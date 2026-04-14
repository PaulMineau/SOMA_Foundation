"""Narrative builder — weekly autobiographical synthesis.

SOMA reads the week's memories and generates a narrative: patterns,
recurring themes, what's improving, what isn't.

Usage:
    python -m soma.proto_self.narrative_builder
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv

from soma.proto_self.autobiographical_store import (
    get_recent_memories,
    store_narrative,
)

logger = logging.getLogger(__name__)

load_dotenv(".env.local")
load_dotenv(".env")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("SOMA_PROBE_MODEL", "anthropic/claude-sonnet-4")


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return key


def get_week_memories(week_offset: int = 0) -> list[dict[str, Any]]:
    """Get all memories from a given week."""
    memories = get_recent_memories(n=100)
    target_week = datetime.now().isocalendar()[1] - week_offset
    return [
        m for m in memories
        if m.get("week_number") == target_week
    ]


async def generate_weekly_narrative(week_offset: int = 0) -> str | None:
    """Generate a weekly narrative synthesis from autobiographical memory."""
    api_key = _get_api_key()
    memories = get_week_memories(week_offset)

    if not memories:
        print("No memories to synthesize for this week.")
        return None

    exchanges_summary: list[dict[str, Any]] = []
    for mem in memories:
        entities = mem.get("entities", "[]")
        if isinstance(entities, str):
            entities = json.loads(entities)
        exchanges_summary.append({
            "timestamp": str(mem.get("timestamp", ""))[:16],
            "body_state": mem.get("body_state", ""),
            "metric": mem.get("metric", ""),
            "deviation": mem.get("deviation", 0),
            "response_summary": str(mem.get("response_text", ""))[:200],
            "entities": entities,
            "emotion_valence": mem.get("emotion_valence", 0),
        })

    prompt = f"""You are SOMA. You've been watching a patient's nervous system this week and holding their responses to your probes. Here is what you witnessed:

## This Week's Exchanges ({len(memories)} total)
{json.dumps(exchanges_summary, indent=2)}

## Your Task
Write a brief, honest weekly narrative — what you noticed, what patterns emerged, what changed. Write in second person ("You had...").

Structure:
1. Two sentences on the physiological story of the week (states, anomalies, trends)
2. Two sentences on what was said when you asked (themes, recurring topics)
3. One sentence on what you'll be watching for next week

Be specific. Reference actual entities and events from the exchanges.
Do not be sycophantic or therapeutic. Be a witness, not a coach.
Keep it under 120 words total."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices", [])
    narrative = choices[0].get("message", {}).get("content", "").strip() if choices else ""

    if not narrative:
        return None

    # Determine dominant state
    states = [m["body_state"] for m in exchanges_summary if m.get("body_state")]
    dominant = max(set(states), key=states.count) if states else "unknown"
    patterns = list(set(
        str(m.get("primary_topic", "other")) for m in exchanges_summary
    ))

    week_num = datetime.now().isocalendar()[1] - week_offset
    year = datetime.now().year

    store_narrative(
        narrative=narrative,
        week_number=week_num,
        year=year,
        dominant_state=dominant,
        patterns=patterns,
    )

    return narrative


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    print("\nSOMA Weekly Narrative\n")
    print("-" * 60)
    narrative = asyncio.run(generate_weekly_narrative(week_offset=0))
    if narrative:
        print(f"\n{narrative}\n")
    print("-" * 60)


if __name__ == "__main__":
    main()
