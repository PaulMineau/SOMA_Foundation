"""Probe generator — takes an anomaly, retrieves memories, generates an intelligent question.

This is SOMA's first spoken word. The quality of the question depends on
how much autobiographical context has accumulated.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any

import httpx
from dotenv import load_dotenv

from soma.proto_self.autobiographical_store import (
    get_recent_memories,
    retrieve_similar_memories,
)
from soma.proto_self.db import DEFAULT_DB_PATH, get_connection
from soma.proto_self.state_classifier import classify_state

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


def get_recent_anomalies(
    hours: int = 48,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Get anomaly events from the last N hours for pattern context."""
    conn = get_connection(db_path)
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    try:
        rows = conn.execute(
            "SELECT detected_at, metric, value, baseline, deviation "
            "FROM anomalies WHERE detected_at > ? "
            "ORDER BY detected_at DESC LIMIT 10",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def get_current_session_label(db_path: str | None = None) -> str:
    """Get the label of the most recently active session."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT label FROM sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row["label"] if row else "unlabeled"


def build_probe_prompt(
    anomaly: dict,
    state_info: dict,
    similar_memories: list[dict],
    recent_anomalies: list[dict],
    session_label: str,
) -> str:
    """Assemble the full context prompt for probe generation."""
    memory_context = ""
    if similar_memories:
        memory_context = "\n\nRelevant past exchanges:\n"
        for mem in similar_memories[:3]:
            ts = mem.get("timestamp", "")[:10]
            memory_context += (
                f"\n[{ts}] {mem.get('metric', '')} = {mem.get('value', '')} "
                f"({mem.get('deviation', 0):+.1f}s)\n"
                f"Response: \"{mem.get('response_text', '')}\"\n"
            )

    pattern_context = ""
    if len(recent_anomalies) > 1:
        metrics = [a["metric"] for a in recent_anomalies]
        most_common = max(set(metrics), key=metrics.count)
        pattern_context = (
            f"\nRecent anomaly pattern (last 48h): "
            f"{len(recent_anomalies)} events, mostly {most_common}."
        )

    hour = datetime.now().hour
    time_of_day = (
        "early morning" if hour < 8 else
        "mid-morning" if hour < 11 else
        "midday" if hour < 13 else
        "mid-afternoon" if hour < 16 else
        "late afternoon" if hour < 18 else
        "evening"
    )

    metric_unit = "ms" if "rmssd" in anomaly.get("metric", "") else "bpm"

    return f"""You are SOMA — a physiological AI that monitors a patient's nervous system and builds an autobiographical understanding of their life through their body's signals.

## Current Anomaly
Metric: {anomaly.get('metric', '').upper()}
Value: {anomaly.get('value', 0)} {metric_unit}
Baseline: {anomaly.get('baseline', 0)}
Deviation: {anomaly.get('deviation', 0):+.1f} standard deviations
Body state: {state_info.get('state', 'unknown').upper()}
Time of day: {time_of_day}
Session label: {session_label}
{pattern_context}
{memory_context}

## Your Task
Generate a single conversational probe. One question, or a brief observation followed by one question.

Rules:
- Sound like a thoughtful friend who has been watching, not a medical device
- Reference specific numbers only if they add meaning
- If there are past similar exchanges, reference the pattern naturally
- Do not use clinical language like "your HRV metrics indicate"
- Do not catastrophize or alarm — this is curiosity, not diagnosis
- Keep it under 60 words
- End with a single open question

Return ONLY the probe text. No preamble, no explanation."""


async def generate_probe(
    anomaly: dict,
    db_path: str | None = None,
    model_path: str | None = None,
) -> tuple[str, dict, list[dict]]:
    """Generate a conversational probe for a detected anomaly.

    Returns (probe_text, state_info, similar_memories).
    """
    api_key = _get_api_key()
    state_info = classify_state(db_path=db_path, model_path=model_path)

    query = (
        f"Body state {state_info['state']}. "
        f"{anomaly.get('metric', '')} = {anomaly.get('value', 0)} "
        f"({anomaly.get('deviation', 0):+.1f}s). "
        f"Time: {datetime.now().strftime('%H:%M')}."
    )

    similar_memories = retrieve_similar_memories(query, n=5, min_deviation=1.0)
    recent_anomalies = get_recent_anomalies(hours=48, db_path=db_path)
    session_label = get_current_session_label(db_path)

    prompt = build_probe_prompt(
        anomaly, state_info, similar_memories, recent_anomalies, session_label
    )

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
                "max_tokens": 200,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices", [])
    probe_text = choices[0].get("message", {}).get("content", "") if choices else ""

    return probe_text.strip(), state_info, similar_memories
