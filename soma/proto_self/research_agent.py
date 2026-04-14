"""Daily research agent — Claude API with web search via OpenRouter.

Runs once daily. Searches the web for recommendations tailored to
the patient's physiological patterns and known preferences.
Stages candidates for review (not auto-merged).

Usage:
    python -m soma.proto_self.research_agent
    python -m soma.proto_self.research_agent --topic "books on consciousness"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv

from soma.proto_self.db import DEFAULT_DB_PATH, get_connection
from soma.proto_self.raen_scorer import score_candidates
from soma.proto_self.soma_profile import build_profile

logger = logging.getLogger(__name__)

load_dotenv(".env.local")
load_dotenv(".env")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("SOMA_RESEARCH_MODEL", "anthropic/claude-sonnet-4")
ADDITIONS_PATH = os.environ.get("SOMA_ADDITIONS_PATH", "data/corpus_additions.json")

# Research topic rotation — cycled by day of year
RESEARCH_TOPICS = [
    "movies and films",
    "books on consciousness, AI, or philosophy of mind",
    "books on causal inference or data science",
    "physical health and recovery activities",
    "meditation and Buddhist practice resources",
    "parenting and early childhood development",
    "books on longevity and health optimization",
    "documentaries about science, nature, or human stories",
    "biohacking and HRV optimization techniques",
    "creative and generative AI tools",
]


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return key


def get_todays_topic() -> str:
    """Rotate through topics by day of year."""
    day = datetime.now().timetuple().tm_yday
    return RESEARCH_TOPICS[day % len(RESEARCH_TOPICS)]


def build_research_prompt(profile: dict, topic: str) -> str:
    """Build the research prompt with full profile context."""
    state = profile["current_state"]
    worked = [w["title"] for w in profile.get("what_worked", [])]
    existing = profile["existing_corpus"]

    identity = profile["identity"]

    return f"""You are SOMA's research agent. Your job is to find excellent {topic} recommendations.

## Patient Profile
- Age {identity['age']}, {identity['location']}, data scientist and AI researcher
- Practices: {', '.join(identity['practices'])}
- Quitting: {', '.join(identity['quitting'])} (important: nothing that glorifies substances)
- Reading: {', '.join(identity['reading_now'])}
- Movie taste: redemption arcs, meaning, human connection ({', '.join(identity['movies_loved'])})
- Values: {', '.join(identity['values'])}
- Interests: {', '.join(identity['interests'])}

## Current Physiological State
{state['state'].upper()} — {state['reason']}
RMSSD: {state.get('rmssd', 'N/A')}ms | RHR: {state.get('rhr', 'N/A')}bpm

## What Has Worked Recently
{json.dumps(worked, indent=2) if worked else "No feedback data yet."}

## Already in Corpus (do not duplicate)
{json.dumps(existing, indent=2)}

## Your Task
Find 5 genuinely excellent {topic} recommendations for this patient.

For each recommendation return a JSON object with these exact fields:
- id: unique string like "res_{datetime.now().strftime('%Y%m%d')}_N"
- type: one of "movie", "book", "activity", "media"
- title: exact title
- why: 1-2 sentences explaining why this fits specifically
- tags: list of relevant tags
- best_states: list from ["depleted", "recovering", "baseline", "restored", "peak"]
- avoid_states: list of states where this would be counterproductive
- duration_min: estimated time in minutes
- source: where you found this (URL or publication name)
- research_date: "{datetime.now().strftime('%Y-%m-%d')}"

Requirements:
- Must be real and verifiable — no hallucinated titles
- Must genuinely fit the values and taste described above
- Must not duplicate anything in the existing corpus
- Prioritize things released or discovered in the last 2 years
- For books: prioritize those available on Kindle or audiobook
- For movies: prioritize streaming availability

Return ONLY a JSON array of 5 objects. No preamble, no explanation, no markdown fences."""


async def run_research(
    topic: str | None = None,
    db_path: str | None = None,
    model_path: str | None = None,
    corpus_path: str | None = None,
) -> list[dict]:
    """Run the research agent. Returns list of staged candidates."""
    api_key = _get_api_key()

    profile = build_profile(db_path=db_path, model_path=model_path, corpus_path=corpus_path)
    topic = topic or get_todays_topic()

    print(f"\nSOMA Research Agent")
    print(f"   Topic: {topic}")
    print(f"   Model: {MODEL}")
    print(f"   Current state: {profile['current_state']['state']}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    prompt = build_research_prompt(profile, topic)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()

    data = resp.json()

    # Log token usage
    usage = data.get("usage", {})
    tokens = usage.get("total_tokens", 0)
    logger.info("Research LLM call: %d tokens", tokens)

    # Extract text
    choices = data.get("choices", [])
    if not choices:
        print("LLM returned no choices")
        return []

    raw_text: str = choices[0].get("message", {}).get("content", "")

    # Parse JSON candidates
    try:
        clean = re.sub(r"```(?:json)?\s*\n?", "", raw_text.strip())
        clean = re.sub(r"\n?```\s*$", "", clean)
        candidates = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"Failed to parse research output: {e}")
        print(f"Raw output:\n{raw_text[:500]}")
        return []

    if not isinstance(candidates, list):
        print(f"Expected JSON array, got {type(candidates)}")
        return []

    # Score candidates
    scored = score_candidates(candidates, profile)

    # Stage for review
    existing_additions: list[dict] = []
    try:
        with open(ADDITIONS_PATH) as f:
            existing_additions = json.load(f)
    except FileNotFoundError:
        pass

    new_additions = [c for c in scored if c["recommended"]]
    all_additions = existing_additions + new_additions

    os.makedirs(os.path.dirname(ADDITIONS_PATH) or ".", exist_ok=True)
    with open(ADDITIONS_PATH, "w") as f:
        json.dump(all_additions, f, indent=2)

    # Log to DB
    conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researched_at TEXT,
            topic TEXT,
            model TEXT,
            candidates_found INTEGER,
            candidates_staged INTEGER,
            tokens_used INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO research_log (researched_at, topic, model, candidates_found, candidates_staged, tokens_used) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), topic, MODEL, len(candidates), len(new_additions), tokens),
    )
    conn.commit()
    conn.close()

    print(f"Research complete")
    print(f"   Found: {len(candidates)} candidates")
    print(f"   Staged for review: {len(new_additions)} (RAEN >= 0.65)")
    print(f"   Tokens used: {tokens}\n")

    for c in new_additions:
        print(f"  [{c['type'].upper()}] {c['title']} — RAEN: {c['raen_total']}")
        print(f"    {c['why']}\n")

    return new_additions


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description="SOMA Daily Research Agent")
    parser.add_argument("--topic", default=None, help="Override today's research topic")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(run_research(topic=args.topic, db_path=args.db))


if __name__ == "__main__":
    main()
