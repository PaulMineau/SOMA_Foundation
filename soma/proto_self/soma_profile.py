"""SOMA Profile — assembles rich context for the research agent.

Combines identity, current physiology, feedback history, and existing corpus
into a single profile dict that becomes the research prompt context.

Usage:
    python -m soma.proto_self.soma_profile
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta

from soma.proto_self.artifact_filter import clean_rr, compute_rhr, compute_rmssd
from soma.proto_self.db import DEFAULT_DB_PATH, get_connection
from soma.proto_self.recommender import CORPUS_PATH
from soma.proto_self.state_classifier import classify_state

logger = logging.getLogger(__name__)


def get_recent_state_summary(
    days: int = 7,
    db_path: str | None = None,
) -> list[dict]:
    """Summarize physiological states over the past N days."""
    conn = get_connection(db_path)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT s.label, COUNT(r.id) as readings, AVG(r.rr_ms) as avg_rr "
        "FROM rr_intervals r "
        "JOIN sessions s ON r.session_id = s.session_id "
        "WHERE r.timestamp > ? GROUP BY s.label",
        (cutoff,),
    ).fetchall()
    conn.close()

    summary = []
    for row in rows:
        avg_rr = row["avg_rr"]
        rhr = round(60000 / avg_rr, 1) if avg_rr else None
        summary.append({
            "label": row["label"],
            "readings": row["readings"],
            "avg_rhr": rhr,
        })
    return summary


def get_what_worked(
    n: int = 10,
    db_path: str | None = None,
) -> list[dict]:
    """Get recommendations that produced 'better' outcomes."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT title, type, state_at_recommendation, "
            "rmssd_before, rmssd_after, outcome "
            "FROM recommendations WHERE outcome = 'better' "
            "ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        conn.close()
        return [
            {
                "title": r["title"],
                "type": r["type"],
                "state": r["state_at_recommendation"],
                "rmssd_delta": round((r["rmssd_after"] or 0) - (r["rmssd_before"] or 0), 1),
                "outcome": r["outcome"],
            }
            for r in rows
        ]
    except Exception:
        conn.close()
        return []


def get_what_didnt_work(
    n: int = 5,
    db_path: str | None = None,
) -> list[dict]:
    """Get recommendations that produced 'worse' or 'same' outcomes."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT title, type, state_at_recommendation, outcome "
            "FROM recommendations WHERE outcome IN ('worse', 'same') "
            "ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        conn.close()
        return [
            {
                "title": r["title"],
                "type": r["type"],
                "state": r["state_at_recommendation"],
                "outcome": r["outcome"],
            }
            for r in rows
        ]
    except Exception:
        conn.close()
        return []


def get_existing_titles(corpus_path: str | None = None) -> list[str]:
    """Return titles already in corpus to avoid duplicates."""
    path = corpus_path or CORPUS_PATH
    with open(path) as f:
        corpus = json.load(f)["entries"]
    return [e["title"] for e in corpus]


def build_profile(
    db_path: str | None = None,
    model_path: str | None = None,
    corpus_path: str | None = None,
) -> dict:
    """Assemble full profile for research prompt injection."""
    current_state = classify_state(db_path=db_path, model_path=model_path)
    state_history = get_recent_state_summary(days=7, db_path=db_path)
    worked = get_what_worked(db_path=db_path)
    didnt_work = get_what_didnt_work(db_path=db_path)
    existing = get_existing_titles(corpus_path)

    profile = {
        "identity": {
            "age": 50,
            "location": "Duvall, WA",
            "sleep_condition": "sleep apnea, CPAP user",
            "practices": ["Tibetan Tonglen meditation", "running", "kettlebell"],
            "quitting": ["nicotine", "THC"],
            "interests": [
                "causal inference", "consciousness research", "AI architecture",
                "recommendation systems", "Buddhist philosophy", "parenting",
                "health optimization", "data science", "biometric tracking",
            ],
            "values": [
                "upstream causation", "compound over deplete",
                "be your own customer", "friction engineering",
                "compassion", "presence with family",
            ],
            "reading_now": ["The Feeling of What Happens (Damasio)", "Causality (Pearl)"],
            "movies_loved": ["Soul", "Shawshank Redemption", "A Man Called Otto"],
        },
        "current_state": current_state,
        "week_summary": state_history,
        "what_worked": worked,
        "what_didnt_work": didnt_work,
        "existing_corpus": existing,
        "generated_at": datetime.now().isoformat(),
    }

    return profile


def main() -> None:
    profile = build_profile()
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
