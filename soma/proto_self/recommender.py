"""State-aware recommendation engine.

Matches current physiological state to curated corpus.
Logs recommendations to DB for feedback tracking.

Usage:
    python -m soma.proto_self.recommender
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import sys
from datetime import datetime
from typing import Any

from soma.proto_self.db import DEFAULT_DB_PATH, get_connection
from soma.proto_self.state_classifier import classify_state

logger = logging.getLogger(__name__)

CORPUS_PATH = os.environ.get("SOMA_CORPUS", "data/corpus.json")


def load_corpus(corpus_path: str | None = None) -> list[dict]:
    """Load the recommendation corpus."""
    path = corpus_path or CORPUS_PATH
    with open(path) as f:
        return json.load(f)["entries"]


def _ensure_recommendations_table(conn: sqlite3.Connection) -> None:
    """Create recommendations table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recommended_at TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            state_at_recommendation TEXT NOT NULL,
            rmssd_before REAL,
            rhr_before REAL,
            followed INTEGER DEFAULT 0,
            rmssd_after REAL,
            rhr_after REAL,
            outcome TEXT,
            feedback_at TEXT
        )
    """)
    conn.commit()


def get_recommendations(
    n: int = 3,
    exclude_ids: list[str] | None = None,
    db_path: str | None = None,
    model_path: str | None = None,
    corpus_path: str | None = None,
) -> dict[str, Any]:
    """Return top N recommendations for current physiological state.

    Returns dict with 'state' info and 'recommendations' list.
    """
    state_info = classify_state(db_path=db_path, model_path=model_path)
    state = state_info["state"]
    corpus = load_corpus(corpus_path)
    exclude_ids = exclude_ids or []

    # Filter: include if state is in best_states, exclude if in avoid_states
    eligible = [
        entry for entry in corpus
        if state in entry["best_states"]
        and state not in entry.get("avoid_states", [])
        and entry["id"] not in exclude_ids
    ]

    # Fallback: anything not in avoid_states
    if not eligible:
        eligible = [
            entry for entry in corpus
            if state not in entry.get("avoid_states", [])
            and entry["id"] not in exclude_ids
        ]

    random.shuffle(eligible)
    selected = eligible[:n]

    return {
        "state": state_info,
        "recommendations": selected,
    }


def log_recommendation(
    entry_id: str,
    title: str,
    rec_type: str,
    state_info: dict,
    db_path: str | None = None,
) -> int:
    """Write recommendation to DB for feedback tracking. Returns the row ID."""
    conn = get_connection(db_path)
    _ensure_recommendations_table(conn)

    cursor = conn.execute(
        "INSERT INTO recommendations "
        "(recommended_at, entry_id, title, type, state_at_recommendation, rmssd_before, rhr_before) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            entry_id,
            title,
            rec_type,
            state_info["state"],
            state_info.get("rmssd"),
            state_info.get("rhr"),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id or 0


def log_feedback(
    recommendation_id: int,
    followed: int,
    outcome: str | None = None,
    db_path: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Log feedback for a recommendation. Returns state after."""
    state_after = classify_state(db_path=db_path, model_path=model_path)

    conn = get_connection(db_path)
    conn.execute(
        "UPDATE recommendations SET "
        "followed=?, rmssd_after=?, rhr_after=?, outcome=?, feedback_at=? "
        "WHERE id=?",
        (
            followed,
            state_after.get("rmssd"),
            state_after.get("rhr"),
            outcome,
            datetime.now().isoformat(),
            recommendation_id,
        ),
    )
    conn.commit()
    conn.close()

    logger.info("Feedback logged for rec %d: followed=%d outcome=%s",
                recommendation_id, followed, outcome)
    return state_after


def get_pending_recommendations(
    n: int = 5,
    db_path: str | None = None,
) -> list[dict]:
    """Get recent recommendations awaiting feedback."""
    conn = get_connection(db_path)
    _ensure_recommendations_table(conn)
    rows = conn.execute(
        "SELECT id, recommended_at, title, type, state_at_recommendation, followed, outcome "
        "FROM recommendations WHERE feedback_at IS NULL "
        "ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recommendation_history(
    n: int = 20,
    db_path: str | None = None,
) -> list[dict]:
    """Get recent recommendation history with outcomes."""
    conn = get_connection(db_path)
    _ensure_recommendations_table(conn)
    rows = conn.execute(
        "SELECT recommended_at, title, type, state_at_recommendation, "
        "rmssd_before, rmssd_after, followed, outcome "
        "FROM recommendations ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def main() -> None:
    result = get_recommendations(n=3)
    state = result["state"]

    print(f"\nCurrent State: {state['state'].upper()}")
    print(f"   {state['reason']}")
    if state["rhr"] is not None:
        print(f"   RHR: {state['rhr']} bpm | RMSSD: {state['rmssd']} ms\n")

    print("SOMA Recommends:\n")

    for i, rec in enumerate(result["recommendations"], 1):
        print(f"  {i}. [{rec['type'].upper()}] {rec['title']}")
        print(f"     {rec['why']}")
        print(f"     Duration: ~{rec['duration_min']} min\n")
        log_recommendation(rec["id"], rec["title"], rec["type"], state)


if __name__ == "__main__":
    main()
