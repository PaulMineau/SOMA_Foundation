"""LanceDB recommendation memory — embeds (state, recommendation, outcome) for learning.

Over time, SOMA learns which recommendations actually work for each physiological state
by searching past outcomes via semantic similarity.

Usage:
    from soma.proto_self.soma_memory import embed_recommendation, find_similar_past
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import lancedb  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TABLE_NAME = "recommendation_memory"


def _get_db_path() -> str:
    return os.environ.get("LANCEDB_PATH", "./data/lancedb")


def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_recommendation(
    rec_entry: dict,
    state_info: dict,
    outcome: str,
    rmssd_after: float | None = None,
    db_path: str | None = None,
) -> None:
    """Store a completed recommendation + outcome as an embedding.

    The embedding encodes: what state, what was recommended, what happened.
    This is the training signal for learning which recommendations work.
    """
    text = (
        f"State: {state_info['state']}. "
        f"RMSSD: {state_info.get('rmssd', 0)}ms. "
        f"RHR: {state_info.get('rhr', 0)}bpm. "
        f"Recommended: {rec_entry['title']}. "
        f"Tags: {', '.join(rec_entry.get('tags', []))}. "
        f"Outcome: {outcome}."
    )

    embedder = _get_embedder()
    vector = np.asarray(embedder.encode(text), dtype=np.float32).tolist()

    db = lancedb.connect(db_path or _get_db_path())
    record = {
        "vector": vector,
        "entry_id": rec_entry["id"],
        "title": rec_entry["title"],
        "type": rec_entry["type"],
        "state": state_info["state"],
        "outcome": outcome,
        "rmssd_before": float(state_info.get("rmssd") or 0),
        "rmssd_after": float(rmssd_after or 0),
        "rhr_before": float(state_info.get("rhr") or 0),
        "recorded_at": datetime.now().isoformat(),
    }

    table_names: list[str] = db.table_names()
    if TABLE_NAME in table_names:
        tbl = db.open_table(TABLE_NAME)
        tbl.add([record])
    else:
        db.create_table(TABLE_NAME, [record])

    logger.info("Embedded: %s -> %s", rec_entry["title"], outcome)


def find_similar_past(
    current_state_info: dict,
    n: int = 5,
    only_positive: bool = True,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Find past recommendations that worked in a similar state.

    Searches by embedding similarity to current state,
    optionally filtering to only 'better' outcomes.
    """
    query_text = (
        f"State: {current_state_info['state']}. "
        f"RMSSD: {current_state_info.get('rmssd', 0)}ms. "
        f"RHR: {current_state_info.get('rhr', 0)}bpm."
    )

    embedder = _get_embedder()
    query_vector = np.asarray(embedder.encode(query_text), dtype=np.float32).tolist()

    db = lancedb.connect(db_path or _get_db_path())
    if TABLE_NAME not in db.table_names():
        return []

    tbl = db.open_table(TABLE_NAME)
    search = tbl.search(query_vector).limit(n)

    if only_positive:
        search = search.where("outcome = 'better'")

    results: list[dict[str, Any]] = search.to_list()
    return results
