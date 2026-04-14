"""LanceDB autobiographical memory — every probe/response exchange becomes searchable memory.

This is Extended Consciousness in code: a persistent, searchable record of
what SOMA noticed, what it asked, and what it was told.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import lancedb  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TABLE_NAME = "autobiographical_memory"
NARRATIVE_TABLE = "narrative_summaries"

_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _get_db_path() -> str:
    return os.environ.get("LANCEDB_PATH", "./data/lancedb")


def get_db(db_path: str | None = None) -> lancedb.DBConnection:
    return lancedb.connect(db_path or _get_db_path())


def embed_text(text: str) -> list[float]:
    return np.asarray(_get_encoder().encode(text), dtype=np.float32).tolist()


def store_exchange(
    anomaly_type: str,
    metric: str,
    value: float,
    baseline: float,
    deviation: float,
    body_state: str,
    probe_text: str,
    response_text: str,
    session_label: str = "",
    entities: list[str] | None = None,
    emotion_valence: float = 0.0,
    db_path: str | None = None,
) -> str:
    """Store a complete probe/response exchange as autobiographical memory."""
    db = get_db(db_path)

    full_exchange = (
        f"Physiological state: {body_state}. "
        f"{metric} = {value} ({deviation:+.1f}s from baseline). "
        f"SOMA asked: {probe_text} "
        f"Response: {response_text}"
    )

    memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    record: dict[str, Any] = {
        "vector": embed_text(full_exchange),
        "memory_id": memory_id,
        "timestamp": datetime.now().isoformat(),
        "anomaly_type": anomaly_type,
        "metric": metric,
        "value": float(value or 0),
        "baseline": float(baseline or 0),
        "deviation": float(deviation or 0),
        "body_state": body_state,
        "probe_text": probe_text,
        "response_text": response_text,
        "entities": json.dumps(entities or []),
        "emotion_valence": float(emotion_valence),
        "full_exchange": full_exchange,
        "session_label": session_label,
        "week_number": datetime.now().isocalendar()[1],
    }

    table_names: list[str] = db.table_names()
    if TABLE_NAME in table_names:
        tbl = db.open_table(TABLE_NAME)
        tbl.add([record])
    else:
        db.create_table(TABLE_NAME, [record])

    logger.info("Memory stored: %s", memory_id)
    return memory_id


def retrieve_similar_memories(
    query_text: str,
    n: int = 5,
    min_deviation: float | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over autobiographical memory."""
    db = get_db(db_path)
    if TABLE_NAME not in db.table_names():
        return []

    tbl = db.open_table(TABLE_NAME)
    query_vector = embed_text(query_text)

    results: list[dict[str, Any]] = (
        tbl.search(query_vector).limit(n + 5).to_list()
    )

    valid = [r for r in results if r.get("response_text", "")]

    if min_deviation is not None:
        valid = [r for r in valid if abs(r.get("deviation", 0)) >= min_deviation]

    return valid[:n]


def get_recent_memories(
    n: int = 10,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return most recent exchanges chronologically."""
    db = get_db(db_path)
    if TABLE_NAME not in db.table_names():
        return []

    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()
    if df.empty:
        return []

    records: list[dict[str, Any]] = df.to_dict("records")
    records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return records[:n]


def store_narrative(
    narrative: str,
    week_number: int,
    year: int,
    dominant_state: str = "unknown",
    patterns: list[str] | None = None,
    db_path: str | None = None,
) -> None:
    """Store a weekly narrative synthesis."""
    db = get_db(db_path)

    record: dict[str, Any] = {
        "vector": embed_text(narrative),
        "narrative_id": f"week_{week_number}_{year}",
        "period": f"Week {week_number}, {year}",
        "summary": narrative,
        "patterns_found": json.dumps(patterns or []),
        "dominant_state": dominant_state,
        "generated_at": datetime.now().isoformat(),
    }

    table_names: list[str] = db.table_names()
    if NARRATIVE_TABLE in table_names:
        tbl = db.open_table(NARRATIVE_TABLE)
        tbl.add([record])
    else:
        db.create_table(NARRATIVE_TABLE, [record])

    logger.info("Narrative stored: %s", record["narrative_id"])
