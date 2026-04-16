"""CPAP data ingestor — stores myAir + EDF data in LanceDB.

Daily summary table + event time-series table for correlation with
Fitbit recovery and Polar HRV.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import lancedb  # type: ignore[import-untyped]
import numpy as np

logger = logging.getLogger(__name__)

SOMA_DB_PATH = os.path.expanduser("~/soma/soma.db")
DAILY_TABLE = "proto_self_cpap_daily"
EVENTS_TABLE = "proto_self_cpap_events"


def _get_db() -> lancedb.DBConnection:
    os.makedirs(os.path.dirname(SOMA_DB_PATH), exist_ok=True)
    return lancedb.connect(SOMA_DB_PATH)


def _build_narrative(record: dict) -> str:
    """Generate natural language description of night's sleep-disordered breathing."""
    parts = []

    ahi = record.get("ahi", 0)
    usage_min = record.get("usage_min", 0)
    usage_hrs = usage_min / 60

    parts.append(f"CPAP usage: {usage_hrs:.1f} hours.")

    if ahi < 1:
        parts.append(f"AHI {ahi:.1f} — excellent control of sleep-disordered breathing.")
    elif ahi < 5:
        parts.append(f"AHI {ahi:.1f} — good control, below diagnostic threshold.")
    elif ahi < 15:
        parts.append(f"AHI {ahi:.1f} — mild residual events despite therapy.")
    elif ahi < 30:
        parts.append(f"AHI {ahi:.1f} — moderate events, therapy may need adjustment.")
    else:
        parts.append(f"AHI {ahi:.1f} — severe residual events, review with clinician.")

    # Compliance
    if usage_hrs >= 4:
        parts.append("Compliant night (>=4h).")
    else:
        parts.append(f"Short usage night — {usage_hrs:.1f}h. Under compliance threshold.")

    # Leak
    leak_p95 = record.get("leak_p95") or record.get("leak_percentile", 0)
    if leak_p95 > 24:
        parts.append(f"High mask leak (p95 {leak_p95:.0f}L/min) — mask fit or strap.")
    elif leak_p95 > 0:
        parts.append(f"Leak under control (p95 {leak_p95:.0f}L/min).")

    # Score
    score = record.get("sleep_score", 0)
    if score:
        parts.append(f"myAir score: {score}/100.")

    return " ".join(parts)


def ensure_daily_table(db: lancedb.DBConnection) -> Any:
    """Create daily CPAP summary table if it doesn't exist."""
    if DAILY_TABLE in db.table_names():
        return db.open_table(DAILY_TABLE)

    schema = {
        "date": "2026-01-01",
        "source": "myair",  # "myair" | "edf" | "merged"
        "ahi": 0.0,
        "usage_min": 0,
        "sleep_score": 0,
        "mask_pair_count": 0,
        "leak_percentile": 0.0,
        "leak_p95": 0.0,
        "median_leak": 0.0,
        "mean_pressure": 0.0,
        "apneas": 0,
        "hypopneas": 0,
        "total_events": 0,
        "duration_min": 0,
        "start_time": "",
        "end_time": "",
        "damasio_layer": "L1_proto_self",
        "narrative": "",
        "vector": np.zeros(384, dtype=np.float32).tolist(),
        "ingested_at": "",
    }

    return db.create_table(DAILY_TABLE, data=[schema])


def ensure_events_table(db: lancedb.DBConnection) -> Any:
    """Create events time-series table if it doesn't exist."""
    if EVENTS_TABLE in db.table_names():
        return db.open_table(EVENTS_TABLE)

    schema = {
        "date": "2026-01-01",
        "timestamp": "",
        "event_type": "apnea",
        "duration_sec": 0.0,
        "magnitude": 0.0,
        "hour_of_night": 0,  # 0-11 from sleep start
    }

    return db.create_table(EVENTS_TABLE, data=[schema])


def ingest_myair_records(records: list[dict]) -> int:
    """Ingest myAir daily summaries into LanceDB.

    Returns count of new records added.
    """
    from sentence_transformers import SentenceTransformer

    db = _get_db()
    table = ensure_daily_table(db)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    added = 0
    for rec in records:
        date = rec.get("startDate")
        if not date:
            continue

        # Skip if already ingested
        try:
            existing = table.search().where(f"date = '{date}'").limit(1).to_list()
            if existing:
                continue
        except Exception:
            pass

        normalized = {
            "date": date,
            "source": "myair",
            "ahi": float(rec.get("ahi", 0) or 0),
            "usage_min": int(rec.get("totalUsage", 0) or 0),
            "sleep_score": int(rec.get("sleepScore", 0) or 0),
            "mask_pair_count": int(rec.get("maskPairCount", 0) or 0),
            "leak_percentile": float(rec.get("leakPercentile", 0) or 0),
            "leak_p95": 0.0,
            "median_leak": 0.0,
            "mean_pressure": 0.0,
            "apneas": 0,
            "hypopneas": 0,
            "total_events": 0,
            "duration_min": int(rec.get("totalUsage", 0) or 0),
            "start_time": "",
            "end_time": "",
            "damasio_layer": "L1_proto_self",
            "narrative": "",
            "vector": np.zeros(384, dtype=np.float32).tolist(),
            "ingested_at": datetime.now().isoformat(),
        }
        normalized["narrative"] = _build_narrative(normalized)
        normalized["vector"] = encoder.encode(normalized["narrative"]).tolist()

        table.add([normalized])
        added += 1
        logger.info("Ingested myAir: %s AHI=%.1f usage=%dmin",
                    date, normalized["ahi"], normalized["usage_min"])

    return added


def ingest_edf_summary(summary: Any) -> None:
    """Ingest an EDF-parsed CPAPNightSummary, merging with existing myAir record if any.

    summary: CPAPNightSummary from edf_parser
    """
    from sentence_transformers import SentenceTransformer

    db = _get_db()
    table = ensure_daily_table(db)
    events_table = ensure_events_table(db)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    date = summary.date

    # Check for existing record (myAir) to merge with
    existing_list = []
    try:
        existing_list = table.search().where(f"date = '{date}'").limit(1).to_list()
    except Exception:
        pass

    base = existing_list[0] if existing_list else {}
    merged = {
        "date": date,
        "source": "merged" if existing_list else "edf",
        # Prefer EDF values (more accurate) with myAir as fallback
        "ahi": float(summary.ahi or base.get("ahi", 0)),
        "usage_min": int(base.get("usage_min", 0) or summary.duration_min),
        "sleep_score": int(base.get("sleep_score", 0)),
        "mask_pair_count": int(base.get("mask_pair_count", 0)),
        "leak_percentile": float(base.get("leak_percentile", 0)),
        "leak_p95": float(summary.p95_leak),
        "median_leak": float(summary.median_leak),
        "mean_pressure": float(summary.mean_pressure),
        "apneas": int(summary.apneas),
        "hypopneas": int(summary.hypopneas),
        "total_events": int(summary.total_events),
        "duration_min": int(summary.duration_min),
        "start_time": summary.start_time.isoformat() if summary.start_time else "",
        "end_time": summary.end_time.isoformat() if summary.end_time else "",
        "damasio_layer": "L1_proto_self",
        "narrative": "",
        "vector": np.zeros(384, dtype=np.float32).tolist(),
        "ingested_at": datetime.now().isoformat(),
    }
    merged["narrative"] = _build_narrative(merged)
    merged["vector"] = encoder.encode(merged["narrative"]).tolist()

    # Delete existing record then add merged
    if existing_list:
        try:
            table.delete(f"date = '{date}'")
        except Exception:
            pass
    table.add([merged])

    # Also store individual events for time-series analysis
    if summary.start_time:
        event_records = []
        for ev in summary.events:
            hour_of_night = int((ev.timestamp - summary.start_time).total_seconds() / 3600)
            event_records.append({
                "date": date,
                "timestamp": ev.timestamp.isoformat(),
                "event_type": ev.event_type,
                "duration_sec": float(ev.duration_sec),
                "magnitude": float(ev.magnitude),
                "hour_of_night": hour_of_night,
            })

        if event_records:
            events_table.add(event_records)

    logger.info("Ingested EDF: %s AHI=%.1f events=%d", date, summary.ahi, summary.total_events)


def get_recent_cpap_days(n: int = 7) -> list[dict]:
    """Get last N days of CPAP summary data."""
    db = _get_db()
    if DAILY_TABLE not in db.table_names():
        return []

    tbl = db.open_table(DAILY_TABLE)
    try:
        df = tbl.to_pandas()
        if df.empty:
            return []
        df = df[df["date"] != "2026-01-01"]  # filter out schema init row
        df = df.sort_values("date", ascending=False)
        return df.head(n).to_dict("records")
    except Exception:
        return []


def get_events_for_date(date: str) -> list[dict]:
    """Get all events for a given night."""
    db = _get_db()
    if EVENTS_TABLE not in db.table_names():
        return []

    tbl = db.open_table(EVENTS_TABLE)
    try:
        df = tbl.to_pandas()
        if df.empty:
            return []
        df = df[df["date"] == date]
        return df.sort_values("timestamp").to_dict("records")
    except Exception:
        return []
