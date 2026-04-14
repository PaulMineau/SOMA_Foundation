"""HRV data storage — writes Polar H10 sessions to LanceDB time-series table."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import lancedb  # type: ignore[import-untyped]

from soma.proto_self.hrv import HRVMetrics, classify_body_state, compute_hrv
from soma.proto_self.polar_reader import PolarSession

logger = logging.getLogger(__name__)

TABLE_RR = "proto_self_rr"  # Raw RR intervals
TABLE_HRV = "proto_self_hrv"  # Computed HRV windows


def _get_db_path() -> str:
    return os.environ.get("LANCEDB_PATH", "./data/lancedb")


def get_db(db_path: str | None = None) -> lancedb.DBConnection:
    """Connect to the LanceDB database."""
    path = db_path or _get_db_path()
    return lancedb.connect(path)


def store_session(
    session: PolarSession,
    db_path: str | None = None,
) -> int:
    """Store raw RR interval data from a Polar session.

    Each RR interval gets its own row with a timestamp,
    enabling time-series queries and windowed feature extraction.

    Returns number of records written.
    """
    if not session.samples:
        logger.info("No samples to store")
        return 0

    db = get_db(db_path)

    records: list[dict[str, Any]] = []
    for sample in session.samples:
        for rr in sample.rr_intervals:
            records.append({
                "timestamp": sample.timestamp,
                "datetime": datetime.fromtimestamp(sample.timestamp).isoformat(),
                "heart_rate": sample.heart_rate,
                "rr_interval_ms": rr,
                "sensor_contact": sample.sensor_contact,
                "device": session.device_name,
                "session_start": datetime.fromtimestamp(session.start_time).isoformat(),
            })

    if not records:
        logger.info("No RR intervals to store")
        return 0

    table_names: list[str] = db.table_names()
    if TABLE_RR in table_names:
        tbl = db.open_table(TABLE_RR)
        tbl.add(records)
        logger.info("Appended %d RR records to '%s'", len(records), TABLE_RR)
    else:
        db.create_table(TABLE_RR, records)
        logger.info("Created table '%s' with %d records", TABLE_RR, len(records))

    return len(records)


def store_hrv_window(
    metrics: HRVMetrics,
    body_state: str,
    window_start: float,
    db_path: str | None = None,
) -> None:
    """Store a computed HRV window with body state classification."""
    db = get_db(db_path)

    record: dict[str, Any] = {
        "timestamp": window_start,
        "datetime": datetime.fromtimestamp(window_start).isoformat(),
        "mean_rr": metrics.mean_rr,
        "sdnn": metrics.sdnn,
        "rmssd": metrics.rmssd,
        "pnn50": metrics.pnn50,
        "mean_hr": metrics.mean_hr,
        "n_intervals": metrics.n_intervals,
        "n_artifacts": metrics.n_artifacts,
        "window_seconds": metrics.window_seconds,
        "body_state": body_state,
    }

    table_names: list[str] = db.table_names()
    if TABLE_HRV in table_names:
        tbl = db.open_table(TABLE_HRV)
        tbl.add([record])
    else:
        db.create_table(TABLE_HRV, [record])

    logger.info(
        "Stored HRV window: RMSSD=%.1f state=%s", metrics.rmssd, body_state
    )


def store_session_with_hrv(
    session: PolarSession,
    db_path: str | None = None,
) -> tuple[int, HRVMetrics | None]:
    """Store raw RR data and compute/store HRV metrics for the session.

    Returns (n_rr_records, hrv_metrics_or_none).
    """
    n_stored = store_session(session, db_path)

    rr = session.all_rr_intervals
    if len(rr) < 3:
        logger.warning("Too few RR intervals for HRV computation (%d)", len(rr))
        return n_stored, None

    try:
        metrics = compute_hrv(rr, window_seconds=session.duration_seconds)
    except ValueError as e:
        logger.warning("HRV computation failed: %s", e)
        return n_stored, None

    body_state = classify_body_state(metrics)
    store_hrv_window(metrics, body_state, session.start_time, db_path)

    return n_stored, metrics
