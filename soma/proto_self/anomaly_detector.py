"""Anomaly detector — Core Consciousness comes online.

Runs continuously alongside polar_logger.py. Watches the live DB feed,
compares to baseline model, writes anomalies when deviations exceed 1.5 std dev.

This is Core Consciousness in code: the system becomes aware of a change.

Usage:
    python -m soma.proto_self.anomaly_detector
    python -m soma.proto_self.anomaly_detector --poll-interval 15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime

from soma.proto_self.artifact_filter import clean_rr, compute_rhr, compute_rmssd
from soma.proto_self.baseline_model import MODEL_PATH
from soma.proto_self.db import DEFAULT_DB_PATH, get_connection

logger = logging.getLogger(__name__)

WINDOW_SIZE = 60  # RR intervals per computation window
DEFAULT_POLL_INTERVAL = 30  # seconds between checks


def load_model(model_path: str | None = None) -> dict:
    """Load the baseline model from JSON."""
    path = model_path or MODEL_PATH
    with open(path) as f:
        return json.load(f)


def _ensure_anomalies_table(conn: sqlite3.Connection) -> None:
    """Create anomalies table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL,
            baseline REAL,
            deviation REAL,
            acknowledged INTEGER DEFAULT 0
        )
    """)
    conn.commit()


def write_anomaly(
    conn: sqlite3.Connection,
    metric: str,
    value: float,
    baseline: float,
    deviation: float,
) -> None:
    """Write an anomaly event to the database."""
    conn.execute(
        "INSERT INTO anomalies (detected_at, metric, value, baseline, deviation) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), metric, value, baseline, round(deviation, 2)),
    )
    conn.commit()
    print(f"  ANOMALY: {metric} = {value} (baseline {baseline}, {deviation:.1f}s)")


def check_window(
    rr_window: list[float],
    model: dict,
    conn: sqlite3.Connection,
) -> None:
    """Check a window of RR intervals against baseline."""
    clean = clean_rr(rr_window)
    if len(clean) < 10:
        return

    rhr = compute_rhr(clean)
    rmssd = compute_rmssd(clean)

    if rhr is not None:
        mean = model["rhr"]["mean"]
        std = model["rhr"]["std"]
        if std > 0:
            deviation = (rhr - mean) / std
            if abs(deviation) > 1.5:
                write_anomaly(conn, "rhr", rhr, mean, deviation)
            else:
                print(f"  RHR: {rhr} bpm ({deviation:+.1f}s) ok")

    if rmssd is not None:
        mean = model["rmssd"]["mean"]
        std = model["rmssd"]["std"]
        if std > 0:
            deviation = (rmssd - mean) / std
            if deviation < -1.5:  # Only flag LOW RMSSD (stress signal)
                write_anomaly(conn, "rmssd", rmssd, mean, deviation)
            else:
                print(f"  RMSSD: {rmssd} ms ({deviation:+.1f}s) ok")


def run_detector(
    db_path: str | None = None,
    model_path: str | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> None:
    """Run the anomaly detector loop."""
    print("SOMA Anomaly Detector — Core Consciousness Online")
    print(f"   Polling every {poll_interval}s\n")

    model = load_model(model_path)
    print(f"   Baseline RHR:   {model['rhr']['mean']} +/- {model['rhr']['std']} bpm")
    print(f"   Baseline RMSSD: {model['rmssd']['mean']} +/- {model['rmssd']['std']} ms\n")

    conn = get_connection(db_path)
    _ensure_anomalies_table(conn)

    last_processed_id = 0

    try:
        while True:
            rows = conn.execute(
                "SELECT id, rr_ms FROM rr_intervals WHERE id > ? ORDER BY id ASC LIMIT ?",
                (last_processed_id, WINDOW_SIZE),
            ).fetchall()

            if rows:
                last_processed_id = rows[-1]["id"]
                rr_window = [r["rr_ms"] for r in rows]
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Window: {len(rr_window)} readings")
                check_window(rr_window, model, conn)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for signal...")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nAnomaly detector stopped.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOMA Anomaly Detector — Core Consciousness"
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    parser.add_argument("--model", default=MODEL_PATH, help="Baseline model JSON path")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help="Seconds between checks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    run_detector(db_path=args.db, model_path=args.model, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
