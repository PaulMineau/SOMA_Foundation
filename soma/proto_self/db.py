"""SQLite storage for SOMA-Cardio RR interval logging.

Schema:
- sessions: labeled recording sessions with start/end timestamps
- rr_intervals: individual RR intervals linked to sessions
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get("SOMA_CARDIO_DB", "data/soma_cardio.db")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection, creating the database if needed."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            label TEXT,
            device_name TEXT,
            device_address TEXT,
            battery_level INTEGER,
            n_intervals INTEGER DEFAULT 0,
            mean_hr REAL,
            rmssd REAL,
            sdnn REAL,
            body_state TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rr_intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            rr_ms REAL NOT NULL,
            hr_bpm INTEGER,
            session_id TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_rr_session
        ON rr_intervals(session_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_rr_timestamp
        ON rr_intervals(timestamp)
    """)
    conn.commit()


def start_session(
    conn: sqlite3.Connection,
    session_id: str,
    label: str = "unlabeled",
    device_name: str = "",
    device_address: str = "",
    battery_level: int | None = None,
) -> None:
    """Record the start of a new session."""
    conn.execute(
        "INSERT INTO sessions (session_id, started_at, label, device_name, device_address, battery_level) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, datetime.now().isoformat(), label, device_name, device_address, battery_level),
    )
    conn.commit()
    logger.info("Session started: %s (label=%s)", session_id, label)


def insert_rr(
    conn: sqlite3.Connection,
    session_id: str,
    rr_ms: float,
    hr_bpm: int,
    timestamp: str | None = None,
) -> None:
    """Insert a single RR interval."""
    ts = timestamp or datetime.now().isoformat()
    conn.execute(
        "INSERT INTO rr_intervals (timestamp, rr_ms, hr_bpm, session_id) VALUES (?, ?, ?, ?)",
        (ts, rr_ms, hr_bpm, session_id),
    )


def end_session(
    conn: sqlite3.Connection,
    session_id: str,
    n_intervals: int = 0,
    mean_hr: float | None = None,
    rmssd: float | None = None,
    sdnn: float | None = None,
    body_state: str | None = None,
) -> None:
    """Record the end of a session with summary metrics."""
    conn.execute(
        "UPDATE sessions SET ended_at=?, n_intervals=?, mean_hr=?, rmssd=?, sdnn=?, body_state=? "
        "WHERE session_id=?",
        (datetime.now().isoformat(), n_intervals, mean_hr, rmssd, sdnn, body_state, session_id),
    )
    conn.commit()
    logger.info(
        "Session ended: %s (%d intervals, RMSSD=%.1f, state=%s)",
        session_id, n_intervals, rmssd or 0, body_state or "unknown",
    )


def get_session_rr(conn: sqlite3.Connection, session_id: str) -> list[float]:
    """Get all RR intervals for a session."""
    rows = conn.execute(
        "SELECT rr_ms FROM rr_intervals WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [row["rr_ms"] for row in rows]


def get_recent_sessions(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """Get recent sessions with summary data."""
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def export_session_csv(
    conn: sqlite3.Connection,
    session_id: str,
    output_path: str | None = None,
) -> str:
    """Export a session's RR intervals to CSV.

    Returns the output file path.
    """
    import csv

    if output_path is None:
        output_path = f"data/rr_export_{session_id}.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = conn.execute(
        "SELECT timestamp, rr_ms, hr_bpm FROM rr_intervals WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rr_ms", "hr_bpm", "session_id"])
        for row in rows:
            writer.writerow([row["timestamp"], row["rr_ms"], row["hr_bpm"], session_id])

    logger.info("Exported %d rows to %s", len(rows), output_path)
    return output_path


def export_daily_csv(
    conn: sqlite3.Connection,
    date_str: str | None = None,
    output_path: str | None = None,
) -> str:
    """Export all RR intervals from a given day to CSV."""
    import csv

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    if output_path is None:
        output_path = f"data/rr_export_{date_str.replace('-', '')}.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = conn.execute(
        "SELECT timestamp, rr_ms, hr_bpm, session_id FROM rr_intervals "
        "WHERE timestamp LIKE ? ORDER BY id",
        (f"{date_str}%",),
    ).fetchall()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rr_ms", "hr_bpm", "session_id"])
        for row in rows:
            writer.writerow([row["timestamp"], row["rr_ms"], row["hr_bpm"], row["session_id"]])

    logger.info("Exported %d daily rows to %s", len(rows), output_path)
    return output_path
