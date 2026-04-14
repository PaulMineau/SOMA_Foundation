"""SOMA Core server — FastAPI running on NAS.

Receives RR interval batches from SOMA Mobile, computes HRV,
detects anomalies, and serves the conversational probe endpoint.

Run:
    uvicorn soma.proto_self.soma_server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from soma.proto_self.db import (
    get_connection,
    get_recent_sessions,
    insert_rr,
    start_session,
)
from soma.proto_self.hrv import classify_body_state, compute_hrv

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("SOMA_CARDIO_DB", "data/soma_cardio.db")

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _init_anomalies_table()
    logger.info("SOMA Core server started, DB: %s", DB_PATH)
    yield


app = FastAPI(title="SOMA Core", version="0.2.0", lifespan=lifespan)


# ── Models ──────────────────────────────────────────────────────────────────


class RRReading(BaseModel):
    timestamp: str
    rr_ms: float
    hr_bpm: int | None = None


class RRBatch(BaseModel):
    session_id: str
    label: str | None = None
    device_id: str = "polar_h10"
    readings: list[RRReading]


class ContextTag(BaseModel):
    session_id: str
    label: str


class ProbeRequest(BaseModel):
    message: str | None = None
    anomaly_id: int | None = None
    generate_only: bool = False


# ── DB + anomaly helpers ────────────────────────────────────────────────────


def _init_anomalies_table() -> None:
    """Ensure the anomalies table exists."""
    conn = get_connection(DB_PATH)
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
    conn.close()


def _compute_baseline(conn: Any) -> dict[str, float | None]:
    """Compute baseline from morning_baseline labeled sessions."""
    rows = conn.execute(
        "SELECT rr.rr_ms FROM rr_intervals rr "
        "JOIN sessions s ON rr.session_id = s.session_id "
        "WHERE s.label = 'morning_baseline' "
        "ORDER BY rr.id DESC LIMIT 500"
    ).fetchall()

    rr_list = [r["rr_ms"] for r in rows]
    if len(rr_list) < 10:
        return {"rhr": None, "rmssd": None, "rhr_std": None, "rmssd_std": None, "n": len(rr_list)}

    try:
        metrics = compute_hrv(rr_list)
    except ValueError:
        return {"rhr": None, "rmssd": None, "rhr_std": None, "rmssd_std": None, "n": len(rr_list)}

    # Compute std dev of HR estimates from RR
    hr_values = [60000.0 / rr for rr in rr_list if 300 < rr < 2000]
    hr_std = (sum((h - metrics.mean_hr) ** 2 for h in hr_values) / len(hr_values)) ** 0.5 if hr_values else 0

    return {
        "rhr": metrics.mean_hr,
        "rmssd": metrics.rmssd,
        "rhr_std": round(hr_std, 1),
        "rmssd_std": metrics.sdnn,  # Using SDNN as proxy for RMSSD variability
        "n": len(rr_list),
    }


def _check_anomaly(
    conn: Any,
    current_hr: float | None,
    current_rmssd: float | None,
) -> dict[str, Any] | None:
    """Check if current readings are anomalous vs. baseline."""
    baseline = _compute_baseline(conn)
    if baseline["rhr"] is None or baseline["rmssd"] is None:
        return None

    threshold = 1.5  # std deviations

    # Check HR anomaly
    if current_hr is not None and baseline["rhr_std"] and baseline["rhr_std"] > 0:
        hr_deviation = abs(current_hr - baseline["rhr"]) / baseline["rhr_std"]
        if hr_deviation > threshold:
            anomaly = {
                "metric": "rhr",
                "value": current_hr,
                "baseline": baseline["rhr"],
                "deviation": round(hr_deviation, 2),
            }
            conn.execute(
                "INSERT INTO anomalies (detected_at, metric, value, baseline, deviation) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), "rhr", current_hr, baseline["rhr"], hr_deviation),
            )
            conn.commit()
            return anomaly

    # Check RMSSD anomaly (low RMSSD = stressed)
    if current_rmssd is not None and baseline["rmssd_std"] and baseline["rmssd_std"] > 0:
        rmssd_deviation = abs(current_rmssd - baseline["rmssd"]) / baseline["rmssd_std"]
        if rmssd_deviation > threshold:
            anomaly = {
                "metric": "rmssd",
                "value": current_rmssd,
                "baseline": baseline["rmssd"],
                "deviation": round(rmssd_deviation, 2),
            }
            conn.execute(
                "INSERT INTO anomalies (detected_at, metric, value, baseline, deviation) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), "rmssd", current_rmssd, baseline["rmssd"], rmssd_deviation),
            )
            conn.commit()
            return anomaly

    return None


# ── Startup ─────────────────────────────────────────────────────────────────


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.post("/ingest/rr")
def ingest_rr(batch: RRBatch) -> dict[str, Any]:
    """Receive RR interval batch from mobile app."""
    conn = get_connection(DB_PATH)

    # Upsert session
    existing = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id=?",
        (batch.session_id,),
    ).fetchone()

    if not existing:
        start_session(
            conn, batch.session_id,
            label=batch.label or "unlabeled",
            device_name=batch.device_id,
        )

    # Insert readings
    inserted = 0
    for r in batch.readings:
        insert_rr(conn, batch.session_id, r.rr_ms, r.hr_bpm or 0, r.timestamp)
        inserted += 1

    conn.commit()

    # Check for anomalies on recent data
    rr_list = [r.rr_ms for r in batch.readings]
    current_hr = None
    current_rmssd = None
    if len(rr_list) >= 3:
        try:
            metrics = compute_hrv(rr_list)
            current_hr = metrics.mean_hr
            current_rmssd = metrics.rmssd
        except ValueError:
            pass

    anomaly = _check_anomaly(conn, current_hr, current_rmssd)

    conn.close()

    result: dict[str, Any] = {
        "status": "ok",
        "inserted": inserted,
        "session_id": batch.session_id,
    }
    if anomaly:
        result["anomaly"] = anomaly

    return result


@app.post("/ingest/context")
def tag_session(tag: ContextTag) -> dict[str, str]:
    """Label a session after the fact."""
    conn = get_connection(DB_PATH)
    conn.execute(
        "UPDATE sessions SET label = ? WHERE session_id = ?",
        (tag.label, tag.session_id),
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "label": tag.label}


@app.get("/status")
def get_status() -> dict[str, Any]:
    """Current HRV state — last 60 readings."""
    conn = get_connection(DB_PATH)
    rows = conn.execute(
        "SELECT rr_ms FROM rr_intervals ORDER BY id DESC LIMIT 60"
    ).fetchall()
    conn.close()

    rr_list = [r["rr_ms"] for r in rows]

    if len(rr_list) < 3:
        return {
            "rhr_bpm": None,
            "rmssd_ms": None,
            "body_state": None,
            "readings_in_window": len(rr_list),
            "timestamp": datetime.now().isoformat(),
        }

    try:
        metrics = compute_hrv(rr_list)
        body_state = classify_body_state(metrics)
    except ValueError:
        return {
            "rhr_bpm": None,
            "rmssd_ms": None,
            "body_state": None,
            "readings_in_window": len(rr_list),
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "rhr_bpm": metrics.mean_hr,
        "rmssd_ms": metrics.rmssd,
        "sdnn_ms": metrics.sdnn,
        "body_state": body_state,
        "readings_in_window": len(rr_list),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/baseline")
def get_baseline() -> dict[str, Any]:
    """Baseline from morning_baseline labeled sessions."""
    conn = get_connection(DB_PATH)
    baseline = _compute_baseline(conn)
    conn.close()

    if baseline["rhr"] is None:
        return {
            "status": "insufficient_data",
            "message": "Run more morning_baseline sessions",
            "sample_size": baseline["n"],
        }

    return {
        "baseline_rhr_bpm": baseline["rhr"],
        "baseline_rmssd_ms": baseline["rmssd"],
        "rhr_std": baseline["rhr_std"],
        "rmssd_std": baseline["rmssd_std"],
        "sample_size": baseline["n"],
        "status": "ok",
    }


@app.get("/anomalies")
def get_anomalies(unacknowledged_only: bool = True) -> list[dict[str, Any]]:
    """Return flagged anomaly events."""
    conn = get_connection(DB_PATH)
    query = "SELECT * FROM anomalies"
    if unacknowledged_only:
        query += " WHERE acknowledged = 0"
    query += " ORDER BY detected_at DESC LIMIT 20"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/anomalies/{anomaly_id}/acknowledge")
def acknowledge_anomaly(anomaly_id: int) -> dict[str, str]:
    """Mark an anomaly as acknowledged."""
    conn = get_connection(DB_PATH)
    conn.execute(
        "UPDATE anomalies SET acknowledged = 1 WHERE id = ?",
        (anomaly_id,),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/sessions")
def list_sessions(limit: int = 10) -> list[dict[str, Any]]:
    """List recent sessions with summary data."""
    conn = get_connection(DB_PATH)
    sessions = get_recent_sessions(conn, limit=limit)
    conn.close()
    return sessions


@app.post("/probe")
async def probe(req: ProbeRequest) -> dict[str, Any]:
    """Generate a probe for an anomaly. Optionally store response as memory."""
    from soma.proto_self.probe_generator import generate_probe
    from soma.proto_self.memory_writer import write_memory

    conn = get_connection(DB_PATH)
    if req.anomaly_id:
        row = conn.execute(
            "SELECT id, detected_at, metric, value, baseline, deviation "
            "FROM anomalies WHERE id = ?",
            (req.anomaly_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, detected_at, metric, value, baseline, deviation "
            "FROM anomalies WHERE acknowledged = 0 "
            "ORDER BY detected_at DESC LIMIT 1"
        ).fetchone()
    conn.close()

    if not row:
        return {"status": "no_anomaly", "message": "No unacknowledged anomalies found"}

    anomaly = dict(row)
    probe_text, state_info, similar_memories = await generate_probe(anomaly)

    if req.generate_only or not req.message:
        return {
            "status": "probe_generated",
            "probe": probe_text,
            "anomaly": anomaly,
            "state": state_info,
            "similar_memories_found": len(similar_memories),
        }

    memory_id, extracted = await write_memory(
        anomaly, state_info, probe_text, req.message
    )

    return {
        "status": "memory_stored",
        "probe": probe_text,
        "memory_id": memory_id,
        "entities_extracted": extracted.get("entities", []),
        "emotion_valence": extracted.get("emotion_valence", 0),
        "primary_topic": extracted.get("primary_topic", "other"),
    }
