"""State classifier — classify current physiological state from baseline.

States:
    depleted    — high stress, low HRV, body under load
    recovering  — returning toward baseline, transitional
    baseline    — normal resting state, nothing remarkable
    restored    — above baseline, parasympathetic dominant
    peak        — significantly above baseline, cognitive/creative window open
    unknown     — insufficient signal

Usage:
    python -m soma.proto_self.state_classifier
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys

from soma.proto_self.artifact_filter import clean_rr, compute_rhr, compute_rmssd
from soma.proto_self.baseline_model import MODEL_PATH
from soma.proto_self.db import DEFAULT_DB_PATH, get_connection

logger = logging.getLogger(__name__)

WINDOW_SIZE = 60


def load_model(model_path: str | None = None) -> dict:
    """Load the baseline model from JSON."""
    path = model_path or MODEL_PATH
    with open(path) as f:
        return json.load(f)


def get_recent_rr(db_path: str | None = None, n: int = WINDOW_SIZE) -> list[float]:
    """Get the N most recent RR intervals from the database."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT rr_ms FROM rr_intervals ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [r["rr_ms"] for r in rows]


def classify_state(
    db_path: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Classify current physiological state by comparing to baseline.

    Returns a dict with:
        state: str — one of depleted/recovering/baseline/restored/peak/unknown
        reason: str — human-readable explanation
        rhr: float | None — current resting heart rate
        rmssd: float | None — current RMSSD
        rhr_z: float — standard deviations from baseline RHR
        rmssd_z: float — standard deviations from baseline RMSSD
    """
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        return {
            "state": "unknown",
            "reason": "No baseline model. Run: python -m soma.proto_self.baseline_model",
            "rhr": None, "rmssd": None, "rhr_z": 0, "rmssd_z": 0,
        }

    rr_raw = get_recent_rr(db_path)
    rr_clean = clean_rr(rr_raw)

    if len(rr_clean) < 10:
        return {
            "state": "unknown",
            "reason": "Insufficient signal",
            "rhr": None, "rmssd": None, "rhr_z": 0, "rmssd_z": 0,
        }

    rhr = compute_rhr(rr_clean)
    rmssd = compute_rmssd(rr_clean)

    if rhr is None or rmssd is None:
        return {
            "state": "unknown",
            "reason": "Could not compute metrics",
            "rhr": rhr, "rmssd": rmssd, "rhr_z": 0, "rmssd_z": 0,
        }

    rhr_mean = model["rhr"]["mean"]
    rhr_std = model["rhr"]["std"]
    rmssd_mean = model["rmssd"]["mean"]
    rmssd_std = model["rmssd"]["std"]

    rhr_z = (rhr - rhr_mean) / rhr_std if rhr_std else 0
    rmssd_z = (rmssd - rmssd_mean) / rmssd_std if rmssd_std else 0

    # Classify
    if rmssd_z < -1.5 or rhr_z > 1.5:
        state = "depleted"
        reason = f"RMSSD {rmssd_z:.1f}s below baseline, RHR {rhr_z:.1f}s above"
    elif rmssd_z < -0.75 or rhr_z > 0.75:
        state = "recovering"
        reason = f"Below baseline but trending. RMSSD {rmssd_z:.1f}s"
    elif rmssd_z > 1.5 and rhr_z < -0.5:
        state = "peak"
        reason = f"RMSSD {rmssd_z:.1f}s above baseline. Cognitive window open."
    elif rmssd_z > 0.5:
        state = "restored"
        reason = "Above baseline. Parasympathetic dominant."
    else:
        state = "baseline"
        reason = "Within normal range."

    # Enrich with Fitbit overnight context if available
    fitbit_context: dict = {}
    try:
        from soma.proto_self.fitbit.fitbit_dashboard import get_today_fitbit
        fb = get_today_fitbit()
        if fb:
            fitbit_context = {
                "sleep_hrs": round(fb.get("sleep_duration_min", 0) / 60, 1),
                "deep_sleep_min": fb.get("deep_sleep_min", 0),
                "sleep_efficiency": fb.get("sleep_efficiency"),
                "overnight_hrv": fb.get("hrv_rmssd"),
                "recovery_score": fb.get("recovery_score", 0),
                "spo2_avg": fb.get("spo2_avg"),
                "steps": fb.get("steps", 0),
            }

            # Fitbit recovery score can shift state assessment
            recovery = fb.get("recovery_score", 5)
            sleep_hrs = fb.get("sleep_duration_min", 0) / 60

            if recovery <= 3 and state in ("baseline", "recovering"):
                state = "depleted"
                reason += f" Fitbit recovery {recovery}/10, sleep {sleep_hrs:.1f}h."
            elif recovery >= 8 and state == "baseline":
                state = "restored"
                reason += f" Fitbit recovery {recovery}/10."
    except Exception:
        pass

    # Enrich with CPAP context if available
    cpap_context: dict = {}
    try:
        from soma.proto_self.cpap.cpap_ingestor import get_recent_cpap_days
        cpap_days = get_recent_cpap_days(n=1)
        if cpap_days:
            cp = cpap_days[0]
            cpap_context = {
                "last_ahi": cp.get("ahi", 0),
                "last_usage_hrs": round(cp.get("usage_min", 0) / 60, 1),
                "last_cpap_score": cp.get("sleep_score", 0),
                "last_leak_p95": cp.get("leak_p95", 0) or cp.get("leak_percentile", 0),
            }

            # High AHI night suggests physiological cost
            ahi = cp.get("ahi", 0)
            usage_min = cp.get("usage_min", 0)

            if ahi >= 15 and state in ("baseline", "restored"):
                state = "recovering"
                reason += f" CPAP AHI {ahi:.1f} last night — hypoxic burden."
            elif usage_min < 240 and ahi >= 10:
                reason += f" CPAP under-used ({usage_min // 60}h) with AHI {ahi:.1f}."
    except Exception:
        pass

    return {
        "state": state,
        "reason": reason,
        "rhr": rhr,
        "rmssd": rmssd,
        "rhr_z": round(rhr_z, 2),
        "rmssd_z": round(rmssd_z, 2),
        "fitbit": fitbit_context,
        "cpap": cpap_context,
    }


def main() -> None:
    s = classify_state()
    print(f"\nCurrent State: {s['state'].upper()}")
    print(f"   Reason: {s['reason']}")
    if s["rhr"] is not None:
        print(f"   RHR: {s['rhr']} bpm ({s['rhr_z']:+.1f}s)")
    if s["rmssd"] is not None:
        print(f"   RMSSD: {s['rmssd']} ms ({s['rmssd_z']:+.1f}s)")

    fb = s.get("fitbit", {})
    if fb:
        print(f"\n   Fitbit overnight:")
        if fb.get("overnight_hrv"):
            print(f"     HRV (overnight): {fb['overnight_hrv']:.1f} ms")
        if fb.get("sleep_hrs"):
            print(f"     Sleep: {fb['sleep_hrs']}h ({fb.get('deep_sleep_min', 0)}m deep)")
        if fb.get("recovery_score"):
            print(f"     Recovery: {fb['recovery_score']}/10")
        if fb.get("spo2_avg"):
            print(f"     SpO2: {fb['spo2_avg']:.1f}%")

    cp = s.get("cpap", {})
    if cp:
        print(f"\n   CPAP last night:")
        if cp.get("last_ahi") is not None:
            print(f"     AHI: {cp['last_ahi']:.1f}")
        if cp.get("last_usage_hrs"):
            print(f"     Usage: {cp['last_usage_hrs']}h")
        if cp.get("last_cpap_score"):
            print(f"     myAir Score: {cp['last_cpap_score']}/100")


if __name__ == "__main__":
    main()
