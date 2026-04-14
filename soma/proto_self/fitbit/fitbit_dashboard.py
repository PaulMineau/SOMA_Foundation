"""Fitbit data access for dashboard and state classifier.

Reads from the LanceDB proto_self_fitbit table populated by soma_fitbit_ingestor.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import lancedb  # type: ignore[import-untyped]

SOMA_DB_PATH = os.path.expanduser("~/soma/soma.db")
TABLE_NAME = "proto_self_fitbit"


def get_fitbit_db() -> lancedb.DBConnection | None:
    if not os.path.exists(SOMA_DB_PATH):
        return None
    return lancedb.connect(SOMA_DB_PATH)


def get_recent_fitbit_days(n: int = 7) -> list[dict[str, Any]]:
    """Get the last N days of Fitbit data, most recent first."""
    db = get_fitbit_db()
    if db is None:
        return []

    table_names = db.table_names()
    if TABLE_NAME not in table_names:
        return []

    tbl = db.open_table(TABLE_NAME)
    try:
        df = tbl.to_pandas()
        if df.empty:
            return []
        # Filter out the init record
        df = df[df["date"] != "2026-01-01"]
        df = df.sort_values("date", ascending=False)
        records = df.head(n).to_dict("records")
        return records
    except Exception:
        return []


def get_today_fitbit() -> dict[str, Any] | None:
    """Get today's Fitbit data (or yesterday's if today not yet available)."""
    records = get_recent_fitbit_days(n=2)
    if not records:
        return None

    today_str = date.today().strftime("%Y-%m-%d")
    yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    for r in records:
        if r.get("date") == today_str:
            return r
    for r in records:
        if r.get("date") == yesterday_str:
            return r

    return records[0] if records else None


def get_fitbit_trends(days: int = 7) -> dict[str, Any]:
    """Compute trends over the last N days."""
    records = get_recent_fitbit_days(n=days)
    if len(records) < 2:
        return {}

    # Reverse to chronological order
    records = list(reversed(records))

    valid_hrv = [r["hrv_rmssd"] for r in records if r.get("hrv_rmssd", 0) > 0]
    valid_rhr = [r["resting_hr"] for r in records if r.get("resting_hr", 0) > 0]
    valid_sleep = [r["sleep_duration_min"] for r in records if r.get("sleep_duration_min", 0) > 0]
    valid_deep = [r["deep_sleep_min"] for r in records if r.get("deep_sleep_min", 0) > 0]
    recovery_scores = [r["recovery_score"] for r in records if r.get("recovery_score", 0) > 0]

    trends: dict[str, Any] = {"days": len(records)}

    if valid_hrv:
        trends["avg_hrv"] = round(sum(valid_hrv) / len(valid_hrv), 1)
    if valid_rhr:
        trends["avg_rhr"] = round(sum(valid_rhr) / len(valid_rhr), 1)
    if valid_sleep:
        trends["avg_sleep_hrs"] = round(sum(valid_sleep) / len(valid_sleep) / 60, 1)
    if valid_deep:
        trends["avg_deep_min"] = round(sum(valid_deep) / len(valid_deep))
    if recovery_scores:
        trends["avg_recovery"] = round(sum(recovery_scores) / len(recovery_scores), 1)

    # Direction trends (compare first half to second half)
    if len(recovery_scores) >= 4:
        mid = len(recovery_scores) // 2
        early = sum(recovery_scores[:mid]) / mid
        late = sum(recovery_scores[mid:]) / (len(recovery_scores) - mid)
        delta = late - early
        trends["recovery_trend"] = "improving" if delta > 0.5 else "declining" if delta < -0.5 else "stable"
        trends["recovery_delta"] = round(delta, 1)

    return trends
