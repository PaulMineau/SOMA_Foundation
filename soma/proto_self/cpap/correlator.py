"""CPAP correlator — links sleep-disordered breathing to next-day recovery.

The hypothesis: high-AHI nights suppress next-day HRV and increase RHR.
This module quantifies that lag-correlation across the patient's history.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def correlate_cpap_to_recovery(days: int = 30) -> dict[str, Any]:
    """Correlate CPAP AHI with next-day Fitbit recovery metrics.

    Pairs each CPAP night with the Fitbit data from the following day.
    Computes simple correlations: AHI vs recovery score, leak vs HRV, etc.

    Returns summary stats + the paired data for charting.
    """
    from soma.proto_self.cpap.cpap_ingestor import get_recent_cpap_days
    from soma.proto_self.fitbit.fitbit_dashboard import get_recent_fitbit_days

    cpap_days = get_recent_cpap_days(n=days)
    fitbit_days = get_recent_fitbit_days(n=days)

    # Index fitbit by date for lookup
    fb_by_date = {r["date"]: r for r in fitbit_days}

    paired: list[dict[str, Any]] = []
    for cpap in cpap_days:
        cpap_date = cpap["date"]
        # CPAP record is for night ending on cpap_date — Fitbit "next day" is same date
        # (since Fitbit date = wake day)
        fb = fb_by_date.get(cpap_date)
        if not fb:
            continue

        paired.append({
            "date": cpap_date,
            "ahi": cpap.get("ahi", 0),
            "usage_min": cpap.get("usage_min", 0),
            "leak_p95": cpap.get("leak_p95", 0) or cpap.get("leak_percentile", 0),
            "cpap_score": cpap.get("sleep_score", 0),
            "recovery_score": fb.get("recovery_score", 0),
            "fitbit_hrv": fb.get("hrv_rmssd", 0),
            "resting_hr": fb.get("resting_hr", 0),
            "deep_sleep_min": fb.get("deep_sleep_min", 0),
            "spo2_avg": fb.get("spo2_avg", 0),
        })

    if len(paired) < 3:
        return {"n": len(paired), "insufficient_data": True, "paired": paired}

    # Simple Pearson-style correlations
    def _pearson(xs: list[float], ys: list[float]) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            return 0.0
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
        denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
        if denom_x == 0 or denom_y == 0:
            return 0.0
        return num / (denom_x * denom_y)

    ahi = [p["ahi"] for p in paired if p["ahi"] > 0]
    recovery = [p["recovery_score"] for p in paired if p["ahi"] > 0]
    hrv = [p["fitbit_hrv"] for p in paired if p["ahi"] > 0 and p["fitbit_hrv"] > 0]
    rhr = [p["resting_hr"] for p in paired if p["ahi"] > 0 and p["resting_hr"] > 0]
    spo2 = [p["spo2_avg"] for p in paired if p["ahi"] > 0 and p["spo2_avg"] > 0]

    # Pair up AHI with corresponding other metric (same indices)
    def _pair_data(other_key: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for p in paired:
            if p["ahi"] > 0 and p.get(other_key, 0) > 0:
                xs.append(p["ahi"])
                ys.append(p[other_key])
        return xs, ys

    ahi_x, rec_y = _pair_data("recovery_score")
    ahi_x2, hrv_y = _pair_data("fitbit_hrv")
    ahi_x3, rhr_y = _pair_data("resting_hr")
    ahi_x4, spo2_y = _pair_data("spo2_avg")

    correlations = {
        "ahi_vs_recovery": round(_pearson(ahi_x, rec_y), 3),
        "ahi_vs_hrv": round(_pearson(ahi_x2, hrv_y), 3),
        "ahi_vs_rhr": round(_pearson(ahi_x3, rhr_y), 3),
        "ahi_vs_spo2": round(_pearson(ahi_x4, spo2_y), 3),
    }

    # Interpretation hints
    insights: list[str] = []
    if correlations["ahi_vs_recovery"] < -0.3:
        insights.append("Strong: higher AHI nights are followed by lower recovery scores.")
    elif correlations["ahi_vs_recovery"] < -0.15:
        insights.append("Moderate: AHI tends to suppress next-day recovery.")

    if correlations["ahi_vs_hrv"] < -0.3:
        insights.append("Higher AHI correlates with lower next-day HRV — autonomic cost is measurable.")

    if correlations["ahi_vs_rhr"] > 0.3:
        insights.append("Higher AHI correlates with elevated next-day resting HR.")

    if correlations["ahi_vs_spo2"] < -0.3:
        insights.append("Higher AHI correlates with lower overnight SpO2 — hypoxic burden.")

    return {
        "n": len(paired),
        "correlations": correlations,
        "insights": insights,
        "paired": paired,
    }


def get_compliance_stats(days: int = 30) -> dict[str, Any]:
    """Compliance = % of nights with >= 4h CPAP usage."""
    from soma.proto_self.cpap.cpap_ingestor import get_recent_cpap_days

    cpap_days = get_recent_cpap_days(n=days)
    if not cpap_days:
        return {"n": 0, "compliance_pct": 0.0, "avg_usage_hrs": 0.0, "avg_ahi": 0.0}

    compliant = sum(1 for d in cpap_days if d.get("usage_min", 0) >= 240)
    total = len(cpap_days)
    usages = [d.get("usage_min", 0) / 60 for d in cpap_days if d.get("usage_min", 0) > 0]
    ahis = [d.get("ahi", 0) for d in cpap_days if d.get("ahi", 0) > 0]

    return {
        "n": total,
        "compliance_pct": round(100 * compliant / total, 1) if total else 0,
        "avg_usage_hrs": round(sum(usages) / len(usages), 1) if usages else 0,
        "avg_ahi": round(sum(ahis) / len(ahis), 2) if ahis else 0,
    }
