"""HRV metric computation from RR intervals.

Computes standard time-domain and frequency-domain HRV metrics
from RR interval data collected by the Polar H10.

Reference: Task Force of ESC/NASPE (1996) — Heart rate variability:
standards of measurement, physiological interpretation and clinical use.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Artifact rejection: RR intervals outside this range are likely noise
MIN_RR_MS = 300.0   # ~200 bpm
MAX_RR_MS = 2000.0  # ~30 bpm
MAX_RR_DIFF_MS = 300.0  # Max acceptable beat-to-beat change


@dataclass(frozen=True)
class HRVMetrics:
    """Standard HRV metrics computed from RR intervals."""

    # Time-domain
    mean_rr: float       # Mean RR interval (ms)
    sdnn: float          # Standard deviation of NN intervals (ms)
    rmssd: float         # Root mean square of successive differences (ms)
    pnn50: float         # % of successive intervals differing by > 50ms
    mean_hr: float       # Mean heart rate (bpm)

    # Counts
    n_intervals: int     # Total RR intervals used
    n_artifacts: int     # Intervals rejected as artifacts

    # Window metadata
    window_seconds: float  # Duration of the analysis window


def clean_rr_intervals(
    rr_intervals: list[float],
) -> NDArray[np.float64]:
    """Remove artifact RR intervals.

    Rejects intervals that are:
    - Outside physiological range (300-2000ms)
    - Sudden jumps > 300ms from the previous interval (ectopic beats)
    """
    if not rr_intervals:
        return np.array([], dtype=np.float64)

    cleaned: list[float] = []
    prev: float | None = None

    for rr in rr_intervals:
        # Range check
        if rr < MIN_RR_MS or rr > MAX_RR_MS:
            continue

        # Successive difference check
        if prev is not None and abs(rr - prev) > MAX_RR_DIFF_MS:
            prev = rr  # Update prev but don't include this interval
            continue

        cleaned.append(rr)
        prev = rr

    return np.array(cleaned, dtype=np.float64)


def compute_hrv(
    rr_intervals: list[float],
    window_seconds: float = 0.0,
) -> HRVMetrics:
    """Compute time-domain HRV metrics from RR intervals.

    Args:
        rr_intervals: Raw RR intervals in milliseconds.
        window_seconds: Duration of the collection window (for metadata).

    Returns:
        HRVMetrics with all computed values.

    Raises:
        ValueError: If fewer than 3 clean RR intervals remain after filtering.
    """
    n_raw = len(rr_intervals)
    clean = clean_rr_intervals(rr_intervals)
    n_clean = len(clean)
    n_artifacts = n_raw - n_clean

    if n_clean < 3:
        raise ValueError(
            f"Insufficient clean RR intervals: {n_clean} "
            f"(need >= 3, had {n_raw} raw, {n_artifacts} artifacts)"
        )

    # Mean RR and HR
    mean_rr = float(np.mean(clean))
    mean_hr = 60000.0 / mean_rr  # Convert ms to bpm

    # SDNN — standard deviation of all NN intervals
    sdnn = float(np.std(clean, ddof=1))

    # Successive differences
    diffs = np.diff(clean)

    # RMSSD — root mean square of successive differences
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))

    # pNN50 — percentage of successive differences > 50ms
    nn50 = int(np.sum(np.abs(diffs) > 50.0))
    pnn50 = (nn50 / len(diffs)) * 100.0

    metrics = HRVMetrics(
        mean_rr=round(mean_rr, 1),
        sdnn=round(sdnn, 1),
        rmssd=round(rmssd, 1),
        pnn50=round(pnn50, 1),
        mean_hr=round(mean_hr, 1),
        n_intervals=n_clean,
        n_artifacts=n_artifacts,
        window_seconds=window_seconds,
    )

    logger.info(
        "HRV: RMSSD=%.1fms SDNN=%.1fms pNN50=%.1f%% HR=%.0fbpm "
        "(%d intervals, %d artifacts)",
        metrics.rmssd, metrics.sdnn, metrics.pnn50, metrics.mean_hr,
        metrics.n_intervals, metrics.n_artifacts,
    )

    return metrics


def classify_body_state(metrics: HRVMetrics) -> str:
    """Classify physiological state from HRV metrics.

    Simple rule-based classification as a baseline before ML models.
    States map to Proto-Self body state vector categories.

    Returns one of: "recovery", "resting", "optimal", "stressed", "fatigued"
    """
    # High RMSSD + low HR = parasympathetic dominance = recovery
    if metrics.rmssd > 60 and metrics.mean_hr < 60:
        return "recovery"

    # Moderate RMSSD + normal HR = resting
    if metrics.rmssd > 40 and metrics.mean_hr < 75:
        return "resting"

    # Good RMSSD + moderate HR = optimal readiness
    if metrics.rmssd > 30 and metrics.mean_hr < 85:
        return "optimal"

    # Low RMSSD + elevated HR = sympathetic dominance = stressed
    if metrics.rmssd < 20 or metrics.mean_hr > 90:
        return "stressed"

    # Low RMSSD + normal/low HR = autonomic depletion
    if metrics.rmssd < 25 and metrics.mean_hr < 75:
        return "fatigued"

    return "resting"  # Default
