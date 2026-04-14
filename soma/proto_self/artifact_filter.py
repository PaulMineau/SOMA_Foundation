"""Artifact filter — clean RR signal before any computation.

Signal quality is the foundation. Every downstream computation depends on this.

Two-stage filter:
1. Range rejection: remove physiologically impossible intervals
2. Ectopic rejection: remove sudden jumps (movement artifact, strap loss, ectopic beats)
"""

from __future__ import annotations


def reject_range(rr_list: list[float], min_rr: float = 300, max_rr: float = 1500) -> list[float]:
    """Remove physiologically impossible intervals.

    300ms = 200 bpm (absolute human maximum)
    1500ms = 40 bpm (very low resting HR)
    """
    return [rr for rr in rr_list if min_rr <= rr <= max_rr]


def reject_ectopic(rr_list: list[float], threshold: float = 0.20) -> list[float]:
    """Remove intervals that differ from their neighbor by more than 20%.

    Catches movement artifact, strap contact loss, ectopic beats.
    """
    if not rr_list:
        return []
    clean = [rr_list[0]]
    for rr in rr_list[1:]:
        if abs(rr - clean[-1]) / clean[-1] < threshold:
            clean.append(rr)
    return clean


def clean_rr(rr_list: list[float]) -> list[float]:
    """Full pipeline: range filter -> ectopic rejection."""
    rr = reject_range(rr_list)
    rr = reject_ectopic(rr)
    return rr


def compute_rmssd(rr_list: list[float]) -> float | None:
    """Compute RMSSD from cleaned RR intervals."""
    rr = clean_rr(rr_list)
    if len(rr) < 2:
        return None
    diffs = [(rr[i + 1] - rr[i]) ** 2 for i in range(len(rr) - 1)]
    return round((sum(diffs) / len(diffs)) ** 0.5, 2)


def compute_rhr(rr_list: list[float]) -> float | None:
    """Compute resting heart rate from cleaned RR intervals."""
    rr = clean_rr(rr_list)
    if not rr:
        return None
    return round(60000 / (sum(rr) / len(rr)), 1)


if __name__ == "__main__":
    test = [820.0, 810.0, 835.0, 2000.0, 800.0, 150.0, 815.0]
    print(f"Raw:   {test}")
    print(f"Clean: {clean_rr(test)}")
    print(f"RMSSD: {compute_rmssd(test)}")
    print(f"RHR:   {compute_rhr(test)}")
