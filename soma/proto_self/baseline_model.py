"""Baseline model — compute and save physiological fingerprint from morning_baseline sessions.

Run after accumulating morning_baseline sessions. Re-run weekly as more data accumulates.

Usage:
    python -m soma.proto_self.baseline_model
    python -m soma.proto_self.baseline_model --min-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

from soma.proto_self.artifact_filter import clean_rr, compute_rhr, compute_rmssd
from soma.proto_self.db import DEFAULT_DB_PATH, get_connection

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("SOMA_BASELINE_MODEL", "data/baseline_model.json")
BASELINE_LABEL = "morning_baseline"
DEFAULT_MIN_SAMPLES = 100


def load_baseline_rr(db_path: str | None = None) -> list[float]:
    """Load all RR intervals from morning_baseline sessions."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT rr.rr_ms FROM rr_intervals rr "
        "JOIN sessions s ON rr.session_id = s.session_id "
        "WHERE s.label = ? ORDER BY rr.timestamp ASC",
        (BASELINE_LABEL,),
    ).fetchall()
    conn.close()
    return [r["rr_ms"] for r in rows]


def compute_stats(values: list[float]) -> tuple[float | None, float | None]:
    """Compute mean and standard deviation."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(mean, 2), round(math.sqrt(variance), 2)


def build_baseline(
    db_path: str | None = None,
    model_path: str | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict | None:
    """Build baseline model from morning_baseline sessions.

    Returns the model dict, or None if insufficient data.
    """
    model_path = model_path or MODEL_PATH

    raw_rr = load_baseline_rr(db_path)
    print(f"Raw samples loaded: {len(raw_rr)}")

    if len(raw_rr) < min_samples:
        print(f"Insufficient data. Need {min_samples} samples, have {len(raw_rr)}.")
        print("Run more morning_baseline sessions first.")
        return None

    clean = clean_rr(raw_rr)
    print(f"Clean samples after artifact rejection: {len(clean)}")

    if len(clean) < 10:
        print("Too few clean samples after artifact rejection.")
        return None

    # Per-beat HR stats
    rhr_values = [round(60000 / rr, 2) for rr in clean]
    mean_rhr, std_rhr = compute_stats(rhr_values)

    # RMSSD in rolling windows of 60 beats
    window_size = 60
    rmssd_values: list[float] = []
    for i in range(0, len(clean) - window_size, window_size):
        window = clean[i : i + window_size]
        rmssd = compute_rmssd(window)
        if rmssd is not None:
            rmssd_values.append(rmssd)

    mean_rmssd, std_rmssd = compute_stats(rmssd_values)

    if mean_rhr is None or std_rhr is None or mean_rmssd is None or std_rmssd is None:
        print("Could not compute baseline stats.")
        return None

    model = {
        "generated_at": datetime.now().isoformat(),
        "label": BASELINE_LABEL,
        "sample_count": len(clean),
        "rhr": {
            "mean": mean_rhr,
            "std": std_rhr,
            "alert_threshold_high": round(mean_rhr + 1.5 * std_rhr, 1),
            "alert_threshold_low": round(mean_rhr - 1.5 * std_rhr, 1),
        },
        "rmssd": {
            "mean": mean_rmssd,
            "std": std_rmssd,
            "alert_threshold_low": round(mean_rmssd - 1.5 * std_rmssd, 1),
        },
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"\nBaseline model saved to {model_path}")
    print(f"   RHR:   {mean_rhr} +/- {std_rhr} bpm")
    print(f"   RMSSD: {mean_rmssd} +/- {std_rmssd} ms")
    print(f"   Alert if RHR > {model['rhr']['alert_threshold_high']} bpm")
    print(f"   Alert if RMSSD < {model['rmssd']['alert_threshold_low']} ms")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOMA — Build baseline physiological model"
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    parser.add_argument("--output", default=MODEL_PATH, help="Output model JSON path")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help="Minimum RR samples required")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    build_baseline(db_path=args.db, model_path=args.output, min_samples=args.min_samples)


if __name__ == "__main__":
    main()
