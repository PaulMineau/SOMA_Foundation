"""Interoception module — reads body signal, produces SomaticEmbedding.

No LLM — this is fast, rule-based + numpy. Reads from existing SQLite
(polar_logger) and computes HRV metrics against personal baseline.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from soma.brain.embeddings import SomaticEmbedding
from soma.proto_self.artifact_filter import clean_rr, compute_rhr, compute_rmssd
from soma.proto_self.db import DEFAULT_DB_PATH, get_connection

logger = logging.getLogger(__name__)

# Default baseline values (overridden when baseline model exists)
DEFAULT_RMSSD_BASELINE = 40.0
DEFAULT_RHR_BASELINE = 72.0


class InteroceptionModule:
    """Processes raw RR intervals into a somatic state embedding."""

    WINDOW_SIZE = 300  # ~5 min of beats at 60bpm

    def __init__(
        self,
        db_path: str | None = None,
        rmssd_baseline: float = DEFAULT_RMSSD_BASELINE,
        rhr_baseline: float = DEFAULT_RHR_BASELINE,
    ) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.rmssd_baseline = rmssd_baseline
        self.rhr_baseline = rhr_baseline
        self._load_baseline()

    def _load_baseline(self) -> None:
        """Try to load personal baseline from the baseline model."""
        try:
            import json
            from soma.proto_self.baseline_model import MODEL_PATH

            with open(MODEL_PATH) as f:
                model = json.load(f)
            self.rmssd_baseline = model["rmssd"]["mean"]
            self.rhr_baseline = model["rhr"]["mean"]
            logger.info(
                "Loaded baseline: RMSSD=%.1f RHR=%.1f",
                self.rmssd_baseline, self.rhr_baseline,
            )
        except Exception:
            logger.info("Using default baseline (no model file found)")

    def fetch_rr_window(self) -> list[float]:
        """Read recent RR intervals from SQLite."""
        conn = get_connection(self.db_path)
        rows = conn.execute(
            "SELECT rr_ms FROM rr_intervals ORDER BY id DESC LIMIT ?",
            (self.WINDOW_SIZE,),
        ).fetchall()
        conn.close()
        return [r["rr_ms"] for r in rows]

    def process(self, rr_intervals: list[float] | None = None) -> SomaticEmbedding:
        """Process RR intervals into a SomaticEmbedding.

        If rr_intervals is None, reads from the database.
        """
        if rr_intervals is None:
            rr_intervals = self.fetch_rr_window()

        rr_clean = clean_rr(rr_intervals)

        if len(rr_clean) < 5:
            return self._empty_embedding("Insufficient signal")

        rmssd = compute_rmssd(rr_clean) or 0.0
        rhr = compute_rhr(rr_clean) or 0.0
        trend = self._compute_trend(rr_clean)
        load = self._compute_load(rmssd, rhr, trend)
        vector = self._encode_vector(rmssd, rhr, trend, load)
        description = self._describe(rmssd, rhr, trend, load)

        return SomaticEmbedding(
            rmssd=rmssd,
            rhr=rhr,
            hrv_trend=trend,
            load=load,
            vector=vector,
            description=description,
        )

    def _compute_trend(self, rr_clean: list[float]) -> float:
        """Compute HRV trend (slope) over the window. Normalized to -1..+1."""
        if len(rr_clean) < 10:
            return 0.0

        # Split into first half and second half, compare RMSSD
        mid = len(rr_clean) // 2
        first_rmssd = compute_rmssd(rr_clean[:mid]) or 0.0
        second_rmssd = compute_rmssd(rr_clean[mid:]) or 0.0

        if first_rmssd == 0:
            return 0.0

        # Positive = improving, negative = declining
        delta_pct = (second_rmssd - first_rmssd) / first_rmssd
        return max(-1.0, min(1.0, delta_pct))

    def _compute_load(self, rmssd: float, rhr: float, trend: float) -> float:
        """Composite stress load 0-1. High when: low HRV + high RHR + declining trend."""
        # Z-score components against personal baseline
        rmssd_z = (self.rmssd_baseline - rmssd) / max(self.rmssd_baseline, 1)  # Inverted: low HRV = high load
        rhr_z = (rhr - self.rhr_baseline) / max(self.rhr_baseline, 1)
        trend_penalty = max(0, -trend)  # Declining trend adds load

        raw_load = (rmssd_z * 0.5) + (rhr_z * 0.3) + (trend_penalty * 0.2)
        return max(0.0, min(1.0, raw_load))

    def _encode_vector(
        self, rmssd: float, rhr: float, trend: float, load: float
    ) -> NDArray[np.float32]:
        """Encode somatic state as a 32-dim vector."""
        # Normalized features repeated/padded to 32 dims
        features = np.array([
            rmssd / 100.0,          # normalize to ~0-1 range
            rhr / 100.0,
            trend,
            load,
            rmssd / max(self.rmssd_baseline, 1),  # ratio to baseline
            rhr / max(self.rhr_baseline, 1),
            1.0 if load > 0.6 else 0.0,  # stress flag
            1.0 if trend < -0.3 else 0.0,  # declining flag
        ], dtype=np.float32)

        # Pad to 32 dims with repeated features + noise for uniqueness
        padded = np.zeros(32, dtype=np.float32)
        padded[:len(features)] = features
        # Fill remaining with feature interactions
        for i in range(len(features), 32):
            padded[i] = features[i % len(features)] * features[(i + 1) % len(features)]

        return padded

    def _describe(self, rmssd: float, rhr: float, trend: float, load: float) -> str:
        """Natural language description for LLM injection."""
        parts: list[str] = []

        # RMSSD context
        rmssd_ratio = rmssd / max(self.rmssd_baseline, 1)
        if rmssd_ratio < 0.5:
            parts.append(f"HRV severely suppressed at {rmssd:.0f}ms (baseline {self.rmssd_baseline:.0f}ms)")
        elif rmssd_ratio < 0.75:
            parts.append(f"HRV below baseline at {rmssd:.0f}ms (baseline {self.rmssd_baseline:.0f}ms)")
        elif rmssd_ratio > 1.3:
            parts.append(f"HRV elevated at {rmssd:.0f}ms (baseline {self.rmssd_baseline:.0f}ms)")
        else:
            parts.append(f"HRV near baseline at {rmssd:.0f}ms")

        # RHR context
        parts.append(f"RHR {rhr:.0f}bpm")

        # Trend
        if trend < -0.2:
            parts.append("declining trend")
        elif trend > 0.2:
            parts.append("improving trend")

        # Load summary
        if load > 0.7:
            parts.append("high somatic load")
        elif load > 0.4:
            parts.append("moderate load")
        else:
            parts.append("low load")

        return ". ".join(parts) + "."

    def _empty_embedding(self, reason: str) -> SomaticEmbedding:
        return SomaticEmbedding(
            rmssd=0.0,
            rhr=0.0,
            hrv_trend=0.0,
            load=0.0,
            vector=np.zeros(32, dtype=np.float32),
            description=f"No somatic data: {reason}",
        )
