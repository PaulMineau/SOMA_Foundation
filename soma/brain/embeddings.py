"""Typed embedding schema — the nervous system protocol of SOMA.

Every embedding that passes between brain modules is a typed dataclass.
Each carries both a machine-readable vector and a human-readable description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SomaticEmbedding:
    """Raw body signal, processed by interoception module."""

    rmssd: float  # HRV metric, milliseconds
    rhr: float  # resting heart rate, bpm
    hrv_trend: float  # slope over last 5 min, normalized -1 to +1
    load: float  # 0.0-1.0, composite stress signal
    vector: NDArray[np.float32]  # 32-dim encoding
    description: str  # "HRV suppressed 18ms, RHR 72, rising load"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ThalamusEmbedding:
    """Routing weights — what to trust this cycle."""

    biosensor_weight: float  # 0.0-1.0
    semantic_weight: float  # 0.0-1.0
    visual_weight: float  # 0.0-1.0 (0 until camera live)
    low_road_flag: bool  # True = bypass cortex, fire amygdala direct
    signal_classification: str  # "resting" | "stress" | "exercise" | "arousal"
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AffectVec:
    """Amygdala output — the raw emotional signal."""

    valence: float  # -1.0 to +1.0
    arousal: float  # 0.0 to 1.0
    dominant_drive: str  # SEEKING | CARE | PLAY | GRIEF | FEAR | RAGE
    low_road_contribution: float  # how much came from raw biosensor
    high_road_contribution: float  # how much came from semantic context
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryContext:
    """Hippocampus retrieval — what's relevant from the past."""

    similar_moments: list[dict[str, Any]]  # top-3 retrieved episodes
    recency_weight: float  # how much recent memory dominates
    pattern_note: str | None  # "this pattern preceded X last time"
    vector: NDArray[np.float32]  # 128-dim
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AffectiveEmbedding:
    """The unified pre-linguistic state — what it feels like to be the patient right now.

    This is what gets injected into every LLM prompt downstream.
    """

    valence: float
    arousal: float
    somatic_load: float
    dominant_drive: str
    vector: NDArray[np.float32]  # 128-dim
    description: str  # injected into PFC system prompt
    source_somatic: SomaticEmbedding
    source_affect: AffectVec
    source_memory: MemoryContext | None
    confidence: float  # how much data backed this cycle
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PFCOutput:
    """Prefrontal cortex output — what SOMA says."""

    recommendation: str | None
    anomaly_flag: bool
    anomaly_description: str | None
    prediction: str | None
    question_for_patient: str | None
    model_used: str  # "claude-sonnet-4" or "qwen3:8b"
    source_embedding: AffectiveEmbedding
    timestamp: datetime = field(default_factory=datetime.now)
