"""Affective embedding space — merges somatic, affect, and memory into unified state.

The AffectiveEmbedding is the core representation: what it feels like to be
the patient right now. Both machine-readable (float vector) and human-readable
(text description injected into every LLM prompt).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from soma.brain.embeddings import (
    AffectiveEmbedding,
    AffectVec,
    MemoryContext,
    SomaticEmbedding,
    ThalamusEmbedding,
)

logger = logging.getLogger(__name__)


class AffectiveSpaceMerger:
    """Merges somatic, affect, and memory into the unified AffectiveEmbedding."""

    def merge(
        self,
        somatic: SomaticEmbedding,
        affect: AffectVec,
        memory: MemoryContext | None,
        routing: ThalamusEmbedding,
    ) -> AffectiveEmbedding:
        """Merge all signals into a unified affective embedding."""
        vector = self._compute_vector(somatic, affect, memory, routing)
        description = self._compose_description(somatic, affect, memory, routing)
        confidence = self._compute_confidence(somatic, routing)

        return AffectiveEmbedding(
            valence=affect.valence,
            arousal=affect.arousal,
            somatic_load=somatic.load,
            dominant_drive=affect.dominant_drive,
            vector=vector,
            description=description,
            source_somatic=somatic,
            source_affect=affect,
            source_memory=memory,
            confidence=confidence,
        )

    def _compute_vector(
        self,
        somatic: SomaticEmbedding,
        affect: AffectVec,
        memory: MemoryContext | None,
        routing: ThalamusEmbedding,
    ) -> NDArray[np.float32]:
        """Combine component vectors into 128-dim unified embedding."""
        # Somatic: 32-dim -> weighted
        somatic_weighted = somatic.vector * routing.biosensor_weight

        # Affect: encode as small vector
        affect_vec = np.array([
            affect.valence,
            affect.arousal,
            affect.low_road_contribution,
            affect.high_road_contribution,
        ], dtype=np.float32) * routing.semantic_weight

        # Memory: 128-dim -> take first 32
        if memory is not None and memory.vector is not None:
            memory_vec = memory.vector[:32] * 0.2
        else:
            memory_vec = np.zeros(32, dtype=np.float32)

        # Concatenate and pad to 128
        raw = np.concatenate([somatic_weighted[:32], affect_vec, memory_vec])
        result = np.zeros(128, dtype=np.float32)
        n = min(len(raw), 128)
        result[:n] = raw[:n]

        return result

    def _compose_description(
        self,
        somatic: SomaticEmbedding,
        affect: AffectVec,
        memory: MemoryContext | None,
        routing: ThalamusEmbedding,
    ) -> str:
        """Compose the natural language description injected into every LLM prompt.

        This is the most important field — it carries the felt state into language.
        """
        parts: list[str] = []

        # Affective state
        valence_word = (
            "positive" if affect.valence > 0.3 else
            "negative" if affect.valence < -0.3 else
            "neutral"
        )
        parts.append(
            f"Current state: arousal {affect.arousal:.2f}, "
            f"valence {valence_word} ({affect.valence:.2f}). "
            f"Dominant drive: {affect.dominant_drive}."
        )

        # Somatic
        parts.append(f"Body signal: {somatic.description}")

        # Routing context
        if routing.low_road_flag:
            parts.append("LOW ROAD ACTIVE — fast threat response bypassing cortex.")
        parts.append(f"Signal class: {routing.signal_classification}.")

        # Memory
        if memory and memory.similar_moments:
            n = len(memory.similar_moments)
            parts.append(f"Memory: {n} similar past moments found.")
            if memory.pattern_note:
                parts.append(f"Pattern: {memory.pattern_note}")

        return " ".join(parts)

    def _compute_confidence(
        self,
        somatic: SomaticEmbedding,
        routing: ThalamusEmbedding,
    ) -> float:
        """How much data backed this cycle. 0 = no signal, 1 = all sensors live."""
        score = 0.0

        # Somatic signal present?
        if somatic.rmssd > 0:
            score += 0.5

        # Routing had data to work with?
        if routing.biosensor_weight > 0.3:
            score += 0.3

        # Non-trivial somatic description?
        if "suppressed" in somatic.description or "elevated" in somatic.description:
            score += 0.2

        return min(1.0, score)
