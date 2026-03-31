"""
Memory Consolidation — Inspired by hippocampal memory replay during sleep.

Memories with high prediction error and high affective intensity get strengthened.
Routine memories decay. The mountain stays. The 47th similar lunch fades.
"""

from __future__ import annotations

import math

from soma.memory.episodic_store import InMemoryEpisodicStore


class Consolidator:
    """Runs consolidation over the episodic store."""

    def __init__(
        self,
        affect_weight: float = 0.5,
        prediction_error_weight: float = 0.5,
        temporal_decay_rate: float = 0.01,
        recency_boost: float = 0.1,
    ) -> None:
        self.affect_weight = affect_weight
        self.prediction_error_weight = prediction_error_weight
        self.temporal_decay_rate = temporal_decay_rate
        self.recency_boost = recency_boost

    def consolidate(self, store: InMemoryEpisodicStore, current_time: float) -> None:
        """Run one consolidation cycle over all memories in the store."""
        for memory in store.all_memories():
            weighted_salience = (
                self.affect_weight * memory.affect_intensity
                + self.prediction_error_weight * memory.prediction_error
            )

            # Interaction term — somatic marker effect
            interaction = memory.affect_intensity * memory.prediction_error
            weighted_salience += interaction

            # Temporal decay
            age_seconds = max(0, current_time - memory.created_at)
            age_days = age_seconds / 86400.0
            temporal_factor = math.exp(-self.temporal_decay_rate * age_days)

            # Access frequency boost (reconsolidation)
            access_boost = 1.0 + (self.recency_boost * memory.access_count)

            memory.consolidation_score = weighted_salience * temporal_factor * access_boost

    def consolidate_n_cycles(
        self,
        store: InMemoryEpisodicStore,
        n_cycles: int,
        start_time: float,
        cycle_interval_days: float = 1.0,
    ) -> None:
        """Run multiple consolidation cycles (simulating multiple nights of sleep)."""
        for i in range(n_cycles):
            cycle_time = start_time + (i * cycle_interval_days * 86400)
            self.consolidate(store, cycle_time)
