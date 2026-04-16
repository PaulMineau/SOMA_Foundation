"""Tests for SOMA Brain — embeddings, interoception, affective space, state bus."""

from __future__ import annotations

import asyncio
import numpy as np

from soma.brain.embeddings import (
    AffectVec,
    AffectiveEmbedding,
    MemoryContext,
    PFCOutput,
    SomaticEmbedding,
    ThalamusEmbedding,
)
from soma.brain.interoception import InteroceptionModule
from soma.brain.affective_space import AffectiveSpaceMerger
from soma.brain.state_bus import StateBus


def _make_somatic(rmssd: float = 40.0, rhr: float = 72.0, load: float = 0.3) -> SomaticEmbedding:
    return SomaticEmbedding(
        rmssd=rmssd, rhr=rhr, hrv_trend=0.0, load=load,
        vector=np.zeros(32, dtype=np.float32),
        description=f"RMSSD {rmssd}ms, RHR {rhr}bpm, load {load}",
    )


def _make_routing(low_road: bool = False) -> ThalamusEmbedding:
    return ThalamusEmbedding(
        biosensor_weight=0.7, semantic_weight=0.3, visual_weight=0.0,
        low_road_flag=low_road, signal_classification="resting",
        description="Normal routing",
    )


def _make_affect(valence: float = 0.0, arousal: float = 0.4) -> AffectVec:
    return AffectVec(
        valence=valence, arousal=arousal, dominant_drive="SEEKING",
        low_road_contribution=0.3, high_road_contribution=0.7,
        description=f"Valence {valence}, arousal {arousal}",
    )


class TestEmbeddings:
    def test_somatic_embedding_fields(self) -> None:
        s = _make_somatic(rmssd=18.0, rhr=80.0, load=0.7)
        assert s.rmssd == 18.0
        assert s.load == 0.7
        assert len(s.vector) == 32

    def test_thalamus_embedding(self) -> None:
        t = _make_routing(low_road=True)
        assert t.low_road_flag is True
        assert t.biosensor_weight + t.semantic_weight + t.visual_weight == 1.0

    def test_affect_vec(self) -> None:
        a = _make_affect(valence=-0.5, arousal=0.8)
        assert a.valence == -0.5
        assert a.dominant_drive == "SEEKING"

    def test_memory_context(self) -> None:
        m = MemoryContext(
            similar_moments=[{"text": "test", "valence": 0.1}],
            recency_weight=0.5,
            pattern_note="Test pattern",
            vector=np.zeros(128, dtype=np.float32),
            description="1 similar moment",
        )
        assert len(m.similar_moments) == 1
        assert m.pattern_note == "Test pattern"


class TestInteroception:
    def test_process_with_data(self) -> None:
        module = InteroceptionModule()
        # Simulate 60 beats at ~75bpm with variation
        rr = [800.0 + (i % 5) * 3 for i in range(60)]
        somatic = module.process(rr)

        assert somatic.rmssd > 0
        assert somatic.rhr > 0
        assert 0.0 <= somatic.load <= 1.0
        assert len(somatic.vector) == 32
        assert len(somatic.description) > 0

    def test_process_insufficient_data(self) -> None:
        module = InteroceptionModule()
        somatic = module.process([800.0, 810.0])

        assert somatic.rmssd == 0.0
        assert "Insufficient" in somatic.description

    def test_high_load_detected(self) -> None:
        module = InteroceptionModule()
        # Fast HR, low variability = high load
        rr = [550.0 + (i % 2) for i in range(60)]
        somatic = module.process(rr)

        assert somatic.load > 0.3  # Should be elevated
        assert somatic.rhr > 100

    def test_trend_computation(self) -> None:
        module = InteroceptionModule()
        # First half: steady, second half: more variable (improving HRV)
        rr = [800.0 + (i % 2) for i in range(30)] + [780.0 + (i % 10) * 5 for i in range(30)]
        somatic = module.process(rr)

        # Should compute some trend (direction depends on variability change)
        assert somatic.rmssd > 0


class TestAffectiveSpace:
    def test_merge_produces_128dim(self) -> None:
        merger = AffectiveSpaceMerger()
        somatic = _make_somatic()
        affect = _make_affect()
        routing = _make_routing()

        embedding = merger.merge(somatic, affect, None, routing)

        assert len(embedding.vector) == 128
        assert embedding.valence == affect.valence
        assert embedding.somatic_load == somatic.load
        assert embedding.source_somatic is somatic
        assert embedding.source_affect is affect

    def test_description_includes_state(self) -> None:
        merger = AffectiveSpaceMerger()
        somatic = _make_somatic(load=0.8)
        affect = _make_affect(valence=-0.5, arousal=0.7)
        routing = _make_routing()

        embedding = merger.merge(somatic, affect, None, routing)

        assert "negative" in embedding.description.lower()
        assert "SEEKING" in embedding.description

    def test_low_road_noted_in_description(self) -> None:
        merger = AffectiveSpaceMerger()
        somatic = _make_somatic(load=0.9)
        affect = _make_affect(valence=-0.8, arousal=0.9)
        routing = _make_routing(low_road=True)

        embedding = merger.merge(somatic, affect, None, routing)

        assert "LOW ROAD" in embedding.description

    def test_confidence_with_signal(self) -> None:
        merger = AffectiveSpaceMerger()
        somatic = _make_somatic(rmssd=40.0)
        affect = _make_affect()
        routing = _make_routing()

        embedding = merger.merge(somatic, affect, None, routing)
        assert embedding.confidence > 0.0

    def test_confidence_without_signal(self) -> None:
        merger = AffectiveSpaceMerger()
        somatic = _make_somatic(rmssd=0.0)
        affect = _make_affect()
        routing = ThalamusEmbedding(
            biosensor_weight=0.1, semantic_weight=0.9, visual_weight=0.0,
            low_road_flag=False, signal_classification="unknown",
            description="No biosensor",
        )

        embedding = merger.merge(somatic, affect, None, routing)
        assert embedding.confidence < 0.5


class TestStateBus:
    def test_publish_and_get(self) -> None:
        bus = StateBus()

        async def _test():
            await bus.publish("test_key", {"value": 42})
            result = await bus.get("test_key")
            assert result == {"value": 42}

        asyncio.run(_test())

    def test_get_all(self) -> None:
        bus = StateBus()

        async def _test():
            await bus.publish("a", 1)
            await bus.publish("b", 2)
            all_state = await bus.get_all()
            assert all_state == {"a": 1, "b": 2}

        asyncio.run(_test())

    def test_sync_access(self) -> None:
        bus = StateBus()

        async def _test():
            await bus.publish("sync_test", "hello")

        asyncio.run(_test())
        assert bus.get_sync("sync_test") == "hello"
