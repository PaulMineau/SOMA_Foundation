"""Tests for Week 4: autobiographical store, probe generation, memory writing."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

from soma.proto_self.autobiographical_store import (
    get_recent_memories,
    retrieve_similar_memories,
    store_exchange,
    store_narrative,
)


def _temp_lancedb() -> str:
    d = tempfile.mkdtemp()
    return d


class TestAutobiographicalStore:
    def test_store_and_retrieve(self) -> None:
        db_path = _temp_lancedb()
        try:
            memory_id = store_exchange(
                anomaly_type="hrv_anomaly",
                metric="rmssd",
                value=18.3,
                baseline=41.0,
                deviation=-2.3,
                body_state="depleted",
                probe_text="Your HRV is low. What's going on?",
                response_text="Rough meeting at work.",
                session_label="work_stress",
                entities=["work", "meeting"],
                emotion_valence=-0.5,
                db_path=db_path,
            )

            assert memory_id.startswith("mem_")

            memories = get_recent_memories(n=5, db_path=db_path)
            assert len(memories) == 1
            assert memories[0]["response_text"] == "Rough meeting at work."
            assert memories[0]["body_state"] == "depleted"
        finally:
            import shutil
            shutil.rmtree(db_path, ignore_errors=True)

    def test_semantic_search(self) -> None:
        db_path = _temp_lancedb()
        try:
            # Store two different exchanges
            store_exchange(
                anomaly_type="hrv_anomaly", metric="rmssd", value=18.0,
                baseline=41.0, deviation=-2.3, body_state="depleted",
                probe_text="HRV low", response_text="Work stress with Joe",
                entities=["Joe", "work"], db_path=db_path,
            )
            store_exchange(
                anomaly_type="hrv_anomaly", metric="rhr", value=90.0,
                baseline=72.0, deviation=2.0, body_state="stressed",
                probe_text="HR elevated", response_text="Just finished a hard run",
                entities=["running", "exercise"], db_path=db_path,
            )

            # Search for work-related memories
            results = retrieve_similar_memories(
                "work stress meeting", n=5, db_path=db_path
            )
            assert len(results) >= 1
        finally:
            import shutil
            shutil.rmtree(db_path, ignore_errors=True)

    def test_store_narrative(self) -> None:
        db_path = _temp_lancedb()
        try:
            store_narrative(
                narrative="You had a rough week. Work stress dominated.",
                week_number=15,
                year=2026,
                dominant_state="depleted",
                patterns=["work", "stress"],
                db_path=db_path,
            )
            # Just verify it doesn't crash — narrative retrieval is via LanceDB
        finally:
            import shutil
            shutil.rmtree(db_path, ignore_errors=True)

    def test_empty_retrieval(self) -> None:
        db_path = _temp_lancedb()
        try:
            memories = get_recent_memories(n=5, db_path=db_path)
            assert memories == []

            similar = retrieve_similar_memories("test", n=5, db_path=db_path)
            assert similar == []
        finally:
            import shutil
            shutil.rmtree(db_path, ignore_errors=True)

    def test_multiple_memories_ordered(self) -> None:
        db_path = _temp_lancedb()
        try:
            import time
            for i in range(3):
                store_exchange(
                    anomaly_type="hrv_anomaly", metric="rmssd",
                    value=20.0 + i, baseline=41.0, deviation=-2.0 + i * 0.5,
                    body_state="depleted",
                    probe_text=f"Probe {i}",
                    response_text=f"Response {i}",
                    db_path=db_path,
                )
                time.sleep(0.01)  # Ensure different timestamps

            memories = get_recent_memories(n=10, db_path=db_path)
            assert len(memories) == 3
            # Most recent first
            assert "Response 2" in memories[0]["response_text"]
        finally:
            import shutil
            shutil.rmtree(db_path, ignore_errors=True)


class TestProbeGenerator:
    def test_build_prompt(self) -> None:
        from soma.proto_self.probe_generator import build_probe_prompt

        anomaly = {"metric": "rmssd", "value": 18.3, "baseline": 41.0, "deviation": -2.3}
        state_info = {"state": "depleted"}
        prompt = build_probe_prompt(anomaly, state_info, [], [], "work_stress")

        assert "RMSSD" in prompt
        assert "18.3" in prompt
        assert "DEPLETED" in prompt.upper()
        assert "60 words" in prompt

    def test_build_prompt_with_memories(self) -> None:
        from soma.proto_self.probe_generator import build_probe_prompt

        anomaly = {"metric": "rmssd", "value": 18.3, "baseline": 41.0, "deviation": -2.3}
        state_info = {"state": "depleted"}
        memories = [
            {"timestamp": "2026-04-10T14:00:00", "metric": "rmssd",
             "value": 19.0, "deviation": -2.1,
             "response_text": "Argument with Joe about causal inference"},
        ]
        prompt = build_probe_prompt(anomaly, state_info, memories, [], "work_stress")

        assert "Joe" in prompt or "causal inference" in prompt
        assert "past exchanges" in prompt.lower()
