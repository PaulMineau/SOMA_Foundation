"""Tests for Week 3b: profile, RAEN scorer, research agent (no API calls)."""

from __future__ import annotations

import json
import os
import random
import tempfile

from soma.proto_self.baseline_model import build_baseline
from soma.proto_self.db import get_connection, insert_rr, start_session
from soma.proto_self.raen_scorer import score_candidate, score_candidates
from soma.proto_self.research_agent import RESEARCH_TOPICS, get_todays_topic, build_research_prompt
from soma.proto_self.soma_profile import build_profile, get_existing_titles


def _temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _seed_db(db_path: str, model_path: str) -> None:
    conn = get_connection(db_path)
    start_session(conn, "baseline_001", label="morning_baseline")
    rng = random.Random(42)
    for i in range(300):
        insert_rr(conn, "baseline_001", round(800.0 + rng.gauss(0, 15), 1), 75)
    conn.commit()
    conn.close()
    build_baseline(db_path=db_path, model_path=model_path, min_samples=50)


class TestSomaProfile:
    def test_build_profile(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            _seed_db(db_path, model_path)
            profile = build_profile(db_path=db_path, model_path=model_path)

            assert "identity" in profile
            assert "current_state" in profile
            assert "existing_corpus" in profile
            assert profile["identity"]["age"] == 50
            assert len(profile["identity"]["interests"]) > 0
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_existing_titles(self) -> None:
        titles = get_existing_titles()
        assert len(titles) > 0
        assert "The Wild Robot" in titles


class TestRAENScorer:
    def _make_profile(self) -> dict:
        return {
            "identity": {
                "interests": ["consciousness research", "AI architecture", "parenting"],
            },
            "current_state": {"state": "baseline"},
            "what_worked": [
                {"type": "activity", "title": "Morning run"},
                {"type": "activity", "title": "Tonglen meditation"},
            ],
            "existing_corpus": ["The Wild Robot", "Soul"],
        }

    def test_score_new_relevant_candidate(self) -> None:
        profile = self._make_profile()
        candidate = {
            "id": "test_001",
            "type": "book",
            "title": "Being You by Anil Seth",
            "tags": ["consciousness research", "neuroscience"],
            "best_states": ["baseline", "restored"],
            "avoid_states": ["depleted"],
            "duration_min": 45,
        }

        scored = score_candidate(candidate, profile)
        assert scored["raen_total"] > 0
        assert scored["raen"]["novelty"] == 8  # New title

    def test_duplicate_scores_low_novelty(self) -> None:
        profile = self._make_profile()
        candidate = {
            "id": "dup_001",
            "type": "movie",
            "title": "The Wild Robot",  # Already in corpus
            "tags": ["restorative"],
            "best_states": ["baseline"],
            "avoid_states": [],
            "duration_min": 102,
        }

        scored = score_candidate(candidate, profile)
        assert scored["raen"]["novelty"] == 2  # Duplicate

    def test_blocked_state_zeros_actionability(self) -> None:
        profile = self._make_profile()
        profile["current_state"]["state"] = "depleted"
        candidate = {
            "id": "block_001",
            "type": "book",
            "title": "Dense Technical Book",
            "tags": [],
            "best_states": ["peak"],
            "avoid_states": ["depleted"],
            "duration_min": 60,
        }

        scored = score_candidate(candidate, profile)
        assert scored["raen"]["actionability"] == 0

    def test_score_candidates_ranks_by_total(self) -> None:
        profile = self._make_profile()
        candidates = [
            {"id": "a", "type": "book", "title": "Low Match", "tags": [],
             "best_states": [], "avoid_states": [], "duration_min": 60},
            {"id": "b", "type": "activity", "title": "High Match",
             "tags": ["consciousness research", "AI architecture"],
             "best_states": ["baseline"], "avoid_states": [], "duration_min": 30},
        ]

        scored = score_candidates(candidates, profile)
        assert scored[0]["title"] == "High Match"


class TestResearchAgent:
    def test_topic_rotation(self) -> None:
        topic = get_todays_topic()
        assert topic in RESEARCH_TOPICS

    def test_build_prompt(self) -> None:
        profile = {
            "identity": {
                "age": 50,
                "location": "Duvall, WA",
                "practices": ["meditation"],
                "quitting": ["nicotine"],
                "interests": ["consciousness research"],
                "values": ["compassion"],
                "reading_now": ["Damasio"],
                "movies_loved": ["Soul"],
            },
            "current_state": {"state": "baseline", "reason": "Normal", "rmssd": 40, "rhr": 72},
            "what_worked": [],
            "existing_corpus": ["The Wild Robot"],
        }

        prompt = build_research_prompt(profile, "movies and films")

        assert "movies and films" in prompt
        assert "consciousness research" in prompt
        assert "The Wild Robot" in prompt
        assert "JSON array" in prompt
