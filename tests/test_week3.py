"""Tests for Week 3: state classifier, recommender, feedback loop."""

from __future__ import annotations

import json
import os
import random
import tempfile

from soma.proto_self.artifact_filter import clean_rr
from soma.proto_self.baseline_model import build_baseline
from soma.proto_self.db import get_connection, insert_rr, start_session
from soma.proto_self.recommender import (
    get_recommendations,
    load_corpus,
    log_feedback,
    log_recommendation,
    get_pending_recommendations,
    get_recommendation_history,
)
from soma.proto_self.state_classifier import classify_state


def _temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _seed_baseline_db(db_path: str, model_path: str) -> None:
    """Create a DB with enough morning_baseline data and build the model."""
    conn = get_connection(db_path)
    start_session(conn, "baseline_001", label="morning_baseline")

    rng = random.Random(42)
    for i in range(300):
        rr = 800.0 + rng.gauss(0, 15)
        insert_rr(conn, "baseline_001", round(rr, 1), 75)
    conn.commit()
    conn.close()

    build_baseline(db_path=db_path, model_path=model_path, min_samples=50)


class TestStateClassifier:
    def test_unknown_without_model(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            result = classify_state(db_path=db_path, model_path=model_path)
            assert result["state"] == "unknown"
        finally:
            os.unlink(db_path)

    def test_unknown_without_data(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            _seed_baseline_db(db_path, model_path)
            # Clear RR data so there's no recent signal
            conn = get_connection(db_path)
            conn.execute("DELETE FROM rr_intervals")
            conn.commit()
            conn.close()

            result = classify_state(db_path=db_path, model_path=model_path)
            assert result["state"] == "unknown"
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_baseline_state_with_normal_data(self) -> None:
        """Normal data should classify as baseline."""
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            _seed_baseline_db(db_path, model_path)
            result = classify_state(db_path=db_path, model_path=model_path)
            # Should be baseline or close — data is the same as the model
            assert result["state"] in ("baseline", "restored", "recovering")
            assert result["rhr"] is not None
            assert result["rmssd"] is not None
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_depleted_state(self) -> None:
        """Very fast HR + low variability should classify as depleted."""
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            _seed_baseline_db(db_path, model_path)

            # Add stressed data (fast HR, low variability)
            conn = get_connection(db_path)
            start_session(conn, "stress_001", label="work_stress")
            for i in range(60):
                insert_rr(conn, "stress_001", 550.0 + (i % 3), 109)
            conn.commit()
            conn.close()

            result = classify_state(db_path=db_path, model_path=model_path)
            assert result["state"] in ("depleted", "recovering")
            assert result["rhr_z"] > 0  # HR above baseline
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestCorpus:
    def test_corpus_loads(self) -> None:
        corpus = load_corpus()
        assert len(corpus) > 0
        for entry in corpus:
            assert "id" in entry
            assert "type" in entry
            assert "title" in entry
            assert "best_states" in entry

    def test_all_states_have_recommendations(self) -> None:
        """Every state should have at least one eligible recommendation."""
        corpus = load_corpus()
        states = ["depleted", "recovering", "baseline", "restored", "peak"]
        for state in states:
            eligible = [
                e for e in corpus
                if state in e["best_states"]
                and state not in e.get("avoid_states", [])
            ]
            assert len(eligible) > 0, f"No recommendations for state: {state}"


class TestRecommender:
    def test_get_recommendations(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            _seed_baseline_db(db_path, model_path)
            result = get_recommendations(n=3, db_path=db_path, model_path=model_path)
            assert "state" in result
            assert "recommendations" in result
            assert len(result["recommendations"]) <= 3
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_log_and_retrieve_recommendation(self) -> None:
        db_path = _temp_db()
        try:
            state_info = {"state": "baseline", "rhr": 75.0, "rmssd": 40.0}
            row_id = log_recommendation("act_001", "Morning run", "activity", state_info, db_path=db_path)
            assert row_id > 0

            pending = get_pending_recommendations(db_path=db_path)
            assert len(pending) == 1
            assert pending[0]["title"] == "Morning run"
        finally:
            os.unlink(db_path)

    def test_log_feedback(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            _seed_baseline_db(db_path, model_path)
            state_info = {"state": "baseline", "rhr": 75.0, "rmssd": 40.0}
            row_id = log_recommendation("act_003", "Tonglen", "activity", state_info, db_path=db_path)

            state_after = log_feedback(row_id, followed=1, outcome="better",
                                       db_path=db_path, model_path=model_path)
            assert state_after["state"] is not None

            # Pending should now be empty (feedback was given)
            pending = get_pending_recommendations(db_path=db_path)
            assert len(pending) == 0

            history = get_recommendation_history(db_path=db_path)
            assert len(history) == 1
            assert history[0]["outcome"] == "better"
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_exclude_ids(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", "_model.json")
        try:
            _seed_baseline_db(db_path, model_path)
            # Get all IDs
            corpus = load_corpus()
            all_ids = [e["id"] for e in corpus]

            # Exclude all but one
            result = get_recommendations(n=10, exclude_ids=all_ids[:-1],
                                         db_path=db_path, model_path=model_path)
            assert len(result["recommendations"]) <= 1
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)
