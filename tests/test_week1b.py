"""Tests for Week 1b: artifact filter, baseline model, anomaly detection."""

from __future__ import annotations

import json
import os
import tempfile

from soma.proto_self.artifact_filter import (
    clean_rr,
    compute_rhr,
    compute_rmssd,
    reject_ectopic,
    reject_range,
)
from soma.proto_self.baseline_model import build_baseline, compute_stats
from soma.proto_self.db import get_connection, insert_rr, start_session


def _temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


class TestArtifactFilter:
    def test_reject_range_removes_extremes(self) -> None:
        rr = [820.0, 200.0, 810.0, 2000.0, 835.0, 150.0, 800.0]
        clean = reject_range(rr)
        assert 200.0 not in clean
        assert 2000.0 not in clean
        assert 150.0 not in clean
        assert len(clean) == 4

    def test_reject_range_keeps_valid(self) -> None:
        rr = [800.0, 810.0, 750.0, 900.0]
        assert reject_range(rr) == rr

    def test_reject_ectopic_removes_jumps(self) -> None:
        rr = [800.0, 810.0, 1200.0, 820.0, 815.0]
        clean = reject_ectopic(rr)
        assert 1200.0 not in clean

    def test_reject_ectopic_empty(self) -> None:
        assert reject_ectopic([]) == []

    def test_clean_rr_full_pipeline(self) -> None:
        rr = [820.0, 810.0, 835.0, 2000.0, 800.0, 150.0, 815.0]
        clean = clean_rr(rr)
        assert 2000.0 not in clean
        assert 150.0 not in clean
        assert len(clean) >= 3

    def test_compute_rmssd(self) -> None:
        rr = [800.0, 810.0, 790.0, 805.0, 815.0]
        rmssd = compute_rmssd(rr)
        assert rmssd is not None
        assert rmssd > 0

    def test_compute_rmssd_insufficient(self) -> None:
        assert compute_rmssd([800.0]) is None
        assert compute_rmssd([]) is None

    def test_compute_rhr(self) -> None:
        rr = [800.0, 810.0, 790.0]
        rhr = compute_rhr(rr)
        assert rhr is not None
        assert 70 < rhr < 80  # ~75 bpm at 800ms

    def test_compute_rhr_empty(self) -> None:
        assert compute_rhr([]) is None


class TestComputeStats:
    def test_basic_stats(self) -> None:
        values = [10.0, 20.0, 30.0]
        mean, std = compute_stats(values)
        assert mean == 20.0
        assert std is not None and std > 0

    def test_empty(self) -> None:
        mean, std = compute_stats([])
        assert mean is None
        assert std is None


class TestBaselineModel:
    def test_insufficient_data(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", ".json")
        try:
            conn = get_connection(db_path)
            start_session(conn, "s1", label="morning_baseline")
            for i in range(10):
                insert_rr(conn, "s1", 800.0 + i, 75)
            conn.commit()
            conn.close()

            result = build_baseline(db_path=db_path, model_path=model_path, min_samples=100)
            assert result is None
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_builds_model_with_enough_data(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", ".json")
        try:
            conn = get_connection(db_path)
            start_session(conn, "s1", label="morning_baseline")
            # Generate 200 realistic RR intervals (~75 bpm with variation)
            import random
            rng = random.Random(42)
            for i in range(200):
                rr = 800.0 + rng.gauss(0, 15)
                insert_rr(conn, "s1", round(rr, 1), 75)
            conn.commit()
            conn.close()

            result = build_baseline(db_path=db_path, model_path=model_path, min_samples=50)
            assert result is not None
            assert "rhr" in result
            assert "rmssd" in result
            assert result["rhr"]["mean"] is not None
            assert result["rmssd"]["mean"] is not None

            # Check file was written
            assert os.path.exists(model_path)
            with open(model_path) as f:
                saved = json.load(f)
            assert saved["rhr"]["mean"] == result["rhr"]["mean"]
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_non_baseline_sessions_excluded(self) -> None:
        db_path = _temp_db()
        model_path = db_path.replace(".db", ".json")
        try:
            conn = get_connection(db_path)

            # Add data labeled as something else
            start_session(conn, "s1", label="post_run")
            for i in range(200):
                insert_rr(conn, "s1", 700.0 + i % 20, 85)
            conn.commit()

            # Add very few morning_baseline
            start_session(conn, "s2", label="morning_baseline")
            for i in range(5):
                insert_rr(conn, "s2", 800.0, 75)
            conn.commit()
            conn.close()

            result = build_baseline(db_path=db_path, model_path=model_path, min_samples=50)
            assert result is None  # Only 5 morning_baseline samples
        finally:
            os.unlink(db_path)
            if os.path.exists(model_path):
                os.unlink(model_path)
