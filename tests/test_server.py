"""Tests for SOMA Core FastAPI server."""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Set DB path before importing server
_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.environ["SOMA_CARDIO_DB"] = _db_path

from soma.proto_self.soma_server import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup():
    """Reset DB between tests."""
    yield
    # Clear tables
    from soma.proto_self.db import get_connection
    conn = get_connection(_db_path)
    conn.execute("DELETE FROM rr_intervals")
    conn.execute("DELETE FROM sessions")
    conn.execute("DELETE FROM anomalies")
    conn.commit()
    conn.close()


class TestIngestRR:
    def test_basic_ingest(self, client: TestClient) -> None:
        resp = client.post("/ingest/rr", json={
            "session_id": "test_001",
            "label": "morning_baseline",
            "device_id": "polar_h10",
            "readings": [
                {"timestamp": "2026-04-14T06:00:00", "rr_ms": 800.0, "hr_bpm": 75},
                {"timestamp": "2026-04-14T06:00:01", "rr_ms": 810.0, "hr_bpm": 74},
                {"timestamp": "2026-04-14T06:00:02", "rr_ms": 790.0, "hr_bpm": 76},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["inserted"] == 3
        assert data["session_id"] == "test_001"

    def test_ingest_creates_session(self, client: TestClient) -> None:
        client.post("/ingest/rr", json={
            "session_id": "test_002",
            "label": "post_run",
            "device_id": "polar_h10",
            "readings": [
                {"timestamp": "2026-04-14T07:00:00", "rr_ms": 700.0},
            ],
        })

        resp = client.get("/sessions")
        sessions = resp.json()
        assert len(sessions) == 1
        assert sessions[0]["label"] == "post_run"

    def test_ingest_duplicate_session(self, client: TestClient) -> None:
        """Second batch to same session should append, not create new session."""
        for i in range(2):
            client.post("/ingest/rr", json={
                "session_id": "dup_test",
                "label": "test",
                "device_id": "polar_h10",
                "readings": [
                    {"timestamp": f"2026-04-14T08:00:0{i}", "rr_ms": 800.0},
                ],
            })

        resp = client.get("/sessions")
        sessions = resp.json()
        assert len(sessions) == 1


class TestStatus:
    def test_empty_status(self, client: TestClient) -> None:
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rhr_bpm"] is None
        assert data["readings_in_window"] == 0

    def test_status_with_data(self, client: TestClient) -> None:
        readings = [
            {"timestamp": f"2026-04-14T06:00:{i:02d}", "rr_ms": 800.0 + i, "hr_bpm": 75}
            for i in range(10)
        ]
        client.post("/ingest/rr", json={
            "session_id": "status_test",
            "label": "test",
            "device_id": "polar_h10",
            "readings": readings,
        })

        resp = client.get("/status")
        data = resp.json()
        assert data["rhr_bpm"] is not None
        assert data["rmssd_ms"] is not None
        assert data["body_state"] is not None
        assert data["readings_in_window"] == 10


class TestBaseline:
    def test_insufficient_data(self, client: TestClient) -> None:
        resp = client.get("/baseline")
        data = resp.json()
        assert data["status"] == "insufficient_data"

    def test_baseline_with_morning_data(self, client: TestClient) -> None:
        readings = [
            {"timestamp": f"2026-04-14T06:00:{i:02d}", "rr_ms": 800.0 + (i % 5), "hr_bpm": 75}
            for i in range(20)
        ]
        client.post("/ingest/rr", json={
            "session_id": "morning_001",
            "label": "morning_baseline",
            "device_id": "polar_h10",
            "readings": readings,
        })

        resp = client.get("/baseline")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["baseline_rhr_bpm"] is not None
        assert data["baseline_rmssd_ms"] is not None


class TestContextTag:
    def test_relabel_session(self, client: TestClient) -> None:
        client.post("/ingest/rr", json={
            "session_id": "relabel_test",
            "label": "unlabeled",
            "device_id": "polar_h10",
            "readings": [{"timestamp": "2026-04-14T09:00:00", "rr_ms": 800.0}],
        })

        resp = client.post("/ingest/context", json={
            "session_id": "relabel_test",
            "label": "meditation",
        })
        assert resp.json()["label"] == "meditation"

        sessions = client.get("/sessions").json()
        assert sessions[0]["label"] == "meditation"


class TestAnomalies:
    def test_empty_anomalies(self, client: TestClient) -> None:
        resp = client.get("/anomalies")
        assert resp.status_code == 200
        assert resp.json() == []


class TestProbe:
    def test_probe_no_anomaly(self, client: TestClient) -> None:
        resp = client.post("/probe", json={"generate_only": True})
        data = resp.json()
        assert data["status"] == "no_anomaly"


class TestSessions:
    def test_list_sessions(self, client: TestClient) -> None:
        for label in ["morning_baseline", "post_run", "meditation"]:
            client.post("/ingest/rr", json={
                "session_id": f"s_{label}",
                "label": label,
                "device_id": "polar_h10",
                "readings": [{"timestamp": "2026-04-14T10:00:00", "rr_ms": 800.0}],
            })

        resp = client.get("/sessions?limit=2")
        sessions = resp.json()
        assert len(sessions) == 2
