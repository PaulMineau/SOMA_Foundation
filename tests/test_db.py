"""Tests for SOMA-Cardio SQLite storage."""

from __future__ import annotations

import os
import tempfile

from soma.proto_self.db import (
    end_session,
    export_session_csv,
    get_connection,
    get_recent_sessions,
    get_session_rr,
    insert_rr,
    start_session,
)


def _temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


class TestSQLiteStorage:
    def test_create_tables(self) -> None:
        db_path = _temp_db()
        try:
            conn = get_connection(db_path)
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            assert "sessions" in tables
            assert "rr_intervals" in tables
            conn.close()
        finally:
            os.unlink(db_path)

    def test_session_lifecycle(self) -> None:
        db_path = _temp_db()
        try:
            conn = get_connection(db_path)

            start_session(conn, "test_001", label="morning_baseline",
                          device_name="Polar H10", device_address="AA:BB")

            insert_rr(conn, "test_001", 800.0, 75)
            insert_rr(conn, "test_001", 810.0, 74)
            insert_rr(conn, "test_001", 790.0, 76)
            conn.commit()

            end_session(conn, "test_001", n_intervals=3,
                        mean_hr=75.0, rmssd=14.1, sdnn=10.0, body_state="resting")

            rr = get_session_rr(conn, "test_001")
            assert rr == [800.0, 810.0, 790.0]

            sessions = get_recent_sessions(conn)
            assert len(sessions) == 1
            assert sessions[0]["label"] == "morning_baseline"
            assert sessions[0]["rmssd"] == 14.1
            assert sessions[0]["body_state"] == "resting"
            assert sessions[0]["ended_at"] is not None

            conn.close()
        finally:
            os.unlink(db_path)

    def test_export_csv(self) -> None:
        db_path = _temp_db()
        try:
            conn = get_connection(db_path)
            start_session(conn, "csv_test", label="test")

            for rr in [800.0, 810.0, 790.0, 805.0]:
                insert_rr(conn, "csv_test", rr, 75)
            conn.commit()

            csv_path = _temp_db().replace(".db", ".csv")
            result = export_session_csv(conn, "csv_test", output_path=csv_path)

            assert os.path.exists(result)
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == 5  # header + 4 rows

            os.unlink(csv_path)
            conn.close()
        finally:
            os.unlink(db_path)

    def test_multiple_sessions(self) -> None:
        db_path = _temp_db()
        try:
            conn = get_connection(db_path)

            start_session(conn, "s1", label="morning")
            insert_rr(conn, "s1", 800.0, 75)
            conn.commit()

            start_session(conn, "s2", label="evening")
            insert_rr(conn, "s2", 750.0, 80)
            conn.commit()

            assert get_session_rr(conn, "s1") == [800.0]
            assert get_session_rr(conn, "s2") == [750.0]

            sessions = get_recent_sessions(conn)
            assert len(sessions) == 2

            conn.close()
        finally:
            os.unlink(db_path)
