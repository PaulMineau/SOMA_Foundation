"""Tests for CPAP integration — ingestor, correlator (no API calls)."""

from __future__ import annotations

from unittest.mock import patch


class TestCorrelator:
    def test_insufficient_data(self) -> None:
        from soma.proto_self.cpap.correlator import correlate_cpap_to_recovery

        # Without any data, should return insufficient
        with patch("soma.proto_self.cpap.cpap_ingestor.get_recent_cpap_days", return_value=[]):
            with patch("soma.proto_self.fitbit.fitbit_dashboard.get_recent_fitbit_days", return_value=[]):
                result = correlate_cpap_to_recovery(days=30)
                assert result["n"] == 0
                assert result.get("insufficient_data") is True

    def test_strong_negative_correlation(self) -> None:
        from soma.proto_self.cpap.correlator import correlate_cpap_to_recovery

        # Simulated: high AHI pairs with low recovery
        cpap_data = [
            {"date": "2026-04-14", "ahi": 20.0, "usage_min": 420},
            {"date": "2026-04-13", "ahi": 2.0, "usage_min": 450},
            {"date": "2026-04-12", "ahi": 18.0, "usage_min": 380},
            {"date": "2026-04-11", "ahi": 3.0, "usage_min": 460},
            {"date": "2026-04-10", "ahi": 15.0, "usage_min": 400},
        ]
        fitbit_data = [
            {"date": "2026-04-14", "recovery_score": 4.0, "hrv_rmssd": 25.0, "resting_hr": 72, "spo2_avg": 92},
            {"date": "2026-04-13", "recovery_score": 8.0, "hrv_rmssd": 55.0, "resting_hr": 58, "spo2_avg": 96},
            {"date": "2026-04-12", "recovery_score": 4.5, "hrv_rmssd": 28.0, "resting_hr": 70, "spo2_avg": 93},
            {"date": "2026-04-11", "recovery_score": 8.5, "hrv_rmssd": 60.0, "resting_hr": 56, "spo2_avg": 97},
            {"date": "2026-04-10", "recovery_score": 5.0, "hrv_rmssd": 32.0, "resting_hr": 68, "spo2_avg": 94},
        ]

        with patch("soma.proto_self.cpap.cpap_ingestor.get_recent_cpap_days", return_value=cpap_data):
            with patch("soma.proto_self.fitbit.fitbit_dashboard.get_recent_fitbit_days", return_value=fitbit_data):
                result = correlate_cpap_to_recovery(days=30)
                assert result["n"] == 5
                # High AHI -> low recovery should give negative correlation
                assert result["correlations"]["ahi_vs_recovery"] < -0.5
                assert len(result["insights"]) > 0

    def test_compliance_stats(self) -> None:
        from soma.proto_self.cpap.correlator import get_compliance_stats

        cpap_data = [
            {"date": "2026-04-14", "ahi": 5.0, "usage_min": 420},  # compliant
            {"date": "2026-04-13", "ahi": 3.0, "usage_min": 180},  # not compliant
            {"date": "2026-04-12", "ahi": 4.0, "usage_min": 480},  # compliant
            {"date": "2026-04-11", "ahi": 8.0, "usage_min": 300},  # compliant (>=240)
        ]

        with patch("soma.proto_self.cpap.cpap_ingestor.get_recent_cpap_days", return_value=cpap_data):
            stats = get_compliance_stats(days=30)
            assert stats["n"] == 4
            assert stats["compliance_pct"] == 75.0  # 3 of 4 >= 4h
            assert stats["avg_ahi"] == 5.0


class TestIngestor:
    def test_build_narrative_excellent_control(self) -> None:
        from soma.proto_self.cpap.cpap_ingestor import _build_narrative

        record = {
            "ahi": 0.5,
            "usage_min": 420,  # 7 hours
            "sleep_score": 95,
            "leak_percentile": 5,
        }
        narrative = _build_narrative(record)
        assert "excellent" in narrative.lower()
        assert "compliant" in narrative.lower()

    def test_build_narrative_poor_control(self) -> None:
        from soma.proto_self.cpap.cpap_ingestor import _build_narrative

        record = {
            "ahi": 22.0,
            "usage_min": 180,  # 3 hours
            "sleep_score": 50,
            "leak_percentile": 30,
        }
        narrative = _build_narrative(record)
        assert "moderate" in narrative.lower() or "severe" in narrative.lower()
        assert "short usage" in narrative.lower() or "under compliance" in narrative.lower()

    def test_build_narrative_mild(self) -> None:
        from soma.proto_self.cpap.cpap_ingestor import _build_narrative

        record = {"ahi": 8.0, "usage_min": 450, "sleep_score": 75, "leak_percentile": 10}
        narrative = _build_narrative(record)
        assert "mild" in narrative.lower()


class TestMyAirClient:
    def test_pkce_generation(self) -> None:
        from soma.proto_self.cpap.myair_client import _generate_pkce

        verifier, challenge = _generate_pkce()
        assert len(verifier) > 20
        assert len(challenge) > 20
        assert verifier != challenge
        # PKCE verifiers/challenges must not contain base64 padding
        assert "=" not in verifier
        assert "=" not in challenge
