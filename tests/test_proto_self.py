"""Tests for Proto-Self: HRV computation, BLE parsing, and body state classification."""

from __future__ import annotations

import numpy as np

from soma.proto_self.hrv import (
    HRVMetrics,
    classify_body_state,
    clean_rr_intervals,
    compute_hrv,
)
from soma.proto_self.polar_reader import HRSample, PolarSession, parse_hr_measurement


class TestParseHRMeasurement:
    """Tests for BLE Heart Rate Measurement parsing (0x2A37)."""

    def test_uint8_hr_with_rr(self) -> None:
        """Parse uint8 heart rate + RR intervals."""
        # Flags: 0x10 (RR present, uint8 HR)
        # HR: 72 bpm
        # RR: 833ms (= 853 raw * 1000/1024)
        flags = 0x10
        hr = 72
        rr_raw = int(833 * 1024 / 1000)  # ~853
        rr_bytes = rr_raw.to_bytes(2, "little")
        data = bytearray([flags, hr]) + bytearray(rr_bytes)

        sample = parse_hr_measurement(data)

        assert sample.heart_rate == 72
        assert len(sample.rr_intervals) == 1
        assert 830 < sample.rr_intervals[0] < 836  # ~833ms

    def test_uint16_hr_format(self) -> None:
        """Parse uint16 heart rate format (bit 0 set)."""
        # Flags: 0x11 (RR present, uint16 HR)
        flags = 0x11
        hr_bytes = (72).to_bytes(2, "little")
        rr_raw = int(800 * 1024 / 1000)
        rr_bytes = rr_raw.to_bytes(2, "little")
        data = bytearray([flags]) + bytearray(hr_bytes) + bytearray(rr_bytes)

        sample = parse_hr_measurement(data)

        assert sample.heart_rate == 72
        assert len(sample.rr_intervals) == 1

    def test_multiple_rr_intervals(self) -> None:
        """Parse multiple RR intervals in one notification."""
        flags = 0x10  # RR present, uint8 HR
        hr = 80
        rr1_raw = int(750 * 1024 / 1000)
        rr2_raw = int(760 * 1024 / 1000)
        data = bytearray([flags, hr])
        data += rr1_raw.to_bytes(2, "little")
        data += rr2_raw.to_bytes(2, "little")

        sample = parse_hr_measurement(data)

        assert sample.heart_rate == 80
        assert len(sample.rr_intervals) == 2

    def test_no_rr_intervals(self) -> None:
        """Parse HR-only notification (no RR bit set)."""
        flags = 0x00  # No RR, uint8 HR
        hr = 65
        data = bytearray([flags, hr])

        sample = parse_hr_measurement(data)

        assert sample.heart_rate == 65
        assert sample.rr_intervals == []

    def test_sensor_contact_flags(self) -> None:
        """Parse sensor contact status bits."""
        # Bits 1-2: sensor contact supported + detected
        flags = 0x06  # Contact supported (bit 1) + contact detected (bit 2)
        data = bytearray([flags, 70])

        sample = parse_hr_measurement(data)
        assert sample.sensor_contact is True

        # Contact supported but NOT detected
        flags = 0x02  # Contact supported (bit 1), not detected (bit 2 = 0)
        data = bytearray([flags, 70])

        sample = parse_hr_measurement(data)
        assert sample.sensor_contact is False


class TestCleanRRIntervals:
    """Tests for RR interval artifact rejection."""

    def test_removes_out_of_range(self) -> None:
        """Intervals outside 300-2000ms are removed."""
        rr = [800.0, 200.0, 850.0, 2500.0, 790.0]
        clean = clean_rr_intervals(rr)

        assert len(clean) == 3
        assert 200.0 not in clean
        assert 2500.0 not in clean

    def test_removes_sudden_jumps(self) -> None:
        """Successive differences > 300ms are removed."""
        rr = [800.0, 810.0, 1200.0, 820.0, 815.0]  # 1200 is a jump
        clean = clean_rr_intervals(rr)

        assert 1200.0 not in clean

    def test_empty_input(self) -> None:
        clean = clean_rr_intervals([])
        assert len(clean) == 0

    def test_all_valid(self) -> None:
        rr = [800.0, 810.0, 805.0, 815.0, 800.0]
        clean = clean_rr_intervals(rr)
        assert len(clean) == 5


class TestComputeHRV:
    """Tests for HRV metric computation."""

    def test_basic_computation(self) -> None:
        """Compute HRV from a simple RR series."""
        # Steady ~75 bpm with slight variation
        rr = [800.0, 810.0, 790.0, 805.0, 815.0, 795.0, 800.0, 810.0, 790.0, 805.0]

        metrics = compute_hrv(rr, window_seconds=8.0)

        assert metrics.n_intervals == 10
        assert metrics.n_artifacts == 0
        assert 790 < metrics.mean_rr < 815
        assert 70 < metrics.mean_hr < 80
        assert metrics.sdnn > 0
        assert metrics.rmssd > 0
        assert 0 <= metrics.pnn50 <= 100
        assert metrics.window_seconds == 8.0

    def test_high_hrv(self) -> None:
        """High variability → high RMSSD."""
        # Alternating short/long intervals (respiratory sinus arrhythmia)
        rr = [750.0, 850.0, 750.0, 850.0, 750.0, 850.0, 750.0, 850.0]

        metrics = compute_hrv(rr)

        assert metrics.rmssd > 50  # High variability

    def test_low_hrv(self) -> None:
        """Very steady intervals → low RMSSD."""
        rr = [800.0, 801.0, 800.0, 801.0, 800.0, 801.0, 800.0, 801.0]

        metrics = compute_hrv(rr)

        assert metrics.rmssd < 5  # Very low variability

    def test_insufficient_intervals_raises(self) -> None:
        """Fewer than 3 clean intervals should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Insufficient"):
            compute_hrv([800.0, 810.0])

    def test_artifacts_counted(self) -> None:
        """Artifacts should be counted but not included."""
        rr = [800.0, 810.0, 100.0, 805.0, 3000.0, 795.0]  # 2 artifacts

        metrics = compute_hrv(rr)

        assert metrics.n_artifacts == 2
        assert metrics.n_intervals == 4


class TestClassifyBodyState:
    """Tests for rule-based body state classification."""

    def test_recovery_state(self) -> None:
        """High RMSSD + low HR = recovery."""
        metrics = HRVMetrics(
            mean_rr=1100.0, sdnn=80.0, rmssd=70.0, pnn50=40.0,
            mean_hr=55.0, n_intervals=100, n_artifacts=0, window_seconds=300.0,
        )
        assert classify_body_state(metrics) == "recovery"

    def test_stressed_state(self) -> None:
        """Low RMSSD or high HR = stressed."""
        metrics = HRVMetrics(
            mean_rr=600.0, sdnn=20.0, rmssd=15.0, pnn50=5.0,
            mean_hr=100.0, n_intervals=100, n_artifacts=0, window_seconds=300.0,
        )
        assert classify_body_state(metrics) == "stressed"

    def test_optimal_state(self) -> None:
        """Good RMSSD + moderate HR = optimal."""
        metrics = HRVMetrics(
            mean_rr=750.0, sdnn=50.0, rmssd=35.0, pnn50=20.0,
            mean_hr=80.0, n_intervals=100, n_artifacts=0, window_seconds=300.0,
        )
        assert classify_body_state(metrics) == "optimal"


class TestPolarSession:
    """Tests for PolarSession data aggregation."""

    def test_all_rr_intervals(self) -> None:
        """all_rr_intervals flattens across samples."""
        session = PolarSession(start_time=1000.0)
        session.samples = [
            HRSample(timestamp=1000.0, heart_rate=70, rr_intervals=[800.0, 810.0], sensor_contact=True),
            HRSample(timestamp=1001.0, heart_rate=72, rr_intervals=[790.0], sensor_contact=True),
        ]

        assert session.all_rr_intervals == [800.0, 810.0, 790.0]
        assert session.sample_count == 2

    def test_duration(self) -> None:
        session = PolarSession(start_time=1000.0)
        session.samples = [
            HRSample(timestamp=1000.0, heart_rate=70, rr_intervals=[], sensor_contact=True),
            HRSample(timestamp=1060.0, heart_rate=70, rr_intervals=[], sensor_contact=True),
        ]

        assert session.duration_seconds == 60.0

    def test_empty_session(self) -> None:
        session = PolarSession()
        assert session.all_rr_intervals == []
        assert session.duration_seconds == 0.0
        assert session.sample_count == 0
