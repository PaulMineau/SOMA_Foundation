"""Polar H10 BLE reader — streams RR intervals and heart rate via Bluetooth.

Uses the standard Heart Rate Measurement characteristic (0x2A37) which provides:
- Heart rate (bpm)
- RR intervals (ms) — the time between successive heartbeats

RR intervals are the raw signal for all HRV computation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice

logger = logging.getLogger(__name__)

# Standard BLE Heart Rate Measurement characteristic UUID
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# Polar H10 Battery Level characteristic
BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"


@dataclass
class HRSample:
    """A single heart rate measurement from the Polar H10."""

    timestamp: float  # Unix timestamp
    heart_rate: int  # bpm
    rr_intervals: list[float]  # milliseconds between beats
    sensor_contact: bool  # whether the sensor has good skin contact


@dataclass
class PolarSession:
    """Accumulated data from a Polar H10 recording session."""

    start_time: float = 0.0
    samples: list[HRSample] = field(default_factory=list)
    device_name: str = ""
    device_address: str = ""
    battery_level: int | None = None

    @property
    def duration_seconds(self) -> float:
        if not self.samples:
            return 0.0
        return self.samples[-1].timestamp - self.start_time

    @property
    def all_rr_intervals(self) -> list[float]:
        """Flat list of all RR intervals across all samples."""
        rr: list[float] = []
        for s in self.samples:
            rr.extend(s.rr_intervals)
        return rr

    @property
    def sample_count(self) -> int:
        return len(self.samples)


def parse_hr_measurement(data: bytearray) -> HRSample:
    """Parse a Heart Rate Measurement characteristic value (BLE 0x2A37).

    Byte layout per Bluetooth SIG spec:
    - Byte 0: Flags
      - Bit 0: HR format (0 = uint8, 1 = uint16)
      - Bit 1-2: Sensor contact status
      - Bit 4: RR interval present
    - Byte 1 (or 1-2): Heart rate value
    - Remaining bytes: RR intervals (uint16 LE, units of 1/1024 sec)
    """
    flags = data[0]
    hr_format_16bit = bool(flags & 0x01)
    sensor_contact_supported = bool(flags & 0x02)
    sensor_contact = bool(flags & 0x04) if sensor_contact_supported else True
    rr_present = bool(flags & 0x10)

    offset = 1
    if hr_format_16bit:
        heart_rate = int.from_bytes(data[offset:offset + 2], "little")
        offset += 2
    else:
        heart_rate = data[offset]
        offset += 1

    # Skip Energy Expended if present (bit 3)
    if flags & 0x08:
        offset += 2

    # Parse RR intervals
    rr_intervals: list[float] = []
    if rr_present:
        while offset + 1 < len(data):
            rr_raw = int.from_bytes(data[offset:offset + 2], "little")
            rr_ms = rr_raw / 1024.0 * 1000.0  # Convert to milliseconds
            rr_intervals.append(rr_ms)
            offset += 2

    return HRSample(
        timestamp=time.time(),
        heart_rate=heart_rate,
        rr_intervals=rr_intervals,
        sensor_contact=sensor_contact,
    )


async def discover_polar(timeout: float = 10.0) -> BLEDevice | None:
    """Scan for a Polar device via BLE.

    Returns the first device with 'Polar' in its name, or None.
    """
    logger.info("Scanning for Polar devices (%.0fs timeout)...", timeout)
    devices = await BleakScanner.discover(timeout=timeout)

    for device in devices:
        if device.name and "Polar" in device.name:
            logger.info(
                "Found Polar device: %s (%s)", device.name, device.address
            )
            return device

    logger.warning("No Polar device found")
    return None


async def read_battery(client: BleakClient) -> int | None:
    """Read battery level from the Polar H10."""
    try:
        data = await client.read_gatt_char(BATTERY_UUID)
        level = data[0]
        logger.info("Battery level: %d%%", level)
        return level
    except Exception:
        logger.debug("Could not read battery level")
        return None


async def stream_hr(
    duration_seconds: float = 60.0,
    device: BLEDevice | None = None,
    on_sample: Callable[[HRSample], None] | None = None,
) -> PolarSession:
    """Connect to Polar H10 and stream heart rate + RR intervals.

    Args:
        duration_seconds: How long to record (seconds).
        device: Pre-discovered BLEDevice. If None, will scan.
        on_sample: Optional callback fired for each HRSample.

    Returns:
        PolarSession with all collected data.
    """
    if device is None:
        device = await discover_polar()
        if device is None:
            raise RuntimeError("No Polar device found. Is the chest strap on?")

    session = PolarSession(
        start_time=time.time(),
        device_name=device.name or "unknown",
        device_address=device.address,
    )

    def _notification_handler(_sender: int, data: bytearray) -> None:
        sample = parse_hr_measurement(data)

        if not sample.sensor_contact:
            logger.debug("No sensor contact — skipping sample")
            return

        session.samples.append(sample)

        if on_sample is not None:
            on_sample(sample)

        if sample.rr_intervals:
            logger.debug(
                "HR=%d bpm, RR=%s ms",
                sample.heart_rate,
                [f"{rr:.0f}" for rr in sample.rr_intervals],
            )

    logger.info(
        "Connecting to %s (%s) for %.0fs...",
        session.device_name, session.device_address, duration_seconds,
    )

    async with BleakClient(device.address) as client:
        session.battery_level = await read_battery(client)

        await client.start_notify(HR_MEASUREMENT_UUID, _notification_handler)
        logger.info("Streaming heart rate data...")

        await asyncio.sleep(duration_seconds)

        await client.stop_notify(HR_MEASUREMENT_UUID)

    logger.info(
        "Session complete: %d samples, %.0fs, %d RR intervals",
        session.sample_count,
        session.duration_seconds,
        len(session.all_rr_intervals),
    )

    return session
