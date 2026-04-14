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

    Uses find_device_by_filter which keeps the scanner context alive,
    producing a device reference that connects reliably on macOS.
    """
    logger.info("Scanning for Polar devices (%.0fs timeout)...", timeout)

    device = await BleakScanner.find_device_by_filter(
        lambda d, _adv: d.name is not None and "Polar" in d.name,
        timeout=timeout,
    )

    if device is not None:
        logger.info(
            "Found Polar device: %s (%s)", device.name, device.address
        )
    else:
        logger.warning("No Polar device found")

    return device


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
    session = PolarSession(start_time=time.time())

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

    # On macOS CoreBluetooth, the scanner must stay alive during connection.
    # We use the scanner context manager to discover + connect in one flow,
    # which keeps the internal CBPeripheral reference valid.
    if device is None:
        logger.info("Scanning for Polar device...")
        scanner = BleakScanner()
        device = await scanner.find_device_by_filter(
            lambda d, _adv: d.name is not None and "Polar" in d.name,
            timeout=15.0,
        )
        if device is None:
            raise RuntimeError("No Polar device found. Is the chest strap on?")

    session.device_name = device.name or "unknown"
    session.device_address = device.address

    logger.info(
        "Connecting to %s (%s) for %.0fs...",
        session.device_name, session.device_address, duration_seconds,
    )

    # Connect with retries — CoreBluetooth can be slow/flaky.
    # Create a fresh BleakClient per attempt with the BLEDevice object.
    max_retries = 3
    client: BleakClient | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client = BleakClient(device, timeout=30.0)
            await client.connect()
            logger.info("Connected on attempt %d", attempt)
            break
        except (TimeoutError, Exception) as e:
            if client and client.is_connected:
                await client.disconnect()
            if attempt < max_retries:
                logger.warning(
                    "Connection attempt %d/%d failed (%s), retrying...",
                    attempt, max_retries, type(e).__name__,
                )
                # Re-discover to get a fresh peripheral reference
                device = await BleakScanner.find_device_by_filter(
                    lambda d, _adv: d.name is not None and "Polar" in d.name,
                    timeout=10.0,
                )
                if device is None:
                    raise RuntimeError("Lost Polar device during retry")
                await asyncio.sleep(1.0)
            else:
                raise RuntimeError(
                    f"Failed to connect to {session.device_name} after "
                    f"{max_retries} attempts. Try: 1) Close Polar Beat/Flow apps, "
                    f"2) System Settings > Bluetooth > Forget the Polar device, "
                    f"3) Make sure strap is wet with skin contact."
                ) from e

    assert client is not None

    try:
        session.battery_level = await read_battery(client)

        await client.start_notify(HR_MEASUREMENT_UUID, _notification_handler)
        logger.info("Streaming heart rate data...")

        await asyncio.sleep(duration_seconds)

        await client.stop_notify(HR_MEASUREMENT_UUID)
    finally:
        await client.disconnect()

    logger.info(
        "Session complete: %d samples, %.0fs, %d RR intervals",
        session.sample_count,
        session.duration_seconds,
        len(session.all_rr_intervals),
    )

    return session
