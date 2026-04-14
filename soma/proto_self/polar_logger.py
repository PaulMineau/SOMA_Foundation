"""Continuous Polar H10 session logger — streams RR intervals to SQLite + CSV.

Usage:
    python -m soma.proto_self.polar_logger morning_baseline
    python -m soma.proto_self.polar_logger post_run
    python -m soma.proto_self.polar_logger meditation

Press Ctrl+C to end session. Labels become training signal for Week 3.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import sys
from datetime import datetime

from soma.proto_self.db import (
    end_session,
    export_daily_csv,
    get_connection,
    insert_rr,
    start_session,
)
from soma.proto_self.hrv import classify_body_state, compute_hrv
from soma.proto_self.polar_reader import (
    HR_MEASUREMENT_UUID,
    discover_polar,
    parse_hr_measurement,
    read_battery,
)

logger = logging.getLogger(__name__)


def _compute_rolling_rmssd(rr_buffer: list[float]) -> float | None:
    """Compute RMSSD from a rolling buffer of RR intervals."""
    if len(rr_buffer) < 2:
        return None
    diffs = [(rr_buffer[i + 1] - rr_buffer[i]) ** 2 for i in range(len(rr_buffer) - 1)]
    return round((sum(diffs) / len(diffs)) ** 0.5, 2)


async def run_session(label: str | None = None, db_path: str | None = None) -> None:
    """Run a continuous Polar H10 logging session.

    Streams RR intervals to SQLite and CSV until Ctrl+C.
    On exit, computes session HRV summary and stores it.
    """
    from bleak import BleakClient, BleakScanner

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = label or "unlabeled"

    rr_buffer: list[float] = []  # Rolling 60-beat window
    all_rr: list[float] = []  # All RR intervals for session summary
    max_buffer = 60

    conn = get_connection(db_path)

    # CSV setup
    csv_dir = "data"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"rr_export_{datetime.now().strftime('%Y%m%d')}.csv")
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)

    print(f"\nSOMA-Cardio Session: {session_id}")
    print(f"   Label: {label}")
    print(f"   Logging to: {conn.execute('PRAGMA database_list').fetchone()[2]}")
    print(f"   CSV: {csv_path}\n")

    # Discover device
    print("Scanning for Polar H10...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, _adv: d.name is not None and "Polar" in d.name,
        timeout=15.0,
    )
    if device is None:
        print("Polar device not found. Is the strap wet and on your chest?")
        conn.close()
        csv_file.close()
        return

    print(f"Found: {device.name} ({device.address})\n")

    # Connect with retry
    client: BleakClient | None = None
    for attempt in range(1, 4):
        try:
            client = BleakClient(device, timeout=30.0)
            await client.connect()
            print(f"Connected (attempt {attempt})")
            break
        except (TimeoutError, Exception) as e:
            if client and client.is_connected:
                await client.disconnect()
            if attempt < 3:
                print(f"Connection attempt {attempt}/3 failed ({type(e).__name__}), retrying...")
                device = await BleakScanner.find_device_by_filter(
                    lambda d, _adv: d.name is not None and "Polar" in d.name,
                    timeout=10.0,
                )
                if device is None:
                    print("Lost Polar device during retry")
                    conn.close()
                    csv_file.close()
                    return
                await asyncio.sleep(1.0)
            else:
                print(f"Failed to connect after 3 attempts. Try forgetting the device in Bluetooth settings.")
                conn.close()
                csv_file.close()
                return

    assert client is not None

    # Read battery
    battery = None
    try:
        from soma.proto_self.polar_reader import BATTERY_UUID
        data = await client.read_gatt_char(BATTERY_UUID)
        battery = data[0]
        print(f"Battery: {battery}%")
    except Exception:
        pass

    # Start session in DB
    start_session(
        conn, session_id, label=label,
        device_name=device.name or "", device_address=device.address,
        battery_level=battery,
    )

    def callback(_sender: int, data: bytearray) -> None:
        sample = parse_hr_measurement(data)

        if not sample.sensor_contact:
            return

        ts = datetime.now().isoformat()
        for rr in sample.rr_intervals:
            insert_rr(conn, session_id, rr, sample.heart_rate, ts)
            writer.writerow([ts, rr, sample.heart_rate, session_id])
            all_rr.append(rr)
            rr_buffer.append(rr)
            if len(rr_buffer) > max_buffer:
                rr_buffer.pop(0)

        conn.commit()
        csv_file.flush()

        if sample.rr_intervals:
            rmssd = _compute_rolling_rmssd(rr_buffer)
            hr = sample.heart_rate
            rr_str = ", ".join(f"{r:.0f}" for r in sample.rr_intervals)
            rmssd_str = f"{rmssd:.1f}" if rmssd else "---"
            print(f"  [{ts[11:19]}]  RR: [{rr_str}]ms  |  HR: {hr} bpm  |  RMSSD: {rmssd_str}ms")

    try:
        await client.start_notify(HR_MEASUREMENT_UUID, callback)
        print(f"\nStreaming... Press Ctrl+C to end session.\n")
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        try:
            await client.stop_notify(HR_MEASUREMENT_UUID)
        except Exception:
            pass
        await client.disconnect()

        # Compute session summary
        mean_hr = None
        rmssd = None
        sdnn = None
        body_state = None

        if len(all_rr) >= 3:
            try:
                metrics = compute_hrv(all_rr)
                mean_hr = metrics.mean_hr
                rmssd = metrics.rmssd
                sdnn = metrics.sdnn
                body_state = classify_body_state(metrics)
            except ValueError:
                pass

        end_session(
            conn, session_id,
            n_intervals=len(all_rr),
            mean_hr=mean_hr,
            rmssd=rmssd,
            sdnn=sdnn,
            body_state=body_state,
        )

        conn.close()
        csv_file.close()

        print(f"\n{'=' * 50}")
        print(f"Session ended: {session_id}")
        print(f"  Intervals: {len(all_rr)}")
        if rmssd is not None:
            print(f"  RMSSD: {rmssd:.1f} ms")
            print(f"  SDNN: {sdnn:.1f} ms")
            print(f"  Mean HR: {mean_hr:.0f} bpm")
            print(f"  Body State: {body_state}")
        print(f"  Data: soma_cardio.db + {csv_path}")
        print()


def main() -> None:
    label = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(run_session(label=label))


if __name__ == "__main__":
    main()
