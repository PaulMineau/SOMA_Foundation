"""ResMed EDF file parser — reads SD card data for time-resolved CPAP signals.

Parses OSCAR-format EDF+ files from ResMed AirSense 10/11 SD cards.
Extracts:
  - Timestamped apnea/hypopnea events
  - Pressure waveforms
  - Flow rate, tidal volume, leak rate
  - SpO2 (if oximeter attached)

Usage:
    python -m soma.proto_self.cpap.edf_parser /path/to/DATALOG/20260414/
    python -m soma.proto_self.cpap.edf_parser ~/ResMed_SDCard/DATALOG --last-n 7

Requires: pip install pyedflib numpy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CPAPEvent:
    """A single timestamped CPAP event (apnea, hypopnea, etc.)."""

    timestamp: datetime
    event_type: str  # "apnea" | "hypopnea" | "flow_limitation" | "snore" | "leak"
    duration_sec: float = 0.0
    magnitude: float = 0.0


@dataclass
class CPAPNightSummary:
    """One night of CPAP data parsed from EDF files."""

    date: str  # YYYY-MM-DD
    start_time: datetime
    end_time: datetime
    duration_min: int
    total_events: int
    apneas: int
    hypopneas: int
    ahi: float  # events per hour
    mean_pressure: float
    median_leak: float
    p95_leak: float
    events: list[CPAPEvent] = field(default_factory=list)
    # Time-series (sampled at 1-sec intervals where available)
    pressure_samples: list[float] = field(default_factory=list)
    leak_samples: list[float] = field(default_factory=list)
    flow_samples: list[float] = field(default_factory=list)


def _parse_edf_file(path: Path) -> dict[str, Any]:
    """Parse a single EDF file and return signals as a dict.

    Uses pyedflib if available, falls back to structured error.
    """
    try:
        import pyedflib
    except ImportError:
        raise RuntimeError(
            "pyedflib not installed. Run: pip install pyedflib numpy"
        )

    with pyedflib.EdfReader(str(path)) as f:
        n_signals = f.signals_in_file
        labels = f.getSignalLabels()
        start_datetime = f.getStartdatetime()

        signals: dict[str, Any] = {
            "path": str(path),
            "start_time": start_datetime,
            "labels": labels,
            "signals": {},
        }

        for i in range(n_signals):
            label = labels[i].strip()
            data = f.readSignal(i)
            sample_rate = f.getSampleFrequency(i)
            signals["signals"][label] = {
                "data": data,
                "sample_rate": sample_rate,
                "n_samples": len(data),
            }

        return signals


def parse_night_directory(night_dir: Path) -> CPAPNightSummary | None:
    """Parse all EDF files in a single night's DATALOG directory.

    ResMed layout:
        DATALOG/YYYYMMDD/YYYYMMDD_HHMMSS_BRP.edf   - breath-level data
        DATALOG/YYYYMMDD/YYYYMMDD_HHMMSS_CSL.edf   - crash log
        DATALOG/YYYYMMDD/YYYYMMDD_HHMMSS_EVE.edf   - events (apnea, hypopnea)
        DATALOG/YYYYMMDD/YYYYMMDD_HHMMSS_PLD.edf   - pressure, leak, duration
        DATALOG/YYYYMMDD/YYYYMMDD_HHMMSS_SAD.edf   - SpO2 (if oximeter attached)
    """
    if not night_dir.is_dir():
        return None

    edf_files = sorted(night_dir.glob("*.edf"))
    if not edf_files:
        return None

    import numpy as np

    date_str = night_dir.name[:4] + "-" + night_dir.name[4:6] + "-" + night_dir.name[6:8]

    events: list[CPAPEvent] = []
    pressure_samples: list[float] = []
    leak_samples: list[float] = []
    flow_samples: list[float] = []
    start_time: datetime | None = None
    end_time: datetime | None = None

    for edf_path in edf_files:
        try:
            parsed = _parse_edf_file(edf_path)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", edf_path.name, e)
            continue

        ts = parsed["start_time"]
        if start_time is None or ts < start_time:
            start_time = ts

        signals = parsed["signals"]

        # Events file (EVE)
        if "_EVE.edf" in edf_path.name:
            # EVE files typically use annotation-style channels
            for label, sig in signals.items():
                lower = label.lower()
                if "apnea" in lower or "hyp" in lower:
                    data = sig["data"]
                    rate = sig["sample_rate"]
                    # Find non-zero samples = events
                    for idx, val in enumerate(data):
                        if val > 0:
                            event_time = ts + timedelta(seconds=idx / rate)
                            event_type = "apnea" if "apnea" in lower else "hypopnea"
                            events.append(CPAPEvent(
                                timestamp=event_time,
                                event_type=event_type,
                                duration_sec=float(val),
                            ))

        # Pressure/leak/duration file (PLD)
        if "_PLD.edf" in edf_path.name:
            for label, sig in signals.items():
                data = sig["data"]
                lower = label.lower()
                if "press" in lower:
                    pressure_samples.extend(float(v) for v in data)
                elif "leak" in lower:
                    leak_samples.extend(float(v) for v in data)

        # Breath-level file (BRP)
        if "_BRP.edf" in edf_path.name:
            for label, sig in signals.items():
                if "flow" in label.lower():
                    data = sig["data"]
                    # Downsample to 1-second averages to keep size manageable
                    rate = sig["sample_rate"]
                    window = int(rate)
                    for i in range(0, len(data), window):
                        chunk = data[i:i + window]
                        flow_samples.append(float(np.mean(chunk)))

        # Track end time
        duration_sec = max(
            (s["n_samples"] / s["sample_rate"]) for s in signals.values()
        ) if signals else 0
        file_end = ts + timedelta(seconds=duration_sec)
        if end_time is None or file_end > end_time:
            end_time = file_end

    if start_time is None or end_time is None:
        return None

    duration_min = int((end_time - start_time).total_seconds() / 60)

    apneas = sum(1 for e in events if e.event_type == "apnea")
    hypopneas = sum(1 for e in events if e.event_type == "hypopnea")
    total_events = apneas + hypopneas
    ahi = (total_events / (duration_min / 60)) if duration_min > 0 else 0.0

    # Pressure stats
    mean_pressure = float(np.mean(pressure_samples)) if pressure_samples else 0.0

    # Leak stats
    if leak_samples:
        leak_arr = np.array(leak_samples)
        median_leak = float(np.median(leak_arr))
        p95_leak = float(np.percentile(leak_arr, 95))
    else:
        median_leak = 0.0
        p95_leak = 0.0

    return CPAPNightSummary(
        date=date_str,
        start_time=start_time,
        end_time=end_time,
        duration_min=duration_min,
        total_events=total_events,
        apneas=apneas,
        hypopneas=hypopneas,
        ahi=round(ahi, 2),
        mean_pressure=round(mean_pressure, 2),
        median_leak=round(median_leak, 2),
        p95_leak=round(p95_leak, 2),
        events=events,
        pressure_samples=pressure_samples,
        leak_samples=leak_samples,
        flow_samples=flow_samples,
    )


def parse_datalog(datalog_dir: str | Path, last_n: int | None = None) -> list[CPAPNightSummary]:
    """Parse a whole DATALOG directory (multiple nights)."""
    datalog_path = Path(os.path.expanduser(str(datalog_dir)))
    if not datalog_path.is_dir():
        raise FileNotFoundError(f"DATALOG dir not found: {datalog_path}")

    # Night directories are YYYYMMDD format
    night_dirs = sorted(
        [d for d in datalog_path.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8],
        reverse=True,
    )

    if last_n:
        night_dirs = night_dirs[:last_n]

    summaries: list[CPAPNightSummary] = []
    for night_dir in night_dirs:
        logger.info("Parsing night: %s", night_dir.name)
        summary = parse_night_directory(night_dir)
        if summary:
            summaries.append(summary)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ResMed SD card EDF files")
    parser.add_argument("datalog_path", help="Path to DATALOG directory or single night dir")
    parser.add_argument("--last-n", type=int, default=None, help="Parse only last N nights")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    path = Path(os.path.expanduser(args.datalog_path))

    # Detect if it's a single night or full DATALOG
    if path.name.isdigit() and len(path.name) == 8:
        summaries = [parse_night_directory(path)] if parse_night_directory(path) else []
    else:
        summaries = parse_datalog(path, last_n=args.last_n)

    print(f"\nParsed {len(summaries)} nights\n")
    for s in summaries:
        print(f"  {s.date}  AHI: {s.ahi:.1f}  "
              f"Duration: {s.duration_min // 60}h{s.duration_min % 60}m  "
              f"Pressure: {s.mean_pressure:.1f}cmH2O  "
              f"Leak p95: {s.p95_leak:.1f}L/min  "
              f"Apneas: {s.apneas}  Hypopneas: {s.hypopneas}")


if __name__ == "__main__":
    main()
