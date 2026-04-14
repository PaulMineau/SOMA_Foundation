# SOMA Week 1 Spec — Polar H10 Logging Layer

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 1 — Body Signal Acquisition  
**Goal:** Establish continuous, reliable RR interval logging from Polar H10 to disk

---

## Architecture Overview

```
Polar H10 (BLE)
    ↓
bleak (Python BLE client)
    ↓
parse_rr() — raw bytes → milliseconds
    ↓
SQLite (soma_cardio.db) + CSV fallback
    ↓
[Week 2] RMSSD computation layer
    ↓
[Week 3] Baseline model + anomaly detection
    ↓
[Week 4] Conversational probe ("did something happen?")
```

---

## Week 1 Deliverables

- [ ] `polar_logger.py` — continuous RR logging to SQLite
- [ ] `soma_cardio.db` — auto-created on first run
- [ ] `rr_export.csv` — daily CSV export for inspection
- [ ] `monitor.py` — live terminal readout (RMSSD rolling window)
- [ ] README section: how to start a session

---

## File: `polar_logger.py`

```python
import asyncio
import sqlite3
import csv
from datetime import datetime
from bleak import BleakScanner, BleakClient

HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DB_PATH = "soma_cardio.db"
CSV_PATH = f"rr_export_{datetime.now().strftime('%Y%m%d')}.csv"


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rr_intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            rr_ms REAL NOT NULL,
            session_id TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            label TEXT
        )
    """)
    conn.commit()


def parse_rr(data):
    flag = data[0]
    rr_start = 3 if (flag & 0x01) else 2
    rr_intervals = []
    for i in range(rr_start, len(data) - 1, 2):
        rr = int.from_bytes(data[i:i+2], "little") / 1024 * 1000
        rr_intervals.append(round(rr, 2))
    return rr_intervals


def compute_rmssd(rr_buffer):
    if len(rr_buffer) < 2:
        return None
    diffs = [(rr_buffer[i+1] - rr_buffer[i])**2 for i in range(len(rr_buffer)-1)]
    return round((sum(diffs) / len(diffs)) ** 0.5, 2)


async def run_session(label=None):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now().isoformat()
    rr_buffer = []  # rolling 60-second window (~60 beats)
    MAX_BUFFER = 60

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    conn.execute(
        "INSERT INTO sessions (session_id, started_at, label) VALUES (?, ?, ?)",
        (session_id, started_at, label or "unlabeled")
    )
    conn.commit()

    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    print(f"\n🫀 SOMA-Cardio Session: {session_id}")
    print(f"   Label: {label or 'unlabeled'}")
    print(f"   Logging to: {DB_PATH}\n")

    def callback(sender, data):
        rr_list = parse_rr(data)
        if not rr_list:
            return

        ts = datetime.now().isoformat()
        for rr in rr_list:
            # Write to DB
            conn.execute(
                "INSERT INTO rr_intervals (timestamp, rr_ms, session_id) VALUES (?, ?, ?)",
                (ts, rr, session_id)
            )
            # Write to CSV
            writer.writerow([ts, rr, session_id])
            # Update rolling buffer
            rr_buffer.append(rr)
            if len(rr_buffer) > MAX_BUFFER:
                rr_buffer.pop(0)

        conn.commit()
        csv_file.flush()

        rmssd = compute_rmssd(rr_buffer)
        hr_est = round(60000 / rr_list[-1], 1) if rr_list else None
        print(f"  [{ts[11:19]}]  RR: {rr_list}ms  |  HR~{hr_est} bpm  |  RMSSD: {rmssd}ms")

    print("🔍 Scanning for Polar H10...")
    devices = await BleakScanner.discover(timeout=10)
    polar = next((d for d in devices if d.name and "Polar" in d.name), None)

    if not polar:
        print("❌ Polar device not found. Is the strap wet and on your chest?")
        return

    print(f"✅ Found: {polar.name} ({polar.address})\n")

    try:
        async with BleakClient(polar.address) as client:
            await client.start_notify(HR_UUID, callback)
            print("📡 Streaming... Press Ctrl+C to end session.\n")
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ended_at = datetime.now().isoformat()
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (ended_at, session_id)
        )
        conn.commit()
        conn.close()
        csv_file.close()
        print(f"\n✅ Session ended. Data saved to {DB_PATH}")


if __name__ == "__main__":
    import sys
    label = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    asyncio.run(run_session(label=label))
```

---

## Usage

### Start a labeled session
```bash
python polar_logger.py morning_baseline
python polar_logger.py post_run
python polar_logger.py work_stress
python polar_logger.py meditation
```

Labels matter — they become the training signal for the baseline model in Week 3.

### Stop a session
`Ctrl+C` — session is cleanly closed, timestamps written.

---

## File: `monitor.py` (Quick Live Check)

```python
import sqlite3
import time

DB_PATH = "soma_cardio.db"

def tail_rr(n=10):
    conn = sqlite3.connect(DB_PATH)
    while True:
        rows = conn.execute(
            f"SELECT timestamp, rr_ms FROM rr_intervals ORDER BY id DESC LIMIT {n}"
        ).fetchall()
        print("\033c", end="")  # clear terminal
        print("🫀 SOMA-Cardio Live Monitor\n")
        for ts, rr in reversed(rows):
            hr = round(60000 / rr, 1)
            print(f"  {ts[11:19]}  |  RR: {rr}ms  |  HR: {hr} bpm")
        time.sleep(2)

if __name__ == "__main__":
    tail_rr()
```

Run in a second terminal while `polar_logger.py` is running.

---

## Schema Reference

### `rr_intervals`
| column | type | notes |
|---|---|---|
| id | INTEGER | auto |
| timestamp | TEXT | ISO 8601 |
| rr_ms | REAL | milliseconds |
| session_id | TEXT | links to sessions |

### `sessions`
| column | type | notes |
|---|---|---|
| session_id | TEXT | YYYYMMDD_HHMMSS |
| started_at | TEXT | ISO 8601 |
| ended_at | TEXT | set on Ctrl+C |
| label | TEXT | e.g. "morning_baseline" |

---

## Week 2 Preview

- Compute RMSSD per session, store aggregate
- Plot HRV trend over time (matplotlib or Streamlit)
- First look at Paul's baseline distribution

## Week 3 Preview

- Morning baseline model (mean RHR, mean RMSSD, std dev)
- Anomaly threshold: > 1.5 std dev = flag
- Log anomaly events to new table

## Week 4 Preview

- Anomaly → conversational probe
- SOMA asks: *"RHR is up 11 bpm from your baseline. Did something happen?"*
- Response logged, fed back into context

---

*SOMA — Sentient Observation & Memory Architecture*  
*Phase 1: The body learns to speak.*
