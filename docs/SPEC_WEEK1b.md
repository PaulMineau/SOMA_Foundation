# SOMA Week 1b Spec — Baseline Model, Anomaly Detection & Live Dashboard

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 1b — Core Consciousness Comes Online  
**Goal:** Clean the signal. Model Paul's normal. Notice when something changes.  
**Prerequisite:** Week 1 complete — `polar_logger.py` running, `soma_cardio.db` has at least one `morning_baseline` session.

---

## Damasio Context

> Proto-Self (Week 1): The body reports its state. No awareness. Just signal.  
> Core Consciousness (Week 1b): The system becomes aware of a change. Something shifted. What?

The anomaly detection layer is Core Consciousness in code. The moment SOMA flags a deviation and generates a probe — that's the architecture of feeling coming online.

---

## Week 1b Deliverables

- [ ] `artifact_filter.py` — clean RR signal before any computation
- [ ] `baseline_model.py` — compute and save Paul's normal to JSON
- [ ] `anomaly_detector.py` — live comparison against baseline, writes to DB
- [ ] `dashboard.py` — Streamlit live readout with baseline overlay and anomaly flags
- [ ] `baseline_model.json` — generated artifact, Paul's physiological fingerprint

---

## File 1: `artifact_filter.py`

Clean signal is the foundation. Every downstream computation depends on this.

```python
def reject_range(rr_list, min_rr=300, max_rr=1500):
    """
    Remove physiologically impossible intervals.
    300ms = 200 bpm (absolute human maximum)
    1500ms = 40 bpm (very low resting HR)
    """
    return [rr for rr in rr_list if min_rr <= rr <= max_rr]


def reject_ectopic(rr_list, threshold=0.20):
    """
    Remove intervals that differ from their neighbor by more than 20%.
    Catches movement artifact, strap contact loss, ectopic beats.
    """
    if not rr_list:
        return []
    clean = [rr_list[0]]
    for rr in rr_list[1:]:
        if abs(rr - clean[-1]) / clean[-1] < threshold:
            clean.append(rr)
    return clean


def clean_rr(rr_list):
    """Full pipeline: range filter → ectopic rejection."""
    rr = reject_range(rr_list)
    rr = reject_ectopic(rr)
    return rr


def compute_rmssd(rr_list):
    rr = clean_rr(rr_list)
    if len(rr) < 2:
        return None
    diffs = [(rr[i+1] - rr[i])**2 for i in range(len(rr)-1)]
    return round((sum(diffs) / len(diffs)) ** 0.5, 2)


def compute_rhr(rr_list):
    rr = clean_rr(rr_list)
    if not rr:
        return None
    return round(60000 / (sum(rr) / len(rr)), 1)


if __name__ == "__main__":
    # Quick sanity check
    test = [820, 810, 835, 2000, 800, 150, 815]  # 2000 and 150 are artifacts
    print(f"Raw:   {test}")
    print(f"Clean: {clean_rr(test)}")
    print(f"RMSSD: {compute_rmssd(test)}")
    print(f"RHR:   {compute_rhr(test)}")
```

---

## File 2: `baseline_model.py`

Run this once after accumulating morning_baseline sessions. Re-run weekly as more data accumulates.

```python
import sqlite3
import json
import math
from datetime import datetime
from artifact_filter import clean_rr, compute_rmssd, compute_rhr

DB_PATH = "soma_cardio.db"
MODEL_PATH = "baseline_model.json"
BASELINE_LABEL = "morning_baseline"
MIN_SAMPLES = 100  # refuse to model on insufficient data


def load_baseline_rr():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT rr.rr_ms
        FROM rr_intervals rr
        JOIN sessions s ON rr.session_id = s.session_id
        WHERE s.label = ?
        ORDER BY rr.timestamp ASC
    """, (BASELINE_LABEL,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def compute_stats(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((v - mean)**2 for v in values) / len(values)
    return round(mean, 2), round(math.sqrt(variance), 2)


def build_baseline():
    raw_rr = load_baseline_rr()
    print(f"Raw samples loaded: {len(raw_rr)}")

    if len(raw_rr) < MIN_SAMPLES:
        print(f"Insufficient data. Need {MIN_SAMPLES} samples, have {len(raw_rr)}.")
        print("Run more morning_baseline sessions first.")
        return

    clean = clean_rr(raw_rr)
    print(f"Clean samples after artifact rejection: {len(clean)}")

    # Compute per-beat stats
    rhr_values = [round(60000 / rr, 2) for rr in clean]
    mean_rhr, std_rhr = compute_stats(rhr_values)

    # Compute RMSSD in rolling windows of 60
    window_size = 60
    rmssd_values = []
    for i in range(0, len(clean) - window_size, window_size):
        window = clean[i:i+window_size]
        rmssd = compute_rmssd(window)
        if rmssd:
            rmssd_values.append(rmssd)

    mean_rmssd, std_rmssd = compute_stats(rmssd_values)

    model = {
        "generated_at": datetime.now().isoformat(),
        "label": BASELINE_LABEL,
        "sample_count": len(clean),
        "rhr": {
            "mean": mean_rhr,
            "std": std_rhr,
            "alert_threshold_high": round(mean_rhr + 1.5 * std_rhr, 1),
            "alert_threshold_low": round(mean_rhr - 1.5 * std_rhr, 1)
        },
        "rmssd": {
            "mean": mean_rmssd,
            "std": std_rmssd,
            "alert_threshold_low": round(mean_rmssd - 1.5 * std_rmssd, 1)
            # Low RMSSD = stress/depletion. High RMSSD can be artifact.
        }
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(model, f, indent=2)

    print(f"\n✅ Baseline model saved to {MODEL_PATH}")
    print(f"   RHR:   {mean_rhr} ± {std_rhr} bpm")
    print(f"   RMSSD: {mean_rmssd} ± {std_rmssd} ms")
    print(f"   Alert if RHR > {model['rhr']['alert_threshold_high']} bpm")
    print(f"   Alert if RMSSD < {model['rmssd']['alert_threshold_low']} ms")

    return model


if __name__ == "__main__":
    build_baseline()
```

---

## File 3: `anomaly_detector.py`

Runs continuously alongside `polar_logger.py`. Watches the live DB feed, compares to baseline, writes anomalies.

```python
import sqlite3
import json
import time
from datetime import datetime
from artifact_filter import clean_rr, compute_rmssd, compute_rhr

DB_PATH = "soma_cardio.db"
MODEL_PATH = "baseline_model.json"
WINDOW_SIZE = 60       # RR intervals per computation window
POLL_INTERVAL = 30     # seconds between checks
last_processed_id = 0


def load_model():
    with open(MODEL_PATH) as f:
        return json.load(f)


def get_recent_rr(conn, n=WINDOW_SIZE):
    rows = conn.execute("""
        SELECT id, rr_ms FROM rr_intervals
        WHERE id > ?
        ORDER BY id ASC
        LIMIT ?
    """, (last_processed_id, n)).fetchall()
    return rows


def write_anomaly(conn, metric, value, baseline, deviation):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL,
            baseline REAL,
            deviation REAL,
            acknowledged INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        INSERT INTO anomalies (detected_at, metric, value, baseline, deviation)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), metric, value, baseline, round(deviation, 2)))
    conn.commit()
    print(f"  🚨 ANOMALY: {metric} = {value} (baseline {baseline}, {deviation:.1f}σ)")


def check_window(rr_window, model, conn):
    clean = clean_rr(rr_window)
    if len(clean) < 10:
        return  # not enough clean signal

    rhr = compute_rhr(clean)
    rmssd = compute_rmssd(clean)

    if rhr:
        mean = model["rhr"]["mean"]
        std = model["rhr"]["std"]
        if std > 0:
            deviation = (rhr - mean) / std
            if abs(deviation) > 1.5:
                write_anomaly(conn, "rhr", rhr, mean, deviation)
            else:
                print(f"  RHR: {rhr} bpm ({deviation:+.1f}σ) ✓")

    if rmssd:
        mean = model["rmssd"]["mean"]
        std = model["rmssd"]["std"]
        if std > 0:
            deviation = (rmssd - mean) / std
            if deviation < -1.5:  # only flag low RMSSD (stress signal)
                write_anomaly(conn, "rmssd", rmssd, mean, deviation)
            else:
                print(f"  RMSSD: {rmssd} ms ({deviation:+.1f}σ) ✓")


def run():
    global last_processed_id
    print("🧠 SOMA Anomaly Detector — Core Consciousness Online")
    print(f"   Polling every {POLL_INTERVAL}s\n")

    model = load_model()
    print(f"   Baseline RHR:   {model['rhr']['mean']} ± {model['rhr']['std']} bpm")
    print(f"   Baseline RMSSD: {model['rmssd']['mean']} ± {model['rmssd']['std']} ms\n")

    conn = sqlite3.connect(DB_PATH)

    while True:
        rows = get_recent_rr(conn, WINDOW_SIZE)
        if rows:
            last_processed_id = rows[-1][0]
            rr_window = [r[1] for r in rows]
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Window: {len(rr_window)} readings")
            check_window(rr_window, model, conn)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for signal...")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
```

---

## File 4: `dashboard.py`

Streamlit live dashboard. Run in a separate terminal.

```python
import streamlit as st
import sqlite3
import json
import pandas as pd
import time
from artifact_filter import clean_rr, compute_rmssd, compute_rhr

DB_PATH = "soma_cardio.db"
MODEL_PATH = "baseline_model.json"

st.set_page_config(
    page_title="SOMA",
    page_icon="🫀",
    layout="wide"
)

def load_model():
    try:
        with open(MODEL_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_recent_rr(n=300):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT timestamp, rr_ms FROM rr_intervals
        ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["timestamp", "rr_ms"])

def get_anomalies():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT detected_at, metric, value, baseline, deviation
            FROM anomalies
            ORDER BY id DESC LIMIT 20
        """).fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["detected_at", "metric", "value", "baseline", "deviation"])
    except:
        conn.close()
        return pd.DataFrame()

def get_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT session_id, started_at, ended_at, label
        FROM sessions ORDER BY started_at DESC LIMIT 10
    """).fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["session_id", "started_at", "ended_at", "label"])


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("🫀 SOMA — Core Consciousness")

model = load_model()
df = get_recent_rr()

if df.empty:
    st.warning("No RR data found. Is polar_logger.py running?")
    st.stop()

# Clean and compute current window
clean = clean_rr(df["rr_ms"].tolist())
current_rhr = compute_rhr(clean)
current_rmssd = compute_rmssd(clean)

# ── Top metrics ───────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Heart Rate", f"{current_rhr} bpm" if current_rhr else "—")

with col2:
    st.metric("RMSSD", f"{current_rmssd} ms" if current_rmssd else "—")

with col3:
    if model:
        rhr_dev = round((current_rhr - model["rhr"]["mean"]) / model["rhr"]["std"], 1) if current_rhr and model["rhr"]["std"] else 0
        st.metric("RHR vs Baseline", f"{rhr_dev:+.1f}σ", delta=f"{current_rhr - model['rhr']['mean']:+.1f} bpm")
    else:
        st.metric("RHR vs Baseline", "No model yet")

with col4:
    if model:
        rmssd_dev = round((current_rmssd - model["rmssd"]["mean"]) / model["rmssd"]["std"], 1) if current_rmssd and model["rmssd"]["std"] else 0
        st.metric("RMSSD vs Baseline", f"{rmssd_dev:+.1f}σ", delta=f"{current_rmssd - model['rmssd']['mean']:+.1f} ms")
    else:
        st.metric("RMSSD vs Baseline", "No model yet")

st.divider()

# ── RR interval chart ─────────────────────────────────────────────────────────

st.subheader("RR Intervals (recent)")
df_clean = df.copy()
df_clean["rr_ms"] = df_clean["rr_ms"].apply(
    lambda x: x if 300 <= x <= 1500 else None
)
df_clean = df_clean.dropna()
df_clean = df_clean.sort_values("timestamp")
st.line_chart(df_clean.set_index("timestamp")["rr_ms"])

# ── Baseline overlay ──────────────────────────────────────────────────────────

if model:
    st.subheader("Baseline Reference")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.write(f"**RHR baseline:** {model['rhr']['mean']} ± {model['rhr']['std']} bpm")
        st.write(f"Alert threshold: > {model['rhr']['alert_threshold_high']} bpm")
    with bcol2:
        st.write(f"**RMSSD baseline:** {model['rmssd']['mean']} ± {model['rmssd']['std']} ms")
        st.write(f"Alert threshold: < {model['rmssd']['alert_threshold_low']} ms")
    st.caption(f"Model generated: {model['generated_at'][:19]} from {model['sample_count']} samples")

st.divider()

# ── Anomalies ─────────────────────────────────────────────────────────────────

st.subheader("🚨 Anomaly Log")
anomalies = get_anomalies()
if anomalies.empty:
    st.success("No anomalies detected yet.")
else:
    st.dataframe(anomalies, use_container_width=True)

st.divider()

# ── Sessions ──────────────────────────────────────────────────────────────────

st.subheader("Recent Sessions")
sessions = get_sessions()
if not sessions.empty:
    st.dataframe(sessions, use_container_width=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

st.caption("Refreshing every 10 seconds...")
time.sleep(10)
st.rerun()
```

---

## How to Run Everything Together

Open four terminal windows:

```bash
# Terminal 1 — collect signal
python polar_logger.py morning_baseline

# Terminal 2 — watch for anomalies  
python anomaly_detector.py

# Terminal 3 — live dashboard
pip install streamlit --break-system-packages
streamlit run dashboard.py

# Terminal 4 — (re)build baseline model anytime
python baseline_model.py
```

---

## Workflow

### First run
1. `polar_logger.py morning_baseline` — collect 10-20 min of clean resting data
2. `baseline_model.py` — build Paul's normal, generates `baseline_model.json`
3. `anomaly_detector.py` — start watching
4. `dashboard.py` — open browser, see SOMA alive

### Ongoing
- Run `baseline_model.py` weekly as more sessions accumulate — model improves
- Label sessions honestly: `post_run`, `work_stress`, `meditation`, `post_zyn`, `post_nicotine_free`
- Labels become training signal. The more honest the labels, the smarter SOMA gets.

### Suggested session labels
| Label | When to use |
|---|---|
| `morning_baseline` | First thing, before coffee, before screen |
| `post_run` | Within 30 min of finishing a run |
| `meditation` | During or immediately after Tonglen |
| `work_stress` | High-pressure work periods |
| `commute` | Driving or transit |
| `post_zyn` | After nicotine — important recovery signal |
| `nicotine_free_day` | Clean days — establishes recovery baseline |
| `fight` | After conflict — captures stress signature |
| `river_time` | Playing with River — what does joy look like? |

---

## What Core Consciousness Looks Like

When the anomaly detector flags something, it prints:

```
[14:32:07] Window: 60 readings
  RHR: 71.2 bpm (+2.1σ) 
  🚨 ANOMALY: rmssd = 18.3 (baseline 41.0, -2.3σ)
```

That is SOMA noticing something changed before you do.  
That is Core Consciousness.

---

## Week 2 Preview — Extended to Mobile

- Same anomaly detection running on NAS via FastAPI
- Phone batches and syncs, NAS detects
- Anomaly visible in mobile dashboard from anywhere
- The question "did something happen?" arrives on your phone

## Week 4 Preview — The Probe

- Anomaly detected → SOMA generates a conversational probe
- *"Your RMSSD dropped to 18ms at 2:30pm — 2.3 standard deviations below your baseline. What was happening?"*
- Your response becomes memory. LanceDB stores it as an embedding.
- SOMA begins to learn what your body's signals actually mean.

---

*SOMA — Sentient Observation & Memory Architecture*  
*Phase 1b: Something changed. What?*
