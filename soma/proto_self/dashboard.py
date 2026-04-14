"""SOMA Core Consciousness Dashboard — Streamlit live readout.

Shows current HRV, baseline overlay, anomaly log, and session history.
Auto-refreshes every 10 seconds.

Usage:
    streamlit run soma/proto_self/dashboard.py
"""

from __future__ import annotations

import os
import sqlite3
import json
import time

import pandas as pd
import streamlit as st

from soma.proto_self.artifact_filter import clean_rr, compute_rhr, compute_rmssd
from soma.proto_self.baseline_model import MODEL_PATH
from soma.proto_self.db import DEFAULT_DB_PATH

DB_PATH = os.environ.get("SOMA_CARDIO_DB", DEFAULT_DB_PATH)

st.set_page_config(
    page_title="SOMA",
    page_icon="",
    layout="wide",
)


def load_model() -> dict | None:
    try:
        with open(MODEL_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_recent_rr(n: int = 300) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, rr_ms FROM rr_intervals ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["timestamp", "rr_ms"])


def get_anomalies() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT detected_at, metric, value, baseline, deviation "
            "FROM anomalies ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["detected_at", "metric", "value", "baseline", "deviation"])
    except sqlite3.OperationalError:
        conn.close()
        return pd.DataFrame()


def get_sessions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT session_id, started_at, ended_at, label, n_intervals, rmssd, body_state "
        "FROM sessions ORDER BY started_at DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return pd.DataFrame(
        rows,
        columns=["session_id", "started_at", "ended_at", "label", "intervals", "rmssd", "body_state"],
    )


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("SOMA — Core Consciousness")

model = load_model()
df = get_recent_rr()

if df.empty:
    st.warning("No RR data found. Start a session: python -m soma.proto_self.polar_logger morning_baseline")
    st.stop()

# Clean and compute current window
clean = clean_rr(df["rr_ms"].tolist())
current_rhr = compute_rhr(clean)
current_rmssd = compute_rmssd(clean)

# ── Top metrics ───────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Heart Rate", f"{current_rhr} bpm" if current_rhr else "---")

with col2:
    st.metric("RMSSD", f"{current_rmssd} ms" if current_rmssd else "---")

with col3:
    if model and current_rhr and model["rhr"]["std"] > 0:
        rhr_dev = round((current_rhr - model["rhr"]["mean"]) / model["rhr"]["std"], 1)
        delta_bpm = round(current_rhr - model["rhr"]["mean"], 1)
        st.metric("RHR vs Baseline", f"{rhr_dev:+.1f}s", delta=f"{delta_bpm:+.1f} bpm")
    else:
        st.metric("RHR vs Baseline", "No model yet")

with col4:
    if model and current_rmssd and model["rmssd"]["std"] > 0:
        rmssd_dev = round((current_rmssd - model["rmssd"]["mean"]) / model["rmssd"]["std"], 1)
        delta_ms = round(current_rmssd - model["rmssd"]["mean"], 1)
        st.metric("RMSSD vs Baseline", f"{rmssd_dev:+.1f}s", delta=f"{delta_ms:+.1f} ms")
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

if not df_clean.empty:
    st.line_chart(df_clean.set_index("timestamp")["rr_ms"])
else:
    st.info("No clean RR intervals to chart.")

# ── Baseline overlay ──────────────────────────────────────────────────────────

if model:
    st.subheader("Baseline Reference")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.write(f"**RHR baseline:** {model['rhr']['mean']} +/- {model['rhr']['std']} bpm")
        st.write(f"Alert threshold: > {model['rhr']['alert_threshold_high']} bpm")
    with bcol2:
        st.write(f"**RMSSD baseline:** {model['rmssd']['mean']} +/- {model['rmssd']['std']} ms")
        st.write(f"Alert threshold: < {model['rmssd']['alert_threshold_low']} ms")
    st.caption(f"Model generated: {model['generated_at'][:19]} from {model['sample_count']} samples")

st.divider()

# ── Anomalies ─────────────────────────────────────────────────────────────────

st.subheader("Anomaly Log")
anomalies = get_anomalies()
if anomalies.empty:
    st.success("No anomalies detected yet.")
else:
    st.dataframe(anomalies, use_container_width=True)

st.divider()

# ── Recommendations ──────────────────────────────────────────────────────────

st.subheader("SOMA Recommends")

try:
    from soma.proto_self.recommender import get_recommendations, log_recommendation

    if st.button("Get Recommendations"):
        result = get_recommendations(n=3)
        rec_state = result["state"]

        st.write(f"**Current state:** {rec_state['state'].upper()} — {rec_state['reason']}")

        for rec in result["recommendations"]:
            with st.expander(f"[{rec['type'].upper()}] {rec['title']} (~{rec['duration_min']} min)"):
                st.write(rec["why"])
                st.write(f"Tags: {', '.join(rec.get('tags', []))}")
                if st.button(f"I did this", key=f"did_{rec['id']}"):
                    log_recommendation(rec["id"], rec["title"], rec["type"], rec_state)
                    st.success("Logged! Run feedback_logger to rate the outcome.")
except Exception as e:
    st.info(f"Recommendations unavailable: {e}")

st.divider()

# ── Reading Queue ────────────────────────────────────────────────────────────

st.subheader("Reading Queue")

try:
    from soma.proto_self.substack_agent import QUEUE_PATH
    from soma.proto_self.state_classifier import classify_state as _classify

    queue_data: list[dict] = []
    try:
        with open(QUEUE_PATH) as _f:
            queue_data = json.load(_f)
    except FileNotFoundError:
        pass

    _current_state = _classify().get("state", "unknown")
    readable = [
        a for a in queue_data
        if (a.get("approved") or a.get("auto_surfaced"))
        and not a.get("read")
        and _current_state in a.get("best_states", [_current_state])
        and _current_state not in a.get("avoid_states", [])
    ]
    readable.sort(key=lambda x: x.get("raen_total", 0), reverse=True)

    if not readable:
        st.info(f"No articles matched for current state: {_current_state.upper()}")
    else:
        st.write(f"Showing {len(readable)} articles for state: **{_current_state.upper()}**")
        for a in readable[:5]:
            auto = "+" if a.get("auto_surfaced") else ""
            with st.expander(f"{auto} {a.get('title', '?')} — {a.get('newsletter', '?')} (~{a.get('duration_min', '?')} min)"):
                st.write(a.get("key_insight", a.get("why", "")))
                st.write(f"**Best for:** {', '.join(a.get('best_states', []))}")
                if a.get("url"):
                    st.markdown(f"[Read article]({a['url']})")
except Exception as e:
    st.info(f"Reading queue unavailable: {e}")

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
