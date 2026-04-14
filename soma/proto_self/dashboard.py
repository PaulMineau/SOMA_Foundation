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

# Clean and compute current Polar window (may be empty if no strap on)
current_rhr = None
current_rmssd = None
if not df.empty:
    clean = clean_rr(df["rr_ms"].tolist())
    current_rhr = compute_rhr(clean)
    current_rmssd = compute_rmssd(clean)

# ── Fitbit Overnight Data ────────────────────────────────────────────────────

_fb_data = None
_fb_trends = None
try:
    from soma.proto_self.fitbit.fitbit_dashboard import (
        get_today_fitbit,
        get_recent_fitbit_days,
        get_fitbit_trends,
    )
    _fb_data = get_today_fitbit()
    _fb_trends = get_fitbit_trends(days=7)
except Exception:
    pass

if _fb_data:
    st.subheader("Overnight Recovery (Fitbit)")

    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        recovery = _fb_data.get("recovery_score", 0)
        st.metric("Recovery", f"{recovery}/10")
    with fc2:
        overnight_hrv = _fb_data.get("hrv_rmssd", 0)
        st.metric("Overnight HRV", f"{overnight_hrv:.0f} ms" if overnight_hrv else "---")
    with fc3:
        sleep_min = _fb_data.get("sleep_duration_min", 0)
        st.metric("Sleep", f"{sleep_min // 60}h {sleep_min % 60}m")
    with fc4:
        deep = _fb_data.get("deep_sleep_min", 0)
        st.metric("Deep Sleep", f"{deep} min")
    with fc5:
        spo2 = _fb_data.get("spo2_avg")
        st.metric("SpO2", f"{spo2:.1f}%" if spo2 else "---")

    # Second row: activity + trends
    fc6, fc7, fc8, fc9 = st.columns(4)
    with fc6:
        rhr = _fb_data.get("resting_hr", 0)
        st.metric("Resting HR", f"{rhr} bpm" if rhr else "---")
    with fc7:
        steps = _fb_data.get("steps", 0)
        st.metric("Steps", f"{steps:,}")
    with fc8:
        eff = _fb_data.get("sleep_efficiency")
        st.metric("Sleep Efficiency", f"{eff}%" if eff else "---")
    with fc9:
        if _fb_trends and "recovery_trend" in _fb_trends:
            trend = _fb_trends["recovery_trend"]
            delta = _fb_trends.get("recovery_delta", 0)
            st.metric("7-Day Trend", trend.capitalize(), delta=f"{delta:+.1f}")
        else:
            st.metric("7-Day Trend", "---")

    # Weekly chart
    try:
        _fb_week = get_recent_fitbit_days(n=7)
        if _fb_week and len(_fb_week) > 1:
            import pandas as _pd
            _fb_df = _pd.DataFrame(list(reversed(_fb_week)))
            _fb_df = _fb_df[_fb_df["date"] != "2026-01-01"]
            if not _fb_df.empty:
                st.subheader("7-Day Recovery + Sleep")
                chart_cols = ["date", "recovery_score", "deep_sleep_min", "hrv_rmssd"]
                chart_df = _fb_df[chart_cols].set_index("date")
                chart_df.columns = ["Recovery (0-10)", "Deep Sleep (min)", "HRV (ms)"]
                st.line_chart(chart_df)
    except Exception:
        pass

    st.divider()

# ── Polar H10 Live Metrics ───────────────────────────────────────────────────

if current_rhr or current_rmssd:
    st.subheader("Live (Polar H10)")

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
elif not _fb_data:
    st.warning("No data yet. Start a Polar session or sync Fitbit data.")
    st.stop()

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

# ── Probe Panel ──────────────────────────────────────────────────────────────

st.subheader("SOMA Probe")

try:
    _anomaly_conn = sqlite3.connect(DB_PATH)
    _pending = _anomaly_conn.execute(
        "SELECT id, detected_at, metric, value, deviation "
        "FROM anomalies WHERE acknowledged = 0 "
        "ORDER BY detected_at DESC LIMIT 1"
    ).fetchall()
    _anomaly_conn.close()

    if _pending:
        _a = _pending[0]
        st.warning(
            f"Anomaly detected: **{_a[2].upper()}** = {_a[3]} "
            f"({_a[4]:+.1f}s) at {str(_a[1])[11:16]}"
        )
        st.info("Run: `python -m soma.proto_self.probe_interface` in terminal")
    else:
        st.success("No pending anomalies.")
except Exception:
    st.info("Anomaly data unavailable.")

st.divider()

# ── Memory Timeline ──────────────────────────────────────────────────────────

st.subheader("Memory Timeline")

try:
    from soma.proto_self.autobiographical_store import get_recent_memories as _get_memories

    _memories = _get_memories(n=5)
    if not _memories:
        st.info("No autobiographical memories yet. Run a probe session to begin.")
    else:
        for _mem in _memories:
            _ts = str(_mem.get("timestamp", ""))[:16].replace("T", " ")
            _state = str(_mem.get("body_state", "")).upper()
            _metric = str(_mem.get("metric", "")).upper()
            _dev = _mem.get("deviation", 0)
            with st.expander(f"[{_ts}] {_state} — {_metric} {_dev:+.1f}s"):
                st.write(f"**SOMA asked:** {_mem.get('probe_text', '')}")
                st.write(f"**Response:** {_mem.get('response_text', '')}")
                _entities = _mem.get("entities", "[]")
                if isinstance(_entities, str):
                    _entities = json.loads(_entities)
                if _entities:
                    st.write(f"**Noted:** {', '.join(str(e) for e in _entities[:5])}")
                _val = _mem.get("emotion_valence", 0)
                _vlabel = "positive" if _val > 0.1 else "negative" if _val < -0.1 else "neutral"
                st.caption(f"Valence: {_vlabel} ({_val:+.2f})")
except Exception as _e:
    st.info(f"Memory timeline unavailable: {_e}")

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
