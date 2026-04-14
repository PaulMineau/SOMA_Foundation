"""
soma_fitbit_ingestor.py — Ingest Fitbit data into SOMA's Proto-Self layer.

Maps physiological signals to Damasio's L1 (Proto-Self) layer in LanceDB.
Each day becomes a vector-searchable record with a narrative embedding,
enabling queries like "days when recovery was high" or "nights I slept poorly."

Architecture:
  Fitbit API → FitbitClient → proto_self_record → LanceDB (soma_proto_self table)

Usage:
  # Ingest today
  python soma_fitbit_ingestor.py

  # Backfill a date range
  python soma_fitbit_ingestor.py --start 2026-01-01 --end 2026-04-14

  # Ingest specific date
  python soma_fitbit_ingestor.py --date 2026-04-13

Dependencies:
  pip install lancedb sentence-transformers requests requests-oauthlib
"""

import argparse
import json
import os
from datetime import date, datetime, timedelta
from typing import Optional

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

from fitbit_client import FitbitClient

# ── Config ────────────────────────────────────────────────────────────────────

SOMA_DB_PATH = os.path.expanduser("~/soma/soma.db")
TABLE_NAME = "proto_self_fitbit"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good for health narratives


# ── Narrative generation ──────────────────────────────────────────────────────

def build_narrative(record: dict) -> str:
    """
    Convert a daily Fitbit summary into a natural language narrative.
    This is what gets embedded — it should be rich enough to support
    semantic queries like 'high stress day' or 'excellent recovery'.
    """
    parts = []

    # Sleep quality
    sleep_min = record.get("sleep_duration_min", 0)
    sleep_hrs = sleep_min / 60
    deep_min = record.get("deep_sleep_min", 0)
    rem_min = record.get("rem_sleep_min", 0)
    efficiency = record.get("sleep_efficiency")
    eff_str = f"{efficiency}% efficiency" if efficiency else "unknown efficiency"

    parts.append(
        f"Sleep: {sleep_hrs:.1f} hours with {deep_min} minutes deep sleep "
        f"and {rem_min} minutes REM, {eff_str}."
    )

    # Classify sleep quality
    if deep_min >= 90 and sleep_hrs >= 7:
        parts.append("Sleep quality was excellent — deep and restorative.")
    elif deep_min >= 60 and sleep_hrs >= 6:
        parts.append("Sleep quality was moderate.")
    elif deep_min < 45 or sleep_hrs < 5.5:
        parts.append("Sleep was poor or fragmented — recovery may be compromised.")

    # Heart rate & HRV (key recovery markers)
    rhr = record.get("resting_hr")
    hrv = record.get("hrv_rmssd")

    if rhr:
        parts.append(f"Resting heart rate: {rhr} bpm.")
        if rhr <= 58:
            parts.append("Resting HR indicates excellent cardiovascular recovery.")
        elif rhr >= 70:
            parts.append("Elevated resting HR — possible fatigue or stress load.")

    if hrv:
        parts.append(f"HRV (RMSSD): {hrv:.1f} ms.")
        if hrv >= 50:
            parts.append("High HRV indicates strong autonomic recovery and low stress.")
        elif hrv >= 30:
            parts.append("HRV in moderate range — adequate but not peak recovery.")
        elif hrv < 20:
            parts.append("Low HRV suggests high stress load or incomplete recovery.")

    # Activity
    steps = record.get("steps", 0)
    azm = record.get("active_zone_minutes", 0)
    very_active = record.get("very_active_min", 0)

    parts.append(f"Activity: {steps:,} steps, {azm} active zone minutes ({very_active} vigorous).")

    if azm >= 40:
        parts.append("High activity day — significant cardiovascular stimulus.")
    elif azm >= 20:
        parts.append("Moderate activity — met movement goals.")
    elif steps < 4000:
        parts.append("Low movement day — mostly sedentary.")

    # SpO2
    spo2 = record.get("spo2_avg")
    if spo2:
        parts.append(f"Overnight SpO2 average: {spo2:.1f}%.")
        if spo2 < 93:
            parts.append("SpO2 below threshold — possible sleep-disordered breathing event.")

    # Overall physiological state classification
    recovery_score = _compute_recovery_score(record)
    parts.append(f"Estimated recovery score: {recovery_score}/10.")

    if recovery_score >= 8:
        parts.append("Proto-self state: high-readiness. Autonomic balance favorable.")
    elif recovery_score >= 5:
        parts.append("Proto-self state: moderate readiness. Normal baseline.")
    else:
        parts.append("Proto-self state: depleted. System under load — conserve resources.")

    return " ".join(parts)


def _compute_recovery_score(record: dict) -> float:
    """
    Heuristic recovery score 0-10 from physiological markers.
    Inspired by Whoop/Oura scoring logic but simplified.
    """
    score = 5.0  # baseline

    # HRV contribution (±2 points)
    hrv = record.get("hrv_rmssd")
    if hrv:
        if hrv >= 60:
            score += 2
        elif hrv >= 45:
            score += 1
        elif hrv < 25:
            score -= 2
        elif hrv < 35:
            score -= 1

    # Resting HR contribution (±2 points)
    rhr = record.get("resting_hr")
    if rhr:
        if rhr <= 55:
            score += 2
        elif rhr <= 62:
            score += 1
        elif rhr >= 72:
            score -= 2
        elif rhr >= 68:
            score -= 1

    # Sleep duration contribution (±1.5 points)
    sleep_hrs = record.get("sleep_duration_min", 0) / 60
    if sleep_hrs >= 7.5:
        score += 1.5
    elif sleep_hrs >= 6.5:
        score += 0.5
    elif sleep_hrs < 5.5:
        score -= 1.5
    elif sleep_hrs < 6.0:
        score -= 0.5

    # Deep sleep contribution (±1 point)
    deep = record.get("deep_sleep_min", 0)
    if deep >= 90:
        score += 1
    elif deep < 45:
        score -= 1

    # SpO2 contribution (±1 point)
    spo2 = record.get("spo2_avg")
    if spo2:
        if spo2 >= 96:
            score += 0.5
        elif spo2 < 93:
            score -= 1.0

    return round(max(0, min(10, score)), 1)


# ── LanceDB schema & storage ──────────────────────────────────────────────────

def get_or_create_table(db, embedding_dim: int):
    """
    Create the proto_self_fitbit table if it doesn't exist.
    LanceDB is schema-on-write so we define it via a sample record.
    """
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)

    sample = {
        "date": "2026-01-01",
        "resting_hr": 0,
        "hrv_rmssd": 0.0,
        "hrv_coverage": 0.0,
        "sleep_duration_min": 0,
        "sleep_efficiency": 0,
        "sleep_start": "",
        "sleep_end": "",
        "deep_sleep_min": 0,
        "light_sleep_min": 0,
        "rem_sleep_min": 0,
        "wake_min": 0,
        "steps": 0,
        "calories": 0,
        "active_zone_minutes": 0,
        "very_active_min": 0,
        "spo2_avg": 0.0,
        "spo2_min": 0.0,
        "recovery_score": 0.0,
        "damasio_layer": "L1_proto_self",
        "narrative": "",
        "vector": np.zeros(embedding_dim, dtype=np.float32).tolist(),
        "ingested_at": "",
    }

    return db.create_table(TABLE_NAME, data=[sample])


def record_to_lancedb_row(record: dict, narrative: str, vector: list) -> dict:
    """Flatten a daily Fitbit summary into a LanceDB row."""
    return {
        "date": record["date"],
        "resting_hr": int(record.get("resting_hr") or 0),
        "hrv_rmssd": float(record.get("hrv_rmssd") or 0.0),
        "hrv_coverage": float(record.get("hrv_coverage") or 0.0),
        "sleep_duration_min": int(record.get("sleep_duration_min") or 0),
        "sleep_efficiency": int(record.get("sleep_efficiency") or 0),
        "sleep_start": str(record.get("sleep_start") or ""),
        "sleep_end": str(record.get("sleep_end") or ""),
        "deep_sleep_min": int(record.get("deep_sleep_min") or 0),
        "light_sleep_min": int(record.get("light_sleep_min") or 0),
        "rem_sleep_min": int(record.get("rem_sleep_min") or 0),
        "wake_min": int(record.get("wake_min") or 0),
        "steps": int(record.get("steps") or 0),
        "calories": int(record.get("calories") or 0),
        "active_zone_minutes": int(record.get("active_zone_minutes") or 0),
        "very_active_min": int(record.get("very_active_min") or 0),
        "spo2_avg": float(record.get("spo2_avg") or 0.0),
        "spo2_min": float(record.get("spo2_min") or 0.0),
        "recovery_score": _compute_recovery_score(record),
        "damasio_layer": "L1_proto_self",
        "narrative": narrative,
        "vector": vector,
        "ingested_at": datetime.utcnow().isoformat(),
    }


# ── Dedup check ───────────────────────────────────────────────────────────────

def date_already_ingested(table, target_date: str) -> bool:
    try:
        results = table.search().where(f"date = '{target_date}'").limit(1).to_list()
        return len(results) > 0
    except Exception:
        return False


# ── Main ingest pipeline ──────────────────────────────────────────────────────

class SomaFitbitIngestor:
    def __init__(self):
        print("🧠 SOMA Fitbit Ingestor initializing...")
        self.fitbit = FitbitClient()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        embedding_dim = self.model.get_sentence_embedding_dimension()

        os.makedirs(os.path.dirname(SOMA_DB_PATH), exist_ok=True)
        self.db = lancedb.connect(SOMA_DB_PATH)
        self.table = get_or_create_table(self.db, embedding_dim)
        print(f"  📦 LanceDB connected: {SOMA_DB_PATH} / {TABLE_NAME}")
        print(f"  🤖 Embedding model: {EMBEDDING_MODEL} ({embedding_dim}d)")

    def ingest_day(self, target_date: str, force: bool = False) -> bool:
        """
        Ingest a single day. Returns True if ingested, False if skipped.
        """
        if not force and date_already_ingested(self.table, target_date):
            print(f"  ⏭️  {target_date} already in SOMA — skipping (use --force to overwrite)")
            return False

        record = self.fitbit.get_daily_summary(target_date)
        narrative = build_narrative(record)
        vector = self.model.encode(narrative).tolist()
        row = record_to_lancedb_row(record, narrative, vector)

        if force:
            # Delete existing record for this date before re-inserting
            try:
                self.table.delete(f"date = '{target_date}'")
            except Exception:
                pass

        self.table.add([row])

        recovery = row["recovery_score"]
        print(f"  ✅ {target_date} — Recovery: {recovery}/10 | HRV: {row['hrv_rmssd']} ms | "
              f"RHR: {row['resting_hr']} bpm | Sleep: {row['sleep_duration_min']//60}h{row['sleep_duration_min']%60}m")
        return True

    def ingest_range(self, start: str, end: str = None, force: bool = False):
        """Backfill a date range."""
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.today()

        total = (end_dt - start_dt).days + 1
        ingested = 0
        skipped = 0

        print(f"\n📅 Ingesting {total} days from {start} to {end_dt.strftime('%Y-%m-%d')}...\n")

        current = start_dt
        while current <= end_dt:
            d = current.strftime("%Y-%m-%d")
            try:
                result = self.ingest_day(d, force=force)
                if result:
                    ingested += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  ❌ {d}: {e}")
            current += timedelta(days=1)

        print(f"\n🏁 Done. {ingested} ingested, {skipped} skipped.")
        self._print_table_stats()

    def query(self, query_text: str, limit: int = 5) -> list:
        """
        Semantic search over Proto-Self records.
        e.g.: ingestor.query("high stress, poor sleep")
        """
        vector = self.model.encode(query_text).tolist()
        results = (
            self.table.search(vector)
            .limit(limit)
            .to_list()
        )
        return results

    def _print_table_stats(self):
        count = self.table.count_rows()
        print(f"\n📊 SOMA Proto-Self table: {count} records total")

        # Most recent 5
        try:
            recent = (
                self.table.search()
                .select(["date", "recovery_score", "hrv_rmssd", "resting_hr", "sleep_duration_min"])
                .limit(5)
                .to_list()
            )
            # Sort by date descending
            recent = sorted(recent, key=lambda x: x["date"], reverse=True)
            print("\n  Recent records:")
            for r in recent:
                hrs = r["sleep_duration_min"] // 60
                mins = r["sleep_duration_min"] % 60
                print(f"    {r['date']} | Recovery: {r['recovery_score']}/10 | "
                      f"HRV: {r['hrv_rmssd']:.0f}ms | RHR: {r['resting_hr']} | Sleep: {hrs}h{mins}m")
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOMA Fitbit Ingestor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", help="Ingest specific date (YYYY-MM-DD)")
    group.add_argument("--start", help="Start of date range (YYYY-MM-DD)")
    group.add_argument("--today", action="store_true", help="Ingest today only")
    parser.add_argument("--end", help="End of date range (YYYY-MM-DD), defaults to today")
    parser.add_argument("--force", action="store_true", help="Overwrite existing records")
    parser.add_argument("--query", help="Semantic search test query")
    parser.add_argument("--stats", action="store_true", help="Show table stats")

    args = parser.parse_args()
    ingestor = SomaFitbitIngestor()

    if args.query:
        print(f"\n🔍 Searching: '{args.query}'")
        results = ingestor.query(args.query)
        for r in results:
            print(f"  {r['date']} — Recovery {r['recovery_score']}/10")
            print(f"    {r['narrative'][:200]}...")
        return

    if args.stats:
        ingestor._print_table_stats()
        return

    if args.date:
        ingestor.ingest_day(args.date, force=args.force)
    elif args.start:
        ingestor.ingest_range(args.start, args.end, force=args.force)
    elif args.today:
        ingestor.ingest_day(date.today().strftime("%Y-%m-%d"), force=args.force)
    else:
        # Default: ingest today
        print(f"\n🌅 Ingesting today ({date.today().strftime('%Y-%m-%d')})...")
        ingestor.ingest_day(date.today().strftime("%Y-%m-%d"), force=args.force)
        ingestor._print_table_stats()


if __name__ == "__main__":
    main()
