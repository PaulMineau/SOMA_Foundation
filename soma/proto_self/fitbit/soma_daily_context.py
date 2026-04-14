"""
soma_daily_context.py — Generate SOMA Proto-Self context for LLM consumption.

Two modes:
  1. Daily cron (run after morning sync) — updates today's record, prints context
  2. Context generation — outputs a structured health summary ready to paste into Claude

Usage:
  # Run daily (add to crontab: 0 8 * * * python soma_daily_context.py)
  python soma_daily_context.py

  # Get last N days of context for LLM
  python soma_daily_context.py --context --days 7

  # Just show today's state
  python soma_daily_context.py --today
"""

import argparse
import os
from datetime import date, datetime, timedelta

import lancedb

SOMA_DB_PATH = os.path.expanduser("~/soma/soma.db")
TABLE_NAME = "proto_self_fitbit"


def load_recent_records(days: int = 7) -> list:
    if not os.path.exists(SOMA_DB_PATH):
        return []
    db = lancedb.connect(SOMA_DB_PATH)
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        results = (
            table.search()
            .where(f"date >= '{cutoff}'")
            .to_list()
        )
        return sorted(results, key=lambda x: x["date"])
    except Exception:
        return []


def format_llm_context(records: list) -> str:
    """
    Format Proto-Self records as structured LLM context.
    Designed to be prepended to Claude prompts for health-aware responses.
    """
    if not records:
        return "No Proto-Self physiological data available."

    lines = ["## SOMA Proto-Self Layer — Physiological Context\n"]
    lines.append(f"Period: {records[0]['date']} → {records[-1]['date']}")
    lines.append(f"Records: {len(records)} days\n")

    # Averages
    valid_hrv = [r["hrv_rmssd"] for r in records if r["hrv_rmssd"] > 0]
    valid_rhr = [r["resting_hr"] for r in records if r["resting_hr"] > 0]
    valid_sleep = [r["sleep_duration_min"] for r in records if r["sleep_duration_min"] > 0]
    valid_recovery = [r["recovery_score"] for r in records]

    if valid_hrv:
        lines.append(f"Avg HRV (RMSSD): {sum(valid_hrv)/len(valid_hrv):.1f} ms")
    if valid_rhr:
        lines.append(f"Avg Resting HR: {sum(valid_rhr)/len(valid_rhr):.1f} bpm")
    if valid_sleep:
        avg_sleep_min = sum(valid_sleep) / len(valid_sleep)
        lines.append(f"Avg Sleep: {avg_sleep_min/60:.1f} hours")
    if valid_recovery:
        lines.append(f"Avg Recovery Score: {sum(valid_recovery)/len(valid_recovery):.1f}/10\n")

    # Daily breakdown
    lines.append("### Daily Breakdown\n")
    for r in records:
        today_marker = " ← TODAY" if r["date"] == date.today().strftime("%Y-%m-%d") else ""
        sleep_hrs = r["sleep_duration_min"] // 60
        sleep_mins = r["sleep_duration_min"] % 60
        lines.append(
            f"**{r['date']}{today_marker}**  "
            f"Recovery: {r['recovery_score']}/10 | "
            f"HRV: {r['hrv_rmssd']:.0f}ms | "
            f"RHR: {r['resting_hr']} bpm | "
            f"Sleep: {sleep_hrs}h{sleep_mins}m "
            f"({r['deep_sleep_min']}m deep) | "
            f"AZM: {r['active_zone_minutes']}"
        )

    # Most recent narrative
    if records:
        latest = records[-1]
        lines.append(f"\n### Today's Proto-Self State\n{latest['narrative']}")

    return "\n".join(lines)


def trend_analysis(records: list) -> str:
    """Simple trend detection over the window."""
    if len(records) < 3:
        return ""

    lines = ["\n### Trends\n"]

    # HRV trend
    hrv = [r["hrv_rmssd"] for r in records if r["hrv_rmssd"] > 0]
    if len(hrv) >= 3:
        early_avg = sum(hrv[:len(hrv)//2]) / (len(hrv)//2)
        late_avg = sum(hrv[len(hrv)//2:]) / (len(hrv) - len(hrv)//2)
        delta = late_avg - early_avg
        direction = "↑ improving" if delta > 3 else "↓ declining" if delta < -3 else "→ stable"
        lines.append(f"HRV trend: {direction} ({delta:+.1f}ms over window)")

    # Recovery trend
    scores = [r["recovery_score"] for r in records]
    early = sum(scores[:len(scores)//2]) / (len(scores)//2)
    late = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
    delta = late - early
    direction = "↑ improving" if delta > 0.5 else "↓ declining" if delta < -0.5 else "→ stable"
    lines.append(f"Recovery trend: {direction} ({delta:+.1f} pts over window)")

    # Sleep trend
    sleep = [r["sleep_duration_min"] for r in records if r["sleep_duration_min"] > 0]
    if len(sleep) >= 3:
        early_avg = sum(sleep[:len(sleep)//2]) / (len(sleep)//2)
        late_avg = sum(sleep[len(sleep)//2:]) / (len(sleep) - len(sleep)//2)
        delta = (late_avg - early_avg) / 60
        direction = "↑ more sleep" if delta > 0.25 else "↓ less sleep" if delta < -0.25 else "→ stable"
        lines.append(f"Sleep trend: {direction} ({delta:+.1f}h over window)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", action="store_true", help="Print LLM-ready context")
    parser.add_argument("--days", type=int, default=7, help="Number of days to include")
    parser.add_argument("--today", action="store_true", help="Show today's record only")
    parser.add_argument("--ingest-first", action="store_true",
                        help="Run ingestor before generating context")
    args = parser.parse_args()

    if args.ingest_first:
        from soma_fitbit_ingestor import SomaFitbitIngestor
        ingestor = SomaFitbitIngestor()
        ingestor.ingest_day(date.today().strftime("%Y-%m-%d"))

    if args.today:
        records = load_recent_records(days=1)
    else:
        records = load_recent_records(days=args.days)

    if not records:
        print("❌ No records found. Run soma_fitbit_ingestor.py first.")
        return

    context = format_llm_context(records)
    trends = trend_analysis(records)
    print(context)
    print(trends)


if __name__ == "__main__":
    main()
