"""Daily CPAP sync — cron script to pull yesterday's myAir data.

Runs at 6:30am (after Fitbit sync at 6am). Fetches previous night's
sleep record from myAir and stores in LanceDB.

Cron setup:
    30 6 * * * cd /path/to/SOMA_Foundation && .venv/bin/python -m soma.proto_self.cpap.daily_sync >> logs/cpap_sync.log 2>&1

Usage:
    python -m soma.proto_self.cpap.daily_sync              # last 7 days from myAir
    python -m soma.proto_self.cpap.daily_sync --days 30     # backfill 30 days
    python -m soma.proto_self.cpap.daily_sync --edf ~/SD/DATALOG  # parse SD card
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta


async def sync_myair(days: int = 7) -> int:
    """Fetch last N days from myAir."""
    from soma.proto_self.cpap.cpap_ingestor import ingest_myair_records
    from soma.proto_self.cpap.myair_client import MyAirClient

    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")

    client = MyAirClient()
    records = await client.get_sleep_records(start_month=start, end_month=end)

    print(f"  Fetched {len(records)} myAir records")
    added = ingest_myair_records(records)
    print(f"  Ingested {added} new records")
    return added


def sync_edf(datalog_path: str, last_n: int = 7) -> int:
    """Parse SD card EDF files for last N nights."""
    from soma.proto_self.cpap.cpap_ingestor import ingest_edf_summary
    from soma.proto_self.cpap.edf_parser import parse_datalog

    summaries = parse_datalog(datalog_path, last_n=last_n)
    print(f"  Parsed {len(summaries)} nights from EDF")

    for summary in summaries:
        ingest_edf_summary(summary)

    return len(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA CPAP daily sync")
    parser.add_argument("--days", type=int, default=7,
                        help="Days of myAir data to fetch")
    parser.add_argument("--edf", type=str, default=None,
                        help="Path to SD card DATALOG dir for richer EDF parsing")
    parser.add_argument("--no-myair", action="store_true",
                        help="Skip myAir poll (EDF only)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    os.makedirs("logs", exist_ok=True)

    print(f"\nSOMA CPAP Daily Sync — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not args.no_myair:
        print("  Polling myAir...")
        try:
            asyncio.run(sync_myair(days=args.days))
        except Exception as e:
            print(f"  myAir sync failed: {e}")

    if args.edf:
        print(f"  Parsing EDF from {args.edf}...")
        try:
            sync_edf(args.edf, last_n=args.days)
        except Exception as e:
            print(f"  EDF parse failed: {e}")

    print("  Done.\n")


if __name__ == "__main__":
    main()
