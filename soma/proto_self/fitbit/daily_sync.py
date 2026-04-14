"""Daily Fitbit sync — run via cron at 6am to ingest yesterday's data.

Fitbit finalizes overnight sleep/HRV data by early morning, so we ingest
the previous day's complete record at 6am.

Cron setup:
    crontab -e
    0 6 * * * cd /Users/paulmineau/git/SOMA/SOMA_Foundation && /Users/paulmineau/git/SOMA/SOMA_Foundation/.venv/bin/python -m soma.proto_self.fitbit.daily_sync >> logs/fitbit_sync.log 2>&1

Usage:
    python -m soma.proto_self.fitbit.daily_sync              # ingest yesterday
    python -m soma.proto_self.fitbit.daily_sync --today       # ingest today instead
    python -m soma.proto_self.fitbit.daily_sync --days 3      # backfill last 3 days
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)


def sync_day(target_date: str, force: bool = False) -> bool:
    """Ingest a single day from Fitbit into SOMA LanceDB."""
    from soma.proto_self.fitbit.soma_fitbit_ingestor import SomaFitbitIngestor

    ingestor = SomaFitbitIngestor()
    return ingestor.ingest_day(target_date, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOMA Fitbit daily sync — ingest yesterday's complete data"
    )
    parser.add_argument("--today", action="store_true",
                        help="Ingest today instead of yesterday")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to backfill (default: 1 = yesterday only)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing records")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    os.makedirs("logs", exist_ok=True)

    print(f"\nSOMA Fitbit Daily Sync — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if args.today:
        target = date.today().strftime("%Y-%m-%d")
        print(f"  Syncing today: {target}")
        sync_day(target, force=args.force)
    else:
        for i in range(args.days, 0, -1):
            target = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            print(f"  Syncing: {target}")
            try:
                sync_day(target, force=args.force)
            except Exception as e:
                print(f"  Failed for {target}: {e}")

    print("  Done.\n")


if __name__ == "__main__":
    main()
