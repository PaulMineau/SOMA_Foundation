"""Live terminal monitor — tails RR intervals from SQLite.

Run in a second terminal while polar_logger.py is recording.

Usage:
    python -m soma.proto_self.monitor
    python -m soma.proto_self.monitor --rows 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from soma.proto_self.db import DEFAULT_DB_PATH


def tail_rr(db_path: str, n: int = 10, refresh: float = 2.0) -> None:
    """Continuously display the most recent RR intervals."""
    import sqlite3

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("Start a session first: python -m soma.proto_self.polar_logger")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("SOMA-Cardio Live Monitor")
    print(f"  DB: {db_path}")
    print(f"  Refreshing every {refresh:.0f}s, showing last {n} intervals")
    print("  Press Ctrl+C to stop\n")

    try:
        while True:
            rows = conn.execute(
                "SELECT r.timestamp, r.rr_ms, r.hr_bpm, s.label "
                "FROM rr_intervals r "
                "JOIN sessions s ON r.session_id = s.session_id "
                "ORDER BY r.id DESC LIMIT ?",
                (n,),
            ).fetchall()

            # Get current session info
            session = conn.execute(
                "SELECT session_id, label, started_at, n_intervals "
                "FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()

            # Clear screen
            print("\033c", end="")
            print("SOMA-Cardio Live Monitor\n")

            if session:
                print(f"  Session: {session['session_id']}  Label: {session['label']}")
                print()

            if not rows:
                print("  No data yet — waiting for RR intervals...\n")
            else:
                # Compute rolling stats from visible rows
                rr_values = [row["rr_ms"] for row in reversed(rows)]
                avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0
                avg_hr = 60000 / avg_rr if avg_rr > 0 else 0

                print(f"  Avg HR: {avg_hr:.0f} bpm  |  Avg RR: {avg_rr:.0f} ms  |  Showing last {len(rows)}\n")

                for row in reversed(rows):
                    ts = row["timestamp"]
                    rr = row["rr_ms"]
                    hr = row["hr_bpm"]
                    label = row["label"] or ""
                    hr_est = 60000 / rr if rr > 0 else 0
                    print(f"  {ts[11:19]}  |  RR: {rr:.0f}ms  |  HR: {hr} bpm  |  {label}")

            print(f"\n  (refreshing every {refresh:.0f}s — Ctrl+C to stop)")
            time.sleep(refresh)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA-Cardio live monitor")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to show")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval (seconds)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    args = parser.parse_args()

    tail_rr(args.db, n=args.rows, refresh=args.refresh)


if __name__ == "__main__":
    main()
