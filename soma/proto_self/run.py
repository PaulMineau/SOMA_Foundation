"""CLI entrypoint for Polar H10 data collection."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from soma.proto_self.hrv import classify_body_state, compute_hrv
from soma.proto_self.polar_reader import HRSample, stream_hr
from soma.proto_self.storage import store_session_with_hrv


def _live_callback(sample: HRSample) -> None:
    """Print live data as it arrives."""
    rr_str = ", ".join(f"{rr:.0f}" for rr in sample.rr_intervals)
    contact = "ok" if sample.sensor_contact else "NO CONTACT"
    print(
        f"  HR: {sample.heart_rate:3d} bpm  "
        f"RR: [{rr_str}] ms  "
        f"({contact})"
    )


async def collect(
    duration: float,
    store: bool = True,
) -> None:
    """Run a Polar H10 collection session."""
    print(f"\nPolar H10 — collecting for {duration:.0f}s")
    print("=" * 50)

    session = await stream_hr(
        duration_seconds=duration,
        on_sample=_live_callback,
    )

    rr = session.all_rr_intervals
    print(f"\n{'=' * 50}")
    print(f"Session: {session.sample_count} samples, "
          f"{len(rr)} RR intervals, "
          f"{session.duration_seconds:.0f}s")

    if session.battery_level is not None:
        print(f"Battery: {session.battery_level}%")

    if len(rr) >= 3:
        metrics = compute_hrv(rr, window_seconds=session.duration_seconds)
        body_state = classify_body_state(metrics)

        print(f"\nHRV Metrics:")
        print(f"  RMSSD:  {metrics.rmssd:.1f} ms")
        print(f"  SDNN:   {metrics.sdnn:.1f} ms")
        print(f"  pNN50:  {metrics.pnn50:.1f}%")
        print(f"  Mean HR: {metrics.mean_hr:.0f} bpm")
        print(f"  Mean RR: {metrics.mean_rr:.0f} ms")
        print(f"  Clean intervals: {metrics.n_intervals} "
              f"({metrics.n_artifacts} artifacts rejected)")
        print(f"\n  Body State: {body_state.upper()}")
    else:
        print("\nInsufficient RR intervals for HRV computation")

    if store:
        n_stored, _ = store_session_with_hrv(session)
        print(f"\nStored {n_stored} RR records to LanceDB")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOMA Proto-Self — Polar H10 data collection"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Collection duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Don't write to LanceDB (dry run)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING for clean output)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    asyncio.run(collect(
        duration=args.duration,
        store=not args.no_store,
    ))


if __name__ == "__main__":
    main()
