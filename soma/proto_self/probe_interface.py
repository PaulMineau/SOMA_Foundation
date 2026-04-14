"""Probe interface — CLI conversation terminal for SOMA.

Usage:
    python -m soma.proto_self.probe_interface              # probe latest anomaly
    python -m soma.proto_self.probe_interface test          # test with synthetic anomaly
    python -m soma.proto_self.probe_interface timeline      # view memory timeline
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime

from soma.proto_self.autobiographical_store import get_recent_memories
from soma.proto_self.db import DEFAULT_DB_PATH, get_connection
from soma.proto_self.probe_generator import generate_probe
from soma.proto_self.memory_writer import write_memory


def get_pending_anomalies(
    limit: int = 3,
    db_path: str | None = None,
) -> list[dict]:
    """Get unacknowledged anomalies to probe."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, detected_at, metric, value, baseline, deviation "
            "FROM anomalies WHERE acknowledged = 0 "
            "ORDER BY detected_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def acknowledge_anomaly(anomaly_id: int, db_path: str | None = None) -> None:
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE anomalies SET acknowledged = 1 WHERE id = ?",
        (anomaly_id,),
    )
    conn.commit()
    conn.close()


def get_current_session_label(db_path: str | None = None) -> str:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT label FROM sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row["label"] if row else "unlabeled"


async def run_probe_session(anomaly: dict | None = None) -> None:
    """Run a single probe/response/memory session."""
    print("\n" + "=" * 60)
    print("  SOMA — Conversational Probe")
    print("=" * 60)

    if anomaly is None:
        pending = get_pending_anomalies(limit=1)
        if not pending:
            print("\nNo pending anomalies to probe.")
            print("(Run anomaly_detector to generate anomaly events,")
            print(" or use 'test' mode: python -m soma.proto_self.probe_interface test)")
            return
        anomaly = pending[0]

    ts = anomaly.get("detected_at", "")[:16].replace("T", " at ")
    metric = anomaly.get("metric", "").upper()
    print(f"\n  Detected: {metric} anomaly at {ts}")
    print(f"  Value: {anomaly.get('value', 0)} ({anomaly.get('deviation', 0):+.1f}s from baseline)\n")
    print("  Generating probe...\n")

    probe_text, state_info, similar_memories = await generate_probe(anomaly)

    print("-" * 60)
    print(f"\n  SOMA:  {probe_text}\n")
    print("-" * 60)

    print("\n  (Type your response. Press Enter on empty line to submit.)")
    print("  (Type 'skip' to acknowledge without responding.)\n")

    lines: list[str] = []
    print("  You:  ", end="", flush=True)
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.lower().strip() == "skip":
            print("\n  Skipped. Anomaly acknowledged.")
            if anomaly.get("id"):
                acknowledge_anomaly(anomaly["id"])
            return
        if line == "" and lines:
            break
        if line:
            lines.append(line)
        if not lines:
            print("        ", end="", flush=True)

    response_text = " ".join(lines).strip()
    if not response_text:
        print("\n  No response recorded.")
        return

    print()

    session_label = get_current_session_label()
    memory_id, extracted = await write_memory(
        anomaly, state_info, probe_text, response_text, session_label
    )

    if anomaly.get("id"):
        acknowledge_anomaly(anomaly["id"])

    print("  Exchange stored as autobiographical memory.")

    entities = extracted.get("entities", [])
    valence = extracted.get("emotion_valence", 0.0)

    if valence < -0.3:
        closing = "That sounds hard. I'll remember this."
    elif valence > 0.3:
        closing = "Good to know. I'll carry this forward."
    else:
        closing = "Got it. This goes into the record."

    if entities:
        notable = [e for e in entities if len(str(e)) > 2][:2]
        if notable:
            closing += f" (noted: {', '.join(str(e) for e in notable)})"

    print(f"\n  SOMA:  {closing}\n")
    print("=" * 60 + "\n")


async def show_memory_timeline(n: int = 10) -> None:
    """Display recent autobiographical memories."""
    memories = get_recent_memories(n=n)
    if not memories:
        print("\nNo autobiographical memories yet.")
        print("Run a probe session to begin building memory.")
        return

    print(f"\nSOMA Memory Timeline — last {len(memories)} exchanges\n")
    print("-" * 60)

    for mem in memories:
        ts = mem.get("timestamp", "")[:16].replace("T", " ")
        state = mem.get("body_state", "").upper()
        metric = mem.get("metric", "").upper()
        dev = mem.get("deviation", 0)
        probe = str(mem.get("probe_text", ""))[:80]
        response = str(mem.get("response_text", ""))[:80]
        entities = json.loads(mem.get("entities", "[]")) if isinstance(mem.get("entities"), str) else mem.get("entities", [])

        print(f"\n  [{ts}] {state} — {metric} {dev:+.1f}s")
        if entities:
            print(f"  Entities: {', '.join(str(e) for e in entities[:4])}")
        print(f"  SOMA:  {probe}...")
        print(f"  Response:  {response}...")

    print("\n" + "-" * 60)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    if len(sys.argv) > 1 and sys.argv[1] == "timeline":
        asyncio.run(show_memory_timeline())
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_anomaly = {
            "id": None,
            "metric": "rmssd",
            "value": 18.3,
            "baseline": 41.0,
            "deviation": -2.3,
            "detected_at": datetime.now().isoformat(),
        }
        asyncio.run(run_probe_session(test_anomaly))
    else:
        asyncio.run(run_probe_session())


if __name__ == "__main__":
    main()
