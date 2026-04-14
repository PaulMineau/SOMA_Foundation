"""Feedback logger — close the loop after a recommendation.

Interactive CLI to record whether you followed SOMA's recommendation
and how it affected your state.

Usage:
    python -m soma.proto_self.feedback_logger
"""

from __future__ import annotations

import sys

from soma.proto_self.recommender import get_pending_recommendations, log_feedback
from soma.proto_self.state_classifier import classify_state


def interactive_feedback() -> None:
    """Interactive feedback session."""
    print("\nSOMA Feedback Loop\n")

    pending = get_pending_recommendations()
    if not pending:
        print("No pending recommendations to give feedback on.")
        return

    print("Recent recommendations awaiting feedback:\n")
    for row in pending:
        print(f"  [{row['id']}] {str(row['recommended_at'])[11:16]} — {row['title']}")

    print()
    rec_id_str = input("Enter recommendation ID to give feedback on: ").strip()
    try:
        rec_id = int(rec_id_str)
    except ValueError:
        print("Invalid ID.")
        return

    followed_str = input("Did you follow this recommendation? (y/n): ").strip().lower()
    followed = 1 if followed_str == "y" else 0

    if followed:
        print("\nHow did you feel afterward?")
        print("  1. Better (HRV improved, felt restored)")
        print("  2. Same (no noticeable change)")
        print("  3. Worse (more depleted)")
        choice = input("Choice (1/2/3): ").strip()
        outcome_map = {"1": "better", "2": "same", "3": "worse"}
        outcome = outcome_map.get(choice, "same")
    else:
        outcome = "skipped"

    state_after = log_feedback(rec_id, followed, outcome)

    print(f"\nFeedback logged.")
    print(f"Current state: {state_after['state'].upper()}")
    if state_after["rmssd"] is not None:
        print(f"   RMSSD: {state_after['rmssd']} ms | RHR: {state_after['rhr']} bpm")


def main() -> None:
    interactive_feedback()


if __name__ == "__main__":
    main()
