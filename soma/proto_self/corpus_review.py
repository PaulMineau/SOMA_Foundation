"""Corpus review — approve or reject staged research additions.

SOMA suggests. You decide. Nothing enters the live corpus without review.

Usage:
    python -m soma.proto_self.corpus_review
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from soma.proto_self.recommender import CORPUS_PATH
from soma.proto_self.research_agent import ADDITIONS_PATH


def load_staged() -> list[dict]:
    try:
        with open(ADDITIONS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def load_corpus(corpus_path: str | None = None) -> dict:
    path = corpus_path or CORPUS_PATH
    with open(path) as f:
        return json.load(f)


def save_corpus(corpus: dict, corpus_path: str | None = None) -> None:
    path = corpus_path or CORPUS_PATH
    with open(path, "w") as f:
        json.dump(corpus, f, indent=2)


def review(corpus_path: str | None = None) -> None:
    """Interactive review of staged research additions."""
    staged = load_staged()
    if not staged:
        print("No staged recommendations to review.")
        return

    corpus = load_corpus(corpus_path)
    approved: list[str] = []
    rejected: list[str] = []

    print(f"\nSOMA Research Review — {len(staged)} candidates\n")
    print("For each: [a]pprove, [r]eject, [s]kip for now, [q]uit\n")

    for entry in staged:
        print(f"{'─' * 60}")
        print(f"[{entry.get('type', '?').upper()}] {entry.get('title', '?')}")
        print(f"Why: {entry.get('why', '?')}")
        print(f"Best for: {', '.join(entry.get('best_states', []))}")
        print(f"Tags: {', '.join(entry.get('tags', []))}")
        print(f"Duration: ~{entry.get('duration_min', '?')} min")
        print(f"RAEN score: {entry.get('raen_total', '?')}")
        if entry.get("source"):
            print(f"Source: {entry['source']}")
        print()

        choice = input("[a/r/s/q]: ").strip().lower()

        if choice == "a":
            # Clean research metadata before merging
            clean_entry = {
                k: v for k, v in entry.items()
                if k not in ("raen", "raen_total", "recommended", "source", "research_date")
            }
            clean_entry["added_at"] = datetime.now().isoformat()
            corpus["entries"].append(clean_entry)
            approved.append(entry["title"])
            print(f"Added: {entry['title']}\n")

        elif choice == "r":
            rejected.append(entry["title"])
            print("Rejected\n")

        elif choice == "q":
            print("Quitting review. Remaining items stay staged.")
            break

        else:
            print("Skipped\n")

    # Save updated corpus
    save_corpus(corpus, corpus_path)

    # Clear approved and rejected from staging
    reviewed_titles = set(approved + rejected)
    remaining = [e for e in staged if e.get("title") not in reviewed_titles]

    os.makedirs(os.path.dirname(ADDITIONS_PATH) or ".", exist_ok=True)
    with open(ADDITIONS_PATH, "w") as f:
        json.dump(remaining, f, indent=2)

    print(f"\n{'─' * 60}")
    print(f"Review complete.")
    print(f"  Approved: {len(approved)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Still staged: {len(remaining)}")
    if approved:
        print(f"\nAdded to corpus:")
        for t in approved:
            print(f"  + {t}")


def main() -> None:
    review()


if __name__ == "__main__":
    main()
