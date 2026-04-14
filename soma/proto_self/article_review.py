"""Article review — manage reading queue and newsletter discovery.

Usage:
    python -m soma.proto_self.article_review              # review all pending
    python -m soma.proto_self.article_review articles      # review articles only
    python -m soma.proto_self.article_review newsletters   # review discovered newsletters
    python -m soma.proto_self.article_review list           # show reading list
    python -m soma.proto_self.article_review list baseline  # filter by state
    python -m soma.proto_self.article_review read 2         # mark article #2 as read
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from soma.proto_self.substack_agent import NEWSLETTERS_PATH, QUEUE_PATH


def load_queue() -> list[dict]:
    try:
        with open(QUEUE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_queue(queue: list[dict]) -> None:
    os.makedirs(os.path.dirname(QUEUE_PATH) or ".", exist_ok=True)
    with open(QUEUE_PATH, "w") as f:
        json.dump(queue, f, indent=2)


def review_articles() -> None:
    """Review staged articles."""
    queue = load_queue()
    pending = [a for a in queue if not a.get("reviewed") and not a.get("auto_surfaced")]

    if not pending:
        print("No articles pending review.")
        return

    print(f"\nArticle Review — {len(pending)} pending\n")
    print("[a]pprove to reading list, [d]ismiss, [s]kip, [q]uit\n")

    for article in pending:
        print(f"{'─' * 60}")
        print(f"{article.get('title', '?')}")
        print(f"   {article.get('newsletter', '?')} — {article.get('author', '?')}")
        print(f"   {article.get('key_insight', article.get('why', ''))}")
        print(f"   Best for: {', '.join(article.get('best_states', []))}")
        print(f"   Read time: ~{article.get('duration_min', '?')} min")
        print(f"   RAEN: {article.get('raen_total', '?')}")
        print(f"   URL: {article.get('url', '?')}\n")

        choice = input("[a/d/s/q]: ").strip().lower()

        if choice == "a":
            article["reviewed"] = True
            article["approved"] = True
            article["reviewed_at"] = datetime.now().isoformat()
            print(f"Added to reading list\n")
        elif choice == "d":
            article["reviewed"] = True
            article["approved"] = False
            article["reviewed_at"] = datetime.now().isoformat()
            print(f"Dismissed\n")
        elif choice == "q":
            break
        else:
            print(f"Skipped\n")

    save_queue(queue)
    approved = sum(1 for a in queue if a.get("approved") and not a.get("read"))
    print(f"\nReading list: {approved} articles ready")


def review_newsletters() -> None:
    """Review discovered newsletters."""
    with open(NEWSLETTERS_PATH) as f:
        newsletters = json.load(f)

    pending = newsletters.get("discovered", [])
    if not pending:
        print("No new newsletters to review.")
        return

    print(f"\nNewsletter Discovery Review — {len(pending)} found\n")
    print("[f]ollow, [d]ismiss, [s]kip, [q]uit\n")

    followed = newsletters["followed"]
    dismissed = newsletters.get("dismissed", [])
    remaining: list[dict] = []

    for nl in pending:
        print(f"{'─' * 60}")
        print(f"{nl.get('name', '?')} by {nl.get('author', '?')}")
        print(f"   {nl.get('description', '')}")
        print(f"   Why: {nl.get('why_patient', '')}")
        print(f"   URL: {nl.get('url', '')}\n")

        choice = input("[f/d/s/q]: ").strip().lower()

        if choice == "f":
            clean = {k: v for k, v in nl.items()
                     if k not in ("discovered_at", "why_patient")}
            clean["notes"] = nl.get("why_patient", "")
            clean["id"] = f"nl_{len(followed) + 1:03d}"
            clean["best_states"] = nl.get("best_states", ["baseline", "restored", "peak"])
            clean["avoid_states"] = nl.get("avoid_states", [])
            followed.append(clean)
            print(f"Now following {nl['name']}\n")
        elif choice == "d":
            dismissed.append(nl.get("name", ""))
            print("Dismissed\n")
        elif choice == "q":
            remaining.append(nl)
            break
        else:
            remaining.append(nl)
            print("Skipped\n")

    newsletters["followed"] = followed
    newsletters["dismissed"] = dismissed
    newsletters["discovered"] = remaining

    with open(NEWSLETTERS_PATH, "w") as f:
        json.dump(newsletters, f, indent=2)


def show_reading_list(state: str | None = None) -> None:
    """Show approved, unread articles. Optionally filter by state."""
    queue = load_queue()
    readable = [
        a for a in queue
        if (a.get("approved") or a.get("auto_surfaced"))
        and not a.get("read")
    ]

    if state:
        readable = [
            a for a in readable
            if state in a.get("best_states", [])
            and state not in a.get("avoid_states", [])
        ]
        readable.sort(key=lambda x: x.get("raen_total", 0), reverse=True)

    if not readable:
        print("Reading list empty." if not state else f"Nothing matched for state: {state}")
        return

    label = f"for {state}" if state else ""
    print(f"\nReading List {label} — {len(readable)} articles\n")

    for i, a in enumerate(readable, 1):
        auto = "+" if a.get("auto_surfaced") else " "
        print(f"  {i}.{auto} {a.get('title', '?')}")
        print(f"     {a.get('newsletter', '?')} — ~{a.get('duration_min', '?')} min")
        print(f"     {a.get('key_insight', a.get('why', ''))}")
        print(f"     {a.get('url', '')}\n")


def mark_read(index: int) -> None:
    """Mark an article as read by its index in the reading list."""
    queue = load_queue()
    readable = [
        a for a in queue
        if (a.get("approved") or a.get("auto_surfaced"))
        and not a.get("read")
    ]
    if index < 1 or index > len(readable):
        print("Invalid index")
        return

    target = readable[index - 1]
    for a in queue:
        if a.get("url") == target.get("url"):
            a["read"] = True
            a["read_at"] = datetime.now().isoformat()
    save_queue(queue)
    print(f"Marked as read: {target.get('title', '?')}")


def main() -> None:
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "articles":
            review_articles()
        elif cmd == "newsletters":
            review_newsletters()
        elif cmd == "list":
            state = sys.argv[2] if len(sys.argv) > 2 else None
            show_reading_list(state)
        elif cmd == "read":
            if len(sys.argv) > 2:
                mark_read(int(sys.argv[2]))
            else:
                print("Usage: article_review read <index>")
        else:
            print(f"Unknown command: {cmd}")
    else:
        review_articles()
        print()
        review_newsletters()


if __name__ == "__main__":
    main()
