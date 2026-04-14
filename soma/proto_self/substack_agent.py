"""Substack agent — daily newsletter fetch, classify, score pipeline.

Fetches recent posts from followed newsletters, classifies with Claude via
OpenRouter, scores with RAEN, and stages for review.

Usage:
    python -m soma.proto_self.substack_agent
    python -m soma.proto_self.substack_agent --discover-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any

import feedparser
import httpx
from dotenv import load_dotenv

from soma.proto_self.raen_scorer import score_candidates
from soma.proto_self.soma_profile import build_profile

logger = logging.getLogger(__name__)

load_dotenv(".env.local")
load_dotenv(".env")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("SOMA_RESEARCH_MODEL", "anthropic/claude-sonnet-4")
NEWSLETTERS_PATH = os.environ.get("SOMA_NEWSLETTERS", "data/newsletters.json")
QUEUE_PATH = os.environ.get("SOMA_ARTICLE_QUEUE", "data/article_queue.json")

AUTO_SURFACE_THRESHOLD = 0.85
STAGE_THRESHOLD = 0.55
MAX_ARTICLE_AGE_DAYS = 3
MAX_ARTICLES_PER_RUN = 10


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return key


def load_newsletters() -> dict:
    with open(NEWSLETTERS_PATH) as f:
        return json.load(f)


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


def _is_recent(published_str: str, max_days: int = MAX_ARTICLE_AGE_DAYS) -> bool:
    """Check if article was published within the window."""
    try:
        from email.utils import parsedate_to_datetime
        pub_date = parsedate_to_datetime(published_str)
        return (datetime.now(pub_date.tzinfo) - pub_date).days <= max_days
    except Exception:
        return True


def fetch_rss_entries(newsletter: dict) -> list[dict]:
    """Parse RSS feed and return recent entries."""
    rss_url = newsletter.get("rss")
    if not rss_url:
        return []

    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        logger.warning("RSS fetch failed for %s: %s", newsletter["name"], e)
        return []

    recent: list[dict] = []
    for entry in feed.entries[:5]:
        published = entry.get("published", "")
        if not _is_recent(published):
            continue
        recent.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "summary": entry.get("summary", "")[:500],
            "published": published,
            "newsletter_id": newsletter["id"],
            "newsletter_name": newsletter["name"],
            "author": newsletter.get("author", ""),
            "newsletter_tags": newsletter.get("tags", []),
            "newsletter_best_states": newsletter.get("best_states", []),
            "newsletter_avoid_states": newsletter.get("avoid_states", []),
            "typical_length_min": newsletter.get("typical_length_min", 10),
        })

    return recent


async def _llm_call(prompt: str, max_tokens: int = 1000) -> str:
    """Make an LLM call via OpenRouter."""
    api_key = _get_api_key()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


async def classify_article(entry: dict, profile: dict) -> dict | None:
    """Ask Claude to classify and score an article."""
    state = profile["current_state"]["state"]
    content = entry.get("summary", "No content available")

    prompt = f"""Analyze this newsletter article for a patient profile.

## Profile (brief)
- Age 50, data scientist, Tibetan Buddhist practitioner
- Interests: causal inference, consciousness, parenting, longevity, AI
- Values: upstream causation, compound over deplete, compassion
- Current state: {state.upper()}
- Avoids: glorification of substances, outrage loops, hustle culture

## Article
Title: {entry['title']}
Author: {entry['author']}
Newsletter: {entry['newsletter_name']}

Content preview:
{content}

Return ONLY a JSON object:
{{"title": "cleaned title", "why": "1-2 sentences why this fits", "tags": ["tag1", "tag2"], "best_states": ["baseline", "restored"], "avoid_states": ["depleted"], "estimated_read_min": 10, "key_insight": "one sentence core idea", "fits_patient": true, "paywall": false}}"""

    try:
        raw = await _llm_call(prompt)
        clean = re.sub(r"```(?:json)?\s*\n?", "", raw.strip())
        clean = re.sub(r"\n?```\s*$", "", clean)
        classification = json.loads(clean)
        return {**entry, **classification}
    except Exception as e:
        logger.warning("Classification failed for %s: %s", entry["title"], e)
        return None


async def discover_newsletters(seed: dict, profile: dict) -> list[dict]:
    """Use Claude to find new newsletter writers matching a discovery seed."""
    newsletters = load_newsletters()
    known = [n["name"] for n in newsletters["followed"]]
    dismissed = newsletters.get("dismissed", [])

    prompt = f"""Find 3 active Substack newsletter writers on: {seed['topic']}

Keywords: {', '.join(seed['keywords'])}
Exclude: {', '.join(seed['exclude_keywords']) if seed['exclude_keywords'] else 'nothing specific'}

For someone who:
- Studies causal inference and AI consciousness
- Practices Tibetan Buddhism
- Has an infant son, values parenting science
- Prefers intellectual depth over hot takes

Already follows: {', '.join(known)}
Previously dismissed: {', '.join(dismissed)}

Return ONLY a JSON array of objects:
[{{"name": "Newsletter Name", "author": "Author", "url": "https://example.substack.com", "rss": "https://example.substack.com/feed", "description": "one sentence", "why_patient": "why this fits", "tags": ["tag1"], "typical_length_min": 10}}]"""

    try:
        raw = await _llm_call(prompt, max_tokens=2000)
        clean = re.sub(r"```(?:json)?\s*\n?", "", raw.strip())
        clean = re.sub(r"\n?```\s*$", "", clean)
        return json.loads(clean)
    except Exception as e:
        logger.warning("Discovery failed for %s: %s", seed["topic"], e)
        return []


async def run_substack_agent(discover_only: bool = False) -> dict[str, int]:
    """Run the full Substack agent pipeline. Returns summary stats."""
    print(f"\nSOMA Substack Agent")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    profile = build_profile()
    newsletters = load_newsletters()
    existing_queue = load_queue()
    existing_urls = {a["url"] for a in existing_queue}

    staged: list[dict] = []
    auto_surfaced: list[dict] = []
    processed = 0

    # 1. Check followed newsletters for new posts
    if not discover_only:
        print("Checking followed newsletters...\n")
        for newsletter in newsletters["followed"]:
            entries = fetch_rss_entries(newsletter)
            for entry in entries:
                if entry["url"] in existing_urls:
                    continue
                if processed >= MAX_ARTICLES_PER_RUN:
                    break

                print(f"  -> {entry['title'][:60]}")
                classified = await classify_article(entry, profile)

                if not classified or not classified.get("fits_patient"):
                    print(f"     x Doesn't fit, skipping")
                    continue

                if classified.get("paywall"):
                    print(f"     Paywalled, skipping")
                    continue

                raen_entry = {
                    "id": f"art_{datetime.now().strftime('%Y%m%d')}_{processed}",
                    "type": "article",
                    "title": classified.get("title", entry["title"]),
                    "why": classified.get("why", ""),
                    "key_insight": classified.get("key_insight", ""),
                    "url": entry["url"],
                    "author": entry["author"],
                    "newsletter": entry["newsletter_name"],
                    "tags": classified.get("tags", entry["newsletter_tags"]),
                    "best_states": classified.get("best_states", entry["newsletter_best_states"]),
                    "avoid_states": classified.get("avoid_states", entry["newsletter_avoid_states"]),
                    "duration_min": classified.get("estimated_read_min", entry["typical_length_min"]),
                    "fetched_at": datetime.now().isoformat(),
                    "published": entry["published"],
                }

                scored_list = score_candidates([raen_entry], profile)
                scored = scored_list[0] if scored_list else raen_entry
                raen_score = scored.get("raen_total", 0)

                print(f"     RAEN: {raen_score}")

                if raen_score >= AUTO_SURFACE_THRESHOLD:
                    scored["auto_surfaced"] = True
                    auto_surfaced.append(scored)
                    print(f"     Auto-surfaced")
                elif raen_score >= STAGE_THRESHOLD:
                    scored["auto_surfaced"] = False
                    staged.append(scored)
                    print(f"     Staged for review")
                else:
                    print(f"     Below threshold")

                processed += 1

    # 2. Newsletter discovery (one seed topic per day)
    print("\nDiscovering new newsletters...\n")
    day_index = datetime.now().timetuple().tm_yday
    seed = newsletters["discovery_seeds"][day_index % len(newsletters["discovery_seeds"])]
    print(f"  Topic: {seed['topic']}\n")

    discovered = await discover_newsletters(seed, profile)
    new_discoveries: list[dict] = []
    known_names = {n["name"] for n in newsletters["followed"]}
    dismissed_names = set(newsletters.get("dismissed", []))

    for nl in discovered:
        if isinstance(nl, dict) and nl.get("name") not in known_names and nl.get("name") not in dismissed_names:
            nl["discovered_at"] = datetime.now().isoformat()
            nl["best_states"] = ["baseline", "restored", "peak"]
            nl["avoid_states"] = []
            new_discoveries.append(nl)
            print(f"  + {nl.get('name', '?')} by {nl.get('author', '?')}")
            print(f"    {nl.get('why_patient', nl.get('description', ''))}\n")

    if new_discoveries:
        newsletters["discovered"] = newsletters.get("discovered", []) + new_discoveries
        with open(NEWSLETTERS_PATH, "w") as f:
            json.dump(newsletters, f, indent=2)

    # 3. Update queue
    all_new = auto_surfaced + staged
    updated_queue = existing_queue + all_new
    save_queue(updated_queue)

    stats = {
        "processed": processed,
        "auto_surfaced": len(auto_surfaced),
        "staged": len(staged),
        "discovered": len(new_discoveries),
        "queue_total": len(updated_queue),
    }

    print(f"\nSubstack agent complete")
    print(f"   Articles processed: {stats['processed']}")
    print(f"   Auto-surfaced: {stats['auto_surfaced']}")
    print(f"   Staged for review: {stats['staged']}")
    print(f"   New newsletters discovered: {stats['discovered']}")
    print(f"   Total in reading queue: {stats['queue_total']}")

    return stats


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description="SOMA Substack Agent")
    parser.add_argument("--discover-only", action="store_true",
                        help="Only discover new newsletters, don't fetch articles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(run_substack_agent(discover_only=args.discover_only))


if __name__ == "__main__":
    main()
