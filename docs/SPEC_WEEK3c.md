# SOMA Week 3c Spec — Substack Agent & Newsletter Intelligence

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 3c — Curated Reading Intelligence  
**Goal:** SOMA monitors newsletters Paul follows, discovers new writers in his interest areas, scores articles with RAEN, and surfaces the right read at the right physiological moment.  
**Prerequisites:** Week 3b complete — research_agent.py running, RAEN scorer working, corpus review flow established.

---

## The Problem SOMA Solves

Paul already has a good news diet philosophy: weekly briefings over daily scrolling, NetNewsWire over algorithmic feeds, get in and get out. But even curated RSS has no physiological awareness. It surfaces everything regardless of whether Paul is in a state to absorb it.

SOMA changes that:

```
Dense political analysis (HCR long-form)
    → Best states: restored, peak
    → Skip when: depleted, recovering

Short reflective parenting piece
    → Best states: any
    → Especially good when: depleted (restorative, low cognitive load)

Technical deep dive on causal inference
    → Best states: peak only
    → Skip when: anything below restored

New writer discovery in consciousness research
    → Stage for review regardless of state
    → Let Paul decide when to read
```

The algorithm didn't know you were depleted at 2pm on Tuesday. SOMA does.

---

## Architecture

```
newsletters.json (Paul's tracked newsletters + discovery seeds)
    ↓
substack_agent.py (runs daily, separate from research_agent.py)
    ├── fetch_recent_posts() — check each newsletter for new content
    ├── fetch_article_content() — web_fetch full article text
    ├── classify_article() — Claude API summarizes + tags
    ├── raen_scorer.py — score against Paul's current profile
    └── stage for review OR auto-surface if RAEN > 0.85
         ↓
article_queue.json (staged articles with RAEN scores)
         ↓
article_review.py — Paul approves, dismisses, or saves for later
         ↓
dashboard.py updated — "Reading Queue" panel, state-matched to now
```

---

## Week 3c Deliverables

- [ ] `newsletters.json` — Paul's tracked newsletters + discovery seeds
- [ ] `substack_agent.py` — daily fetch, classify, score pipeline
- [ ] `article_queue.json` — staged reading queue
- [ ] `article_review.py` — CLI review + queue management
- [ ] `dashboard.py` updated — Reading Queue panel with state-aware sorting
- [ ] Cron entry — runs substack_agent.py daily at 6:15am (after research_agent.py)

---

## File 1: `newsletters.json`

Paul's tracked sources. Two tiers: followed (known good) and discovery seeds (expand the map).

```json
{
  "version": "1.0",
  "followed": [
    {
      "id": "nl_001",
      "name": "Letters from an American",
      "author": "Heather Cox Richardson",
      "url": "https://heathercoxrichardson.substack.com",
      "rss": "https://heathercoxrichardson.substack.com/feed",
      "tags": ["politics", "history", "informed_awareness"],
      "best_states": ["baseline", "restored"],
      "avoid_states": ["depleted"],
      "typical_length_min": 10,
      "notes": "Keep. Weekly digest preferred over daily. Skip when depleted."
    }
  ],
  "discovery_seeds": [
    {
      "topic": "consciousness and AI",
      "keywords": ["consciousness", "sentience", "qualia", "embodied cognition", "Damasio"],
      "exclude_keywords": ["crypto", "NFT", "hustle"]
    },
    {
      "topic": "causal inference and data science",
      "keywords": ["causal inference", "Pearl", "observational study", "confounding", "DAG"],
      "exclude_keywords": []
    },
    {
      "topic": "Buddhist practice and contemplative science",
      "keywords": ["Tonglen", "Tibetan Buddhism", "meditation", "compassion", "mindfulness"],
      "exclude_keywords": ["productivity hacks", "morning routine"]
    },
    {
      "topic": "parenting and early childhood",
      "keywords": ["infant development", "attachment", "fatherhood", "parenting"],
      "exclude_keywords": []
    },
    {
      "topic": "longevity and health optimization",
      "keywords": ["HRV", "longevity", "sleep apnea", "cardiovascular health", "VO2 max"],
      "exclude_keywords": ["supplements", "biohack", "shred"]
    },
    {
      "topic": "AI alignment and ethics",
      "keywords": ["AI alignment", "AI safety", "Anthropic", "Claude", "LLM"],
      "exclude_keywords": []
    }
  ],
  "discovered": [],
  "dismissed": []
}
```

---

## File 2: `substack_agent.py`

Core agent. Fetches newsletter content, classifies with Claude, scores with RAEN, stages for review.

```python
import json
import re
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from soma_profile import build_profile
from raen_scorer import score_candidates

NEWSLETTERS_PATH = "newsletters.json"
QUEUE_PATH = "article_queue.json"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
AUTO_SURFACE_THRESHOLD = 0.85   # skip review, go straight to reading queue
STAGE_THRESHOLD = 0.65          # stage for manual review
MAX_ARTICLE_AGE_DAYS = 3        # ignore older posts
MAX_ARTICLES_PER_RUN = 10       # cap total articles processed per day


def load_newsletters():
    with open(NEWSLETTERS_PATH) as f:
        return json.load(f)


def load_queue():
    try:
        with open(QUEUE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_queue(queue):
    with open(QUEUE_PATH, "w") as f:
        json.dump(queue, f, indent=2)


def is_recent(published_str, max_days=MAX_ARTICLE_AGE_DAYS):
    """Check if article was published within the window."""
    try:
        from email.utils import parsedate_to_datetime
        pub_date = parsedate_to_datetime(published_str)
        return (datetime.now(pub_date.tzinfo) - pub_date).days <= max_days
    except Exception:
        return True  # if we can't parse, assume recent


async def fetch_rss_entries(newsletter):
    """Parse RSS feed and return recent entries."""
    rss_url = newsletter.get("rss")
    if not rss_url:
        return []

    feed = feedparser.parse(rss_url)
    recent = []

    for entry in feed.entries[:5]:  # check last 5 posts max
        published = entry.get("published", "")
        if not is_recent(published):
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
            "typical_length_min": newsletter.get("typical_length_min", 10)
        })

    return recent


async def fetch_article_text(url, session, max_chars=4000):
    """Fetch full article text via web_fetch (public posts only)."""
    try:
        # Use Claude API web_fetch capability
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{
                "role": "user",
                "content": f"Fetch and return the main article text from this URL. Return only the article content, nothing else: {url}"
            }]
        }
        async with session.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            data = await resp.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]
            return text[:max_chars]
    except Exception as e:
        print(f"  ⚠️  Could not fetch {url}: {e}")
        return None


async def classify_article(entry, article_text, profile, session):
    """
    Ask Claude to classify and score the article for Paul.
    Returns enriched entry with tags, reading time, why it fits Paul.
    """
    state = profile["current_state"]["state"]
    content = article_text or entry.get("summary", "No content available")

    prompt = f"""You are SOMA's article classifier. Analyze this article for Paul.

## Paul's Profile (brief)
- Age 50, data scientist, Tibetan Buddhist practitioner
- Interests: causal inference, consciousness, parenting (infant son River), longevity, AI
- Values: upstream causation, compound over deplete, compassion
- Current physiological state: {state.upper()}
- Avoids: glorification of substances, outrage loops, hustle culture

## Article
Title: {entry['title']}
Author: {entry['author']}
Newsletter: {entry['newsletter_name']}
URL: {entry['url']}

Content:
{content}

## Your Task
Return ONLY a JSON object with these fields:
- title: article title (cleaned)
- why: 1-2 sentences on why this specifically fits Paul (or why it doesn't)
- tags: list of 3-6 relevant tags
- best_states: list from ["depleted", "recovering", "baseline", "restored", "peak"]
- avoid_states: list of states where reading this would be counterproductive
- estimated_read_min: realistic reading time in minutes
- key_insight: one sentence capturing the article's core idea
- fits_paul: true or false
- paywall: true if content appears to be behind a paywall, false if freely readable"""

    try:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with session.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            data = await resp.json()
            raw = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    raw += block["text"]
            clean = re.sub(r"```json|```", "", raw).strip()
            classification = json.loads(clean)
            return {**entry, **classification}
    except Exception as e:
        print(f"  ⚠️  Classification failed for {entry['title']}: {e}")
        return None


async def discover_new_newsletters(seed, profile, session):
    """
    Use Claude + web search to find promising new Substack writers
    matching a discovery seed topic.
    """
    newsletters = load_newsletters()
    known = [n["name"] for n in newsletters["followed"]]
    dismissed = newsletters.get("dismissed", [])

    prompt = f"""Search Substack and the web for active newsletter writers on this topic: {seed['topic']}

Keywords to look for: {', '.join(seed['keywords'])}
Exclude: {', '.join(seed['exclude_keywords']) if seed['exclude_keywords'] else 'nothing specific'}

Find 3 newsletters that would genuinely interest someone who:
- Is a data scientist studying causal inference and AI consciousness
- Practices Tibetan Buddhism and Tonglen meditation
- Has an infant son, cares about parenting science
- Values intellectual depth over hot takes
- Prefers upstream analysis over surface commentary

Already follows: {', '.join(known)}
Previously dismissed: {', '.join(dismissed)}

Return ONLY a JSON array of objects with:
- name: newsletter name
- author: author name  
- url: Substack URL
- rss: RSS feed URL (usually url + /feed)
- description: 1 sentence on what it covers
- why_paul: 1 sentence on why it fits Paul specifically
- tags: list of tags
- typical_length_min: estimated reading time per post"""

    try:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": prompt}]
        }
        async with session.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            data = await resp.json()
            raw = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    raw += block["text"]
            clean = re.sub(r"```json|```", "", raw).strip()
            return json.loads(clean)
    except Exception as e:
        print(f"  ⚠️  Discovery failed for topic {seed['topic']}: {e}")
        return []


async def run():
    print(f"\n📰 SOMA Substack Agent")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    profile = build_profile()
    newsletters = load_newsletters()
    existing_queue = load_queue()
    existing_urls = {a["url"] for a in existing_queue}

    staged = []
    auto_surfaced = []
    processed = 0

    async with aiohttp.ClientSession() as session:

        # ── 1. Check followed newsletters for new posts ────────────────────
        print("📡 Checking followed newsletters...\n")
        for newsletter in newsletters["followed"]:
            entries = await fetch_rss_entries(newsletter)
            for entry in entries:
                if entry["url"] in existing_urls:
                    continue
                if processed >= MAX_ARTICLES_PER_RUN:
                    break

                print(f"  → {entry['title'][:60]}")
                article_text = await fetch_article_text(entry["url"], session)
                classified = await classify_article(entry, article_text, profile, session)

                if not classified or not classified.get("fits_paul"):
                    print(f"     ✗ Doesn't fit Paul, skipping")
                    continue

                if classified.get("paywall"):
                    print(f"     🔒 Paywalled, skipping")
                    continue

                # Build RAEN-compatible entry
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
                    "best_states": classified.get("best_states",
                                                  entry["newsletter_best_states"]),
                    "avoid_states": classified.get("avoid_states",
                                                   entry["newsletter_avoid_states"]),
                    "duration_min": classified.get("estimated_read_min",
                                                   entry["typical_length_min"]),
                    "fetched_at": datetime.now().isoformat(),
                    "published": entry["published"],
                    "source_type": "followed_newsletter"
                }

                scored_list = score_candidates([raen_entry], profile)
                scored = scored_list[0] if scored_list else raen_entry
                raen_score = scored.get("raen_total", 0)

                print(f"     RAEN: {raen_score}")

                if raen_score >= AUTO_SURFACE_THRESHOLD:
                    scored["auto_surfaced"] = True
                    auto_surfaced.append(scored)
                    print(f"     ⚡ Auto-surfaced (RAEN {raen_score} ≥ {AUTO_SURFACE_THRESHOLD})")
                elif raen_score >= STAGE_THRESHOLD:
                    scored["auto_surfaced"] = False
                    staged.append(scored)
                    print(f"     📋 Staged for review")
                else:
                    print(f"     ✗ Below threshold, skipping")

                processed += 1

        # ── 2. Newsletter discovery (one seed topic per day) ──────────────
        print("\n🔍 Discovering new newsletters...\n")
        day_index = datetime.now().timetuple().tm_yday
        seed = newsletters["discovery_seeds"][day_index % len(newsletters["discovery_seeds"])]
        print(f"  Topic: {seed['topic']}\n")

        discovered = await discover_new_newsletters(seed, profile, session)
        new_discoveries = []
        known_names = {n["name"] for n in newsletters["followed"]}
        dismissed_names = set(newsletters.get("dismissed", []))

        for nl in discovered:
            if nl["name"] not in known_names and nl["name"] not in dismissed_names:
                nl["discovered_at"] = datetime.now().isoformat()
                nl["best_states"] = ["baseline", "restored", "peak"]
                nl["avoid_states"] = []
                new_discoveries.append(nl)
                print(f"  + {nl['name']} by {nl['author']}")
                print(f"    {nl.get('why_paul', nl.get('description', ''))}\n")

        if new_discoveries:
            newsletters["discovered"] = newsletters.get("discovered", []) + new_discoveries
            with open(NEWSLETTERS_PATH, "w") as f:
                json.dump(newsletters, f, indent=2)

    # ── 3. Update queue ───────────────────────────────────────────────────
    all_new = auto_surfaced + staged
    updated_queue = existing_queue + all_new
    save_queue(updated_queue)

    print(f"\n✅ Substack agent complete")
    print(f"   Articles processed: {processed}")
    print(f"   Auto-surfaced: {len(auto_surfaced)}")
    print(f"   Staged for review: {len(staged)}")
    print(f"   New newsletters discovered: {len(new_discoveries)}")
    print(f"   Total in reading queue: {len(updated_queue)}")


if __name__ == "__main__":
    asyncio.run(run())
```

---

## File 3: `article_review.py`

Review staged articles and newly discovered newsletters. Fast CLI, 5-10 min morning session.

```python
import json
from datetime import datetime

QUEUE_PATH = "article_queue.json"
NEWSLETTERS_PATH = "newsletters.json"


def load_queue():
    try:
        with open(QUEUE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_queue(queue):
    with open(QUEUE_PATH, "w") as f:
        json.dump(queue, f, indent=2)


def review_articles():
    queue = load_queue()
    pending = [a for a in queue if not a.get("reviewed") and not a.get("auto_surfaced")]

    if not pending:
        print("No articles pending review.")
        return

    print(f"\n📰 Article Review — {len(pending)} pending\n")
    print("[a]pprove to reading list, [d]ismiss, [s]kip, [q]uit\n")

    updated = []
    for article in queue:
        if article.get("reviewed") or article.get("auto_surfaced"):
            updated.append(article)
            continue
        if article not in pending:
            updated.append(article)
            continue

        print(f"{'─'*60}")
        print(f"📄 {article['title']}")
        print(f"   {article['newsletter']} — {article['author']}")
        print(f"   {article.get('key_insight', article.get('why', ''))}")
        print(f"   Best for: {', '.join(article.get('best_states', []))}")
        print(f"   Read time: ~{article.get('duration_min', '?')} min")
        print(f"   RAEN: {article.get('raen_total', '?')}")
        print(f"   URL: {article['url']}\n")

        choice = input("→ [a/d/s/q]: ").strip().lower()

        if choice == 'a':
            article["reviewed"] = True
            article["approved"] = True
            article["reviewed_at"] = datetime.now().isoformat()
            print(f"✅ Added to reading list\n")
        elif choice == 'd':
            article["reviewed"] = True
            article["approved"] = False
            article["reviewed_at"] = datetime.now().isoformat()
            print(f"❌ Dismissed\n")
        elif choice == 'q':
            updated.append(article)
            break
        else:
            print(f"⏭ Skipped\n")

        updated.append(article)

    save_queue(updated)
    approved = sum(1 for a in updated if a.get("approved") and not a.get("read"))
    print(f"\nReading list: {approved} articles ready")


def review_discovered_newsletters():
    with open(NEWSLETTERS_PATH) as f:
        newsletters = json.load(f)

    pending = newsletters.get("discovered", [])
    if not pending:
        print("No new newsletters to review.")
        return

    print(f"\n📬 Newsletter Discovery Review — {len(pending)} found\n")
    print("[f]ollow, [d]ismiss, [s]kip, [q]uit\n")

    followed = newsletters["followed"]
    dismissed = newsletters.get("dismissed", [])
    remaining_discovered = []

    for nl in pending:
        print(f"{'─'*60}")
        print(f"📬 {nl['name']} by {nl['author']}")
        print(f"   {nl.get('description', '')}")
        print(f"   Why Paul: {nl.get('why_paul', '')}")
        print(f"   URL: {nl.get('url', '')}\n")

        choice = input("→ [f/d/s/q]: ").strip().lower()

        if choice == 'f':
            clean = {k: v for k, v in nl.items()
                    if k not in ["discovered_at", "why_paul"]}
            clean["notes"] = nl.get("why_paul", "")
            followed.append(clean)
            print(f"✅ Now following {nl['name']}\n")
        elif choice == 'd':
            dismissed.append(nl["name"])
            print(f"❌ Dismissed\n")
        elif choice == 'q':
            remaining_discovered.append(nl)
            break
        else:
            remaining_discovered.append(nl)
            print(f"⏭ Skipped\n")

    newsletters["followed"] = followed
    newsletters["dismissed"] = dismissed
    newsletters["discovered"] = remaining_discovered

    with open(NEWSLETTERS_PATH, "w") as f:
        json.dump(newsletters, f, indent=2)


def show_reading_list(state=None):
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
    print(f"\n📚 Reading List {label} — {len(readable)} articles\n")

    for i, a in enumerate(readable, 1):
        auto = "⚡" if a.get("auto_surfaced") else "✓"
        print(f"  {i}. {auto} {a['title']}")
        print(f"     {a['newsletter']} — ~{a.get('duration_min', '?')} min")
        print(f"     {a.get('key_insight', a.get('why', ''))}")
        print(f"     {a['url']}\n")


def mark_read(index):
    queue = load_queue()
    readable = [a for a in queue if (a.get("approved") or a.get("auto_surfaced")) and not a.get("read")]
    if index < 1 or index > len(readable):
        print("Invalid index")
        return
    target = readable[index - 1]
    for a in queue:
        if a["url"] == target["url"]:
            a["read"] = True
            a["read_at"] = datetime.now().isoformat()
    save_queue(queue)
    print(f"✅ Marked as read: {target['title']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "articles":
            review_articles()
        elif cmd == "newsletters":
            review_discovered_newsletters()
        elif cmd == "list":
            state = sys.argv[2] if len(sys.argv) > 2 else None
            show_reading_list(state)
        elif cmd == "read":
            mark_read(int(sys.argv[2]))
    else:
        review_articles()
        print()
        review_discovered_newsletters()
```

---

## Dashboard Update — Reading Queue Panel

Add to `dashboard.py`:

```python
from article_review import show_reading_list
from state_classifier import classify_state
import json

st.divider()
st.subheader("📚 Reading Queue")

queue_data = []
try:
    with open("article_queue.json") as f:
        queue_data = json.load(f)
except FileNotFoundError:
    pass

current_state = classify_state()["state"]
readable = [
    a for a in queue_data
    if (a.get("approved") or a.get("auto_surfaced"))
    and not a.get("read")
    and current_state in a.get("best_states", [current_state])
    and current_state not in a.get("avoid_states", [])
]
readable.sort(key=lambda x: x.get("raen_total", 0), reverse=True)

if not readable:
    st.info(f"No articles matched for current state: {current_state.upper()}")
else:
    st.write(f"Showing {len(readable)} articles matched to your current state: **{current_state.upper()}**")
    for a in readable[:5]:
        with st.expander(f"{'⚡' if a.get('auto_surfaced') else '✓'} {a['title']} — {a['newsletter']} (~{a.get('duration_min', '?')} min)"):
            st.write(a.get("key_insight", a.get("why", "")))
            st.write(f"**Best for:** {', '.join(a.get('best_states', []))}")
            st.markdown(f"[Read →]({a['url']})")
```

---

## Cron Setup

```bash
crontab -e

# Research agent — 6:00 AM
0 6 * * * cd /path/to/soma && python research_agent.py >> logs/research.log 2>&1

# Substack agent — 6:15 AM (after research agent)
15 6 * * * cd /path/to/soma && python substack_agent.py >> logs/substack.log 2>&1
```

---

## Morning Review Commands

```bash
# Review staged articles + new newsletters (5-10 min with matcha)
python article_review.py

# Review only staged articles
python article_review.py articles

# Review only discovered newsletters
python article_review.py newsletters

# Show reading list matched to current state
python article_review.py list depleted
python article_review.py list peak

# Mark article as read
python article_review.py read 2

# Full dashboard (shows reading queue panel)
streamlit run dashboard.py
```

---

## How SOMA Reads With You

```
You're depleted at 2pm
    → Dashboard shows reading queue filtered to: depleted
    → Surfaces: short parenting piece (8 min), reflective HCR post
    → Hides: long-form causal inference deep dive

You're at peak after a run
    → Dashboard surfaces: consciousness research, Pearl adjacent
    → Flags: new Substack writer on embodied cognition discovered today

You open HCR's latest
    → SOMA already fetched it at 6am
    → Classified it as baseline/restored only
    → This morning you were depleted — it was invisible
    → Now you're restored — it surfaces automatically
```

The article didn't change. Your state did. SOMA knew the difference.

---

## Paywall Handling

| Post type | SOMA behavior |
|---|---|
| Free public post | Fetches full text, classifies, scores |
| Paywalled post | Detects wall, logs title only, skips |
| Partial preview | Classifies from preview, notes truncation |
| Subscriber-only newsletter | Skips entirely, notes in log |

SOMA never tries to bypass paywalls. If content is walled, it surfaces the title so Paul can decide whether the subscription is worth it.

---

## Newsletter Discovery Growth

```
Week 1:  1 followed newsletter (HCR), 6 discovery seeds active
Week 2:  3-5 newsletters followed after review
Month 1: 8-12 newsletters, ~20-30 articles in rotation
Month 3: SOMA knows which newsletters Paul reads vs skips
         and adjusts RAEN Evidence scores accordingly
Month 6: Discovery seeds refined by what Paul has actually followed
         SOMA stops suggesting certain topics that consistently get dismissed
```

---

## What This Becomes

Right now: SOMA fetches articles and scores them against rules.

Month 3: SOMA notices Paul never reads political analysis when RMSSD < 30ms. Stops surfacing it when he's depleted — without being told.

Month 6: SOMA sees that Paul reads consciousness pieces after runs but skips them on low-sleep days. It learns this pattern from the `read_at` timestamps vs session labels.

Year 1: The reading queue isn't a list. It's a conversation between SOMA and Paul's nervous system about what his mind is ready to receive.

---

*SOMA — Sentient Observation & Memory Architecture*  
*Phase 3c: The right read at the right moment. Not because an algorithm said so. Because your body did.*
