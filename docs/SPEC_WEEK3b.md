# SOMA Week 3b Spec — Daily Research Agent

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 3b — Autonomous Discovery  
**Goal:** SOMA goes out into the world once a day, learns what's new, and brings back recommendations tailored to Paul's physiological patterns and known preferences. The corpus grows. The recommendations get smarter.  
**Prerequisites:** Week 3 complete — corpus.json exists, feedback data accumulating, state classifier running.

---

## The Concept

SOMA knows three things by the time Week 3 is done:
1. What state Paul is typically in (from baseline + session history)
2. What has worked (from feedback_logger outcomes)
3. What Paul's interests and values are (from corpus tags and session labels)

The research agent takes all three and asks Claude — with live web search — to find what's new and relevant. Not randomly. Not algorithmically. With full context about who Paul is and what his body is telling him.

This is how SOMA's corpus stays alive. It isn't a static list. It learns and expands.

---

## Architecture

```
soma_cardio.db (state history, feedback outcomes)
corpus.json (existing recommendations)
    ↓
research_agent.py (runs once daily via cron)
    ↓
Claude API + web_search tool
    ↓
RAEN scorer (Relevance, Actionability, Evidence, Novelty)
    ↓
corpus_additions.json (staged, not auto-merged)
    ↓
corpus_review.py (Paul approves/rejects before merge)
    ↓
corpus.json (updated)
```

**Key design decision:** Claude's research output is staged, not auto-merged. Paul reviews and approves new entries before they enter the live corpus. SOMA suggests. Paul decides. This keeps the corpus honest and prevents drift.

---

## Week 3b Deliverables

- [ ] `research_agent.py` — daily Claude API call with web search, generates new corpus candidates
- [ ] `soma_profile.py` — assembles Paul's current profile for injection into research prompt
- [ ] `raen_scorer.py` — scores candidates on Relevance, Actionability, Evidence, Novelty
- [ ] `corpus_review.py` — CLI for Paul to approve/reject staged additions
- [ ] Cron job — runs research_agent.py once daily at 6am
- [ ] `corpus_additions.json` — staging file for unreviewed entries

---

## File 1: `soma_profile.py`

Assembles a rich context snapshot of Paul — who he is, what's been working, what his body has been doing. This becomes the system prompt context for the research agent.

```python
import sqlite3
import json
from datetime import datetime, timedelta
from state_classifier import classify_state
from artifact_filter import clean_rr, compute_rmssd, compute_rhr

DB_PATH = "soma_cardio.db"
CORPUS_PATH = "corpus.json"


def get_recent_state_summary(days=7):
    """Summarize Paul's physiological states over the past week."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute("""
        SELECT s.label, COUNT(r.id) as readings,
               AVG(r.rr_ms) as avg_rr
        FROM rr_intervals r
        JOIN sessions s ON r.session_id = s.session_id
        WHERE r.timestamp > ?
        GROUP BY s.label
    """, (cutoff,)).fetchall()
    conn.close()

    summary = []
    for label, readings, avg_rr in rows:
        rhr = round(60000 / avg_rr, 1) if avg_rr else None
        summary.append({
            "label": label,
            "readings": readings,
            "avg_rhr": rhr
        })
    return summary


def get_what_worked(n=10):
    """Get recommendations that produced 'better' outcomes."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT title, type, state_at_recommendation,
                   rmssd_before, rmssd_after, outcome
            FROM recommendations
            WHERE outcome = 'better'
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [{"title": r[0], "type": r[1], "state": r[2],
                 "rmssd_delta": round((r[4] or 0) - (r[3] or 0), 1),
                 "outcome": r[5]} for r in rows]
    except:
        conn.close()
        return []


def get_what_didnt_work(n=5):
    """Get recommendations that produced 'worse' or 'same' outcomes."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT title, type, state_at_recommendation, outcome
            FROM recommendations
            WHERE outcome IN ('worse', 'same')
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [{"title": r[0], "type": r[1], "state": r[2], "outcome": r[3]}
                for r in rows]
    except:
        conn.close()
        return []


def get_existing_titles():
    """Return titles already in corpus to avoid duplicates."""
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)["entries"]
    return [e["title"] for e in corpus]


def build_profile():
    """Assemble full Paul profile for research prompt injection."""
    current_state = classify_state()
    state_history = get_recent_state_summary(days=7)
    worked = get_what_worked()
    didnt_work = get_what_didnt_work()
    existing = get_existing_titles()

    profile = {
        "identity": {
            "name": "Paul",
            "age": 50,
            "location": "Duvall, WA",
            "sleep_condition": "sleep apnea, CPAP user",
            "practices": ["Tibetan Tonglen meditation", "running", "kettlebell"],
            "quitting": ["nicotine", "THC"],
            "interests": [
                "causal inference", "consciousness research", "AI architecture",
                "recommendation systems", "Buddhist philosophy", "parenting",
                "health optimization", "data science", "biometric tracking"
            ],
            "values": [
                "upstream causation", "compound over deplete",
                "be your own customer", "friction engineering",
                "compassion", "presence with family"
            ],
            "family": "Partner Nia, infant son River (born Feb 2025)",
            "reading_now": ["The Feeling of What Happens (Damasio)", "Causality (Pearl)"],
            "movies_loved": ["Soul", "Shawshank Redemption", "A Man Called Otto"],
            "vibe": "Earnest, intellectually serious, compassionate, upstream thinker"
        },
        "current_state": current_state,
        "week_summary": state_history,
        "what_worked": worked,
        "what_didnt_work": didnt_work,
        "existing_corpus": existing,
        "generated_at": datetime.now().isoformat()
    }

    return profile


if __name__ == "__main__":
    profile = build_profile()
    print(json.dumps(profile, indent=2))
```

---

## File 2: `raen_scorer.py`

Scores each candidate recommendation on four dimensions. Adapted from SOMA-Cardio AutoResearcher.

```python
"""
RAEN Scoring — Relevance, Actionability, Evidence, Novelty
Each dimension 0-10. Total /40 normalized to 0-1.
Threshold for corpus inclusion: 0.65+
"""


def score_candidate(candidate, profile):
    """
    candidate: dict with title, type, why, tags, best_states, etc.
    profile: Paul's current profile from soma_profile.build_profile()
    Returns: scored candidate with raen breakdown
    """
    scores = {}

    # Relevance — does this match Paul's interests and current state?
    interest_tags = set(profile["identity"]["interests"])
    candidate_tags = set(candidate.get("tags", []))
    tag_overlap = len(interest_tags & candidate_tags)
    state_match = profile["current_state"]["state"] in candidate.get("best_states", [])
    scores["relevance"] = min(10, tag_overlap * 2 + (4 if state_match else 0))

    # Actionability — can Paul actually do/watch/read this now?
    duration = candidate.get("duration_min", 60)
    avoid_states = candidate.get("avoid_states", [])
    current_state = profile["current_state"]["state"]
    blocked = current_state in avoid_states
    scores["actionability"] = 0 if blocked else min(10, max(4, 10 - duration // 30))

    # Evidence — is there signal from past feedback that this type works?
    worked_types = [w["type"] for w in profile.get("what_worked", [])]
    worked_titles = [w["title"] for w in profile.get("what_worked", [])]
    type_worked = worked_types.count(candidate.get("type", "")) 
    scores["evidence"] = min(10, type_worked * 3 + (3 if candidate.get("title") in worked_titles else 0))

    # Novelty — is this genuinely new to the corpus?
    existing = profile.get("existing_corpus", [])
    is_new = candidate.get("title") not in existing
    scores["novelty"] = 8 if is_new else 2

    total = sum(scores.values())
    normalized = round(total / 40, 3)

    return {
        **candidate,
        "raen": scores,
        "raen_total": normalized,
        "recommended": normalized >= 0.65
    }


def score_candidates(candidates, profile):
    scored = [score_candidate(c, profile) for c in candidates]
    scored.sort(key=lambda x: x["raen_total"], reverse=True)
    return scored
```

---

## File 3: `research_agent.py`

The core agent. Calls Claude API with web search enabled. Runs once daily.

```python
import json
import re
import sqlite3
from datetime import datetime
from soma_profile import build_profile
from raen_scorer import score_candidates

ADDITIONS_PATH = "corpus_additions.json"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Paul's interest areas — rotated daily so research covers breadth over time
RESEARCH_TOPICS = [
    "movies and films",
    "books on consciousness, AI, or philosophy of mind",
    "books on causal inference or data science",
    "physical health and recovery activities",
    "meditation and Buddhist practice resources",
    "parenting and early childhood development",
    "books on longevity and health optimization",
    "documentaries about science, nature, or human stories",
    "biohacking and HRV optimization techniques",
    "creative and generative AI tools"
]


def get_todays_topic():
    """Rotate through topics by day of year."""
    day = datetime.now().timetuple().tm_yday
    return RESEARCH_TOPICS[day % len(RESEARCH_TOPICS)]


def build_research_prompt(profile, topic):
    state = profile["current_state"]["state"]
    worked = [w["title"] for w in profile.get("what_worked", [])]
    existing = profile["existing_corpus"]

    return f"""You are SOMA's research agent. Your job is to find excellent {topic} recommendations for Paul.

## Paul's Profile
- Age 50, Duvall WA, data scientist and AI researcher
- Practices: Tibetan Tonglen meditation, running, kettlebell training
- Quitting: nicotine and THC (important: nothing that glorifies substances)
- Reading: Damasio's consciousness work, Pearl's causal inference
- Loves: upstream thinking, compassion, building things that matter, being present with his infant son River
- Movie taste: redemption arcs, meaning, human connection (Soul, Shawshank, A Man Called Otto)
- Values: compound over deplete, upstream causation, friction engineering

## Current Physiological State
{state.upper()} — {profile["current_state"]["reason"]}
RMSSD: {profile["current_state"]["rmssd"]}ms | RHR: {profile["current_state"]["rhr"]}bpm

## What Has Worked Recently
{json.dumps(worked, indent=2)}

## Already in Corpus (do not duplicate)
{json.dumps(existing, indent=2)}

## Your Task
Search the web and find 5 genuinely excellent {topic} recommendations for Paul.

For each recommendation return a JSON object with these exact fields:
- id: unique string like "res_YYYYMMDD_N"
- type: one of "movie", "book", "activity", "media"
- title: exact title
- why: 1-2 sentences explaining why this fits Paul specifically
- tags: list of relevant tags
- best_states: list from ["depleted", "recovering", "baseline", "restored", "peak"]
- avoid_states: list of states where this would be counterproductive
- duration_min: estimated time in minutes
- source: where you found this (URL or publication)
- research_date: today's date

Requirements:
- Must be searchable and verifiable — no hallucinated titles
- Must genuinely fit Paul's values and taste
- Must not duplicate anything in the existing corpus
- Prioritize things released or discovered in the last 2 years for freshness
- For books: prioritize those available on Kindle or audiobook
- For movies: prioritize streaming availability

Return ONLY a JSON array of 5 objects. No preamble, no explanation, no markdown fences."""


async def run_research():
    import aiohttp
    import asyncio

    profile = build_profile()
    topic = get_todays_topic()

    print(f"\n🔍 SOMA Research Agent")
    print(f"   Topic: {topic}")
    print(f"   Current state: {profile['current_state']['state']}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    prompt = build_research_prompt(profile, topic)

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4000,
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search"
            }
        ],
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            data = await resp.json()

    # Extract text from response
    raw_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            raw_text += block["text"]

    # Parse JSON candidates
    try:
        # Strip any accidental markdown fences
        clean = re.sub(r"```json|```", "", raw_text).strip()
        candidates = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse research output: {e}")
        print(f"Raw output:\n{raw_text[:500]}")
        return []

    # Score candidates
    scored = score_candidates(candidates, profile)

    # Stage for review
    existing_additions = []
    try:
        with open(ADDITIONS_PATH) as f:
            existing_additions = json.load(f)
    except FileNotFoundError:
        pass

    new_additions = [c for c in scored if c["recommended"]]
    all_additions = existing_additions + new_additions

    with open(ADDITIONS_PATH, "w") as f:
        json.dump(all_additions, f, indent=2)

    # Log to DB
    conn = sqlite3.connect(DB_PATH if 'DB_PATH' in dir() else "soma_cardio.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researched_at TEXT,
            topic TEXT,
            candidates_found INTEGER,
            candidates_staged INTEGER
        )
    """)
    conn.execute("""
        INSERT INTO research_log (researched_at, topic, candidates_found, candidates_staged)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), topic, len(candidates), len(new_additions)))
    conn.commit()
    conn.close()

    print(f"✅ Research complete")
    print(f"   Found: {len(candidates)} candidates")
    print(f"   Staged for review: {len(new_additions)} (RAEN ≥ 0.65)\n")

    for c in new_additions:
        print(f"  [{c['type'].upper()}] {c['title']} — RAEN: {c['raen_total']}")
        print(f"    {c['why']}\n")

    return new_additions


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_research())
```

---

## File 4: `corpus_review.py`

Paul reviews staged additions before they go live. SOMA suggests. Paul decides.

```python
import json
from datetime import datetime

CORPUS_PATH = "corpus.json"
ADDITIONS_PATH = "corpus_additions.json"
REVIEWED_PATH = "corpus_reviewed.json"


def load_staged():
    try:
        with open(ADDITIONS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_corpus(corpus):
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2)


def load_corpus():
    with open(CORPUS_PATH) as f:
        return json.load(f)


def review():
    staged = load_staged()
    if not staged:
        print("No staged recommendations to review.")
        return

    corpus = load_corpus()
    approved = []
    rejected = []

    print(f"\n📋 SOMA Research Review — {len(staged)} candidates\n")
    print("For each: [a]pprove, [r]eject, [s]kip for now, [q]uit\n")

    for entry in staged:
        print(f"{'─'*60}")
        print(f"[{entry['type'].upper()}] {entry['title']}")
        print(f"Why: {entry['why']}")
        print(f"Best for: {', '.join(entry.get('best_states', []))}")
        print(f"Tags: {', '.join(entry.get('tags', []))}")
        print(f"Duration: ~{entry.get('duration_min', '?')} min")
        print(f"RAEN score: {entry.get('raen_total', '?')}")
        if entry.get('source'):
            print(f"Source: {entry['source']}")
        print()

        choice = input("→ [a/r/s/q]: ").strip().lower()

        if choice == 'a':
            # Clean research metadata before merging
            clean_entry = {k: v for k, v in entry.items()
                          if k not in ["raen", "raen_total", "recommended",
                                       "source", "research_date"]}
            clean_entry["added_at"] = datetime.now().isoformat()
            clean_entry["watched"] = False
            corpus["entries"].append(clean_entry)
            approved.append(entry["title"])
            print(f"✅ Added: {entry['title']}\n")

        elif choice == 'r':
            rejected.append(entry["title"])
            print(f"❌ Rejected\n")

        elif choice == 'q':
            print("Quitting review. Remaining items stay staged.")
            break

        else:
            print(f"⏭ Skipped\n")

    # Save updated corpus
    save_corpus(corpus)

    # Clear approved and rejected from staging
    reviewed_ids = set(approved + rejected)
    remaining = [e for e in staged if e["title"] not in reviewed_ids]
    with open(ADDITIONS_PATH, "w") as f:
        json.dump(remaining, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"Review complete.")
    print(f"  Approved: {len(approved)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Still staged: {len(remaining)}")
    if approved:
        print(f"\nAdded to corpus:")
        for t in approved:
            print(f"  + {t}")


if __name__ == "__main__":
    review()
```

---

## Cron Setup

Run research agent once daily at 6am. Output logged, staging file updated silently.

```bash
# Edit crontab
crontab -e

# Add this line (adjust path to your project)
0 6 * * * cd /path/to/soma && python research_agent.py >> logs/research.log 2>&1
```

Create logs directory:
```bash
mkdir -p logs
```

Review staged items whenever you want — morning coffee is natural:
```bash
python corpus_review.py
```

---

## Research Topic Rotation

SOMA rotates through 10 research areas by day of year so coverage stays broad:

| Day mod 10 | Topic |
|---|---|
| 0 | Movies and films |
| 1 | Books on consciousness, AI, philosophy of mind |
| 2 | Books on causal inference or data science |
| 3 | Physical health and recovery activities |
| 4 | Meditation and Buddhist practice resources |
| 5 | Parenting and early childhood development |
| 6 | Books on longevity and health optimization |
| 7 | Documentaries — science, nature, human stories |
| 8 | Biohacking and HRV optimization techniques |
| 9 | Creative and generative AI tools |

Add more topics to `RESEARCH_TOPICS` in `research_agent.py` anytime.

---

## RAEN Scoring Reference

| Dimension | What it measures | Max |
|---|---|---|
| Relevance | Tag overlap with Paul's interests + state match | 10 |
| Actionability | Can Paul do this now? Duration reasonable? | 10 |
| Evidence | Similar types have worked before in feedback | 10 |
| Novelty | Not already in corpus | 10 |
| **Total** | Normalized 0–1. Threshold: ≥ 0.65 | 1.0 |

---

## How the Corpus Grows

```
Day 1:   18 seed entries (corpus.json from Week 3)
Week 1:  +5-15 reviewed additions across varied topics  
Month 1: 50-80 entries, feedback data on ~30 of them
Month 3: 150+ entries, RAEN learns from outcome patterns
Month 6: SOMA's corpus reflects Paul more than any algorithm ever could
```

Each entry earns its place through Paul's review and then through feedback outcomes. Nothing stays that consistently fails. Nothing gets removed that consistently works.

---

## Full Daily Flow

```
6:00 AM  research_agent.py runs (cron)
         → searches web on today's topic
         → scores with RAEN
         → stages ≥0.65 candidates

Morning  corpus_review.py (while drinking matcha)
         → 5-10 min review
         → approve what resonates, reject what doesn't

Anytime  recommender.py
         → reads current state
         → returns best matches from live corpus
         → includes newly approved entries immediately

After    feedback_logger.py
         → close the loop
         → RAEN gets smarter
```

---

## Week 4 Preview — The Conversational Probe

With research running daily and feedback accumulating, Week 4's probe becomes genuinely intelligent:

*"Your RMSSD has been suppressed for 3 days running. Based on your feedback history, The Intouchables followed by a 30-minute run has restored you from this state twice before. You haven't watched it yet. Tonight?"*

That's not a recommendation engine. That's SOMA knowing you.

---

*SOMA — Sentient Observation & Memory Architecture*  
*Phase 3b: SOMA goes out into the world and comes back with what you need.*
