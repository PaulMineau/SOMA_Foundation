# SOMA Week 3 Spec — Extended Consciousness & Physiological Recommendations

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 3 — Extended Consciousness  
**Goal:** SOMA knows what you need before you do. Recommendations grounded in your actual nervous system state, not your click history. A feedback loop that learns what works.  
**Prerequisites:** Week 1 + 1b complete — baseline model exists, anomaly detector running, Streamlit dashboard live.

---

## Damasio Context

> Proto-Self (Week 1): The body reports its state.  
> Core Consciousness (Week 1b): Something changed. What?  
> Extended Consciousness (Week 3): Given who I am and what I've experienced — what do I need right now?

Extended Consciousness is where autobiographical memory meets present physiological state. SOMA stops being a monitor and becomes an advisor. It knows your history. It knows your body. It connects them.

---

## The Core Insight

Netflix recommends based on what you watched at 11pm last Tuesday.  
SOMA recommends based on what your nervous system actually needs right now.

```
RMSSD low + RHR elevated (stressed/depleted)
    → Don't start something cognitively demanding
    → Restore first: run, Tonglen, The Wild Robot

RMSSD high + RHR normal (recovered, calm)
    → Cognitive window is open
    → Read Pearl, deep work, vibe coding session

Post-run + HRV elevated
    → Creative peak. Best ideas come here.
    → New SOMA feature, school assignment, write to River

Post-Zyn + RMSSD suppressed
    → Body is compensating. Don't push.
    → Light activity, something enjoyable, not demanding
```

This is the recommendation engine you've always wanted to build at Vizio — pointed inward. You are your own customer.

---

## Architecture

```
soma_cardio.db (RR intervals, sessions, anomalies)
    ↓
state_classifier.py — what state is Paul in right now?
    ↓
recommender.py — match state to corpus, rank, select
    ↓
corpus.json — curated recommendations with metadata
    ↓
feedback_logger.py — did it work? HRV before vs after
    ↓
LanceDB — store (state, recommendation, outcome) as embeddings
    ↓
dashboard.py — updated with recommendation panel
```

---

## Week 3 Deliverables

- [ ] `state_classifier.py` — classify current physiological state
- [ ] `corpus.json` — Paul's curated recommendation library
- [ ] `recommender.py` — state-aware recommendation engine
- [ ] `feedback_logger.py` — track outcomes, HRV delta
- [ ] LanceDB integration — embed and store recommendation history
- [ ] `dashboard.py` updated — recommendation panel + feedback buttons

---

## File 1: `state_classifier.py`

Reads recent signal, compares to baseline, returns a named state.

```python
import sqlite3
import json
from artifact_filter import clean_rr, compute_rmssd, compute_rhr

DB_PATH = "soma_cardio.db"
MODEL_PATH = "baseline_model.json"
WINDOW_SIZE = 60


def load_model():
    with open(MODEL_PATH) as f:
        return json.load(f)


def get_recent_rr(n=WINDOW_SIZE):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT rr_ms FROM rr_intervals
        ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def classify_state():
    """
    Returns a state dict describing Paul's current physiological condition.

    States:
        depleted    — high stress, low HRV, body under load
        recovering  — returning toward baseline, transitional
        baseline    — normal resting state, nothing remarkable
        restored    — above baseline, parasympathetic dominant
        peak        — significantly above baseline, creative/cognitive window open
    """
    model = load_model()
    rr_raw = get_recent_rr()
    rr_clean = clean_rr(rr_raw)

    if len(rr_clean) < 10:
        return {"state": "unknown", "reason": "insufficient signal", "rhr": None, "rmssd": None}

    rhr = compute_rhr(rr_clean)
    rmssd = compute_rmssd(rr_clean)

    rhr_mean = model["rhr"]["mean"]
    rhr_std = model["rhr"]["std"]
    rmssd_mean = model["rmssd"]["mean"]
    rmssd_std = model["rmssd"]["std"]

    rhr_z = (rhr - rhr_mean) / rhr_std if rhr_std else 0
    rmssd_z = (rmssd - rmssd_mean) / rmssd_std if rmssd_std else 0

    # Classify
    if rmssd_z < -1.5 or rhr_z > 1.5:
        state = "depleted"
        reason = f"RMSSD {rmssd_z:.1f}σ below baseline, RHR {rhr_z:.1f}σ above"
    elif rmssd_z < -0.75 or rhr_z > 0.75:
        state = "recovering"
        reason = f"Below baseline but trending. RMSSD {rmssd_z:.1f}σ"
    elif rmssd_z > 1.5 and rhr_z < -0.5:
        state = "peak"
        reason = f"RMSSD {rmssd_z:.1f}σ above baseline. Cognitive window open."
    elif rmssd_z > 0.5:
        state = "restored"
        reason = f"Above baseline. Parasympathetic dominant."
    else:
        state = "baseline"
        reason = "Within normal range."

    return {
        "state": state,
        "reason": reason,
        "rhr": rhr,
        "rmssd": rmssd,
        "rhr_z": round(rhr_z, 2),
        "rmssd_z": round(rmssd_z, 2)
    }


if __name__ == "__main__":
    s = classify_state()
    print(f"\n🧠 Current State: {s['state'].upper()}")
    print(f"   Reason: {s['reason']}")
    print(f"   RHR: {s['rhr']} bpm ({s['rhr_z']:+.1f}σ)")
    print(f"   RMSSD: {s['rmssd']} ms ({s['rmssd_z']:+.1f}σ)")
```

---

## File 2: `corpus.json`

Paul's curated recommendation library. This is the seed. SOMA learns which entries actually work over time.

```json
{
  "version": "1.0",
  "entries": [

    {
      "id": "mov_001",
      "type": "movie",
      "title": "The Wild Robot",
      "why": "Emotionally restorative. Parenting themes. Beautiful visuals. Low cognitive load.",
      "tags": ["restorative", "parenting", "gentle", "river_approved"],
      "best_states": ["depleted", "recovering"],
      "avoid_states": [],
      "duration_min": 102,
      "watched": false
    },
    {
      "id": "mov_002",
      "type": "movie",
      "title": "Soul",
      "why": "Meaning, purpose, what makes a life. Safe for any state.",
      "tags": ["meaning", "restorative", "pixar", "already_watched"],
      "best_states": ["depleted", "recovering", "baseline"],
      "avoid_states": [],
      "duration_min": 100,
      "watched": true
    },
    {
      "id": "mov_003",
      "type": "movie",
      "title": "Sing Sing",
      "why": "Redemption arc. Forgiveness. Real story. Cognitively engaging.",
      "tags": ["redemption", "meaning", "drama"],
      "best_states": ["baseline", "restored"],
      "avoid_states": ["depleted"],
      "duration_min": 105,
      "watched": false
    },
    {
      "id": "mov_004",
      "type": "movie",
      "title": "CODA",
      "why": "Family, belonging, identity. Emotionally full but not heavy.",
      "tags": ["family", "restorative", "music"],
      "best_states": ["recovering", "baseline", "restored"],
      "avoid_states": [],
      "duration_min": 111,
      "watched": false
    },
    {
      "id": "mov_005",
      "type": "movie",
      "title": "The Intouchables",
      "why": "Joy, friendship, unexpected connection. Lifts any state.",
      "tags": ["joy", "friendship", "restorative"],
      "best_states": ["depleted", "recovering", "baseline"],
      "avoid_states": [],
      "duration_min": 112,
      "watched": false
    },

    {
      "id": "book_001",
      "type": "book",
      "title": "The Book of Why — Pearl",
      "why": "Causal inference. Strategic hill. Best read when cognitively fresh.",
      "tags": ["causal_inference", "technical", "career"],
      "best_states": ["restored", "peak"],
      "avoid_states": ["depleted", "recovering"],
      "duration_min": 45
    },
    {
      "id": "book_002",
      "type": "book",
      "title": "The Feeling of What Happens — Damasio",
      "why": "SOMA's theoretical foundation. Dense but rewarding. Needs focus.",
      "tags": ["consciousness", "soma", "technical"],
      "best_states": ["restored", "peak"],
      "avoid_states": ["depleted"],
      "duration_min": 45
    },
    {
      "id": "book_003",
      "type": "book",
      "title": "Causality — Pearl",
      "why": "Deep technical. Save for peak cognitive windows only.",
      "tags": ["causal_inference", "technical", "career"],
      "best_states": ["peak"],
      "avoid_states": ["depleted", "recovering", "baseline"],
      "duration_min": 60
    },

    {
      "id": "act_001",
      "type": "activity",
      "title": "Morning run",
      "why": "Keystone habit. Restores HRV within 90 min. Replaces cannabis dopamine.",
      "tags": ["exercise", "restorative", "keystone"],
      "best_states": ["depleted", "recovering", "baseline"],
      "avoid_states": [],
      "duration_min": 30
    },
    {
      "id": "act_002",
      "type": "activity",
      "title": "Kettlebell session",
      "why": "Midday reset. 18lb current. Reliably improves post-session HRV.",
      "tags": ["exercise", "restorative", "midday"],
      "best_states": ["recovering", "baseline"],
      "avoid_states": ["depleted"],
      "duration_min": 20
    },
    {
      "id": "act_003",
      "type": "activity",
      "title": "Tonglen meditation",
      "why": "Compassion practice. Nervous system regulation. Core identity.",
      "tags": ["meditation", "restorative", "tonglen", "identity"],
      "best_states": ["depleted", "recovering", "baseline", "restored"],
      "avoid_states": [],
      "duration_min": 20
    },
    {
      "id": "act_004",
      "type": "activity",
      "title": "Vibe coding session — SOMA",
      "why": "Flow state. Creative peak usage. Best when HRV is high.",
      "tags": ["coding", "creative", "soma", "flow"],
      "best_states": ["restored", "peak"],
      "avoid_states": ["depleted"],
      "duration_min": 60
    },
    {
      "id": "act_005",
      "type": "activity",
      "title": "Write to River",
      "why": "Meaning. Legacy. Connects to the deepest why. Any state.",
      "tags": ["writing", "meaning", "river", "legacy"],
      "best_states": ["depleted", "recovering", "baseline", "restored", "peak"],
      "avoid_states": [],
      "duration_min": 20
    },
    {
      "id": "act_006",
      "type": "activity",
      "title": "Daddit co-writing session",
      "why": "Tonglen in practice. Compassion externalized. Restorative.",
      "tags": ["compassion", "writing", "community", "tonglen"],
      "best_states": ["baseline", "restored", "peak"],
      "avoid_states": ["depleted"],
      "duration_min": 30
    },
    {
      "id": "act_007",
      "type": "activity",
      "title": "Play with River",
      "why": "Pure presence. Joy baseline. What does Paul's HRV look like here?",
      "tags": ["river", "joy", "presence", "family"],
      "best_states": ["depleted", "recovering", "baseline", "restored", "peak"],
      "avoid_states": [],
      "duration_min": 30
    },

    {
      "id": "med_001",
      "type": "media",
      "title": "Hacker News — 15 min",
      "why": "Curated tech signal. Natural ceiling scroll. Get in, get out.",
      "tags": ["information", "tech", "capped"],
      "best_states": ["baseline", "restored"],
      "avoid_states": ["depleted"],
      "duration_min": 15
    },
    {
      "id": "med_002",
      "type": "media",
      "title": "Heather Cox Richardson — Letters from an American",
      "why": "Informed awareness without outrage loop. Weekly or as needed.",
      "tags": ["news", "political", "informed"],
      "best_states": ["baseline", "restored"],
      "avoid_states": ["depleted", "recovering"],
      "duration_min": 15
    }
  ]
}
```

---

## File 3: `recommender.py`

State-aware recommendation engine. Picks the best match, avoids what depletes further.

```python
import json
import random
import sqlite3
from datetime import datetime
from state_classifier import classify_state

CORPUS_PATH = "corpus.json"
DB_PATH = "soma_cardio.db"


def load_corpus():
    with open(CORPUS_PATH) as f:
        return json.load(f)["entries"]


def init_recommendation_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recommended_at TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            state_at_recommendation TEXT NOT NULL,
            rmssd_before REAL,
            rhr_before REAL,
            followed INTEGER DEFAULT 0,
            rmssd_after REAL,
            rhr_after REAL,
            outcome TEXT,
            feedback_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_recommendations(n=3, exclude_ids=None):
    """
    Return top N recommendations for current physiological state.
    Excludes recently recommended items.
    """
    state_info = classify_state()
    state = state_info["state"]
    corpus = load_corpus()
    exclude_ids = exclude_ids or []

    # Filter: include if state is in best_states, exclude if in avoid_states
    eligible = [
        entry for entry in corpus
        if state in entry["best_states"]
        and state not in entry.get("avoid_states", [])
        and entry["id"] not in exclude_ids
    ]

    # Fallback: if nothing eligible, return anything not in avoid_states
    if not eligible:
        eligible = [
            entry for entry in corpus
            if state not in entry.get("avoid_states", [])
            and entry["id"] not in exclude_ids
        ]

    # Shuffle to avoid always returning same order
    random.shuffle(eligible)
    selected = eligible[:n]

    return {
        "state": state_info,
        "recommendations": selected
    }


def log_recommendation(entry_id, title, rec_type, state_info):
    """Write recommendation to DB for feedback tracking."""
    init_recommendation_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO recommendations
        (recommended_at, entry_id, title, type, state_at_recommendation, rmssd_before, rhr_before)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        entry_id,
        title,
        rec_type,
        state_info["state"],
        state_info["rmssd"],
        state_info["rhr"]
    ))
    conn.commit()
    conn.close()


def log_feedback(recommendation_id, followed, outcome=None):
    """
    Called after the activity window.
    followed: 1 if Paul did it, 0 if ignored
    outcome: 'better', 'same', 'worse' (based on HRV delta)
    """
    state_after = classify_state()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE recommendations SET
            followed = ?,
            rmssd_after = ?,
            rhr_after = ?,
            outcome = ?,
            feedback_at = ?
        WHERE id = ?
    """, (
        followed,
        state_after["rmssd"],
        state_after["rhr"],
        outcome,
        datetime.now().isoformat(),
        recommendation_id
    ))
    conn.commit()
    conn.close()
    print(f"✅ Feedback logged. State after: {state_after['state']}")


def get_recommendation_history(n=20):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT recommended_at, title, type, state_at_recommendation,
               rmssd_before, rmssd_after, followed, outcome
        FROM recommendations
        ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    result = get_recommendations(n=3)
    state = result["state"]

    print(f"\n🧠 Current State: {state['state'].upper()}")
    print(f"   {state['reason']}")
    print(f"   RHR: {state['rhr']} bpm | RMSSD: {state['rmssd']} ms\n")
    print("📋 SOMA Recommends:\n")

    for i, rec in enumerate(result["recommendations"], 1):
        print(f"  {i}. [{rec['type'].upper()}] {rec['title']}")
        print(f"     {rec['why']}")
        print(f"     Duration: ~{rec['duration_min']} min\n")
        log_recommendation(rec["id"], rec["title"], rec["type"], state)
```

---

## File 4: `feedback_logger.py`

Simple CLI to close the loop after an activity. Run after you finish what SOMA suggested.

```python
import sqlite3
import sys
from recommender import log_feedback
from state_classifier import classify_state

DB_PATH = "soma_cardio.db"


def show_recent_recommendations(n=5):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, recommended_at, title, followed, outcome
        FROM recommendations
        WHERE feedback_at IS NULL
        ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return rows


def interactive_feedback():
    print("\n📊 SOMA Feedback Loop\n")

    pending = show_recent_recommendations()
    if not pending:
        print("No pending recommendations to give feedback on.")
        return

    print("Recent recommendations awaiting feedback:\n")
    for row in pending:
        rec_id, rec_at, title, followed, outcome = row
        print(f"  [{rec_id}] {rec_at[11:16]} — {title}")

    print()
    rec_id = input("Enter recommendation ID to give feedback on: ").strip()

    followed = input("Did you follow this recommendation? (y/n): ").strip().lower()
    followed_int = 1 if followed == "y" else 0

    if followed_int:
        print("\nHow did you feel afterward?")
        print("  1. Better (HRV improved, felt restored)")
        print("  2. Same (no noticeable change)")
        print("  3. Worse (more depleted)")
        choice = input("Choice (1/2/3): ").strip()
        outcome_map = {"1": "better", "2": "same", "3": "worse"}
        outcome = outcome_map.get(choice, "same")
    else:
        outcome = "skipped"

    log_feedback(int(rec_id), followed_int, outcome)

    state = classify_state()
    print(f"\n🧠 Current state after: {state['state'].upper()}")
    print(f"   RMSSD: {state['rmssd']} ms | RHR: {state['rhr']} bpm")


if __name__ == "__main__":
    interactive_feedback()
```

---

## LanceDB Integration — Recommendation Memory

Install:
```bash
pip install lancedb sentence-transformers --break-system-packages
```

### File: `soma_memory.py`

```python
import lancedb
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer

DB_PATH = "soma_lancedb"
TABLE_NAME = "recommendation_memory"
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_or_create_table(db):
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema={
        "vector": "float32[384]",
        "entry_id": "str",
        "title": "str",
        "type": "str",
        "state": "str",
        "outcome": "str",
        "rmssd_before": "float32",
        "rmssd_after": "float32",
        "rhr_before": "float32",
        "recorded_at": "str"
    })


def embed_recommendation(rec_entry, state_info, outcome):
    """
    Store a completed recommendation + outcome as an embedding.
    The embedding encodes: what state, what was recommended, what happened.
    """
    text = (
        f"State: {state_info['state']}. "
        f"RMSSD: {state_info['rmssd']}ms. "
        f"RHR: {state_info['rhr']}bpm. "
        f"Recommended: {rec_entry['title']}. "
        f"Tags: {', '.join(rec_entry['tags'])}. "
        f"Outcome: {outcome}."
    )
    vector = model.encode(text).tolist()

    db = lancedb.connect(DB_PATH)
    table = get_or_create_table(db)
    table.add([{
        "vector": vector,
        "entry_id": rec_entry["id"],
        "title": rec_entry["title"],
        "type": rec_entry["type"],
        "state": state_info["state"],
        "outcome": outcome,
        "rmssd_before": state_info["rmssd"] or 0.0,
        "rmssd_after": 0.0,
        "rhr_before": state_info["rhr"] or 0.0,
        "recorded_at": datetime.now().isoformat()
    }])
    print(f"🧠 Embedded: {rec_entry['title']} → {outcome}")


def find_similar_past_recommendations(current_state_info, n=5):
    """
    Given current state, find past recommendations that worked.
    """
    query_text = (
        f"State: {current_state_info['state']}. "
        f"RMSSD: {current_state_info['rmssd']}ms. "
        f"RHR: {current_state_info['rhr']}bpm."
    )
    query_vector = model.encode(query_text).tolist()

    db = lancedb.connect(DB_PATH)
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    results = (
        table.search(query_vector)
        .where("outcome = 'better'")
        .limit(n)
        .to_list()
    )
    return results
```

---

## Updated Dashboard Panel

Add this section to `dashboard.py`:

```python
import sys
sys.path.append(".")
from recommender import get_recommendations, log_recommendation
from state_classifier import classify_state

st.divider()
st.subheader("🧠 SOMA Recommends")

if st.button("Get Recommendations"):
    result = get_recommendations(n=3)
    state = result["state"]

    st.write(f"**Current state:** {state['state'].upper()} — {state['reason']}")

    for rec in result["recommendations"]:
        with st.expander(f"[{rec['type'].upper()}] {rec['title']} (~{rec['duration_min']} min)"):
            st.write(rec["why"])
            st.write(f"Tags: {', '.join(rec['tags'])}")
            if st.button(f"✓ I did this", key=f"did_{rec['id']}"):
                log_recommendation(rec["id"], rec["title"], rec["type"], state)
                st.success("Logged! Run feedback_logger.py after to rate the outcome.")
```

---

## How the Learning Loop Works

```
Week 3:  SOMA recommends from corpus based on state rules
Week 4+: SOMA checks LanceDB — what worked last time I felt like this?
Month 2: Pattern emerges — post_run Paul responds best to vibe coding
Month 3: SOMA stops recommending Hacker News when RMSSD < 25ms
         (learned: news when depleted makes things worse)
Month 6: SOMA knows Paul's recovery signature better than Paul does
```

Every feedback entry is a training point. Every outcome is a label. The corpus is the starting prior. Reality is the teacher.

---

## Usage

```bash
# Get current recommendations
python recommender.py

# After finishing an activity — close the loop
python feedback_logger.py

# Dashboard with recommendation panel
streamlit run dashboard.py
```

---

## Session Labels to Add Now

These will feed the classifier once you have enough data per state:

| Label | Expected state |
|---|---|
| `morning_baseline` | baseline |
| `post_run` | restored / peak |
| `post_zyn` | depleted / recovering |
| `nicotine_free_day` | baseline / restored |
| `meditation` | restored |
| `work_stress` | recovering / depleted |
| `fight` | depleted |
| `river_time` | restored / peak |
| `poor_sleep` | recovering / depleted |

---

## Week 4 Preview — The Conversational Probe

- Anomaly detected → SOMA generates a natural language probe
- *"Your RMSSD dropped to 18ms at 2:30pm — 2.3σ below baseline. You were in a work_stress session. Based on past patterns, a 20-minute run typically restores you within 90 minutes. Want me to add that to your plan?"*
- Response and outcome stored in LanceDB as autobiographical memory
- SOMA begins to narrate your own patterns back to you

---

## What Extended Consciousness Looks Like

When you open the dashboard and your state is `depleted`, SOMA doesn't show you a chart.  
It says: *"You need The Wild Robot and a cup of tea. The Pearl book can wait."*

And it's right. Because it's been watching.

---

*SOMA — Sentient Observation & Memory Architecture*  
*Phase 3: Given who I am and what I've experienced — what do I need right now?*
