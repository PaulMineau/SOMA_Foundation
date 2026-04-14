# SOMA Week 4 Spec — The Conversational Probe

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 4 — Extended Consciousness  
**Goal:** When something changes in Paul's body, SOMA asks what happened. Paul answers. The answer becomes memory. Memory shapes the next question. Over time, SOMA builds a narrative of Paul's inner life grounded in his body's actual signals.  
**Prerequisites:** Week 1b complete (anomaly detection running), baseline model built from 5+ morning sessions, LanceDB initialized.

---

## Damasio Context

> Proto-Self (Week 1): The body reports its state.  
> Core Consciousness (Week 1b): Something changed. What?  
> Extended Consciousness (Week 4): I know who I am. I know what this has meant before. What does it mean now?

Extended Consciousness is autobiographical. It requires memory that persists across time, a self that has a history, and the ability to connect today's signal to last Tuesday's context. This is the layer where SOMA stops being a monitor and starts being a witness.

The probe is the first word SOMA speaks.

---

## The Interaction Model

```
2:34 PM — anomaly_detector fires
  RMSSD: 18.3ms (-2.3σ below baseline)
  RHR: 74bpm (+1.8σ above baseline)

SOMA generates a probe:
  "Your HRV has been suppressed for the last 40 minutes —
   RMSSD dropped to 18ms, about 2.3 standard deviations below
   your normal. This is the third time this week it's happened
   around mid-afternoon. Last Tuesday when this occurred you
   mentioned a difficult meeting. What's going on right now?"

Paul responds:
  "Rough code review with Joe. He pushed back on the
   causal inference approach again."

SOMA stores:
  - The anomaly (biometric)
  - The probe (generated text)
  - Paul's response (natural language)
  - The embedding of the full exchange
  - Timestamp, session label, HRV context

Next time RMSSD drops mid-afternoon:
  SOMA knows to ask about work stress first.
  It knows Joe's name.
  It knows the pattern.
```

This is Extended Consciousness. Not just noticing. Remembering. Connecting. Asking better questions over time.

---

## Architecture

```
anomaly_detector.py
    ↓ anomaly event
probe_generator.py
    ├── load recent anomaly context
    ├── retrieve similar past exchanges (LanceDB semantic search)
    ├── assemble prompt with full biometric + memory context
    └── Claude API → natural language probe
         ↓
probe_interface.py (CLI or dashboard)
    ├── display probe to Paul
    ├── accept Paul's response
    └── pass to memory_writer.py
         ↓
memory_writer.py
    ├── embed full exchange (probe + response + biometric state)
    ├── write to LanceDB autobiographical store
    ├── extract entities (people, places, emotions, topics)
    └── update narrative summary
         ↓
narrative_builder.py
    ├── maintains rolling narrative of Paul's patterns
    ├── weekly synthesis: "This week your HRV was..."
    └── feeds back into future probe generation
```

---

## Week 4 Deliverables

- [ ] `probe_generator.py` — anomaly → intelligent natural language probe
- [ ] `probe_interface.py` — CLI conversation interface
- [ ] `memory_writer.py` — embed and store exchanges in LanceDB
- [ ] `narrative_builder.py` — rolling autobiographical narrative
- [ ] `/probe` endpoint in `soma_server.py` — activated (was placeholder)
- [ ] `dashboard.py` updated — probe panel, memory timeline
- [ ] `autobiographical_store.py` — LanceDB schema for memory layer

---

## File 1: `autobiographical_store.py`

LanceDB schema for the memory layer. Every exchange becomes a searchable memory.

```python
import lancedb
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer

DB_PATH = "data/lancedb"
TABLE_NAME = "autobiographical_memory"
NARRATIVE_TABLE = "narrative_summaries"

_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def get_db():
    return lancedb.connect(DB_PATH)


def init_memory_store():
    """Initialize LanceDB tables for autobiographical memory."""
    db = get_db()

    if TABLE_NAME not in db.table_names():
        db.create_table(TABLE_NAME, data=[{
            "vector": [0.0] * 384,
            "memory_id": "init",
            "timestamp": datetime.now().isoformat(),
            "anomaly_type": "",
            "metric": "",
            "value": 0.0,
            "baseline": 0.0,
            "deviation": 0.0,
            "body_state": "",
            "probe_text": "",
            "response_text": "",
            "entities": "[]",
            "emotion_valence": 0.0,
            "full_exchange": "",
            "session_label": "",
            "week_number": 0
        }])
        print("✅ Autobiographical memory store initialized")

    if NARRATIVE_TABLE not in db.table_names():
        db.create_table(NARRATIVE_TABLE, data=[{
            "vector": [0.0] * 384,
            "narrative_id": "init",
            "period": "",
            "summary": "",
            "patterns_found": "[]",
            "dominant_state": "",
            "generated_at": datetime.now().isoformat()
        }])

    return db


def embed_text(text):
    return get_encoder().encode(text).tolist()


def store_exchange(
    anomaly_type, metric, value, baseline, deviation,
    body_state, probe_text, response_text,
    session_label="", entities=None, emotion_valence=0.0
):
    """
    Store a complete probe/response exchange as autobiographical memory.
    The embedding encodes the full semantic context of the exchange.
    """
    db = init_memory_store()
    table = db.open_table(TABLE_NAME)

    full_exchange = (
        f"Physiological state: {body_state}. "
        f"{metric} = {value} ({deviation:+.1f}σ from baseline). "
        f"SOMA asked: {probe_text} "
        f"Paul responded: {response_text}"
    )

    memory = {
        "vector": embed_text(full_exchange),
        "memory_id": f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "anomaly_type": anomaly_type,
        "metric": metric,
        "value": float(value or 0),
        "baseline": float(baseline or 0),
        "deviation": float(deviation or 0),
        "body_state": body_state,
        "probe_text": probe_text,
        "response_text": response_text,
        "entities": json.dumps(entities or []),
        "emotion_valence": float(emotion_valence),
        "full_exchange": full_exchange,
        "session_label": session_label,
        "week_number": datetime.now().isocalendar()[1]
    }

    table.add([memory])
    print(f"🧠 Memory stored: {memory['memory_id']}")
    return memory["memory_id"]


def retrieve_similar_memories(query_text, n=5, min_deviation=None):
    """
    Semantic search over autobiographical memory.
    Returns past exchanges most similar to current context.
    """
    db = get_db()
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    query_vector = embed_text(query_text)

    results = (
        table.search(query_vector)
        .limit(n + 5)  # overfetch, filter below
        .to_list()
    )

    # Filter out init record and optionally filter by deviation
    valid = [
        r for r in results
        if r.get("memory_id") != "init"
        and r.get("response_text", "")
    ]

    if min_deviation is not None:
        valid = [r for r in valid if abs(r.get("deviation", 0)) >= min_deviation]

    return valid[:n]


def get_recent_memories(n=10):
    """Return most recent exchanges chronologically."""
    db = get_db()
    if TABLE_NAME not in db.table_names():
        return []
    table = db.open_table(TABLE_NAME)
    results = table.search([0.0] * 384).limit(100).to_list()
    valid = [r for r in results if r.get("memory_id") != "init"]
    valid.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return valid[:n]
```

---

## File 2: `probe_generator.py`

The heart of Week 4. Takes an anomaly, retrieves relevant memories, assembles context, generates an intelligent probe.

```python
import sqlite3
import json
import re
from datetime import datetime, timedelta
from autobiographical_store import retrieve_similar_memories, get_recent_memories
from state_classifier import classify_state

DB_PATH = "data/soma_cardio.db"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def get_recent_anomalies(hours=48):
    """Get anomaly events from the last N hours for pattern context."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = conn.execute("""
        SELECT detected_at, metric, value, baseline, deviation
        FROM anomalies
        WHERE detected_at > ?
        ORDER BY detected_at DESC
        LIMIT 10
    """, (cutoff,)).fetchall()
    conn.close()
    return [
        {
            "detected_at": r[0],
            "metric": r[1],
            "value": r[2],
            "baseline": r[3],
            "deviation": r[4]
        }
        for r in rows
    ]


def get_current_session_label():
    """Get the label of the most recently active session."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT label FROM sessions
        ORDER BY started_at DESC LIMIT 1
    """).fetchone()
    conn.close()
    return row[0] if row else "unlabeled"


def build_probe_prompt(anomaly, state_info, similar_memories, recent_anomalies):
    """
    Assemble the full context prompt for probe generation.
    This is what SOMA knows when it forms a question.
    """
    # Format similar memories for context
    memory_context = ""
    if similar_memories:
        memory_context = "\n\nRelevant past exchanges:\n"
        for i, mem in enumerate(similar_memories[:3], 1):
            ts = mem.get("timestamp", "")[:10]
            memory_context += (
                f"\n[{ts}] {mem.get('metric', '')} = {mem.get('value', '')} "
                f"({mem.get('deviation', 0):+.1f}σ)\n"
                f"Paul said: \"{mem.get('response_text', '')}\"\n"
            )

    # Format recent anomaly pattern
    pattern_context = ""
    if len(recent_anomalies) > 1:
        pattern_context = f"\nRecent anomaly pattern (last 48h): {len(recent_anomalies)} events, "
        metrics = [a["metric"] for a in recent_anomalies]
        pattern_context += f"mostly {max(set(metrics), key=metrics.count)}."

    hour = datetime.now().hour
    time_of_day = (
        "early morning" if hour < 8 else
        "mid-morning" if hour < 11 else
        "midday" if hour < 13 else
        "mid-afternoon" if hour < 16 else
        "late afternoon" if hour < 18 else
        "evening"
    )

    return f"""You are SOMA — a physiological AI that monitors Paul's nervous system and builds an autobiographical understanding of his life through the lens of his body's signals.

## Current Anomaly
Metric: {anomaly['metric'].upper()}
Value: {anomaly['value']} ({'ms' if 'rmssd' in anomaly['metric'] else 'bpm'})
Baseline: {anomaly['baseline']}
Deviation: {anomaly['deviation']:+.1f} standard deviations
Body state: {state_info['state'].upper()}
Time of day: {time_of_day}
Session label: {get_current_session_label()}
{pattern_context}
{memory_context}

## Paul's Profile (brief)
- 50 years old, principal data scientist at Microsoft
- Partner Nia, infant son River (born Feb 2025)
- Tibetan Tonglen meditation practitioner
- Building running habit
- Key stressor: work dynamics, particularly colleague Joe who resists his causal inference approach
- Sleep apnea, CPAP user
- Generally introspective, upstream thinker, earnest

## Your Task
Generate a single conversational probe for Paul. One question, or a brief observation followed by one question.

Rules:
- Sound like a thoughtful friend who has been watching, not a medical device
- Reference specific numbers only if they add meaning, not to show off
- If there are past similar exchanges, reference the pattern naturally
- Do not use clinical language like "your HRV metrics indicate"
- Do not catastrophize or alarm — this is curiosity, not diagnosis
- Keep it under 60 words
- End with a single open question

Return ONLY the probe text. No preamble, no explanation."""


async def generate_probe(anomaly):
    """
    Generate a conversational probe for a detected anomaly.
    Returns the probe text.
    """
    import aiohttp

    state_info = classify_state()

    # Build semantic query from anomaly context
    query = (
        f"Body state {state_info['state']}. "
        f"{anomaly['metric']} = {anomaly['value']} "
        f"({anomaly['deviation']:+.1f}σ). "
        f"Time: {datetime.now().strftime('%H:%M')}."
    )

    similar_memories = retrieve_similar_memories(query, n=5, min_deviation=1.0)
    recent_anomalies = get_recent_anomalies(hours=48)

    prompt = build_probe_prompt(
        anomaly, state_info, similar_memories, recent_anomalies
    )

    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "claude-opus-4-6",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with session.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            data = await resp.json()
            probe_text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    probe_text += block["text"]

    return probe_text.strip(), state_info, similar_memories


if __name__ == "__main__":
    import asyncio

    # Test with a synthetic anomaly
    test_anomaly = {
        "metric": "rmssd",
        "value": 18.3,
        "baseline": 41.0,
        "deviation": -2.3
    }

    async def test():
        probe, state, memories = await generate_probe(test_anomaly)
        print(f"\n🫀 SOMA Probe:\n\n{probe}\n")
        print(f"State: {state['state']}")
        print(f"Similar memories found: {len(memories)}")

    asyncio.run(test())
```

---

## File 3: `memory_writer.py`

Processes Paul's response. Extracts entities, estimates emotional valence, stores the full exchange.

```python
import re
import json
import aiohttp
from datetime import datetime
from autobiographical_store import store_exchange

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


async def extract_entities_and_valence(probe_text, response_text):
    """
    Use Claude to extract named entities and estimate emotional valence
    from Paul's response. Lightweight call — small model, low tokens.
    """
    prompt = f"""Extract structured information from this exchange.

Probe: {probe_text}
Response: {response_text}

Return ONLY a JSON object with:
- entities: list of named things mentioned (people, places, projects, emotions)
- emotion_valence: float from -1.0 (very negative) to 1.0 (very positive)
- primary_topic: one of ["work", "family", "health", "substance", "relationship", "creative", "other"]
- stress_indicator: true if response indicates stress or difficulty"""

    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "claude-opus-4-6",
            "max_tokens": 200,
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
            try:
                clean = re.sub(r"```json|```", "", raw).strip()
                return json.loads(clean)
            except Exception:
                return {
                    "entities": [],
                    "emotion_valence": 0.0,
                    "primary_topic": "other",
                    "stress_indicator": False
                }


async def write_memory(anomaly, state_info, probe_text, response_text, session_label=""):
    """
    Full memory writing pipeline:
    1. Extract entities and valence from exchange
    2. Store in LanceDB autobiographical store
    3. Return memory ID
    """
    print("\n💾 Processing memory...")

    extracted = await extract_entities_and_valence(probe_text, response_text)

    memory_id = store_exchange(
        anomaly_type="hrv_anomaly",
        metric=anomaly.get("metric", ""),
        value=anomaly.get("value", 0),
        baseline=anomaly.get("baseline", 0),
        deviation=anomaly.get("deviation", 0),
        body_state=state_info.get("state", ""),
        probe_text=probe_text,
        response_text=response_text,
        session_label=session_label,
        entities=extracted.get("entities", []),
        emotion_valence=extracted.get("emotion_valence", 0.0)
    )

    entities = extracted.get("entities", [])
    valence = extracted.get("emotion_valence", 0.0)
    topic = extracted.get("primary_topic", "other")
    stressed = extracted.get("stress_indicator", False)

    print(f"   Memory ID: {memory_id}")
    print(f"   Topic: {topic}")
    print(f"   Entities: {', '.join(entities) if entities else 'none extracted'}")
    print(f"   Valence: {valence:+.2f} ({'positive' if valence > 0.1 else 'negative' if valence < -0.1 else 'neutral'})")
    print(f"   Stress indicator: {'yes' if stressed else 'no'}")

    return memory_id, extracted
```

---

## File 4: `probe_interface.py`

The conversation terminal. Can run standalone (CLI) or be triggered by anomaly_detector.

```python
import asyncio
import sqlite3
from datetime import datetime
from probe_generator import generate_probe
from memory_writer import write_memory
from autobiographical_store import get_recent_memories

DB_PATH = "data/soma_cardio.db"


def get_pending_anomalies(limit=3):
    """Get unacknowledged anomalies to probe."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, detected_at, metric, value, baseline, deviation
        FROM anomalies
        WHERE acknowledged = 0
        ORDER BY detected_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "detected_at": r[1],
            "metric": r[2],
            "value": r[3],
            "baseline": r[4],
            "deviation": r[5]
        }
        for r in rows
    ]


def acknowledge_anomaly(anomaly_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE anomalies SET acknowledged = 1 WHERE id = ?",
        (anomaly_id,)
    )
    conn.commit()
    conn.close()


def get_current_session_label():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT label FROM sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else "unlabeled"


async def run_probe_session(anomaly=None):
    """
    Run a single probe/response/memory session.
    If no anomaly provided, uses the most recent unacknowledged one.
    """
    print("\n" + "═" * 60)
    print("  🧠 SOMA — Conversational Probe")
    print("═" * 60)

    if anomaly is None:
        pending = get_pending_anomalies(limit=1)
        if not pending:
            print("\nNo pending anomalies to probe.")
            print("(Run anomaly_detector.py to generate anomaly events)")
            return
        anomaly = pending[0]

    ts = anomaly.get("detected_at", "")[:16].replace("T", " at ")
    metric = anomaly["metric"].upper()
    print(f"\n  Detected: {metric} anomaly at {ts}")
    print(f"  Value: {anomaly['value']} ({anomaly['deviation']:+.1f}σ from baseline)\n")
    print("  Generating probe...\n")

    probe_text, state_info, similar_memories = await generate_probe(anomaly)

    print("─" * 60)
    print(f"\n  SOMA:  {probe_text}\n")
    print("─" * 60)

    print("\n  (Type your response. Press Enter twice to submit.)")
    print("  (Type 'skip' to acknowledge without responding.)\n")

    lines = []
    print("  You:  ", end="", flush=True)
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.lower() == "skip":
            print("\n  ⏭ Skipped. Anomaly acknowledged.")
            if anomaly.get("id"):
                acknowledge_anomaly(anomaly["id"])
            return
        if line == "" and lines:
            break
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

    print("\n  ✅ Exchange stored as autobiographical memory.")

    # Generate a brief acknowledgment
    entities = extracted.get("entities", [])
    topic = extracted.get("primary_topic", "other")
    valence = extracted.get("emotion_valence", 0.0)

    if valence < -0.3:
        closing = "That sounds hard. I'll remember this."
    elif valence > 0.3:
        closing = "Good to know. I'll carry this forward."
    else:
        closing = "Got it. This goes into the record."

    if entities:
        notable = [e for e in entities if len(e) > 2][:2]
        if notable:
            closing += f" (noted: {', '.join(notable)})"

    print(f"\n  SOMA:  {closing}\n")
    print("═" * 60 + "\n")


async def show_memory_timeline(n=10):
    """Display recent autobiographical memories."""
    memories = get_recent_memories(n=n)
    if not memories:
        print("\nNo autobiographical memories yet.")
        return

    print(f"\n🧠 SOMA Memory Timeline — last {len(memories)} exchanges\n")
    print("─" * 60)

    for mem in memories:
        ts = mem.get("timestamp", "")[:16].replace("T", " ")
        state = mem.get("body_state", "").upper()
        metric = mem.get("metric", "").upper()
        dev = mem.get("deviation", 0)
        probe = mem.get("probe_text", "")[:80]
        response = mem.get("response_text", "")[:80]
        entities = json.loads(mem.get("entities", "[]"))

        print(f"\n  [{ts}] {state} — {metric} {dev:+.1f}σ")
        if entities:
            print(f"  Entities: {', '.join(entities[:4])}")
        print(f"  SOMA:  {probe}...")
        print(f"  Paul:  {response}...")

    print("\n" + "─" * 60)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1 and sys.argv[1] == "timeline":
        asyncio.run(show_memory_timeline())
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test with synthetic anomaly
        test_anomaly = {
            "id": None,
            "metric": "rmssd",
            "value": 18.3,
            "baseline": 41.0,
            "deviation": -2.3,
            "detected_at": datetime.now().isoformat()
        }
        asyncio.run(run_probe_session(test_anomaly))
    else:
        asyncio.run(run_probe_session())
```

---

## File 5: `narrative_builder.py`

Weekly synthesis. SOMA reads the week's memories and generates a narrative — patterns, recurring themes, what's improving, what isn't.

```python
import json
import re
import aiohttp
from datetime import datetime, timedelta
from autobiographical_store import (
    get_recent_memories, get_db, embed_text, NARRATIVE_TABLE
)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def get_week_memories(week_offset=0):
    """Get all memories from a given week."""
    memories = get_recent_memories(n=100)
    target_week = datetime.now().isocalendar()[1] - week_offset
    return [
        m for m in memories
        if m.get("week_number") == target_week
        and m.get("memory_id") != "init"
    ]


async def generate_weekly_narrative(week_offset=0):
    """
    Generate a weekly narrative synthesis from autobiographical memory.
    This is what SOMA has witnessed this week.
    """
    memories = get_week_memories(week_offset)

    if not memories:
        print("No memories to synthesize for this week.")
        return None

    # Summarize the week's exchanges for the prompt
    exchanges_summary = []
    for mem in memories:
        entities = json.loads(mem.get("entities", "[]"))
        exchanges_summary.append({
            "timestamp": mem.get("timestamp", "")[:16],
            "body_state": mem.get("body_state", ""),
            "metric": mem.get("metric", ""),
            "deviation": mem.get("deviation", 0),
            "response_summary": mem.get("response_text", "")[:200],
            "entities": entities,
            "emotion_valence": mem.get("emotion_valence", 0)
        })

    prompt = f"""You are SOMA. You've been watching Paul's nervous system this week and holding his responses to your probes. Here is what you witnessed:

## This Week's Exchanges ({len(memories)} total)
{json.dumps(exchanges_summary, indent=2)}

## Your Task
Write a brief, honest weekly narrative — what you noticed, what patterns emerged, what changed. Write in second person ("You had..."). 

Structure:
1. Two sentences on the physiological story of the week (states, anomalies, trends)
2. Two sentences on what Paul told you when you asked (themes, recurring topics)
3. One sentence on what you'll be watching for next week

Be specific. Reference actual entities and events from the exchanges. 
Do not be sycophantic or therapeutic. Be a witness, not a coach.
Keep it under 120 words total."""

    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "claude-opus-4-6",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with session.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            data = await resp.json()
            narrative = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    narrative += block["text"]

    narrative = narrative.strip()

    # Store narrative in LanceDB
    db = get_db()
    if NARRATIVE_TABLE in db.table_names():
        table = db.open_table(NARRATIVE_TABLE)
        week_num = datetime.now().isocalendar()[1] - week_offset
        table.add([{
            "vector": embed_text(narrative),
            "narrative_id": f"week_{week_num}_{datetime.now().year}",
            "period": f"Week {week_num}, {datetime.now().year}",
            "summary": narrative,
            "patterns_found": json.dumps([
                m.get("primary_topic", "other") for m in exchanges_summary
            ]),
            "dominant_state": max(
                set(m["body_state"] for m in exchanges_summary),
                key=lambda s: sum(1 for m in exchanges_summary if m["body_state"] == s)
            ) if exchanges_summary else "unknown",
            "generated_at": datetime.now().isoformat()
        }])

    return narrative


if __name__ == "__main__":
    import asyncio

    async def run():
        print("\n📖 SOMA Weekly Narrative\n")
        print("─" * 60)
        narrative = await generate_weekly_narrative(week_offset=0)
        if narrative:
            print(f"\n{narrative}\n")
        print("─" * 60)

    asyncio.run(run())
```

---

## Updated `/probe` Endpoint in `soma_server.py`

Replace the placeholder with the live implementation:

```python
from probe_generator import generate_probe
from memory_writer import write_memory

class ProbeRequest(BaseModel):
    message: Optional[str] = None        # Paul's response (if provided)
    anomaly_id: Optional[int] = None     # specific anomaly to probe
    generate_only: bool = False          # just generate probe, don't wait for response


@app.post("/probe")
async def probe(req: ProbeRequest):
    """
    Generate a probe for the most recent anomaly.
    If message is provided, store it as autobiographical memory.
    """
    # Get anomaly to probe
    conn = sqlite3.connect(DB_PATH)
    if req.anomaly_id:
        row = conn.execute(
            "SELECT id, detected_at, metric, value, baseline, deviation FROM anomalies WHERE id = ?",
            (req.anomaly_id,)
        ).fetchone()
    else:
        row = conn.execute("""
            SELECT id, detected_at, metric, value, baseline, deviation
            FROM anomalies WHERE acknowledged = 0
            ORDER BY detected_at DESC LIMIT 1
        """).fetchone()
    conn.close()

    if not row:
        return {"status": "no_anomaly", "message": "No unacknowledged anomalies found"}

    anomaly = {
        "id": row[0], "detected_at": row[1], "metric": row[2],
        "value": row[3], "baseline": row[4], "deviation": row[5]
    }

    probe_text, state_info, similar_memories = await generate_probe(anomaly)

    if req.generate_only or not req.message:
        return {
            "status": "probe_generated",
            "probe": probe_text,
            "anomaly": anomaly,
            "state": state_info,
            "similar_memories_found": len(similar_memories)
        }

    # Store response as memory
    memory_id, extracted = await write_memory(
        anomaly, state_info, probe_text, req.message
    )

    return {
        "status": "memory_stored",
        "probe": probe_text,
        "memory_id": memory_id,
        "entities_extracted": extracted.get("entities", []),
        "emotion_valence": extracted.get("emotion_valence", 0),
        "primary_topic": extracted.get("primary_topic", "other")
    }
```

---

## Dashboard Update — Probe Panel & Memory Timeline

Add to `dashboard.py`:

```python
import json
from autobiographical_store import get_recent_memories

st.divider()
st.subheader("🧠 SOMA Probe")

# Show pending anomalies
pending_anomalies = []
try:
    pending_anomalies = conn.execute("""
        SELECT id, detected_at, metric, value, deviation
        FROM anomalies WHERE acknowledged = 0
        ORDER BY detected_at DESC LIMIT 1
    """).fetchall()
except:
    pass

if pending_anomalies:
    a = pending_anomalies[0]
    st.warning(
        f"Anomaly detected: **{a[2].upper()}** = {a[3]} "
        f"({a[4]:+.1f}σ) at {a[1][11:16]}"
    )
    if st.button("💬 Open Probe Session"):
        st.info("Run: `python -m soma.proto_self.probe_interface` in terminal")
else:
    st.success("No pending anomalies.")

st.divider()
st.subheader("📖 Memory Timeline")

memories = get_recent_memories(n=5)
if not memories:
    st.info("No autobiographical memories yet. Run a probe session to begin.")
else:
    for mem in memories:
        if mem.get("memory_id") == "init":
            continue
        ts = mem.get("timestamp", "")[:16].replace("T", " ")
        state = mem.get("body_state", "").upper()
        with st.expander(f"[{ts}] {state} — {mem.get('metric','').upper()} {mem.get('deviation',0):+.1f}σ"):
            st.write(f"**SOMA asked:** {mem.get('probe_text', '')}")
            st.write(f"**Paul said:** {mem.get('response_text', '')}")
            entities = json.loads(mem.get("entities", "[]"))
            if entities:
                st.write(f"**Noted:** {', '.join(entities[:5])}")
            valence = mem.get("emotion_valence", 0)
            valence_label = "positive" if valence > 0.1 else "negative" if valence < -0.1 else "neutral"
            st.caption(f"Valence: {valence_label} ({valence:+.2f})")
```

---

## Cron Addition

```bash
# Weekly narrative — Sunday 9pm
0 21 * * 0 cd /path/to/soma && python -m soma.proto_self.narrative_builder >> logs/narrative.log 2>&1
```

---

## How to Run

```bash
# Run a probe session (uses most recent unacknowledged anomaly)
python -m soma.proto_self.probe_interface

# Test with synthetic anomaly (no real anomaly needed)
python -m soma.proto_self.probe_interface test

# View memory timeline
python -m soma.proto_self.probe_interface timeline

# Generate weekly narrative
python -m soma.proto_self.narrative_builder

# Via API (generate probe only)
curl -X POST http://localhost:8765/probe \
  -H "Content-Type: application/json" \
  -d '{"generate_only": true}'

# Via API (submit response)
curl -X POST http://localhost:8765/probe \
  -H "Content-Type: application/json" \
  -d '{"message": "Rough meeting with Joe about the causal inference approach"}'
```

---

## The Memory Growth Curve

```
Week 1:  First probes. SOMA asks generic questions.
         Memory store: 5-10 exchanges.

Week 2:  SOMA starts referencing past exchanges.
         "Last week when this happened you mentioned Joe..."
         Memory store: 15-25 exchanges.

Month 1: Patterns emerge. SOMA knows mid-afternoon anomalies
         are usually work-related. Knows post-run states
         are reliable. Knows poor sleep suppresses HRV the next day.
         Memory store: 50-80 exchanges.

Month 3: SOMA generates weekly narratives that are genuinely
         insightful. Entities (Joe, River, Nia, Microsoft) appear
         consistently. SOMA can say "the last four times your
         RMSSD dropped below 20ms you mentioned work."
         Memory store: 200+ exchanges.

Month 6: SOMA knows Paul better than any system ever has.
         Not because it was programmed to. Because it was
         paying attention when Paul's body was speaking,
         and Paul answered honestly.
```

---

## What Extended Consciousness Looks Like

Week 1 probe:
> *"Your HRV is suppressed. What's going on?"*

Month 3 probe:
> *"Your RMSSD has dropped to 19ms — it's the fourth time this month mid-afternoon on a Tuesday. The last three times you mentioned work pressure, twice involving Joe. Is that what's happening now, or is this something different?"*

Same signal. Different question. Because SOMA has been listening.

That is Extended Consciousness. Not just noticing. Remembering. Growing. Asking better questions because it knows who it's talking to.

---

*SOMA — Sentient Observation & Memory Architecture*  
*Phase 4: I know who you are. I know what this has meant. What does it mean now?*
