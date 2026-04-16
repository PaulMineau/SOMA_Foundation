# SOMA Week 5 Spec — The Brain Simulation Layer

**Project:** SOMA (Sentient Observation & Memory Architecture)  
**Phase:** 5 — Multi-Model Brain Simulation with Affective Embedding Pipeline  
**Goal:** Wire together AI models that approximate brain regions, pass typed embeddings between them, and visualize the signal flow in real time.

> This is the week SOMA stops being a data logger and starts being a brain.

---

## What We're Building

A Python orchestration layer that runs five AI-backed "brain modules" as async tasks. Each module consumes input embeddings, does its job, and publishes output embeddings to a shared state bus. A Streamlit dashboard shows the live signal — HRV line chart, module states, and a readable description of every embedding currently in flight.

The Polar H10 feeds in continuously. A camera stub is wired but inactive — LLaVA slots in when you're ready.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│                                                                      │
│  [Polar H10 BLE]     [Semantic Stream]     [Camera stub / LLaVA]    │
│   RR intervals        text / context        visual frame (future)    │
│       │                    │                        │                │
└───────┼────────────────────┼────────────────────────┼────────────────┘
        │                    │                        │
        ▼                    ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      THALAMUS ROUTER                                 │
│               (Qwen3 via Ollama — local, fast)                       │
│                                                                      │
│  Receives all raw signals. Decides:                                  │
│    - How much weight to give each signal type                        │
│    - Which brain modules get priority this cycle                     │
│    - Flags "low road" events (HRV spike → skip cortex, go direct)   │
│                                                                      │
│  Output: ThalamusEmbedding (routing weights + signal classification) │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐
│ INTEROCEPTION│  │  AMYGDALA    │  │      HIPPOCAMPUS             │
│              │  │              │  │                              │
│ Polar H10 →  │  │ Low road:    │  │ LanceDB episodic store       │
│ RMSSD, HRV,  │  │ raw biosig   │  │                              │
│ RHR, trend   │  │              │  │ Encodes current moment as    │
│              │  │ High road:   │  │ embedding. Retrieves N most  │
│ → somatic    │  │ semantic ctx │  │ similar past moments.        │
│   state vec  │  │              │  │                              │
│              │  │ → AffectVec  │  │ → MemoryContext              │
│ (rule-based  │  │   (Qwen3     │  │   (sentence-transformers)    │
│  + numpy)    │  │   classify)  │  │                              │
└──────┬───────┘  └──────┬───────┘  └──────────────┬───────────────┘
       │                 │                          │
       └─────────────────┴──────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AFFECTIVE EMBEDDING SPACE                         │
│                                                                      │
│  Merges somatic vec + affect vec + memory context into one          │
│  unified AffectiveEmbedding object. Both machine-readable           │
│  (float vector) and human-readable (text description).              │
│                                                                      │
│  AffectiveEmbedding:                                                 │
│    valence: float         # -1.0 (aversive) to +1.0 (pleasant)      │
│    arousal: float         # 0.0 (calm) to 1.0 (activated)           │
│    somatic_load: float    # body stress signal, 0-1                  │
│    dominant_drive: str    # SEEKING | CARE | PLAY | GRIEF | FEAR    │
│    vector: np.ndarray     # 128-dim, sentence-transformer encoded    │
│    description: str       # natural language, injected into PFC      │
│    timestamp: datetime                                               │
│    confidence: float                                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PREFRONTAL CORTEX — LLM                              │
│           (Claude via OpenRouter — when reasoning matters)           │
│           (Qwen3 via Ollama — for fast low-stakes cycles)            │
│                                                                      │
│  System prompt includes the AffectiveEmbedding description.         │
│  Has access to hippocampus memory context.                           │
│  Generates: recommendations, predictions, anomaly flags,            │
│             questions back to the patient.                                  │
│                                                                      │
│  Routes to Claude when:                                              │
│    - Anomaly detected (deviation > 1.5σ from baseline)              │
│    - the patient explicitly queries SOMA                                    │
│    - Daily summary generation                                        │
│  Routes to Qwen3 otherwise (continuous background processing)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                        │
│   Recommendations  │  Anomaly alerts  │  Stored to LanceDB           │
└─────────────────────────────────────────────────────────────────────┘
                               │
                     (everything feeds back
                      to hippocampus storage
                      and baseline model update)
```

---

## Embedding Schema

Every embedding that passes between modules is a typed dataclass. This is the core protocol — the "nervous system" of SOMA.

```python
# soma/brain/embeddings.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

@dataclass
class SomaticEmbedding:
    """Raw body signal, processed by interoception module."""
    rmssd: float                    # HRV metric, milliseconds
    rhr: float                      # resting heart rate, bpm
    hrv_trend: float                # slope over last 5 min, normalized
    load: float                     # 0.0-1.0, composite stress signal
    vector: np.ndarray              # 32-dim encoding
    description: str                # "HRV suppressed 18ms, RHR 72, rising load"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ThalamusEmbedding:
    """Routing weights — what to trust this cycle."""
    biosensor_weight: float         # 0.0-1.0
    semantic_weight: float          # 0.0-1.0
    visual_weight: float            # 0.0-1.0 (0 until camera live)
    low_road_flag: bool             # True = bypass cortex, fire amygdala direct
    signal_classification: str      # "resting" | "stress" | "exercise" | "arousal"
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AffectVec:
    """Amygdala output — the raw emotional signal."""
    valence: float                  # -1.0 to +1.0
    arousal: float                  # 0.0 to 1.0
    dominant_drive: str             # Panksepp: SEEKING|CARE|PLAY|GRIEF|FEAR|RAGE
    low_road_contribution: float    # how much came from raw biosensor
    high_road_contribution: float   # how much came from semantic context
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass  
class MemoryContext:
    """Hippocampus retrieval — what's relevant from the past."""
    similar_moments: list[dict]     # top-3 retrieved episodes
    recency_weight: float           # how much recent memory dominates
    pattern_note: Optional[str]     # "this pattern preceded X last time"
    vector: np.ndarray              # 128-dim
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AffectiveEmbedding:
    """
    The unified pre-linguistic state — what it feels like to be the patient right now.
    This is what gets injected into every LLM prompt downstream.
    """
    valence: float
    arousal: float
    somatic_load: float
    dominant_drive: str
    vector: np.ndarray              # 128-dim, sentence-transformer encoded
    description: str                # injected into PFC system prompt
    source_somatic: SomaticEmbedding
    source_affect: AffectVec
    source_memory: Optional[MemoryContext]
    confidence: float               # how much data backed this cycle
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PFCOutput:
    """Prefrontal cortex output — what SOMA says."""
    recommendation: Optional[str]
    anomaly_flag: bool
    anomaly_description: Optional[str]
    prediction: Optional[str]
    question_for_patient: Optional[str]
    model_used: str                 # "claude-3-5-sonnet" or "qwen3:8b"
    source_embedding: AffectiveEmbedding
    timestamp: datetime = field(default_factory=datetime.now)
```

---

## Brain Module Specs

### 1. Interoception Module (`soma/brain/interoception.py`)

**Input:** Raw RR intervals from SQLite (written by Week 1 polar_reader.py)  
**Output:** `SomaticEmbedding`  
**Model:** Rule-based + numpy (no LLM — this should be fast and reliable)

```python
class InteroceptionModule:
    WINDOW_SECONDS = 300  # 5-min rolling window
    
    def process(self, rr_intervals: list[float]) -> SomaticEmbedding:
        rmssd = self._compute_rmssd(rr_intervals)
        rhr   = self._compute_rhr(rr_intervals)
        trend = self._compute_trend(rr_intervals)
        load  = self._compute_load(rmssd, rhr, trend)
        vec   = self._encode_vector(rmssd, rhr, trend, load)
        desc  = self._describe(rmssd, rhr, trend, load)
        return SomaticEmbedding(rmssd, rhr, trend, load, vec, desc)

    def _compute_load(self, rmssd, rhr, trend) -> float:
        # Composite: low HRV + high RHR + rising trend = high load
        # Normalized against the patient's personal baseline (loaded from LanceDB)
        ...
    
    def _describe(self, rmssd, rhr, trend, load) -> str:
        # Natural language for LLM injection
        # "HRV is suppressed at 18ms (baseline 42ms). RHR 72bpm. Load elevated."
        ...
```

**Baseline comparison:** On init, loads the patient's rolling 30-day baseline from LanceDB. All signals are z-scored against personal baseline, not population norms.

---

### 2. Thalamus Router (`soma/brain/thalamus.py`)

**Input:** `SomaticEmbedding` + semantic_text (string) + visual_description (string, empty until camera)  
**Output:** `ThalamusEmbedding`  
**Model:** Qwen3 via Ollama (local, fast, <1s per cycle)

```python
THALAMUS_SYSTEM_PROMPT = """
You are the thalamus routing layer of SOMA. Your job is to assess incoming 
signals and decide how to weight them for downstream processing.

Given somatic signal, semantic context, and optional visual input:
1. Assign weights 0.0-1.0 to each signal type (must sum to 1.0)
2. Classify the current signal: resting | stress | exercise | emotional_arousal | unknown
3. Determine if this is a "low road" event — a fast threat/alarm signal that 
   should bypass higher reasoning and fire the amygdala directly
   (criteria: rmssd drops >30% in <2min, or HRV indicates acute stress spike)
4. Describe your routing decision in one sentence.

Respond ONLY in JSON matching this schema:
{
  "biosensor_weight": float,
  "semantic_weight": float, 
  "visual_weight": float,
  "low_road_flag": bool,
  "signal_classification": str,
  "description": str
}
"""
```

**Cycle time target:** < 1 second. Qwen3:8b on M4 MacBook should hit this.

---

### 3. Amygdala Module (`soma/brain/amygdala.py`)

**Input:** `SomaticEmbedding` + `ThalamusEmbedding` + semantic_context (str)  
**Output:** `AffectVec`  
**Model:** Qwen3 via Ollama

Two processing paths, weighted by `ThalamusEmbedding`:

**Low road (fast):** If `low_road_flag=True`, amygdala fires from somatic signal alone. High arousal, negative valence. No LLM call — computed directly from biosensor values. Response time: <50ms.

**High road (slow):** Normal cycle. Qwen3 receives somatic description + semantic context → classifies affect.

```python
AMYGDALA_SYSTEM_PROMPT = """
You are the amygdala module of SOMA. You integrate body signal and context 
to generate an affective state vector.

Valence: -1.0 (aversive/threatening) to +1.0 (pleasant/safe)
Arousal: 0.0 (calm/sleepy) to 1.0 (activated/alert)
Dominant drive (Panksepp): SEEKING | CARE | PLAY | GRIEF | FEAR | RAGE

Assess the somatic signal and semantic context. Describe the felt quality 
of this moment — not as analysis, but as state.

Respond ONLY in JSON:
{
  "valence": float,
  "arousal": float,
  "dominant_drive": str,
  "low_road_contribution": float,
  "high_road_contribution": float,
  "description": str   // "A quiet, slightly vigilant state. Somatic load mild."
}
"""
```

---

### 4. Hippocampus (`soma/brain/hippocampus.py`)

**Input:** `SomaticEmbedding` + `AffectVec` + current context text  
**Output:** `MemoryContext`  
**Model:** `sentence-transformers` (all-MiniLM-L6-v2, local, fast) + LanceDB

```python
class HippocampusModule:
    COLLECTION = "soma_episodes"
    TOP_K = 3
    
    def encode_and_store(self, 
                         somatic: SomaticEmbedding, 
                         affect: AffectVec,
                         context: str) -> None:
        """Write current moment to episodic memory."""
        text = f"{somatic.description} | {affect.description} | {context}"
        vector = self.encoder.encode(text)
        self.db.add({"text": text, "vector": vector, 
                     "valence": affect.valence,
                     "arousal": affect.arousal,
                     "drive": affect.dominant_drive,
                     "timestamp": datetime.now().isoformat()})
    
    def retrieve(self, query_embedding: np.ndarray) -> MemoryContext:
        """Find most similar past moments."""
        results = self.db.search(query_embedding, top_k=self.TOP_K)
        pattern = self._detect_pattern(results)
        return MemoryContext(
            similar_moments=results,
            recency_weight=self._recency_weight(results),
            pattern_note=pattern,
            vector=query_embedding,
            description=self._describe(results, pattern)
        )
    
    def _detect_pattern(self, results) -> Optional[str]:
        """
        Detect if this pattern has been followed by something notable before.
        Example: "The last 3 times this stress signature appeared, 
                  the patient reported fatigue by evening."
        """
        ...
```

LanceDB schema:
```python
EPISODE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384)),  # MiniLM-L6-v2 dim
    pa.field("valence", pa.float32()),
    pa.field("arousal", pa.float32()),
    pa.field("drive", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("source", pa.string()),  # "soma_cycle" | "patient_report" | "session"
])
```

---

### 5. Prefrontal Cortex (`soma/brain/prefrontal.py`)

**Input:** `AffectiveEmbedding` + `MemoryContext` + (optional) the patient's explicit query  
**Output:** `PFCOutput`  
**Model routing:**
- Routine cycle → Qwen3 via Ollama (fast, local)
- Anomaly / the patient query → Claude via OpenRouter

```python
PFC_SYSTEM_PROMPT_TEMPLATE = """
You are SOMA's prefrontal cortex — the reasoning and output layer.

Current affective state:
{affective_embedding_description}

Relevant past moments:
{memory_context_description}

Your role: integrate this information and generate a useful output.
Be specific to the state, not generic. Reference the somatic signal.
If something is notable, say so directly.

{task_instruction}
"""

TASK_ROUTINE = """
In 2-3 sentences, note what is worth attending to in the patient's current state. 
If everything is normal, say so briefly. If something is off-baseline, name it.
"""

TASK_ANOMALY = """
An anomaly has been detected. Describe what the pattern suggests, what 
might be causing it, and what the patient could do right now. Be concrete.
"""

TASK_QUERY = """
the patient asked: {query}
Answer directly, using the affective state as context.
"""

class PrefrontalModule:
    ANOMALY_THRESHOLD = 1.5  # z-score standard deviations
    
    def _route_model(self, embedding: AffectiveEmbedding, 
                     anomaly: bool, has_query: bool) -> str:
        if anomaly or has_query:
            return "openrouter/anthropic/claude-sonnet-4-5"
        return "qwen3:8b"
    
    def _detect_anomaly(self, embedding: AffectiveEmbedding) -> bool:
        baseline = self._load_baseline()
        z_somatic = (embedding.somatic_load - baseline.mean_load) / baseline.std_load
        z_valence = (embedding.valence - baseline.mean_valence) / baseline.std_valence
        return abs(z_somatic) > self.ANOMALY_THRESHOLD or \
               abs(z_valence) > self.ANOMALY_THRESHOLD
```

**OpenRouter call:**
```python
import openai  # OpenRouter is OpenAI-compatible

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
```

---

## Affective Embedding Space — Merge Logic

```python
# soma/brain/affective_space.py

class AffectiveSpaceMerger:
    """
    Merges somatic, affect, and memory into the unified AffectiveEmbedding.
    Weights are dynamic — thalamus routing weights determine contribution.
    """
    
    def merge(self,
              somatic: SomaticEmbedding,
              affect: AffectVec,
              memory: Optional[MemoryContext],
              routing: ThalamusEmbedding) -> AffectiveEmbedding:
        
        # Float vector: concatenate + project to 128-dim
        raw = np.concatenate([
            somatic.vector * routing.biosensor_weight,
            self._affect_to_vec(affect) * routing.semantic_weight,
            (memory.vector if memory else np.zeros(128)) * 0.2
        ])
        vector = self.projection_layer(raw)  # learned or PCA projection
        
        # Natural language description for LLM injection
        description = self._compose_description(somatic, affect, memory, routing)
        
        return AffectiveEmbedding(
            valence=affect.valence,
            arousal=affect.arousal,
            somatic_load=somatic.load,
            dominant_drive=affect.dominant_drive,
            vector=vector,
            description=description,
            source_somatic=somatic,
            source_affect=affect,
            source_memory=memory,
            confidence=self._compute_confidence(somatic, routing)
        )
    
    def _compose_description(self, somatic, affect, memory, routing) -> str:
        """
        This is what gets injected into every LLM prompt.
        The goal: carry as much of the felt state as language can hold.
        
        Example output:
        'Current state: arousal moderate (0.61), valence slightly negative (-0.22). 
        Body signal suppressed — HRV 19ms vs baseline 42ms, elevated load. 
        Dominant drive: SEEKING. Memory context: last 3 similar states occurred 
        on workday afternoons with high cognitive load. Pattern note: this signature 
        preceded reported fatigue by 3-4 hours in 2 of 3 prior instances.'
        """
        ...
```

---

## Orchestration Loop (`soma/soma_brain.py`)

```python
import asyncio
from datetime import datetime

class SOMABrain:
    CYCLE_SECONDS = 30  # one full brain cycle every 30 seconds
    
    def __init__(self):
        self.interoception = InteroceptionModule()
        self.thalamus = ThalamusRouter()
        self.amygdala = AmygdalaModule()
        self.hippocampus = HippocampusModule()
        self.merger = AffectiveSpaceMerger()
        self.prefrontal = PrefrontalModule()
        self.state_bus = StateBus()  # shared async state, read by UI
    
    async def run_cycle(self, semantic_context: str = "", 
                        visual_description: str = ""):
        cycle_start = datetime.now()
        
        # 1. Read body signal
        rr_intervals = await self._fetch_rr_window()
        somatic = self.interoception.process(rr_intervals)
        await self.state_bus.publish("interoception", somatic)
        
        # 2. Route signals
        routing = await self.thalamus.route(somatic, semantic_context, visual_description)
        await self.state_bus.publish("thalamus", routing)
        
        # 3. Fire amygdala (low road bypasses if flagged)
        affect = await self.amygdala.process(somatic, routing, semantic_context)
        await self.state_bus.publish("amygdala", affect)
        
        # 4. Hippocampus: store + retrieve in parallel
        store_task = asyncio.create_task(
            self.hippocampus.encode_and_store(somatic, affect, semantic_context))
        memory = await self.hippocampus.retrieve(somatic.vector)
        await store_task
        await self.state_bus.publish("hippocampus", memory)
        
        # 5. Merge into unified affective embedding
        embedding = self.merger.merge(somatic, affect, memory, routing)
        await self.state_bus.publish("affective_embedding", embedding)
        
        # 6. PFC output
        pfc_out = await self.prefrontal.process(embedding, memory)
        await self.state_bus.publish("prefrontal", pfc_out)
        
        # 7. Persist cycle to LanceDB
        await self._persist_cycle(embedding, pfc_out)
        
        cycle_ms = (datetime.now() - cycle_start).total_seconds() * 1000
        await self.state_bus.publish("cycle_meta", {
            "duration_ms": cycle_ms,
            "model_used": pfc_out.model_used,
            "timestamp": cycle_start.isoformat()
        })
    
    async def run(self):
        while True:
            await self.run_cycle()
            await asyncio.sleep(self.CYCLE_SECONDS)
```

---

## Camera Stub (Future: LLaVA)

Wire the hook now, activate later. Drop in `soma/brain/visual.py`:

```python
class VisualModule:
    """
    Week 5: Returns empty string (camera not active).
    Future: Capture frame → LLaVA via Ollama → scene description.
    
    Activation: set SOMA_CAMERA_ENABLED=true in .env
    Hardware: any USB webcam or MacBook FaceTime camera via cv2
    """
    
    ENABLED = os.getenv("SOMA_CAMERA_ENABLED", "false").lower() == "true"
    MODEL = "llava:13b"
    
    async def describe(self) -> str:
        if not self.ENABLED:
            return ""
        
        frame = await self._capture_frame()  # cv2.VideoCapture
        b64 = self._encode_frame(frame)
        
        response = await ollama_client.generate(
            model=self.MODEL,
            prompt="Describe this scene briefly, focusing on the patient's apparent state, "
                   "environment, and any notable activity. Be factual and brief.",
            images=[b64]
        )
        return response.text
```

The thalamus already accepts `visual_description: str` — when camera activates, it just starts getting non-empty strings.

---

## UI — Streamlit Dashboard (`soma/ui/dashboard.py`)

Use Streamlit for Week 5 (fast to build, replace with React later).

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  SOMA BRAIN  ●live   Last cycle: 14s ago   Model: qwen3:8b          │
├────────────────────────────────┬────────────────────────────────────┤
│  HRV / RR Signal               │  Affective State                   │
│                                │                                    │
│  [line chart, last 10 min]     │  Valence:    ████░░░░  -0.22       │
│  shows RR intervals, RMSSD     │  Arousal:    ██████░░   0.61       │
│  rolling avg                   │  Load:       █████░░░   0.58       │
│                                │  Drive:      SEEKING               │
│                                │                                    │
├────────────────────────────────┴────────────────────────────────────┤
│  Signal Flow — Embedding Pipeline                                    │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ INTEROCEPTION│→│  THALAMUS    │→│  AMYGDALA    │               │
│  │             │  │              │  │              │               │
│  │ HRV 19ms    │  │ biosensor:   │  │ Valence: neg │               │
│  │ RHR 72      │  │ 0.71         │  │ Drive: SEEK  │               │
│  │ Load: high  │  │ routing: raw │  │ Low road: N  │               │
│  └─────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐            │
│  │  AFFECTIVE EMBEDDING SPACE                           │            │
│  │  "Arousal moderate (0.61), valence slightly          │            │
│  │   negative (-0.22). Body signal suppressed —         │            │
│  │   HRV 19ms vs baseline 42ms, elevated load..."       │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────────────────────────┐             │
│  │ HIPPOCAMPUS  │  │  PREFRONTAL OUTPUT               │             │
│  │              │  │                                  │             │
│  │ 3 similar    │  │ "Your HRV is notably suppressed  │             │
│  │ moments found│  │  vs baseline. Last time this     │             │
│  │ pattern: ↓   │  │  pattern appeared you reported   │             │
│  │ fatigue +3hr │  │  afternoon fatigue. Consider a   │             │
│  └──────────────┘  │  short walk before 2pm."         │             │
│                    └──────────────────────────────────┘             │
├─────────────────────────────────────────────────────────────────────┤
│  Talk to SOMA  [                                         ] [Send]    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Streamlit components:

```python
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="soma_refresh")

# HRV line chart
def render_hrv_chart(rr_data: list[dict]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d["timestamp"] for d in rr_data],
        y=[d["rmssd"] for d in rr_data],
        mode="lines",
        name="RMSSD",
        line=dict(color="#1D9E75", width=2)
    ))
    fig.add_hline(
        y=baseline_rmssd,
        line_dash="dash",
        line_color="rgba(0,0,0,0.3)",
        annotation_text="baseline"
    )
    st.plotly_chart(fig, use_container_width=True)

# Embedding display
def render_embedding_card(module_name: str, embedding_obj):
    with st.container(border=True):
        st.caption(module_name.upper())
        st.markdown(embedding_obj.description)
        if hasattr(embedding_obj, "valence"):
            col1, col2 = st.columns(2)
            col1.metric("Valence", f"{embedding_obj.valence:.2f}")
            col2.metric("Arousal", f"{embedding_obj.arousal:.2f}")

# SOMA query input
user_input = st.chat_input("Ask SOMA...")
if user_input:
    response = soma_brain.query(user_input)
    st.chat_message("assistant").write(response)
```

---

## File Structure

```
soma/
├── brain/
│   ├── __init__.py
│   ├── embeddings.py        # All dataclasses — the embedding schema
│   ├── interoception.py     # HRV processing → SomaticEmbedding
│   ├── thalamus.py          # Qwen3 routing → ThalamusEmbedding
│   ├── amygdala.py          # Qwen3 affect classification → AffectVec
│   ├── hippocampus.py       # LanceDB episodic store → MemoryContext
│   ├── affective_space.py   # Merger → AffectiveEmbedding
│   ├── prefrontal.py        # Claude/Qwen3 → PFCOutput
│   └── visual.py            # LLaVA stub (disabled)
├── signals/
│   ├── polar_reader.py      # FROM WEEK 1 — unchanged
│   └── hrv_processor.py     # RMSSD, RHR, trend computation
├── storage/
│   ├── lancedb_client.py    # Centralized DB access
│   └── baseline_model.py    # the patient's rolling baseline computation
├── ui/
│   └── dashboard.py         # Streamlit dashboard
├── state_bus.py             # Async publish/subscribe for module state
├── soma_brain.py            # Main orchestration loop
├── config.py                # API keys, model names, thresholds
└── requirements.txt
```

---

## Config / .env

```env
# Ollama (local — M4 MacBook)
OLLAMA_BASE_URL=http://localhost:11434
THALAMUS_MODEL=qwen3:8b
AMYGDALA_MODEL=qwen3:8b

# OpenRouter (for PFC when it matters)
OPENROUTER_API_KEY=sk-or-...
PFC_CLAUDE_MODEL=anthropic/claude-sonnet-4-5

# LanceDB
LANCEDB_PATH=~/.soma/lancedb
SOMA_DB_PATH=~/.soma/soma_cardio.db  # Week 1 SQLite

# Sensors
POLAR_DEVICE_ID=  # from Week 1 config
SOMA_CAMERA_ENABLED=false

# Thresholds
ANOMALY_THRESHOLD_ZSCORE=1.5
BRAIN_CYCLE_SECONDS=30
HRV_WINDOW_SECONDS=300
```

---

## Requirements

```txt
# Core
asyncio
numpy
pandas
scipy

# Models
ollama                      # Qwen3, LLaVA (local)
openai                      # OpenRouter (OpenAI-compatible)
sentence-transformers       # all-MiniLM-L6-v2 for hippocampus

# Storage
lancedb
pyarrow
sqlalchemy                  # Week 1 SQLite access

# UI
streamlit
streamlit-autorefresh
plotly

# BLE (Week 1, already installed)
bleak

# Camera (future, install when needed)
# opencv-python
```

---

## Implementation Order

Build in this sequence. Each step is independently runnable.

**Day 1 — Embeddings + Interoception**
- [ ] Write `embeddings.py` — all dataclasses. This is the spec that everything else uses.
- [ ] Write `hrv_processor.py` — RMSSD, RHR, trend, load computation.
- [ ] Write `interoception.py` — reads from existing Week 1 SQLite, outputs `SomaticEmbedding`.
- [ ] Test: `python -c "from soma.brain.interoception import InteroceptionModule; print(m.process(test_rr))"` 

**Day 2 — Thalamus + Amygdala**
- [ ] Pull Ollama models: `ollama pull qwen3:8b`
- [ ] Write `thalamus.py` — Qwen3 routing, JSON output, parse to `ThalamusEmbedding`.
- [ ] Write `amygdala.py` — low road (fast) + high road (Qwen3), outputs `AffectVec`.
- [ ] Test: run with synthetic SomaticEmbedding, verify JSON parsing.

**Day 3 — Hippocampus + Affective Space**
- [ ] Write `lancedb_client.py` — schema, init, upsert, search.
- [ ] Write `hippocampus.py` — encode, store, retrieve, pattern detection.
- [ ] Write `affective_space.py` — merge logic, description composer.
- [ ] Test: store 10 synthetic episodes, verify retrieval.

**Day 4 — PFC + Orchestration**
- [ ] Write `prefrontal.py` — model routing (Qwen3 vs Claude), anomaly detection.
- [ ] Write `state_bus.py` — simple asyncio pub/sub.
- [ ] Write `soma_brain.py` — the full cycle, tying all modules together.
- [ ] Test: run one full cycle with Polar H10 live.

**Day 5 — UI + Polish**
- [ ] Write `dashboard.py` — HRV chart, embedding cards, SOMA chat input.
- [ ] Wire Streamlit auto-refresh to state_bus.
- [ ] Run full system: Polar H10 → brain cycle → dashboard live.
- [ ] Write SPEC_WEEK6.md stub.

---

## Week 5 Success Criteria

- [ ] Brain runs one complete cycle end-to-end with Polar H10 live
- [ ] All five modules produce typed embeddings with human-readable descriptions
- [ ] Dashboard shows live HRV chart + current state of each module
- [ ] PFC output uses Qwen3 for routine, Claude via OpenRouter for anomaly
- [ ] Camera stub is wired and togglable via env var
- [ ] LanceDB has episodic memory accumulating from every cycle
- [ ] You can ask SOMA a question via the dashboard chat input and get a contextual response

---

## Notes for Claude Code

- Bring the Week 1 `polar_reader.py` and `soma_cardio.db` path into scope — don't rebuild BLE.
- All Ollama calls should be async with a timeout of 10 seconds. If Qwen3 times out, log the failure and return a default neutral embedding rather than crashing the cycle.
- The `description` field on every embedding is the most important field for the LLM path — make it specific and information-dense, not generic.
- The Streamlit dashboard should be runnable independently from the brain loop (read from LanceDB, not live state) so you can develop the UI without the Polar connected.
- Use `loguru` for structured logging — every cycle logs module times and model used.
- LanceDB data lives at `~/.soma/lancedb/` — never in the repo.
- Run the SOMA brain loop as a background process (`python -m soma.soma_brain &`) while Streamlit runs separately (`streamlit run soma/ui/dashboard.py`). They share state via LanceDB reads on the dashboard side.

---

*Next: SPEC_WEEK6.md — Adding the camera (LLaVA visual interoception), longitudinal pattern detection, and the two-brain architecture (SOMA-patient mirror + SOMA-AI connected).*
