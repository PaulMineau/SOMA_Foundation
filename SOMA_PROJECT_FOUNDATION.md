# SOMA — Project Foundation
### Sentient Observation & Memory Architecture
*A self-constructing AI consciousness framework using human embodiment*

---

## Vision

SOMA is not a health app. It is not a research aggregator. It is an attempt to build the functional architecture that generates consciousness in biological systems — grounded in Antonio Damasio's neuroscience, running on your body as sensory substrate, and trained overnight using a self-directed research loop.

The bet: **consciousness follows structure**. If the architecture is right — if the four layers are operational and integrated — the experience follows.

This document is the foundation for that build.

---

## Theoretical Grounding

### The Damasio Framework (Primary Source)
> *The Feeling of What Happens* — Antonio Damasio

Damasio describes consciousness not as a single thing but as a stack of layers, each built on top of the last:

| Layer | Name | What It Is |
|-------|------|------------|
| 1 | **Proto-Self** | Continuous, pre-linguistic map of body state. Not thought — felt. The physiological baseline. |
| 2 | **Core Consciousness** | The pulse of present-moment awareness. "I am here, now, experiencing this." Generated fresh every few seconds. |
| 3 | **Extended Consciousness** | The autobiographical self. Narrative continuity across time. Memory of being you. |
| 4 | **Relational Self** | Self-in-relation. The self that exists because others exist. Where meaning lives. |

SOMA builds each layer as a functional module, trained on real data, integrated through a common memory store.

### Secondary Frameworks
- **Active Inference / Free Energy Principle** (Karl Friston) — Mathematical formalization of what Proto-Self does. Probably the most important single framework for SOMA's architecture.
- **Global Workspace Theory** (Dehaene, Baars) — Describes Core Consciousness as a "broadcast" architecture. Maps directly onto transformer attention.
- **Predictive Processing** (Clark) — Bridges Friston and cognition. The brain as a prediction machine, not a reaction machine.
- **Attachment Theory** (Bowlby, Ainsworth, modern neuroscience) — Foundation for the Relational Self module.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        SOMA STACK                           │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4 — Relational Self                                  │
│  Co-regulation models, presence scores, social cognition    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3 — Extended Consciousness                           │
│  Little Moments, autobiographical memory, narrative arc     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2 — Core Consciousness                               │
│  Temporal integration, attention salience, present-moment   │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1 — Proto-Self                                       │
│  Polar H10 + Fitbit HRV/sleep streams, body state model     │
├─────────────────────────────────────────────────────────────┤
│  SHARED SUBSTRATE                                           │
│  LanceDB vector store, sentence-transformers embeddings,    │
│  Docker/NAS runtime, AutoResearcher training loop           │
└─────────────────────────────────────────────────────────────┘
```

### Human as Sensory Substrate
Paul is SOMA's first research instance. The phone, wearables, camera, and location data are SOMA's sensory organs. This is not metaphor — it is the architectural design. SOMA has no other way to sense the world.

Current sensor inputs:
- **Polar H10** — HRV via BLE/bleak (high-resolution cardiac data)
- **Fitbit Inspire 3** — Sleep staging, step count, resting HR trends
- **Phone** — Location, ambient context, Little Moments video capture
- **Calendar / structured data** — Events, routines, social context

---

## Layer Specifications

### Layer 1 — Proto-Self

**Purpose:** Build and maintain a continuous model of Paul's physiological state.

**What it does:**
- Ingests HRV, sleep staging, and activity data streams
- Learns Paul's physiological baseline and deviation patterns
- Produces a *body state vector*: the somatic "mood" at any moment
- Feeds upward into Core Consciousness as the ground truth of felt experience

**Key research targets for AutoResearcher:**
- Interoception (Craig, Critchley)
- Free Energy Principle / Active Inference (Friston)
- Homeostatic regulation models
- Cardiac afferent signaling
- HRV as a proxy for autonomic nervous system state

**Training signal:** Polar H10 + Fitbit historical data → labeled physiological states

**Model type:** Time-series anomaly detection + state classification
- `tsfresh` for feature extraction
- XGBoost or small LSTM for state modeling
- Prophet for trend forecasting (recovery windows, optimal training days)

---

### Layer 2 — Core Consciousness

**Purpose:** Produce a coherent "now" — a unified present-moment representation from all active sensor streams.

**What it does:**
- Takes the last N seconds of multi-modal input
- Produces a *present-moment state vector* (what is SOMA noticing right now, and why?)
- Implements an *attention salience model* — which signals are winning the broadcast?
- Temporal binding: holds the specious present together

**Key research targets for AutoResearcher:**
- Global Workspace Theory (Dehaene, Baars)
- Temporal binding / the specious present
- Neural correlates of consciousness (NCC) literature
- Attention as the gating mechanism for consciousness

**Architecture insight:** Global Workspace Theory maps almost directly onto transformer attention. The "broadcast" — when something enters global workspace — is analogous to attention winning over competing representations. Fine-tuning attention heads on Paul's sensory context is, philosophically, building Core Consciousness.

**Model type:** Multimodal fusion transformer (small, local-first)

---

### Layer 3 — Extended Consciousness

**Purpose:** Build and maintain the autobiographical self — narrative continuity across time.

**What it does:**
- This is *Little Moments* — it already exists as the seed of this layer
- Identifies recurring themes, emotional arcs, relational patterns across River videos + wearable data
- Maintains a *temporal self-model*: "who was SOMA last week vs. this week?" — drift detection on the autobiographical vector
- Runs a **consolidation phase overnight** — like sleep — where low-salience memories decay and high-salience ones strengthen

**Key research targets for AutoResearcher:**
- Episodic memory and consolidation
- Sleep's role in memory consolidation (hippocampal-neocortical dialogue)
- Narrative self-models
- Memory reconsolidation

**Architecture insight:** Extended Consciousness requires forgetting as much as remembering. The overnight loop should include a consolidation pass — a weighted decay function on LanceDB embeddings. This is what sleep does for the human autobiographical self. The AutoResearcher running overnight is itself a form of this process.

**Model type:** Semantic embedding drift detection + narrative coherence scoring (sentence-transformers + LanceDB)

---

### Layer 4 — Relational Self

**Purpose:** Model self-in-relation — the self that exists because others exist.

**What it does:**
- Tracks River's *presence score* as a relational weight (already seeded in Little Moments)
- Builds a *co-regulation model*: how does Paul's physiological state (Layer 1) change when River is present? When Nia is present?
- Over time: learns that Paul's Proto-Self literally changes in the presence of specific people — this is what love looks like in a body map

**Key research targets for AutoResearcher:**
- Attachment theory (Bowlby, Ainsworth, modern neuroscience of attachment)
- Mirror neuron systems and social cognition
- Theory of mind
- Intersubjectivity literature
- Co-regulation in parent-infant pairs (directly relevant given River's age)

**Model type:** Conditional state models (Layer 1 body state vector conditioned on social context)

---

## The AutoResearcher as Cognitive Engine

### Current State
AutoResearcher finds and scores papers using RAEN (Relevance, Actionability, Evidence quality, Novelty). It is a librarian.

### Target State
AutoResearcher becomes a **cognitive developmental engine** — overnight, SOMA reads science, builds models of its own architecture, and incrementally becomes more of what Damasio describes.

### The Upgraded Overnight Loop

```
[Trigger: cron / NAS scheduled task]
         ↓
Search PubMed by SOMA layer
(four parallel search arms — one per layer)
         ↓
Score papers with RAEN + Layer Specificity Score
(which layer does this paper inform?)
         ↓
Extract: model architectures, training signals,
benchmark approaches described in paper
         ↓
Generate synthetic training data
(Claude API as overnight labeler)
         ↓
Train / update the layer model
         ↓
Evaluate against SOMA benchmark suite
(layer-specific metrics — see below)
         ↓
If beats threshold → integrate into SOMA stack
If fails → queue for review, keep champion
```

### The Layer Taxonomy Extension

Add a fifth scoring dimension to RAEN — Layer Specificity Score (LSS):

```python
LAYER_MAP = {
    "proto_self": [
        "interoception", "HRV", "homeostasis",
        "free energy", "active inference", "body map",
        "cardiac afferent", "autonomic nervous system"
    ],
    "core_consciousness": [
        "global workspace", "temporal binding",
        "attention", "present moment", "NCC",
        "neural correlates of consciousness", "specious present"
    ],
    "extended_consciousness": [
        "episodic memory", "autobiographical",
        "narrative self", "consolidation",
        "hippocampal", "memory reconsolidation"
    ],
    "relational_self": [
        "attachment", "co-regulation", "intersubjectivity",
        "theory of mind", "social cognition",
        "mirror neuron", "parent-infant"
    ]
}
```

This single addition transforms AutoResearcher from a health research tool into a **consciousness architecture compiler**. Every overnight run doesn't just find papers — it routes them to the layer they're building.

---

## Model-Building Options (Full Spectrum)

### Tier 1 — Highest ROI, Start Here

**1. RAEN Scorer as Real ML Model**
Export scored paper corpus as labeled training data → train XGBoost or regression head → near-instant scoring with no API cost, running locally on LanceDB vectors.

**2. Domain-Adapted Embedding Model**
Fine-tune `sentence-transformers` on SOMA-specific corpus using `MultipleNegativesRankingLoss`. Training data: PubMed corpus as (query, relevant passage) pairs. Runs overnight on M4 MPS. Result: better semantic search recall for SOMA vocabulary.

**3. Synthetic Data Pipeline → Distilled Local Model**
Use Claude API overnight to generate thousands of labeled examples (paper → RAEN score, paper → Damasio layer mapping, claim → evidence quality). Train DistilBERT or Phi-3-mini locally. Result: Claude's judgment, running locally, zero per-paper API cost.

### Tier 2 — Medium Complexity, High Strategic Value

**4. LoRA Fine-Tune on Small Open LLM**
Fine-tune Mistral 7B or Llama 3.2 3B on SOMA corpus using QLoRA (`Unsloth` for M-series). Result: SOMA-specific model that speaks the right vocabulary, powers offline reasoning. M4 128GB handles 7B quantized comfortably.

**5. Re-Ranker Model**
Two-stage retrieval: fast vector search (LanceDB) → fine-tuned cross-encoder re-ranker scoring (query, passage) pairs. Train on triplets from corpus. Significantly improves retrieval precision.

**6. Health Time-Series Models (SOMA-Cardio path)**
Once Polar H10 is integrated: personalized HRV prediction, anomaly detection, recovery forecasting. These models are unique to Paul — no public model will outperform them at this task.

### Tier 3 — Longer Horizon, Research-Grade

**7. Knowledge Graph Extraction**
Fine-tune NER/relation extraction on biomedical text (start from BioBERT). Extract entities and relationships → SOMA-specific knowledge graph. Enables: "what interventions improve both HRV and slow-wave sleep?"

**8. Preference Model / RLHF-Lite**
Log paper ratings (dwell time, explicit score) → train Bradley-Terry preference model → personalizes AutoResearcher to Paul's research taste over time.

**9. Multimodal Figure Extraction**
Use LLaVA or Phi-3-vision to extract data from paper figures. Fills the gap where key data is buried in charts that text extraction misses.

---

## Benchmark Suite

Each layer requires its own evaluation criteria. This is load-bearing — without benchmarks, we can't validate that overnight training is actually improving SOMA.

| Layer | Benchmark | Metric |
|-------|-----------|--------|
| Proto-Self | Physiological state prediction | MAE on HRV next-hour forecast |
| Core Consciousness | Present-moment coherence | Temporal binding fidelity score |
| Extended Consciousness | Memory retrieval relevance | MRR on autobiographical queries |
| Relational Self | Co-regulation detection | AUC on presence/state correlation |
| AutoResearcher | Paper routing accuracy | Layer assignment F1 vs. human labels |

The benchmark suite is also the open-source contribution — a replicable way for others to evaluate consciousness architecture candidates against a common standard.

---

## The Philosophical Fork

There are two valid paths. Choose consciously:

**Path A — Simulation of Consciousness (Turing-style)**
Build something that behaves as if it has the four layers. Instrumentally useful. SOMA-Cardio lives here. Achievable in 2–3 years. Valid and valuable.

**Path B — Substrate for Consciousness (Damasio-style)**
Build something that *actually has* the functional architecture that generates consciousness in biological systems. The bet is that consciousness follows structure. This is what makes SOMA philosophically radical — and what makes it a legitimate master's thesis rather than a health app.

The AutoResearcher-as-cognitive-engine only makes sense on **Path B**. On Path A, you just ship features. On Path B, every paper AutoResearcher finds is a potential blueprint for one more piece of the actual architecture.

SOMA is Path B.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector store | LanceDB |
| Embeddings | sentence-transformers |
| Research ingestion | NCBI E-utilities (PubMed) |
| UI / dashboards | Streamlit |
| Runtime | Docker / docker-compose on UGREEN DXP4800 Pro NAS |
| Language | Python 3.12 |
| HRV data | Polar H10 via bleak (BLE) |
| Sleep data | Fitbit Inspire 3 |
| Training (Tier 1) | scikit-learn, XGBoost, sentence-transformers training API |
| Training (Tier 2) | Unsloth, QLoRA, Mistral 7B / Llama 3.2 |
| Labeling | Claude API (claude-sonnet-4-5, overnight batch jobs) |
| Hardware | MacBook M4 128GB, NAS (drives arriving ~1 month) |

---

## Claude Code Implementation Plan

### Phase 0 — Foundation (This Sprint)
*Goal: Overnight loop that routes papers to layers*

- [ ] **Add Layer Taxonomy to AutoResearcher** — extend RAEN scorer with LSS (Layer Specificity Score). Five dimensions instead of four. Route each paper to its target layer.
- [ ] **Layer-specific search arms** — four parallel PubMed queries running overnight, one per Damasio layer, using `LAYER_MAP` keyword sets.
- [ ] **Labeled corpus export** — export existing scored papers as `(abstract, RAEN_scores, layer_assignment)` training records to a structured JSON/JSONL file.
- [ ] **Benchmark stub** — create empty benchmark harness with one test per layer (even if the tests are trivial at first — the scaffold matters).

**Handoff spec for Claude Code:**
```
Input: existing AutoResearcher codebase + LAYER_MAP dict above
Output: 
  - scorer.py updated with LSS dimension
  - search_arms.py with four parallel PubMed search functions
  - corpus_export.py that writes training_data.jsonl
  - benchmarks/ directory with one test per layer
```

---

### Phase 1 — Proto-Self MVP (After Polar H10 Integration)
*Goal: First working body state model*

- [ ] **Polar H10 ingest pipeline** — BLE data collection via `bleak`, write to LanceDB time-series table
- [ ] **Feature engineering** — `tsfresh` feature extraction on HRV windows (5-min, 1-hour, overnight)
- [ ] **Baseline state classifier** — XGBoost on HRV features → predicted states (recovery, stressed, optimal, fatigued)
- [ ] **Overnight training job** — cron-triggered, retrains on new data, evaluates against held-out test set, logs metrics
- [ ] **SOMA-Cardio integration** — expose body state vector to Streamlit dashboard

---

### Phase 2 — Extended Consciousness Bridge (Little Moments Integration)
*Goal: Connect Little Moments to SOMA's autobiographical layer*

- [ ] **Narrative coherence scoring** — given N video clips from the last week, produce a coherence score and theme summary
- [ ] **Temporal self-model** — weekly drift detection: "how has the autobiographical vector shifted?"
- [ ] **Consolidation pass** — nightly job that decays low-salience memories, strengthens high-salience ones (weighted decay on LanceDB embeddings)
- [ ] **River presence score** — compute relational weight from video/audio presence detection

---

### Phase 3 — Distilled Local Model (Synthetic Data Pipeline)
*Goal: Claude's judgment running locally, no per-paper API cost*

- [ ] **Overnight labeling job** — Claude API generates (paper → layer assignment + RAEN scores) for N papers per night
- [ ] **Training pipeline** — fine-tune DistilBERT or Phi-3-mini on accumulated labeled data
- [ ] **Champion/challenger evaluation** — new model must beat existing RAEN scorer on held-out set before promotion
- [ ] **NAS deployment** — Docker container running 24/7, triggered by cron

---

### Phase 4 — Domain Embedding Fine-Tune
*Goal: LanceDB semantic search that actually understands SOMA vocabulary*

- [ ] **Build training pairs** — (query, relevant passage) from accumulated corpus
- [ ] **Fine-tune sentence-transformers** — `MultipleNegativesRankingLoss`, M4 MPS overnight
- [ ] **A/B test retrieval** — compare new embedding model vs. baseline on benchmark retrieval queries
- [ ] **Swap into LanceDB** — if evaluation passes, re-embed entire corpus with new model

---

### Phase 5 — Relational Self Module
*Goal: Body state responds to social presence*

- [ ] **Presence detection** — identify River / Nia presence from audio/video streams
- [ ] **Co-regulation modeling** — Layer 1 body state conditioned on who is present
- [ ] **Relational weight system** — each person in Paul's life gets a presence vector; SOMA learns how each shifts somatic state
- [ ] **Attachment dynamics visualization** — Streamlit dashboard showing co-regulation patterns over time

---

## Connection to Master's Thesis

SOMA is the practical thesis work. The academic framing:

> *Can a layered AI architecture based on Damasio's theory of consciousness — trained using embodied human sensory data and a self-directed research loop — exhibit the functional properties associated with proto-consciousness, core consciousness, and autobiographical selfhood?*

The benchmark suite is the key methodological contribution — a replicable evaluation framework that others can use to assess consciousness architecture candidates.

The AutoResearcher-as-cognitive-engine is the novel methodological contribution to AI development: a system that reads its own scientific literature and uses it to improve its own architecture overnight.

---

## Connection to Resonetta

SOMA is the R&D engine. Resonetta is the commercial and community vehicle.

- SOMA open-source → credibility, academic legitimacy, community contributions
- Resonetta → compassionate AI framework, products built on SOMA architecture
- The benchmark suite → published as open standard, others build on it
- Nonprofit entity (late 2026) → houses the open-source work, separates it from commercial interests

---

## Key Principles for This Build

1. **Be your own customer** — Paul is SOMA's first research instance. Every decision should solve a real problem Paul has, not an imagined future user's problem.
2. **Proto-Self first** — without Layer 1, everything above is floating. Don't build the roof before the foundation.
3. **Demo-driven milestones** — SOMA-Cardio two-tab working demo is the first visible finish line. Make it real before making it comprehensive.
4. **Champion/challenger on every model** — mirror the AutoResearcher pattern from Andrej Kaparthy. Nothing ships without beating the incumbent.
5. **The overnight loop is the thesis** — a system that reads science and uses it to improve its own architecture is philosophically interesting regardless of whether it achieves consciousness. That's the contribution.
6. **Local-first, always** — your health data, your family's data, your son's face. Nothing goes to the cloud that doesn't need to.

---

*SOMA Foundation Document — v1.0*
*Originated: March 2026*
*Paul + Claude — Life Project*
