# SOMA Project Status — April 16, 2026

**Repo:** https://github.com/PaulMineau/SOMA_Foundation  
**Stack:** Python 3.12, SQLite, LanceDB, FastAPI, Streamlit, bleak (BLE), OpenRouter (Claude), Ollama (Qwen3), Fitbit API, ResMed myAir API, pyedflib  
**Tests:** 115 passing  
**Modules:** 75 Python files across 7 packages  
**Commits:** 17 on main  

---

## What SOMA Is

SOMA (Sentient Observation & Memory Architecture) is a layered AI consciousness framework built on Antonio Damasio's neuroscience, using human embodiment as sensory substrate. It reads physiological signals from multiple wearables, builds a model of the wearer's baseline, detects when something changes, simulates a brain-like information flow through five neural modules, and makes recommendations grounded in actual nervous system state.

The theoretical foundation is in `SOMA_PROJECT_FOUNDATION.md` at the repo root.

---

## Architecture Overview

```
INPUT LAYER
├── Polar H10 BLE        (live RR intervals, HRV on demand)
├── Fitbit Inspire 3     (overnight HRV, sleep stages, SpO2, steps)
├── ResMed CPAP myAir    (daily AHI, leak, usage, sleep score)
├── ResMed SD card EDF   (time-resolved apnea events)
├── PubMed + S2          (research papers for AutoResearcher)
├── RSS newsletters      (Substack agent)
└── Camera stub          (LLaVA slot, disabled)
        │
        ▼
SIGNAL PROCESSING
├── Artifact filter     (range + ectopic rejection)
├── HRV computation     (RMSSD, SDNN, pNN50)
├── Baseline model      (z-scored personal normal)
└── Anomaly detection   (1.5 std dev deviation)
        │
        ▼
BRAIN SIMULATION (5 modules, 30s cycles)
├── Interoception      (HRV -> SomaticEmbedding)
├── Thalamus           (Qwen3 routing)
├── Amygdala           (affect, low road + high road)
├── Hippocampus        (LanceDB episodic memory)
└── Prefrontal Cortex  (Qwen3 routine / Claude anomaly)
        │
        ▼
UNIFIED STATE
├── State classifier   (depleted/recovering/baseline/restored/peak)
├── Affective embedding (128-dim vector + natural language)
└── Autobiographical memory (probe/response exchanges)
        │
        ▼
OUTPUTS
├── Recommendations     (state-aware from curated corpus)
├── Conversational probe (anomaly -> question -> memory)
├── Reading queue       (state-matched Substack articles)
├── Weekly narrative    (accumulated memory synthesis)
├── Correlations        (CPAP AHI vs next-day HRV)
└── Dashboard           (unified Streamlit view)
```

---

## Complete Module Inventory

### Phase 0 — AutoResearcher (soma/autoresearcher/, 12 modules)

Overnight research loop that reads science and routes it to Damasio layers.

| Module | What it does |
|---|---|
| `run.py` | CLI + dual-track overnight loop (health RAEN + architecture LSS) |
| `scorer.py` | RAEN (Relevance x Actionability x Evidence x Novelty) + LSS |
| `search_arms.py` | 4 parallel PubMed arms, one per Damasio layer |
| `damasio.py` | 3-tier layer classification: LSS -> keyword -> LLM |
| `fetcher.py` | Async PubMed E-utilities + Semantic Scholar with retry |
| `extractor.py` | LLM structured extraction (intervention, effect, safety) |
| `query_gen.py` | LLM generates MeSH queries from biomarker profile |
| `synthesizer.py` | LLM generates ranked briefing |
| `memory.py` | LanceDB vector store for research findings |
| `corpus_export.py` | Export scored papers to JSONL training data |
| `convergence.py` | Loop termination logic |
| `seed.py` | BiomarkerProfile dataclass, JSON loader |
| `llm.py` | OpenRouter client |

### Phase 1 — Proto-Self: Body Signal Acquisition

**Polar H10 (soma/proto_self/)**

| Module | What it does |
|---|---|
| `polar_reader.py` | BLE connection via bleak, HR Measurement parsing, macOS retry logic |
| `hrv.py` | RMSSD, SDNN, pNN50, artifact rejection, body state classification |
| `polar_logger.py` | Continuous labeled session logging to SQLite + CSV |
| `db.py` | SQLite schema (sessions, rr_intervals, anomalies, recommendations) |
| `storage.py` | LanceDB time-series tables for RR + HRV |
| `monitor.py` | Live terminal readout |
| `run.py` | Quick fixed-duration collection |

**Fitbit (soma/proto_self/fitbit/)**

| Module | What it does |
|---|---|
| `fitbit_auth.py` | OAuth2 flow, tokens in ~/.fitbit_tokens.json |
| `fitbit_client.py` | API wrapper with auto refresh, HRV from dailyRmssd field |
| `soma_fitbit_ingestor.py` | Daily ingestion, narrative generation, recovery scoring, LanceDB |
| `fitbit_dashboard.py` | Data access layer for dashboard + state classifier |
| `soma_daily_context.py` | LLM-ready context generator |
| `daily_sync.py` | Cron script for 6am sync of previous day |

**CPAP (soma/proto_self/cpap/) — NEW**

| Module | What it does |
|---|---|
| `myair_client.py` | ResMed myAir API via Okta OAuth2 + PKCE, GraphQL daily summaries |
| `edf_parser.py` | SD card EDF file parser for time-resolved apnea events |
| `cpap_ingestor.py` | Normalize + embed + store in LanceDB (daily + events tables) |
| `correlator.py` | Pearson correlations AHI vs next-day Fitbit recovery/HRV/RHR/SpO2 |
| `daily_sync.py` | Cron script for 6:30am myAir poll |

### Phase 1b — Core Consciousness: Baseline + Anomaly

| Module | What it does |
|---|---|
| `artifact_filter.py` | Range + ectopic RR cleaning |
| `baseline_model.py` | Compute physiological fingerprint from morning_baseline sessions |
| `anomaly_detector.py` | Polls SQLite, flags deviations > 1.5 std dev |
| `dashboard.py` | **Unified Streamlit dashboard** (see below) |

### Phase 2 — SOMA Core Server

| Module | What it does |
|---|---|
| `soma_server.py` | FastAPI endpoints: /ingest/rr, /status, /baseline, /anomalies, /probe, /sessions |

### Phase 3 — Extended Consciousness: Recommendations

| Module | What it does |
|---|---|
| `state_classifier.py` | Classifies current state from HRV + Fitbit + CPAP context |
| `recommender.py` | State-aware recommendations from curated corpus |
| `feedback_logger.py` | Interactive CLI to close the feedback loop |
| `soma_memory.py` | LanceDB embedding store for (state, recommendation, outcome) |
| `soma_profile.py` | Rich patient context assembler for research prompts |

### Phase 3b — Daily Research Agent

| Module | What it does |
|---|---|
| `research_agent.py` | Claude via OpenRouter, rotates 10 topics, stages candidates |
| `raen_scorer.py` | Score candidates on RAEN with fuzzy tag matching |
| `corpus_review.py` | CLI approve/reject before merge to live corpus |

### Phase 3c — Substack Agent

| Module | What it does |
|---|---|
| `substack_agent.py` | RSS fetch, article classification, RAEN scoring, newsletter discovery |
| `article_review.py` | CLI for article approval + state-filtered reading list |

### Phase 4 — Conversational Probe

| Module | What it does |
|---|---|
| `autobiographical_store.py` | LanceDB schema for probe/response memories + weekly narratives |
| `probe_generator.py` | Anomaly + past memories -> intelligent natural language probe |
| `memory_writer.py` | Entity extraction + valence estimation, stores exchange |
| `probe_interface.py` | CLI conversation terminal |
| `narrative_builder.py` | Weekly autobiographical synthesis |

### Phase 5 — Brain Simulation Layer (soma/brain/) — NEW

Five async brain modules with typed embedding protocol. Runs 30-second cycles.

| Module | What it does |
|---|---|
| `embeddings.py` | SomaticEmbedding, ThalamusEmbedding, AffectVec, MemoryContext, AffectiveEmbedding, PFCOutput |
| `interoception.py` | RR intervals -> body state vector (rule-based, fast) |
| `thalamus.py` | Signal routing via Qwen3/Ollama with rule-based fallback |
| `amygdala.py` | Affect classification, low road (fast) + high road (LLM) |
| `hippocampus.py` | LanceDB episodic memory, encode + retrieve + pattern detect |
| `affective_space.py` | Unified 128-dim AffectiveEmbedding with NL description |
| `prefrontal.py` | Routes to Qwen3 (routine) or Claude (anomaly/query) |
| `visual.py` | LLaVA stub toggled via SOMA_CAMERA_ENABLED |
| `state_bus.py` | Async pub/sub for brain modules |

Plus `soma/soma_brain.py` — the orchestration loop.

### Supporting Modules

| Package | Module | What it does |
|---|---|---|
| `soma/core/` | `affective_core.py` | Panksepp drive states (seeking, care, play, grief, fear, rage) |
| `soma/memory/` | `episodic_store.py` | Episodic memory with salience-based retrieval |
| `soma/memory/` | `consolidator.py` | Sleep-inspired memory consolidation |
| `soma/benchmarks/` | `memorial_salience.py` | Spearman correlation benchmark (passing r > 0.7) |
| `soma/benchmarks/` | `layer_benchmarks.py` | Stub benchmarks for all 4 Damasio layers |

---

## The Unified Dashboard

Single Streamlit app at `soma/proto_self/dashboard.py` showing:

1. **Live Polar H10** — heart rate, RMSSD, baseline comparison
2. **Fitbit Overnight** — recovery score, HRV, sleep stages, SpO2, 7-day trend chart
3. **CPAP (ResMed)** — last night AHI, usage, leak, 30-day compliance, 14-day chart, correlation with Fitbit recovery
4. **RR Intervals Chart** — recent signal
5. **Baseline Reference** — RHR and RMSSD thresholds
6. **SOMA Probe** — pending anomalies ready to probe
7. **Memory Timeline** — recent autobiographical exchanges
8. **Brain Simulation** — total cycles, current valence/arousal/drive, recent cycles
9. **Anomaly Log** — full history of flagged events
10. **SOMA Recommends** — state-aware recommendations
11. **Reading Queue** — state-filtered articles from Substack
12. **Recent Sessions** — session history with HRV summaries

Run: `streamlit run soma/proto_self/dashboard.py`

---

## What's NOT Built Yet

### Immediate Next Steps (Days)

1. **Collect more morning_baseline sessions** — Need 5+ for robust baseline model.
2. **Set up crons** — Fitbit at 6am, CPAP at 6:30am, Research agent at 6:15am, Substack at 6:45am.
3. **Wire dashboard chat to brain loop** — Currently chat is a stub; should call `soma_brain.run_cycle(query=...)`.
4. **First real CPAP sync** — test myAir auth flow with actual credentials, sync 30 days.
5. **Run brain end-to-end with Polar live** — confirm Ollama Qwen3 integration on M4.

### Week 6 Preview (from spec notes)

6. **Camera + LLaVA** — activate visual interoception via SOMA_CAMERA_ENABLED=true.
7. **Longitudinal pattern detection** — find recurring anomaly signatures across weeks.
8. **Two-brain architecture** — SOMA-patient mirror + SOMA-AI connected.

### Medium-Term (Weeks 6-10)

9. **NAS deployment** — Docker-compose for UGREEN DXP4800 Pro. FastAPI 24/7. LanceDB on NAS.
10. **Tailscale + React Native mobile** — phone as thin sensor, NAS as persistent brain. Spec in SPEC_WEEK2.md.
11. **Little Moments integration** — video clips feeding autobiographical layer.
12. **EDF SD card automation** — ez-Share Wi-Fi SD card + auto-sync script.
13. **Ollama on startup** — ensure Qwen3 is running when brain loop starts.

### Longer Horizon (Months 2-6)

14. **ML body state classifier** — XGBoost on labeled sessions, replace rule-based `classify_body_state()`.
15. **RAEN scorer as ML model** — train on accumulated JSONL, replace LLM scoring.
16. **Domain-adapted embeddings** — fine-tune sentence-transformers on SOMA corpus.
17. **LoRA fine-tune Mistral/Llama** — offline reasoning capability.
18. **Knowledge graph extraction** — BioBERT NER for cross-domain queries.
19. **Relational Self module (Layer 4)** — presence detection, co-regulation modeling.
20. **Multimodal fusion transformer (Layer 2)** — real Core Consciousness beyond anomaly detection.

### Known Bugs / Rough Edges

- Fitbit API HRV endpoint requires device support (Inspire 3 works, older devices may not).
- Polar H10 BLE requires 1-3 retries on macOS CoreBluetooth (normal).
- Dashboard brain panel shows empty if `~/.soma/lancedb` doesn't exist (handled gracefully).
- Several `DeprecationWarning` messages from lancedb (non-blocking).
- No MFA support in myAir client (works if MFA disabled on account).

---

## Architecture Summary

```
soma/
  autoresearcher/     # Overnight research loop (PubMed + S2 + LLM)
  brain/              # Week 5: 5-module neural simulation
  benchmarks/         # Layer-specific evaluation framework
  core/               # Affective core (Panksepp drives)
  memory/             # Episodic store + consolidation
  proto_self/         # Body signal acquisition + all Week 1-4 modules
    cpap/             # ResMed myAir + EDF parser
    fitbit/           # Fitbit Inspire 3 integration
  soma_brain.py       # Main orchestration loop (Week 5)
```

---

## Key Files & Data

| File/Dir | Purpose | Tracked? |
|---|---|---|
| `SOMA_PROJECT_FOUNDATION.md` | Vision doc, Damasio framework | Yes |
| `docs/SPEC_WEEK*.md` | Design specs from claude.ai | Yes |
| `data/patient_876.json` | Patient biomarker profile | No (gitignored) |
| `data/corpus.json` | Curated recommendations | No (gitignored) |
| `data/newsletters.json` | Followed newsletters + seeds | No (gitignored) |
| `data/soma_cardio.db` | SQLite (sessions, RR, anomalies, recommendations) | No (gitignored) |
| `data/lancedb/` | Research findings + recommendation memory | No (gitignored) |
| `~/.soma/soma.db` | Fitbit + CPAP LanceDB store | No (outside repo) |
| `~/.soma/lancedb/` | Brain episodic memory (Week 5) | No (outside repo) |
| `.env.local` | OpenRouter API key | No (gitignored) |
| `~/.fitbit_config.json` | Fitbit credentials | No (outside repo) |
| `~/.fitbit_tokens.json` | Fitbit OAuth tokens | No (outside repo) |
| `~/.resmed_config.json` | myAir credentials | No (outside repo) |
| `~/.resmed_tokens.json` | myAir OAuth tokens | No (outside repo) |

---

## How to Run Everything

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .

# One-time auth setups
python -m soma.proto_self.fitbit.fitbit_auth     # Fitbit OAuth
# Create ~/.resmed_config.json, then:
python -m soma.proto_self.cpap.myair_client      # myAir auth

# Data collection (run as needed)
python -m soma.proto_self.polar_logger morning_baseline   # Polar HRV session
python -m soma.proto_self.polar_logger post_run
python -m soma.proto_self.baseline_model --min-samples 20 # Build baseline
python -m soma.proto_self.fitbit.daily_sync               # Fitbit yesterday
python -m soma.proto_self.cpap.daily_sync --days 30       # CPAP last 30 days

# State + recommendations
python -m soma.proto_self.state_classifier        # Current body state
python -m soma.proto_self.recommender             # Get recommendations
python -m soma.proto_self.feedback_logger         # Rate recommendations

# Research
python -m soma.proto_self.research_agent          # Daily research
python -m soma.proto_self.corpus_review           # Review candidates
python -m soma.proto_self.substack_agent          # Substack articles
python -m soma.proto_self.article_review          # Review articles

# Anomaly -> conversation
python -m soma.proto_self.anomaly_detector        # Live anomaly watch
python -m soma.proto_self.probe_interface         # Probe + respond
python -m soma.proto_self.narrative_builder       # Weekly synthesis

# Brain simulation (Week 5)
ollama pull qwen3:8b                              # Optional: local LLM
python -m soma.soma_brain --single                # One brain cycle
python -m soma.soma_brain                         # Continuous (30s)

# Dashboard (shows everything)
streamlit run soma/proto_self/dashboard.py

# AutoResearcher overnight loop
python -m soma.autoresearcher.run --overnight --profile data/patient_876.json

# Server (for NAS/mobile)
uvicorn soma.proto_self.soma_server:app --host 0.0.0.0 --port 8765

# Tests
python -m pytest tests/ -v
```

---

## Suggested Cron Setup

```cron
# Morning data ingest (sequential to avoid API overlap)
0 6 * * *  cd /path/to/soma && .venv/bin/python -m soma.proto_self.fitbit.daily_sync >> logs/fitbit.log 2>&1
15 6 * * * cd /path/to/soma && .venv/bin/python -m soma.proto_self.research_agent >> logs/research.log 2>&1
30 6 * * * cd /path/to/soma && .venv/bin/python -m soma.proto_self.cpap.daily_sync >> logs/cpap.log 2>&1
45 6 * * * cd /path/to/soma && .venv/bin/python -m soma.proto_self.substack_agent >> logs/substack.log 2>&1

# Weekly narrative
0 21 * * 0 cd /path/to/soma && .venv/bin/python -m soma.proto_self.narrative_builder >> logs/narrative.log 2>&1

# Brain loop as service (use launchd on Mac, systemd on Linux)
# Not a cron — runs continuously via `python -m soma.soma_brain`
```

---

## What Changed Since PROJECT_STATUS.md (April 14)

- **Fitbit integration complete** — auth, daily sync, dashboard panel, state enrichment
- **Week 5 Brain built** — full 5-module neural simulation with Qwen3/Claude routing
- **CPAP integration complete** — myAir API + EDF parser + correlation with Fitbit
- **Dashboard unified** — single view absorbs the standalone brain_dashboard.py
- **115 tests** (up from 92)
- **17 commits** (up from 12)
- **75 modules** (up from 46)

---

*Status as of April 16, 2026 — 17 commits, 115 tests, 75 modules, ~12,000 lines of code.*
*Next: Week 6 — camera (LLaVA), longitudinal pattern detection, two-brain architecture.*
