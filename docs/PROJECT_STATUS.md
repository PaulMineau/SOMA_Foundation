# SOMA Project Status — April 14, 2026

**Repo:** https://github.com/PaulMineau/SOMA_Foundation  
**Stack:** Python 3.12, SQLite, LanceDB, FastAPI, Streamlit, bleak (BLE), OpenRouter (Claude API)  
**Tests:** 85 passing  
**Modules:** 46 Python files across 6 packages  

---

## What SOMA Is

SOMA (Sentient Observation & Memory Architecture) is a layered AI consciousness framework built on Antonio Damasio's neuroscience, using a human body as its sensory substrate. It reads physiological signals (HRV, heart rate) from a Polar H10 chest strap, builds a model of the wearer's baseline, detects when something changes, and makes recommendations grounded in actual nervous system state.

The theoretical foundation is in `SOMA_PROJECT_FOUNDATION.md` at the repo root.

---

## What's Built and Working

### Phase 0 — AutoResearcher (Complete)

**Package:** `soma/autoresearcher/` (12 modules)

The overnight research loop that reads science and routes it to Damasio layers.

| Module | What it does |
|---|---|
| `run.py` | CLI entrypoint. Dual-track overnight loop: health queries (RAEN scored) + architecture queries (LSS scored) run in parallel |
| `scorer.py` | RAEN scoring (Relevance x Actionability x Evidence x Novelty) + LSS (Layer Specificity Score). Architecture papers scored separately with `score_architecture_paper()` |
| `search_arms.py` | 4 parallel PubMed search arms, one per Damasio layer (16 queries total) |
| `damasio.py` | 3-tier layer classification: LSS -> keyword rules -> LLM fallback |
| `fetcher.py` | Async PubMed E-utilities + Semantic Scholar with retry/rate limiting |
| `extractor.py` | LLM structured extraction (intervention, effect, safety) |
| `query_gen.py` | LLM generates MeSH queries from biomarker profile |
| `synthesizer.py` | LLM generates ranked briefing from scored papers |
| `memory.py` | LanceDB vector store for research findings |
| `corpus_export.py` | Export scored papers to JSONL training data |
| `convergence.py` | Loop termination logic |
| `seed.py` | BiomarkerProfile dataclass, JSON loader |
| `llm.py` | OpenRouter client (claude-opus-4-6) |

**Status:** Fully working. Has been run live. First dual-track overnight loop scored 3 health papers + 77 architecture papers. Training corpus accumulating in `data/training_data.jsonl`.

**Run:** `python -m soma.autoresearcher.run --overnight --profile data/patient_876.json`

---

### Phase 1 — Proto-Self: Body Signal Acquisition (Complete)

**Package:** `soma/proto_self/` (core BLE + HRV modules)

| Module | What it does |
|---|---|
| `polar_reader.py` | Polar H10 BLE connection via bleak. Parses HR Measurement characteristic (0x2A37). Handles macOS CoreBluetooth quirks with retry + re-discovery |
| `hrv.py` | Time-domain HRV metrics (RMSSD, SDNN, pNN50). Artifact rejection. Rule-based body state classification (recovery/resting/optimal/stressed/fatigued) |
| `polar_logger.py` | Continuous labeled session logging. Streams to SQLite + daily CSV. Ctrl+C to stop. Labels become training signal |
| `db.py` | SQLite schema (sessions, rr_intervals, anomalies, recommendations, research_log). Session lifecycle, CSV export |
| `storage.py` | LanceDB time-series tables for raw RR + computed HRV windows |
| `monitor.py` | Live terminal readout tailing SQLite |
| `run.py` | Quick fixed-duration collection with HRV summary |

**Status:** Fully working. First live reading: RMSSD=111.5ms, SDNN=96.9ms, HR=77bpm, body state=OPTIMAL. BLE connection requires 1-3 retry attempts on macOS.

**Run:** `python -m soma.proto_self.polar_logger morning_baseline`

---

### Phase 1b — Core Consciousness: Baseline + Anomaly Detection (Complete)

| Module | What it does |
|---|---|
| `artifact_filter.py` | Two-stage RR cleaning: range rejection (300-1500ms) + ectopic rejection (>20% beat-to-beat change) |
| `baseline_model.py` | Computes physiological fingerprint from `morning_baseline` sessions. Saves to JSON with alert thresholds (1.5 std dev). Per-beat HR stats + rolling 60-beat RMSSD windows |
| `anomaly_detector.py` | Polls SQLite, compares live readings to baseline, writes anomaly events when RHR or RMSSD exceed 1.5 std dev |
| `dashboard.py` | Streamlit live dashboard: current HR/RMSSD, baseline overlay, anomaly log, recommendations, reading queue, session history. Auto-refreshes every 10s |

**Status:** Built and tested. Needs more `morning_baseline` sessions to build a robust baseline model (minimum 100 RR intervals, ideally 300+).

**Run:**
- `python -m soma.proto_self.baseline_model --min-samples 20`
- `python -m soma.proto_self.anomaly_detector`
- `streamlit run soma/proto_self/dashboard.py`

---

### Phase 2 — SOMA Core Server (Complete)

| Module | What it does |
|---|---|
| `soma_server.py` | FastAPI server for NAS deployment. Endpoints: POST /ingest/rr (receive batches from mobile), GET /status (current HRV), GET /baseline (morning baseline distribution), GET /anomalies (flagged events), POST /ingest/context (relabel sessions), GET /sessions (list), POST /probe (Week 4 placeholder). Anomaly detection runs on each ingest |

**Status:** Built and tested (11 endpoint tests passing). Not yet deployed to NAS. Tailscale setup and React Native mobile app are documented in SPEC_WEEK2.md but not yet implemented.

**Run:** `uvicorn soma.proto_self.soma_server:app --host 0.0.0.0 --port 8765`

**Not yet done:**
- Tailscale tunnel between NAS and phone
- React Native mobile app (scaffold + BLE + smart routing)
- Docker deployment to UGREEN DXP4800 Pro NAS

---

### Phase 3 — Extended Consciousness: Recommendations (Complete)

| Module | What it does |
|---|---|
| `state_classifier.py` | Classifies current state (depleted/recovering/baseline/restored/peak) by comparing live HRV to baseline model via z-scores |
| `recommender.py` | State-aware recommendation engine. Matches physiological state to curated corpus. Logs recommendations to DB |
| `feedback_logger.py` | Interactive CLI to close the loop: did you follow the recommendation? How did you feel after? |
| `soma_memory.py` | LanceDB embedding store for (state, recommendation, outcome) tuples. Learns which recommendations work via semantic similarity search |
| `soma_profile.py` | Assembles rich patient context (identity, physiology, feedback history, existing corpus) for research prompt injection |

**Data:** `data/corpus.json` — 17+ curated recommendations (movies, books, activities, media) with per-state matching rules and tags.

**Status:** Fully working. The learning loop is: recommend -> follow -> feedback -> embed -> improve.

---

### Phase 3b — Daily Research Agent (Complete)

| Module | What it does |
|---|---|
| `research_agent.py` | Daily Claude API call via OpenRouter. Rotates through 10 topic areas by day of year. Generates 5 candidates, scores with RAEN, stages for review |
| `raen_scorer.py` | Scores candidates on Relevance (fuzzy tag match to interests), Actionability (duration + state), Evidence (feedback history, base score of 4 when no data), Novelty (not in corpus). Threshold >= 0.55 |
| `corpus_review.py` | Interactive CLI: approve/reject staged additions before merge to live corpus |

**Data:** `data/corpus_additions.json` (staging file, gitignored)

**Status:** Working. First live research run produced candidates. RAEN scorer was tuned after initial run (fuzzy matching, base evidence score, lower threshold).

**Run:** `python -m soma.proto_self.research_agent --topic "books on consciousness"`

---

### Phase 3c — Substack Agent & Newsletter Intelligence (Complete)

| Module | What it does |
|---|---|
| `substack_agent.py` | Fetches RSS from followed newsletters, classifies articles via OpenRouter, scores with RAEN, auto-surfaces high-scoring articles (>0.85), stages others (>0.55). Discovers new newsletters daily by rotating through 6 topic seeds |
| `article_review.py` | CLI for article approval, newsletter follow/dismiss, reading list filtered by state, mark-as-read |

**Data:** `data/newsletters.json` — 1 followed newsletter (HCR) + 6 discovery seeds + 3 discovered newsletters (from first live run)

**Status:** Working. First run discovered 3 newsletters (The Dharma Dispatch, Mind & Life Perspectives, Tricycle Wisdom). Dashboard shows reading queue filtered by physiological state.

---

### Supporting Modules

| Package | Module | What it does |
|---|---|---|
| `soma/core/` | `affective_core.py` | Panksepp drive states (seeking, care, play, grief, fear, rage) with activation and decay. Seeds Layer 2 |
| `soma/memory/` | `episodic_store.py` | Episodic memory store with salience-based retrieval (affect x prediction error) |
| `soma/memory/` | `consolidator.py` | Sleep-inspired memory consolidation: strengthens salient memories, decays routine ones |
| `soma/benchmarks/` | `memorial_salience.py` | Benchmark: Spearman correlation between retrieval rank and ground-truth salience. Currently passing (r > 0.7) |
| `soma/benchmarks/` | `layer_benchmarks.py` | Stub benchmarks for all 4 Damasio layers + AutoResearcher routing accuracy |

---

## What's NOT Built Yet

### Immediate Next Steps

1. **Collect more baseline data** — Need 5+ labeled `morning_baseline` sessions to build a robust baseline model. This is the most important thing to do right now.

2. **Week 4: Conversational Probe** — When an anomaly is detected, SOMA generates a natural language probe: *"Your RMSSD dropped to 18ms at 2:30pm. What was happening?"* Response gets stored as autobiographical memory in LanceDB. The `/probe` endpoint in `soma_server.py` is a placeholder waiting for this.

3. **Cron automation** — `research_agent.py` at 6am, `substack_agent.py` at 6:15am, `baseline_model.py` weekly. Currently manual.

### Medium-Term (Weeks 5-8)

4. **NAS deployment** — Docker-compose for UGREEN DXP4800 Pro. FastAPI server running 24/7. LanceDB on NAS storage.

5. **Tailscale + React Native mobile app** — Phone as thin sensor node, NAS as persistent brain. Spec is in SPEC_WEEK2.md with full code scaffolds (React Native BLE collector, smart WiFi/cellular router, offline sync queue).

6. **Little Moments integration** — Connect video clips of River to the autobiographical layer (Extended Consciousness). Narrative coherence scoring, temporal self-model, consolidation pass.

7. **Fitbit Inspire 3 integration** — Sleep staging, step count, resting HR trends. Complements Polar H10 (which is session-based) with 24/7 passive data.

### Longer Horizon (Months 2-6)

8. **ML body state classifier** — Replace rule-based `classify_body_state()` with XGBoost trained on labeled sessions. This is the Phase 1 goal from the foundation doc.

9. **RAEN scorer as ML model** — Train XGBoost on the accumulated JSONL training data, replacing LLM calls for paper scoring.

10. **Domain-adapted embeddings** — Fine-tune sentence-transformers on SOMA corpus for better semantic search.

11. **LoRA fine-tune on small LLM** — Mistral 7B or Llama 3.2 on SOMA corpus for offline reasoning.

12. **Knowledge graph extraction** — BioBERT NER/relation extraction for "what interventions improve both HRV and slow-wave sleep?"

13. **Relational Self module (Layer 4)** — Presence detection (River/Nia), co-regulation modeling (how does HRV change when they're present?), attachment dynamics visualization.

14. **Multimodal fusion transformer (Layer 2)** — Present-moment state vector from all active sensor streams. The Core Consciousness module proper, beyond anomaly detection.

---

## Architecture Summary

```
soma/
  autoresearcher/    # Overnight research loop (PubMed + S2 + LLM scoring)
  proto_self/        # Body signal acquisition + all Week 1-3c modules
  core/              # Affective core (Panksepp drives)
  memory/            # Episodic store + consolidation
  benchmarks/        # Layer-specific evaluation framework
```

```
Polar H10 (BLE) --> polar_logger.py --> soma_cardio.db (SQLite)
                                            |
                                    baseline_model.py --> baseline_model.json
                                            |
                                    anomaly_detector.py --> anomalies table
                                            |
                                    state_classifier.py --> current state
                                            |
                    recommender.py <-- corpus.json <-- corpus_review.py <-- research_agent.py
                         |
                    dashboard.py (Streamlit)
                         |
                    article_review.py <-- substack_agent.py <-- newsletters.json
```

---

## Key Files

| File | Purpose |
|---|---|
| `SOMA_PROJECT_FOUNDATION.md` | Vision doc, Damasio framework, full architecture spec |
| `data/patient_876.json` | Patient biomarker profile (gitignored) |
| `data/corpus.json` | Curated recommendation library (17+ entries) |
| `data/newsletters.json` | Followed newsletters + discovery seeds |
| `data/soma_cardio.db` | SQLite with sessions, RR intervals, anomalies, recommendations (gitignored) |
| `data/lancedb/` | LanceDB vector store for research findings + recommendation memory (gitignored) |
| `.env.local` | OpenRouter API key (gitignored) |

---

## How to Run

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Collect HRV data (Polar H10 on chest)
python -m soma.proto_self.polar_logger morning_baseline

# Build baseline model
python -m soma.proto_self.baseline_model --min-samples 20

# Check current state
python -m soma.proto_self.state_classifier

# Get recommendations
python -m soma.proto_self.recommender

# Run research agent
python -m soma.proto_self.research_agent --topic "books on consciousness"

# Review research candidates
python -m soma.proto_self.corpus_review

# Run Substack agent
python -m soma.proto_self.substack_agent

# Review articles
python -m soma.proto_self.article_review

# Dashboard
streamlit run soma/proto_self/dashboard.py

# Anomaly detector (runs alongside logger)
python -m soma.proto_self.anomaly_detector

# Overnight AutoResearcher loop
python -m soma.autoresearcher.run --overnight --profile data/patient_876.json

# SOMA Core server (for NAS/mobile)
uvicorn soma.proto_self.soma_server:app --host 0.0.0.0 --port 8765

# Run tests
python -m pytest tests/ -v
```

---

*Status as of April 14, 2026 — 12 commits, 85 tests, 46 modules, ~8000 lines of code.*
