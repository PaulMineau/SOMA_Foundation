# SOMA — Sentient Observation & Memory Architecture

Monorepo for the SOMA consciousness architecture framework. Built on Damasio's four-layer theory of consciousness, using human embodiment (wearables, sensors) as sensory substrate.

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| LLM | `claude-opus-4-6` via OpenRouter (`OPENROUTER_API_KEY`) |
| Vector store | LanceDB 0.13+ (local-first) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Research ingestion | NCBI E-utilities (PubMed) |
| UI | Streamlit |
| Runtime | Docker / docker-compose on UGREEN DXP4800 Pro NAS |

## Package Structure

```
soma/
├── __init__.py
├── autoresearcher/          # AutoResearcher — overnight research loop
│   ├── __init__.py
│   ├── seed.py              # BiomarkerProfile dataclass + loader
│   ├── query_gen.py         # LLM → PubMed/S2 query strings
│   ├── fetcher.py           # Async PubMed + Semantic Scholar fetch
│   ├── extractor.py         # LLM → structured paper fields
│   ├── scorer.py            # RAEN + LSS scoring (5 dimensions)
│   ├── damasio.py           # Classify finding to SOMA layer
│   ├── search_arms.py       # Layer-specific parallel PubMed search
│   ├── corpus_export.py     # Export scored papers to training JSONL
│   ├── convergence.py       # Loop termination logic
│   ├── synthesizer.py       # LLM → ranked briefing
│   ├── memory.py            # LanceDB read/write
│   ├── llm.py               # OpenRouter LLM client
│   └── run.py               # CLI entrypoint + loop orchestration
├── core/                    # SOMA consciousness layers
│   ├── __init__.py
│   └── affective_core.py    # Panksepp drive states (Layer 2 seed)
├── memory/                  # Episodic memory + consolidation
│   ├── __init__.py
│   ├── episodic_store.py    # Episodic memory store
│   └── consolidator.py      # Sleep-inspired memory consolidation
└── benchmarks/              # SOMA benchmark suite
    ├── __init__.py
    ├── memorial_salience.py # Test 2: Memorial Salience Alignment
    └── layer_benchmarks.py  # Layer-specific benchmark stubs
```

## Commands

```bash
# Install
pip install -e .

# Run one research loop
python -m soma.autoresearcher.run --profile data/patient_876.json --max-iterations 5

# Run overnight loop (all 4 layer search arms)
python -m soma.autoresearcher.run --overnight

# Export training corpus
python -m soma.autoresearcher.corpus_export --output data/training_data.jsonl

# Launch Streamlit UI
streamlit run app.py

# Run tests
pytest tests/ -v

# Run benchmarks
python -m soma.benchmarks.memorial_salience
python -m soma.benchmarks.layer_benchmarks

# Type check
mypy soma/ --strict
```

## Architecture

### Damasio Layers
1. **Proto-Self** — Continuous body state model (HRV, sleep, physiology)
2. **Core Consciousness** — Present-moment awareness, attention salience
3. **Extended Consciousness** — Autobiographical memory, narrative continuity
4. **Relational Self** — Self-in-relation, co-regulation models

### AutoResearcher Loop
1. Seed → load BiomarkerProfile
2. Query gen → LLM generates PubMed queries (or layer-specific search arms)
3. Fetch → async PubMed + S2
4. Extract → LLM structured extraction
5. Score → RAEN (Relevance, Actionability, Evidence, Novelty) + LSS (Layer Specificity Score)
6. Damasio classify → route paper to consciousness layer
7. Convergence check → stop or iterate
8. Synthesize → ranked briefing
9. Store → LanceDB

### Scoring: RAEN + LSS
- **R** (Relevance): Semantic similarity to profile
- **A** (Actionability): Concrete, safe intervention?
- **E** (Evidence quality): Study type + citation + funding
- **N** (Novelty): 1 - similarity to known interventions
- **LSS** (Layer Specificity): How strongly does this paper map to a specific Damasio layer?

## Coding Conventions

- All async/await for I/O (httpx, LanceDB writes)
- `dataclasses` for all data models — no raw dicts in business logic
- Type hints required everywhere — `mypy --strict` must pass
- Rate limiting: `asyncio.Semaphore(3)` for PubMed, `asyncio.Semaphore(1)` for S2
- Log every LLM call with token count
- Never store raw PII outside local filesystem
- Local-first always — health data stays on-device

## Key Principles

1. Proto-Self first — build the foundation before the roof
2. Champion/challenger on every model — nothing ships without beating the incumbent
3. The overnight loop is the thesis — a system that reads science to improve its own architecture
4. Be your own customer — the first research instance drives design decisions
5. Demo-driven milestones
