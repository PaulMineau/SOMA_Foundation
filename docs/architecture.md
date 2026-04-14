# Architecture — SOMA AutoResearcher

## Core Loop

```
BiomarkerProfile
       │
       ▼
 Query Generation ◄────────────────────────────────┐
       │                                             │
       ▼                                             │ new queries
 Fetch Papers (PubMed + S2)                         │ from findings
       │                                             │
       ▼                                             │
 Extract Structured Fields                          │
       │                                             │
       ▼                                             │
  Score (RAEN)                                      │
       │                                             │
       ▼                                             │
 Convergence Check ─── NO ──────────────────────────┘
       │
      YES
       │
       ▼
  Synthesize Briefing
       │
       ▼
  Damasio Layer Tag
       │
       ▼
  Write to LanceDB (SOMA memory)
```

---

## Component Specs

### seed.py — BiomarkerProfile

```python
@dataclass
class BiomarkerProfile:
    # Lab values
    shbg_nmol_l: float           # elevated
    homocysteine_umol_l: float   # borderline high
    vitamin_d_ng_ml: float       # insufficient
    free_testosterone_pg_ml: float
    
    # Sleep
    sleep_efficiency_pct: float  # from Fitbit
    deep_sleep_minutes: float
    
    # Context
    age: int                     # 50
    sex: str                     # "male"
    conditions: list[str]        # ["sleep_apnea", "elevated_shbg"]
    current_supplements: list[str]  # ["D3_K2", "boron", "EPA", "methylated_B_complex"]
    recovery_stage_months: int   # months since condition onset
    
    # Meta
    research_focus: list[str]    # ["cardiovascular", "testosterone", "sleep", "homocysteine"]
    
    def to_embedding_text(self) -> str:
        """Natural language description for semantic matching."""
        return (
            f"50-year-old male with sleep apnea, elevated SHBG {self.shbg_nmol_l} nmol/L, "
            f"borderline homocysteine {self.homocysteine_umol_l} umol/L, insufficient vitamin D, "
            f"CPAP user. "
            f"Research focus: {', '.join(self.research_focus)}."
        )
```

### query_gen.py — LLM Query Generator

System prompt:
```
You are a clinical literature search specialist. Given a patient biomarker profile, 
generate 6-8 targeted PubMed search queries. Each query should use MeSH terms where 
possible. Prioritize intersection queries (e.g., "SHBG AND testosterone AND sleep apnea") 
over single-concept queries. Return JSON: {"queries": ["query1", "query2", ...]}.
```

Output: `list[str]` of PubMed-ready search strings.

On iteration 2+: include top 3 findings from previous round in prompt context. Ask LLM to generate follow-up queries based on gaps in evidence.

### fetcher.py — Paper Fetcher

PubMed E-utilities flow:
1. `esearch.fcgi` → get PMIDs (max 20 per query)
2. `esummary.fcgi` → get title, abstract, authors, journal, year, study type
3. If `score > 0.6` after abstract scoring: `efetch.fcgi` → full text (where available)

Rate limiting: `asyncio.Semaphore(3)` — 3 concurrent PubMed requests max.

Semantic Scholar: use `/graph/v1/paper/search` endpoint. Field: `title,abstract,year,citationCount,publicationTypes`.

Dedup by PMID or DOI before scoring.

```python
@dataclass
class Paper:
    pmid: str | None
    doi: str | None
    title: str
    abstract: str
    year: int
    study_type: str           # "RCT", "meta-analysis", "observational", "case", "review"
    citation_count: int
    industry_funded: bool     # extracted by LLM from methods section if available
    full_text: str | None
    source: str               # "pubmed" | "semantic_scholar"
```

### extractor.py — Structured Extraction

For each paper, LLM extracts:

```python
@dataclass  
class PaperExtract:
    intervention: str         # "Methylated B-complex supplementation"
    population_description: str  # "Adult males with hyperhomocysteinemia"
    effect_size: float | None # Cohen's d or % change if reported
    effect_direction: str     # "positive" | "negative" | "null" | "mixed"
    outcome_measure: str      # "homocysteine reduction"
    safe_for_profile: bool    # LLM judgment given patient conditions
    actionable: bool          # concrete intervention available OTC or via prescription
    conflicts_with_supplements: list[str]  # any conflicts with current_supplements
```

Prompt: structured JSON extraction. If field unavailable, use `null`. Never hallucinate effect sizes — if not in paper, return `null`.

### scorer.py — RAEN Scoring

```python
def score_paper(
    paper: Paper,
    extract: PaperExtract,
    profile: BiomarkerProfile,
    known_actions_embedding: np.ndarray,
    embedder: SentenceTransformer,
) -> RAENScore:
    
    # R — Relevance (semantic similarity)
    paper_text = f"{extract.population_description}. {extract.intervention}."
    paper_emb = embedder.encode(paper_text)
    profile_emb = embedder.encode(profile.to_embedding_text())
    R = float(cosine_similarity([paper_emb], [profile_emb])[0][0])
    
    # A — Actionability
    if not extract.actionable or not extract.safe_for_profile:
        A = 0.0
    else:
        effect_bonus = min(abs(extract.effect_size or 0.0) / 50.0, 0.3)
        A = 0.7 + effect_bonus if extract.effect_direction == "positive" else 0.3
    
    # E — Evidence quality
    E_base = {
        "cochrane": 1.0, "meta-analysis": 0.8, "RCT": 0.85,
        "observational": 0.5, "case": 0.2, "review": 0.4
    }.get(paper.study_type.lower(), 0.3)
    E = E_base - (0.15 if paper.industry_funded else 0.0)
    
    # N — Novelty (inverse of similarity to known actions)
    paper_emb_flat = embedder.encode(extract.intervention)
    novelty_sim = float(cosine_similarity([paper_emb_flat], [known_actions_embedding])[0][0])
    N = 1.0 - novelty_sim
    
    score = R * A * E * N
    return RAENScore(R=R, A=A, E=E, N=N, total=score)
```

### convergence.py — Loop Termination

```python
def should_converge(
    score_history: list[float],  # RAEN totals from this round
    iteration: int,
    max_iterations: int,
) -> bool:
    if iteration >= max_iterations:
        return True
    if len(score_history) < 10:
        return False
    last_10 = score_history[-10:]
    marginal_gain = max(last_10) - min(last_10)
    return marginal_gain < 0.05
```

### synthesizer.py — Briefing Generator

Output format (Streamlit renders as markdown):

```
## SOMA Health Brief — {date}

### Top Interventions
1. **{intervention}** — RAEN: {score:.2f}
   Evidence: {study_type}, {year}. Effect: {effect_size}.
   Action: {concrete next step}.
   SOMA layer: {Proto-Self | Core | Extended | Relational}

### Gaps Identified
- {topic} — literature sparse, suggest clinical consultation

### Already Optimized
- {interventions already in current_supplements with strong evidence}
```

### damasio.py — SOMA Layer Classifier

```python
LAYER_RULES = {
    "Proto-Self": [
        "autonomic regulation", "sleep physiology", "heart rate", 
        "blood pressure", "cardiovascular", "homeostasis"
    ],
    "Core Consciousness": [
        "alertness", "wakefulness", "cortisol", "circadian", 
        "cognitive performance", "working memory"
    ],
    "Extended Consciousness": [
        "long-term memory", "neuroprotection", "dementia risk",
        "executive function", "homocysteine neurotoxicity"
    ],
    "Relational Self": [
        "testosterone", "libido", "mood", "depression", 
        "social behavior", "empathy"
    ]
}
```

LLM makes final classification when keyword match is ambiguous. Returns one of the four layer strings.

### memory.py — LanceDB

Table schema: `soma_research`

| Field | Type |
|---|---|
| id | str (PMID or DOI) |
| title | str |
| abstract | str |
| intervention | str |
| outcome | str |
| raen_total | float |
| raen_r | float |
| raen_a | float |
| raen_e | float |
| raen_n | float |
| soma_layer | str |
| year | int |
| study_type | str |
| embedding | vector[384] |
| briefing_date | str (ISO) |

---

## Docker Deployment (NAS)

```yaml
# docker-compose.yml
services:
  autoresearcher:
    build: .
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - LANCEDB_PATH=/data/lancedb
    volumes:
      - /mnt/nas/soma/lancedb:/data/lancedb
      - /mnt/nas/soma/profiles:/data/profiles
    ports:
      - "8501:8501"
    command: streamlit run app.py --server.port 8501
```

Cron job on NAS: run research loop weekly, generate new briefing, write to LanceDB.

---

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-...
LANCEDB_PATH=./data/lancedb           # local dev
PROFILE_PATH=./data/patient_876.json
MAX_ITERATIONS=5
ABSTRACT_SCORE_THRESHOLD=0.6
LOG_LEVEL=INFO
```
