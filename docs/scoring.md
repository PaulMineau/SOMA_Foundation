# Scoring Function — RAEN

## Objective

Maximize expected health-adjusted life years per unit of behavioral change required, weighted by evidence confidence.

Operationalized as: `S = R × A × E × N`

All components are [0.0, 1.0]. Final score is their product.

---

## R — Relevance

**What it measures**: How closely does this paper's study population and intervention match the patient's biomarker profile?

**Method**: Cosine similarity between:
- `paper_embedding`: encode `"{population_description}. {intervention}."`
- `profile_embedding`: encode `profile.to_embedding_text()`

**Model**: `all-MiniLM-L6-v2` (fast, runs locally, 384 dimensions)

**Thresholds**:
- `R < 0.4` → discard paper before scoring A/E/N (saves compute)
- `R 0.4–0.6` → score but note "population mismatch" in synthesis
- `R > 0.7` → strong match, include in briefing candidates

**Edge cases**:
- Papers on general male aging with no sleep apnea mention → R ≈ 0.5 (relevant but not exact)
- Papers on 50yo males with elevated SHBG → R ≈ 0.85 (strong match)
- Papers on female populations → R ≈ 0.2 (filter out unless intervention is universal)

---

## A — Actionability

**What it measures**: Can the patient actually do something about this, safely, this week?

**Binary preconditions** (if either is false → A = 0.0):
1. `extract.actionable == True`: paper describes a concrete intervention (supplement, lifestyle change, procedure)
2. `extract.safe_for_profile == True`: LLM judges intervention safe given conditions + contraindications

**Scoring when preconditions met**:

```
base_A = 0.7

# Effect size bonus (max +0.3)
if effect_size is not None:
    effect_bonus = min(abs(effect_size) / 50.0, 0.3)
else:
    effect_bonus = 0.0

# Direction modifier
if effect_direction == "positive":
    A = base_A + effect_bonus
elif effect_direction == "null":
    A = 0.2
elif effect_direction == "negative":
    A = 0.0  # harmful — discard
elif effect_direction == "mixed":
    A = 0.4
```

**Supplement conflicts**: If `intervention` conflicts with `current_supplements` (e.g., "avoid B12 with metformin"), flag in synthesis but don't zero out A — let the patient decide.

---

## E — Evidence Quality

**What it measures**: How trustworthy is this evidence?

**Base scores by study type**:

| Study Type | E_base | Notes |
|---|---|---|
| Cochrane review | 1.00 | Gold standard |
| Meta-analysis | 0.80 | Check heterogeneity |
| RCT | 0.85 | Best individual study type |
| Observational | 0.50 | Confounders likely |
| Case series | 0.20 | Hypothesis generating only |
| Review (narrative) | 0.40 | Expert opinion, not evidence |
| Unknown | 0.30 | Default pessimistic |

**Industry funding penalty**: `-0.15` if `industry_funded == True`

**Citation count bonus** (secondary, minor):
- `citation_count > 100` → `+0.05`
- `citation_count > 500` → `+0.10`
- Max bonus: `+0.10` (don't let popularity override study design)

**Final E**: Clamp to [0.0, 1.0]

---

## N — Novelty

**What it measures**: Is this something the patient doesn't already know and act on?

**Method**: 
1. Embed `extract.intervention` → `paper_action_emb`
2. Compute cosine similarity against `known_interventions_embedding` (mean of all embeddings from `known_interventions_acted_on` in profile)
3. `N = 1.0 - similarity`

**Intuition**:
- "Take methylated B12" when patient already takes it → similarity ≈ 0.9 → N ≈ 0.1
- "Tongkat Ali for SHBG reduction" → similarity ≈ 0.2 → N ≈ 0.8 (novel)
- "Creatine monohydrate for sleep apnea" → similarity ≈ 0.1 → N ≈ 0.9 (novel, unexpected)

**Override**: If `R × A × E` is very high (> 0.6) but N is low, still include in synthesis with label "Already optimized — strong supporting evidence." Validation is valuable even when not novel.

---

## Final Score Interpretation

| Score | Interpretation |
|---|---|
| 0.7–1.0 | Strong candidate — include as top recommendation |
| 0.4–0.7 | Worth reviewing — include in briefing |
| 0.2–0.4 | Marginal — include only if top candidates are sparse |
| 0.0–0.2 | Discard |

**Synthesis sort order**: Descending by total score. Top 5–10 appear in briefing.

---

## What This Function Is NOT

- Not a replacement for clinical judgment
- Not designed to identify drug interactions (flag for clinician)
- Not calibrated for populations other than the active patient profile
- Not a diagnostic tool — descriptive only

The briefing always ends with: "Discuss top-scored interventions with your physician before changing protocols."
