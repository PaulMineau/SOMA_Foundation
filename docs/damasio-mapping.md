# Damasio Layer Mapping — SOMA Classification

## Purpose

Every research finding gets tagged to one of Damasio's four consciousness layers. This is SOMA's unique contribution — it maps biomedical evidence onto a phenomenological architecture of the self.

Source text: Damasio, A. (1999). *The Feeling of What Happens: Body and Emotion in the Making of Consciousness.*

---

## The Four Layers

### Proto-Self
**Damasio's definition**: The continuous, moment-to-moment mapping of the body's internal state. Automatic, pre-conscious. The biological substrate of being alive.

**SOMA mapping — assign here when the finding primarily affects**:
- Autonomic nervous system (heart rate, blood pressure, HRV)
- Sleep architecture and physiology (deep sleep, REM, AHI)
- Cardiovascular function
- Metabolic homeostasis
- Cellular inflammation
- Hormonal baseline regulation (cortisol rhythms, testosterone production)

**Examples**:
- "EPA reduces cardiovascular inflammation" → Proto-Self
- "CPAP reduces nocturnal hypoxemia" → Proto-Self
- "SHBG normalization restores free testosterone" → Proto-Self

---

### Core Consciousness
**Damasio's definition**: The moment-to-moment feeling of being in a body, arising from the Proto-Self being modified by an object or event. The "pulse of knowing."

**SOMA mapping — assign here when the finding primarily affects**:
- Alertness and wakefulness
- Cortisol morning response
- Circadian rhythm regulation
- Cognitive performance in the immediate present
- Working memory and attention
- Mood in the present moment

**Examples**:
- "Magnesium improves sleep onset and morning alertness" → Core Consciousness
- "Vitamin D affects circadian gene expression" → Core Consciousness
- "Morning cortisol spike from nicotine cessation" → Core Consciousness

---

### Extended Consciousness
**Damasio's definition**: The capacity to hold a personal past and anticipate a personal future. Autobiography. The narrative self extended in time.

**SOMA mapping — assign here when the finding primarily affects**:
- Long-term memory formation
- Neuroprotection over years/decades
- Dementia and cognitive decline risk
- Executive function and planning capacity
- Homocysteine neurotoxicity (long-term neural damage)
- Future health trajectory modeling

**Examples**:
- "Elevated homocysteine doubles dementia risk over 10 years" → Extended Consciousness
- "EPA neuroprotection in aging males" → Extended Consciousness
- "Vitamin D deficiency associated with accelerated cognitive decline" → Extended Consciousness

---

### Relational Self
**Damasio's definition**: The self as it exists in relation to others. Social emotions, empathy, the capacity to show up as a person in relationship.

**SOMA mapping — assign here when the finding primarily affects**:
- Testosterone and its role in mood, libido, motivation
- Depression and emotional regulation
- Social behavior and bonding
- Capacity for presence in relationships
- Stress response in social contexts
- Recovery from addiction (relational repair)

**Examples**:
- "Low free testosterone associated with depression and social withdrawal" → Relational Self
- "Recovery from alcohol use improves relational attunement" → Relational Self
- "Sleep deprivation reduces emotional empathy" → Relational Self

---

## Classification Rules

### Primary Rule
One finding, one layer. Assign to the layer most directly affected by the mechanism of action.

### Tiebreaker: Upstream Wins
If a finding spans multiple layers, assign to the most upstream layer. Example: "CPAP therapy reduces cardiovascular risk AND improves morning alertness AND reduces depression." Cardiovascular is Proto-Self, morning alertness is Core, depression is Relational. Assign to **Proto-Self** — the CPAP mechanism is fundamentally autonomic/physiological.

### LLM Classification Prompt

```
You are classifying a biomedical research finding into Damasio's four consciousness layers 
for the SOMA personal health system.

Layers:
- Proto-Self: autonomic regulation, sleep physiology, cardiovascular, metabolism, hormonal baseline
- Core Consciousness: alertness, circadian rhythm, cortisol, immediate cognitive performance, working memory
- Extended Consciousness: long-term memory, neuroprotection, dementia risk, executive function, future health trajectory
- Relational Self: testosterone/mood/libido, depression, social behavior, emotional regulation, recovery

Finding: {intervention} → {outcome_measure} — {effect_direction} effect

Return JSON: {"layer": "<one of the four layer names>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}
```

---

## Why This Matters for SOMA

Tagging findings to layers enables SOMA to answer: "What is the state of my Proto-Self right now, and what interventions most directly improve it?" The Extended Consciousness layer becomes a longevity map. The Relational Self layer directly answers: "What is the patient's capacity to be fully present for the people who matter most?"

This is not metaphor. It is a clinical ontology built on phenomenological foundations.
