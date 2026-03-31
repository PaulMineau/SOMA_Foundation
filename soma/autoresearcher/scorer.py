"""RAEN + LSS scoring: Relevance x Actionability x Evidence x Novelty + Layer Specificity."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from soma.autoresearcher.extractor import PaperExtract
from soma.autoresearcher.fetcher import Paper
from soma.autoresearcher.seed import BiomarkerProfile

logger = logging.getLogger(__name__)

# Evidence base scores by study type (lowercase keys)
_EVIDENCE_BASE: dict[str, float] = {
    "cochrane": 1.0,
    "meta-analysis": 0.8,
    "rct": 0.85,
    "observational": 0.5,
    "case": 0.2,
    "review": 0.4,
    "unknown": 0.3,
}

RELEVANCE_DISCARD_THRESHOLD = 0.4

# Layer Specificity Score (LSS) — keyword sets per Damasio layer
LAYER_MAP: dict[str, list[str]] = {
    "proto_self": [
        "interoception", "HRV", "homeostasis",
        "free energy", "active inference", "body map",
        "cardiac afferent", "autonomic nervous system",
        "heart rate variability", "sleep physiology",
        "cardiovascular", "metabolic", "hormonal baseline",
        "inflammation", "deep sleep", "nocturnal",
    ],
    "core_consciousness": [
        "global workspace", "temporal binding",
        "attention", "present moment", "NCC",
        "neural correlates of consciousness", "specious present",
        "alertness", "wakefulness", "circadian",
        "working memory", "cognitive performance", "cortisol",
    ],
    "extended_consciousness": [
        "episodic memory", "autobiographical",
        "narrative self", "consolidation",
        "hippocampal", "memory reconsolidation",
        "neuroprotection", "dementia risk", "executive function",
        "cognitive decline", "brain aging", "long-term memory",
    ],
    "relational_self": [
        "attachment", "co-regulation", "intersubjectivity",
        "theory of mind", "social cognition",
        "mirror neuron", "parent-infant",
        "empathy", "emotional regulation", "social behavior",
        "mood", "relational",
    ],
}


@dataclass(frozen=True)
class RAENScore:
    """Computed RAEN + LSS score for a single paper."""

    R: float  # noqa: N815 — Relevance
    A: float  # noqa: N815 — Actionability
    E: float  # noqa: N815 — Evidence quality
    N: float  # noqa: N815 — Novelty
    LSS: float  # Layer Specificity Score
    total: float
    primary_layer: str  # which Damasio layer this paper best maps to
    layer_scores: dict[str, float]  # per-layer keyword match scores


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_relevance(
    extract: PaperExtract,
    profile: BiomarkerProfile,
    embedder: SentenceTransformer,
) -> float:
    """R — cosine similarity between paper and profile embeddings."""
    paper_text = f"{extract.population_description}. {extract.intervention}."
    profile_text = profile.to_embedding_text()

    paper_emb = np.asarray(embedder.encode(paper_text), dtype=np.float32)
    profile_emb = np.asarray(embedder.encode(profile_text), dtype=np.float32)

    sim: float = float(
        cosine_similarity(
            paper_emb.reshape(1, -1), profile_emb.reshape(1, -1)
        )[0][0]
    )
    return _clamp(sim)


def compute_actionability(extract: PaperExtract) -> float:
    """A — can the patient act on this safely?"""
    if not extract.actionable or not extract.safe_for_profile:
        return 0.0

    base_a = 0.7

    if extract.effect_direction == "positive":
        effect_bonus = min(abs(extract.effect_size or 0.0) / 50.0, 0.3)
        return _clamp(base_a + effect_bonus)

    if extract.effect_direction == "null":
        return 0.2

    if extract.effect_direction == "negative":
        return 0.0

    if extract.effect_direction == "mixed":
        return 0.4

    return 0.3


def compute_evidence(paper: Paper) -> float:
    """E — evidence quality based on study type, funding, and citations."""
    base = _EVIDENCE_BASE.get(paper.study_type.lower(), 0.3)

    if paper.industry_funded:
        base -= 0.15

    if paper.citation_count > 500:
        base += 0.10
    elif paper.citation_count > 100:
        base += 0.05

    return _clamp(base)


def compute_novelty(
    extract: PaperExtract,
    known_actions_embedding: NDArray[np.float32],
    embedder: SentenceTransformer,
) -> float:
    """N — 1 minus similarity to known interventions."""
    intervention_emb = np.asarray(
        embedder.encode(extract.intervention), dtype=np.float32
    )
    sim: float = float(
        cosine_similarity(
            intervention_emb.reshape(1, -1),
            known_actions_embedding.reshape(1, -1),
        )[0][0]
    )
    return _clamp(1.0 - sim)


def compute_layer_specificity(
    paper: Paper,
    extract: PaperExtract,
) -> tuple[float, str, dict[str, float]]:
    """LSS — how strongly does this paper map to a specific Damasio layer?

    Scans the paper title, abstract, intervention, and outcome for layer keywords.
    Returns (lss_score, primary_layer, per_layer_scores).

    LSS is high when a paper maps clearly to ONE layer (specificity)
    and low when it's ambiguous or matches no layer.
    """
    text = (
        f"{paper.title} {paper.abstract} "
        f"{extract.intervention} {extract.outcome_measure} "
        f"{extract.population_description}"
    ).lower()

    layer_scores: dict[str, float] = {}
    for layer_key, keywords in LAYER_MAP.items():
        hits = sum(1 for kw in keywords if kw.lower() in text)
        layer_scores[layer_key] = hits / len(keywords) if keywords else 0.0

    # Primary layer = highest scoring
    if not layer_scores or max(layer_scores.values()) == 0:
        return 0.0, "proto_self", layer_scores

    primary_layer = max(layer_scores, key=lambda k: layer_scores[k])
    top_score = layer_scores[primary_layer]

    # LSS rewards specificity: high score in one layer, low in others
    scores_list = sorted(layer_scores.values(), reverse=True)
    if len(scores_list) >= 2 and scores_list[0] > 0:
        # Ratio of top to runner-up — higher = more specific
        specificity = 1.0 - (scores_list[1] / scores_list[0]) if scores_list[0] > 0 else 0.0
        lss = _clamp(top_score * (0.5 + 0.5 * specificity))
    else:
        lss = _clamp(top_score)

    return lss, primary_layer, layer_scores


def build_known_actions_embedding(
    profile: BiomarkerProfile,
    embedder: SentenceTransformer,
) -> NDArray[np.float32]:
    """Compute mean embedding of all known interventions acted on."""
    actions = list(profile.known_interventions_acted_on)
    if not actions:
        dim: int = embedder.get_sentence_embedding_dimension() or 384
        return np.zeros(dim, dtype=np.float32)

    embeddings = np.asarray(embedder.encode(actions), dtype=np.float32)
    mean_emb: NDArray[np.float32] = np.mean(embeddings, axis=0).astype(
        np.float32
    )
    return mean_emb


def score_paper(
    paper: Paper,
    extract: PaperExtract,
    profile: BiomarkerProfile,
    known_actions_embedding: NDArray[np.float32],
    embedder: SentenceTransformer,
) -> RAENScore:
    """Score a paper using RAEN + LSS.

    S = R x A x E x N (the core score). LSS is tracked separately
    for layer routing — it doesn't multiply into the total because
    layer specificity is orthogonal to paper quality.
    """
    r = compute_relevance(extract, profile, embedder)
    a = compute_actionability(extract)
    e = compute_evidence(paper)
    n = compute_novelty(extract, known_actions_embedding, embedder)
    lss, primary_layer, layer_scores = compute_layer_specificity(paper, extract)

    total = r * a * e * n

    logger.info(
        "RAEN+LSS for '%s': R=%.2f A=%.2f E=%.2f N=%.2f LSS=%.2f layer=%s total=%.3f",
        paper.title[:60],
        r, a, e, n, lss, primary_layer, total,
    )
    return RAENScore(
        R=r, A=a, E=e, N=n,
        LSS=lss, total=total,
        primary_layer=primary_layer,
        layer_scores=layer_scores,
    )
