"""Damasio layer classifier — maps findings to SOMA consciousness layers."""

from __future__ import annotations

import logging
from typing import Any

from soma.autoresearcher.extractor import PaperExtract
from soma.autoresearcher.llm import llm_call, parse_json_response
from soma.autoresearcher.scorer import RAENScore

logger = logging.getLogger(__name__)

VALID_LAYERS = (
    "Proto-Self",
    "Core Consciousness",
    "Extended Consciousness",
    "Relational Self",
)

# Map from scorer's internal layer keys to display names
LAYER_KEY_TO_NAME: dict[str, str] = {
    "proto_self": "Proto-Self",
    "core_consciousness": "Core Consciousness",
    "extended_consciousness": "Extended Consciousness",
    "relational_self": "Relational Self",
}

# Keyword rules for fast classification before LLM fallback
LAYER_RULES: dict[str, list[str]] = {
    "Proto-Self": [
        "autonomic regulation",
        "sleep physiology",
        "heart rate",
        "blood pressure",
        "cardiovascular",
        "homeostasis",
        "sleep apnea",
        "CPAP",
        "HRV",
        "inflammation",
        "metabolic",
        "EPA",
        "omega-3",
        "SHBG",
        "hormonal baseline",
        "deep sleep",
        "nocturnal hypoxemia",
    ],
    "Core Consciousness": [
        "alertness",
        "wakefulness",
        "cortisol",
        "circadian",
        "cognitive performance",
        "working memory",
        "attention",
        "morning",
        "sleep onset",
        "magnesium",
    ],
    "Extended Consciousness": [
        "long-term memory",
        "neuroprotection",
        "dementia risk",
        "executive function",
        "homocysteine neurotoxicity",
        "cognitive decline",
        "neurodegeneration",
        "Alzheimer",
        "brain aging",
    ],
    "Relational Self": [
        "testosterone",
        "libido",
        "mood",
        "depression",
        "social behavior",
        "empathy",
        "emotional regulation",
        "anxiety",
        "relational",
        "recovery",
        "addiction",
    ],
}

CLASSIFICATION_SYSTEM_PROMPT = """\
You are classifying a biomedical research finding into Damasio's four \
consciousness layers for the SOMA personal health system.

Layers:
- Proto-Self: autonomic regulation, sleep physiology, cardiovascular, \
metabolism, hormonal baseline
- Core Consciousness: alertness, circadian rhythm, cortisol, immediate \
cognitive performance, working memory
- Extended Consciousness: long-term memory, neuroprotection, dementia risk, \
executive function, future health trajectory
- Relational Self: testosterone/mood/libido, depression, social behavior, \
emotional regulation, recovery

Rules:
- One finding, one layer.
- Assign to the layer most directly affected by the mechanism of action.
- Tiebreaker: upstream wins (Proto-Self > Core > Extended > Relational).

Return JSON: {"layer": "<one of the four layer names>", "confidence": \
<0.0-1.0>, "reasoning": "<one sentence>"}
"""


def classify_from_score(score: RAENScore) -> str:
    """Fast classification using the LSS layer scores from the scorer.

    Uses the primary_layer from LSS computation, converting from
    internal key format to display name.
    """
    return LAYER_KEY_TO_NAME.get(score.primary_layer, "Proto-Self")


def _keyword_classify(extract: PaperExtract) -> str | None:
    """Attempt classification by keyword match. Returns None if ambiguous."""
    text = (
        f"{extract.intervention} {extract.outcome_measure} "
        f"{extract.population_description}"
    ).lower()

    matches: dict[str, int] = {}
    for layer, keywords in LAYER_RULES.items():
        count = sum(1 for kw in keywords if kw.lower() in text)
        if count > 0:
            matches[layer] = count

    if not matches:
        return None

    if len(matches) == 1:
        layer = next(iter(matches))
        logger.info("Keyword classification: %s (unambiguous)", layer)
        return layer

    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    if sorted_matches[0][1] >= sorted_matches[1][1] + 2:
        layer = sorted_matches[0][0]
        logger.info(
            "Keyword classification: %s (dominant, %d vs %d)",
            layer,
            sorted_matches[0][1],
            sorted_matches[1][1],
        )
        return layer

    return None


async def classify_layer(
    extract: PaperExtract,
    score: RAENScore | None = None,
) -> str:
    """Classify a finding into one of Damasio's four consciousness layers.

    Uses LSS score first (if available), then keyword matching,
    then falls back to LLM for ambiguous cases.
    """
    # Use LSS from scorer if available and confident
    if score is not None and score.LSS > 0.2:
        layer = classify_from_score(score)
        logger.info("LSS classification: %s (LSS=%.2f)", layer, score.LSS)
        return layer

    # Try fast keyword classification
    keyword_result = _keyword_classify(extract)
    if keyword_result is not None:
        return keyword_result

    # LLM fallback
    finding = (
        f"{extract.intervention} → {extract.outcome_measure} "
        f"— {extract.effect_direction} effect"
    )

    raw = await llm_call(
        system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
        user_prompt=f"Finding: {finding}",
        max_tokens=256,
        temperature=0.2,
    )

    parsed: dict[str, Any] = parse_json_response(raw)
    layer = str(parsed.get("layer", ""))
    confidence = float(parsed.get("confidence", 0.0))
    reasoning = str(parsed.get("reasoning", ""))

    if layer not in VALID_LAYERS:
        logger.warning(
            "LLM returned invalid layer '%s', defaulting to Proto-Self",
            layer,
        )
        layer = "Proto-Self"

    logger.info(
        "LLM classification: %s (confidence=%.2f, reason='%s')",
        layer,
        confidence,
        reasoning[:80],
    )
    return layer
