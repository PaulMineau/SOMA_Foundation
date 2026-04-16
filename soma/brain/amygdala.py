"""Amygdala module — generates affective state from body signal + context.

Two processing paths:
- Low road (fast): direct from biosensor, no LLM, <50ms
- High road (slow): Qwen3 classifies affect from somatic + semantic context
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import httpx

from soma.brain.embeddings import AffectVec, SomaticEmbedding, ThalamusEmbedding

logger = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("AMYGDALA_MODEL", "qwen3:8b")

SYSTEM_PROMPT = """You are the amygdala module of SOMA. Integrate body signal and context to generate an affective state.

Valence: -1.0 (aversive/threatening) to +1.0 (pleasant/safe)
Arousal: 0.0 (calm/sleepy) to 1.0 (activated/alert)
Dominant drive (Panksepp): SEEKING | CARE | PLAY | GRIEF | FEAR | RAGE

Assess the somatic signal and semantic context. Describe the felt quality of this moment.

Respond ONLY in JSON:
{"valence": float, "arousal": float, "dominant_drive": str, "low_road_contribution": float, "high_road_contribution": float, "description": str}

/no_think"""


def _low_road_affect(somatic: SomaticEmbedding) -> AffectVec:
    """Fast path — compute affect directly from biosensor values."""
    # High load = negative valence, high arousal
    valence = -somatic.load  # load 0.8 -> valence -0.8
    arousal = min(1.0, somatic.load + 0.2)

    if somatic.load > 0.8:
        drive = "FEAR"
    elif somatic.load > 0.5:
        drive = "SEEKING"  # Vigilant seeking
    else:
        drive = "SEEKING"

    return AffectVec(
        valence=round(valence, 2),
        arousal=round(arousal, 2),
        dominant_drive=drive,
        low_road_contribution=1.0,
        high_road_contribution=0.0,
        description=f"Low road: valence {valence:.2f}, arousal {arousal:.2f}. "
                    f"Body under load ({somatic.load:.2f}). Drive: {drive}.",
    )


def _rule_based_affect(somatic: SomaticEmbedding) -> AffectVec:
    """Fallback when Ollama is unavailable."""
    if somatic.load > 0.6:
        valence, arousal, drive = -0.4, 0.6, "SEEKING"
    elif somatic.load < 0.2:
        valence, arousal, drive = 0.3, 0.2, "CARE"
    else:
        valence, arousal, drive = 0.0, 0.4, "SEEKING"

    return AffectVec(
        valence=valence,
        arousal=arousal,
        dominant_drive=drive,
        low_road_contribution=0.5,
        high_road_contribution=0.5,
        description=f"Rule-based affect: valence {valence}, arousal {arousal}, {drive}.",
    )


async def process(
    somatic: SomaticEmbedding,
    routing: ThalamusEmbedding,
    semantic_context: str = "",
) -> AffectVec:
    """Process affect through amygdala. Low road if flagged, else high road."""
    # Low road — bypass LLM
    if routing.low_road_flag:
        logger.info("Amygdala LOW ROAD: somatic load %.2f", somatic.load)
        return _low_road_affect(somatic)

    # High road — Qwen3 classification
    prompt = (
        f"Somatic state: {somatic.description}\n"
        f"Semantic context: {semantic_context or 'none'}\n"
        f"Thalamus routing: {routing.description}\n"
        f"Signal classification: {routing.signal_classification}"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
    except Exception as e:
        logger.warning("Amygdala Ollama failed: %s, using rules", e)
        return _rule_based_affect(somatic)

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        data = json.loads(clean)
        return AffectVec(
            valence=max(-1.0, min(1.0, float(data.get("valence", 0)))),
            arousal=max(0.0, min(1.0, float(data.get("arousal", 0.5)))),
            dominant_drive=str(data.get("dominant_drive", "SEEKING")),
            low_road_contribution=float(data.get("low_road_contribution", 0.3)),
            high_road_contribution=float(data.get("high_road_contribution", 0.7)),
            description=str(data.get("description", "Affect classified via high road.")),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Amygdala parse failed: %s", e)
        return _rule_based_affect(somatic)
