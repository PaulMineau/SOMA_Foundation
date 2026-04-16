"""Thalamus router — routes signals to appropriate brain modules.

Uses Qwen3 via Ollama for fast local inference (<1s).
Falls back to rule-based routing if Ollama is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import httpx

from soma.brain.embeddings import SomaticEmbedding, ThalamusEmbedding

logger = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("THALAMUS_MODEL", "qwen3:8b")

SYSTEM_PROMPT = """You are the thalamus routing layer of SOMA. Assess incoming signals and decide how to weight them.

Given somatic signal, semantic context, and optional visual input:
1. Assign weights 0.0-1.0 to each signal type (must sum to ~1.0)
2. Classify the signal: resting | stress | exercise | emotional_arousal | unknown
3. Determine if this is a "low road" event — a fast alarm that should bypass higher reasoning (criteria: HRV drops >30% below baseline, or somatic load > 0.7)
4. Describe your routing decision in one sentence.

Respond ONLY in JSON:
{"biosensor_weight": float, "semantic_weight": float, "visual_weight": float, "low_road_flag": bool, "signal_classification": str, "description": str}

/no_think"""


async def _ollama_call(prompt: str, system: str = SYSTEM_PROMPT) -> str | None:
    """Call Ollama with timeout. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
    except Exception as e:
        logger.warning("Thalamus Ollama call failed: %s", e)
        return None


def _rule_based_routing(somatic: SomaticEmbedding) -> ThalamusEmbedding:
    """Fallback when Ollama is unavailable."""
    low_road = somatic.load > 0.7

    if somatic.load > 0.6:
        classification = "stress"
    elif somatic.rhr > 100:
        classification = "exercise"
    elif somatic.load < 0.2:
        classification = "resting"
    else:
        classification = "unknown"

    return ThalamusEmbedding(
        biosensor_weight=0.7,
        semantic_weight=0.3,
        visual_weight=0.0,
        low_road_flag=low_road,
        signal_classification=classification,
        description=f"Rule-based routing: {classification}, load={somatic.load:.2f}",
    )


async def route(
    somatic: SomaticEmbedding,
    semantic_context: str = "",
    visual_description: str = "",
) -> ThalamusEmbedding:
    """Route signals through thalamus. Uses Ollama if available, else rules."""
    prompt = (
        f"Somatic signal: {somatic.description}\n"
        f"Semantic context: {semantic_context or 'none'}\n"
        f"Visual input: {visual_description or 'none (camera inactive)'}\n"
        f"Somatic load: {somatic.load:.2f}, RMSSD: {somatic.rmssd:.0f}ms, RHR: {somatic.rhr:.0f}bpm"
    )

    raw = await _ollama_call(prompt)
    if raw is None:
        return _rule_based_routing(somatic)

    try:
        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        data = json.loads(clean)
        return ThalamusEmbedding(
            biosensor_weight=float(data.get("biosensor_weight", 0.7)),
            semantic_weight=float(data.get("semantic_weight", 0.3)),
            visual_weight=float(data.get("visual_weight", 0.0)),
            low_road_flag=bool(data.get("low_road_flag", False)),
            signal_classification=str(data.get("signal_classification", "unknown")),
            description=str(data.get("description", "Thalamus routing complete")),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Thalamus parse failed: %s, falling back to rules", e)
        return _rule_based_routing(somatic)
