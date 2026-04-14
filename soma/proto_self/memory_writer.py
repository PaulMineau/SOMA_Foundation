"""Memory writer — processes responses, extracts entities, stores exchanges.

After SOMA asks and the patient answers, this module:
1. Extracts named entities and emotional valence via LLM
2. Stores the full exchange as autobiographical memory in LanceDB
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

from soma.proto_self.autobiographical_store import store_exchange

logger = logging.getLogger(__name__)

load_dotenv(".env.local")
load_dotenv(".env")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("SOMA_PROBE_MODEL", "anthropic/claude-sonnet-4")


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return key


async def extract_entities_and_valence(
    probe_text: str,
    response_text: str,
) -> dict[str, Any]:
    """Extract named entities and emotional valence from an exchange."""
    api_key = _get_api_key()

    prompt = f"""Extract structured information from this exchange.

Probe: {probe_text}
Response: {response_text}

Return ONLY a JSON object with:
- entities: list of named things mentioned (people, places, projects, emotions)
- emotion_valence: float from -1.0 (very negative) to 1.0 (very positive)
- primary_topic: one of ["work", "family", "health", "substance", "relationship", "creative", "other"]
- stress_indicator: true if response indicates stress or difficulty"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        raw = choices[0].get("message", {}).get("content", "") if choices else ""

        clean = re.sub(r"```(?:json)?\s*\n?", "", raw.strip())
        clean = re.sub(r"\n?```\s*$", "", clean)
        return json.loads(clean)
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
        return {
            "entities": [],
            "emotion_valence": 0.0,
            "primary_topic": "other",
            "stress_indicator": False,
        }


async def write_memory(
    anomaly: dict,
    state_info: dict,
    probe_text: str,
    response_text: str,
    session_label: str = "",
    db_path: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Full memory writing pipeline.

    Returns (memory_id, extracted_data).
    """
    extracted = await extract_entities_and_valence(probe_text, response_text)

    memory_id = store_exchange(
        anomaly_type="hrv_anomaly",
        metric=anomaly.get("metric", ""),
        value=anomaly.get("value", 0),
        baseline=anomaly.get("baseline", 0),
        deviation=anomaly.get("deviation", 0),
        body_state=state_info.get("state", ""),
        probe_text=probe_text,
        response_text=response_text,
        session_label=session_label,
        entities=extracted.get("entities", []),
        emotion_valence=extracted.get("emotion_valence", 0.0),
        db_path=db_path,
    )

    logger.info(
        "Memory %s: topic=%s valence=%.2f entities=%s",
        memory_id,
        extracted.get("primary_topic"),
        extracted.get("emotion_valence", 0),
        extracted.get("entities", []),
    )

    return memory_id, extracted
