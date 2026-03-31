"""Shared LLM client for OpenRouter (claude-opus-4-6)."""

from __future__ import annotations

import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env.local then .env (first found wins per variable)
load_dotenv(".env.local")
load_dotenv(".env")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-opus-4-6"


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set"
        )
    return key


async def llm_call(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Make an async LLM call to OpenRouter and return the text response.

    Logs token usage per CLAUDE.md conventions.
    """
    api_key = _get_api_key()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()

    data = resp.json()

    # Log token usage
    usage = data.get("usage", {})
    tokens_used = usage.get("total_tokens", 0)
    logger.info("LLM call: %d tokens", tokens_used)

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("LLM returned no choices")

    content: str = choices[0].get("message", {}).get("content", "")
    return content


def parse_json_response(text: str) -> dict[str, object]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Try direct parse first
    try:
        result: dict[str, object] = json.loads(cleaned)
        return result
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON object from the text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
    if match:
        result = json.loads(match.group())
        return result

    raise json.JSONDecodeError("No valid JSON found in LLM response", cleaned, 0)
