"""Prefrontal cortex — reasoning and output layer.

Routes to Qwen3 (local, fast) for routine cycles.
Routes to Claude via OpenRouter for anomalies and explicit queries.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv

from soma.brain.embeddings import AffectiveEmbedding, MemoryContext, PFCOutput

logger = logging.getLogger(__name__)

load_dotenv(".env.local")
load_dotenv(".env")

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QWEN_MODEL = os.environ.get("PFC_LOCAL_MODEL", "qwen3:8b")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CLAUDE_MODEL = os.environ.get("PFC_CLAUDE_MODEL", "anthropic/claude-sonnet-4")
ANOMALY_THRESHOLD = float(os.environ.get("ANOMALY_THRESHOLD_ZSCORE", "1.5"))

PFC_SYSTEM_TEMPLATE = """You are SOMA's prefrontal cortex — the reasoning and output layer.

Current affective state:
{affective_description}

Relevant past moments:
{memory_description}

Your role: integrate this information and generate a useful output.
Be specific to the state, not generic. Reference the somatic signal.
If something is notable, say so directly.

{task_instruction}

Respond ONLY in JSON:
{{"recommendation": str_or_null, "anomaly_flag": bool, "anomaly_description": str_or_null, "prediction": str_or_null, "question": str_or_null}}"""

TASK_ROUTINE = "In 2-3 sentences, note what is worth attending to in the current state. If everything is normal, say so briefly."
TASK_ANOMALY = "An anomaly has been detected. Describe what the pattern suggests, what might be causing it, and what to do right now. Be concrete."
TASK_QUERY = "The patient asked: {query}\nAnswer directly, using the affective state as context."


class PrefrontalModule:
    """Reasoning layer — generates recommendations, anomaly flags, and answers."""

    def detect_anomaly(self, embedding: AffectiveEmbedding) -> bool:
        """Check if current state is anomalous."""
        return embedding.somatic_load > 0.7 or abs(embedding.valence) > 0.7

    async def process(
        self,
        embedding: AffectiveEmbedding,
        memory: MemoryContext | None = None,
        query: str | None = None,
    ) -> PFCOutput:
        """Generate PFC output for this cycle."""
        is_anomaly = self.detect_anomaly(embedding)
        use_claude = is_anomaly or query is not None

        # Build prompt
        if query:
            task = TASK_QUERY.format(query=query)
        elif is_anomaly:
            task = TASK_ANOMALY
        else:
            task = TASK_ROUTINE

        memory_desc = memory.description if memory else "No memory context."
        prompt = PFC_SYSTEM_TEMPLATE.format(
            affective_description=embedding.description,
            memory_description=memory_desc,
            task_instruction=task,
        )

        if use_claude:
            model_used = CLAUDE_MODEL
            raw = await self._call_claude(prompt)
        else:
            model_used = QWEN_MODEL
            raw = await self._call_ollama(prompt)

        if raw is None:
            return self._default_output(embedding, model_used)

        return self._parse_output(raw, embedding, model_used, is_anomaly)

    async def _call_ollama(self, prompt: str) -> str | None:
        """Call local Qwen3 via Ollama."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={
                        "model": QWEN_MODEL,
                        "prompt": prompt,
                        "system": "You are SOMA's prefrontal cortex. Respond only in JSON. /no_think",
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except Exception as e:
            logger.warning("PFC Ollama failed: %s", e)
            return None

    async def _call_claude(self, prompt: str) -> str | None:
        """Call Claude via OpenRouter for important decisions."""
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("No OPENROUTER_API_KEY, falling back to Ollama")
            return await self._call_ollama(prompt)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": CLAUDE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0.5,
                    },
                )
                resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            return choices[0].get("message", {}).get("content", "") if choices else None
        except Exception as e:
            logger.warning("PFC Claude failed: %s, falling back to Ollama", e)
            return await self._call_ollama(prompt)

    def _parse_output(
        self, raw: str, embedding: AffectiveEmbedding, model: str, is_anomaly: bool
    ) -> PFCOutput:
        try:
            clean = re.sub(r"```(?:json)?\s*\n?", "", raw.strip())
            clean = re.sub(r"\n?```\s*$", "", clean)
            data = json.loads(clean)

            return PFCOutput(
                recommendation=data.get("recommendation"),
                anomaly_flag=data.get("anomaly_flag", is_anomaly),
                anomaly_description=data.get("anomaly_description"),
                prediction=data.get("prediction"),
                question_for_patient=data.get("question"),
                model_used=model,
                source_embedding=embedding,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("PFC parse failed: %s", e)
            # Use raw text as recommendation
            return PFCOutput(
                recommendation=raw[:500] if raw else None,
                anomaly_flag=is_anomaly,
                anomaly_description=None,
                prediction=None,
                question_for_patient=None,
                model_used=model,
                source_embedding=embedding,
            )

    def _default_output(self, embedding: AffectiveEmbedding, model: str) -> PFCOutput:
        return PFCOutput(
            recommendation="System processing. No output this cycle.",
            anomaly_flag=False,
            anomaly_description=None,
            prediction=None,
            question_for_patient=None,
            model_used=model,
            source_embedding=embedding,
        )
