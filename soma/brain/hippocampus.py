"""Hippocampus — episodic memory store + retrieval.

Encodes each brain cycle as a searchable episode in LanceDB.
Retrieves similar past moments and detects patterns.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import lancedb  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from soma.brain.embeddings import AffectVec, MemoryContext, SomaticEmbedding

logger = logging.getLogger(__name__)

LANCEDB_PATH = os.environ.get("LANCEDB_PATH", os.path.expanduser("~/.soma/lancedb"))
TABLE_NAME = "soma_episodes"
TOP_K = 3

_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


class HippocampusModule:
    """Encodes moments to episodic memory and retrieves similar past moments."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or LANCEDB_PATH
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.encoder = _get_encoder()

    def _get_table(self) -> Any:
        if TABLE_NAME in self.db.table_names():
            return self.db.open_table(TABLE_NAME)
        return None

    def encode_text(self, text: str) -> NDArray[np.float32]:
        return np.asarray(self.encoder.encode(text), dtype=np.float32)

    async def encode_and_store(
        self,
        somatic: SomaticEmbedding,
        affect: AffectVec,
        context: str = "",
    ) -> None:
        """Write current moment to episodic memory."""
        text = f"{somatic.description} | {affect.description} | {context}"
        vector = self.encode_text(text).tolist()

        record = {
            "vector": vector,
            "text": text,
            "valence": float(affect.valence),
            "arousal": float(affect.arousal),
            "drive": affect.dominant_drive,
            "rmssd": float(somatic.rmssd),
            "rhr": float(somatic.rhr),
            "load": float(somatic.load),
            "source": "soma_cycle",
            "timestamp": datetime.now().isoformat(),
        }

        table = self._get_table()
        if table is not None:
            table.add([record])
        else:
            self.db.create_table(TABLE_NAME, [record])

        logger.debug("Episode stored: %s", text[:80])

    async def retrieve(
        self,
        query_vector: NDArray[np.float32] | None = None,
        query_text: str | None = None,
        top_k: int = TOP_K,
    ) -> MemoryContext:
        """Find most similar past moments."""
        table = self._get_table()
        if table is None:
            return self._empty_context()

        if query_vector is None and query_text is not None:
            query_vector = self.encode_text(query_text)
        elif query_vector is None:
            return self._empty_context()

        # Pad/truncate to match embedding dim if needed
        expected_dim = self.encoder.get_sentence_embedding_dimension() or 384
        if len(query_vector) != expected_dim:
            padded = np.zeros(expected_dim, dtype=np.float32)
            n = min(len(query_vector), expected_dim)
            padded[:n] = query_vector[:n]
            query_vector = padded

        try:
            results = table.search(query_vector.tolist()).limit(top_k + 2).to_list()
        except Exception as e:
            logger.warning("Hippocampus search failed: %s", e)
            return self._empty_context()

        # Filter valid results
        valid = [r for r in results if r.get("text")][:top_k]

        pattern = self._detect_pattern(valid)
        recency = self._recency_weight(valid)

        # Build 128-dim context vector from retrieved memories
        context_vec = np.zeros(128, dtype=np.float32)
        if valid:
            vecs = [np.array(r.get("vector", [0] * 128)[:128], dtype=np.float32) for r in valid]
            context_vec = np.mean(vecs, axis=0).astype(np.float32)

        description = self._describe(valid, pattern)

        return MemoryContext(
            similar_moments=[
                {
                    "text": r.get("text", ""),
                    "valence": r.get("valence", 0),
                    "arousal": r.get("arousal", 0),
                    "drive": r.get("drive", ""),
                    "timestamp": r.get("timestamp", ""),
                }
                for r in valid
            ],
            recency_weight=recency,
            pattern_note=pattern,
            vector=context_vec,
            description=description,
        )

    def _detect_pattern(self, results: list[dict]) -> str | None:
        """Detect if this pattern has notable prior outcomes."""
        if len(results) < 2:
            return None

        drives = [r.get("drive", "") for r in results]
        valences = [r.get("valence", 0) for r in results]

        # Check if similar states consistently negative
        if all(v < -0.2 for v in valences):
            most_common_drive = max(set(drives), key=drives.count) if drives else "unknown"
            return f"Past similar states were consistently negative, driven by {most_common_drive}."

        if all(v > 0.2 for v in valences):
            return "Past similar states were consistently positive."

        return None

    def _recency_weight(self, results: list[dict]) -> float:
        """How much recent memory dominates the retrieved set."""
        if not results:
            return 0.0
        # Simple: if most recent result is very recent (today), high weight
        try:
            latest = results[0].get("timestamp", "")
            if latest and latest[:10] == datetime.now().strftime("%Y-%m-%d"):
                return 0.8
            return 0.4
        except Exception:
            return 0.5

    def _describe(self, results: list[dict], pattern: str | None) -> str:
        """Generate human-readable description of memory context."""
        if not results:
            return "No similar past moments found."

        parts = [f"{len(results)} similar past moments found."]

        for i, r in enumerate(results[:3], 1):
            text = r.get("text", "")[:100]
            ts = r.get("timestamp", "")[:10]
            parts.append(f"[{ts}] {text}")

        if pattern:
            parts.append(f"Pattern: {pattern}")

        return " ".join(parts)

    def _empty_context(self) -> MemoryContext:
        return MemoryContext(
            similar_moments=[],
            recency_weight=0.0,
            pattern_note=None,
            vector=np.zeros(128, dtype=np.float32),
            description="No episodic memory available yet.",
        )
