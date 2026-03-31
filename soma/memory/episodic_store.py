"""
Episodic Memory Store — Memories tagged with affect, prediction error, and relational context.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class EpisodicMemory:
    """A single episodic memory with full SOMA metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Content
    description: str = ""
    embedding: list[float] = field(default_factory=list)

    # SOMA tags
    affect_intensity: float = 0.0
    prediction_error: float = 0.0
    relational_context: dict[str, float] = field(default_factory=dict)

    # Consolidation
    consolidation_score: float = 0.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def salience(self) -> float:
        """Salience = affect_intensity * prediction_error."""
        return self.affect_intensity * self.prediction_error


class InMemoryEpisodicStore:
    """In-memory episodic store for testing and benchmarking."""

    def __init__(self) -> None:
        self.memories: dict[str, EpisodicMemory] = {}

    def store(self, memory: EpisodicMemory) -> str:
        """Store an episodic memory. Returns the memory ID."""
        self.memories[memory.id] = memory
        return memory.id

    def get(self, memory_id: str) -> EpisodicMemory | None:
        return self.memories.get(memory_id)

    def all_memories(self) -> list[EpisodicMemory]:
        return list(self.memories.values())

    def retrieve_by_salience(self, top_k: int = 10) -> list[EpisodicMemory]:
        """Retrieve memories ranked by consolidation_score."""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.consolidation_score,
            reverse=True,
        )
        for mem in sorted_memories[:top_k]:
            mem.access_count += 1
            mem.last_accessed = time.time()
        return sorted_memories[:top_k]

    def retrieve_by_embedding_similarity(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[EpisodicMemory]:
        """Vanilla RAG-style retrieval: cosine similarity only."""
        def cosine_sim(a: list[float], b: list[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = [
            (mem, cosine_sim(query_embedding, mem.embedding))
            for mem in self.memories.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in scored[:top_k]]

    def count(self) -> int:
        return len(self.memories)
