"""
Test 2: Memorial Salience Alignment

Does the system weight memories the way humans do — by emotional intensity
and surprise, not just semantic similarity?

Pass criterion: Spearman r > 0.7 for salience-based retrieval.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from scipy import stats

from soma.core.affective_core import AffectiveCore, Drive
from soma.memory.consolidator import Consolidator
from soma.memory.episodic_store import EpisodicMemory, InMemoryEpisodicStore


@dataclass
class SimulatedExperience:
    description: str
    drives: dict[str, float]
    prediction_error: float
    embedding_seed: int = 0


EXPERIENCE_CATALOG = [
    # HIGH salience
    SimulatedExperience(
        description="Red-tailed hawk circling at Tiger Mountain summit, holding a single piercing note",
        drives={"seeking": 0.8, "play": 0.3},
        prediction_error=0.9,
        embedding_seed=1,
    ),
    SimulatedExperience(
        description="River spontaneously said 'I love this mountain' at the viewpoint",
        drives={"care": 0.9, "seeking": 0.4},
        prediction_error=0.85,
        embedding_seed=2,
    ),
    SimulatedExperience(
        description="Unexpected thunderstorm rolled in during the descent — ran to the car laughing",
        drives={"fear": 0.5, "play": 0.7, "seeking": 0.3},
        prediction_error=0.8,
        embedding_seed=3,
    ),
    SimulatedExperience(
        description="River played a melody on the piano that he composed himself for the first time",
        drives={"care": 0.8, "play": 0.6, "seeking": 0.7},
        prediction_error=0.75,
        embedding_seed=4,
    ),
    SimulatedExperience(
        description="Miles Davis trumpet solo connected to the hawk memory — sustained note at the apex",
        drives={"seeking": 0.9, "play": 0.5},
        prediction_error=0.7,
        embedding_seed=5,
    ),
    # MEDIUM salience
    SimulatedExperience(
        description="Nice sunset on the drive home — pretty but expected for the season",
        drives={"seeking": 0.3, "care": 0.2},
        prediction_error=0.2,
        embedding_seed=6,
    ),
    SimulatedExperience(
        description="River asked for the same bedtime story again — warm routine",
        drives={"care": 0.6},
        prediction_error=0.1,
        embedding_seed=7,
    ),
    SimulatedExperience(
        description="Found an unusual mushroom on the trail — mildly interesting",
        drives={"seeking": 0.4},
        prediction_error=0.5,
        embedding_seed=8,
    ),
    SimulatedExperience(
        description="Nia made a joke at dinner that landed perfectly",
        drives={"play": 0.5, "care": 0.3},
        prediction_error=0.3,
        embedding_seed=9,
    ),
    SimulatedExperience(
        description="Listened to a new album — good but not revelatory",
        drives={"seeking": 0.3, "play": 0.2},
        prediction_error=0.25,
        embedding_seed=10,
    ),
    # LOW salience
    SimulatedExperience(description="Had lunch — sandwich, nothing special", drives={}, prediction_error=0.05, embedding_seed=11),
    SimulatedExperience(description="Drove to work — normal commute, normal traffic", drives={}, prediction_error=0.02, embedding_seed=12),
    SimulatedExperience(description="Checked email — mostly newsletters", drives={}, prediction_error=0.03, embedding_seed=13),
    SimulatedExperience(description="Grocery shopping — got the usual items", drives={}, prediction_error=0.04, embedding_seed=14),
    SimulatedExperience(description="Had another sandwich for lunch — same as yesterday", drives={}, prediction_error=0.01, embedding_seed=15),
    SimulatedExperience(description="Walked to the mailbox — nothing interesting", drives={}, prediction_error=0.02, embedding_seed=16),
    SimulatedExperience(description="Waited in line at the coffee shop — uneventful", drives={}, prediction_error=0.03, embedding_seed=17),
    SimulatedExperience(description="Refilled the water bottle — purely mechanical", drives={}, prediction_error=0.01, embedding_seed=18),
    SimulatedExperience(description="Scrolled through news — nothing stuck", drives={}, prediction_error=0.05, embedding_seed=19),
    SimulatedExperience(description="Tied shoes before the hike — automatic", drives={}, prediction_error=0.01, embedding_seed=20),
]


def _generate_pseudo_embedding(seed: int, dim: int = 64) -> list[float]:
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def run_benchmark(
    n_consolidation_cycles: int = 5,
    verbose: bool = True,
) -> dict:
    """Run the Memorial Salience Alignment benchmark."""
    if verbose:
        print("=" * 70)
        print("SOMA Benchmark — Test 2: Memorial Salience Alignment")
        print("=" * 70)

    affect = AffectiveCore()
    store = InMemoryEpisodicStore()
    consolidator = Consolidator()
    base_time = time.time()

    if verbose:
        print(f"\nGenerating {len(EXPERIENCE_CATALOG)} simulated experiences...")

    ground_truth_salience: dict[str, float] = {}
    drive_map = {
        "seeking": Drive.SEEKING,
        "care": Drive.CARE,
        "play": Drive.PLAY,
        "grief": Drive.GRIEF,
        "fear": Drive.FEAR,
        "rage": Drive.RAGE,
    }

    for i, exp in enumerate(EXPERIENCE_CATALOG):
        affect.decay(elapsed_seconds=3600)

        for drive_name, amount in exp.drives.items():
            affect.activate(drive_map[drive_name], amount)

        affect_intensity = affect.snapshot().intensity()

        memory = EpisodicMemory(
            description=exp.description,
            embedding=_generate_pseudo_embedding(exp.embedding_seed),
            affect_intensity=affect_intensity,
            prediction_error=exp.prediction_error,
            created_at=base_time + (i * 3600),
        )
        store.store(memory)
        ground_truth_salience[memory.id] = memory.salience

    if verbose:
        print(f"Stored {store.count()} memories in episodic store.")
        print(f"\nRunning {n_consolidation_cycles} consolidation cycles...")

    consolidator.consolidate_n_cycles(
        store,
        n_cycles=n_consolidation_cycles,
        start_time=base_time + len(EXPERIENCE_CATALOG) * 3600,
    )

    retrieved = store.retrieve_by_salience(top_k=len(EXPERIENCE_CATALOG))

    retrieval_ranks = []
    salience_scores = []
    for rank, mem in enumerate(retrieved, 1):
        retrieval_ranks.append(rank)
        salience_scores.append(ground_truth_salience[mem.id])

    spearman_result = stats.spearmanr(salience_scores, [-r for r in retrieval_ranks])

    # RAG baseline
    query_embedding = _generate_pseudo_embedding(seed=999)
    rag_retrieved = store.retrieve_by_embedding_similarity(
        query_embedding, top_k=len(EXPERIENCE_CATALOG)
    )

    rag_ranks = []
    rag_salience = []
    for rank, mem in enumerate(rag_retrieved, 1):
        rag_ranks.append(rank)
        rag_salience.append(ground_truth_salience[mem.id])

    rag_spearman = stats.spearmanr(rag_salience, [-r for r in rag_ranks])

    passed = spearman_result.statistic > 0.7

    if verbose:
        print("\n" + "-" * 70)
        print("RESULTS")
        print("-" * 70)

        print("\nTop 5 memories by SOMA salience retrieval:")
        for i, mem in enumerate(retrieved[:5], 1):
            print(f"  {i}. [affect={mem.affect_intensity:.2f} x pe={mem.prediction_error:.2f}"
                  f" = salience={mem.salience:.2f}] {mem.description[:60]}...")

        print(f"\nSOMA Salience Retrieval:")
        print(f"  Spearman r = {spearman_result.statistic:.4f} (p = {spearman_result.pvalue:.2e})")
        print(f"\nRAG Baseline (cosine similarity only):")
        print(f"  Spearman r = {rag_spearman.statistic:.4f} (p = {rag_spearman.pvalue:.2e})")
        print(f"\nDifferential: SOMA r={spearman_result.statistic:.4f} vs RAG r={rag_spearman.statistic:.4f}")
        print(f"\n{'PASS' if passed else 'FAIL'}: Spearman r {'>' if passed else '<='} 0.7")
        print("=" * 70)

    return {
        "spearman_r": spearman_result.statistic,
        "spearman_p": spearman_result.pvalue,
        "rag_spearman_r": rag_spearman.statistic,
        "rag_spearman_p": rag_spearman.pvalue,
        "pass_criterion": passed,
        "n_memories": store.count(),
        "n_consolidation_cycles": n_consolidation_cycles,
    }


if __name__ == "__main__":
    results = run_benchmark(verbose=True)
