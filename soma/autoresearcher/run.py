"""CLI entrypoint — runs the full SOMA AutoResearcher loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from soma.autoresearcher.convergence import should_converge
from soma.autoresearcher.extractor import PaperExtract, extract_paper
from soma.autoresearcher.fetcher import Paper, fetch_papers
from soma.autoresearcher.query_gen import generate_followup_queries, generate_queries
from soma.autoresearcher.scorer import (
    RAENScore,
    build_known_actions_embedding,
    score_paper,
)
from soma.autoresearcher.seed import BiomarkerProfile, load_profile

logger = logging.getLogger(__name__)

RELEVANCE_DISCARD_THRESHOLD = 0.4


@dataclass
class ScoredPaper:
    """A paper with its extraction and RAEN score."""

    paper: Paper
    extract: PaperExtract
    score: RAENScore
    soma_layer: str = "Proto-Self"


@dataclass
class LoopState:
    """Accumulated state across research loop iterations."""

    all_scored: list[ScoredPaper] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)

    def add(self, sp: ScoredPaper) -> None:
        self.all_scored.append(sp)
        self.score_history.append(sp.score.total)

        if sp.paper.pmid:
            self.seen_ids.add(sp.paper.pmid)
        if sp.paper.doi:
            self.seen_ids.add(sp.paper.doi)

    def is_seen(self, paper: Paper) -> bool:
        if paper.pmid and paper.pmid in self.seen_ids:
            return True
        if paper.doi and paper.doi in self.seen_ids:
            return True
        return False

    def top_findings(self, n: int = 3) -> list[str]:
        """Return top N finding summaries for follow-up query generation."""
        ranked = sorted(
            self.all_scored, key=lambda sp: sp.score.total, reverse=True
        )
        summaries: list[str] = []
        for sp in ranked[:n]:
            summaries.append(
                f"{sp.extract.intervention} — "
                f"{sp.extract.outcome_measure} "
                f"(RAEN={sp.score.total:.2f}, {sp.paper.study_type})"
            )
        return summaries


async def run_loop(
    profile: BiomarkerProfile,
    max_iterations: int = 5,
) -> list[ScoredPaper]:
    """Execute the full research loop (profile-driven queries)."""
    from sentence_transformers import SentenceTransformer

    logger.info(
        "Starting research loop for profile '%s' (max %d iterations)",
        profile.profile_id,
        max_iterations,
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    known_emb = build_known_actions_embedding(profile, embedder)
    state = LoopState()

    for iteration in range(max_iterations):
        logger.info("=== Iteration %d/%d ===", iteration + 1, max_iterations)

        if iteration == 0:
            queries = await generate_queries(profile)
        else:
            queries = await generate_followup_queries(
                profile, state.top_findings()
            )

        papers = await fetch_papers(queries)
        new_papers = [p for p in papers if not state.is_seen(p)]
        logger.info(
            "Fetched %d papers (%d new)", len(papers), len(new_papers)
        )

        if not new_papers:
            logger.info("No new papers found — stopping early")
            break

        for paper in new_papers:
            if not paper.abstract:
                continue

            try:
                extract = await extract_paper(paper, profile)
            except Exception:
                logger.exception(
                    "Extraction failed for '%s'", paper.title[:50]
                )
                continue

            raen = score_paper(
                paper, extract, profile, known_emb, embedder
            )

            if raen.R < RELEVANCE_DISCARD_THRESHOLD:
                continue

            state.add(ScoredPaper(paper=paper, extract=extract, score=raen))

        if should_converge(state.score_history, iteration, max_iterations):
            break

    # Classify each finding to a SOMA layer
    from soma.autoresearcher.damasio import classify_layer

    for sp in state.all_scored:
        try:
            sp.soma_layer = await classify_layer(sp.extract, sp.score)
        except Exception:
            logger.exception(
                "Layer classification failed for '%s'",
                sp.extract.intervention[:50],
            )
            sp.soma_layer = "Proto-Self"

    # Store to LanceDB
    from soma.autoresearcher.memory import store_findings

    findings = [
        (sp.paper, sp.extract, sp.score, sp.soma_layer)
        for sp in state.all_scored
    ]
    stored = store_findings(findings, embedder)
    logger.info("Stored %d findings to LanceDB", stored)

    ranked = sorted(
        state.all_scored, key=lambda sp: sp.score.total, reverse=True
    )
    logger.info(
        "Loop complete: %d papers scored, top score %.3f",
        len(ranked),
        ranked[0].score.total if ranked else 0.0,
    )
    return ranked


async def run_overnight(
    profile: BiomarkerProfile,
) -> list[ScoredPaper]:
    """Run the overnight loop: all 4 layer search arms + score + store + export.

    This is the core Phase 0 overnight loop:
    1. Run all 4 Damasio layer search arms in parallel
    2. Extract and score each paper
    3. Classify to layers
    4. Store to LanceDB
    5. Export training corpus
    """
    from sentence_transformers import SentenceTransformer

    from soma.autoresearcher.corpus_export import export_scored_papers
    from soma.autoresearcher.damasio import classify_layer
    from soma.autoresearcher.memory import store_findings
    from soma.autoresearcher.search_arms import run_all_search_arms

    logger.info("=== SOMA Overnight Loop Starting ===")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    known_emb = build_known_actions_embedding(profile, embedder)

    # 1. Run all search arms in parallel
    arm_results = await run_all_search_arms()

    # 2. Extract and score all papers
    all_scored: list[ScoredPaper] = []

    for layer_key, arm_result in arm_results.items():
        logger.info(
            "Processing %d papers from '%s' search arm",
            len(arm_result.papers), layer_key,
        )

        for paper in arm_result.papers:
            if not paper.abstract:
                continue

            try:
                extract = await extract_paper(paper, profile)
            except Exception:
                logger.exception(
                    "Extraction failed for '%s'", paper.title[:50]
                )
                continue

            raen = score_paper(
                paper, extract, profile, known_emb, embedder
            )

            if raen.R < RELEVANCE_DISCARD_THRESHOLD:
                continue

            sp = ScoredPaper(paper=paper, extract=extract, score=raen)

            # 3. Classify layer (using LSS first, then keyword/LLM fallback)
            try:
                sp.soma_layer = await classify_layer(extract, raen)
            except Exception:
                logger.exception("Layer classification failed")
                sp.soma_layer = "Proto-Self"

            all_scored.append(sp)

    # 4. Store to LanceDB
    findings = [
        (sp.paper, sp.extract, sp.score, sp.soma_layer)
        for sp in all_scored
    ]
    stored = store_findings(findings, embedder)
    logger.info("Stored %d findings to LanceDB", stored)

    # 5. Export training corpus
    export_records: list[dict[str, Any]] = []
    for sp in all_scored:
        export_records.append({
            "abstract": sp.paper.abstract[:2000],
            "title": sp.paper.title,
            "intervention": sp.extract.intervention,
            "outcome": sp.extract.outcome_measure,
            "raen_total": sp.score.total,
            "raen_r": sp.score.R,
            "raen_a": sp.score.A,
            "raen_e": sp.score.E,
            "raen_n": sp.score.N,
            "raen_lss": sp.score.LSS,
            "soma_layer": sp.soma_layer,
            "primary_layer_key": sp.score.primary_layer,
            "study_type": sp.paper.study_type,
            "year": sp.paper.year,
        })
    exported = export_scored_papers(export_records)
    logger.info("Exported %d records to training corpus", exported)

    ranked = sorted(all_scored, key=lambda sp: sp.score.total, reverse=True)

    # Summary by layer
    layer_counts: dict[str, int] = {}
    for sp in all_scored:
        layer_counts[sp.soma_layer] = layer_counts.get(sp.soma_layer, 0) + 1

    logger.info("=== Overnight Loop Complete ===")
    logger.info("Total papers scored: %d", len(all_scored))
    logger.info("Papers by layer: %s", json.dumps(layer_counts, indent=2))
    if ranked:
        logger.info("Top score: %.3f", ranked[0].score.total)

    return ranked


def print_briefing(ranked: list[ScoredPaper]) -> None:
    """Print a simple text briefing to stdout."""
    if not ranked:
        print("No papers scored above threshold.")
        return

    print("\n" + "=" * 60)
    print("SOMA AutoResearcher — Briefing")
    print("=" * 60)

    top = ranked[:10]
    for i, sp in enumerate(top, 1):
        print(f"\n{i}. {sp.extract.intervention}")
        print(f"   RAEN: {sp.score.total:.3f} "
              f"(R={sp.score.R:.2f} A={sp.score.A:.2f} "
              f"E={sp.score.E:.2f} N={sp.score.N:.2f} "
              f"LSS={sp.score.LSS:.2f})")
        print(f"   Evidence: {sp.paper.study_type}, {sp.paper.year}")
        print(f"   SOMA layer: {sp.soma_layer}")
        print(f"   Outcome: {sp.extract.outcome_measure}")
        print(f"   Direction: {sp.extract.effect_direction}", end="")
        if sp.extract.effect_size is not None:
            print(f" (effect size: {sp.extract.effect_size}%)")
        else:
            print()
        if not sp.extract.safe_for_profile:
            print("   !! Safety flag — discuss with physician")
        if sp.extract.conflicts_with_supplements:
            print(
                f"   Conflicts: {', '.join(sp.extract.conflicts_with_supplements)}"
            )

    print("\n" + "-" * 60)
    print(
        "Discuss top-scored interventions with your physician "
        "before changing protocols."
    )
    print("-" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOMA AutoResearcher — autonomous health literature agent"
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("PROFILE_PATH", "data/patient_876.json"),
        help="Path to biomarker profile JSON",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.environ.get("MAX_ITERATIONS", "5")),
        help="Maximum research loop iterations",
    )
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Run overnight loop (all 4 Damasio layer search arms)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    profile = load_profile(args.profile)

    if args.overnight:
        ranked = asyncio.run(run_overnight(profile))
    else:
        ranked = asyncio.run(run_loop(profile, args.max_iterations))

    print_briefing(ranked)


if __name__ == "__main__":
    main()
