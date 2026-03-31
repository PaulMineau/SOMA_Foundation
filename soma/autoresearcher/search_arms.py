"""Layer-specific parallel PubMed search arms.

Each Damasio layer gets its own search arm with targeted keyword queries.
All four arms run in parallel during the overnight loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from soma.autoresearcher.fetcher import (
    Paper,
    dedup_papers,
    fetch_pubmed,
    HTTPX_TIMEOUT,
    PUBMED_DELAY,
)

logger = logging.getLogger(__name__)


# PubMed MeSH-style queries per Damasio layer
LAYER_QUERIES: dict[str, list[str]] = {
    "proto_self": [
        '("heart rate variability" OR HRV) AND (interoception OR "autonomic nervous system")',
        '("free energy principle" OR "active inference") AND (homeostasis OR "body map")',
        '("cardiac afferent" OR baroreceptor) AND (brain OR consciousness)',
        '(interoception) AND ("predictive processing" OR "predictive coding")',
    ],
    "core_consciousness": [
        '("global workspace theory" OR "global neuronal workspace") AND (attention OR consciousness)',
        '("temporal binding" OR "specious present") AND (neural OR cognitive)',
        '("neural correlates of consciousness" OR NCC) AND (attention OR awareness)',
        '(consciousness) AND ("transformer" OR "attention mechanism") AND (artificial intelligence)',
    ],
    "extended_consciousness": [
        '("episodic memory") AND (consolidation OR "hippocampal replay")',
        '("autobiographical memory" OR "narrative self") AND (neural OR brain)',
        '("memory reconsolidation") AND (sleep OR "hippocampal-neocortical")',
        '("memory consolidation") AND sleep AND (systems OR hippocampus)',
    ],
    "relational_self": [
        '("attachment theory" OR "attachment style") AND (neuroscience OR neural)',
        '("co-regulation" OR "interpersonal synchrony") AND (physiology OR cardiac)',
        '("theory of mind" OR mentalizing) AND (neural OR brain)',
        '("parent-infant" OR "mother-infant") AND (co-regulation OR synchrony OR attachment)',
    ],
}


@dataclass
class SearchArmResult:
    """Result from a single layer search arm."""

    layer: str
    papers: list[Paper]
    queries_run: int
    total_fetched: int


async def run_search_arm(
    layer: str,
    queries: list[str] | None = None,
    timeout: float = HTTPX_TIMEOUT,
) -> SearchArmResult:
    """Run PubMed searches for a single Damasio layer.

    Args:
        layer: Layer key (e.g., "proto_self").
        queries: Override queries. If None, uses LAYER_QUERIES.
        timeout: HTTP client timeout.
    """
    if queries is None:
        queries = LAYER_QUERIES.get(layer, [])

    if not queries:
        logger.warning("No queries defined for layer '%s'", layer)
        return SearchArmResult(layer=layer, papers=[], queries_run=0, total_fetched=0)

    all_papers: list[Paper] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for query in queries:
            try:
                papers = await fetch_pubmed(client, query)
                all_papers.extend(papers)
                logger.info(
                    "Layer '%s' query fetched %d papers: %s",
                    layer, len(papers), query[:60],
                )
            except Exception:
                logger.exception(
                    "Search arm '%s' failed for query: %s", layer, query[:60]
                )
            await asyncio.sleep(PUBMED_DELAY)

    deduped = dedup_papers(all_papers)

    logger.info(
        "Search arm '%s' complete: %d papers (%d after dedup) from %d queries",
        layer, len(all_papers), len(deduped), len(queries),
    )

    return SearchArmResult(
        layer=layer,
        papers=deduped,
        queries_run=len(queries),
        total_fetched=len(deduped),
    )


async def run_all_search_arms(
    timeout: float = HTTPX_TIMEOUT,
) -> dict[str, SearchArmResult]:
    """Run all four Damasio layer search arms in parallel.

    Returns a dict mapping layer key to SearchArmResult.
    """
    logger.info("Starting all 4 layer search arms in parallel")

    tasks = {
        layer: asyncio.create_task(run_search_arm(layer, timeout=timeout))
        for layer in LAYER_QUERIES
    }

    results: dict[str, SearchArmResult] = {}
    for layer, task in tasks.items():
        try:
            results[layer] = await task
        except Exception:
            logger.exception("Search arm '%s' failed entirely", layer)
            results[layer] = SearchArmResult(
                layer=layer, papers=[], queries_run=0, total_fetched=0
            )

    total_papers = sum(r.total_fetched for r in results.values())
    logger.info(
        "All search arms complete: %d total papers across %d layers",
        total_papers, len(results),
    )

    return results
