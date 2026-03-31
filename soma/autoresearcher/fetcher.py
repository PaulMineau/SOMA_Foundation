"""Async paper fetcher for PubMed E-utilities and Semantic Scholar."""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
S2_BASE = "https://api.semanticscholar.org/graph/v1"

MAX_PUBMED_RESULTS = 20
HTTPX_TIMEOUT = 30.0

# Rate limit delays (seconds between requests)
PUBMED_DELAY = 0.35  # 3 req/s max
S2_DELAY = 1.1  # 1 req/s max

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


@dataclass
class Paper:
    """Structured representation of a fetched paper."""

    pmid: str | None
    doi: str | None
    title: str
    abstract: str
    year: int
    study_type: str  # "RCT", "meta-analysis", "observational", "case", "review"
    citation_count: int
    industry_funded: bool
    full_text: str | None
    source: str  # "pubmed" | "semantic_scholar"


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """Make an HTTP request with retry on 429 rate limiting."""
    resp: httpx.Response | None = None
    for attempt in range(MAX_RETRIES):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code != 429:
            return resp
        wait = RETRY_BACKOFF * (2 ** attempt)
        logger.warning(
            "Rate limited (429) on %s, retrying in %.1fs (attempt %d/%d)",
            url.split("?")[0].split("/")[-1],
            wait,
            attempt + 1,
            MAX_RETRIES,
        )
        await asyncio.sleep(wait)

    assert resp is not None
    return resp


# ---------------------------------------------------------------------------
# PubMed E-utilities
# ---------------------------------------------------------------------------


async def _pubmed_esearch(
    client: httpx.AsyncClient, query: str
) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    resp = await _request_with_retry(
        client,
        "GET",
        f"{PUBMED_BASE}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": str(MAX_PUBMED_RESULTS),
            "retmode": "json",
        },
    )
    if resp.status_code == 429:
        logger.warning("PubMed esearch rate limited after retries: '%s'", query)
        return []
    resp.raise_for_status()

    data: dict[str, Any] = resp.json()
    id_list: list[str] = data.get("esearchresult", {}).get("idlist", [])
    logger.info("PubMed esearch '%s': %d PMIDs", query, len(id_list))
    return id_list


async def _pubmed_esummary(
    client: httpx.AsyncClient, pmids: list[str]
) -> list[Paper]:
    """Fetch summary metadata for a batch of PMIDs."""
    if not pmids:
        return []

    resp = await _request_with_retry(
        client,
        "GET",
        f"{PUBMED_BASE}/esummary.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        },
    )
    if resp.status_code == 429:
        logger.warning("PubMed esummary rate limited after retries")
        return []
    resp.raise_for_status()

    data: dict[str, Any] = resp.json()
    result_block: dict[str, Any] = data.get("result", {})

    papers: list[Paper] = []
    for pmid in pmids:
        rec: dict[str, Any] | None = result_block.get(pmid)
        if rec is None:
            continue

        title: str = rec.get("title", "")
        year = _extract_year(rec)
        doi = _extract_doi(rec)
        study_type = _guess_study_type_from_pubtype(rec.get("pubtype", []))

        papers.append(
            Paper(
                pmid=pmid,
                doi=doi,
                title=title,
                abstract="",
                year=year,
                study_type=study_type,
                citation_count=0,
                industry_funded=False,
                full_text=None,
                source="pubmed",
            )
        )

    return papers


async def _pubmed_efetch_abstracts(
    client: httpx.AsyncClient, pmids: list[str]
) -> dict[str, str]:
    """Fetch abstracts for a batch of PMIDs. Returns {pmid: abstract}."""
    if not pmids:
        return {}

    resp = await _request_with_retry(
        client,
        "GET",
        f"{PUBMED_BASE}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        },
    )
    if resp.status_code == 429:
        logger.warning("PubMed efetch rate limited after retries")
        return {}
    resp.raise_for_status()

    abstracts: dict[str, str] = {}
    try:
        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            if pmid_el is None or pmid_el.text is None:
                continue
            pmid = pmid_el.text

            abstract_parts: list[str] = []
            for abstract_text in article.findall(".//AbstractText"):
                if abstract_text.text:
                    abstract_parts.append(abstract_text.text)

            if abstract_parts:
                abstracts[pmid] = " ".join(abstract_parts)
    except ET.ParseError:
        logger.warning("Failed to parse PubMed XML response")

    return abstracts


def _extract_year(rec: dict[str, Any]) -> int:
    """Extract publication year from esummary record."""
    pubdate: str = rec.get("pubdate", "")
    if pubdate and len(pubdate) >= 4:
        try:
            return int(pubdate[:4])
        except ValueError:
            pass
    sortpubdate: str = rec.get("sortpubdate", "")
    if sortpubdate and len(sortpubdate) >= 4:
        try:
            return int(sortpubdate[:4])
        except ValueError:
            pass
    return 0


def _extract_doi(rec: dict[str, Any]) -> str | None:
    """Extract DOI from esummary articleids."""
    for aid in rec.get("articleids", []):
        if aid.get("idtype") == "doi":
            val: str = aid.get("value", "")
            if val:
                return val
    return None


def _guess_study_type_from_pubtype(pubtypes: list[str]) -> str:
    """Map PubMed publication types to our study type taxonomy."""
    pubtypes_lower = [pt.lower() for pt in pubtypes]
    if any("meta-analysis" in pt for pt in pubtypes_lower):
        return "meta-analysis"
    if any("randomized controlled trial" in pt for pt in pubtypes_lower):
        return "RCT"
    if any("review" in pt for pt in pubtypes_lower):
        return "review"
    if any("case reports" in pt for pt in pubtypes_lower):
        return "case"
    if any("observational" in pt for pt in pubtypes_lower):
        return "observational"
    return "unknown"


async def fetch_pubmed(
    client: httpx.AsyncClient, query: str
) -> list[Paper]:
    """Run a PubMed search and return Papers with abstracts."""
    pmids = await _pubmed_esearch(client, query)
    if not pmids:
        return []

    await asyncio.sleep(PUBMED_DELAY)
    papers = await _pubmed_esummary(client, pmids)

    await asyncio.sleep(PUBMED_DELAY)
    abstracts = await _pubmed_efetch_abstracts(client, pmids)

    for paper in papers:
        if paper.pmid and paper.pmid in abstracts:
            paper.abstract = abstracts[paper.pmid]

    logger.info(
        "PubMed fetch complete for '%s': %d papers with abstracts",
        query,
        sum(1 for p in papers if p.abstract),
    )
    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

S2_FIELDS = "title,abstract,year,citationCount,publicationTypes,externalIds"


async def fetch_semantic_scholar(
    client: httpx.AsyncClient, query: str
) -> list[Paper]:
    """Search Semantic Scholar and return Papers."""
    resp = await _request_with_retry(
        client,
        "GET",
        f"{S2_BASE}/paper/search",
        params={
            "query": query,
            "limit": str(MAX_PUBMED_RESULTS),
            "fields": S2_FIELDS,
        },
    )
    if resp.status_code == 429:
        logger.warning("S2 rate limited after retries for query '%s'", query)
        return []
    resp.raise_for_status()

    data: dict[str, Any] = resp.json()
    raw_papers: list[dict[str, Any]] = data.get("data", [])

    papers: list[Paper] = []
    for rec in raw_papers:
        title: str = rec.get("title") or ""
        abstract: str = rec.get("abstract") or ""
        year: int = rec.get("year") or 0

        ext_ids: dict[str, str] = rec.get("externalIds") or {}
        pmid = ext_ids.get("PubMed")
        doi = ext_ids.get("DOI")

        pub_types: list[str] = rec.get("publicationTypes") or []
        study_type = _guess_s2_study_type(pub_types)

        papers.append(
            Paper(
                pmid=pmid,
                doi=doi,
                title=title,
                abstract=abstract,
                year=year,
                study_type=study_type,
                citation_count=rec.get("citationCount", 0) or 0,
                industry_funded=False,
                full_text=None,
                source="semantic_scholar",
            )
        )

    logger.info("S2 fetch complete for '%s': %d papers", query, len(papers))
    return papers


def _guess_s2_study_type(pub_types: list[str]) -> str:
    """Map Semantic Scholar publication types to our taxonomy."""
    types_lower = [pt.lower() for pt in pub_types]
    if "review" in types_lower:
        return "review"
    if "casereport" in types_lower:
        return "case"
    if "journalarticle" in types_lower:
        return "unknown"
    return "unknown"


# ---------------------------------------------------------------------------
# Unified fetch + dedup
# ---------------------------------------------------------------------------


def dedup_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers by PMID or DOI, preferring PubMed source."""
    seen_pmids: set[str] = set()
    seen_dois: set[str] = set()
    result: list[Paper] = []

    sorted_papers = sorted(papers, key=lambda p: p.source != "pubmed")

    for paper in sorted_papers:
        if paper.pmid and paper.pmid in seen_pmids:
            continue
        if paper.doi and paper.doi in seen_dois:
            continue

        if paper.pmid:
            seen_pmids.add(paper.pmid)
        if paper.doi:
            seen_dois.add(paper.doi)
        result.append(paper)

    deduped = len(papers) - len(result)
    if deduped > 0:
        logger.info("Deduplication removed %d papers", deduped)
    return result


async def fetch_papers(
    queries: list[str],
    timeout: float = HTTPX_TIMEOUT,
) -> list[Paper]:
    """Fetch papers from PubMed and Semantic Scholar for all queries."""
    all_papers: list[Paper] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for query in queries:
            try:
                papers = await fetch_pubmed(client, query)
                all_papers.extend(papers)
            except Exception:
                logger.exception("PubMed fetch failed for '%s'", query)
            await asyncio.sleep(PUBMED_DELAY)

        for query in queries:
            try:
                papers = await fetch_semantic_scholar(client, query)
                all_papers.extend(papers)
            except Exception:
                logger.exception("S2 fetch failed for '%s'", query)
            await asyncio.sleep(S2_DELAY)

    deduped = dedup_papers(all_papers)
    logger.info(
        "Total: %d papers fetched, %d after dedup",
        len(all_papers),
        len(deduped),
    )
    return deduped
