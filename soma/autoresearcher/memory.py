"""LanceDB read/write for SOMA Extended Consciousness layer."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import lancedb  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from soma.autoresearcher.extractor import PaperExtract
from soma.autoresearcher.fetcher import Paper
from soma.autoresearcher.scorer import RAENScore

logger = logging.getLogger(__name__)

TABLE_NAME = "soma_research"
EMBEDDING_DIM = 384


def _get_db_path() -> str:
    return os.environ.get("LANCEDB_PATH", "./data/lancedb")


def get_db(db_path: str | None = None) -> lancedb.DBConnection:
    """Connect to the LanceDB database."""
    path = db_path or _get_db_path()
    logger.info("Connecting to LanceDB at %s", path)
    db: lancedb.DBConnection = lancedb.connect(path)
    return db


def _paper_id(paper: Paper) -> str:
    """Get a stable identifier for a paper (PMID preferred, then DOI)."""
    if paper.pmid:
        return f"pmid:{paper.pmid}"
    if paper.doi:
        return f"doi:{paper.doi}"
    return f"title:{paper.title[:80]}"


def _build_record(
    paper: Paper,
    extract: PaperExtract,
    score: RAENScore,
    soma_layer: str,
    embedding: list[float],
    briefing_date: str | None = None,
) -> dict[str, Any]:
    """Build a flat dict record for LanceDB insertion."""
    return {
        "id": _paper_id(paper),
        "title": paper.title,
        "abstract": paper.abstract[:2000],
        "intervention": extract.intervention,
        "outcome": extract.outcome_measure,
        "raen_total": float(score.total),
        "raen_r": float(score.R),
        "raen_a": float(score.A),
        "raen_e": float(score.E),
        "raen_n": float(score.N),
        "raen_lss": float(score.LSS),
        "soma_layer": soma_layer,
        "primary_layer_key": score.primary_layer,
        "year": paper.year,
        "study_type": paper.study_type,
        "vector": embedding,
        "briefing_date": briefing_date or date.today().isoformat(),
    }


def store_findings(
    findings: list[tuple[Paper, PaperExtract, RAENScore, str]],
    embedder: SentenceTransformer,
    db_path: str | None = None,
) -> int:
    """Write scored findings to LanceDB."""
    if not findings:
        logger.info("No findings to store")
        return 0

    db = get_db(db_path)
    briefing_date = date.today().isoformat()

    records: list[dict[str, Any]] = []
    for paper, extract, score, layer in findings:
        embed_text = f"{extract.intervention}. {extract.outcome_measure}."
        embedding = np.asarray(
            embedder.encode(embed_text), dtype=np.float32
        ).tolist()

        records.append(
            _build_record(paper, extract, score, layer, embedding, briefing_date)
        )

    table_names: list[str] = db.table_names()
    if TABLE_NAME in table_names:
        tbl = db.open_table(TABLE_NAME)
        tbl.add(records)
        logger.info("Appended %d records to '%s'", len(records), TABLE_NAME)
    else:
        db.create_table(TABLE_NAME, records)
        logger.info("Created table '%s' with %d records", TABLE_NAME, len(records))

    return len(records)


def search_similar(
    query: str,
    embedder: SentenceTransformer,
    limit: int = 10,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar findings by semantic similarity."""
    db = get_db(db_path)

    table_names: list[str] = db.table_names()
    if TABLE_NAME not in table_names:
        logger.info("Table '%s' does not exist yet", TABLE_NAME)
        return []

    tbl = db.open_table(TABLE_NAME)
    query_emb = np.asarray(
        embedder.encode(query), dtype=np.float32
    ).tolist()

    results: list[dict[str, Any]] = (
        tbl.search(query_emb).limit(limit).to_list()
    )

    logger.info("Search '%s': %d results", query[:50], len(results))
    return results


def get_all_findings(
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve all stored findings."""
    db = get_db(db_path)

    table_names: list[str] = db.table_names()
    if TABLE_NAME not in table_names:
        return []

    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()
    records: list[dict[str, Any]] = df.to_dict("records")
    logger.info("Retrieved %d total findings", len(records))
    return records


def get_findings_by_layer(
    soma_layer: str,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve findings filtered by SOMA layer."""
    db = get_db(db_path)

    table_names: list[str] = db.table_names()
    if TABLE_NAME not in table_names:
        return []

    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()
    filtered = df[df["soma_layer"] == soma_layer]
    records: list[dict[str, Any]] = filtered.to_dict("records")
    logger.info(
        "Found %d findings for layer '%s'", len(records), soma_layer
    )
    return records
