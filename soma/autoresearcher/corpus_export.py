"""Export scored papers to JSONL for ML training data.

Outputs records as (abstract, RAEN_scores, layer_assignment) for downstream
model training (RAEN scorer, domain embeddings, distilled local model).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from soma.autoresearcher.memory import get_all_findings

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "data/training_data.jsonl"


def export_from_lancedb(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    db_path: str | None = None,
    min_raen: float = 0.0,
) -> int:
    """Export all stored findings from LanceDB to JSONL training format.

    Each line is a JSON object with:
    - abstract: str
    - title: str
    - intervention: str
    - outcome: str
    - raen_total: float
    - raen_r, raen_a, raen_e, raen_n: float
    - soma_layer: str
    - study_type: str
    - year: int
    - export_date: str

    Args:
        output_path: Path to write JSONL file.
        db_path: Override LanceDB path.
        min_raen: Minimum RAEN total score to include.

    Returns:
        Number of records exported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    findings = get_all_findings(db_path=db_path)

    if not findings:
        logger.info("No findings to export")
        return 0

    export_date = date.today().isoformat()
    exported = 0

    with output_path.open("w", encoding="utf-8") as f:
        for rec in findings:
            raen_total = rec.get("raen_total", 0.0)
            if raen_total < min_raen:
                continue

            abstract = rec.get("abstract", "")
            if not abstract:
                continue

            record: dict[str, Any] = {
                "abstract": abstract,
                "title": rec.get("title", ""),
                "intervention": rec.get("intervention", ""),
                "outcome": rec.get("outcome", ""),
                "raen_total": raen_total,
                "raen_r": rec.get("raen_r", 0.0),
                "raen_a": rec.get("raen_a", 0.0),
                "raen_e": rec.get("raen_e", 0.0),
                "raen_n": rec.get("raen_n", 0.0),
                "soma_layer": rec.get("soma_layer", ""),
                "study_type": rec.get("study_type", ""),
                "year": rec.get("year", 0),
                "export_date": export_date,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1

    logger.info("Exported %d records to %s", exported, output_path)
    return exported


def export_scored_papers(
    scored_papers: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> int:
    """Export a list of scored paper dicts directly to JSONL.

    Use this when you have in-memory results (e.g., from a just-completed
    overnight run) rather than reading back from LanceDB.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_date = date.today().isoformat()
    exported = 0

    # Append mode — don't overwrite previous training data
    with output_path.open("a", encoding="utf-8") as f:
        for rec in scored_papers:
            rec["export_date"] = export_date
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            exported += 1

    logger.info("Appended %d records to %s", exported, output_path)
    return exported


def main() -> None:
    """CLI entrypoint for corpus export."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export SOMA research corpus to JSONL training data"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path"
    )
    parser.add_argument(
        "--min-raen", type=float, default=0.0, help="Minimum RAEN score filter"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    count = export_from_lancedb(output_path=args.output, min_raen=args.min_raen)
    print(f"Exported {count} records to {args.output}")


if __name__ == "__main__":
    main()
