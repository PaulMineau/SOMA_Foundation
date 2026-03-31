"""Structured extraction from papers via LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from soma.autoresearcher.fetcher import Paper
from soma.autoresearcher.llm import llm_call, parse_json_response
from soma.autoresearcher.seed import BiomarkerProfile

logger = logging.getLogger(__name__)


@dataclass
class PaperExtract:
    """Structured fields extracted from a paper by LLM."""

    intervention: str
    population_description: str
    effect_size: float | None  # Cohen's d or % change if reported
    effect_direction: str  # "positive" | "negative" | "null" | "mixed"
    outcome_measure: str
    safe_for_profile: bool  # LLM judgment given patient conditions
    actionable: bool  # concrete intervention available OTC or via prescription
    conflicts_with_supplements: list[str] = field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """\
You are a biomedical data extraction specialist. Extract structured fields \
from a research paper abstract for a personal health optimization system.

Patient context is provided so you can judge safety and actionability.

Return JSON with exactly these fields:
{
  "intervention": "<the concrete intervention described, e.g. 'Tongkat ali 200mg daily'>",
  "population_description": "<study population, e.g. 'Males aged 45-55 with elevated SHBG'>",
  "effect_size": <numeric percent change or Cohen's d if reported, or null if not in paper>,
  "effect_direction": "<positive | negative | null | mixed>",
  "outcome_measure": "<primary outcome measured, e.g. 'SHBG reduction'>",
  "safe_for_profile": <true | false>,
  "actionable": <true | false>,
  "conflicts_with_supplements": ["<supplement name>", ...]
}

Rules:
- "intervention": The specific treatment or action studied. If the paper is a \
review with no single intervention, use the most prominent one discussed.
- "effect_size": ONLY include if explicitly stated in the abstract. Never \
estimate or hallucinate effect sizes. Use null if not reported.
- "effect_direction": "positive" if the intervention improved the target \
outcome, "negative" if it worsened it, "null" if no significant effect, \
"mixed" if results were mixed.
- "safe_for_profile": false if the intervention conflicts with the patient's \
contraindications, has addiction potential (patient is in recovery), or \
poses cardiovascular risk. When in doubt, mark false.
- "actionable": true only if the intervention is a concrete, available action \
(supplement, lifestyle change, OTC product, or prescription). false for \
theoretical mechanisms or surgical procedures without clear access.
- "conflicts_with_supplements": List any of the patient's current supplements \
that may interact with the intervention. Empty list if none.
"""


def _build_extraction_prompt(
    paper: Paper, profile: BiomarkerProfile
) -> str:
    """Build the user prompt for extraction."""
    parts = [
        "Extract structured fields from this paper.\n",
        f"Title: {paper.title}",
        f"Year: {paper.year}",
        f"Study type: {paper.study_type}",
        f"\nAbstract:\n{paper.abstract}\n",
        "Patient context:",
        f"  Age: {profile.age}, Sex: {profile.sex}",
        f"  Conditions: {', '.join(profile.conditions)}",
        f"  Current supplements: {', '.join(profile.current_supplements)}",
        f"  Contraindications: {', '.join(profile.contraindications)}",
        f"  Recovery sensitivity: {profile.recovery_sensitivity}",
        f"  Cardiovascular risk: {profile.cardiovascular_risk}",
    ]
    return "\n".join(parts)


async def extract_paper(
    paper: Paper,
    profile: BiomarkerProfile,
) -> PaperExtract:
    """Extract structured fields from a paper using LLM."""
    user_prompt = _build_extraction_prompt(paper, profile)

    raw = await llm_call(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1024,
        temperature=0.2,
    )

    parsed: dict[str, Any] = parse_json_response(raw)

    effect_size_raw = parsed.get("effect_size")
    effect_size: float | None = (
        float(effect_size_raw) if effect_size_raw is not None else None
    )

    conflicts_raw = parsed.get("conflicts_with_supplements", [])
    conflicts: list[str] = (
        list(conflicts_raw) if isinstance(conflicts_raw, list) else []
    )

    extract = PaperExtract(
        intervention=str(parsed.get("intervention", "")),
        population_description=str(parsed.get("population_description", "")),
        effect_size=effect_size,
        effect_direction=str(parsed.get("effect_direction", "null")),
        outcome_measure=str(parsed.get("outcome_measure", "")),
        safe_for_profile=bool(parsed.get("safe_for_profile", False)),
        actionable=bool(parsed.get("actionable", False)),
        conflicts_with_supplements=conflicts,
    )

    logger.info(
        "Extracted from '%s': intervention='%s', direction=%s, actionable=%s",
        paper.title[:50],
        extract.intervention[:40],
        extract.effect_direction,
        extract.actionable,
    )
    return extract
