"""LLM-powered PubMed/Semantic Scholar query generation."""

from __future__ import annotations

import logging
from typing import Any

from soma.autoresearcher.llm import llm_call, parse_json_response
from soma.autoresearcher.seed import BiomarkerProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a clinical literature search specialist. Given a patient biomarker \
profile, generate 6-8 targeted PubMed search queries. Each query should use \
MeSH terms where possible. Prioritize intersection queries \
(e.g., "SHBG AND testosterone AND sleep apnea") over single-concept queries.

Return JSON: {"queries": ["query1", "query2", ...]}

Rules:
- Each query must be a valid PubMed search string.
- Use MeSH terms and Boolean operators (AND, OR).
- Cover all research_focus areas from the profile.
- Prioritize multi-concept intersection queries over broad single-concept ones.
- Generate exactly 6-8 queries.
"""

FOLLOWUP_SYSTEM_PROMPT = """\
You are a clinical literature search specialist. Based on the patient profile \
and findings from the previous research iteration, generate 4-6 follow-up \
PubMed search queries that address gaps in the evidence so far.

Return JSON: {"queries": ["query1", "query2", ...]}

Rules:
- Each query must be a valid PubMed search string.
- Use MeSH terms and Boolean operators (AND, OR).
- Focus on gaps: topics where evidence was sparse or contradictory.
- Do NOT repeat queries that already produced strong results.
- Generate exactly 4-6 queries.
"""


def _build_profile_summary(profile: BiomarkerProfile) -> str:
    """Build a concise profile summary for the LLM prompt."""
    lines = [
        f"Age: {profile.age}, Sex: {profile.sex}",
        f"SHBG: {profile.shbg_nmol_l} nmol/L (normal 17-56)",
        f"Homocysteine: {profile.homocysteine_umol_l} umol/L (optimal <10)",
        f"Vitamin D: {profile.vitamin_d_ng_ml} ng/mL (optimal 40-60)",
    ]
    if profile.free_testosterone_pg_ml is not None:
        lines.append(
            f"Free testosterone: {profile.free_testosterone_pg_ml} pg/mL"
        )
    lines.append(f"Conditions: {', '.join(profile.conditions)}")
    lines.append(
        f"Current supplements: {', '.join(profile.current_supplements)}"
    )
    lines.append(f"Recovery stage: {profile.recovery_stage_months} months")
    lines.append(f"Research focus: {', '.join(profile.research_focus)}")
    lines.append(
        f"Contraindications: {', '.join(profile.contraindications)}"
    )
    return "\n".join(lines)


async def generate_queries(
    profile: BiomarkerProfile,
) -> list[str]:
    """Generate initial PubMed/S2 search queries from the biomarker profile."""
    user_prompt = (
        "Generate search queries for this patient profile:\n\n"
        + _build_profile_summary(profile)
    )

    raw = await llm_call(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1024,
        temperature=0.2,
    )

    parsed: dict[str, Any] = parse_json_response(raw)
    queries: list[str] = parsed.get("queries", [])

    if not queries:
        raise RuntimeError("LLM returned no queries")

    logger.info("Generated %d initial queries", len(queries))
    for i, q in enumerate(queries):
        logger.debug("  Query %d: %s", i + 1, q)

    return queries


async def generate_followup_queries(
    profile: BiomarkerProfile,
    previous_findings: list[str],
) -> list[str]:
    """Generate follow-up queries based on gaps from previous iteration."""
    findings_text = "\n".join(
        f"- {f}" for f in previous_findings[:3]
    )

    user_prompt = (
        "Generate follow-up search queries based on this profile and "
        "findings from the previous iteration.\n\n"
        "Patient profile:\n"
        + _build_profile_summary(profile)
        + "\n\nPrevious top findings:\n"
        + findings_text
        + "\n\nGenerate queries that address gaps in the evidence."
    )

    raw = await llm_call(
        system_prompt=FOLLOWUP_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1024,
        temperature=0.2,
    )

    parsed: dict[str, Any] = parse_json_response(raw)
    queries: list[str] = parsed.get("queries", [])

    if not queries:
        raise RuntimeError("LLM returned no follow-up queries")

    logger.info("Generated %d follow-up queries", len(queries))
    return queries
