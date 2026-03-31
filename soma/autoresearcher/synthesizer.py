"""LLM-powered briefing synthesizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from soma.autoresearcher.extractor import PaperExtract
from soma.autoresearcher.fetcher import Paper
from soma.autoresearcher.llm import llm_call, parse_json_response
from soma.autoresearcher.scorer import RAENScore
from soma.autoresearcher.seed import BiomarkerProfile

logger = logging.getLogger(__name__)


@dataclass
class BriefingEntry:
    """A single entry in the synthesized briefing."""

    rank: int
    intervention: str
    raen_score: float
    raen_breakdown: str
    study_type: str
    year: int
    effect_summary: str
    action_step: str
    soma_layer: str
    safe_for_profile: bool
    conflicts: list[str]


@dataclass
class Briefing:
    """Complete synthesized briefing."""

    date: str
    entries: list[BriefingEntry]
    gaps: list[str]
    already_optimized: list[str]

    def to_markdown(self) -> str:
        """Render as markdown for Streamlit or stdout."""
        lines: list[str] = [
            f"## SOMA Health Brief — {self.date}",
            "",
            "### Top Interventions",
        ]

        for entry in self.entries:
            lines.append(
                f"{entry.rank}. **{entry.intervention}** — "
                f"RAEN: {entry.raen_score:.2f}"
            )
            lines.append(
                f"   Evidence: {entry.study_type}, {entry.year}. "
                f"Effect: {entry.effect_summary}."
            )
            lines.append(f"   Action: {entry.action_step}.")
            lines.append(f"   SOMA layer: {entry.soma_layer}")
            if not entry.safe_for_profile:
                lines.append(
                    "   **Warning**: Safety flag — discuss with physician"
                )
            if entry.conflicts:
                lines.append(
                    f"   Conflicts: {', '.join(entry.conflicts)}"
                )
            lines.append("")

        if self.gaps:
            lines.append("### Gaps Identified")
            for gap in self.gaps:
                lines.append(
                    f"- {gap} — literature sparse, "
                    "suggest clinical consultation"
                )
            lines.append("")

        if self.already_optimized:
            lines.append("### Already Optimized")
            for item in self.already_optimized:
                lines.append(f"- {item}")
            lines.append("")

        lines.append("---")
        lines.append(
            "*Discuss top-scored interventions with your physician "
            "before changing protocols.*"
        )
        return "\n".join(lines)


SYNTHESIS_SYSTEM_PROMPT = """\
You are a health research synthesis specialist for the SOMA personal health \
system. Given scored research findings, generate:

1. A concrete "action step" for each top intervention — what should the \
patient do this week?
2. A list of "gaps" — research focus areas where evidence was sparse or \
contradictory.
3. A list of "already optimized" — interventions the patient already takes \
that have strong supporting evidence.

Return JSON:
{
  "action_steps": {
    "<intervention name>": "<concrete next step>"
  },
  "gaps": ["<topic 1>", "<topic 2>"],
  "already_optimized": ["<supplement/intervention with strong evidence>"]
}

Rules:
- Action steps must be specific, safe, and actionable this week.
- If safe_for_profile is false, the action step should be "Discuss with \
physician before starting".
- Gaps are research_focus areas with fewer than 2 strong papers (RAEN > 0.4).
- Already optimized: current supplements backed by at least one paper with \
RAEN > 0.3.
"""


def _build_synthesis_prompt(
    findings: list[tuple[Paper, PaperExtract, RAENScore, str]],
    profile: BiomarkerProfile,
) -> str:
    """Build the user prompt for synthesis."""
    parts: list[str] = [
        "Synthesize findings for this patient.\n",
        f"Research focus areas: {', '.join(profile.research_focus)}",
        f"Current supplements: {', '.join(profile.current_supplements)}",
        f"Contraindications: {', '.join(profile.contraindications)}",
        f"Recovery sensitivity: {profile.recovery_sensitivity}",
        "\nTop findings (ranked by RAEN score):\n",
    ]

    for i, (paper, extract, score, layer) in enumerate(findings[:10], 1):
        parts.append(
            f"{i}. Intervention: {extract.intervention}\n"
            f"   Outcome: {extract.outcome_measure}\n"
            f"   Direction: {extract.effect_direction}, "
            f"Effect size: {extract.effect_size}\n"
            f"   Study type: {paper.study_type}, Year: {paper.year}\n"
            f"   RAEN: {score.total:.3f} "
            f"(R={score.R:.2f} A={score.A:.2f} "
            f"E={score.E:.2f} N={score.N:.2f})\n"
            f"   Safe: {extract.safe_for_profile}, "
            f"SOMA layer: {layer}\n"
        )

    return "\n".join(parts)


async def synthesize_briefing(
    findings: list[tuple[Paper, PaperExtract, RAENScore, str]],
    profile: BiomarkerProfile,
) -> Briefing:
    """Generate a structured briefing from scored and classified findings."""
    user_prompt = _build_synthesis_prompt(findings, profile)

    raw = await llm_call(
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=2048,
        temperature=0.7,
    )

    parsed: dict[str, Any] = parse_json_response(raw)
    action_steps: dict[str, str] = {
        str(k): str(v)
        for k, v in (parsed.get("action_steps") or {}).items()
    }
    gaps: list[str] = [str(g) for g in (parsed.get("gaps") or [])]
    already_optimized: list[str] = [
        str(a) for a in (parsed.get("already_optimized") or [])
    ]

    entries: list[BriefingEntry] = []
    for rank, (paper, extract, score, layer) in enumerate(
        findings[:10], 1
    ):
        action = _find_action_step(extract.intervention, action_steps)

        effect_parts: list[str] = [extract.effect_direction]
        if extract.effect_size is not None:
            effect_parts.append(f"{extract.effect_size}%")

        entries.append(
            BriefingEntry(
                rank=rank,
                intervention=extract.intervention,
                raen_score=score.total,
                raen_breakdown=(
                    f"R={score.R:.2f} A={score.A:.2f} "
                    f"E={score.E:.2f} N={score.N:.2f} "
                    f"LSS={score.LSS:.2f}"
                ),
                study_type=paper.study_type,
                year=paper.year,
                effect_summary=", ".join(effect_parts),
                action_step=action,
                soma_layer=layer,
                safe_for_profile=extract.safe_for_profile,
                conflicts=extract.conflicts_with_supplements,
            )
        )

    briefing = Briefing(
        date=date.today().isoformat(),
        entries=entries,
        gaps=gaps,
        already_optimized=already_optimized,
    )

    logger.info(
        "Briefing synthesized: %d entries, %d gaps, %d already optimized",
        len(entries),
        len(gaps),
        len(already_optimized),
    )
    return briefing


def _find_action_step(
    intervention: str, action_steps: dict[str, str]
) -> str:
    """Find the best matching action step for an intervention."""
    if intervention in action_steps:
        return action_steps[intervention]

    intervention_lower = intervention.lower()
    for key, value in action_steps.items():
        if (
            key.lower() in intervention_lower
            or intervention_lower in key.lower()
        ):
            return value

    return "Review evidence and discuss with physician"
