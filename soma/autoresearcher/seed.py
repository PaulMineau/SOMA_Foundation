"""BiomarkerProfile dataclass and JSON loader."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_CONDITIONS: frozenset[str] = frozenset(
    [
        "obstructive_sleep_apnea",
        "central_sleep_apnea",
        "elevated_shbg",
        "low_free_testosterone",
        "hypogonadism",
        "borderline_hyperhomocysteinemia",
        "vitamin_d_insufficiency",
        "vitamin_d_deficiency",
        "nicotine_cessation_active",
        "nicotine_cessation_complete",
        "alcohol_use_disorder_recovery",
        "cannabis_cessation",
        "insulin_resistance",
        "metabolic_syndrome",
        "hypothyroidism",
    ]
)


class ProfileValidationError(Exception):
    """Raised when a biomarker profile fails validation."""


@dataclass(frozen=True)
class BiomarkerProfile:
    """Immutable biomarker profile seeded from lab data."""

    # Identity
    profile_id: str
    updated: str

    # Lab values (required)
    shbg_nmol_l: float
    homocysteine_umol_l: float
    vitamin_d_ng_ml: float

    # Lab values (optional — may not yet have results)
    free_testosterone_pg_ml: float | None = None
    total_testosterone_ng_dl: float | None = None

    # Sleep
    deep_sleep_minutes: float = 0.0
    sleep_efficiency_pct: float | None = None
    apnea_diagnosis: bool = False
    cpap_compliance: str = "unknown"

    # Demographics
    age: int = 0
    sex: str = "unknown"

    # Lists
    conditions: tuple[str, ...] = ()
    current_supplements: tuple[str, ...] = ()
    current_medications: tuple[str, ...] = ()
    known_interventions_acted_on: tuple[str, ...] = ()
    contraindications: tuple[str, ...] = ()
    research_focus: tuple[str, ...] = ()

    # Recovery
    recovery_stage_months: int = 0
    recovery_sensitivity: bool = False

    # Risk
    cardiovascular_risk: str = "unknown"
    risk_note: str = ""

    def to_embedding_text(self) -> str:
        """Natural language description for semantic matching."""
        parts: list[str] = [
            f"{self.age}-year-old {self.sex}",
        ]
        if self.apnea_diagnosis:
            parts.append("with sleep apnea")
        parts.append(
            f"elevated SHBG {self.shbg_nmol_l} nmol/L"
            if self.shbg_nmol_l > 70
            else f"SHBG {self.shbg_nmol_l} nmol/L"
        )
        parts.append(
            f"borderline homocysteine {self.homocysteine_umol_l} umol/L"
            if 10 <= self.homocysteine_umol_l <= 15
            else f"homocysteine {self.homocysteine_umol_l} umol/L"
        )
        if self.vitamin_d_ng_ml < 30:
            parts.append("insufficient vitamin D")

        condition_labels = [c.replace("_", " ") for c in self.conditions]
        if condition_labels:
            parts.append(f"conditions: {', '.join(condition_labels)}")

        if self.research_focus:
            parts.append(
                f"Research focus: {', '.join(f.replace('_', ' ') for f in self.research_focus)}."
            )

        return ", ".join(parts)


def _require_field(data: dict[str, Any], key: str) -> Any:
    """Raise if a required field is missing or None."""
    val = data.get(key)
    if val is None:
        raise ProfileValidationError(f"Required field missing or null: '{key}'")
    return val


def _validate_conditions(conditions: list[str]) -> None:
    """Validate that all conditions are in the controlled vocabulary."""
    invalid = [c for c in conditions if c not in VALID_CONDITIONS]
    if invalid:
        raise ProfileValidationError(
            f"Invalid conditions (not in controlled vocabulary): {invalid}"
        )


def load_profile(path: str | Path) -> BiomarkerProfile:
    """Load and validate a BiomarkerProfile from a JSON file."""
    path = Path(path)
    logger.info("Loading biomarker profile from %s", path)

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    # Extract nested sections
    lab: dict[str, Any] = raw.get("lab_values", {})
    sleep: dict[str, Any] = raw.get("sleep", {})
    demo: dict[str, Any] = raw.get("demographics", {})
    risk: dict[str, Any] = raw.get("risk_flags", {})

    # Validate required lab values
    shbg = _require_field(lab, "shbg_nmol_l")
    homocysteine = _require_field(lab, "homocysteine_umol_l")
    vitamin_d = _require_field(lab, "vitamin_d_ng_ml")

    # Validate conditions
    conditions: list[str] = raw.get("conditions", [])
    _validate_conditions(conditions)

    # Validate demographics
    age = _require_field(demo, "age")
    sex = _require_field(demo, "sex")

    profile = BiomarkerProfile(
        profile_id=_require_field(raw, "profile_id"),
        updated=_require_field(raw, "updated"),
        shbg_nmol_l=float(shbg),
        homocysteine_umol_l=float(homocysteine),
        vitamin_d_ng_ml=float(vitamin_d),
        free_testosterone_pg_ml=(
            float(lab["free_testosterone_pg_ml"])
            if lab.get("free_testosterone_pg_ml") is not None
            else None
        ),
        total_testosterone_ng_dl=(
            float(lab["total_testosterone_ng_dl"])
            if lab.get("total_testosterone_ng_dl") is not None
            else None
        ),
        deep_sleep_minutes=float(sleep.get("deep_sleep_minutes_avg", 0)),
        sleep_efficiency_pct=(
            float(sleep["sleep_efficiency_pct"])
            if sleep.get("sleep_efficiency_pct") is not None
            else None
        ),
        apnea_diagnosis=bool(sleep.get("apnea_diagnosis", False)),
        cpap_compliance=str(sleep.get("cpap_compliance", "unknown")),
        age=int(age),
        sex=str(sex),
        conditions=tuple(conditions),
        current_supplements=tuple(raw.get("current_supplements", [])),
        current_medications=tuple(raw.get("current_medications", [])),
        known_interventions_acted_on=tuple(
            raw.get("known_interventions_acted_on", [])
        ),
        contraindications=tuple(raw.get("contraindications", [])),
        research_focus=tuple(raw.get("research_focus", [])),
        recovery_stage_months=int(raw.get("recovery_stage_months", 0)),
        recovery_sensitivity=bool(risk.get("recovery_sensitivity", False)),
        cardiovascular_risk=str(risk.get("cardiovascular_risk", "unknown")),
        risk_note=str(risk.get("note", "")),
    )

    logger.info(
        "Profile loaded: %s (updated %s)", profile.profile_id, profile.updated
    )
    return profile
