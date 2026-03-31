# Biomarker Profile Schema

## paul_profile.json (seed file)

```json
{
  "profile_id": "paul_2026_q1",
  "updated": "2026-03-16",
  
  "lab_values": {
    "shbg_nmol_l": 78.0,
    "homocysteine_umol_l": 11.2,
    "vitamin_d_ng_ml": 22.0,
    "free_testosterone_pg_ml": null,
    "total_testosterone_ng_dl": null
  },
  
  "sleep": {
    "device": "Fitbit Inspire 3",
    "deep_sleep_minutes_avg": 90,
    "sleep_efficiency_pct": null,
    "cpap_compliance": "improving",
    "apnea_diagnosis": true,
    "ahi_treated": null
  },
  
  "demographics": {
    "age": 50,
    "sex": "male"
  },
  
  "conditions": [
    "obstructive_sleep_apnea",
    "elevated_shbg",
    "borderline_hyperhomocysteinemia",
    "vitamin_d_insufficiency",
    "nicotine_cessation_active",
    "alcohol_use_disorder_recovery",
    "cannabis_cessation"
  ],
  
  "current_supplements": [
    "D3_K2_5000IU",
    "boron_3mg",
    "thorne_super_EPA",
    "methylated_B_complex"
  ],
  
  "current_medications": [],
  
  "recovery_stage_months": 6,
  
  "research_focus": [
    "free_testosterone_optimization",
    "SHBG_reduction",
    "homocysteine_reduction",
    "sleep_apnea_cardiovascular_risk",
    "cardiovascular_longevity",
    "neuroprotection"
  ],
  
  "known_interventions_acted_on": [
    "vitamin_D_supplementation",
    "EPA_supplementation",
    "methylated_B12_supplementation",
    "methylfolate_supplementation",
    "boron_supplementation",
    "exercise_running",
    "CPAP_therapy",
    "alcohol_cessation",
    "cannabis_cessation",
    "nicotine_cessation"
  ],
  
  "contraindications": [
    "high_dose_testosterone_therapy_without_monitoring",
    "stimulants_with_cardiovascular_risk",
    "substances_with_addiction_potential"
  ],
  
  "risk_flags": {
    "cardiovascular_risk": "moderate_elevated",
    "recovery_sensitivity": true,
    "note": "No interventions with addiction potential. Flag any stimulant-class compounds."
  }
}
```

## Schema Validation Rules

- `shbg_nmol_l` — required. Normal range 17-56 nmol/L. Elevated = > 70.
- `homocysteine_umol_l` — required. Optimal < 10. Borderline 10-15. High > 15.
- `vitamin_d_ng_ml` — required. Deficient < 20. Insufficient 20-29. Optimal 40-60.
- `conditions` — must be snake_case, from controlled vocabulary (see below)
- `current_supplements` — feeds novelty filter. Interventions matching this list score N ≈ 0.
- `known_interventions_acted_on` — broader than supplements; includes lifestyle changes.
- `contraindications` — any paper with intervention in this list gets `safe_for_profile = false`.
- `recovery_sensitivity` — when `true`, flag any compound with dependency risk in synthesis output.

## Controlled Vocabulary — Conditions

```
obstructive_sleep_apnea
central_sleep_apnea
elevated_shbg
low_free_testosterone
hypogonadism
borderline_hyperhomocysteinemia
vitamin_d_insufficiency
vitamin_d_deficiency
nicotine_cessation_active
nicotine_cessation_complete
alcohol_use_disorder_recovery
cannabis_cessation
insulin_resistance
metabolic_syndrome
hypothyroidism
```

## Updating the Profile

After each lab draw or Fitbit sync, update `paul_profile.json` and re-run the research loop. The novelty filter and relevance scores will automatically recalibrate.

Track `updated` date — the agent logs research sessions against profile version.
