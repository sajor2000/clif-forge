"""Vendored CLIF Consortium cohort-dashboard priors (source-absent tables).

Source: https://clif-icu.com/cohort Summary Statistics / Advanced Support
(retrieved 2026-08-02). Used only for tables local source CLIF extract does not
ship; fitted pack blocks always win when present.
"""

from __future__ import annotations

__all__ = [
    "DASHBOARD_CITATION",
    "DASHBOARD_RETRIEVED_AT",
    "DASHBOARD_SOURCE_URL",
    "absent_table_rates",
    "admission_type_mix",
    "advanced_support",
    "demographics",
    "encounter_mix",
    "vent_setting_medians",
]

DASHBOARD_SOURCE_URL = "https://clif-icu.com/cohort"
DASHBOARD_RETRIEVED_AT = "2026-08-02"
DASHBOARD_CITATION = (
    "CLIF Consortium Cohort Dashboard Summary Statistics / Advanced Support "
    f"({DASHBOARD_SOURCE_URL}, retrieved {DASHBOARD_RETRIEVED_AT}). "
    "Aggregated, de-identified consortium rates. Used here only for CLIF tables "
    "absent from the local source CLIF extract."
)

#: Critically-ill cohort demographics (dashboard Summary Statistics).
demographics: dict[str, float] = {
    "n_patients": 808_749.0,
    "n_hospitalizations": 1_029_400.0,
    "n_icu_hospitalizations": 933_058.0,
    "hospital_mortality": 0.258,
    "age_median": 66.0,
    "age_q1": 47.0,
    "age_q3": 79.0,
    "female_frac": 0.455,
    "white_frac": 0.633,
    "hispanic_frac": 0.058,
    "icu_los_days_median": 1.7,
    "icu_los_days_q1": 0.8,
    "icu_los_days_q3": 5.0,
    "hosp_los_days_median": 6.4,
    "hosp_los_days_q1": 2.6,
    "hosp_los_days_q3": 18.6,
    "sofa_median": 3.0,
    "charlson_median": 2.0,
}

#: Encounter-type mix among critically ill hospitalizations.
encounter_mix: dict[str, float] = {
    "icu": 0.516,
    "advanced_respiratory": 0.251,
    "vasoactive": 0.192,
    "other_critically_ill": 0.041,
}

#: Admission-type mix (dashboard Admission Types).
admission_type_mix: dict[str, float] = {
    "ed": 0.652,
    "osh": 0.145,
    "elective": 0.079,
    "direct": 0.058,
}

#: Advanced support prevalence among hospitalizations (dashboard Advanced Support).
advanced_support: dict[str, float] = {
    "imv": 0.306,
    "vasopressor": 0.253,
    "crrt": 0.034,
    "norepinephrine": 0.179,
    "phenylephrine": 0.110,
    "vasopressin": 0.073,
    "epinephrine": 0.068,
    "dopamine": 0.015,
}

#: Initial ventilator setting medians (dashboard Respiratory Support).
vent_setting_medians: dict[str, float] = {
    "fio2_set": 0.4,
    "peep_set": 5.0,
    "tidal_volume_set": 450.0,
    "resp_rate_set": 16.0,
}

#: Stay-level rates for source-absent prior-driven tables (literature-scaled to
#: dashboard acuity where no dashboard cell exists).
absent_table_rates: dict[str, float] = {
    # Rare / Concept tables: low per-stay rates consistent with ICU acuity.
    "invasive_hemodynamics": 0.04,
    "transfusion": 0.12,
    "key_icu_orders": 0.35,
    "therapy_details_sessions_per_order": 2.0,
    "clinical_trial": 0.02,
    "microbiology_nonculture": 0.08,
    "patient_diagnosis_chronic_codes": 3.0,
    "place_based_index": 1.0,  # one row per patient when emitted
    "provider_roles_per_stay": 2.0,
    "intake_output_hourly": 1.0,
}
