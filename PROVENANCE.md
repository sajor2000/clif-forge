# Provenance of generated CLIF 2.1 tables (R4, R14)

`clif-forge` generates every table from an **aggregate parameter pack** and a
latent acuity **spine** — no real patient record ever leaves the fit stage. This
document records, per table, *how* its content is produced, so downstream users
can distinguish empirically-fitted structure from documented priors.

Three provenance classes:

- **fitted** — sampled from parameters fit over real CLIF at the fit stage
  (U5) and stored in the versioned parameter pack.
- **spine-derived** — no separate fitted block; structure is a deterministic or
  heuristic function of the fitted latent spine (organ-support trajectory,
  organ-failure flags, outcome), with un-fitted fields set to documented clinical
  constants (R15).
- **prior-driven** — no fit and no consortium prior file; content comes from
  documented literature / clinical-norm rates keyed to spine acuity (R14).
- **derived** — not an independent event stream at all. CLIF defines the table as
  hanging off another table's rows via a key that table minted
  (``organism_id``, ``med_order_id``), so it is folded from those rows within the
  same encounter and joins back to them exactly.

| Table | Provenance | Basis |
|-------|------------|-------|
| `patient` | fitted | pack demographic marginals; language from documented English-skew prior; birth_date from admission−age |
| `hospitalization` | fitted | pack admission/discharge marginals + spine LOS/outcome (AE4); age from pack quantiles |
| `vitals` | fitted | pack per-state AR(1) physiology |
| `labs` | fitted | pack Gaussian-copula (correlation + log-normal marginals + presence); mCIDE unit/order crosswalks; collect/result timing priors |
| `medication_admin_continuous` | fitted | pack per-med infusion hazards + spine couplings |
| *(latent spine)* | fitted | pack semi-Markov state model + per-level flag prevalences |
| `adt` | spine-derived | acuity RLE into ward/ICU segments |
| `respiratory_support` | spine-derived | support-ladder device sequence + R10 matrix, AE1/AE2; mCIDE name examples; IMV `*_obs` jitter |
| `patient_assessments` | spine-derived | RASS↔sedation, GCS↔neuro-failure flag |
| `position` | spine-derived | prone↔severe-hypoxemia (resp+IMV) |
| `medication_admin_intermittent` | spine-derived | documented antibiotic schedule over the stay |
| `microbiology_culture` | spine-derived | documented per-ICU-day culture rate |
| `crrt_therapy` | spine-derived | CRRT during renal-failure windows + documented rates |
| `code_status` | spine-derived | de-escalation before death (spine outcome); **plan intends fitted — no pack block yet** |
| `ecmo_mcs` | prior-driven | ECMO-tier acuity + documented adult VV-ECMO device norms |
| `invasive_hemodynamics` | prior-driven | PA-catheter events during cv-failure windows |
| `transfusion` | prior-driven | peak-acuity-scaled rate + documented product volumes |
| `key_icu_orders` | prior-driven | PT/OT rehab orders for a subset of ICU stays |
| `therapy_details` | prior-driven | documented PT/OT session elements |
| `provider` | prior-driven | one attending + one nurse spanning each stay |
| `hospital_diagnosis` | spine-derived | ICD-10-CM codes from organ-failure flags; POA from flag onset |
| `patient_diagnosis` | spine-derived | chronic problem list + encounter dx mirroring the same flags |
| `intake_output` | spine-derived | hourly balance; oliguria on renal flag, resuscitation on cv flag |
| `microbiology_nonculture` | prior-driven | documented per-stay molecular-panel rate; mCIDE targets |
| `patient_procedures` | prior-driven | rare, ventilated-stay-only draw from the vendored CPT code list |
| `place_based_index` | prior-driven | one latent deprivation draw on published ADI/SVI scales |
| `clinical_trial` | prior-driven | documented enrolment rate for ventilated stays; synthetic trial registry |
| `microbiology_susceptibility` | derived | panel per isolate grown by `microbiology_culture` (`organism_id`) |
| `medication_orders` | derived | folded from both med-admin tables on `med_order_id` |

Two notes on what the classes above do *not* claim:

- A **derived** table inherits its parent's provenance for structure but still
  uses documented priors for content — the susceptibility panels and per-organism
  resistance rates (MRSA-level oxacillin resistance vs near-zero vancomycin
  resistance) are literature norms, not fitted.
- `invasive_hemodynamics` values are drawn per shock phenotype (cardiogenic vs
  distributive), assigned once per stay. The phenotype split and both value ranges
  are documented physiology, not fitted.

**Release gate:** any public release of a generated dataset or the parameter pack
requires credentialed-data and Rush compliance confirmation.
Fitting locally to an aggregate pack is permitted; releasing derived artifacts is
gated.
