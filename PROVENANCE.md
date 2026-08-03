# Provenance of generated CLIF 2.1 tables (R4, R14)

`clif-forge` generates every table from an **aggregate parameter pack** and a
latent acuity **spine** — no real patient record ever leaves the fit stage. This
document records, per table, *how* its content is produced, so downstream users
can distinguish empirically-fitted structure from documented priors.

Provenance classes:

- **fitted (MIMIC)** — sampled from parameters fit over local MIMIC-IV Ext CLIF
  (`~/Data/clif-mimic`) into packs such as `data/param_packs/mimic_all28`.
- **spine-derived** — structure is a function of the fitted latent spine; un-fitted
  fields use pack params when present, else documented clinical constants (R15).
- **dashboard-prior** — table absent from local MIMIC; rates from the CLIF Consortium
  cohort dashboard ([clif-icu.com/cohort](https://clif-icu.com/cohort)), vendored in
  `clifforge.reference.dashboard_priors` (retrieved 2026-08-02).
- **derived** — folded from a parent table's keys (`organism_id`, `med_order_id`).

| Table | Provenance | Basis |
|-------|------------|-------|
| `patient` | fitted (MIMIC) | pack demographic marginals; language English-skew prior; birth_date from admission−age |
| `hospitalization` | fitted (MIMIC) | pack admission/discharge marginals + age quantiles + spine LOS/outcome |
| `vitals` | fitted (MIMIC) | pack per-state AR(1) physiology |
| `labs` | fitted (MIMIC) | pack Gaussian-copula + mCIDE unit/order crosswalks |
| `medication_admin_continuous` | fitted (MIMIC) | pack per-med infusion hazards + spine couplings |
| *(latent spine)* | fitted (MIMIC) | pack semi-Markov state model + flag prevalences |
| `adt` | fitted (MIMIC) | pack location/hospital_type marginals; **validated** `arrival_location_marginal` + `direct_icu_frac` + `enrich_locations`; spine `admission_route_marginal` couples admission type to front door |
| `code_status` | fitted (MIMIC) | pack outcome-conditional DNR/AND rates |
| `respiratory_support` | fitted (MIMIC) | pack device/mode marginals; **validated** gated `niv` + `enrich_devices`; in-bounds set-value quantiles |
| `medication_admin_intermittent` | fitted (MIMIC) | pack stay prevalence + top-K med_category |
| `microbiology_culture` | fitted (MIMIC) | pack cultures/ICU-day + fluid/method marginals; organism prior when MIMIC null |
| `crrt_therapy` | fitted (MIMIC) | pack stay prevalence + mode/rate quantiles; renal-flag windows |
| `ecmo_mcs` | fitted (MIMIC) | pack stay prevalence + device/mcs/flow quantiles |
| `patient_assessments` | fitted (MIMIC) | pack assessment_category marginal; RASS/GCS still spine-coupled |
| `position` | fitted (MIMIC) | pack prone rates (overall + among IMV) |
| `hospital_diagnosis` | prior + fitted pad | MIMIC stay-prevalence disease/cancer priors (acuity-scaled); spine acute codes; top-K marginal pads density only |
| `patient_procedures` | fitted (MIMIC) | pack stay prevalence + top-K procedure_code |
| `invasive_hemodynamics` | dashboard-prior | PA-catheter stay rate + phenotype ranges |
| `transfusion` | dashboard-prior | acuity-scaled rate anchored to dashboard stay rate |
| `key_icu_orders` | dashboard-prior | rehab order stay fraction |
| `therapy_details` | dashboard-prior | PT/OT session elements (same rehab gate) |
| `provider` | dashboard-prior | attending + nurse spanning each stay |
| `patient_diagnosis` | prior | same MIMIC stay-prevalence chronic/cancer priors as hospital_diagnosis; encounter dx from spine |
| `intake_output` | dashboard-prior | hourly balance; oliguria/resuscitation on spine flags |
| `microbiology_nonculture` | dashboard-prior | per-stay molecular-panel rate |
| `place_based_index` | dashboard-prior | one deprivation draw (ADI/SVI scales) |
| `clinical_trial` | dashboard-prior | enrolment rate for ventilated stays |
| `microbiology_susceptibility` | derived | panel per isolate from `microbiology_culture` |
| `medication_orders` | derived | folded from both med-admin tables on `med_order_id` |

Notes:

- MIMIC-fitted clinical table blocks are **not** replaced by anonymous network-median
  priors. ICU generation runs the validated :func:`recalibrate_mimic_icu` path
  (same spine tempering / LOS sojourns / gated NIV / terminal deterioration as
  ``recalibrate_to_network_median``, targeted at MIMIC ICU rates) and restores
  fitted ADT ``arrival_location_marginal`` + ``direct_icu_frac``.
- Dashboard priors apply only to tables absent from the local MIMIC extract.
- Derived tables inherit parent structure; susceptibility resistance panels remain
  literature norms.

**Release gate:** any public release of a generated dataset or the parameter pack
requires credentialed-data and Rush compliance confirmation.
Fitting locally to an aggregate pack is permitted; releasing derived artifacts is
gated.
