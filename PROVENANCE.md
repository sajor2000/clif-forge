# Provenance of generated CLIF 2.1 tables (R4, R14)

`clif-forge` generates every table from an **aggregate parameter pack** and a
latent acuity **spine** — no real patient record ever leaves the fit stage. This
document is the **authoritative catalog** of the 28 CLIF 2.1 tables (+ spine):
*how* each is produced, *which pack path* it uses, and *what the MIMIC realism
work changed*.

## Packs

| Pack | Role |
|------|------|
| [`base_pack/`](base_pack/) | Shareable aggregate pack shipped in the package; default for presets / network-median ICU |
| [`data/param_packs/mimic_all28`](data/param_packs/mimic_all28/) | Local MIMIC-IV Ext CLIF fit (all clinical tables present in that extract). Seeds the committed samples |

ICU samples and MIMIC-targeted generation run
`recalibrate_mimic_icu` (spine tempering, LOS sojourns, gated NIV, terminal
deterioration, ADT front-door arrivals). Full-hospital mode uses
`recalibrate_to_full_hospital`. Presets without a MIMIC pack still use
`recalibrate_to_network_median` on `base_pack/`.

## Provenance classes

- **fitted (MIMIC)** — sampled from parameters fit over local MIMIC-IV Ext CLIF
  (`~/Data/clif-mimic`) into packs such as `mimic_all28`.
- **prior** — documented MIMIC stay-prevalence or clinical-norm rates (not a thin
  fitted top-K alone); still keyed to the spine where required (KTD-6).
- **dashboard-prior** — table absent from local MIMIC; rates from the CLIF
  Consortium cohort dashboard ([clif-icu.com/cohort](https://clif-icu.com/cohort)),
  vendored in `clifforge.reference.dashboard_priors` (retrieved 2026-08-02).
- **derived** — folded from a parent table's keys (`organism_id`, `med_order_id`).

## Latent spine (`_truth.parquet`)

Not a CLIF table. Per-interval support level (0–5), organ-failure flags
(`resp` / `cv` / `renal` / `neuro`), outcome, `admission_route`, and
`resp_phenotype`. Downstream generators **read the spine**; they do not invent
acuity or pathways themselves.

| Field | Meaning |
|-------|---------|
| `support_level` | Semi-Markov organ-support ladder (ward → IMV / vaso / CRRT / ECMO tiers) |
| `*_flag` | Interval organ-failure indicators driving labs, vitals, therapies, dx |
| `outcome` | `alive` / `expired` (terminal archetypes shape the dying trajectory) |
| `admission_route` | Couples hospitalization admission type to ADT arrival location |
| `resp_phenotype` | `type1` (NC→HFNC→IMV), `type2_ohs` / `type2_hf` / `type2_copd` (NIPPV→IMV), or `unspecified` |

## All 28 CLIF 2.1 tables

| Table | Provenance | What it is / basis |
|-------|------------|-------------------|
| `patient` | fitted (MIMIC) | Demographics; language English-skew prior; `birth_date` from admission−age |
| `hospitalization` | fitted (MIMIC) | Admission/discharge marginals + age quantiles + spine LOS/outcome |
| `adt` | fitted (MIMIC) | Location/hospital_type marginals; validated `arrival_location_marginal` + `direct_icu_frac`; `admission_route` couples front door |
| `vitals` | fitted (MIMIC) | Per-state AR(1); cv→shock hemodynamics, resp→SpO₂ pull |
| `labs` | fitted (MIMIC) | Gaussian-copula + mCIDE unit/order crosswalks; renal/shock bumps, soft lactate cap |
| `respiratory_support` | fitted (MIMIC) | Device/mode marginals; **gated NIV**; phenotype ladders (HFNC vs NIPPV); IMV set-value quantiles |
| `medication_admin_continuous` | fitted (MIMIC) | Infusion hazards + spine couplings; **sedation\|IMV ≈ MIMIC** |
| `medication_admin_intermittent` | fitted (MIMIC) | Stay prevalence + top-K `med_category` |
| `medication_orders` | derived | Folded from both med-admin tables on `med_order_id` |
| `patient_assessments` | fitted (MIMIC) | Assessment-category marginal; RASS/GCS still spine-coupled |
| `position` | fitted (MIMIC) | Prone rates (overall + among IMV) |
| `microbiology_culture` | fitted (MIMIC) | Cultures/ICU-day + fluid/method; organism prior when MIMIC null |
| `microbiology_nonculture` | dashboard-prior | Per-stay molecular-panel rate |
| `microbiology_susceptibility` | derived | Panel per isolate from `microbiology_culture` (`organism_id`) |
| `crrt_therapy` | fitted (MIMIC) | Stay prevalence + mode/rate quantiles; renal-flag windows + creat gate |
| `code_status` | fitted (MIMIC) | Outcome-conditional DNR/AND rates |
| `ecmo_mcs` | fitted (MIMIC) | Stay prevalence + device/mcs/flow quantiles |
| `invasive_hemodynamics` | dashboard-prior | PA-catheter stay rate + cardiogenic vs distributive ranges |
| `transfusion` | dashboard-prior | Acuity-scaled rate anchored to dashboard stay rate |
| `key_icu_orders` | dashboard-prior | Rehab order stay fraction |
| `therapy_details` | dashboard-prior | PT/OT session elements (same rehab gate) |
| `provider` | dashboard-prior | Attending + nurse spanning each stay |
| `hospital_diagnosis` | prior + fitted pad | MIMIC stay-prevalence disease/cancer priors (acuity-scaled); soft acute flag codes; top-K marginal **pads density only** (~18 codes/stay) |
| `patient_diagnosis` | prior | Same chronic/cancer priors as hospital_diagnosis; encounter dx from spine |
| `patient_procedures` | fitted (MIMIC) | Stay prevalence + top-K `procedure_code` |
| `intake_output` | dashboard-prior | Hourly balance; oliguria / resuscitation on spine flags |
| `place_based_index` | dashboard-prior | One deprivation draw (ADI/SVI scales) |
| `clinical_trial` | dashboard-prior | Enrolment rate for ventilated stays |

Output filenames follow CLIF 2.1 maturity:
`clif_<table>_2.1_<beta|concept|untiered>.parquet`.

## What we did — MIMIC realism (2026-08)

Work on branch `cursor/rename-truth-spine` brought generation into the MIMIC ICU
statistical region and tightened longitudinal coherence:

1. **Fit path** — `mimic_all28` pack + MIMIC estimators for clinical tables present
   in the local extract; dashboard priors only for MIMIC-absent tables.
2. **Recalibrate** — `recalibrate_mimic_icu` (not anonymous network-median overwrite
   of fitted MIMIC blocks): IMV/mortality/NIV/ADT targets, terminal mix, CRRT gate,
   `resp_phenotype_marginal`, sedation knobs.
3. **Trajectories** — sicker↔sicker coupling: vitals/labs track spine flags; soft
   L4→cv / L5→renal; terminal archetypes with MIMIC-ish invent rates.
4. **Respiratory pathways** — type1 NC→HFNC→IMV vs type2 NIPPV→IMV; NIV stay-gated
   so NIPPV/HFNC stay rates match MIMIC (± few pp).
5. **Sedation** — continuous sedatives paired with IMV (`sedation_per_imv`).
6. **Diagnoses** — dropped universal pneumonia principal; MIMIC disease/cancer
   stay prevalences within ±5 pp; codes/stay ~18; sicker stays carry more chronics.
7. **Validation** — `scripts/validate_against_real.py` probes for trajectories,
   coherence (vaso\|IMV, sedation\|IMV, decedent physiology, CRRT\|creat), plus
   `scripts/audit_realism_sources.py`.
8. **Samples** — `sample_dataset/` and `sample_full_hospital/` regenerated from
   `mimic_all28` (n=5000, seed 42) via the validated recalibrate paths.

## Notes

- MIMIC-fitted clinical blocks are **not** replaced by network-median priors when
  generating ICU data from `mimic_all28`.
- Dashboard priors apply only to tables absent from the local MIMIC extract.
- Derived tables inherit parent structure; susceptibility resistance panels remain
  literature norms.
- Demo / hand-prior runs without a MIMIC pack still use documented constants so
  unit tests and tiny demos stay self-contained.

**Release gate:** any public release of a generated dataset or the parameter pack
requires credentialed-data and Rush compliance confirmation.
Fitting locally to an aggregate pack is permitted; releasing derived artifacts is
gated.
