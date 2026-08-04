# Provenance of generated CLIF 2.1 tables (R4, R14)

`clif-forge` generates every table from an **aggregate parameter pack** and a
latent acuity **spine** — no real patient record ever leaves the fit stage. This
document is the **authoritative catalog** of the 28 CLIF 2.1 tables (+ spine):
*how* each is produced, *which pack path* it uses, and *what the fitted ICU realism
work changed*.

## Packs

| Pack | Role |
|------|------|
| [`base_pack/`](base_pack/) | Shareable aggregate pack shipped in the package; default for presets / network-median ICU |
| [`data/param_packs/icu_all28`](data/param_packs/icu_all28/) | Local source CLIF extract fit (all clinical tables present in that extract). Seeds the committed samples |

ICU samples and fitted-ICU-targeted generation run
`recalibrate_fitted_icu` (spine tempering, LOS sojourns, gated NIV, terminal
deterioration, ADT front-door arrivals). Full-hospital mode uses
`recalibrate_to_full_hospital`. Presets without a reference pack still use
`recalibrate_to_network_median` on `base_pack/`.

## Provenance classes

- **fitted** — sampled from parameters fit over a local source CLIF extract
  (`~/Data/clif-source`) into packs such as `icu_all28`.
- **prior** — documented reference stay-prevalence or clinical-norm rates (not a thin
  fitted top-K alone); still keyed to the spine where required (KTD-6).
- **dashboard-prior** — table absent from the local source extract; rates from the CLIF
  Consortium cohort dashboard ([clif-icu.com/cohort](https://clif-icu.com/cohort)),
  vendored in `clifforge.reference.dashboard_priors` (retrieved 2026-08-02).
- **derived** — folded from a parent table's keys (`organism_id`, `med_order_id`).

## Latent spine (in-memory)

Not a CLIF table and not written to disk. Per-interval support level (0–5),
organ-failure flags (`resp` / `cv` / `renal` / `neuro`), outcome,
`admission_route`, and `resp_phenotype`. Downstream generators **read the
spine**; they do not invent acuity or pathways themselves.

| Field | Meaning |
|-------|---------|
| support_level | Semi-Markov organ-support ladder (ward → IMV / vaso / CRRT / ECMO tiers) |
| organ-failure flags | Interval `resp` / `cv` / `renal` / `neuro` indicators driving labs, vitals, therapies, dx |
| outcome | `alive` / `expired` (terminal archetypes shape the dying trajectory) |
| admission_route | Couples hospitalization admission type to ADT arrival location |
| resp_phenotype | `type1` (NC→HFNC→IMV), `type2_ohs` / `type2_hf` / `type2_copd` (NIPPV→IMV), or `unspecified` |

## All 28 CLIF 2.1 tables

| Table | Provenance | What it is / basis |
|-------|------------|-------------------|
| `patient` | fitted | Demographics; language English-skew prior; `birth_date` from admission−age |
| `hospitalization` | fitted | Admission/discharge marginals + age quantiles + spine LOS/outcome |
| `adt` | fitted | Location/hospital_type marginals; validated `arrival_location_marginal` + `direct_icu_frac`; `admission_route` couples front door |
| `vitals` | fitted | Per-state AR(1); cv→shock hemodynamics, resp→SpO₂ pull |
| `labs` | fitted | Gaussian-copula + mCIDE unit/order crosswalks; renal/shock bumps, soft lactate cap |
| `respiratory_support` | fitted | Device/mode marginals; **gated NIV**; phenotype ladders (HFNC vs NIPPV); IMV set-value quantiles |
| `medication_admin_continuous` | fitted | Infusion hazards + spine couplings; **sedation\|IMV ≈ reference** |
| `medication_admin_intermittent` | fitted | Stay prevalence + top-K `med_category` |
| `medication_orders` | derived | Folded from both med-admin tables on `med_order_id` |
| `patient_assessments` | fitted | Assessment-category marginal; RASS/GCS still spine-coupled |
| `position` | fitted | Prone rates (overall + among IMV) |
| `microbiology_culture` | fitted | Cultures/ICU-day + fluid/method; organism prior when source null |
| `microbiology_nonculture` | dashboard-prior† | Per-stay molecular-panel rate; pack-prefer when fitted |
| `microbiology_susceptibility` | derived | Panel per isolate from `microbiology_culture` (`organism_id`) |
| `crrt_therapy` | fitted | Stay prevalence + mode/rate quantiles; renal-flag windows + creat gate |
| `code_status` | fitted | Outcome-conditional DNR/AND rates |
| `ecmo_mcs` | fitted | Stay prevalence + device/mcs/flow quantiles |
| `invasive_hemodynamics` | dashboard-prior† | PA-catheter stay rate + cardiogenic vs distributive ranges |
| `transfusion` | dashboard-prior† | Acuity-scaled rate anchored to dashboard stay rate |
| `key_icu_orders` | dashboard-prior† | Rehab order stay fraction |
| `therapy_details` | dashboard-prior† | PT/OT session elements (same rehab gate) |
| `provider` | dashboard-prior† | Attending + nurse spanning each stay |
| `hospital_diagnosis` | prior + fitted pad | reference stay-prevalence disease/cancer priors (acuity-scaled); soft acute flag codes; top-K marginal **pads density only** (~18 codes/stay) |
| `patient_diagnosis` | prior | Same chronic/cancer priors as hospital_diagnosis; encounter dx from spine |
| `patient_procedures` | fitted | Stay prevalence + top-K `procedure_code` |
| `intake_output` | dashboard-prior† | Hourly balance; oliguria / resuscitation on spine flags |
| `place_based_index` | dashboard-prior† | One deprivation draw (ADI/SVI scales) |
| `clinical_trial` | dashboard-prior† | Enrolment rate for ventilated stays |

† Hybrid: generators prefer `pack.tables[<name>].params` when a fitted block is
present; otherwise dashboard / literature priors. The local source extract
(`~/Data/clif-source`) still omits these nine tables, so `icu_all28` remains
dashboard-prior for them until those parquets are staged and `run_fit` is re-run.

Output filenames for deliverable parquet follow CLIF 2.1 maturity badges only:
`clif_<table>_2.1_{beta|concept}.parquet`. The three DDL tables without a website
badge (`clinical_trial`, `patient_diagnosis`, `place_based_index`) remain
generated in memory for tests but are omitted from disk. Geography on
`hospitalization` shares the synthetic neighbourhood catalog with in-memory
`place_based_index`. Lab LOINC codes are curated from LOINC 2.82 (ACTIVE terms).

## What we did — fitted ICU realism (2026-08)

Work on branch `cursor/rename-truth-spine` brought generation into the reference ICU
statistical region and tightened longitudinal coherence:

1. **Fit path** — `icu_all28` pack + reference estimators for clinical tables present
   in the local extract; dashboard priors only for source-absent tables (hybrid
   pack-prefer path ready when those tables appear under `--real-dir`).
2. **Recalibrate** — `recalibrate_fitted_icu` (not anonymous network-median overwrite
   of fitted reference blocks): IMV/mortality/NIV/ADT targets, terminal mix, CRRT gate,
   `resp_phenotype_marginal`, sedation knobs; optional `ecmo_stay` for teaching presets.
3. **Trajectories** — sicker↔sicker coupling: vitals/labs track spine flags; soft
   L4→cv / L5→renal; terminal archetypes; **HR↔BP correlated AR(1) innovations**;
   WBC leukocytosis bump on IMV-tier / resp_flag.
4. **Respiratory pathways** — type1 NC→HFNC→IMV vs type2 NIPPV→IMV; NIV stay-gated
   so NIPPV/HFNC stay rates match reference (± few pp).
5. **Sedation** — continuous sedatives paired with IMV (`sedation_per_imv`).
6. **Diagnoses** — dropped universal pneumonia principal; reference disease/cancer
   stay prevalences within ±5 pp; codes/stay ~18; sicker stays carry more chronics.
7. **Validation** — `scripts/validate_against_real.py` probes for trajectories,
   coherence (vaso\|IMV, sedation\|IMV, decedent physiology, CRRT\|creat), plus
   `scripts/audit_realism_sources.py`; **CI always-on** envelope lock on `base_pack`
   (`tests/eval/test_base_pack_envelope.py`).
8. **Samples** — `sample_dataset/` and `sample_full_hospital/` regenerated from
   `icu_all28` (n=5000, seed 42) via the validated recalibrate paths. Empty
   `ecmo_mcs` at n=5k is expected; use preset `rare-support` for teaching.
9. **Rare-event teaching** — `presets/rare-support.toml` elevates ECMO/CRRT without
   changing default network-median rates.

## Notes

- fitted clinical blocks are **not** replaced by network-median priors when
  generating ICU data from `icu_all28`.
- Dashboard priors apply only to tables absent from the local source CLIF extract;
  generators prefer pack params when a fitted block exists.
- Derived tables inherit parent structure; susceptibility resistance panels remain
  literature norms.
- Demo / hand-prior runs without a reference pack still use documented constants so
  unit tests and tiny demos stay self-contained.
- Teaching presets (`rare-support`) are **not** network-rate calibrated.

**Release gate:** any public release of a generated dataset or the parameter pack
requires credentialed-data and Rush compliance confirmation.
Fitting locally to an aggregate pack is permitted; releasing derived artifacts is
gated.
