# CLIFForge demo evaluation report (n=100, seed 42)

Generated 2026-08-01. All data is fully synthetic — sampled from an aggregate parameter pack, never copied from a real record.

**Reference:** a second independent synthetic draw (seed 43). These comparative numbers measure **generator self-consistency across seeds**, NOT fidelity to real patient data — computing real-data fidelity requires a credentialed real CLIF reference and is gated by the credentialed DUA.

## 1. Dataset

28 tables, 7,260 rows total.

| Table | Rows | Columns |
|---|---:|---:|
| `adt` | 127 | 8 |
| `clinical_trial` | 2 | 7 |
| `code_status` | 134 | 4 |
| `crrt_therapy` | 91 | 10 |
| `ecmo_mcs` | 63 | 12 |
| `hospital_diagnosis` | 409 | 5 |
| `hospitalization` | 100 | 10 |
| `intake_output` | 850 | 5 |
| `invasive_hemodynamics` | 50 | 5 |
| `key_icu_orders` | 74 | 5 |
| `labs` | 168 | 11 |
| `medication_admin_continuous` | 372 | 13 |
| `medication_admin_intermittent` | 136 | 13 |
| `medication_orders` | 296 | 15 |
| `microbiology_culture` | 8 | 13 |
| `microbiology_nonculture` | 60 | 18 |
| `microbiology_susceptibility` | 0 | 6 |
| `patient` | 100 | 11 |
| `patient_assessments` | 270 | 8 |
| `patient_diagnosis` | 304 | 7 |
| `patient_procedures` | 4 | 6 |
| `place_based_index` | 200 | 4 |
| `position` | 108 | 4 |
| `provider` | 200 | 6 |
| `respiratory_support` | 171 | 21 |
| `therapy_details` | 90 | 5 |
| `transfusion` | 70 | 8 |
| `vitals` | 2,803 | 6 |

## 2. Validation

All tables pass the primary (pandera) conformance gate.

| Table | Primary (pandera) | Secondary (clifpy) | Note |
|---|---|---|---|
| `adt` | pass | skipped | clifpy secondary gate disabled by caller |
| `clinical_trial` | pass | skipped | clifpy secondary gate disabled by caller |
| `code_status` | pass | skipped | clifpy secondary gate disabled by caller |
| `crrt_therapy` | pass | skipped | clifpy secondary gate disabled by caller |
| `ecmo_mcs` | pass | skipped | clifpy secondary gate disabled by caller |
| `hospital_diagnosis` | pass | skipped | clifpy secondary gate disabled by caller |
| `hospitalization` | pass | skipped | clifpy secondary gate disabled by caller |
| `intake_output` | pass | skipped | clifpy secondary gate disabled by caller |
| `invasive_hemodynamics` | pass | skipped | clifpy secondary gate disabled by caller |
| `key_icu_orders` | pass | skipped | clifpy secondary gate disabled by caller |
| `labs` | pass | skipped | clifpy secondary gate disabled by caller |
| `medication_admin_continuous` | pass | skipped | clifpy secondary gate disabled by caller |
| `medication_admin_intermittent` | pass | skipped | clifpy secondary gate disabled by caller |
| `medication_orders` | pass | skipped | clifpy secondary gate disabled by caller |
| `microbiology_culture` | pass | skipped | clifpy secondary gate disabled by caller |
| `microbiology_nonculture` | pass | skipped | clifpy secondary gate disabled by caller |
| `microbiology_susceptibility` | pass | skipped | clifpy secondary gate disabled by caller |
| `patient` | pass | skipped | clifpy secondary gate disabled by caller |
| `patient_assessments` | pass | skipped | clifpy secondary gate disabled by caller |
| `patient_diagnosis` | pass | skipped | clifpy secondary gate disabled by caller |
| `patient_procedures` | pass | skipped | clifpy secondary gate disabled by caller |
| `place_based_index` | pass | skipped | clifpy secondary gate disabled by caller |
| `position` | pass | skipped | clifpy secondary gate disabled by caller |
| `provider` | pass | skipped | clifpy secondary gate disabled by caller |
| `respiratory_support` | pass | skipped | clifpy secondary gate disabled by caller |
| `therapy_details` | pass | skipped | clifpy secondary gate disabled by caller |
| `transfusion` | pass | skipped | clifpy secondary gate disabled by caller |
| `vitals` | pass | skipped | clifpy secondary gate disabled by caller |

## 3. Fidelity

SDMetrics column-shape and column-pair similarity (1.0 = identical distribution). Mean quality across 26 tables: **0.955**.

| Table | Quality | Column shapes | Column pairs | Cols | Pairs |
|---|---:|---:|---:|---:|---:|
| `microbiology_culture` | 0.756 | 0.800 | 0.713 | 5 | 10 |
| `invasive_hemodynamics` | 0.837 | 0.823 | 0.851 | 3 | 1 |
| `patient` | 0.905 | 0.932 | 0.879 | 8 | 20 |
| `respiratory_support` | 0.911 | 0.918 | 0.905 | 16 | 16 |
| `microbiology_nonculture` | 0.914 | 0.929 | 0.898 | 9 | 20 |
| `hospitalization` | 0.923 | 0.930 | 0.917 | 5 | 6 |
| `position` | 0.926 | 0.926 | 0.926 | 2 | 1 |
| `crrt_therapy` | 0.934 | 0.917 | 0.951 | 7 | 11 |
| `patient_diagnosis` | 0.952 | 0.967 | 0.936 | 3 | 3 |
| `ecmo_mcs` | 0.954 | 0.929 | 0.979 | 10 | 21 |
| `adt` | 0.972 | 0.981 | 0.962 | 4 | 6 |
| `transfusion` | 0.973 | 0.964 | 0.982 | 4 | 3 |
| `medication_orders` | 0.973 | 0.977 | 0.969 | 8 | 20 |
| `hospital_diagnosis` | 0.978 | 0.981 | 0.975 | 4 | 2 |
| `labs` | 0.983 | 0.980 | 0.985 | 6 | 10 |
| `vitals` | 0.984 | 0.985 | 0.984 | 4 | 3 |
| `place_based_index` | 0.987 | 0.973 | 1.000 | 3 | 1 |
| `intake_output` | 0.988 | 0.986 | 0.990 | 3 | 1 |
| `code_status` | 0.993 | 0.993 | 0.993 | 2 | 1 |
| `medication_admin_continuous` | 0.993 | 0.992 | 0.993 | 10 | 20 |
| `patient_assessments` | 0.994 | 0.987 | 1.000 | 4 | 3 |
| `medication_admin_intermittent` | 0.999 | 0.999 | 0.999 | 10 | 20 |
| `clinical_trial` | 1.000 | 1.000 | - | 1 | 0 |
| `key_icu_orders` | 1.000 | 1.000 | 1.000 | 3 | 3 |
| `provider` | 1.000 | 1.000 | 1.000 | 2 | 1 |
| `therapy_details` | 1.000 | 1.000 | 1.000 | 3 | 3 |

## 4. Privacy

| Metric | Value | Reading |
|---|---:|---|
| DCR (median) | 2.159 | distance to the closest reference record |
| DCR (5th pct) | 1.185 | closest-match tail — none detected |
| NN-distance ratio (median) | 0.897 | near 1.0 = not singling out one record |
| Identifiability | 0.440 | fraction of reference records closer to a synthetic than to another reference record |

## 5. Utility (TSTR)

In-hospital mortality, LightGBM, both models scored on the same reference test split.

| Metric | Value |
|---|---:|
| TSTR AUC (trained on synthetic) | 0.610 |
| TRTR AUC (trained on reference) | 0.708 |
| Utility gap (TRTR - TSTR) | +0.098 |
| Features / test rows | 12 / 50 |
