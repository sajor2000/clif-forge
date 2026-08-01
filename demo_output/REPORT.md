# CLIFForge demo evaluation report (n=100, seed 42)

Generated 2026-08-01. All data is fully synthetic — sampled from an aggregate parameter pack, never copied from a real record.

**Reference:** a second independent synthetic draw (seed 43). These comparative numbers measure **generator self-consistency across seeds**, NOT fidelity to real patient data — computing real-data fidelity requires a credentialed real CLIF reference and is gated by the credentialed DUA.

## 1. Dataset

28 tables, 57,118 rows total.

| Table | Rows | Columns |
|---|---:|---:|
| `adt` | 972 | 8 |
| `clinical_trial` | 4 | 7 |
| `code_status` | 126 | 4 |
| `crrt_therapy` | 184 | 10 |
| `ecmo_mcs` | 205 | 12 |
| `hospital_diagnosis` | 463 | 5 |
| `hospitalization` | 100 | 10 |
| `intake_output` | 7,160 | 5 |
| `invasive_hemodynamics` | 324 | 5 |
| `key_icu_orders` | 290 | 5 |
| `labs` | 6,790 | 6 |
| `medication_admin_continuous` | 3,859 | 13 |
| `medication_admin_intermittent` | 1,011 | 13 |
| `medication_orders` | 1,299 | 15 |
| `microbiology_culture` | 140 | 13 |
| `microbiology_nonculture` | 56 | 18 |
| `microbiology_susceptibility` | 269 | 6 |
| `patient` | 100 | 8 |
| `patient_assessments` | 2,122 | 8 |
| `patient_diagnosis` | 394 | 7 |
| `patient_procedures` | 3 | 6 |
| `place_based_index` | 200 | 4 |
| `position` | 750 | 4 |
| `provider` | 200 | 6 |
| `respiratory_support` | 1,051 | 12 |
| `therapy_details` | 326 | 5 |
| `transfusion` | 85 | 8 |
| `vitals` | 28,635 | 6 |

## 2. Validation

All tables pass the primary (pandera) conformance gate.

| Table | Primary (pandera) | Secondary (clifpy) | Note |
|---|---|---|---|
| `adt` | pass | failed | clifpy reported 4 advisory issue(s) for 'adt' (recorded, not blocking — pandera  |
| `clinical_trial` | pass | skipped | no clifpy validator for 'clinical_trial' (pandera alone gates this table) |
| `code_status` | pass | failed | clifpy reported 2 advisory issue(s) for 'code_status' (recorded, not blocking —  |
| `crrt_therapy` | pass | failed | clifpy reported 2 advisory issue(s) for 'crrt_therapy' (recorded, not blocking — |
| `ecmo_mcs` | pass | failed | clifpy reported 5 advisory issue(s) for 'ecmo_mcs' (recorded, not blocking — pan |
| `hospital_diagnosis` | pass | failed | clifpy reported 1 advisory issue(s) for 'hospital_diagnosis' (recorded, not bloc |
| `hospitalization` | pass | failed | clifpy reported 9 advisory issue(s) for 'hospitalization' (recorded, not blockin |
| `intake_output` | pass | skipped | no clifpy validator for 'intake_output' (pandera alone gates this table) |
| `invasive_hemodynamics` | pass | skipped | no clifpy validator for 'invasive_hemodynamics' (pandera alone gates this table) |
| `key_icu_orders` | pass | skipped | no clifpy validator for 'key_icu_orders' (pandera alone gates this table) |
| `labs` | pass | failed | clifpy reported 7 advisory issue(s) for 'labs' (recorded, not blocking — pandera |
| `medication_admin_continuous` | pass | failed | clifpy reported 16 advisory issue(s) for 'medication_admin_continuous' (recorded |
| `medication_admin_intermittent` | pass | failed | clifpy reported 4 advisory issue(s) for 'medication_admin_intermittent' (recorde |
| `medication_orders` | pass | skipped | no clifpy validator for 'medication_orders' (pandera alone gates this table) |
| `microbiology_culture` | pass | failed | clifpy reported 7 advisory issue(s) for 'microbiology_culture' (recorded, not bl |
| `microbiology_nonculture` | pass | failed | clifpy reported 4 advisory issue(s) for 'microbiology_nonculture' (recorded, not |
| `microbiology_susceptibility` | pass | failed | clifpy reported 2 advisory issue(s) for 'microbiology_susceptibility' (recorded, |
| `patient` | pass | failed | clifpy reported 4 advisory issue(s) for 'patient' (recorded, not blocking — pand |
| `patient_assessments` | pass | failed | clifpy reported 5 advisory issue(s) for 'patient_assessments' (recorded, not blo |
| `patient_diagnosis` | pass | skipped | no clifpy validator for 'patient_diagnosis' (pandera alone gates this table) |
| `patient_procedures` | pass | failed | clifpy reported 1 advisory issue(s) for 'patient_procedures' (recorded, not bloc |
| `place_based_index` | pass | skipped | no clifpy validator for 'place_based_index' (pandera alone gates this table) |
| `position` | pass | failed | clifpy reported 1 advisory issue(s) for 'position' (recorded, not blocking — pan |
| `provider` | pass | skipped | no clifpy validator for 'provider' (pandera alone gates this table) |
| `respiratory_support` | pass | failed | clifpy reported 9 advisory issue(s) for 'respiratory_support' (recorded, not blo |
| `therapy_details` | pass | skipped | no clifpy validator for 'therapy_details' (pandera alone gates this table) |
| `transfusion` | pass | skipped | no clifpy validator for 'transfusion' (pandera alone gates this table) |
| `vitals` | pass | failed | clifpy reported 2 advisory issue(s) for 'vitals' (recorded, not blocking — pande |

## 3. Fidelity

SDMetrics column-shape and column-pair similarity (1.0 = identical distribution). Mean quality across 28 tables: **0.936**.

| Table | Quality | Column shapes | Column pairs | Cols | Pairs |
|---|---:|---:|---:|---:|---:|
| `patient_procedures` | 0.500 | 0.667 | 0.333 | 2 | 1 |
| `clinical_trial` | 0.833 | 0.833 | - | 1 | 0 |
| `patient` | 0.877 | 0.917 | 0.837 | 6 | 15 |
| `microbiology_culture` | 0.880 | 0.899 | 0.861 | 7 | 20 |
| `hospitalization` | 0.884 | 0.914 | 0.853 | 5 | 6 |
| `microbiology_susceptibility` | 0.889 | 0.924 | 0.853 | 5 | 10 |
| `invasive_hemodynamics` | 0.902 | 0.896 | 0.909 | 3 | 1 |
| `labs` | 0.934 | 0.944 | 0.925 | 3 | 1 |
| `transfusion` | 0.935 | 0.942 | 0.927 | 4 | 3 |
| `crrt_therapy` | 0.948 | 0.936 | 0.959 | 7 | 11 |
| `code_status` | 0.953 | 0.953 | 0.953 | 2 | 1 |
| `microbiology_nonculture` | 0.954 | 0.959 | 0.948 | 9 | 20 |
| `patient_diagnosis` | 0.957 | 0.973 | 0.941 | 3 | 3 |
| `position` | 0.962 | 0.962 | 0.962 | 2 | 1 |
| `respiratory_support` | 0.964 | 0.951 | 0.978 | 7 | 7 |
| `hospital_diagnosis` | 0.973 | 0.975 | 0.971 | 4 | 2 |
| `key_icu_orders` | 0.974 | 0.979 | 0.969 | 3 | 3 |
| `medication_orders` | 0.976 | 0.981 | 0.970 | 8 | 20 |
| `intake_output` | 0.976 | 0.980 | 0.972 | 3 | 1 |
| `medication_admin_continuous` | 0.976 | 0.981 | 0.972 | 10 | 20 |
| `ecmo_mcs` | 0.980 | 0.967 | 0.992 | 10 | 21 |
| `place_based_index` | 0.983 | 0.967 | 1.000 | 3 | 1 |
| `vitals` | 0.996 | 0.996 | 0.996 | 4 | 3 |
| `patient_assessments` | 0.998 | 0.996 | 1.000 | 4 | 3 |
| `adt` | 0.999 | 0.999 | 0.998 | 4 | 6 |
| `medication_admin_intermittent` | 1.000 | 1.000 | 1.000 | 10 | 20 |
| `provider` | 1.000 | 1.000 | 1.000 | 2 | 1 |
| `therapy_details` | 1.000 | 1.000 | 1.000 | 3 | 3 |

## 4. Privacy

| Metric | Value | Reading |
|---|---:|---|
| DCR (median) | 6.244 | distance to the closest reference record |
| DCR (5th pct) | 2.158 | closest-match tail — none detected |
| NN-distance ratio (median) | 0.948 | near 1.0 = not singling out one record |
| Identifiability | 0.550 | fraction of reference records closer to a synthetic than to another reference record |

## 5. Utility (TSTR)

In-hospital mortality, LightGBM, both models scored on the same reference test split.

| Metric | Value |
|---|---:|
| TSTR AUC (trained on synthetic) | 0.345 |
| TRTR AUC (trained on reference) | 0.607 |
| Utility gap (TRTR - TSTR) | +0.262 |
| Features / test rows | 54 / 50 |
