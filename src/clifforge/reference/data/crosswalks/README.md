# Curated crosswalks (not consortium mCIDE)

| File | Role | Provenance |
|------|------|------------|
| `lab_category_loinc.csv` | CLIF `lab_category` → LOINC + specimen prior | Codes validated **ACTIVE** against LOINC **2.82** (`LoincTable/Loinc.csv`) |
| `micro_assay_loinc.csv` | Micro fluid / organism target → LOINC | Same LOINC 2.82 table |

Specimen category/name columns are a documented synthetic prior (serum/blood/plasma/…),
not an mCIDE list. Never invent LOINC strings outside these files.

## Adversarial ICU review (2026-08-03)

Reviewed every mapping as an intensivist would: analyte, specimen system, method,
and whether the code is what is actually ordered/resulted in ICU care — not merely
“a LOINC that mentions the word.”

### First-pass mistakes already corrected before this review

| Category | Wrong first pick | Why wrong | Correct |
|----------|------------------|-----------|---------|
| eosinophils_percent | 711-2 | Absolute count, not % | 713-8 |
| lymphocytes_percent | 731-0 | Absolute count, not % | 736-9 |
| monocytes_percent | 742-7 | Absolute count, not % | 5905-5 |
| glucose_fingerstick | 41604-0 | Fasting capillary | 41653-7 (glucometer) |
| ldh | 2532-0 | DISCOURAGED in 2.82 | 14804-9 |
| so2_mixed_venous | 2709-4 | Capillary blood | 19224-5 (BldMV) |
| so2_central_venous | 2711-0 | Peripheral venous | 97549-0 (BldCV) |
| bilirubin_conjugated | 1968-7 | “Direct” (= glucuronidated + albumin-bound) | 15152-2 (conjugated) |
| lactate | 2524-7 | Serum/plasma | 32693-4 (whole blood POC) |

### Fixes from this adversarial pass

| Mapping | Was | Now | Rationale |
|---------|-----|-----|-----------|
| `ptt` | 3173-2 (aPTT in **Blood**) | **14979-9** (aPTT in **PPP**) + plasma | Matches INR/PT coagulation panel; ICU coags are platelet-poor plasma |
| `respiratory_tract` culture | 6463-4 (generic Specimen) | **624-7** (Sputum respiratory culture) | Classic ICU respiratory culture order |
| `nasopharynx_upperairway` | 6463-4 | **43214-6** (Nasopharynx culture) | Site-specific |
| `larynx` | 6463-4 | **626-2** (Throat culture) | Closest charted site for laryngeal cultures |
| `skin` | 6462-6 (Wound) | **620-5** (Skin aerobe culture) | Skin ≠ wound |

### Deliberate nearest-neighbor (documented, not ideal)

| Mapping | Code | Why kept |
|---------|------|----------|
| `blood_buffy` | 600-7 (Blood culture) | No ACTIVE bacterial buffy-coat culture LOINC; blood culture is the clinical nearest |
| `brain` | 606-4 (CSF culture) | CLIF fluid “brain” in ICU practice is CNS/CSF workup, not tissue biopsy culture |
| `genito_urinary_tract` / renal pelvis fluids | 630-4 (Urine culture) | ICU GU cultures are overwhelmingly urine |

### Confirmed correct (no change)

Chem/CBC/ABG/coag/troponin core: albumin through wbc as in the CSV — Ser/Plas for chem,
Bld for CBC, BldA/BldV for blood gas, PPP for INR/PT/aPTT, whole-blood lactate,
cardiac troponin I/T, standard CRP (not hs-CRP). Molecular panels: C. diff toxin-gene
NAA `54067-4`, SARS-CoV-2 RNA `94500-6`, RSV RNA `92131-2`.
