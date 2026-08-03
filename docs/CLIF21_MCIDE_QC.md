# CLIF 2.1.0 mCIDE deep QC

Pin re-diff: CLIF commit `966bc5fb…` equals tag `v2.1.0`; website maturity
`888353c0…`. Vendor dry-run: identical.

| Bucket | Count | Deliverable? |
|--------|------:|:------------:|
| beta | 9 | yes |
| concept | 16 | yes |
| untiered (null badge) | 3 | **no** — generated in memory only |

Deliverable parquet is **only** `clif_<table>_2.1_{beta|concept}.parquet` (25 tables).
Share packages must **not** include `_truth.parquet` (generator-internal spine) or
untiered DDL tables. Audit with `scripts/audit_share_package.py`.

## Gap closure (KNOWN_GAPS empty)

| Former gap | Resolution |
|------------|------------|
| Hospitalization geography (8) | Synthetic ZIP↔FIPS↔tract catalog; linked to in-memory `place_based_index` deprivation |
| Labs LOINC + specimen | Curated map validated **ACTIVE** against LOINC **2.82** |
| Micro culture/nonculture LOINC | Same LOINC 2.82 curated assay map |
| RS R10 set fields | Emitted; AC-VC→flow; Pressure Control→PC+Ti; NIPPV PIP alt |
| `vent_brand_name` | Hamilton Medical models: C6, G5, C3, C1, T1 |
| CRRT `dialysis_machine_name` | Prismaflex / PrisMax / NxStage / multiFiltratePRO |

## Automated gates

`audit_clif21` + coverage + mCIDE + RS/CRRT unit tests; every written `clif_*.parquet`
matches `*_2.1_(beta|concept).parquet`.
