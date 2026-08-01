# Committed synthetic CLIF 2.1 full-hospital sample (~8,000 encounters)

A **fully synthetic**, CLIF 2.1–conformant sample of a whole **hospital
population** — not just the ICU. Where `../sample_dataset/` is an ICU cohort
(every stay an ICU stay), this is the ward/ED/stepdown/ICU mix a real hospital
sees, with realistic patient flow. It lets you inspect and test code against a
full-hospital shape without generating anything or holding any credential.

- **~8,000 encounters**, ~30 MB, 29 parquet files (all 28 canonical CLIF 2.1 tables + a
  synthetic-only `truth` benchmarking table).
- A representative draw of the shared **full-hospital** dataset (same generator,
  same `full_hospital` mode).

## Patient flow (matches a real hospital)

- **~15% reach the ICU**; most stays never leave ward/ED acuity.
- **Arrivals are ER/OR-dominant** — ED ~57%, OR/procedural ~11%, direct-to-floor
  ward ~21%, direct-to-ICU ~6%, stepdown ~6%.
- **~20% direct admissions** (`admission_type_category`).
- **< 10% transfer into the ICU** (arrive elsewhere, deteriorate) — the rest of
  ICU access is direct/planned admits.
- **Hospital LOS median ~64 h**, in-hospital mortality ~2.2% — the low-acuity
  full-population regime, not the sicker ICU cohort.

## Guarantees

- **Fully synthetic.** No real patient records; no record maps to a real individual.
- **100% CLIF 2.1 mCIDE-conformant** — every table passes the schema + vocabulary +
  physiologic-bounds gate; zero orphan rows.
- **Reproducible byte-for-byte** — regenerate it from the committed base pack, spec,
  and seed (see below); the `manifest.json` records per-table content hashes.

## Reproduce it

```bash
uv run clif-forge generate \
    --spec sample_full_hospital/spec.toml --base-pack base_pack \
    --n-patients 8000 --seed 42 --out ./reproduced
```

The result matches this directory byte-for-byte. The only difference from the ICU
sample's recipe is `mode = "full_hospital"` in `spec.toml`.

Not real data — do not use for clinical decisions or epidemiologic conclusions.
