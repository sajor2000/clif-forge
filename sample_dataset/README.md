# Committed synthetic CLIF 2.1 sample (~5,000 encounters)

A **fully synthetic**, CLIF 2.1–conformant sample committed directly to the repo —
sized so every file stays under GitHub's **50 MB soft limit** (vitals ≈ 42 MB). It
lets you inspect realistic, multi-table output and test code without generating
anything or holding any credential.

- **5,000 ICU encounters**, ~8.9M rows, ~76 MB, 29 parquet files — **all 28
  tables defined by the canonical CLIF 2.1 DDL**, plus `_truth.parquet`, a
  synthetic-only benchmarking artifact that is not a CLIF table.
- A representative draw of the shared **master** dataset (same generator, same
  network-median statistics, smaller `n`).

## Guarantees

- **Fully synthetic.** No real patient records; no record maps to a real individual.
- **100% CLIF 2.1 mCIDE-conformant** — every table passes the schema + vocabulary +
  physiologic-bounds gate; zero orphan rows.
- **Complete against the canonical schema.** Tables and columns come from the CLIF
  consortium's own DDL, not from the website's prose dictionary, and every
  canonical column is either populated or listed as a documented omission in
  `tests/generate/test_canonical_coverage.py`.
- **Joins that mean something.** `microbiology_susceptibility` resolves to the
  isolate its culture grew, and `medication_orders` is in exact correspondence
  with the administrations given under it — every order has administrations and
  every administration has an order.
- **Reproducible byte-for-byte** — regenerate it from the committed base pack, spec,
  and seed (see below); the `manifest.json` records per-table content hashes.
  Reproducibility is *within a code version*: the generators share one RNG stream
  per encounter, so changing a generator shifts every draw downstream of it.
- **Realistic** — autocorrelated vitals, realistic length-of-stay and measurement
  density, and deterioration-toward-death dynamics.

## Reproduce it

```bash
uv run clif-forge generate \
    --spec sample_dataset/spec.toml --base-pack base_pack \
    --n-patients 5000 --seed 42 --out ./reproduced
```

The result matches this directory byte-for-byte (compare `manifest.json` hashes).

## Files

One `clif_<table>_2.1_<beta|concept>.parquet` per canonical CLIF 2.1 table (28),
plus `_truth.parquet` (synthetic-only latent labels for benchmarking),
`manifest.json` (provenance + hashes), and `spec.toml` (the recipe). Only real
CLIF 2.1 tables carry the `clif_` prefix, so `glob("clif_*.parquet")` loads the
dataset without picking up the spine. See `../DATA_DICTIONARY`-style column
details in the tables themselves.

Not real data — do not use for clinical decisions or epidemiologic conclusions.
