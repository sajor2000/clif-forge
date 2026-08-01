# Committed synthetic CLIF 2.1 sample (~10,000 encounters)

A **fully synthetic**, CLIF 2.1–conformant sample committed directly to the repo —
the largest that stays under GitHub's 100 MB/file limit (vitals ≈ 87 MB). It lets
you inspect realistic, multi-table output and test code without generating anything
or holding any credential.

- **10,000 ICU encounters**, ~16.4M rows, ~138 MB, 20 parquet files (19 CLIF tables
  + a synthetic-only `truth` benchmarking table).
- A representative draw of the shared **master** dataset (same generator, same
  network-median statistics).

## Guarantees

- **Fully synthetic.** No real patient records; no record maps to a real individual.
- **100% CLIF 2.1 mCIDE-conformant** — every table passes the schema + vocabulary +
  physiologic-bounds gate; zero orphan rows.
- **Reproducible byte-for-byte** — regenerate it from the committed base pack, spec,
  and seed (see below); the `manifest.json` records per-table content hashes.
- **Realistic** — autocorrelated vitals, realistic length-of-stay and measurement
  density, and deterioration-toward-death dynamics.

## Reproduce it

```bash
uv run clif-forge generate \
    --spec sample_dataset/spec.toml --base-pack base_pack \
    --n-patients 10000 --seed 42 --out ./reproduced
```

The result matches this directory byte-for-byte (compare `manifest.json` hashes).

## Files

One `clif_<table>_2.1_<beta|concept>.parquet` per CLIF 2.1 table, plus
`_truth.parquet` (synthetic-only latent labels for benchmarking), `manifest.json`
(provenance + hashes), and `spec.toml` (the recipe). Only real CLIF 2.1 tables
carry the `clif_` prefix, so `glob("clif_*.parquet")` loads the dataset without
picking up the spine. See `../DATA_DICTIONARY`-style column details in the tables
themselves.