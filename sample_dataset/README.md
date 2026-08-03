# Committed synthetic CLIF 2.1 sample (~5,000 encounters)

A **fully synthetic**, CLIF 2.1–conformant sample committed directly to the repo —
sized so every file stays under GitHub's **50 MB soft limit** (vitals ≈ 41 MB). It
lets you inspect realistic, multi-table output and test code without generating
anything or holding any credential.

- **5,000 ICU encounters**, all 28 canonical CLIF 2.1 tables plus `_truth.parquet`.
- Built from the **`mimic_all28`** parameter pack (fit on local MIMIC-IV Ext CLIF)
  through the validated **`recalibrate_mimic_icu`** path: spine tempering, LOS
  sojourns, gated NIV, terminal deterioration, and ADT front-door arrivals
  (`arrival_location_marginal` + `direct_icu_frac`).

## Guarantees

- **Fully synthetic.** No real patient records; no record maps to a real individual.
- **100% CLIF 2.1 mCIDE-conformant** — every table passes the schema + vocabulary +
  physiologic-bounds gate; zero orphan rows.
- **Joins that mean something.** `microbiology_susceptibility` resolves to the
  isolate its culture grew, and `medication_orders` is in exact correspondence
  with the administrations given under it.
- **Reproducible byte-for-byte** — regenerate from the committed pack, spec, and
  seed (see below).
- **MIMIC-empirical rates** — IMV / mortality / NIV / ADT arrivals track the
  local MIMIC ICU cohort within the validated recalibrate envelope.

## Reproduce it

```bash
uv run clif-forge generate \
    --spec sample_dataset/spec.toml \
    --base-pack data/param_packs/mimic_all28 \
    --n-patients 5000 --seed 42 --out ./reproduced
```

## What's not here

The full-size master (~85k encounters) is not in git. See the repo README for
how to generate a larger cohort from the same pack.
