# Committed synthetic CLIF 2.1 full-hospital sample (~5,000 encounters)

A **fully synthetic**, CLIF 2.1–conformant sample of a whole **hospital
population** — not just the ICU. Where `../sample_dataset/` is an ICU cohort
(every stay an ICU stay), this is the ward/ED/stepdown/ICU mix a real hospital
sees, with realistic patient flow.

- **5,000 encounters**, the **25** website-badged beta/concept CLIF 2.1 tables +
  `_truth.parquet` (untiered DDL tables omitted from deliverable parquet).
- Built from **`icu_all28`** (fitted clinical tables) through the validated
  **`recalibrate_to_full_hospital`** path (coupled `admission_route_marginal` →
  ADT front door + ward-dominant peak acuity).

**What each table is and how it was produced:** see repo-root
[`PROVENANCE.md`](../PROVENANCE.md).

## Reproduce it

```bash
uv run clif-forge generate \
    --spec sample_full_hospital/spec.toml \
    --base-pack data/param_packs/icu_all28 \
    --n-patients 5000 --seed 42 --out ./reproduced
```
