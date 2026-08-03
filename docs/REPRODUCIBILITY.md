# Reproducibility & Methods

Every artifact in this repository — the master dataset, the shareable base pack,
and every derivative variant — is produced by a **deterministic, scripted, seeded**
pipeline. Given the same inputs and seed, the output is **byte-for-byte identical**
(verified: regenerating any dataset with the same spec + seed produces identical
parquet files and identical content hashes).

## The pipeline

```
real CLIF ──fit──▶ base pack ──derive──▶ population pack ──recalibrate──▶ generation pack ──sample──▶ dataset + manifest
 (credentialed)   (aggregate)   (age/med/demographics)   (ICU or full-hospital)  (seeded, chunked)
```

Each stage is a documented function with no hidden state:

| Stage | Entry point | What it does | Determinism |
|---|---|---|---|
| **Fit** | `clif-forge fit` / `clifforge.fit.run_fit` | Learns aggregate parameters only (marginals, transitions, per-state physiology, correlations, prevalences) from real CLIF; **no row-level data retained**, every cell gated at n ≥ 20. Lab presence is conditioned on the clinically-defined ICU cohort (ADT location) and carries a fitted presence-correlation so co-ordered panels (metabolic panel, arterial blood gas) are generated together rather than independently. Lab values use an empirical quantile (inverse-CDF) marginal driven through the copula so skewed/multimodal shapes match real exactly — emitted only for labs with ≥ 20 records per grid interval (sparser labs keep the log-normal marginal, keeping the grid a bounded aggregate). Tables absent from the local extract stay **dashboard-prior**; when present, fit writes hybrid **pack-prefer** blocks (stay prevalence separate from intensity among positive stays). | Seeded patient split; robust estimators. |
| **Derive** | `clifforge.generate.populations.derive_chicago_population` | Re-weights demographics, shifts age quantiles, fits the med marginal. Deep-copies the input pack. | Pure function of pack + aggregate real stats. |
| **Recalibrate** | `recalibrate_fitted_icu` / `recalibrate_to_network_median` (ICU) / `recalibrate_to_full_hospital` (whole hospital) | Reshapes acuity/LOS/rates to the target population; adds length-aware generator paths and terminal deterioration. reference ICU samples use `recalibrate_fitted_icu` (preserves fitted reference blocks; gated NIV; RF phenotypes; ADT arrivals). Network-median presets still use `recalibrate_to_network_median` on `base_pack/` (including rare ECMO stay prevalence ≈ 0.09%). Full-hospital is ward-dominant and couples spine `admission_route` to ADT arrival. Optional `ecmo_stay` / `rare-support` elevate rare events for teaching. Deep-copies the input pack. | Documented parameters; no randomness. |
| **Sample** | `clif-forge generate` / `clifforge.generate.orchestrator.generate_dataset` | Draws every table from the pack via a single `SeedSequence(seed)` spawned per encounter; conformance-gated before write. Generators **pack-prefer**: fitted params when present, else dashboard priors. | `(seed, n, id_offset)` fully determines output (R22, AE6). |

## Reproduce the committed artifacts

**The ICU sample** (`sample_dataset/`, ~5k encounters) — from `icu_all28` +
`recalibrate_fitted_icu` (see `sample_dataset/spec.toml`):

```bash
uv run clif-forge generate \
    --spec sample_dataset/spec.toml \
    --base-pack data/param_packs/icu_all28 \
    --n-patients 5000 --seed 42 --out ./reproduced
# reproduced/manifest.json content hashes == sample_dataset/manifest.json
```

**The full-hospital sample** (`sample_full_hospital/`, ~5k encounters) — same
pack through `recalibrate_to_full_hospital`:

```bash
uv run clif-forge generate \
    --spec sample_full_hospital/spec.toml \
    --base-pack data/param_packs/icu_all28 \
    --n-patients 5000 --seed 42 --out ./reproduced-full
```

**A derivative variant** — any preset or spec from the shareable `base_pack/`:

```bash
uv run clif-forge presets   # high-acuity, older-cohort, sepsis-heavy, rare-support
uv run clif-forge generate --preset high-acuity --n-patients 5000 --seed 7 --out ./variant
```

**Per-table catalog** (fitted / prior / derived + fitted ICU realism notes):
[`PROVENANCE.md`](../PROVENANCE.md). Domain vocabulary: [`CONCEPTS.md`](../CONCEPTS.md).

**The base pack** (credentialed, one-time authoring step, needs real CLIF):

```bash
uv run python scripts/build_base_pack.py \
    --fitted-pack <fitted-pack> --real-dir <real-clif-dir> --out base_pack
```

**The full-size masters** (credentialed) — the ICU cohort (default) or, with
`--full-hospital`, the whole-hospital population:

```bash
uv run python scripts/generate_deliverable.py \
    --base-pack <fitted-pack> --real-dir <real-clif-dir> --out <dir>            # ICU master
uv run python scripts/generate_deliverable.py --full-hospital \
    --base-pack <fitted-pack> --real-dir <real-clif-dir> --out <dir> --n 365000 # full hospital
```

## Provenance record

Every generated dataset writes a `manifest.json` sidecar recording the generator
version, the resolved spec, the seed, and a per-table row count + SHA-256 content
hash. This makes each dataset reproducible from its recipe and makes any two
datasets provably distinct (`clifforge.manifest.datasets_are_distinct`).

## What the tests guarantee

- **Determinism** — identical `(seed, n, id_offset)` reproduces byte-identical output
  (`tests/test_cli.py`, `tests/generate/test_variants_end_to_end.py`).
- **Realism targets** — LOS, life-support rates, vitals autocorrelation, and lab
  presence stay in the real range (`tests/eval/test_realism_targets.py`, skip-guarded
  on the local pack).
- **Always-on network-median envelope** — shareable `base_pack` + seed-99 cohort
  asserts published bands *and* pinned `measured_*` anchors (mortality, IMV, vaso,
  CRRT, ABG Jaccard, ECMO rarity) in CI without DUA
  (`tests/eval/test_base_pack_envelope.py`, `tests/eval/network_median_envelope.json`).
- **Conformance** — every generated table passes the CLIF 2.1 schema + mCIDE + bounds
  gate before it is written; a non-conformant dataset cannot be produced.
- **No mutation** — `derive_*` and `recalibrate_*` never mutate their input pack, so
  one base pack seeds unlimited independent variants.

## Data-use note

The fit stage requires credentialed real CLIF data and emits only aggregate
parameters. The committed `base_pack/` and synthetic datasets are aggregate-derived /
fully synthetic (no PHI); their release is governed by the recorded compliance
determination (see the release gate and `COMPLIANCE_ACK.md`).
