"""Generate a synthetic CLIF ICU dataset — the off-the-shelf base, or a derivative.

Runs the end-to-end chain — fitted base pack -> population derivation ->
network-median recalibration -> chunked generation -> streaming concat — and
writes one ``clif_<table>_2.1_<maturity>.parquet`` per table. Chunking keeps peak memory bounded
for cohorts of ~100M+ rows.

With no rate/demographic flags it reproduces the **base** dataset (the shared,
off-the-shelf synthetic ICU cohort). Every flag below lets you spin a
**derivative** on the three axes users ask for:

* **size** — ``--n``
* **demographics** — ``--age-shift``, ``--hispanic-frac`` (race via the API)
* **illness rates** — ``--imv-rate``, ``--mortality-scale``, ``--vaso-frac``,
  ``--crrt-prob``, ``--prone-severe``
* **population** — ``--full-hospital`` swaps the ICU-only cohort for the full
  hospital population (ward/ED/stepdown/ICU mix, ~15.6% ICU, short ward-dominant
  LOS, ~2.1% mortality); the demographic/organ knobs still apply.

The generated dataset is a source-derived artifact and is **not** committed; only
this script is. Requires the staged real dataset (for the derivation's aggregate
age/med fit) and a fitted base pack.

Examples:
    # the off-the-shelf base dataset
    uv run python scripts/generate_deliverable.py \
        --base-pack data/param_packs/clif_refit --real-dir ~/Data/clif \
        --out ~/Desktop/clif_synthetic_chicago_icu

    # a 20k high-acuity, higher-mortality, more-Hispanic derivative
    uv run python scripts/generate_deliverable.py \
        --base-pack data/param_packs/clif_refit --real-dir ~/Data/clif \
        --out ~/Desktop/derivative --n 20000 \
        --imv-rate 0.55 --mortality-scale 1.4 --vaso-frac 0.45 --hispanic-frac 0.45

    # the full hospital population (ward/ED/stepdown/ICU mix)
    uv run python scripts/generate_deliverable.py \
        --base-pack data/param_packs/clif_refit --real-dir ~/Data/clif \
        --out ~/Desktop/clif_synthetic_full_hospital --full-hospital
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.filenames import is_deliverable_table, table_parquet_filename
from clifforge.generate.orchestrator import TRUTH_FILENAME, generate_dataset
from clifforge.generate.populations import (
    CHICAGO_ETHNICITY_TARGET,
    derive_chicago_population,
)
from clifforge.generate.recalibrate import (
    recalibrate_to_full_hospital,
    recalibrate_to_network_median,
)

BASE_SEED = 2025


def _build_pack(args: argparse.Namespace) -> ParamPack:
    """Chain base -> population derivation -> recalibration with the CLI overrides.

    ``--full-hospital`` routes through :func:`recalibrate_to_full_hospital` (the
    ward/ED/stepdown/ICU population) instead of the ICU-only network-median
    transform. The shared demographic and organ-support knobs still apply; the
    ICU-specific ``--imv-rate`` / ``--mortality-scale`` levers do not map onto the
    full-population transform (its ICU fraction and low mortality are set by the
    calibrated peak/mortality targets) and are ignored in that mode.
    """
    base = ParamPack.load(args.base_pack)
    ethnicity = None
    if args.hispanic_frac is not None:
        non_hisp = max(0.0, 1.0 - args.hispanic_frac - 0.02)
        ethnicity = {"Hispanic": args.hispanic_frac, "Non-Hispanic": non_hisp, "Unknown": 0.02}
    derived = derive_chicago_population(
        base,
        Path(args.real_dir).expanduser(),
        age_shift_years=args.age_shift,
        ethnicity_target=ethnicity or CHICAGO_ETHNICITY_TARGET,
    )
    flags = {"resp_flag": 0.5, "cv_flag": args.vaso_frac, "renal_flag": 0.05, "neuro_flag": 0.2}
    if args.full_hospital:
        return recalibrate_to_full_hospital(
            derived,
            flag_target_prevalence=flags,
            crrt_prob=args.crrt_prob,
            prone_prob_severe=args.prone_severe,
        )
    return recalibrate_to_network_median(
        derived,
        peak_imv_target=args.imv_rate,
        mortality_scale=args.mortality_scale,
        flag_target_prevalence=flags,
        crrt_prob=args.crrt_prob,
        prone_prob_severe=args.prone_severe,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-pack", required=True, help="Fitted base parameter pack directory.")
    ap.add_argument("--real-dir", required=True, help="Real CLIF dir (derivation inputs).")
    ap.add_argument("--out", required=True, help="Output directory for the dataset.")
    ap.add_argument("--n", type=int, default=85_248, help="Number of encounters (size).")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=8_000,
        help="Encounters per streamed chunk (RAM dial; smaller uses less memory). Default 8000.",
    )
    ap.add_argument(
        "--full-hospital",
        action="store_true",
        help="Generate the full hospital population (ward/ED/stepdown/ICU mix) instead "
        "of the ICU-only cohort. Ignores --imv-rate / --mortality-scale (the "
        "full-population ICU fraction and mortality are set by calibrated targets).",
    )
    # demographics
    ap.add_argument(
        "--age-shift", type=float, default=2.5, help="Years to shift the age quantiles."
    )
    ap.add_argument("--hispanic-frac", type=float, default=None, help="Target Hispanic fraction.")
    # illness rates (defaults = the off-the-shelf network-median base)
    ap.add_argument("--imv-rate", type=float, default=0.28, help="Reaches-IMV peak target.")
    ap.add_argument(
        "--mortality-scale", type=float, default=0.66, help="Peak-mortality multiplier."
    )
    ap.add_argument(
        "--vaso-frac", type=float, default=0.27, help="Cardiovascular-failure (vasopressor) rate."
    )
    ap.add_argument(
        "--crrt-prob", type=float, default=0.29, help="CRRT fraction among renal-failure stays."
    )
    ap.add_argument(
        "--prone-severe", type=float, default=0.026, help="Prone probability in severe hypoxemia."
    )
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    parts = out / "_parts"
    if out.exists():
        shutil.rmtree(out)
    parts.mkdir(parents=True)

    pack = _build_pack(args)

    chunk = args.chunk_size
    n_chunks = (args.n + chunk - 1) // chunk
    table_names: list[str] = []
    for c in range(n_chunks):
        size = min(chunk, args.n - c * chunk)
        ds = generate_dataset(pack, n_patients=size, seed=BASE_SEED + c, id_offset=c * chunk)
        for name, frame in ds.tables.items():
            if not is_deliverable_table(name):
                continue
            (parts / name).mkdir(exist_ok=True)
            frame.write_parquet(parts / name / f"chunk_{c:03d}.parquet")
        (parts / "truth").mkdir(exist_ok=True)
        ds.truth.write_parquet(parts / "truth" / f"chunk_{c:03d}.parquet")
        table_names = [n for n in ds.tables if is_deliverable_table(n)]
        print(f"chunk {c + 1}/{n_chunks} ({min((c + 1) * chunk, args.n):,}/{args.n:,})", flush=True)

    total = 0
    for name in [*table_names, "truth"]:
        dst = out / (TRUTH_FILENAME if name == "truth" else table_parquet_filename(name))
        pl.scan_parquet(str(parts / name / "*.parquet")).sink_parquet(dst)
        rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
        total += rows
        print(f"  {dst.stem:33s} {rows:>12,} rows", flush=True)
    shutil.rmtree(parts)
    print(f"\nDONE -> {out}  |  {args.n:,} encounters | {total:,} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
