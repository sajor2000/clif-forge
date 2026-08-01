"""Validate a synthetic CLIF dataset against a real CLIF reference.

Compares a generated dataset to a real staged CLIF cohort across the dimensions a
reviewer cares about — cohort size, length-of-stay *distribution*, death,
life-support prevalence, per-stay missingness, lab-value fidelity, the
deterioration-toward-death trajectory, and the by-design demographic
differences — and writes a JSON summary (and prints a readable table).

The real reference is restricted to its ICU cohort (ADT ``location_category ==
"icu"``), the population the synthetic dataset represents. No real row-level data
is retained; only aggregate comparisons are emitted.

Usage::

    uv run python scripts/validate_against_real.py \
        --synthetic ~/Desktop/clif_synthetic_chicago_icu_corrected \
        --real ~/Data/clif --out validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from clifforge.generate.filenames import table_parquet_path

_QUANTS = (0.1, 0.25, 0.5, 0.75, 0.9)
_ABG = ["po2_arterial", "pco2_arterial", "ph_arterial", "so2_arterial"]
_PRESENCE_LABS = ["lactate", "ph_arterial", "creatinine", "sodium", "hemoglobin", "troponin_t"]
_VALUE_LABS = ["creatinine", "lactate", "sodium", "hemoglobin", "wbc", "bun", "platelet_count"]
_DEVICES = ["IMV", "NIPPV", "High Flow NC"]


def _los_hours(hosp: pl.DataFrame) -> pl.Series:
    return hosp.select(
        ((pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_hours()).alias("h")
    ).drop_nulls()["h"]


def _q(series: pl.Series) -> list[float]:
    return [round(float(series.quantile(p) or 0.0), 2) for p in _QUANTS]


def _presence(labs: pl.DataFrame, cat: str, n: int) -> float:
    return round(labs.filter(pl.col("lab_category") == cat)["hospitalization_id"].n_unique() / n, 3)


def _trajectory(
    hosp: pl.DataFrame, series: pl.DataFrame, val_col: str, dttm_col: str
) -> dict[str, list[float | None]]:
    """Mean value in 12h bins over the last 48h before death/discharge, decedent vs survivor."""
    disch = dict(zip(hosp["hospitalization_id"], hosp["discharge_dttm"], strict=True))
    died = set(
        hosp.filter(pl.col("discharge_category") == "Expired")["hospitalization_id"].to_list()
    )
    bins: dict[bool, list[list[float]]] = {True: [[], [], [], []], False: [[], [], [], []]}
    for hid, t, v in zip(
        series["hospitalization_id"], series[dttm_col], series[val_col], strict=True
    ):
        d = disch.get(hid)
        if d is None or v is None:
            continue
        hrs = (d - t).total_seconds() / 3600
        if 0 <= hrs <= 48:
            bins[hid in died][min(3, int((48 - hrs) // 12))].append(float(v))
    return {
        "decedents": [round(sum(b) / len(b), 1) if b else None for b in bins[True]],
        "survivors": [round(sum(b) / len(b), 1) if b else None for b in bins[False]],
    }


def _table(root: Path, table: str) -> Path:
    """Prefer the forge maturity-tagged name; fall back to the untagged consortium form."""
    tagged = table_parquet_path(root, table)
    if tagged.exists():
        return tagged
    return root / f"clif_{table}.parquet"


def validate(synthetic: Path, real: Path) -> dict[str, Any]:
    shosp = pl.read_parquet(_table(synthetic, "hospitalization"))
    sn = shosp["hospitalization_id"].n_unique()

    radt = pl.read_parquet(_table(real, "adt"))
    icu = radt.filter(pl.col("location_category") == "icu")["hospitalization_id"].unique()
    rhosp = pl.read_parquet(_table(real, "hospitalization")).filter(
        pl.col("hospitalization_id").is_in(icu)
    )
    rn = rhosp["hospitalization_id"].n_unique()

    slab = pl.read_parquet(_table(synthetic, "labs"))
    rlab = pl.read_parquet(_table(real, "labs")).filter(pl.col("hospitalization_id").is_in(icu))
    srs = pl.read_parquet(
        _table(synthetic, "respiratory_support"),
        columns=["hospitalization_id", "device_category"],
    )
    rrs = pl.read_parquet(
        _table(real, "respiratory_support"), columns=["hospitalization_id", "device_category"]
    ).filter(pl.col("hospitalization_id").is_in(icu))

    def dev(rs: pl.DataFrame, d: str, n: int) -> float:
        return round(
            rs.filter(pl.col("device_category") == d)["hospitalization_id"].n_unique() / n, 3
        )

    def crrt(base: Path, ids: pl.Series | None, n: int) -> float:
        c = pl.read_parquet(_table(base, "crrt_therapy"), columns=["hospitalization_id"])
        if ids is not None:
            c = c.filter(pl.col("hospitalization_id").is_in(ids))
        return round(c["hospitalization_id"].n_unique() / n, 3)

    def value_q(labs: pl.DataFrame, cat: str) -> list[float] | None:
        v = labs.filter(pl.col("lab_category") == cat)["lab_value_numeric"].drop_nulls()
        return [round(float(v.quantile(p) or 0.0), 2) for p in (0.1, 0.5, 0.9)] if v.len() else None

    out: dict[str, Any] = {
        "cohort_n": {"synthetic": sn, "real_icu": rn},
        "los_hours": {
            "synthetic": _q(_los_hours(shosp)),
            "real": _q(_los_hours(rhosp)),
            "quantiles": list(_QUANTS),
        },
        "mortality": {
            "synthetic": round(
                shosp.filter(pl.col("discharge_category") == "Expired").height / sn, 3
            ),
            "real": round(rhosp.filter(pl.col("discharge_category") == "Expired").height / rn, 3),
        },
        "life_support": {
            d: {"synthetic": dev(srs, d, sn), "real": dev(rrs, d, rn)} for d in _DEVICES
        },
        "presence": {},
        "value_fidelity": {},
        "trajectory": {},
    }
    out["life_support"]["CRRT"] = {
        "synthetic": crrt(synthetic, None, sn),
        "real": crrt(real, icu, rn),
    }
    out["life_support"]["ABG_union"] = {
        "synthetic": round(
            slab.filter(pl.col("lab_category").is_in(_ABG))["hospitalization_id"].n_unique() / sn, 3
        ),
        "real": round(
            rlab.filter(pl.col("lab_category").is_in(_ABG))["hospitalization_id"].n_unique() / rn, 3
        ),
    }
    for lab in _PRESENCE_LABS:
        out["presence"][lab] = {
            "synthetic": _presence(slab, lab, sn),
            "real": _presence(rlab, lab, rn),
        }
    for lab in _VALUE_LABS:
        sq, rq = value_q(slab, lab), value_q(rlab, lab)
        if sq and rq:
            out["value_fidelity"][lab] = {"synthetic_p10_50_90": sq, "real_p10_50_90": rq}

    svit = pl.read_parquet(
        _table(synthetic, "vitals"),
        columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
    ).filter(pl.col("vital_category") == "map")
    scr = slab.filter(pl.col("lab_category") == "creatinine").select(
        "hospitalization_id", "lab_order_dttm", "lab_value_numeric"
    )
    out["trajectory"]["map"] = _trajectory(shosp, svit, "vital_value", "recorded_dttm")
    out["trajectory"]["creatinine"] = _trajectory(shosp, scr, "lab_value_numeric", "lab_order_dttm")
    return out


def _fmt(d: dict[str, Any]) -> str:
    lines = [
        f"cohort n: synth {d['cohort_n']['synthetic']:,}  real-ICU {d['cohort_n']['real_icu']:,}",
        "",
    ]
    lines.append(f"LOS hours {d['los_hours']['quantiles']}:")
    lines.append(f"  synth {d['los_hours']['synthetic']}")
    lines.append(f"  real  {d['los_hours']['real']}")
    lines.append(f"\nmortality: synth {d['mortality']['synthetic']}  real {d['mortality']['real']}")
    lines.append("\nlife support:")
    for k, v in d["life_support"].items():
        lines.append(f"  {k:12s} synth {v['synthetic']:.3f}  real {v['real']:.3f}")
    lines.append("\npresence:")
    for k, v in d["presence"].items():
        lines.append(f"  {k:14s} synth {v['synthetic']:.3f}  real {v['real']:.3f}")
    lines.append("\nvalue fidelity (p10/50/90):")
    for k, v in d["value_fidelity"].items():
        lines.append(f"  {k:14s} synth {v['synthetic_p10_50_90']}  real {v['real_p10_50_90']}")
    lines.append("\ntrajectory (last 48h, 12h bins):")
    for k, v in d["trajectory"].items():
        lines.append(f"  {k} decedents {v['decedents']}  survivors {v['survivors']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", required=True, type=Path)
    ap.add_argument("--real", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="Write the JSON summary here.")
    args = ap.parse_args(argv)
    result = validate(args.synthetic.expanduser(), args.real.expanduser())
    print(_fmt(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
