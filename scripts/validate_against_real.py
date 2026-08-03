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

    # Local MIMIC-IV Ext CLIF extract (untagged ``clif_<table>.parquet``)::

    uv run python scripts/validate_against_real.py \
        --synthetic sample_dataset \
        --real ~/Data/clif-mimic --out validation_mimic.json
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


_TRAJ_VITALS = ("map", "heart_rate", "spo2", "respiratory_rate")
_TRAJ_LABS = ("creatinine", "lactate", "bun", "ph_arterial")
_VASO_CATS = frozenset(
    {"norepinephrine", "vasopressin", "epinephrine", "phenylephrine", "dopamine"}
)

#: Clinical notes after MIMIC calibration choices.
CLINICAL_QUESTIONS: tuple[str, ...] = (
    "Decedent physiology kept steeper than MIMIC (terminal_vaso=0.35, "
    "terminal_renal=0.40). CRRT gate crrt_prob=0.95 → stay CRRT ~6–8% vs MIMIC 4%.",
)


def _trajectory(
    hosp: pl.DataFrame, series: pl.DataFrame, val_col: str, dttm_col: str
) -> dict[str, list[float | None]]:
    """Mean value in 12h bins over the last 48h before death/discharge, decedent vs survivor."""
    if series.height == 0 or dttm_col not in series.columns or val_col not in series.columns:
        return {"decedents": [None, None, None, None], "survivors": [None, None, None, None]}
    disch = hosp.select("hospitalization_id", "discharge_dttm")
    died = set(
        hosp.filter(pl.col("discharge_category") == "Expired")["hospitalization_id"].to_list()
    )
    joined = series.join(disch, on="hospitalization_id", how="inner").drop_nulls(val_col)
    if joined.height == 0:
        return {"decedents": [None, None, None, None], "survivors": [None, None, None, None]}
    hrs = (pl.col("discharge_dttm") - pl.col(dttm_col)).dt.total_hours()
    tagged = joined.with_columns(
        hrs.alias("_hrs"),
        pl.col("hospitalization_id").is_in(list(died)).alias("_died"),
    ).filter(pl.col("_hrs").is_between(0, 48))
    if tagged.height == 0:
        return {"decedents": [None, None, None, None], "survivors": [None, None, None, None]}
    tagged = tagged.with_columns(
        ((48 - pl.col("_hrs")) // 12).clip(0, 3).cast(pl.Int32).alias("_bin")
    )
    out: dict[str, list[float | None]] = {}
    for key, died_flag in (("decedents", True), ("survivors", False)):
        means: list[float | None] = []
        for b in range(4):
            cell = tagged.filter((pl.col("_died") == died_flag) & (pl.col("_bin") == b))
            means.append(
                round(float(cell[val_col].mean()), 1) if cell.height else None  # type: ignore[arg-type]
            )
        out[key] = means
    return out


def _event_prevalence_trajectory(
    hosp: pl.DataFrame,
    events: pl.DataFrame,
    *,
    dttm_col: str,
) -> dict[str, list[float | None]]:
    """Share of stays with ≥1 event in each last-48h 12h bin (decedent vs survivor)."""
    empty = {"decedents": [None, None, None, None], "survivors": [None, None, None, None]}
    if events.height == 0 or dttm_col not in events.columns:
        return empty
    disch = hosp.select("hospitalization_id", "discharge_dttm")
    died = set(
        hosp.filter(pl.col("discharge_category") == "Expired")["hospitalization_id"].to_list()
    )
    n_dec = max(1, len(died))
    n_surv = max(1, hosp["hospitalization_id"].n_unique() - len(died))
    joined = events.select("hospitalization_id", dttm_col).join(
        disch, on="hospitalization_id", how="inner"
    )
    hrs = (pl.col("discharge_dttm") - pl.col(dttm_col)).dt.total_hours()
    tagged = joined.with_columns(hrs.alias("_hrs")).filter(pl.col("_hrs").is_between(0, 48))
    if tagged.height == 0:
        return empty
    tagged = tagged.with_columns(
        ((48 - pl.col("_hrs")) // 12).clip(0, 3).cast(pl.Int32).alias("_bin"),
        pl.col("hospitalization_id").is_in(list(died)).alias("_died"),
    )
    out: dict[str, list[float | None]] = {}
    for key, died_flag, denom in (
        ("decedents", True, n_dec),
        ("survivors", False, n_surv),
    ):
        rates: list[float | None] = []
        for b in range(4):
            cell = tagged.filter((pl.col("_died") == died_flag) & (pl.col("_bin") == b))
            n_stay = cell["hospitalization_id"].n_unique()
            rates.append(round(n_stay / denom, 3) if denom else None)
        out[key] = rates
    return out


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
        columns=["hospitalization_id", "device_category", "recorded_dttm"],
    )
    rrs = pl.read_parquet(
        _table(real, "respiratory_support"),
        columns=["hospitalization_id", "device_category", "recorded_dttm"],
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
    )
    rvit = pl.read_parquet(
        _table(real, "vitals"),
        columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
    ).filter(pl.col("hospitalization_id").is_in(icu))

    for cat in _TRAJ_VITALS:
        out["trajectory"][cat] = {
            "synthetic": _trajectory(
                shosp,
                svit.filter(pl.col("vital_category") == cat),
                "vital_value",
                "recorded_dttm",
            ),
            "real": _trajectory(
                rhosp,
                rvit.filter(pl.col("vital_category") == cat),
                "vital_value",
                "recorded_dttm",
            ),
        }
    for cat in _TRAJ_LABS:
        out["trajectory"][cat] = {
            "synthetic": _trajectory(
                shosp,
                slab.filter(pl.col("lab_category") == cat).select(
                    "hospitalization_id", "lab_order_dttm", "lab_value_numeric"
                ),
                "lab_value_numeric",
                "lab_order_dttm",
            ),
            "real": _trajectory(
                rhosp,
                rlab.filter(pl.col("lab_category") == cat).select(
                    "hospitalization_id", "lab_order_dttm", "lab_value_numeric"
                ),
                "lab_value_numeric",
                "lab_order_dttm",
            ),
        }

    # Support intensity over last 48h: stay-share with IMV / vaso in each bin.
    def _vaso_events(base: Path, ids: pl.Series | None) -> pl.DataFrame:
        path = _table(base, "medication_admin_continuous")
        if not path.exists():
            return pl.DataFrame(schema={"hospitalization_id": pl.String, "admin_dttm": pl.Datetime})
        mac = pl.read_parquet(
            path, columns=["hospitalization_id", "admin_dttm", "med_category", "med_group"]
        )
        if ids is not None:
            mac = mac.filter(pl.col("hospitalization_id").is_in(ids))
        return mac.filter(
            pl.col("med_category").is_in(list(_VASO_CATS)) | (pl.col("med_group") == "vasoactives")
        ).select("hospitalization_id", "admin_dttm")

    def _imv_events(rs: pl.DataFrame) -> pl.DataFrame:
        if "recorded_dttm" not in rs.columns:
            return pl.DataFrame(
                schema={"hospitalization_id": pl.String, "recorded_dttm": pl.Datetime}
            )
        return rs.filter(pl.col("device_category") == "IMV").select(
            "hospitalization_id", "recorded_dttm"
        )

    out["trajectory"]["imv_prevalence"] = {
        "synthetic": _event_prevalence_trajectory(
            shosp, _imv_events(srs), dttm_col="recorded_dttm"
        ),
        "real": _event_prevalence_trajectory(rhosp, _imv_events(rrs), dttm_col="recorded_dttm"),
    }
    out["trajectory"]["vaso_prevalence"] = {
        "synthetic": _event_prevalence_trajectory(
            shosp, _vaso_events(synthetic, None), dttm_col="admin_dttm"
        ),
        "real": _event_prevalence_trajectory(
            rhosp, _vaso_events(real, icu), dttm_col="admin_dttm"
        ),
    }

    # Demographics (patient-level; real restricted to ICU-cohort patients).
    spat = pl.read_parquet(_table(synthetic, "patient"))
    rpat = pl.read_parquet(_table(real, "patient")).filter(
        pl.col("patient_id").is_in(
            rhosp.select("patient_id").unique().to_series()
            if "patient_id" in rhosp.columns
            else []
        )
    )

    def _frac(df: pl.DataFrame, col: str, value: str) -> float | None:
        if col not in df.columns or df.height == 0:
            return None
        return round(df.filter(pl.col(col) == value).height / df.height, 3)

    out["demographics"] = {
        "female": {
            "synthetic": _frac(spat, "sex_category", "Female"),
            "real": _frac(rpat, "sex_category", "Female"),
        },
        "white": {
            "synthetic": _frac(spat, "race_category", "White"),
            "real": _frac(rpat, "race_category", "White"),
        },
    }
    if "age_at_admission" in shosp.columns and "age_at_admission" in rhosp.columns:
        out["demographics"]["age_median"] = {
            "synthetic": round(float(shosp["age_at_admission"].median() or 0.0), 1),
            "real": round(float(rhosp["age_at_admission"].median() or 0.0), 1),
        }

    # Full device_category mix (row share) for MIMIC respiratory_support audits.
    def _device_mix(rs: pl.DataFrame) -> dict[str, float]:
        if rs.height == 0 or "device_category" not in rs.columns:
            return {}
        vc = rs.drop_nulls("device_category")["device_category"].value_counts()
        total = int(vc["count"].sum())
        if total == 0:
            return {}
        return {
            str(row[0]): round(int(row[1]) / total, 4)
            for row in vc.iter_rows()
        }

    out["device_category_mix"] = {
        "synthetic": _device_mix(srs),
        "real": _device_mix(rrs),
    }

    # Culture yield: share of cultures with a non-null organism (MIMIC may be empty).
    def _culture_yield(base: Path, ids: pl.Series | None) -> float | None:
        path = _table(base, "microbiology_culture")
        if not path.exists():
            return None
        cult = pl.read_parquet(path)
        if ids is not None and "hospitalization_id" in cult.columns:
            cult = cult.filter(pl.col("hospitalization_id").is_in(ids))
        if cult.height == 0 or "organism_category" not in cult.columns:
            return None
        non_null = cult.filter(pl.col("organism_category").is_not_null()).height
        return round(non_null / cult.height, 3)

    out["culture_yield"] = {
        "synthetic": _culture_yield(synthetic, None),
        "real": _culture_yield(real, icu),
    }

    # ADT front-door mix (validated arrival method): first location_category share.
    def _arrival_mix(base: Path, ids: pl.Series | None) -> dict[str, float]:
        path = _table(base, "adt")
        if not path.exists():
            return {}
        adt = pl.read_parquet(
            path, columns=["hospitalization_id", "location_category", "in_dttm"]
        )
        if ids is not None:
            adt = adt.filter(pl.col("hospitalization_id").is_in(ids))
        if adt.height == 0:
            return {}
        first = (
            adt.sort("in_dttm")
            .group_by("hospitalization_id")
            .agg(pl.col("location_category").first().alias("arr"))
        )
        vc = first["arr"].value_counts()
        total = int(vc["count"].sum())
        if total == 0:
            return {}
        return {str(row[0]): round(int(row[1]) / total, 4) for row in vc.iter_rows()}

    out["adt_arrival"] = {
        "synthetic": _arrival_mix(synthetic, None),
        "real": _arrival_mix(real, icu),
    }

    # Longitudinal coherence: sicker pairs with sicker across tables (not just rates).
    out["coherence"] = {
        "synthetic": _coherence(synthetic, shosp),
        "real": _coherence(real, rhosp),
    }
    return out


def _coherence(base: Path, hosp: pl.DataFrame) -> dict[str, float | None]:
    """Stay-level pairing metrics: IMV↔vaso, renal labs↔CRRT, decedent physiology."""
    n = hosp["hospitalization_id"].n_unique()
    if n == 0:
        return {}
    ids = hosp["hospitalization_id"]
    expired = set(
        hosp.filter(pl.col("discharge_category") == "Expired")["hospitalization_id"].to_list()
    )
    n_dec = len(expired)
    n_surv = n - n_dec

    def _safe_table(name: str, columns: list[str]) -> pl.DataFrame | None:
        path = _table(base, name)
        if not path.exists():
            return None
        df = pl.read_parquet(path, columns=columns)
        return df.filter(pl.col("hospitalization_id").is_in(ids))

    rs = _safe_table(
        "respiratory_support", ["hospitalization_id", "device_category", "recorded_dttm"]
    )
    mac = _safe_table(
        "medication_admin_continuous",
        ["hospitalization_id", "med_category", "med_group", "admin_dttm"],
    )
    labs = _safe_table(
        "labs", ["hospitalization_id", "lab_category", "lab_value_numeric", "lab_order_dttm"]
    )
    vit = _safe_table(
        "vitals", ["hospitalization_id", "vital_category", "vital_value", "recorded_dttm"]
    )
    crrt = _safe_table("crrt_therapy", ["hospitalization_id"])

    out: dict[str, float | None] = {}

    imv: set[str] = set()
    vaso: set[str] = set()
    if rs is not None:
        imv = set(
            rs.filter(pl.col("device_category") == "IMV")["hospitalization_id"].unique().to_list()
        )
    if mac is not None:
        if "med_category" in mac.columns:
            vaso |= set(
                mac.filter(pl.col("med_category").is_in(list(_VASO_CATS)))["hospitalization_id"]
                .unique()
                .to_list()
            )
        if "med_group" in mac.columns:
            vaso |= set(
                mac.filter(pl.col("med_group") == "vasoactives")["hospitalization_id"]
                .unique()
                .to_list()
            )

    if imv:
        out["vaso_given_imv"] = round(len(imv & vaso) / len(imv), 3)
    if vaso:
        out["imv_given_vaso"] = round(len(imv & vaso) / len(vaso), 3)

    # P(any continuous sedative | any IMV) — ICU doctor pairing.
    sed: set[str] = set()
    _SED = {
        "propofol",
        "fentanyl",
        "dexmedetomidine",
        "midazolam",
        "ketamine",
        "remifentanil",
        "morphine",
        "hydromorphone",
        "lorazepam",
        "pentobarbital",
    }
    if mac is not None and "med_category" in mac.columns:
        sed = set(
            mac.filter(pl.col("med_category").is_in(list(_SED)))["hospitalization_id"]
            .unique()
            .to_list()
        )
    if imv:
        out["sedation_given_imv"] = round(len(imv & sed) / len(imv), 3)
    if sed:
        out["imv_given_sedation"] = round(len(imv & sed) / len(sed), 3)
    if n_dec:
        out["imv_given_expired"] = round(len(imv & expired) / n_dec, 3)
        out["vaso_given_expired"] = round(len(vaso & expired) / n_dec, 3)
    if n_surv:
        surv = set(ids.to_list()) - expired
        out["imv_given_survivor"] = round(len(imv & surv) / n_surv, 3)
        out["vaso_given_survivor"] = round(len(vaso & surv) / n_surv, 3)

    if labs is not None and crrt is not None:
        creat = labs.filter(pl.col("lab_category") == "creatinine")
        if creat.height:
            peak = creat.group_by("hospitalization_id").agg(
                pl.col("lab_value_numeric").max().alias("peak")
            )
            # Clinical AKI-range threshold (mg/dL), not cohort quartile — quartile
            # is dominated by max-over-LOS sampling noise and is not comparable
            # across synth vs real with different panel density.
            high = set(peak.filter(pl.col("peak") >= 2.0)["hospitalization_id"].to_list())
            on_crrt = set(crrt["hospitalization_id"].unique().to_list())
            out["crrt_given_high_creat"] = (
                round(len(high & on_crrt) / len(high), 3) if high else None
            )
            out["high_creat_given_crrt"] = (
                round(len(high & on_crrt) / len(on_crrt), 3) if on_crrt else None
            )

    if labs is not None and vaso:
        lac = labs.filter(pl.col("lab_category") == "lactate")
        if lac.height:
            peak = lac.group_by("hospitalization_id").agg(
                pl.col("lab_value_numeric").max().alias("peak")
            )
            # Clinical hyperlactatemia threshold (mmol/L), not cohort quartile.
            high = set(peak.filter(pl.col("peak") >= 2.5)["hospitalization_id"].to_list())
            out["vaso_given_high_lactate"] = (
                round(len(high & vaso) / len(high), 3) if high else None
            )

    if expired and "discharge_dttm" in hosp.columns:
        disch = hosp.select("hospitalization_id", "discharge_dttm")
        n_ref = n_dec

        def _window_means(
            df: pl.DataFrame,
            *,
            dttm_col: str,
            val_col: str,
            category_col: str,
            category: str,
        ) -> pl.DataFrame:
            joined = df.filter(pl.col(category_col) == category).join(
                disch, on="hospitalization_id", how="inner"
            )
            if joined.height == 0:
                return pl.DataFrame(
                    schema={
                        "hospitalization_id": pl.String,
                        "early": pl.Float64,
                        "late": pl.Float64,
                    }
                )
            hrs = (pl.col("discharge_dttm") - pl.col(dttm_col)).dt.total_hours()
            return (
                joined.with_columns(hrs.alias("_hrs"))
                .filter(pl.col("hospitalization_id").is_in(list(expired)))
                .group_by("hospitalization_id")
                .agg(
                    pl.col(val_col)
                    .filter(pl.col("_hrs").is_between(24, 48))
                    .mean()
                    .alias("early"),
                    pl.col(val_col)
                    .filter(pl.col("_hrs").is_between(0, 24))
                    .mean()
                    .alias("late"),
                )
            )

        def _frac_dir(
            windows: pl.DataFrame, *, rises: bool, key: str, min_delta: float
        ) -> None:
            """Share of decedents with a clinically meaningful early→late change."""
            if not windows.height or not n_ref:
                return
            if rises:
                n_hit = windows.filter(
                    pl.col("early").is_not_null()
                    & pl.col("late").is_not_null()
                    & (pl.col("late") > pl.col("early") + min_delta)
                ).height
            else:
                n_hit = windows.filter(
                    pl.col("early").is_not_null()
                    & pl.col("late").is_not_null()
                    & (pl.col("late") < pl.col("early") - min_delta)
                ).height
            out[key] = round(n_hit / n_ref, 3)

        if vit is not None:
            _frac_dir(
                _window_means(
                    vit,
                    dttm_col="recorded_dttm",
                    val_col="vital_value",
                    category_col="vital_category",
                    category="map",
                ),
                rises=False,
                key="decedent_map_falls",
                min_delta=5.0,
            )
            _frac_dir(
                _window_means(
                    vit,
                    dttm_col="recorded_dttm",
                    val_col="vital_value",
                    category_col="vital_category",
                    category="heart_rate",
                ),
                rises=True,
                key="decedent_hr_rises",
                min_delta=5.0,
            )
            _frac_dir(
                _window_means(
                    vit,
                    dttm_col="recorded_dttm",
                    val_col="vital_value",
                    category_col="vital_category",
                    category="spo2",
                ),
                rises=False,
                key="decedent_spo2_falls",
                min_delta=2.0,
            )
        if labs is not None:
            _frac_dir(
                _window_means(
                    labs,
                    dttm_col="lab_order_dttm",
                    val_col="lab_value_numeric",
                    category_col="lab_category",
                    category="creatinine",
                ),
                rises=True,
                key="decedent_creat_rises",
                min_delta=0.3,
            )
            _frac_dir(
                _window_means(
                    labs,
                    dttm_col="lab_order_dttm",
                    val_col="lab_value_numeric",
                    category_col="lab_category",
                    category="lactate",
                ),
                rises=True,
                key="decedent_lactate_rises",
                min_delta=0.5,
            )

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
    lines.append("\ntrajectory (last 48h, 12h bins) — synth vs real:")
    for k, v in d["trajectory"].items():
        if "synthetic" in v and "real" in v:
            lines.append(f"  {k}:")
            lines.append(
                f"    decedents  synth {v['synthetic']['decedents']}  "
                f"real {v['real']['decedents']}"
            )
            lines.append(
                f"    survivors  synth {v['synthetic']['survivors']}  "
                f"real {v['real']['survivors']}"
            )
        else:
            # Backward-compatible single-arm shape
            lines.append(f"  {k} decedents {v['decedents']}  survivors {v['survivors']}")
    if "demographics" in d:
        lines.append("\ndemographics:")
        for k, v in d["demographics"].items():
            lines.append(f"  {k:14s} synth {v['synthetic']}  real {v['real']}")
    if "culture_yield" in d:
        cy = d["culture_yield"]
        lines.append(
            f"\nculture yield (non-null organism): synth {cy['synthetic']}  real {cy['real']}"
        )
    if "adt_arrival" in d:
        lines.append("\nADT arrival (first location):")
        keys = sorted(
            set(d["adt_arrival"]["synthetic"]) | set(d["adt_arrival"]["real"])
        )
        for k in keys:
            s = d["adt_arrival"]["synthetic"].get(k, 0.0)
            r = d["adt_arrival"]["real"].get(k, 0.0)
            lines.append(f"  {k:12s} synth {s:.3f}  real {r:.3f}")
    if "coherence" in d:
        lines.append("\nlongitudinal coherence (sicker↔sicker):")
        keys = sorted(set(d["coherence"]["synthetic"]) | set(d["coherence"]["real"]))
        for k in keys:
            s = d["coherence"]["synthetic"].get(k)
            r = d["coherence"]["real"].get(k)
            flag = ""
            if s is not None and r is not None and abs(float(s) - float(r)) > 0.15:
                flag = "  ⚠"
            lines.append(f"  {k:28s} synth {s}  real {r}{flag}")
    lines.append("\nclinical questions (no single MIMIC answer):")
    for q in CLINICAL_QUESTIONS:
        lines.append(f"  • {q}")
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
