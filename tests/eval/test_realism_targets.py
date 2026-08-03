"""Local DUA realism regression guard (U6, R7).

Generates a cohort from the re-fit -> Chicago -> network-median chain and
asserts aggregate realism targets. Requires the local DUA-derived pack
(``data/param_packs/chicago_v2``); skipped in CI / fresh clones.

**CI always-on lock:** see ``test_base_pack_envelope.py`` +
``network_median_envelope.json`` (shareable ``base_pack``, no DUA data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import generate_dataset
from clifforge.generate.recalibrate import recalibrate_to_network_median

_PACK = Path("data/param_packs/chicago_v2")
_VASO = {
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "epinephrine",
    "dopamine",
    "dobutamine",
    "angiotensin",
    "milrinone",
    "isoproterenol",
}

pytestmark = pytest.mark.skipif(
    not _PACK.exists(),
    reason="requires the local DUA-derived chicago_v2 pack (not redistributable)",
)


@pytest.fixture(scope="module")
def cohort() -> dict[str, pl.DataFrame]:
    pack = recalibrate_to_network_median(ParamPack.load(str(_PACK)))
    return generate_dataset(pack, n_patients=4000, seed=99).tables


def _stay_rate(ds: dict[str, pl.DataFrame], table: str, expr: pl.Expr | None = None) -> float:
    d = ds[table] if expr is None else ds[table].filter(expr)
    return d["hospitalization_id"].n_unique() / ds["hospitalization"].height


def test_length_of_stay_in_range(cohort: dict[str, pl.DataFrame]) -> None:
    h = cohort["hospitalization"].with_columns(
        ((pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_seconds() / 3600).alias("h")
    )
    icu = (
        cohort["adt"]
        .filter(pl.col("location_category") == "icu")
        .with_columns(
            ((pl.col("out_dttm") - pl.col("in_dttm")).dt.total_seconds() / 3600).alias("h")
        )
        .group_by("hospitalization_id")
        .agg(pl.col("h").sum())
    )
    assert 120 < float(np.median(h["h"].to_list())) < 210  # hospital LOS ~165h
    assert 35 < float(np.median(icu["h"].to_list())) < 65  # ICU LOS ~47h


def test_life_support_rates_in_network_median_envelope(cohort: dict[str, pl.DataFrame]) -> None:
    h = cohort["hospitalization"]
    mortality = h.filter(pl.col("discharge_category") == "Expired").height / h.height
    assert 0.07 < mortality < 0.13  # network-median ~0.095
    assert (
        0.30 < _stay_rate(cohort, "respiratory_support", pl.col("device_category") == "IMV") < 0.50
    )
    assert (
        0.24
        < _stay_rate(
            cohort, "medication_admin_continuous", pl.col("med_category").is_in(list(_VASO))
        )
        < 0.40
    )
    assert 0.02 < _stay_rate(cohort, "crrt_therapy") < 0.07


def test_vitals_are_autocorrelated(cohort: dict[str, pl.DataFrame]) -> None:
    # U1: a patient's vital series is a smooth AR(1) walk, not white noise (~0).
    hr = (
        cohort["vitals"]
        .filter(pl.col("vital_category") == "heart_rate")
        .sort(["hospitalization_id", "recorded_dttm"])
    )
    acs = []
    for hid in hr["hospitalization_id"].unique().to_list()[:200]:
        s = hr.filter(pl.col("hospitalization_id") == hid)["vital_value"].to_numpy()
        if len(s) > 10:
            acs.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
    assert np.nanmean(acs) > 0.4  # strongly positive lag-1 autocorrelation


def test_lab_presence_matches_icu_reality(cohort: dict[str, pl.DataFrame]) -> None:
    # U2: near-universal chemistry presence in the ICU cohort.
    creat = _stay_rate(cohort, "labs", pl.col("lab_category") == "creatinine")
    assert creat > 0.9  # was ~0.73 before ICU-conditioning
