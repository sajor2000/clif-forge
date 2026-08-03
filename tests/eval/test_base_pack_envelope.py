"""Always-on network-median envelope lock against the shareable ``base_pack``.

Unlike ``test_realism_targets.py`` (skipped without DUA ``chicago_v2``), this
module runs in CI: load committed ``base_pack/``, recalibrate to network median,
generate a fixed-seed cohort, and assert the published envelope in
``network_median_envelope.json``.

``measured`` anchors in that JSON pin the seed-99 means so widening ``lo``/``hi``
alone cannot green a broken generator without also re-measuring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import generate_dataset
from clifforge.generate.recalibrate import recalibrate_to_network_median

_BASE = Path("base_pack")
_ENVELOPE_PATH = Path(__file__).with_name("network_median_envelope.json")
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
_ABG = ("ph_arterial", "pco2_arterial", "po2_arterial")


def _require_base_pack() -> None:
    if _BASE.exists():
        return
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(
            "base_pack/ is required in CI for the network-median envelope lock; "
            "skipping would erase the realism gate"
        )
    pytest.skip("requires the shipped base pack")


pytestmark = pytest.mark.usefixtures("_base_pack_gate")


@pytest.fixture(scope="module")
def _base_pack_gate() -> None:
    _require_base_pack()


@pytest.fixture(scope="module")
def envelope() -> dict:
    return json.loads(_ENVELOPE_PATH.read_text())


@pytest.fixture(scope="module")
def cohort(envelope: dict) -> dict[str, pl.DataFrame]:
    pack = recalibrate_to_network_median(ParamPack.load(str(_BASE)))
    n = int(envelope.get("n_patients_ci", 900))
    seed = int(envelope.get("seed", 99))
    return generate_dataset(pack, n_patients=n, seed=seed).tables


def _stay_rate(ds: dict[str, pl.DataFrame], table: str, expr: pl.Expr | None = None) -> float:
    d = ds[table] if expr is None else ds[table].filter(expr)
    return d["hospitalization_id"].n_unique() / ds["hospitalization"].height


def _near(actual: float, expected: float, slack: float) -> None:
    assert abs(actual - expected) <= slack, f"{actual=} vs {expected=} (slack={slack})"


def test_length_of_stay_in_range(cohort: dict[str, pl.DataFrame], envelope: dict) -> None:
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
    hosp_med = float(np.median(h["h"].to_list()))
    icu_med = float(np.median(icu["h"].to_list()))
    hosp = envelope["hospital_los_hours_median"]
    icu_e = envelope["icu_los_hours_median"]
    assert hosp["lo"] < hosp_med < hosp["hi"]
    assert icu_e["lo"] < icu_med < icu_e["hi"]
    slack = float(envelope["measured_slack"]["los_hours"])
    _near(hosp_med, float(envelope["measured"]["hospital_los_hours_median"]), slack)
    _near(icu_med, float(envelope["measured"]["icu_los_hours_median"]), slack)


def test_life_support_rates_in_network_median_envelope(
    cohort: dict[str, pl.DataFrame], envelope: dict
) -> None:
    h = cohort["hospitalization"]
    mortality = h.filter(pl.col("discharge_category") == "Expired").height / h.height
    mort = envelope["mortality"]
    assert mort["lo"] < mortality < mort["hi"]
    imv = envelope["imv_stay"]
    imv_rate = _stay_rate(cohort, "respiratory_support", pl.col("device_category") == "IMV")
    assert imv["lo"] < imv_rate < imv["hi"]
    vaso = envelope["vaso_stay"]
    vaso_rate = _stay_rate(
        cohort, "medication_admin_continuous", pl.col("med_category").is_in(list(_VASO))
    )
    assert vaso["lo"] < vaso_rate < vaso["hi"]
    crrt = envelope["crrt_stay"]
    crrt_rate = _stay_rate(cohort, "crrt_therapy")
    assert crrt["lo"] < crrt_rate < crrt["hi"]
    rate_slack = float(envelope["measured_slack"]["rate"])
    m = envelope["measured"]
    _near(mortality, float(m["mortality"]), rate_slack)
    _near(imv_rate, float(m["imv_stay"]), rate_slack)
    _near(vaso_rate, float(m["vaso_stay"]), rate_slack)
    _near(crrt_rate, float(m["crrt_stay"]), rate_slack)


def test_vitals_are_autocorrelated(cohort: dict[str, pl.DataFrame], envelope: dict) -> None:
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
    mean_ac = float(np.nanmean(acs))
    assert mean_ac > envelope["heart_rate_ar1_lag1_mean"]["lo"]
    _near(
        mean_ac,
        float(envelope["measured"]["heart_rate_ar1_lag1_mean"]),
        float(envelope["measured_slack"]["ar1"]),
    )


def test_lab_presence_matches_icu_reality(
    cohort: dict[str, pl.DataFrame], envelope: dict
) -> None:
    creat = _stay_rate(cohort, "labs", pl.col("lab_category") == "creatinine")
    assert creat > envelope["creatinine_presence"]["lo"]
    _near(
        creat,
        float(envelope["measured"]["creatinine_presence"]),
        float(envelope["measured_slack"]["rate"]),
    )


def test_abg_panel_co_occurs(cohort: dict[str, pl.DataFrame], envelope: dict) -> None:
    """Arterial blood-gas members should co-occur within stays (presence copula)."""
    labs = cohort["labs"].filter(pl.col("lab_category").is_in(list(_ABG)))
    assert not labs.is_empty(), "ABG labs must be present in base_pack CI cohort"
    present = (
        labs.select(["hospitalization_id", "lab_category"])
        .unique()
        .with_columns(pl.lit(1).alias("p"))
        .pivot(on="lab_category", index="hospitalization_id", values="p", aggregate_function="max")
        .fill_null(0)
    )
    cols = [c for c in _ABG if c in present.columns]
    assert len(cols) >= 2, f"expected >=2 ABG categories, got {cols}"
    jaccards: list[float] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            both = int(((present[a] == 1) & (present[b] == 1)).sum())
            either = int(((present[a] == 1) | (present[b] == 1)).sum())
            if either:
                jaccards.append(both / either)
    mean_j = float(np.mean(jaccards))
    assert jaccards and mean_j >= envelope["abg_jaccard_min"]["lo"]
    _near(
        mean_j,
        float(envelope["measured"]["abg_jaccard_mean"]),
        float(envelope["measured_slack"]["jaccard"]),
    )


def test_ecmo_empty_allowed_at_sample_scale(
    cohort: dict[str, pl.DataFrame], envelope: dict
) -> None:
    """At network-median rates, ECMO stay prevalence stays near zero at CI scale."""
    n = cohort["hospitalization"].height
    limit = int(envelope["ecmo_mcs_empty_ok_at_n_le"])
    assert "ecmo_mcs" in cohort
    if n <= limit:
        stay_rate = cohort["ecmo_mcs"]["hospitalization_id"].n_unique() / n
        # Network median ≈0.09%; allow a small Poisson tail at n≈900.
        assert stay_rate < 0.02
        _near(
            stay_rate,
            float(envelope["measured"]["ecmo_stay"]),
            float(envelope["measured_slack"]["rate"]),
        )
