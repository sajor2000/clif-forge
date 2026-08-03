"""Unit tests for the Tier 3 vitals generator (U10, R9/R12/R22).

Driven by a small hand-built pack plus directly-constructed ``SpineFrame``s (no
real data, no run_fit): AR(1) values stay inside outlier bounds, blood pressure
falls under the cv-failure/high-acuity states (coupling), cadence is denser in
ICU intervals, categories are exact mCIDE members, datetimes are tz-aware UTC,
the frame passes the gate, and output is seed-reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import vitals
from clifforge.generate.tables.vitals import (
    VITALS,
    sample_vitals,
    vitals_frame,
)
from clifforge.reference import bounds, categories

_GRID = 1.0
_ADMIT = datetime(2021, 1, 1, tzinfo=UTC)


def _ar1(mean: float, phi: float = 0.5, sigma: float = 2.0) -> dict[str, float]:
    return {"mean": mean, "phi": phi, "sigma": sigma}


# Full 6-state blocks. sbp/map means fall at the high-acuity states (>=4, where the
# cv-failure flag is set) — the coupling under test.
def _six(means: dict[int, float], **kw: float) -> dict[str, dict[str, float]]:
    return {str(s): _ar1(m, **kw) for s, m in means.items()}


_SBP = _six({0: 125, 1: 123, 2: 118, 3: 112, 4: 92, 5: 90})
_MAP = _six({0: 88, 1: 87, 2: 84, 3: 80, 4: 65, 5: 63})
_HR = _six({0: 78, 1: 82, 2: 90, 3: 98, 4: 105, 5: 110})
_DBP = _six({0: 72, 1: 71, 2: 68, 3: 64, 4: 55, 5: 54})
_RR = _six({0: 16, 1: 17, 2: 20, 3: 24, 4: 28, 5: 30})
_SPO2 = _six({0: 98, 1: 97, 2: 95, 3: 93, 4: 91, 5: 90}, sigma=20.0)  # big sigma -> exercises clamp
_TEMP = _six({0: 37.0, 1: 37.1, 2: 37.4, 3: 37.8, 4: 38.2, 5: 38.5}, sigma=0.5)


def _pack(grid_step_hours: float = _GRID) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "vitals": {
                "n_records": 1000,
                "fitted": True,
                "params": {
                    "sbp_ar1_by_state": _SBP,
                    "map_ar1_by_state": _MAP,
                    "heart_rate_ar1_by_state": _HR,
                    "dbp_ar1_by_state": _DBP,
                    "respiratory_rate_ar1_by_state": _RR,
                    "spo2_ar1_by_state": _SPO2,
                    "temp_c_ar1_by_state": _TEMP,
                },
            },
            "spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}},
        },
    )


def _spine(levels: list[int], hid: str = "H0", cv: bool = False) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[False] * n,
        cv_flag=[lvl >= 4 or cv for lvl in levels],
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_is_deterministic_under_fixed_seed() -> None:
    pack = _pack()
    sp = _spine([0, 1, 3, 4, 2, 0, 3])
    a = sample_vitals(sp, pack, np.random.default_rng(7))
    b = sample_vitals(sp, pack, np.random.default_rng(7))
    assert a == b


def test_frame_reproducible_byte_for_byte() -> None:
    pack = _pack()
    sp = _spine([0, 1, 2, 3, 4, 3, 2, 1, 0], hid="H1")
    a = vitals_frame(sample_vitals(sp, pack, np.random.default_rng(3)))
    b = vitals_frame(sample_vitals(sp, pack, np.random.default_rng(3)))
    assert a.equals(b)


def test_all_values_within_outlier_bounds() -> None:
    pack = _pack()
    # A long, acuity-varying stay exercises every state and the spo2 clamp.
    sp = _spine([0, 1, 2, 3, 4, 5] * 10, hid="H2")
    for o in sample_vitals(sp, pack, np.random.default_rng(0)):
        lo, hi = bounds("vitals", o.vital_category)
        assert lo <= o.vital_value <= hi


def test_sbp_falls_under_cv_failure_states() -> None:
    pack = _pack()
    low = _spine([1] * 60, hid="Hlow")  # ward, no cv failure
    high = _spine([4] * 60, hid="Hhigh")  # vasopressor, cv-failure flag set
    low_sbp = [
        o.vital_value
        for o in sample_vitals(low, pack, np.random.default_rng(1))
        if o.vital_category == "sbp"
    ]
    high_sbp = [
        o.vital_value
        for o in sample_vitals(high, pack, np.random.default_rng(1))
        if o.vital_category == "sbp"
    ]
    assert low_sbp and high_sbp
    assert float(np.mean(high_sbp)) < float(np.mean(low_sbp)) - 15  # clear downward shift


def test_cadence_is_denser_in_icu_intervals() -> None:
    pack = _pack()
    # First 30 intervals ward (level 0), last 30 ICU (level 3).
    sp = _spine([0] * 30 + [3] * 30, hid="H3")
    obs = sample_vitals(sp, pack, np.random.default_rng(2), admit_dttm=_ADMIT)
    ward_rows = icu_rows = 0
    for o in obs:
        idx = int((o.recorded_dttm - _ADMIT).total_seconds() / 3600 / _GRID)
        if idx >= 30:
            icu_rows += 1
        else:
            ward_rows += 1
    n_vit = sum(1 for v in VITALS if f"{v}_ar1_by_state" in pack.tables["vitals"]["params"])
    icu_rate = icu_rows / (30 * n_vit)
    ward_rate = ward_rows / (30 * n_vit)
    assert icu_rate > ward_rate


def test_all_timestamps_within_stay() -> None:
    pack = _pack()
    sp = _spine([0, 2, 3, 4, 1], hid="H4")
    los = timedelta(hours=len(sp.support_level) * _GRID)
    for o in sample_vitals(sp, pack, np.random.default_rng(0), admit_dttm=_ADMIT):
        assert _ADMIT <= o.recorded_dttm < _ADMIT + los


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    ok = set(categories("vitals", "vital_category"))
    sp = _spine([0, 1, 2, 3, 4, 5], hid="H5")
    seen = {o.vital_category for o in sample_vitals(sp, pack, np.random.default_rng(0))}
    assert seen and seen <= ok


def test_names_echo_categories() -> None:
    pack = _pack()
    sp = _spine([2, 3, 4], hid="H6")
    for o in sample_vitals(sp, pack, np.random.default_rng(0)):
        assert o.vital_name == o.vital_category


def test_only_fitted_vitals_are_emitted() -> None:
    # A pack missing temp_c must emit no temp_c rows but still emit heart_rate.
    pack = _pack()
    del pack.tables["vitals"]["params"]["temp_c_ar1_by_state"]
    sp = _spine([1, 2, 3, 4], hid="H7")
    seen = {o.vital_category for o in sample_vitals(sp, pack, np.random.default_rng(0))}
    assert "temp_c" not in seen
    assert "heart_rate" in seen


def test_missing_state_falls_back_to_nearest() -> None:
    # sbp fit only for states 0 and 5; a level-3 stay resolves to nearest (5).
    pack = _pack()
    pack.tables["vitals"]["params"]["sbp_ar1_by_state"] = {
        "0": _ar1(200.0, sigma=1.0),
        "5": _ar1(80.0, sigma=1.0),
    }
    sp = _spine([3] * 40, hid="H8")
    sbp = [
        o.vital_value
        for o in sample_vitals(sp, pack, np.random.default_rng(0))
        if o.vital_category == "sbp"
    ]
    assert sbp
    assert float(np.mean(sbp)) < 120  # nearest to level 3 is state 5 (mean 80), not 0 (mean 200)


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    obs: list = []
    for i in range(20):
        levels = [0, 1, 2, 3, 4, 5, 3, 1][: 3 + (i % 5)]
        obs += sample_vitals(_spine(levels, hid=f"H{i}"), pack, np.random.default_rng(i))
    frame = vitals_frame(obs)
    dtype = frame.schema["recorded_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    report = gate.validate(frame, "vitals", run_secondary=False)
    assert report.pandera_passed


def test_missing_vitals_block_raises() -> None:
    empty = ParamPack(manifest={}, tables={"spine": {"params": {"state_model": {}}}})
    sp = _spine([1, 2], hid="H9")
    try:
        sample_vitals(sp, empty, np.random.default_rng(0))
    except ValueError as exc:
        assert "vitals" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a pack with no vitals block")


def test_module_exports() -> None:
    assert set(vitals.__all__) == {"VITALS", "VitalObservation", "sample_vitals", "vitals_frame"}


def test_hemodynamic_innovations_are_cross_correlated(monkeypatch: pytest.MonkeyPatch) -> None:
    """HR and SBP residual series co-move more than independent noise would."""
    monkeypatch.setattr(vitals, "_EMIT_PROB_ICU", 1.0)
    monkeypatch.setattr(vitals, "_EMIT_PROB_WARD", 1.0)
    pack = _pack()
    levels = [2] * 120
    obs = sample_vitals(_spine(levels, hid="Hc", cv=False), pack, np.random.default_rng(7))
    frame = vitals_frame(obs).with_columns(
        pl.col("recorded_dttm").dt.truncate("1h").alias("hour")
    )
    wide = (
        frame.filter(pl.col("vital_category").is_in(["heart_rate", "sbp"]))
        .pivot(
            on="vital_category",
            index="hour",
            values="vital_value",
            aggregate_function="first",
        )
        .drop_nulls()
        .sort("hour")
    )
    assert wide.height > 40
    dhr = np.diff(wide["heart_rate"].to_numpy())
    dsbp = np.diff(wide["sbp"].to_numpy())
    corr = float(np.corrcoef(dhr, dsbp)[0, 1])
    assert corr > 0.25
