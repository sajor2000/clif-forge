"""Unit tests for the Tier 4 respiratory_support generator (U12, R10/AE1/AE2/R22)."""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import respiratory_support as rs
from clifforge.generate.tables.respiratory_support import (
    DEVICE_SET_FIELDS,
    respiratory_support_frame,
    sample_respiratory_support,
)
from clifforge.reference import bounds, categories

_GRID = 1.0
_TRACH_N = 72  # must match _TRACH_MIN_IMV_INTERVALS

# The R10 device x mode matrix, written independently of the generator's own
# DEVICE_SET_FIELDS so a wrong entry there is caught rather than tautologically
# confirmed. Covers every device the generator can actually emit.
_R10_EXPECTED_SET_FIELDS = {
    "IMV": {"fio2_set", "tidal_volume_set", "resp_rate_set"},
    "High Flow NC": {"fio2_set", "lpm_set"},
    "Nasal Cannula": {"lpm_set"},
    "Room Air": set(),
    "Trach Collar": set(),
}


def _pack(grid_step_hours: float = _GRID) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(levels: list[int], hid: str = "H0", resp: bool = False) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[resp] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine([0, 1, 2, 3, 3, 1])
    assert sample_respiratory_support(sp, pack, np.random.default_rng(0)) == (
        sample_respiratory_support(sp, pack, np.random.default_rng(0))
    )


def test_each_device_populates_exactly_its_matrix_fields() -> None:
    pack = _pack()
    # A base-device spine plus a long-IMV-then-wean spine reaches every emittable
    # device, including Trach Collar. Assert against the independent R10 table.
    seen: set[str] = set()
    for levels, hid in (([0, 1, 2, 3], "Hm"), ([3] * _TRACH_N + [1], "Ht")):
        for row in sample_respiratory_support(
            _spine(levels, hid=hid), pack, np.random.default_rng(0)
        ):
            assert set(row.set_values) == _R10_EXPECTED_SET_FIELDS[row.device_category]
            seen.add(row.device_category)
    # The generator's own matrix must agree with the independent expectation for
    # every device it can emit (guards DEVICE_SET_FIELDS against silent drift).
    for device in seen:
        assert set(DEVICE_SET_FIELDS[device]) == _R10_EXPECTED_SET_FIELDS[device]
    assert {"Room Air", "Nasal Cannula", "High Flow NC", "IMV", "Trach Collar"} <= seen


def test_set_values_within_bounds() -> None:
    pack = _pack()
    sp = _spine([0, 1, 2, 3, 2, 1], hid="Hb", resp=True)
    for row in sample_respiratory_support(sp, pack, np.random.default_rng(1)):
        for field, value in row.set_values.items():
            lo, hi = bounds("respiratory_support", field)
            assert lo <= value <= hi


def test_ae1_trach_collar_implies_imv_off() -> None:
    pack = _pack()
    # Long IMV places a trach, then weaning to low-flow becomes a Trach Collar.
    sp = _spine([3] * _TRACH_N + [1] * 5, hid="Hae1")
    rows = sample_respiratory_support(sp, pack, np.random.default_rng(0))
    collars = [r for r in rows if r.device_category == "Trach Collar"]
    assert collars  # weaning produced a Trach Collar
    for r in collars:
        assert r.tracheostomy == 1
        assert r.set_values == {}  # IMV off: no ventilator set fields
        assert r.device_category != "IMV"


def test_ae2_tracheostomy_latches_and_persists() -> None:
    pack = _pack()
    # IMV long enough to trach, wean, then return to IMV: trach must stay 1.
    sp = _spine([3] * _TRACH_N + [1] * 5 + [3] * 5, hid="Hae2")
    rows = sample_respiratory_support(sp, pack, np.random.default_rng(0))
    latched = False
    for r in rows:
        if r.tracheostomy == 1:
            latched = True
        if latched:
            assert r.tracheostomy == 1  # never resets once set
    assert latched and rows[-1].tracheostomy == 1


def test_severe_hypoxemia_raises_imv() -> None:
    pack = _pack()
    with_resp = sample_respiratory_support(
        _spine([2] * 20, hid="Hr", resp=True), pack, np.random.default_rng(0)
    )
    without = sample_respiratory_support(
        _spine([2] * 20, hid="Hn", resp=False), pack, np.random.default_rng(0)
    )
    assert any(r.device_category == "IMV" for r in with_resp)
    assert all(r.device_category != "IMV" for r in without)


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    dev_ok = set(categories("respiratory_support", "device_category"))
    mode_ok = set(categories("respiratory_support", "mode_category"))
    sp = _spine([0, 1, 2, 3] * 3 + [2, 1], hid="Hc", resp=True)
    for r in sample_respiratory_support(sp, pack, np.random.default_rng(0)):
        assert r.device_category in dev_ok
        if r.mode_category is not None:
            assert r.mode_category in mode_ok


def test_imv_emits_obs_within_bounds_and_others_do_not() -> None:
    pack = _pack()
    rows = sample_respiratory_support(
        _spine([0, 1, 3, 3, 1], hid="Hobs"), pack, np.random.default_rng(0)
    )
    assert any(r.device_category == "IMV" and r.obs_values for r in rows)
    for r in rows:
        if r.device_category == "IMV":
            assert set(r.obs_values) == {
                "tidal_volume_obs",
                "resp_rate_obs",
                "plateau_pressure_obs",
                "peak_inspiratory_pressure_obs",
                "peep_obs",
                "minute_vent_obs",
                "mean_airway_pressure_obs",
            }
            for field, value in r.obs_values.items():
                lo, hi = bounds("respiratory_support", field)
                assert lo <= value <= hi
        else:
            assert r.obs_values == {}


def test_frame_carries_device_and_mode_names() -> None:
    pack = _pack()
    rows = sample_respiratory_support(
        _spine([0, 1, 2, 3], hid="Hn"), pack, np.random.default_rng(0)
    )
    frame = respiratory_support_frame(rows)
    assert "device_name" in frame.columns and "mode_name" in frame.columns
    assert frame["device_name"].null_count() == 0
    imv = frame.filter(pl.col("device_category") == "IMV")
    if imv.height:
        assert imv["tidal_volume_obs"].null_count() == 0


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    rows: list = []
    for i in range(15):
        levels = [0, 1, 2, 3, 3, 2, 1][: 3 + (i % 5)]
        rows += sample_respiratory_support(
            _spine(levels, hid=f"H{i}", resp=bool(i % 2)), pack, np.random.default_rng(i)
        )
    frame = respiratory_support_frame(rows)
    dtype = frame.schema["recorded_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "respiratory_support", run_secondary=False).pandera_passed


def test_empty_spine_yields_no_rows() -> None:
    pack = _pack()
    empty = SpineFrame(
        hospitalization_id="H0",
        support_level=[],
        resp_flag=[],
        cv_flag=[],
        renal_flag=[],
        neuro_flag=[],
        outcome="alive",
    )
    assert sample_respiratory_support(empty, pack, np.random.default_rng(0)) == []


def test_module_exports() -> None:
    assert set(rs.__all__) == {
        "DEVICE_SET_FIELDS",
        "RespiratorySupportRow",
        "respiratory_support_frame",
        "sample_respiratory_support",
    }
