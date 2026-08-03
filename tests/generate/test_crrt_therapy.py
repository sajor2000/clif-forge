"""Unit tests for the Tier 5 crrt_therapy generator (U18, R9/R12/R22)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import crrt_therapy as ct
from clifforge.generate.tables.crrt_therapy import crrt_therapy_frame, sample_crrt_therapy
from clifforge.reference import bounds, categories

_RATE_FIELDS = (
    "blood_flow_rate",
    "pre_filter_replacement_fluid_rate",
    "post_filter_replacement_fluid_rate",
    "dialysate_flow_rate",
)


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(renal: list[bool], hid: str = "H0") -> SpineFrame:
    n = len(renal)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[5 if r else 3 for r in renal],
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=renal,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine([True, True, False, True])
    a = sample_crrt_therapy(sp, pack, np.random.default_rng(0))
    b = sample_crrt_therapy(sp, pack, np.random.default_rng(0))
    assert a == b


def test_crrt_only_after_sustained_renal_failure() -> None:
    """CRRT charting starts only after 12 renal-flag hours (creatinine gate)."""
    pack = _pack()
    # 8h renal → still gated off; then more renal → emit only after the threshold.
    renal = [True] * 8 + [False, False] + [True] * 10
    rows = sample_crrt_therapy(_spine(renal), pack, np.random.default_rng(0))
    admit = datetime(2020, 1, 1, tzinfo=UTC)
    covered = {int((r.recorded_dttm - admit).total_seconds() // 3600) for r in rows}
    expected: set[int] = set()
    renal_hours = 0.0
    for i, flag in enumerate(renal):
        if flag:
            renal_hours += 1.0
            if renal_hours >= 12.0:
                expected.add(i)
    assert covered == expected
    assert len(rows) == len(expected)


def test_brief_renal_failure_yields_no_crrt() -> None:
    pack = _pack()
    assert sample_crrt_therapy(_spine([True] * 8), pack, np.random.default_rng(0)) == []


def test_no_renal_failure_yields_no_crrt() -> None:
    pack = _pack()
    assert sample_crrt_therapy(_spine([False] * 8), pack, np.random.default_rng(0)) == []


def test_rates_within_bounds() -> None:
    pack = _pack()
    rows = sample_crrt_therapy(_spine([True] * 40), pack, np.random.default_rng(1))
    assert rows  # sustained renal must clear the 12h gate
    for r in rows:
        frame_vals = {
            "blood_flow_rate": r.blood_flow_rate,
            "pre_filter_replacement_fluid_rate": r.pre_filter_replacement_fluid_rate,
            "post_filter_replacement_fluid_rate": r.post_filter_replacement_fluid_rate,
            "dialysate_flow_rate": r.dialysate_flow_rate,
        }
        for field, value in frame_vals.items():
            lo, hi = bounds("crrt_therapy", field)
            assert lo <= value <= hi
        assert r.ultrafiltration_out >= 0


def test_mode_is_mcide_member() -> None:
    pack = _pack()
    ok = set(categories("crrt_therapy", "crrt_mode_category"))
    rows = sample_crrt_therapy(_spine([True] * 20), pack, np.random.default_rng(0))
    assert rows
    for r in rows:
        assert r.crrt_mode_category in ok


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    rows: list = []
    for i in range(15):
        # Contiguous renal blocks long enough to clear the 12h sustained gate.
        renal = [True] * 20
        rows += sample_crrt_therapy(_spine(renal, hid=f"H{i}"), pack, np.random.default_rng(i))
    assert rows
    frame = crrt_therapy_frame(rows)
    dtype = frame.schema["recorded_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "crrt_therapy", run_secondary=False).pandera_passed


def test_module_exports() -> None:
    assert set(ct.__all__) == {"CrrtRow", "crrt_therapy_frame", "sample_crrt_therapy"}
