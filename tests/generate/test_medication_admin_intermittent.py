"""Unit tests for the Tier 5 medication_admin_intermittent generator (U16, R11/R22)."""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import medication_admin_intermittent as mai
from clifforge.generate.tables.medication_admin_intermittent import (
    medication_admin_intermittent_frame,
    sample_medication_admin_intermittent,
)
from clifforge.reference import categories

# Continuous-only infusions that must never appear as discrete rows here.
_CONTINUOUS_MEDS = {"norepinephrine", "propofol"}


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(n: int, hid: str = "H0") -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[2] * n,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def _cohort(pack: ParamPack, rng: np.random.Generator, n_stays: int) -> list:
    rows = []
    for i in range(n_stays):
        rows += sample_medication_admin_intermittent(_spine(48, hid=f"H{i}"), pack, rng)
    return rows


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine(48)
    a = sample_medication_admin_intermittent(sp, pack, np.random.default_rng(0))
    b = sample_medication_admin_intermittent(sp, pack, np.random.default_rng(0))
    assert a == b


def test_all_rows_are_discrete_administrations() -> None:
    pack = _pack()
    rows = _cohort(pack, np.random.default_rng(1), 40)
    assert rows
    for r in rows:
        assert r.mar_action_category == "given"  # discrete, never an infusion action
        assert r.med_dose > 0  # a real dose, not a rate stop


def test_disjoint_from_continuous_infusions() -> None:
    pack = _pack()
    rows = _cohort(pack, np.random.default_rng(2), 40)
    for r in rows:
        assert r.med_category not in _CONTINUOUS_MEDS


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    med_ok = set(categories("medication_admin_intermittent", "med_category"))
    route_ok = set(categories("medication_admin_intermittent", "med_route_category"))
    action_ok = set(categories("medication_admin_intermittent", "mar_action_category"))
    rows = _cohort(pack, np.random.default_rng(3), 20)
    assert rows
    for r in rows:
        assert r.med_category in med_ok
        assert r.med_route_category in route_ok
        assert r.mar_action_category in action_ok


def test_prevalence_subset_of_stays() -> None:
    pack = _pack()
    rng = np.random.default_rng(4)
    on_abx = sum(
        bool(sample_medication_admin_intermittent(_spine(24, hid=f"H{i}"), pack, rng))
        for i in range(2000)
    )
    assert 0.4 < on_abx / 2000 < 0.6  # ~half of stays, per the prevalence constant


def test_doses_scheduled_across_stay() -> None:
    pack = _pack()
    # Force an antibiotic stay by drawing until one lands, then check cadence.
    rng = np.random.default_rng(0)
    rows: list = []
    for i in range(10):
        rows = sample_medication_admin_intermittent(_spine(48, hid=f"H{i}"), pack, rng)
        if rows:
            break
    vanco = sorted(r.admin_dttm for r in rows if r.med_category == "vancomycin")
    assert len(vanco) >= 2
    gap = (vanco[1] - vanco[0]).total_seconds() / 3600
    assert gap == 12.0  # q12h schedule


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    frame = medication_admin_intermittent_frame(_cohort(pack, np.random.default_rng(5), 30))
    dtype = frame.schema["admin_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "medication_admin_intermittent", run_secondary=False).pandera_passed


def test_module_exports() -> None:
    assert set(mai.__all__) == {
        "ORDERED_INTERVAL_HOURS",
        "MedIntermittentRow",
        "medication_admin_intermittent_frame",
        "sample_medication_admin_intermittent",
    }
