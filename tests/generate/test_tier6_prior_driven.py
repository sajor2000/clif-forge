"""Tests for the Tier 6 Concept tables (U20; R5/R14/R22).

Six tables generated from documented literature/clinical rates or dashboard
priors keyed to spine acuity (some later promoted to fitted when a source pack
block exists). Each is checked for its acuity coupling, exact mCIDE membership
where the dictionary defines it, conformance-gate pass, and seed reproducibility.
A provenance check confirms each is labeled fitted or dashboard-prior.
"""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import invasive_hemodynamics as ih
from clifforge.generate.tables.ecmo_mcs import ecmo_mcs_frame, sample_ecmo_mcs
from clifforge.generate.tables.invasive_hemodynamics import (
    invasive_hemodynamics_frame,
    sample_invasive_hemodynamics,
)
from clifforge.generate.tables.key_icu_orders import key_icu_orders_frame, sample_key_icu_orders
from clifforge.generate.tables.provider import provider_frame, sample_provider
from clifforge.generate.tables.therapy_details import (
    sample_therapy_details,
    therapy_details_frame,
)
from clifforge.generate.tables.transfusion import sample_transfusion, transfusion_frame
from clifforge.reference import categories


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(
    levels: list[int],
    *,
    cv: bool = False,
    hid: str = "H0",
    outcome: str = "alive",
) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[False] * n,
        cv_flag=[cv] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome=outcome,
    )


# --- ecmo_mcs ---------------------------------------------------------------- #
def test_ecmo_only_at_top_of_ladder() -> None:
    pack = _pack()
    with_ecmo = sample_ecmo_mcs(_spine([3, 4, 5, 5, 4]), pack, np.random.default_rng(0))
    without = sample_ecmo_mcs(_spine([3, 4, 4, 3]), pack, np.random.default_rng(0))
    assert len(with_ecmo) == 2  # two level-5 intervals
    assert without == []


def test_ecmo_deterministic_and_gates() -> None:
    pack = _pack()
    sp = _spine([5] * 6, hid="He")
    a = sample_ecmo_mcs(sp, pack, np.random.default_rng(1))
    b = sample_ecmo_mcs(sp, pack, np.random.default_rng(1))
    assert a == b
    assert gate.validate(ecmo_mcs_frame(a), "ecmo_mcs", run_secondary=False).pandera_passed


def _hemo_with_catheter(
    levels: list[int],
    pack: ParamPack,
    *,
    cv: bool = True,
    hid: str = "H0",
) -> list:
    """Draw until the stay-level PA-catheter prevalence gate accepts a stay."""
    for seed in range(200):
        rows = sample_invasive_hemodynamics(
            _spine(levels, cv=cv, hid=hid), pack, np.random.default_rng(seed)
        )
        if rows:
            return rows
    return []


# --- invasive_hemodynamics --------------------------------------------------- #
def test_hemodynamics_only_during_cv_failure() -> None:
    pack = _pack()
    shock = _hemo_with_catheter([4] * 12, pack, cv=True)
    stable = sample_invasive_hemodynamics(
        _spine([4] * 12, cv=False), pack, np.random.default_rng(0)
    )
    assert shock and stable == []


def test_hemodynamics_categories_and_gate() -> None:
    pack = _pack()
    ok = set(categories("invasive_hemodynamics", "measure_category"))
    rows = _hemo_with_catheter([4] * 30, pack, cv=True)
    assert rows and all(r.measure_category in ok for r in rows)
    assert gate.validate(
        invasive_hemodynamics_frame(rows), "invasive_hemodynamics", run_secondary=False
    ).pandera_passed


def test_hemodynamics_records_what_was_measured() -> None:
    """Each row must carry a value in its measure's physiologic range.

    A hemodynamics table that logs that a CVP was taken but not what it read is
    not usable for anything, which is what this table used to emit.
    """
    pack = _pack()
    rows = _hemo_with_catheter([4] * 60, pack, cv=True)
    assert rows
    for row in rows:
        spans = [ih._RANGES[p][row.measure_category] for p in ih._RANGES]
        assert min(lo for lo, _ in spans) <= row.measure_value <= max(hi for _, hi in spans)


def test_shock_phenotype_is_consistent_within_a_stay() -> None:
    """Filling pressure and cardiac output must move in opposite directions.

    Cardiogenic shock backs pressure up behind a failing pump; distributive shock
    runs empty and fast. Drawing them independently would erase both phenotypes
    and leave a cohort of uniformly mid-range numbers.
    """
    pack = _pack()
    high_cvp_with_low_output = 0
    low_cvp_with_high_output = 0
    for seed in range(80):
        rows = sample_invasive_hemodynamics(
            _spine([4] * 200, cv=True), pack, np.random.default_rng(seed)
        )
        cvp = [r.measure_value for r in rows if r.measure_category == "cvp"]
        co = [
            r.measure_value for r in rows if r.measure_category == "cardiac_output_thermodilution"
        ]
        if not cvp or not co:
            continue
        mean_cvp, mean_co = sum(cvp) / len(cvp), sum(co) / len(co)
        # The two phenotypes are disjoint on this pair; a stay must sit in one.
        assert not (mean_cvp > 10 and mean_co > 5), "a failing pump cannot also run fast"
        high_cvp_with_low_output += mean_cvp > 10 and mean_co < 5
        low_cvp_with_high_output += mean_cvp < 10 and mean_co > 5
    assert high_cvp_with_low_output and low_cvp_with_high_output


# --- transfusion ------------------------------------------------------------- #
def test_transfusion_scales_with_peak_acuity() -> None:
    pack = _pack()
    rng = np.random.default_rng(3)
    high = sum(
        len(sample_transfusion(_spine([5] * 24, hid=f"H{i}"), pack, rng)) for i in range(300)
    )
    low = sum(len(sample_transfusion(_spine([1] * 24, hid=f"L{i}"), pack, rng)) for i in range(300))
    assert high > low  # sicker (higher peak_level) stays transfuse more


def test_transfusion_ordering_and_gate() -> None:
    pack = _pack()
    rows = sample_transfusion(_spine([5] * 48, hid="Ht"), pack, np.random.default_rng(0))
    for r in rows:
        assert r.transfusion_start_dttm < r.transfusion_end_dttm
        assert r.volume_transfused > 0
    starts = [r.transfusion_start_dttm for r in rows]
    assert starts == sorted(starts)
    assert gate.validate(transfusion_frame(rows), "transfusion", run_secondary=False).pandera_passed


# --- key_icu_orders ---------------------------------------------------------- #
def test_key_icu_orders_categories_and_subset() -> None:
    pack = _pack()
    ok = set(categories("key_icu_orders", "order_category"))
    rng = np.random.default_rng(4)
    with_orders = 0
    for i in range(400):
        rows = sample_key_icu_orders(_spine([2, 3, 3, 2], hid=f"H{i}"), pack, rng)
        with_orders += bool(rows)
        for r in rows:
            assert r.order_category in ok
    # Dashboard prior: ~35% of ICU stays get a rehab consult (± sampling noise).
    assert 0.25 < with_orders / 400 < 0.45


def test_key_icu_orders_starts_with_evaluation_and_gates() -> None:
    pack = _pack()
    # Draw until an ordered stay lands, then check structure.
    rng = np.random.default_rng(0)
    rows: list = []
    for i in range(10):
        rows = sample_key_icu_orders(_spine([2] * 96, hid=f"H{i}"), pack, rng)
        if rows:
            break
    assert any(r.order_category == "PT_evaluation" for r in rows)
    assert any(r.order_category == "PT_treat" for r in rows)  # daily treatments follow
    assert gate.validate(
        key_icu_orders_frame(rows), "key_icu_orders", run_secondary=False
    ).pandera_passed


def test_no_icu_no_orders() -> None:
    pack = _pack()
    assert sample_key_icu_orders(_spine([0, 1, 1, 0]), pack, np.random.default_rng(0)) == []


# --- therapy_details --------------------------------------------------------- #
def test_therapy_details_gate_and_tz_aware_timestamp() -> None:
    """``session_start_dttm`` is a real UTC timestamp, not an ISO string.

    It used to be a string only because the website's untyped prose dictionary
    defaulted it to one; the canonical DDL types it DATETIME.
    """
    pack = _pack()
    rng = np.random.default_rng(0)
    rows: list = []
    for i in range(10):
        rows = sample_therapy_details(_spine([2] * 96, hid=f"H{i}"), pack, rng)
        if rows:
            break
    assert rows
    assert all(r.session_start_dttm.tzinfo is not None for r in rows)
    frame = therapy_details_frame(rows)
    dtype = frame.schema["session_start_dttm"]
    assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "therapy_details", run_secondary=False).pandera_passed


# --- provider ---------------------------------------------------------------- #
def test_provider_covers_every_stay() -> None:
    pack = _pack()
    rows = sample_provider(_spine([2, 3, 4, 2], hid="Hp"), pack, np.random.default_rng(0))
    roles = {r.provider_role_category for r in rows}
    assert roles == {"Attending", "Nurse"}
    assert len({r.provider_id for r in rows}) == 2
    for r in rows:
        assert r.start_dttm < r.stop_dttm
    assert gate.validate(provider_frame(rows), "provider", run_secondary=False).pandera_passed


# --- provenance -------------------------------------------------------------- #
def test_provenance_marks_all_tier6_tables() -> None:
    """Tier-6 Concept tables are fitted (when a source block exists) or dashboard-prior."""
    text = pathlib.Path("PROVENANCE.md").read_text(encoding="utf-8")
    expected = {
        "ecmo_mcs": "fitted",
        "invasive_hemodynamics": "dashboard-prior",
        "transfusion": "dashboard-prior",
        "key_icu_orders": "dashboard-prior",
        "therapy_details": "dashboard-prior",
        "provider": "dashboard-prior",
    }
    for table, label in expected.items():
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| `{table}`"))
        assert label in line, f"{table}: expected {label!r} in {line!r}"


def test_all_tier6_datetimes_are_tz_aware() -> None:
    pack = _pack()
    rng = np.random.default_rng(0)
    frames = {
        "ecmo_mcs": (ecmo_mcs_frame(sample_ecmo_mcs(_spine([5] * 4), pack, rng)), "recorded_dttm"),
        "invasive_hemodynamics": (
            invasive_hemodynamics_frame(_hemo_with_catheter([4] * 12, pack, cv=True)),
            "recorded_dttm",
        ),
        "provider": (provider_frame(sample_provider(_spine([3] * 8), pack, rng)), "start_dttm"),
    }
    for _name, (frame, col) in frames.items():
        dtype = frame.schema[col]
        assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
