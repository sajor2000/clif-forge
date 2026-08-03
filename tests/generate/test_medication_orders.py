"""Unit tests for the derived medication_orders generator (R22).

The point of this table is the join: an order must correspond exactly to the
administrations given under it. Most of what is checked here is that
correspondence, because a medication_orders table that does not join cleanly to
the admin tables is worse than no table at all — it looks usable and is not.
"""

from __future__ import annotations

import numpy as np

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import medication_orders as mo
from clifforge.generate.tables.medication_admin_continuous import (
    sample_medication_admin_continuous,
)
from clifforge.generate.tables.medication_admin_intermittent import (
    ORDERED_INTERVAL_HOURS,
    sample_medication_admin_intermittent,
)
from clifforge.generate.tables.medication_orders import (
    medication_orders_frame,
    sample_medication_orders,
)

_CONTINUOUS = "medication_admin_continuous"
_INTERMITTENT = "medication_admin_intermittent"


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(n: int, hid: str = "H0", level: int = 4) -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[level] * n,
        resp_flag=[True] * n,
        cv_flag=[True] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def _cohort(n_stays: int, seed: int = 0, n_int: int = 48) -> tuple[list, list, list]:
    """Return (continuous admins, intermittent admins, derived orders)."""
    pack, rng = _pack(), np.random.default_rng(seed)
    cont, inter, orders = [], [], []
    for i in range(n_stays):
        spine = _spine(n_int, hid=f"H{i}")
        c = sample_medication_admin_continuous(spine, pack, rng, hospitalization_id=f"H{i}")
        m = sample_medication_admin_intermittent(spine, pack, rng, hospitalization_id=f"H{i}")
        cont += c
        inter += m
        orders += sample_medication_orders(
            {_CONTINUOUS: c, _INTERMITTENT: m}, pack, rng, hospitalization_id=f"H{i}"
        )
    return cont, inter, orders


def test_is_deterministic() -> None:
    assert _cohort(20, seed=4)[2] == _cohort(20, seed=4)[2]


def test_orders_and_administrations_are_in_exact_correspondence() -> None:
    cont, inter, orders = _cohort(40, seed=1)
    administered = {r.med_order_id for r in cont} | {r.med_order_id for r in inter}
    assert administered, "test cohort produced no administrations"
    assert {o.med_order_id for o in orders} == administered


def test_one_row_per_order() -> None:
    _cont, _inter, orders = _cohort(40, seed=2)
    ids = [o.med_order_id for o in orders]
    assert len(ids) == len(set(ids))


def test_order_window_brackets_its_own_administrations() -> None:
    cont, inter, orders = _cohort(40, seed=3)
    by_id: dict[str, list] = {}
    for row in [*cont, *inter]:
        by_id.setdefault(row.med_order_id, []).append(row.admin_dttm)
    for order in orders:
        times = by_id[order.med_order_id]
        assert order.order_start_dttm == min(times)
        assert order.order_end_dttm == max(times)


def test_the_order_precedes_the_first_dose() -> None:
    """You cannot administer a drug before it was ordered."""
    _cont, _inter, orders = _cohort(40, seed=5)
    assert orders
    for order in orders:
        assert order.ordered_dttm < order.order_start_dttm


def test_dose_is_the_first_administration_not_an_average() -> None:
    """A continuous order's later rows are rate changes and a zero-dose stop."""
    cont, inter, orders = _cohort(40, seed=6)
    first_dose = {}
    for row in sorted([*cont, *inter], key=lambda r: r.admin_dttm, reverse=True):
        first_dose[row.med_order_id] = row.med_dose
    for order in orders:
        assert order.med_dose == first_dose[order.med_order_id]


def test_infusions_are_continuous_and_antibiotics_carry_their_ordered_interval() -> None:
    """A course cut short by discharge still reports the frequency that was ordered."""
    cont, inter, orders = _cohort(60, seed=8, n_int=6)  # short stays truncate courses
    continuous_ids = {r.med_order_id for r in cont}
    intermittent_med = {r.med_order_id: r.med_category for r in inter}
    assert intermittent_med, "test cohort produced no intermittent administrations"

    single_dose_orders = 0
    doses: dict[str, int] = {}
    for row in inter:
        doses[row.med_order_id] = doses.get(row.med_order_id, 0) + 1

    for order in orders:
        if order.med_order_id in continuous_ids:
            assert order.med_frequency == mo._CONTINUOUS_FREQUENCY
        else:
            med = intermittent_med[order.med_order_id]
            assert order.med_frequency == f"q{round(ORDERED_INTERVAL_HOURS[med])}h"
            single_dose_orders += doses[order.med_order_id] == 1
    assert single_dose_orders, "expected some truncated single-dose courses at this LOS"


def test_med_group_is_the_vendored_rollup() -> None:
    from clifforge.reference import loader

    _cont, _inter, orders = _cohort(40, seed=10)
    groups = {
        source: loader.crosswalk(source, "med_category", "med_group")
        for source in (_CONTINUOUS, _INTERMITTENT)
    }
    for order in orders:
        expected = {g[order.med_category] for g in groups.values() if order.med_category in g}
        assert order.med_group in expected


def test_prn_is_false_because_nothing_generated_is_as_needed() -> None:
    _cont, _inter, orders = _cohort(30, seed=12)
    frame = medication_orders_frame(orders)
    assert not frame["prn"].any()


def test_order_status_category_is_null_pending_upstream_vocabulary() -> None:
    _cont, _inter, orders = _cohort(30, seed=14)
    frame = medication_orders_frame(orders)
    assert frame["med_order_status_category"].is_null().all()


def test_frame_passes_the_conformance_gate() -> None:
    _cont, _inter, orders = _cohort(30, seed=16)
    gate.validate(medication_orders_frame(orders), "medication_orders")


def test_empty_input_yields_an_empty_typed_frame() -> None:
    frame = medication_orders_frame([])
    assert frame.height == 0
    assert "med_order_id" in frame.columns


def test_module_exports() -> None:
    assert set(mo.__all__) == {
        "MedicationOrderRow",
        "medication_orders_frame",
        "sample_medication_orders",
    }
