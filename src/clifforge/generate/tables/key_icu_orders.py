"""Tier 6 ``key_icu_orders`` generator (U20; prior-driven, R5, R14, KTD-6).

The mCIDE ``order_category`` for this table is the PT/OT rehab set
(``PT_evaluation``/``PT_treat``/``OT_evaluation``/``OT_treat``). Early mobilization
is ordered for a subset of ICU stays, so a documented fraction of stays with ICU
time receive an evaluation followed by treatment orders spread across the stay
(R15 — prior-driven, marked in ``PROVENANCE.md``). ``order_category`` values are
exact mCIDE members (R5). The spine supplies only the stay horizon (KTD-6);
reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import ICU_MIN_SUPPORT_LEVEL, UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame
from clifforge.reference.dashboard_priors import absent_table_rates as _DASH_RATES

__all__ = ["OrderRow", "key_icu_orders_frame", "sample_key_icu_orders"]

#: Fraction of ICU stays that get a rehab consult (dashboard-prior).
_REHAB_PROB = _DASH_RATES["key_icu_orders"]
_TREAT_INTERVAL_HOURS = 24.0  # rehab treatments are ~daily once ordered
_STATUS = "Completed"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class OrderRow:
    """One key ICU (rehab) order event."""

    hospitalization_id: str
    order_dttm: datetime
    order_category: str
    order_status_name: str


def sample_key_icu_orders(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[OrderRow]:
    """Emit PT/OT rehab orders for a subset of ICU stays (R5, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    icu_intervals = [i for i, lvl in enumerate(spine.support_level) if lvl >= ICU_MIN_SUPPORT_LEVEL]
    if not icu_intervals or rng.random() >= _REHAB_PROB:
        return []

    start_idx, end_idx = icu_intervals[0], icu_intervals[-1]

    def at(idx: int) -> datetime:
        return admit_dttm + timedelta(hours=idx * grid_step)

    rows = [
        OrderRow(hid, at(start_idx), "PT_evaluation", _STATUS),
        OrderRow(hid, at(start_idx), "OT_evaluation", _STATUS),
    ]
    # Daily PT/OT treatments after the evaluation, through the ICU stay.
    stride = max(1, round(_TREAT_INTERVAL_HOURS / grid_step))
    for idx in range(start_idx + stride, end_idx + 1, stride):
        rows.append(OrderRow(hid, at(idx), "PT_treat", _STATUS))
        rows.append(OrderRow(hid, at(idx), "OT_treat", _STATUS))
    rows.sort(key=lambda r: (r.order_dttm, r.order_category))
    return rows


def key_icu_orders_frame(rows: list[OrderRow]) -> pl.DataFrame:
    """Stack orders into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "order_dttm": [r.order_dttm for r in rows],
            "order_name": [r.order_category for r in rows],
            "order_category": [r.order_category for r in rows],
            "order_status_name": [r.order_status_name for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "order_dttm": UTC_DATETIME,
            "order_name": pl.String,
            "order_category": pl.String,
            "order_status_name": pl.String,
        },
    )
