"""``medication_orders`` generator (derived; R14, KTD-6).

An order is what the administrations happened *under*, so this table is built by
folding the same encounter's administration rows on ``med_order_id`` — the key
CLIF 2.1 defines for exactly this linkage ("foreign key to link this table to
other medication tables"). Deriving it is what makes the join true by
construction: every order has at least one administration, every administration
points at a real order, and the order window brackets its own doses.

Both administration tables feed in, because CLIF says so: ``med_category`` here is
the "combined CDE of ``medication_admin_continuous`` and
``medication_admin_intermittent``". A continuous infusion and a scheduled
antibiotic produce structurally different orders, and the difference is carried
honestly rather than flattened:

* ``order_start_dttm`` / ``order_end_dttm`` bracket the administrations observed
  for that order. A continuous infusion's window spans its start-to-stop rows; an
  intermittent order's spans its first to last dose.
* ``med_dose`` / ``med_dose_unit`` are taken from the **first** administration.
  A continuous order's later rows are rate changes and a zero-dose stop, so
  averaging them would report a dose nobody ordered.
* ``med_frequency`` is ``continuous`` for infusions and the *ordered* dosing
  interval for intermittent meds. It is looked up from the prescribing schedule,
  not measured off the administrations: a stay that ends before the second dose
  is due leaves one row, and reading the frequency off that would report a
  one-time dose where a q8h course was ordered and cut short by discharge.

``prn`` is ``False`` throughout: every administration this generator can see comes
from a scheduled infusion or a scheduled antibiotic course, so claiming any of
them were as-needed would be an invention (R15). ``med_order_status_category`` is
left null — CLIF 2.1 marks its vocabulary "Under-development", so there is no
permissible value to emit.

Reproducible under a fixed ``rng`` (R22); in fact fully determined by its parents.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME
from clifforge.generate.tables.medication_admin_intermittent import ORDERED_INTERVAL_HOURS
from clifforge.reference import loader

__all__ = ["MedicationOrderRow", "medication_orders_frame", "sample_medication_orders"]

_CONTINUOUS = "medication_admin_continuous"
_INTERMITTENT = "medication_admin_intermittent"

#: Orders are placed shortly before the first dose is given.
_ORDER_LEAD_MINUTES = (5.0, 45.0)

_CONTINUOUS_FREQUENCY = "continuous"
_STATUS_NAME = "Completed"
_ROUTE_NAME = "Intravenous"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class MedicationOrderRow:
    """One medication order, folded from the administrations given under it."""

    hospitalization_id: str
    med_order_id: str
    order_start_dttm: datetime
    order_end_dttm: datetime
    ordered_dttm: datetime
    med_category: str
    med_group: str
    med_route_category: str
    med_dose: float
    med_dose_unit: str
    med_frequency: str


def sample_medication_orders(
    parents: dict[str, list[Any]],
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    patient_id: str | None = None,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[MedicationOrderRow]:
    """Fold this encounter's administrations into one row per order (R22)."""
    groups: dict[str, tuple[str, list[Any]]] = {}
    for source in (_CONTINUOUS, _INTERMITTENT):
        for row in parents[source]:
            groups.setdefault(row.med_order_id, (source, []))[1].append(row)

    med_groups = {
        source: loader.crosswalk(source, "med_category", "med_group")
        for source in (_CONTINUOUS, _INTERMITTENT)
    }

    orders: list[MedicationOrderRow] = []
    for order_id, (source, admins) in groups.items():
        admins.sort(key=lambda r: r.admin_dttm)
        first, last = admins[0], admins[-1]
        lead = timedelta(minutes=float(rng.uniform(*_ORDER_LEAD_MINUTES)))
        orders.append(
            MedicationOrderRow(
                hospitalization_id=first.hospitalization_id,
                med_order_id=order_id,
                order_start_dttm=first.admin_dttm,
                order_end_dttm=last.admin_dttm,
                ordered_dttm=first.admin_dttm - lead,
                med_category=first.med_category,
                med_group=med_groups[source][first.med_category],
                med_route_category=first.med_route_category,
                med_dose=first.med_dose,
                med_dose_unit=first.med_dose_unit,
                med_frequency=(
                    _CONTINUOUS_FREQUENCY
                    if source == _CONTINUOUS
                    else f"q{round(ORDERED_INTERVAL_HOURS[first.med_category])}h"
                ),
            )
        )
    orders.sort(key=lambda o: (o.ordered_dttm, o.med_order_id))
    return orders


def medication_orders_frame(rows: list[MedicationOrderRow]) -> pl.DataFrame:
    """Stack medication orders into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "med_order_id": [r.med_order_id for r in rows],
            "order_start_dttm": [r.order_start_dttm for r in rows],
            "order_end_dttm": [r.order_end_dttm for r in rows],
            "ordered_dttm": [r.ordered_dttm for r in rows],
            "med_name": [r.med_category for r in rows],
            "med_category": [r.med_category for r in rows],
            "med_group": [r.med_group for r in rows],
            "med_order_status_name": [_STATUS_NAME] * len(rows),
            # Vocabulary is "Under-development" upstream: no value is emittable.
            "med_order_status_category": [None] * len(rows),
            "med_route_name": [_ROUTE_NAME] * len(rows),
            "med_dose": [r.med_dose for r in rows],
            "med_dose_unit": [r.med_dose_unit for r in rows],
            "med_frequency": [r.med_frequency for r in rows],
            "prn": [False] * len(rows),
        },
        schema={
            "hospitalization_id": pl.String,
            "med_order_id": pl.String,
            "order_start_dttm": UTC_DATETIME,
            "order_end_dttm": UTC_DATETIME,
            "ordered_dttm": UTC_DATETIME,
            "med_name": pl.String,
            "med_category": pl.String,
            "med_group": pl.String,
            "med_order_status_name": pl.String,
            "med_order_status_category": pl.String,
            "med_route_name": pl.String,
            "med_dose": pl.Float64,
            "med_dose_unit": pl.String,
            "med_frequency": pl.String,
            "prn": pl.Boolean,
        },
    )
