"""Tier 5 ``medication_admin_intermittent`` generator (U16; R11, KTD-6).

Intermittent meds are **discrete** administrations (scheduled doses / boluses),
strictly disjoint from the rate-encoded infusions in
``medication_admin_continuous`` (R11): every row here is a single dose at one
``admin_dttm`` with ``mar_action_category = "given"`` — never a rate, never an
infusion start/stop.

No fitted block exists, so a documented ICU antibiotic schedule drives a subset
of stays (an infection prevalence constant, un-fitted like the adt constants):
flagged stays receive scheduled vancomycin/cefepime doses across their length of
stay. The meds chosen are not vasopressors or sedatives, keeping the
continuous/intermittent split unambiguous. The spine supplies only the stay
horizon (KTD-6). Output is reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame
from clifforge.reference import loader

__all__ = [
    "ORDERED_INTERVAL_HOURS",
    "MedIntermittentRow",
    "medication_admin_intermittent_frame",
    "sample_medication_admin_intermittent",
]

_TABLE = "medication_admin_intermittent"
#: Fraction of stays on antibiotics (documented prevalence, un-fitted).
_ABX_PROB = 0.5
#: (med_category, dosing interval hours, dose mg) — discrete scheduled antibiotics.
_ANTIBIOTIC_SCHEDULE: tuple[tuple[str, float, float], ...] = (
    ("vancomycin", 12.0, 1000.0),
    ("cefepime", 8.0, 2000.0),
)

#: med_category -> the *ordered* dosing interval in hours.
#:
#: Exported for ``medication_orders``, which needs the frequency that was
#: prescribed. That is not always recoverable from the administrations: a stay
#: that ends before the second dose is due leaves a single row, and reading the
#: frequency off it would report a one-time dose where a q8h course was ordered
#: and then truncated by discharge.
ORDERED_INTERVAL_HOURS: dict[str, float] = {
    med: interval for med, interval, _dose in _ANTIBIOTIC_SCHEDULE
}
_DOSE_UNIT = "mg"
_ROUTE = "iv"
_ACTION = "given"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class MedIntermittentRow:
    """One discrete medication administration."""

    hospitalization_id: str
    med_order_id: str
    admin_dttm: datetime
    med_category: str
    med_route_category: str
    med_dose: float
    med_dose_unit: str
    mar_action_category: str


def sample_medication_admin_intermittent(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[MedIntermittentRow]:
    """Emit one hospitalization's discrete med administrations (R11, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    los_hours = spine.n_intervals * grid_step_hours(pack)

    if rng.random() >= _ABX_PROB:
        return []  # not on antibiotics this stay

    rows: list[MedIntermittentRow] = []
    for med, interval_hours, dose in _ANTIBIOTIC_SCHEDULE:
        order_id = f"{hid}-{med}"
        elapsed = 0.0
        while elapsed < los_hours:
            rows.append(
                MedIntermittentRow(
                    hospitalization_id=hid,
                    med_order_id=order_id,
                    admin_dttm=admit_dttm + timedelta(hours=elapsed),
                    med_category=med,
                    med_route_category=_ROUTE,
                    med_dose=dose,
                    med_dose_unit=_DOSE_UNIT,
                    mar_action_category=_ACTION,
                )
            )
            elapsed += interval_hours
    rows.sort(key=lambda r: (r.admin_dttm, r.med_category))
    return rows


def medication_admin_intermittent_frame(rows: list[MedIntermittentRow]) -> pl.DataFrame:
    """Stack discrete administrations into one conformant frame.

    ``med_group`` and ``mar_action_group`` are read from the same vendored mCIDE
    files that define the categories, so the roll-ups stay exact. Note the groups
    differ from the continuous table's: an intermittent antibiotic rolls up to
    ``CMS_sepsis_qualifying_antibiotics``, which has no continuous counterpart.
    """
    med_group = loader.crosswalk(_TABLE, "med_category", "med_group")
    action_group = loader.crosswalk(_TABLE, "mar_action_category", "mar_action_group")
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "med_order_id": [r.med_order_id for r in rows],
            "admin_dttm": [r.admin_dttm for r in rows],
            "med_name": [r.med_category for r in rows],
            "med_category": [r.med_category for r in rows],
            "med_group": [med_group[r.med_category] for r in rows],
            "med_route_name": [r.med_route_category for r in rows],
            "med_route_category": [r.med_route_category for r in rows],
            "med_dose": [r.med_dose for r in rows],
            "med_dose_unit": [r.med_dose_unit for r in rows],
            "mar_action_name": [r.mar_action_category for r in rows],
            "mar_action_category": [r.mar_action_category for r in rows],
            "mar_action_group": [action_group[r.mar_action_category] for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "med_order_id": pl.String,
            "admin_dttm": UTC_DATETIME,
            "med_name": pl.String,
            "med_category": pl.String,
            "med_group": pl.String,
            "med_route_name": pl.String,
            "med_route_category": pl.String,
            "med_dose": pl.Float64,
            "med_dose_unit": pl.String,
            "mar_action_name": pl.String,
            "mar_action_category": pl.String,
            "mar_action_group": pl.String,
        },
    )
