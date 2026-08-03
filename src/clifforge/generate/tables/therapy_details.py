"""Tier 6 ``therapy_details`` generator (U20; prior-driven, R14, KTD-6).

Rehab therapy detail rows accompany the mobilization ordered for a subset of ICU
stays. There is no fitted block and CLIF 2.1 leaves the element columns free text
(no mCIDE list), so documented PT/OT session elements are used (R15 —
prior-driven, marked in ``PROVENANCE.md``). The spine supplies only the stay
horizon (KTD-6); reproducible under a fixed ``rng`` (R22).

``session_start_dttm`` is a real tz-aware UTC timestamp. It was previously
emitted as an ISO-8601 *string*, because the CLIF website's prose dictionary
documents this table without a Data Type column and the schema generator defaulted
those to string. The canonical DDL types it ``DATETIME``, so the string was an
artifact of the wrong source, not a property of CLIF.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import (
    ICU_MIN_SUPPORT_LEVEL,
    UTC_DATETIME,
    grid_step_hours,
    pack_table_params,
)
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference.dashboard_priors import absent_table_rates as _DASH_RATES

__all__ = ["TherapyDetailRow", "sample_therapy_details", "therapy_details_frame"]

_REHAB_PROB = _DASH_RATES["key_icu_orders"]
_SESSION_INTERVAL_HOURS = 24.0
#: Documented PT/OT session elements (element_category -> value).
_SESSION_ELEMENTS: tuple[tuple[str, str], ...] = (
    ("mobility", "sat_at_edge_of_bed"),
    ("activity_tolerance", "moderate"),
)

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class TherapyDetailRow:
    """One therapy-session detail element."""

    hospitalization_id: str
    session_start_dttm: datetime
    therapy_element_category: str
    therapy_element_value: str


def sample_therapy_details(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[TherapyDetailRow]:
    """Emit therapy-session detail rows for a subset of ICU stays (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    icu_intervals = [i for i, lvl in enumerate(spine.support_level) if lvl >= ICU_MIN_SUPPORT_LEVEL]
    params = pack_table_params(pack, "therapy_details")
    rehab_prob = float(params.get("stay_prevalence", _REHAB_PROB))
    if not icu_intervals or rng.random() >= rehab_prob:
        return []

    stride = max(1, round(_SESSION_INTERVAL_HOURS / grid_step))
    element_marginal = params.get("therapy_element_category_marginal")
    rows: list[TherapyDetailRow] = []
    for idx in range(icu_intervals[0], icu_intervals[-1] + 1, stride):
        session_start = admit_dttm + timedelta(hours=idx * grid_step)
        if isinstance(element_marginal, dict) and element_marginal:
            cat = categorical(element_marginal, rng)
            val = str(params.get("therapy_element_value_by_category", {}).get(cat, cat))
            rows.append(TherapyDetailRow(hid, session_start, cat, val))
            # Keep roughly two elements per session when fitted.
            cat2 = categorical(element_marginal, rng)
            val2 = str(params.get("therapy_element_value_by_category", {}).get(cat2, cat2))
            rows.append(TherapyDetailRow(hid, session_start, cat2, val2))
        else:
            for category, value in _SESSION_ELEMENTS:
                rows.append(TherapyDetailRow(hid, session_start, category, value))
    return rows


def therapy_details_frame(rows: list[TherapyDetailRow]) -> pl.DataFrame:
    """Stack therapy detail rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "session_start_dttm": [r.session_start_dttm for r in rows],
            "therapy_element_name": [r.therapy_element_category for r in rows],
            "therapy_element_category": [r.therapy_element_category for r in rows],
            "therapy_element_value": [r.therapy_element_value for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "session_start_dttm": UTC_DATETIME,
            "therapy_element_name": pl.String,
            "therapy_element_category": pl.String,
            "therapy_element_value": pl.String,
        },
    )
