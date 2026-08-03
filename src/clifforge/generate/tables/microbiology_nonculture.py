"""``microbiology_nonculture`` generator (prior-driven; R5, R14, KTD-6).

Non-culture microbiology is the rapid molecular panel: C. difficile PCR on stool,
respiratory PCR for SARS-CoV-2 or RSV. These are sparse, near-admission tests, so
a documented per-stay rate draws a small Poisson count and each test picks a
target from the vendored mCIDE ``organism_category`` list (all three 2.1 values)
paired with the specimen it is actually run on.

``organism_group`` is read from the mCIDE file's own ``organism_group`` column
rather than restated here — the consortium publishes the roll-up alongside the
category, and it is not the identity mapping one might assume (``sars_cov2``
rolls up to ``viral_other``).

Every category emitted is an exact mCIDE member (R5): ``fluid_category``,
``method_category`` (whose 2.1 vocabulary is the single value ``pcr``),
``organism_category``, and ``result_category``.

``reference_low`` / ``reference_high`` are null and ``result_units`` is null
because these panels are **qualitative** — detected / not detected. A reference
range for a yes/no answer would be an invention (R15), so the honest encoding is
the absence of one. This is also why the table is emitted with real ``DATETIME``
columns: CLIF's canonical DDL types them so, even though the website's prose
dictionary leaves this table entirely untyped.

The spine supplies only the stay horizon (KTD-6); reproducible under a fixed
``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.loinc import micro_loinc_code
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference import loader
from clifforge.reference.dashboard_priors import absent_table_rates as _DASH_RATES

__all__ = [
    "NonCultureEvent",
    "microbiology_nonculture_frame",
    "sample_microbiology_nonculture",
]

_TABLE = "microbiology_nonculture"

#: Expected panels per stay (dashboard-prior). Admission-workup tests, per stay.
_PANELS_PER_STAY = _DASH_RATES["microbiology_nonculture"]

#: The 2.1 method vocabulary for this table has exactly one member.
_METHOD = "pcr"

#: mCIDE organism_category -> (mCIDE fluid_category, order name as charted).
_TARGET_SPECIMEN: dict[str, tuple[str, str]] = {
    "sars_cov2": ("nasopharynx_upperairway", "SARS-COV-2 (COVID-19) QUALITATIVE PCR (NAAT)"),
    "respiratory_syncytial_virus": ("nasopharynx_upperairway", "INFLUENZA & RSV PCR (NAAT)"),
    "clostridium_difficile": ("feces_stool", "C. DIFFICILE PCR, SCREENING"),
}
_TARGET_MARGINAL = {
    "sars_cov2": 0.5,
    "clostridium_difficile": 0.3,
    "respiratory_syncytial_virus": 0.2,
}

#: Molecular panels are mostly negative; a small share are technically invalid.
_RESULT_MARGINAL = {"not_detected": 0.85, "detected": 0.12, "indeterminate": 0.03}

#: Reported result strings paired with each mCIDE result_category.
_RESULT_NAME = {
    "not_detected": "Negative",
    "detected": "Positive",
    "indeterminate": "Invalid",
}

#: PCR turnaround is hours, not the days a culture takes.
_TURNAROUND_HOURS = (1.0, 12.0)
_COLLECT_DELAY_MINUTES = (5.0, 60.0)
#: Admission workup: ordered within the first day, or on arrival for short stays.
_ORDER_WINDOW_HOURS = 24.0

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class NonCultureEvent:
    """One non-culture (molecular) microbiology test."""

    patient_id: str
    hospitalization_id: str
    order_dttm: datetime
    collect_dttm: datetime
    result_dttm: datetime
    fluid_category: str
    micro_order_name: str
    organism_category: str
    organism_group: str
    result_category: str


def sample_microbiology_nonculture(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    patient_id: str | None = None,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[NonCultureEvent]:
    """Emit one hospitalization's sparse non-culture panels (R5, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    pid = patient_id if patient_id is not None else hid
    los_hours = spine.n_intervals * grid_step_hours(pack)
    if los_hours <= 0:
        return []

    organism_group = loader.crosswalk(_TABLE, "organism_category", "organism_group")

    events: list[NonCultureEvent] = []
    for _ in range(int(rng.poisson(_PANELS_PER_STAY))):
        target = categorical(_TARGET_MARGINAL, rng)
        fluid, order_name = _TARGET_SPECIMEN[target]
        order = admit_dttm + timedelta(
            hours=float(rng.random()) * min(los_hours, _ORDER_WINDOW_HOURS)
        )
        collect = order + timedelta(minutes=float(rng.uniform(*_COLLECT_DELAY_MINUTES)))
        result = collect + timedelta(hours=float(rng.uniform(*_TURNAROUND_HOURS)))
        events.append(
            NonCultureEvent(
                patient_id=pid,
                hospitalization_id=hid,
                order_dttm=order,
                collect_dttm=collect,
                result_dttm=result,
                fluid_category=fluid,
                micro_order_name=order_name,
                organism_category=target,
                organism_group=organism_group[target],
                result_category=categorical(_RESULT_MARGINAL, rng),
            )
        )
    events.sort(key=lambda e: e.order_dttm)
    return events


def microbiology_nonculture_frame(events: list[NonCultureEvent]) -> pl.DataFrame:
    """Stack non-culture events into one conformant frame."""
    n = len(events)
    return pl.DataFrame(
        {
            "patient_id": [e.patient_id for e in events],
            "hospitalization_id": [e.hospitalization_id for e in events],
            "result_dttm": [e.result_dttm for e in events],
            "collect_dttm": [e.collect_dttm for e in events],
            "order_dttm": [e.order_dttm for e in events],
            "fluid_name": [e.fluid_category for e in events],
            "fluid_category": [e.fluid_category for e in events],
            "method_name": [_METHOD] * n,
            "method_category": [_METHOD] * n,
            "micro_order_name": [e.micro_order_name for e in events],
            "organism_category": [e.organism_category for e in events],
            "organism_group": [e.organism_group for e in events],
            "result_name": [_RESULT_NAME[e.result_category] for e in events],
            "result_category": [e.result_category for e in events],
            # Qualitative assay: no reference interval and no unit to report.
            "reference_low": [None] * n,
            "reference_high": [None] * n,
            "result_units": [None] * n,
            "lab_loinc_code": [micro_loinc_code(e.organism_category) for e in events],
        },
        schema={
            "patient_id": pl.String,
            "hospitalization_id": pl.String,
            "result_dttm": UTC_DATETIME,
            "collect_dttm": UTC_DATETIME,
            "order_dttm": UTC_DATETIME,
            "fluid_name": pl.String,
            "fluid_category": pl.String,
            "method_name": pl.String,
            "method_category": pl.String,
            "micro_order_name": pl.String,
            "organism_category": pl.String,
            "organism_group": pl.String,
            "result_name": pl.String,
            "result_category": pl.String,
            "reference_low": pl.Float64,
            "reference_high": pl.Float64,
            "result_units": pl.String,
            "lab_loinc_code": pl.String,
        },
    )
