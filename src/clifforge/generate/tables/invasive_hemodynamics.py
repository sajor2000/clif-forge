"""Tier 6 ``invasive_hemodynamics`` generator (U20; prior-driven, R5, R14, KTD-6).

Pulmonary-artery-catheter measurements are placed in patients with
cardiovascular failure, so measurement events are emitted at a charting cadence
during the spine's ``cv_flag`` windows. There is no fitted block, so cadence uses
a documented constant (R15 — prior-driven, marked in ``PROVENANCE.md``).

``measure_value`` carries the actual measurement. (It was previously omitted on
the grounds that the dictionary defined no value column; that was an artifact of
sourcing the dictionary from the CLIF website's prose, and the canonical DDL
types it ``DOUBLE``. A hemodynamics table that records *that* a CVP was taken but
never *what it was* is not usable for anything.)

**Values are drawn per shock phenotype, because the numbers invert between them.**
Every row here is charted during a ``cv_flag`` window, so the patient is in
circulatory failure — but cardiogenic and distributive shock look opposite on a
PA catheter. A failing heart backs up: high filling pressures (CVP, PCWP, PA
pressures) with a low cardiac output. Vasodilatory shock does the reverse: low
filling pressures with a high output. Averaging the two would produce a cohort of
uniformly mid-range numbers in which neither phenotype exists, and any
hemodynamic classification done on the output would find nothing. The phenotype
is therefore drawn **once per encounter** and every measurement in that stay is
consistent with it.

Ranges are documented physiology (R15 — prior-driven, marked in
``PROVENANCE.md``); CLIF ships no outlier-threshold file for this table.
``measure_category`` values are exact mCIDE members (R5). The spine supplies only
the cv-failure signal (KTD-6); reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference.dashboard_priors import absent_table_rates as _DASH_RATES

__all__ = [
    "HemodynamicRow",
    "invasive_hemodynamics_frame",
    "sample_invasive_hemodynamics",
]

_MEASURE_INTERVAL_HOURS = 6.0  # hemodynamics charted a few times a day
#: Unconditional stay-level PA-catheter prevalence (clif-icu.com dashboard).
_STAY_PREVALENCE = _DASH_RATES["invasive_hemodynamics"]
#: Approximate ICU cv-failure stay share under recalibrate; converts dashboard
#: stay rate into P(PA catheter | any cv_flag) — only shocked patients get lines.
_CV_STAY_SHARE = 0.27
_CV_CONDITIONAL_PREVALENCE = min(1.0, _STAY_PREVALENCE / _CV_STAY_SHARE)
#: A standard PA-catheter measure set, weighted toward the routinely-charted ones.
_MEASURE_MARGINAL = {
    "cvp": 0.3,
    "pa_systolic": 0.15,
    "pa_diastolic": 0.15,
    "pa_mean": 0.15,
    "pcwp": 0.15,
    "cardiac_output_thermodilution": 0.1,
}

#: Documented measurement ranges by shock phenotype. Pressures are mmHg; cardiac
#: output is L/min. Cardiogenic shock backs pressure up behind a failing pump;
#: distributive (septic) shock runs empty and fast.
_CARDIOGENIC = "cardiogenic"
_DISTRIBUTIVE = "distributive"
_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    _CARDIOGENIC: {
        "cvp": (12.0, 20.0),
        "pa_systolic": (35.0, 55.0),
        "pa_diastolic": (18.0, 28.0),
        "pa_mean": (25.0, 38.0),
        "pcwp": (18.0, 28.0),
        "cardiac_output_thermodilution": (2.5, 4.0),
    },
    _DISTRIBUTIVE: {
        "cvp": (2.0, 8.0),
        "pa_systolic": (18.0, 30.0),
        "pa_diastolic": (6.0, 13.0),
        "pa_mean": (10.0, 19.0),
        "pcwp": (6.0, 12.0),
        "cardiac_output_thermodilution": (6.0, 10.0),
    },
}
#: Septic (distributive) shock is the more common ICU presentation.
_PHENOTYPE_MARGINAL = {_DISTRIBUTIVE: 0.65, _CARDIOGENIC: 0.35}

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class HemodynamicRow:
    """One invasive hemodynamic measurement."""

    hospitalization_id: str
    recorded_dttm: datetime
    measure_category: str
    measure_value: float


def _measure_intervals(cv_flag: list[bool], grid_step: float) -> list[int]:
    """cv-failure interval indices at which to chart a hemodynamic measure."""
    intervals: list[int] = []
    last: int | None = None
    for idx, cv in enumerate(cv_flag):
        if not cv:
            continue
        if last is None or (idx - last) * grid_step >= _MEASURE_INTERVAL_HOURS:
            intervals.append(idx)
            last = idx
    return intervals


def sample_invasive_hemodynamics(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[HemodynamicRow]:
    """Emit PA-catheter measurements during cv-failure windows (R5, R22).

    Clinical prior: invasive hemo is for shock — gate on ``cv_flag`` with
    P(catheter|cv) from the dashboard stay rate, never on non-shock stays.
    """
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    if not any(spine.cv_flag):
        return []
    if rng.random() >= _CV_CONDITIONAL_PREVALENCE:
        return []
    grid_step = grid_step_hours(pack)
    intervals = _measure_intervals(spine.cv_flag, grid_step)
    if not intervals:
        return []

    # One shock phenotype per stay: a patient does not alternate between a failing
    # pump and a vasodilated circulation from one measurement to the next.
    phenotype = categorical(_PHENOTYPE_MARGINAL, rng)
    ranges = _RANGES[phenotype]

    rows: list[HemodynamicRow] = []
    for idx in intervals:
        measure = categorical(_MEASURE_MARGINAL, rng)
        lo, hi = ranges[measure]
        rows.append(
            HemodynamicRow(
                hospitalization_id=hid,
                recorded_dttm=admit_dttm + timedelta(hours=idx * grid_step),
                measure_category=measure,
                measure_value=round(float(rng.uniform(lo, hi)), 1),
            )
        )
    return rows


def invasive_hemodynamics_frame(rows: list[HemodynamicRow]) -> pl.DataFrame:
    """Stack hemodynamic measurement events into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "measure_name": [r.measure_category for r in rows],
            "measure_category": [r.measure_category for r in rows],
            "measure_value": [r.measure_value for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "recorded_dttm": UTC_DATETIME,
            "measure_name": pl.String,
            "measure_category": pl.String,
            "measure_value": pl.Float64,
        },
    )
