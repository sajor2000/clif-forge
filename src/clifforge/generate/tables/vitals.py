"""Tier 3 ``vitals`` generator (U10; R9, R12, KTD-6).

Vitals are physiology, so each vital is a per-state **AR(1)** walk driven by the
latent spine: at every grid interval the mean/phi/sigma are chosen by that
interval's ``support_level`` (the acuity state the pack was fit against), and the
process is advanced on the **same fixed grid the pack was fitted on**
(``spine.params.state_model.grid_step_hours``) — the phi/sigma parameters are only
valid at that sampling interval, so fitting and generating on a common grid is
what keeps the AR(1) autocorrelation faithful.

The continuous walk runs on every interval; observed rows are then **thinned** to
a realistic irregular cadence that is denser inside ICU intervals (acuity
``>= ICU_MIN_SUPPORT_LEVEL``, the same threshold the adt generator uses) and
sparser on the ward, with sub-interval jitter so timestamps are not pinned to
grid boundaries. Every emitted value is clamped into the consortium outlier
bounds (R9, ``reference.bounds``).

**Coupling (R12/KTD-6).** sbp/map track acuity purely through per-state means:
the high-acuity states the cardiovascular-failure flag marks (vasopressor and
above) carry the lower fitted blood-pressure means, so blood pressure falls when
the spine's cv-failure flag is set — an emergent consequence of state selection,
not an invented offset (R15). The spine is the only cross-table channel; this
generator never reads another table's output.

Un-fitted columns (``meas_site_name``) are omitted rather than fabricated (R15;
the schema is permissive). Output is reproducible byte-for-byte under a fixed
``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import ICU_MIN_SUPPORT_LEVEL, UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame
from clifforge.reference import bounds

__all__ = ["VITALS", "VitalObservation", "sample_vitals", "vitals_frame"]

#: The fitted vitals, in emission order. Only these have ``<vital>_ar1_by_state``
#: blocks in the pack; height_cm/weight_kg are mCIDE members but were not fitted
#: by U5, so they are omitted rather than fabricated (R15).
VITALS = ("heart_rate", "sbp", "dbp", "map", "respiratory_rate", "spo2", "temp_c")


#: Per-interval probability that a vital is observed. Un-fitted cadence heuristics
#: (like the adt hospital constants): dense but not certain in the ICU, sparse on
#: the ward. Reproducible because the draw is seeded.
_EMIT_PROB_ICU = 0.85
_EMIT_PROB_WARD = 0.30

#: Measurement site for the vitals where the site changes how the number reads.
#: Free text: CLIF gives ``meas_site_name`` no mCIDE. Vitals absent from this map
#: have no meaningful site and are emitted null.
_MEAS_SITE: dict[str, str] = {
    "temp_c": "core",
    "sbp": "arterial line",
    "dbp": "arterial line",
    "map": "arterial line",
    "spo2": "finger",
}

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class VitalObservation:
    """One observed vital reading in the long ``vitals`` table."""

    hospitalization_id: str
    recorded_dttm: datetime
    vital_name: str
    vital_category: str
    vital_value: float


def _vitals_params(pack: ParamPack) -> dict[str, Any]:
    block = pack.tables.get("vitals")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'vitals' block to sample from")
    params: dict[str, Any] = block["params"]
    return params


def _state_params(by_state: dict[str, dict[str, float]], level: int) -> dict[str, float]:
    """AR(1) params for ``level``; fall back to the nearest fitted state.

    Suppression (fit min_n gate) can leave a (vital, state) cell unfit, so a state
    absent from the block resolves to the nearest available state by numeric
    distance (ties resolve to the lower state) — deterministic, never invented.
    """
    key = str(level)
    if key in by_state:
        return by_state[key]
    available = sorted(int(k) for k in by_state)
    nearest = min(available, key=lambda s: (abs(s - level), s))
    return by_state[str(nearest)]


def sample_vitals(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[VitalObservation]:
    """Emit one hospitalization's observed vitals as a long list of rows (R9, R22).

    For each fitted vital an AR(1) walk advances over every grid interval using
    per-``support_level`` params; each interval is emitted with an ICU/ward-
    dependent probability at a jittered sub-interval timestamp, clamped into the
    outlier bounds. ``hospitalization_id`` defaults to the spine's own id.
    """
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    params = _vitals_params(pack)

    observations: list[VitalObservation] = []
    for vital in VITALS:
        key = f"{vital}_ar1_by_state"
        by_state = params.get(key)
        if not by_state:
            continue
        lower, upper = bounds("vitals", vital)

        value: float | None = None
        for interval_idx, level in enumerate(spine.support_level):
            state = _state_params(by_state, level)
            mean, phi, sigma = state["mean"], state["phi"], state["sigma"]
            if value is None:
                value = mean  # warm-start at the state mean
            else:
                value = mean + phi * (value - mean) + sigma * float(rng.standard_normal())
            value = min(max(value, lower), upper)  # clamp into outlier bounds (R9)

            is_icu = level >= ICU_MIN_SUPPORT_LEVEL
            emit_prob = _EMIT_PROB_ICU if is_icu else _EMIT_PROB_WARD
            if rng.random() < emit_prob:
                jitter = rng.random() * grid_step  # sub-interval, stays within LOS
                recorded = admit_dttm + timedelta(hours=interval_idx * grid_step + jitter)
                observations.append(
                    VitalObservation(
                        hospitalization_id=hid,
                        recorded_dttm=recorded,
                        vital_name=vital,
                        vital_category=vital,
                        vital_value=round(value, 4),
                    )
                )

    observations.sort(key=lambda o: (o.recorded_dttm, o.vital_category))
    return observations


def vitals_frame(observations: list[VitalObservation]) -> pl.DataFrame:
    """Stack observed vitals into one conformant long ``vitals`` frame.

    ``meas_site_name`` is populated only where a site is meaningful. CLIF's own
    mCIDE description of ``temp_c`` says the site "should be indicated in
    meas_site_name", because an oral and a core temperature are not
    interchangeable readings; blood pressure carries the same distinction between
    an arterial line and a cuff. A heart rate has no comparable site, so it is
    left null rather than filled with a placeholder.
    """
    return pl.DataFrame(
        {
            "hospitalization_id": [o.hospitalization_id for o in observations],
            "recorded_dttm": [o.recorded_dttm for o in observations],
            "vital_name": [o.vital_name for o in observations],
            "vital_category": [o.vital_category for o in observations],
            "vital_value": [o.vital_value for o in observations],
            "meas_site_name": [_MEAS_SITE.get(o.vital_category) for o in observations],
        },
        schema={
            "hospitalization_id": pl.String,
            "recorded_dttm": UTC_DATETIME,
            "vital_name": pl.String,
            "vital_category": pl.String,
            "vital_value": pl.Float64,
            "meas_site_name": pl.String,
        },
    )
