"""Tier 4 ``medication_admin_continuous`` generator (U13; R11, AE3, KTD-6).

The pack fits only per-med **infusion hazards** (``stop_hazard`` and mean run
length), not start rates or doses, so continuous infusions are driven by the
latent spine with those hazards layered on: a **vasopressor** (norepinephrine)
runs during the spine's cardiovascular-failure windows, and a **sedative**
(propofol) runs during invasive ventilation (``support_level >= 3``). Both couple
to the same spine that drives U10 hypotension, so norepinephrine co-occurs with
low blood pressure without either table reading the other (KTD-6).

**R11 / AE3 — rate-encoded, stop = new zero-dose row, no boluses.** ``med_dose``
is an infusion *rate* (``med_dose_unit`` a rate unit); a stop is emitted as a
**new** row with ``med_dose = 0`` and ``mar_action_category = "stop"`` — prior
rows are never mutated and there are no bolus rows. Within an active window the
pack ``stop_hazard`` can end an infusion, which restarts if the coupling still
holds, producing realistic on/off cycling.

Doses are documented clinical rate ranges (un-fitted, like the adt constants),
not invented distributions (R15). ``med_order_id`` is synthesized per infusion.
Output is reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import (
    ICU_MIN_SUPPORT_LEVEL,
    IMV_MIN_SUPPORT_LEVEL,
    UTC_DATETIME,
    grid_step_hours,
)
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference import loader

__all__ = [
    "MedAdminRow",
    "medication_admin_continuous_frame",
    "sample_medication_admin_continuous",
]

_TABLE = "medication_admin_continuous"
_DOSE_UNIT = "mcg/kg/min"
_ROUTE = "iv"
#: (med_category, documented rate range) for the two spine-coupled infusions.
_VASOPRESSOR = "norepinephrine"
_SEDATIVE = "propofol"
_DOSE_RANGE: dict[str, tuple[float, float]] = {
    _VASOPRESSOR: (0.02, 0.5),
    _SEDATIVE: (5.0, 50.0),
}
_DEFAULT_STOP_HAZARD = 0.2

# --- fitted path (used when the pack carries a med_category_marginal) --------- #
#: Vasopressors/inotropes — a sampled infusion of one of these is *placed* in a
#: cardiovascular-failure window, preserving the R12 coupling while the med itself
#: is drawn from the fitted marginal (so the med_category distribution matches real).
_VASOPRESSORS = frozenset(
    {
        "norepinephrine",
        "phenylephrine",
        "vasopressin",
        "epinephrine",
        "dopamine",
        "dobutamine",
        "angiotensin",
        "milrinone",
        "isoproterenol",
    }
)
#: Continuous sedatives/analgesics — placed in invasive-ventilation windows.
_SEDATIVES = frozenset(
    {
        "propofol",
        "fentanyl",
        "dexmedetomidine",
        "midazolam",
        "ketamine",
        "remifentanil",
        "morphine",
        "hydromorphone",
        "lorazepam",
        "pentobarbital",
    }
)
#: Documented infusion volume per ICU interval, and mean dose-change events per
#: infusion (calibrated so mar_action shares ~ real: dose_change > start = stop).
_INFUSIONS_PER_ICU_INTERVAL = 0.3
_DOSE_CHANGE_MEAN = 1.28
_FLUID_UNIT = "mL/hr"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class MedAdminRow:
    """One continuous-med administration event (a start, or a zero-dose stop)."""

    hospitalization_id: str
    med_order_id: str
    admin_dttm: datetime
    med_category: str
    med_route_category: str
    med_dose: float
    med_dose_unit: str
    mar_action_category: str


def _stop_hazard(pack: ParamPack, med: str) -> float:
    block = pack.tables.get("medication_admin_continuous")
    if block is None or "params" not in block:
        return _DEFAULT_STOP_HAZARD
    hazards = block["params"].get("infusion_hazards", {})
    return float(hazards.get(med, {}).get("stop_hazard", _DEFAULT_STOP_HAZARD))


def _infusion_rows(
    hid: str,
    med: str,
    active: list[bool],
    pack: ParamPack,
    rng: np.random.Generator,
    admit_dttm: datetime,
    grid_step: float,
    order_seq: Iterator[int],
) -> list[MedAdminRow]:
    """Emit start/stop rows for one med gated on its per-interval ``active`` mask."""
    lo, hi = _DOSE_RANGE[med]
    stop_hazard = _stop_hazard(pack, med)
    rows: list[MedAdminRow] = []
    running = False
    order_id = ""

    def row(dose: float, action: str, t: int) -> MedAdminRow:
        return MedAdminRow(
            hospitalization_id=hid,
            med_order_id=order_id,
            admin_dttm=admit_dttm + timedelta(hours=t * grid_step),
            med_category=med,
            med_route_category=_ROUTE,
            med_dose=round(dose, 4),
            med_dose_unit=_DOSE_UNIT,
            mar_action_category=action,
        )

    for t, on in enumerate(active):
        if on and not running:
            running = True
            order_id = f"{hid}-{med}-{next(order_seq)}"
            rows.append(row(float(rng.uniform(lo, hi)), "start", t))
        elif on and running and rng.random() < stop_hazard:
            rows.append(row(0.0, "stop", t))  # AE3: new zero-dose stop row
            running = False
        elif not on and running:
            rows.append(row(0.0, "stop", t))
            running = False
    if running:
        rows.append(row(0.0, "stop", len(active)))  # stop at discharge
    return rows


def _dose_for(med: str, rng: np.random.Generator) -> tuple[float, str]:
    """Documented un-fitted dose + unit by med class (dose is not the fidelity target)."""
    if med in _VASOPRESSORS:
        return round(float(rng.uniform(0.02, 0.5)), 4), _DOSE_UNIT
    if med in _SEDATIVES:
        return round(float(rng.uniform(5.0, 50.0)), 2), _DOSE_UNIT
    return round(float(rng.uniform(10.0, 250.0)), 1), _FLUID_UNIT


def _fitted_infusions(
    hid: str,
    spine: SpineFrame,
    params: dict[str, Any],
    rng: np.random.Generator,
    admit_dttm: datetime,
    grid_step: float,
    order_seq: Iterator[int],
) -> list[MedAdminRow]:
    """Marginal-driven infusions: med drawn from the fitted marginal, *placed* by acuity.

    Sampling *which* drug from ``med_category_marginal`` makes the med_category
    distribution match real; placing vasopressors in cv-failure windows and
    sedatives in ventilation windows preserves the R12 couplings. Each infusion is
    a start + dose-changes + a zero-dose stop (AE3), matching the real mar_action mix.
    """
    marginal: dict[str, float] = params["med_category_marginal"]
    icu = [t for t, lvl in enumerate(spine.support_level) if lvl >= ICU_MIN_SUPPORT_LEVEL]
    if not icu:
        return []
    cv = [t for t in icu if spine.cv_flag[t]]
    imv = [t for t, lvl in enumerate(spine.support_level) if lvl >= IMV_MIN_SUPPORT_LEVEL]

    def pick_interval(med: str) -> int:
        if med in _VASOPRESSORS and cv:
            return cv[int(rng.integers(len(cv)))]
        if med in _SEDATIVES and imv:
            return imv[int(rng.integers(len(imv)))]
        return icu[int(rng.integers(len(icu)))]

    def at(interval: int) -> datetime:
        return admit_dttm + timedelta(hours=(interval + float(rng.random())) * grid_step)

    rows: list[MedAdminRow] = []

    def emit(med: str, start_t: int) -> None:
        """One infusion: start + dose-changes + a zero-dose stop (AE3)."""
        unit = _dose_for(med, rng)[1]
        oid = f"{hid}-{next(order_seq)}"

        def make(dttm: datetime, dose: float, action: str) -> MedAdminRow:
            return MedAdminRow(hid, oid, dttm, med, _ROUTE, round(dose, 4), unit, action)

        rows.append(make(at(start_t), _dose_for(med, rng)[0], "start"))
        for _c in range(int(rng.poisson(_DOSE_CHANGE_MEAN))):
            rows.append(make(at(start_t), _dose_for(med, rng)[0], "dose_change"))
        rows.append(make(at(start_t), 0.0, "stop"))

    # Guaranteed vasopressor for a cardiovascular-failure stay (R12): the marginal
    # alone samples vasopressors only at their base frequency, so without this the
    # vasopressor *stay*-rate tracks the med marginal instead of cv-failure
    # prevalence. This restores the coupling on top of the marginal draws.
    if cv:
        emit(_VASOPRESSOR, cv[int(rng.integers(len(cv)))])

    # Guaranteed sedative for most IMV stays (ICU doctor logic: ventilated →
    # sedated). Rate defaults to reference P(sedation|IMV)≈0.85 so we pair without
    # saturating every short/comfort IMV window.
    sed_imv_prob = float(params.get("sedation_imv_prob", 1.0))
    if imv and rng.random() < sed_imv_prob:
        emit(_SEDATIVE, imv[int(rng.integers(len(imv)))])

    # Per-interval infusion volume. When a derived pack sets
    # ``vasopressor_per_stay`` the vasopressor classes are removed from this
    # LOS-scaling draw and left entirely to the per-stay cv path above: otherwise
    # a realistic multi-day ICU stay accumulates so many marginal draws that
    # nearly every stay eventually samples a pressor, inflating the vasopressor
    # *stay*-rate far above its real prevalence. ``sedation_per_imv`` does the
    # same for sedatives — only IMV stays draw them (paired with ventilation).
    draw_marginal = marginal
    if params.get("vasopressor_per_stay"):
        # Confine vasopressor use to cardiovascular-failure stays so the *stay*
        # prevalence tracks cv-failure rate instead of climbing with stay length.
        # Non-cv stays draw no pressors; cv stays draw from a marginal with the
        # pressor weight boosted so the vasopressor *row-share* — concentrated in
        # ~a third of stays — still matches the real cohort-wide share.
        if cv:
            boost = float(params.get("vasopressor_cv_boost", 1.0))
            draw_marginal = {
                m: (w * boost if m in _VASOPRESSORS else w) for m, w in draw_marginal.items()
            }
        else:
            draw_marginal = {m: w for m, w in draw_marginal.items() if m not in _VASOPRESSORS}
    if params.get("sedation_per_imv", True):
        # Sedatives only via the stay-level IMV Bernoulli above — strip from the
        # LOS-scaling draw so long IMV stays don't saturate to sedation|IMV=1.0
        # (reference is ~0.85).
        draw_marginal = {m: w for m, w in draw_marginal.items() if m not in _SEDATIVES}
    total = sum(draw_marginal.values())
    draw_marginal = {m: w / total for m, w in draw_marginal.items()} if total > 0 else marginal
    n = int(rng.poisson(_INFUSIONS_PER_ICU_INTERVAL * len(icu)))
    for _ in range(n):
        med = categorical(draw_marginal, rng)
        if med in _SEDATIVES and not imv:
            continue
        emit(med, pick_interval(med))

    rows.sort(key=lambda r: (r.admin_dttm, r.med_category, r.mar_action_category))
    return rows


def sample_medication_admin_continuous(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[MedAdminRow]:
    """Emit one hospitalization's continuous-med rows (R11, AE3, R22).

    Uses the fitted marginal path when the pack carries ``med_category_marginal``;
    otherwise the documented two-med (vasopressor + sedative) coupling path — which
    keeps age-less/marginal-less packs (including the demo) byte-identical.
    """
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    order_seq = iter(range(10**6))

    block = pack.tables.get("medication_admin_continuous", {})
    params = block.get("params", {}) if isinstance(block, dict) else {}
    if params.get("med_category_marginal"):
        return _fitted_infusions(hid, spine, params, rng, admit_dttm, grid_step, order_seq)

    vaso_active = list(spine.cv_flag)
    sed_active = [level >= IMV_MIN_SUPPORT_LEVEL for level in spine.support_level]
    # Optional stay-level Bernoulli (``sedation_imv_prob``); default 1.0 keeps
    # demo/default packs fully paired. Recalibrate sets ~0.85 to match the reference.
    if any(sed_active) and rng.random() >= float(params.get("sedation_imv_prob", 1.0)):
        sed_active = [False] * len(sed_active)
    rows = _infusion_rows(
        hid, _VASOPRESSOR, vaso_active, pack, rng, admit_dttm, grid_step, order_seq
    )
    rows += _infusion_rows(hid, _SEDATIVE, sed_active, pack, rng, admit_dttm, grid_step, order_seq)
    rows.sort(key=lambda r: (r.admin_dttm, r.med_category, r.mar_action_category))
    return rows


def medication_admin_continuous_frame(rows: list[MedAdminRow]) -> pl.DataFrame:
    """Stack med-admin events into one conformant frame.

    ``med_group`` and ``mar_action_group`` are roll-ups of the categories, read
    from the same vendored mCIDE files that define the categories themselves
    rather than restated here — the consortium already publishes both mappings,
    so deriving them is exact and cannot drift out of sync with the category list.
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
