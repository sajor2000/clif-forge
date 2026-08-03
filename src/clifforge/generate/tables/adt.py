"""Tier 2 ``adt`` (admission/discharge/transfer) generator (U9; R8).

The ``adt`` table records a hospitalization's location movements. There is no
fitted ``adt`` block in the parameter pack, so movements are derived **entirely
from the latent spine** (KTD-6): the per-interval acuity trajectory is
run-length-encoded into contiguous ``ward`` / ``icu`` location segments, placed
on the same ``admit_dttm`` + grid timeline the hospitalization generator uses so
the last ``out_dttm`` coincides with the encounter's discharge.

Acuity -> location heuristic: an interval whose support level is at or above
:data:`ICU_MIN_SUPPORT_LEVEL` (high-flow O2 / NIV and above) is ``icu``, else
``ward``. This is the plan's "ICU windows align with high-acuity spine segments"
made concrete; it is a heuristic, not a fitted mapping, because the source cohort ``adt``
location structure was not fitted by U5.

The resulting ICU segments are exposed via :func:`icu_windows` — the sole channel
by which later tiers (vitals, labs, …) restrict observations to ICU time, read
through the spine/orchestrator rather than by cross-reading this table.

Un-fitted fields use documented source-cohort-appropriate constants rather than invented
distributions (R15): ``hospital_type = "academic"`` (the source cohort is a single academic
center), ``location_type = "medical_icu"`` for ICU rows (null off-ICU).

**Arrival location.** By default every movement is a deterministic function of the
spine, reproducible under any ``rng`` (R22). A derived pack may instead carry an
``arrival_location_marginal`` (an ED/OR-dominant mix over ``location_category``) and
a ``direct_icu_frac``: the *first* (admission) segment's location is then drawn from
the passed ``rng`` — ED-dominant with ward / procedural / stepdown variety — rather
than hardcoded to the emergency department, so the full hospital population arrives
through a realistic spread of front doors. A stay whose spine reaches invasive
ventilation is routed straight to ``icu`` (a direct ICU admit) with probability
``direct_icu_frac``; the remainder of reaches-ICU stays arrive elsewhere and
transfer into ICU later, so transfer-to-ICU is ``reaches_icu * (1 - direct_icu_frac)``.
Intervals after the first keep the spine-driven mapping. When no
``arrival_location_marginal`` is set (the ICU master and demo packs) the ``rng`` is
unused and the admission location is unchanged, preserving byte-for-byte output.

**Coupled admission route.** A spine may carry an ``admission_route`` (drawn once per
stay from the pack's ``admission_route_marginal``, KTD-6) that *supersedes* the
arrival marginal at the admission interval: the route maps deterministically to a
location (see :data:`_ROUTE_TO_ARRIVAL`) so the ADT front door and the hospitalization
``admission_type_category`` agree per stay, and an outside-hospital transfer (``osh``)
arrives straight at the higher-level ``icu``. Precedence at idx 0 is therefore
``admission_route`` (if set) > ``arrival_location_marginal`` (if set) > default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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

__all__ = ["ICU_MIN_SUPPORT_LEVEL", "AdtMovement", "adt_frame", "icu_windows", "sample_adt"]


#: source-cohort-appropriate constants for fields with no fitted distribution (R15).
_HOSPITAL_TYPE = "academic"
_ICU_LOCATION_TYPE = "medical_icu"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)

#: Coupled admission-route -> admission (first) ADT location. When the spine carries
#: an ``admission_route`` (drawn once per stay from the pack's ``admission_route_marginal``,
#: KTD-6) it maps deterministically to the arrival location, so the hospitalization's
#: ``admission_type_category`` and this ADT front door agree per stay. ``osh`` (an
#: outside-hospital transfer) arrives straight at the higher-level ``icu``; ``elective``
#: (a scheduled OR / post-op admit) at ``procedural``; ``direct`` / ``facility`` at the
#: ``ward``; ``ed`` / ``other`` through the emergency department. Values are exact mCIDE
#: ``location_category`` members.
_ROUTE_TO_ARRIVAL: dict[str, str] = {
    "ed": "ed",
    "elective": "procedural",
    "direct": "ward",
    "osh": "icu",
    "facility": "ward",
    "other": "ed",
}


@dataclass(frozen=True)
class AdtMovement:
    """One contiguous location stay within a hospitalization."""

    hospitalization_id: str
    hospital_id: str
    hospital_type: str
    in_dttm: datetime
    out_dttm: datetime
    location_name: str
    location_category: str
    location_type: str | None


def _category_for(level: int, idx: int, enrich: bool) -> str:
    """Location category for an interval.

    Default (un-fitted source-cohort pack): ``icu`` at/above the ICU threshold, else
    ``ward``. Enriched (a derived pack sets ``enrich_locations``): the admission
    interval is ``ed`` (patients arrive through the emergency department), the
    high-flow/NIV tier is ``stepdown``, and invasive ventilation and above is
    ``icu`` — matching the real ward/ed/icu/stepdown location mix.
    """
    if not enrich:
        return "icu" if level >= ICU_MIN_SUPPORT_LEVEL else "ward"
    if idx == 0:
        return "ed"
    if level >= IMV_MIN_SUPPORT_LEVEL:
        return "icu"
    if level >= ICU_MIN_SUPPORT_LEVEL:
        return "stepdown"
    return "ward"


def _location_segments(
    support_level: list[int], *, enrich: bool = False, arrival: str | None = None
) -> list[tuple[str, int]]:
    """Run-length-encode the acuity trajectory into ``(category, n_intervals)``.

    ``arrival`` overrides only the admission interval's category (idx 0); all later
    intervals keep the spine-driven mapping. Consecutive equal categories are still
    collapsed, so a ``ward`` arrival that continues at ward acuity merges naturally.
    """
    segments: list[tuple[str, int]] = []
    for idx, level in enumerate(support_level):
        if idx == 0 and arrival is not None:
            category = arrival
        else:
            category = _category_for(level, idx, enrich)
        if segments and segments[-1][0] == category:
            prev_cat, prev_n = segments[-1]
            segments[-1] = (prev_cat, prev_n + 1)
        else:
            segments.append((category, 1))
    return segments


def _arrival_category(
    spine: SpineFrame, params: dict[str, object], rng: np.random.Generator
) -> str | None:
    """Draw the admission-interval location, or ``None`` to keep the default mapping.

    Precedence at the admission interval (idx 0):

    1. **Coupled admission route** — when the spine carries an ``admission_route``
       (drawn per stay from the pack's ``admission_route_marginal``, KTD-6) it maps
       deterministically to the arrival location via :data:`_ROUTE_TO_ARRIVAL`, so
       the arrival and the hospitalization ``admission_type_category`` agree per stay
       (``osh`` arrives straight at ``icu``). The ``rng`` is **not** drawn — the
       route already fixed the front door.
    2. **``arrival_location_marginal``** — when there is no route but the pack carries
       this marginal, the front door is sampled from the passed ``rng`` (R22): a spine
       that reaches invasive ventilation is a direct ICU admit (``icu``) with
       probability ``direct_icu_frac``, otherwise the arrival is drawn from the
       marginal (an ED/OR-dominant mix with ward / stepdown variety).
    3. **Default** — absent both, returns ``None`` and the caller keeps the unchanged
       spine mapping, so the ``rng`` is never drawn and output stays byte-for-byte
       identical (ICU master / demo).
    """
    if spine.admission_route:
        return _ROUTE_TO_ARRIVAL.get(spine.admission_route, "ed")
    marginal = params.get("arrival_location_marginal")
    if not isinstance(marginal, dict) or not marginal:
        return None
    reaches_icu = bool(spine.support_level) and max(spine.support_level) >= IMV_MIN_SUPPORT_LEVEL
    direct_icu_frac = float(params.get("direct_icu_frac", 0.0))  # type: ignore[arg-type]
    if reaches_icu and rng.random() < direct_icu_frac:
        return "icu"
    return categorical(marginal, rng)


def sample_adt(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    hospital_id: str = "HOSP0",
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[AdtMovement]:
    """Emit one hospitalization's ordered, contiguous location movements (R8, R22).

    Segments tile ``[admit_dttm, admit_dttm + n_intervals * grid_step]`` with no
    gaps or overlaps; the terminal ``out_dttm`` equals the hospitalization's
    discharge (both use the pack grid). ``hospitalization_id`` defaults to the
    spine's own id. The ``rng`` is drawn only when the pack carries an
    ``arrival_location_marginal`` (see :func:`_arrival_category`); otherwise the
    result is a pure function of the spine.
    """
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    block = pack.tables.get("adt", {})
    params = block.get("params", {}) if isinstance(block, dict) else {}
    enrich = bool(params.get("enrich_locations"))
    arrival = _arrival_category(spine, params, rng)

    hosp_type_marginal = params.get("hospital_type_marginal")
    if isinstance(hosp_type_marginal, dict) and hosp_type_marginal:
        hospital_type = categorical(hosp_type_marginal, rng)
    else:
        hospital_type = _HOSPITAL_TYPE
    loc_type_marginal = params.get("location_type_marginal")

    movements: list[AdtMovement] = []
    cursor = admit_dttm
    for category, n_int in _location_segments(spine.support_level, enrich=enrich, arrival=arrival):
        out = cursor + timedelta(hours=n_int * grid_step)
        if category == "icu":
            if isinstance(loc_type_marginal, dict) and loc_type_marginal:
                location_type = categorical(loc_type_marginal, rng)
            else:
                location_type = _ICU_LOCATION_TYPE
        else:
            location_type = None
        movements.append(
            AdtMovement(
                hospitalization_id=hid,
                hospital_id=hospital_id,
                hospital_type=hospital_type,
                in_dttm=cursor,
                out_dttm=out,
                location_name=category,
                location_category=category,
                location_type=location_type,
            )
        )
        cursor = out
    return movements


def icu_windows(movements: list[AdtMovement]) -> dict[str, list[tuple[datetime, datetime]]]:
    """Map ``hospitalization_id -> [(in_dttm, out_dttm), …]`` over ICU stays only.

    This is the channel later tiers use to keep observations inside ICU time
    (KTD-6). A hospitalization with no ICU segment is absent from the mapping.
    """
    windows: dict[str, list[tuple[datetime, datetime]]] = {}
    for m in movements:
        if m.location_category == "icu":
            windows.setdefault(m.hospitalization_id, []).append((m.in_dttm, m.out_dttm))
    return windows


def adt_frame(movements: list[AdtMovement]) -> pl.DataFrame:
    """Stack movements into one conformant ``adt`` frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [m.hospitalization_id for m in movements],
            "hospital_id": [m.hospital_id for m in movements],
            "hospital_type": [m.hospital_type for m in movements],
            "in_dttm": [m.in_dttm for m in movements],
            "out_dttm": [m.out_dttm for m in movements],
            "location_name": [m.location_name for m in movements],
            "location_category": [m.location_category for m in movements],
            "location_type": [m.location_type for m in movements],
        },
        schema={
            "hospitalization_id": pl.String,
            "hospital_id": pl.String,
            "hospital_type": pl.String,
            "in_dttm": UTC_DATETIME,
            "out_dttm": UTC_DATETIME,
            "location_name": pl.String,
            "location_category": pl.String,
            "location_type": pl.String,
        },
    )
