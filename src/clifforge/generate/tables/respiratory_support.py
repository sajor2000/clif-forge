"""Tier 4 ``respiratory_support`` generator (U12; R10, AE1, AE2, KTD-6).

There is no fitted ``respiratory_support`` block in the pack, so the device/mode
sequence is derived **entirely from the latent spine** (like adt): the organ-
support ladder in ``support_level`` *is* the respiratory-support trajectory
(0 room-air, 1 low-flow O2, 2 high-flow/NIV, 3+ IMV), refined by the spine's
respiratory-failure flag. Contiguous device runs are RLE'd into one row each on
the admit+grid timeline.

**R10 — device × mode set matrix.** Each device populates exactly its expected
``*_set`` fields (:data:`DEVICE_SET_FIELDS`) and nulls the rest. Values are
documented clinical constants with jitter (un-fitted, like the adt constants),
kept inside the consortium bounds — not invented distributions (R15).

**AE1 — Trach Collar implies IMV off.** Once a tracheostomy is present and the
patient weans off full ventilation, the device becomes ``Trach Collar`` (a
weaning device, IMV off), not a fresh oxygen device.

**AE2 — tracheostomy latches.** ``tracheostomy`` is INT 0/1 and, once set after a
sustained IMV run, persists for the rest of the encounter; a later return to
ventilation rides the existing trach rather than emitting a new intubation
transition.

**Coupling (KTD-6/KTD-7).** At the high-flow/NIV boundary (level 2) the spine's
respiratory-failure flag escalates the device to IMV, so severe respiratory
failure raises IMV prevalence. The spine is the only cross-table channel.

``device_name`` / ``mode_name`` take the first token from the vendored mCIDE
``*_name_examples`` column (else echo the category). On IMV rows, observed vent
readings (``*_obs``) are a small jitter around the paired set values (or a
documented in-bounds prior when R10 leaves the paired set null), clamped to
consortium outlier bounds. Off-matrix ``*_set`` fields and ``vent_brand_name``
remain deliberate omissions.

Output is reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import IMV_MIN_SUPPORT_LEVEL, UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame
from clifforge.reference import bounds, loader

__all__ = [
    "DEVICE_SET_FIELDS",
    "RespiratorySupportRow",
    "respiratory_support_frame",
    "sample_respiratory_support",
]

#: Sustained consecutive IMV intervals after which a tracheostomy is placed and
#: latches on (documented heuristic; the pack does not fit trach timing).
_TRACH_MIN_IMV_INTERVALS = 72

#: R10 device -> the exact ``*_set`` fields it populates; all others are null.
DEVICE_SET_FIELDS: dict[str, tuple[str, ...]] = {
    "IMV": ("fio2_set", "tidal_volume_set", "resp_rate_set"),
    "NIPPV": ("fio2_set", "peep_set", "pressure_support_set"),
    "CPAP": ("fio2_set", "peep_set"),
    "High Flow NC": ("fio2_set", "lpm_set"),
    "Nasal Cannula": ("lpm_set",),
    "Face Mask": (),
    "Trach Collar": (),
    "Room Air": (),
}

#: When a derived pack sets ``enrich_devices``, each stay draws one non-invasive
#: oxygen device (used at the low-flow/NIV tiers) from these documented options,
#: so the device_category mix includes Face Mask / NIPPV rather than only High
#: Flow NC — matching the real distribution while keeping segments stable.
_LOW_FLOW_DEVICES = ("Nasal Cannula", "Nasal Cannula", "Face Mask")
_NIV_DEVICES = ("High Flow NC", "High Flow NC", "High Flow NC", "NIPPV", "CPAP", "Face Mask")

#: Device -> ventilator mode_category (only ventilated devices carry a mode).
_DEVICE_MODE: dict[str, str | None] = {
    "IMV": "Assist Control-Volume Control",
    "NIPPV": "Pressure Support/CPAP",
}

#: All ``*_set`` columns the frame carries (union of the matrix targets).
_SET_COLUMNS = (
    "fio2_set",
    "lpm_set",
    "tidal_volume_set",
    "resp_rate_set",
    "peep_set",
    "pressure_support_set",
)

#: Observed vent columns emitted on IMV rows (null elsewhere).
_OBS_COLUMNS = (
    "tidal_volume_obs",
    "resp_rate_obs",
    "plateau_pressure_obs",
    "peak_inspiratory_pressure_obs",
    "peep_obs",
    "minute_vent_obs",
    "mean_airway_pressure_obs",
)

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class RespiratorySupportRow:
    """One contiguous device segment within a hospitalization."""

    hospitalization_id: str
    device_id: str
    recorded_dttm: datetime
    device_category: str
    mode_category: str | None
    tracheostomy: int
    set_values: dict[str, float]  # only the device's matrix fields, all in-bounds
    obs_values: dict[str, float]  # IMV-only observed readings, empty otherwise


def _device_for(level: int, resp_failure: bool, *, l2_noninvasive: bool = False) -> str:
    """Base device for an interval from acuity, escalated by respiratory failure.

    At the high-flow/NIV tier (level 2), respiratory failure escalates support.
    By default that escalation is to IMV (the original coupling). When a derived
    pack sets ``l2_resp_noninvasive`` the level-2 escalation is **non-invasive**
    (NIPPV) instead — reserving IMV for the intubation tier (level >= 3) so that
    IMV prevalence tracks the spine's ventilator acuity rather than the
    respiratory-failure flag. This matters once stays reach realistic length: a
    flag active across a long level-2 dwell would otherwise mint days of spurious
    IMV for a patient the spine never intubated.
    """
    if level >= IMV_MIN_SUPPORT_LEVEL:
        return "IMV"
    if level == 2:
        if resp_failure:
            return "NIPPV" if l2_noninvasive else "IMV"  # non-invasive vs intubation
        return "High Flow NC"
    if level == 1:
        return "Nasal Cannula"
    return "Room Air"


def _device_gated(level: int, l2_niv: str | None, low_flow: str) -> str:
    """Base device when non-invasive support is a per-stay gated property.

    The ICU floor (level 2) baseline is **low-flow** — non-invasive support
    (NIPPV / High Flow NC) is assigned once per stay at its real per-stay
    prevalence (``l2_niv``) rather than minted for every floor stay. IMV is
    reserved for the intubation tier (level >= 3), so the floor never produces
    spurious ventilation regardless of the respiratory-failure flag.
    """
    if level >= IMV_MIN_SUPPORT_LEVEL:
        return "IMV"
    if level == 2:
        return l2_niv if l2_niv is not None else low_flow
    if level == 1:
        return low_flow
    return "Room Air"


def _set_values(device: str, rng: np.random.Generator) -> dict[str, float]:
    """Documented in-bounds settings for the device's matrix fields (un-fitted)."""
    pool = {
        "fio2_set": round(float(rng.uniform(0.3, 0.7)), 2),
        "lpm_set": round(
            float(rng.uniform(30.0, 55.0) if device == "High Flow NC" else rng.uniform(1.0, 6.0)), 1
        ),
        "tidal_volume_set": round(float(rng.uniform(380.0, 500.0)), 0),
        "resp_rate_set": round(float(rng.uniform(12.0, 22.0)), 0),
        "peep_set": round(float(rng.uniform(5.0, 12.0)), 0),
        "pressure_support_set": round(float(rng.uniform(5.0, 15.0)), 0),
    }
    return {field: pool[field] for field in DEVICE_SET_FIELDS[device]}


def _clamp_obs(field: str, value: float) -> float:
    lo, hi = bounds("respiratory_support", field)
    return min(max(value, lo), hi)


def _obs_values(
    device: str, set_vals: dict[str, float], rng: np.random.Generator
) -> dict[str, float]:
    """IMV-only observed readings: jitter from paired sets + documented priors."""
    if device != "IMV":
        return {}
    tv = set_vals["tidal_volume_set"]
    rr = set_vals["resp_rate_set"]
    tv_obs = _clamp_obs("tidal_volume_obs", tv * float(rng.uniform(0.92, 1.08)))
    rr_obs = _clamp_obs("resp_rate_obs", rr * float(rng.uniform(0.9, 1.1)))
    peep = _clamp_obs("peep_obs", float(rng.uniform(5.0, 12.0)))
    plateau = _clamp_obs("plateau_pressure_obs", peep + float(rng.uniform(8.0, 18.0)))
    pip = _clamp_obs(
        "peak_inspiratory_pressure_obs", plateau + float(rng.uniform(2.0, 8.0))
    )
    map_ = _clamp_obs(
        "mean_airway_pressure_obs", (peep + plateau) / 2.0 * float(rng.uniform(0.9, 1.1))
    )
    minute = _clamp_obs("minute_vent_obs", (tv_obs * rr_obs) / 1000.0)
    return {
        "tidal_volume_obs": round(tv_obs, 0),
        "resp_rate_obs": round(rr_obs, 0),
        "plateau_pressure_obs": round(plateau, 1),
        "peak_inspiratory_pressure_obs": round(pip, 1),
        "peep_obs": round(peep, 0),
        "minute_vent_obs": round(minute, 2),
        "mean_airway_pressure_obs": round(map_, 1),
    }


@lru_cache(maxsize=1)
def _device_names() -> dict[str, str]:
    raw = loader.crosswalk("respiratory_support", "device_category", "device_name_examples")
    return {k: (v.split(",")[0].strip() if v else k) for k, v in raw.items()}


@lru_cache(maxsize=1)
def _mode_names() -> dict[str, str]:
    raw = loader.crosswalk("respiratory_support", "mode_category", "mode_name_examples")
    return {k: (v.split(",")[0].strip() if v else k) for k, v in raw.items()}


def sample_respiratory_support(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[RespiratorySupportRow]:
    """Emit one hospitalization's respiratory-support rows (R10, AE1, AE2, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    if not spine.support_level:
        return []  # empty spine -> no device segments (match the sibling generators)

    # Per-stay non-invasive devices (drawn once so segments stay stable), used
    # only when a derived pack enables device enrichment.
    block = pack.tables.get("respiratory_support", {})
    params = block.get("params", {}) if isinstance(block, dict) else {}
    enrich = bool(params.get("enrich_devices"))
    l2_noninvasive = bool(params.get("l2_resp_noninvasive"))
    low_flow = (
        _LOW_FLOW_DEVICES[int(rng.integers(len(_LOW_FLOW_DEVICES)))] if enrich else "Nasal Cannula"
    )
    niv = _NIV_DEVICES[int(rng.integers(len(_NIV_DEVICES)))] if enrich else "High Flow NC"

    # New gated path: when the pack carries an ``niv`` target, non-invasive support
    # is assigned once per stay at its real per-stay prevalence (rather than minted
    # for every ICU-floor stay, which over-produced NIPPV/HFNC several-fold). Drawn
    # here so the assignment is stable across the stay's segments.
    niv_target = params.get("niv")
    l2_niv: str | None = None
    if isinstance(niv_target, dict):
        r = rng.random()
        p_nippv = float(niv_target.get("nippv_prob", 0.0))
        p_hfnc = float(niv_target.get("hfnc_prob", 0.0))
        l2_niv = "NIPPV" if r < p_nippv else ("High Flow NC" if r < p_nippv + p_hfnc else None)

    # Per-interval (device, tracheostomy) with the latch + AE1 weaning rule.
    trach = 0
    imv_run = 0
    timeline: list[tuple[str, int]] = []
    for level, resp in zip(spine.support_level, spine.resp_flag, strict=True):
        if niv_target is not None:
            device = _device_gated(level, l2_niv, low_flow)
        else:
            device = _device_for(level, resp, l2_noninvasive=l2_noninvasive)
            if enrich and device == "Nasal Cannula":
                device = low_flow
            elif enrich and device == "High Flow NC":
                device = niv
        if device == "IMV":
            imv_run += 1
            if imv_run >= _TRACH_MIN_IMV_INTERVALS:
                trach = 1  # latches on (AE2)
        else:
            imv_run = 0
            if trach == 1 and level >= 1:
                device = "Trach Collar"  # weaning with a trach in place (AE1)
        timeline.append((device, trach))

    # RLE contiguous (device, trach) runs into one row each.
    rows: list[RespiratorySupportRow] = []
    seg_start = 0
    for idx in range(len(timeline) + 1):
        if idx < len(timeline) and timeline[idx] == timeline[seg_start]:
            continue
        device, seg_trach = timeline[seg_start]
        sets = _set_values(device, rng)
        rows.append(
            RespiratorySupportRow(
                hospitalization_id=hid,
                device_id=f"{hid}-D{len(rows)}",
                recorded_dttm=admit_dttm + timedelta(hours=seg_start * grid_step),
                device_category=device,
                mode_category=_DEVICE_MODE.get(device),
                tracheostomy=seg_trach,
                set_values=sets,
                obs_values=_obs_values(device, sets, rng),
            )
        )
        seg_start = idx
    return rows


def respiratory_support_frame(rows: list[RespiratorySupportRow]) -> pl.DataFrame:
    """Stack device segments into one conformant ``respiratory_support`` frame."""
    device_names = _device_names()
    mode_names = _mode_names()
    schema: dict[str, pl.DataType] = {
        "hospitalization_id": pl.String(),
        "device_id": pl.String(),
        "recorded_dttm": UTC_DATETIME,
        "device_name": pl.String(),
        "device_category": pl.String(),
        "mode_name": pl.String(),
        "mode_category": pl.String(),
        "tracheostomy": pl.Int64(),
    }
    data: dict[str, list[object]] = {name: [] for name in schema}
    for col in _SET_COLUMNS:
        schema[col] = pl.Float64()
        data[col] = []
    for col in _OBS_COLUMNS:
        schema[col] = pl.Float64()
        data[col] = []
    for r in rows:
        data["hospitalization_id"].append(r.hospitalization_id)
        data["device_id"].append(r.device_id)
        data["recorded_dttm"].append(r.recorded_dttm)
        data["device_name"].append(device_names.get(r.device_category, r.device_category))
        data["device_category"].append(r.device_category)
        data["mode_name"].append(
            mode_names.get(r.mode_category, r.mode_category) if r.mode_category else None
        )
        data["mode_category"].append(r.mode_category)
        data["tracheostomy"].append(r.tracheostomy)
        for col in _SET_COLUMNS:
            data[col].append(r.set_values.get(col))
        for col in _OBS_COLUMNS:
            data[col].append(r.obs_values.get(col))
    return pl.DataFrame(data, schema=schema)
