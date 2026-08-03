"""Tier 5 ``crrt_therapy`` generator (U18; R9, R12, KTD-6).

Continuous renal replacement is a wide device-parameter table charted while a
patient is on CRRT. No fitted block exists, so CRRT sessions are driven by the
spine's **renal-failure flag** (KTD-6/R12): a wide row is emitted per charting
interval where the flag is set, and none otherwise. The mode is CVVHDF (the
common ICU modality); blood/dialysate/replacement-fluid rates and ultrafiltration
are documented in-bounds constants with jitter (un-fitted, like the adt
constants), kept inside the consortium outlier bounds (R9) — not invented
distributions (R15).

``device_id`` is synthesized per stay. Output is reproducible byte-for-byte under
a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.device_catalogs import DIALYSIS_MACHINE_NAMES, pick_catalog
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference import bounds

__all__ = ["CrrtRow", "crrt_therapy_frame", "sample_crrt_therapy"]

_CRRT_MODE = "cvvhdf"
#: Require sustained renal failure before CRRT so creat has risen (reference:
#: high_creat|CRRT ≈ 0.97 — nearly all CRRT stays are top-quartile creatinine).
_MIN_RENAL_HOURS_BEFORE_CRRT = 12.0

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CrrtRow:
    """One CRRT charting row (device parameters at one time)."""

    hospitalization_id: str
    device_id: str
    recorded_dttm: datetime
    crrt_mode_category: str
    blood_flow_rate: float
    pre_filter_replacement_fluid_rate: float
    post_filter_replacement_fluid_rate: float
    dialysate_flow_rate: float
    ultrafiltration_out: float


def sample_crrt_therapy(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[CrrtRow]:
    """Emit one hospitalization's CRRT rows during renal-failure windows (R12, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    device_id = f"{hid}-CRRT"

    # Prefer the validated ``crrt_prob`` gate (P(CRRT|renal)) when the pack carries
    # it — same knob ``recalibrate_to_network_median`` sets. Fall back to empirical
    # ``stay_prevalence`` for packs that only store the stay-level rate.
    block = pack.tables.get("crrt_therapy", {})
    params = block.get("params", {}) if isinstance(block, dict) else {}
    if "crrt_prob" in params:
        crrt_prob = float(params["crrt_prob"])
        if crrt_prob < 1.0 and rng.random() >= crrt_prob:
            return []
    elif "stay_prevalence" in params:
        if rng.random() >= float(params["stay_prevalence"]):
            return []
    else:
        crrt_prob = float(params.get("crrt_prob", 1.0))
        if crrt_prob < 1.0 and rng.random() >= crrt_prob:
            return []

    def _draw_rate(field: str, lo: float, hi: float) -> float:
        try:
            blo, bhi = bounds("crrt_therapy", field)
            lo, hi = max(lo, blo), min(hi, bhi)
        except Exception:
            pass
        edges = params.get(f"{field}_quantile_bin_edges")
        if isinstance(edges, list) and len(edges) >= 2:
            clean = [float(e) for e in edges if e is not None and lo <= float(e) <= hi]
            if len(clean) >= 2:
                i = int(rng.integers(0, len(clean) - 1))
                a, b = clean[i], clean[i + 1]
                if a > b:
                    a, b = b, a
                return round(float(a if a == b else rng.uniform(a, b)), 1)
        return round(float(rng.uniform(lo, hi)), 1)

    mode_marginal = params.get("crrt_mode_category_marginal")
    mode = (
        categorical(mode_marginal, rng)
        if isinstance(mode_marginal, dict) and mode_marginal
        else _CRRT_MODE
    )

    renal_hours = sum(1 for r in spine.renal_flag if r) * grid_step
    if renal_hours < _MIN_RENAL_HOURS_BEFORE_CRRT:
        return []

    rows: list[CrrtRow] = []
    for idx, renal in enumerate(spine.renal_flag):
        if not renal:
            continue
        # Start CRRT only after the sustained-renal threshold within the stay.
        hours_so_far = sum(1 for r in spine.renal_flag[: idx + 1] if r) * grid_step
        if hours_so_far < _MIN_RENAL_HOURS_BEFORE_CRRT:
            continue
        rows.append(
            CrrtRow(
                hospitalization_id=hid,
                device_id=device_id,
                recorded_dttm=admit_dttm + timedelta(hours=idx * grid_step),
                crrt_mode_category=mode,
                blood_flow_rate=_draw_rate("blood_flow_rate", 180.0, 240.0),
                pre_filter_replacement_fluid_rate=_draw_rate(
                    "pre_filter_replacement_fluid_rate", 400.0, 800.0
                ),
                post_filter_replacement_fluid_rate=_draw_rate(
                    "post_filter_replacement_fluid_rate", 400.0, 800.0
                ),
                dialysate_flow_rate=_draw_rate("dialysate_flow_rate", 1500.0, 2500.0),
                ultrafiltration_out=_draw_rate("ultrafiltration_out", 50.0, 250.0),
            )
        )
    return rows


def crrt_therapy_frame(rows: list[CrrtRow]) -> pl.DataFrame:
    """Stack CRRT rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "device_id": [r.device_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "dialysis_machine_name": [
                pick_catalog(DIALYSIS_MACHINE_NAMES, r.device_id) for r in rows
            ],
            "crrt_mode_name": [r.crrt_mode_category for r in rows],
            "crrt_mode_category": [r.crrt_mode_category for r in rows],
            "blood_flow_rate": [r.blood_flow_rate for r in rows],
            "pre_filter_replacement_fluid_rate": [
                r.pre_filter_replacement_fluid_rate for r in rows
            ],
            "post_filter_replacement_fluid_rate": [
                r.post_filter_replacement_fluid_rate for r in rows
            ],
            "dialysate_flow_rate": [r.dialysate_flow_rate for r in rows],
            "ultrafiltration_out": [r.ultrafiltration_out for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "device_id": pl.String,
            "recorded_dttm": UTC_DATETIME,
            "dialysis_machine_name": pl.String,
            "crrt_mode_name": pl.String,
            "crrt_mode_category": pl.String,
            "blood_flow_rate": pl.Float64,
            "pre_filter_replacement_fluid_rate": pl.Float64,
            "post_filter_replacement_fluid_rate": pl.Float64,
            "dialysate_flow_rate": pl.Float64,
            "ultrafiltration_out": pl.Float64,
        },
    )
