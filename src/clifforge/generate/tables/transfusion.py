"""Tier 6 ``transfusion`` generator (U20; prior-driven, R14, KTD-6).

Blood-product transfusions concentrate in sicker patients, so a documented base
rate is scaled by the spine's peak acuity (``peak_level``) into a small Poisson
count per stay. When a fitted pack block is present, stay prevalence / component
marginals / volume edges win; otherwise dashboard + adult-product norms apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours, pack_table_params
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference.dashboard_priors import absent_table_rates as _DASH_RATES

__all__ = ["TransfusionRow", "sample_transfusion", "transfusion_frame"]

_TRANSFUSION_BASE_RATE = _DASH_RATES["transfusion"] * 5.0
_COMPONENT_MARGINAL = {"RBC": 0.6, "FFP": 0.25, "Platelets": 0.15}
_COMPONENT_VOLUME = {"RBC": 300.0, "FFP": 250.0, "Platelets": 300.0}
_VOLUME_UNITS = "mL"

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class TransfusionRow:
    """One blood-product transfusion."""

    hospitalization_id: str
    transfusion_start_dttm: datetime
    transfusion_end_dttm: datetime
    component_name: str
    volume_transfused: float
    volume_units: str
    product_code: str


def _draw_volume(params: dict[str, Any], component: str, rng: np.random.Generator) -> float:
    edges = params.get("volume_transfused_quantile_bin_edges")
    if isinstance(edges, list) and len(edges) >= 2:
        i = int(rng.integers(0, len(edges) - 1))
        a, b = float(edges[i]), float(edges[i + 1])
        if a > b:
            a, b = b, a
        return round(float(a if a == b else rng.uniform(a, b)), 1)
    base = _COMPONENT_VOLUME.get(component, 300.0)
    return round(base * float(rng.uniform(0.85, 1.1)), 1)


def sample_transfusion(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[TransfusionRow]:
    """Emit a stay's transfusions, scaled by peak acuity (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    los_hours = spine.n_intervals * grid_step_hours(pack)
    if los_hours <= 0:
        return []

    params = pack_table_params(pack, "transfusion")
    gated = "stay_prevalence" in params
    if gated:
        if rng.random() >= float(params["stay_prevalence"]):
            return []
        lam = max(0.5, float(params.get("events_per_positive_stay", 1.5)))
    else:
        lam = _TRANSFUSION_BASE_RATE * (spine.peak_level / 5.0)

    component_marginal = params.get("component_name_marginal")
    if not isinstance(component_marginal, dict) or not component_marginal:
        component_marginal = _COMPONENT_MARGINAL

    # After a stay-level gate, Poisson must be zero-truncated so prevalence is honored.
    n = int(rng.poisson(lam))
    if gated:
        n = max(1, n)
    rows: list[TransfusionRow] = []
    for k in range(n):
        component = categorical(component_marginal, rng)
        start = admit_dttm + timedelta(hours=float(rng.random()) * los_hours)
        end = start + timedelta(hours=float(rng.uniform(1.0, 3.0)))
        rows.append(
            TransfusionRow(
                hospitalization_id=hid,
                transfusion_start_dttm=start,
                transfusion_end_dttm=end,
                component_name=component,
                volume_transfused=_draw_volume(params, component, rng),
                volume_units=_VOLUME_UNITS,
                product_code=f"{component[:3].upper()}-{hid}-{k}",
            )
        )
    rows.sort(key=lambda r: r.transfusion_start_dttm)
    return rows


def transfusion_frame(rows: list[TransfusionRow]) -> pl.DataFrame:
    """Stack transfusions into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "transfusion_start_dttm": [r.transfusion_start_dttm for r in rows],
            "transfusion_end_dttm": [r.transfusion_end_dttm for r in rows],
            "component_name": [r.component_name for r in rows],
            "attribute_name": [r.component_name for r in rows],
            "volume_transfused": [r.volume_transfused for r in rows],
            "volume_units": [r.volume_units for r in rows],
            "product_code": [r.product_code for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "transfusion_start_dttm": UTC_DATETIME,
            "transfusion_end_dttm": UTC_DATETIME,
            "component_name": pl.String,
            "attribute_name": pl.String,
            "volume_transfused": pl.Float64,
            "volume_units": pl.String,
            "product_code": pl.String,
        },
    )
