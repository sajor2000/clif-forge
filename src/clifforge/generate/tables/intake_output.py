"""``intake_output`` generator (prior-driven; R14, KTD-6).

Fluid balance is charted continuously in the ICU, so this table is dense rather
than sparse: intake and output are recorded on a documented hourly cadence for
every ICU interval of the stay.

It is coupled to the spine in the one way that matters clinically. Urine output
is the bedside signal of renal function, so a stay carrying ``renal_flag`` charts
oliguria — output falls to a documented oliguric range instead of the normal one —
while intake rises, because a shocked or oliguric patient is being resuscitated.
Emitting a flat normal balance for a patient the spine says is in acute kidney
failure would contradict the ``labs`` creatinine and the ``crrt_therapy`` rows
generated from that same flag.

``in_out_flag`` is the canonical 0/1 integer (1 = intake, 0 = output), and
``amount`` is mL. CLIF 2.1 gives this table no mCIDE, so ``fluid_name`` is free
text drawn from documented ICU fluids (R15, recorded in ``PROVENANCE.md``).

Reproducible under a fixed ``rng`` (R22).
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

__all__ = ["IntakeOutputRow", "intake_output_frame", "sample_intake_output"]

_INTAKE = 1
_OUTPUT = 0

#: Fluid balance is charted hourly.
_CHART_INTERVAL_HOURS = 1.0

#: Intake sources and their documented per-hour volume ranges (mL).
_INTAKE_FLUIDS = {
    "IV Fluid - Lactated Ringers": 0.4,
    "IV Fluid - Normal Saline": 0.35,
    "Enteral Feed": 0.25,
}
_INTAKE_ML_PER_HOUR = (40.0, 125.0)
#: A resuscitated patient runs positive; the spine's cv/renal flags mark them.
_INTAKE_ML_PER_HOUR_RESUSCITATED = (100.0, 250.0)

_URINE = "Urine"
#: Normal adult urine output is ~0.5-1.5 mL/kg/hr; oliguria is <0.5 mL/kg/hr.
#: Expressed here as absolute mL/hr for an ~80 kg adult.
_URINE_ML_PER_HOUR = (40.0, 120.0)
_URINE_ML_PER_HOUR_OLIGURIC = (5.0, 30.0)
#: Near-anuria during L5 / multi-organ failure (CRRT-tier acuity).
_URINE_ML_PER_HOUR_ANURIC = (0.0, 5.0)
_ANURIA_SUPPORT_LEVEL = 5

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class IntakeOutputRow:
    """One charted fluid intake or output event."""

    hospitalization_id: str
    intake_dttm: datetime
    fluid_name: str
    amount: float
    in_out_flag: int


def sample_intake_output(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[IntakeOutputRow]:
    """Chart hourly intake and urine output across the ICU stay (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)
    params = pack_table_params(pack, "intake_output")
    chart_hours = float(params.get("chart_interval_hours", _CHART_INTERVAL_HOURS))
    stride = max(1, round(chart_hours / grid_step))
    fluid_marginal = params.get("fluid_name_marginal")
    if not isinstance(fluid_marginal, dict) or not fluid_marginal:
        fluid_marginal = _INTAKE_FLUIDS

    rows: list[IntakeOutputRow] = []
    for idx in range(0, spine.n_intervals, stride):
        if spine.support_level[idx] < ICU_MIN_SUPPORT_LEVEL:
            continue
        at = admit_dttm + timedelta(hours=idx * grid_step)
        oliguric = spine.renal_flag[idx]
        anuric = oliguric and spine.support_level[idx] >= _ANURIA_SUPPORT_LEVEL
        resuscitated = oliguric or spine.cv_flag[idx]

        intake_range = _INTAKE_ML_PER_HOUR_RESUSCITATED if resuscitated else _INTAKE_ML_PER_HOUR
        amount_edges = params.get("amount_quantile_bin_edges_by_in_out_flag", {}).get(str(_INTAKE))
        # Pack edges only when physiology is ordinary; resus/oliguria keep clinical ranges.
        if (
            not resuscitated
            and isinstance(amount_edges, list)
            and len(amount_edges) >= 2
        ):
            i = int(rng.integers(0, len(amount_edges) - 1))
            a, b = float(amount_edges[i]), float(amount_edges[i + 1])
            if a > b:
                a, b = b, a
            intake_amt = round(float(a if a == b else rng.uniform(a, b)), 1)
        else:
            intake_amt = round(float(rng.uniform(*intake_range)), 1)
        intake_fluid_marginal = {
            k: v for k, v in fluid_marginal.items() if k != _URINE
        } or _INTAKE_FLUIDS
        rows.append(
            IntakeOutputRow(
                hospitalization_id=hid,
                intake_dttm=at,
                fluid_name=categorical(intake_fluid_marginal, rng),
                amount=intake_amt,
                in_out_flag=_INTAKE,
            )
        )
        if anuric:
            urine_range = _URINE_ML_PER_HOUR_ANURIC
        elif oliguric:
            urine_range = _URINE_ML_PER_HOUR_OLIGURIC
        else:
            urine_range = _URINE_ML_PER_HOUR
        out_edges = params.get("amount_quantile_bin_edges_by_in_out_flag", {}).get(str(_OUTPUT))
        if isinstance(out_edges, list) and len(out_edges) >= 2 and not (oliguric or anuric):
            i = int(rng.integers(0, len(out_edges) - 1))
            a, b = float(out_edges[i]), float(out_edges[i + 1])
            if a > b:
                a, b = b, a
            urine_amt = round(float(a if a == b else rng.uniform(a, b)), 1)
        else:
            urine_amt = round(float(rng.uniform(*urine_range)), 1)
        rows.append(
            IntakeOutputRow(
                hospitalization_id=hid,
                intake_dttm=at,
                fluid_name=_URINE,
                amount=urine_amt,
                in_out_flag=_OUTPUT,
            )
        )
    return rows


def intake_output_frame(rows: list[IntakeOutputRow]) -> pl.DataFrame:
    """Stack fluid-balance rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "intake_dttm": [r.intake_dttm for r in rows],
            "fluid_name": [r.fluid_name for r in rows],
            "amount": [r.amount for r in rows],
            "in_out_flag": [r.in_out_flag for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "intake_dttm": UTC_DATETIME,
            "fluid_name": pl.String,
            "amount": pl.Float64,
            "in_out_flag": pl.Int64,
        },
    )
