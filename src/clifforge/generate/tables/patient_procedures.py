"""``patient_procedures`` generator (prior-driven; R14, KTD-6).

This is the **billing** record of procedures, not a bedside log: CLIF 2.1 keys it
on ``procedure_code`` / ``procedure_code_format`` with a billing and a performing
provider, and timestamps it with ``procedure_billed_dttm``, which the DDL notes
"may differ from actual procedure time".

Codes are **real CPT codes read from the consortium's own vendored code list**
(``mCIDE/patient_procedures/clif_patient_procedure_codes.csv``) rather than
invented ones — see :func:`clifforge.reference.loader.code_list`. That list is a
deliberately narrow, high-acuity set (pulmonary decortication, thoracoscopy, and
lung/heart/liver/kidney transplant), so procedures are emitted **rarely** and only
for stays that reach the mechanical-ventilation rungs of the support ladder.
Billing a lung transplant to a routine ward stay would be worse than emitting no
row at all.

Billing is retrospective, so ``procedure_billed_dttm`` lands after the procedure,
during or shortly after the stay. Providers are string ids (CLIF leaves both
provider columns optional and unconstrained). Reproducible under a fixed ``rng``
(R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame
from clifforge.reference import loader

__all__ = ["ProcedureRow", "patient_procedures_frame", "sample_patient_procedures"]

_TABLE = "patient_procedures"

#: The vendored code list is major thoracic surgery and transplant, so it applies
#: only to ventilated patients (support level 3+ on the ladder).
_MIN_SUPPORT_LEVEL = 3
#: Fraction of eligible stays that receive one of these procedures. Deliberately
#: small: these are rare operations, not routine ICU care.
_PROCEDURE_PROB = 0.04

#: Billing lands after the procedure, up to a few days later.
_BILLING_LAG_HOURS = (12.0, 96.0)

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class ProcedureRow:
    """One billed procedure."""

    hospitalization_id: str
    billing_provider_id: str
    performing_provider_id: str
    procedure_code: str
    procedure_code_format: str
    procedure_billed_dttm: datetime


def sample_patient_procedures(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[ProcedureRow]:
    """Emit a rare billed procedure for a high-acuity stay (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    if spine.n_intervals <= 0 or spine.peak_level < _MIN_SUPPORT_LEVEL:
        return []
    if rng.random() >= _PROCEDURE_PROB:
        return []

    codes = loader.code_list(_TABLE)
    entry = codes[int(rng.integers(0, len(codes)))]

    grid_step = grid_step_hours(pack)
    # Performed somewhere in the stay, then billed afterwards.
    performed_idx = int(rng.integers(0, spine.n_intervals))
    performed = admit_dttm + timedelta(hours=performed_idx * grid_step)
    billed = performed + timedelta(hours=float(rng.uniform(*_BILLING_LAG_HOURS)))

    return [
        ProcedureRow(
            hospitalization_id=hid,
            billing_provider_id=f"{hid}-BILL",
            performing_provider_id=f"{hid}-PERF",
            procedure_code=entry["procedure_code"],
            procedure_code_format=entry["procedure_code_format"],
            procedure_billed_dttm=billed,
        )
    ]


def patient_procedures_frame(rows: list[ProcedureRow]) -> pl.DataFrame:
    """Stack billed procedures into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "billing_provider_id": [r.billing_provider_id for r in rows],
            "performing_provider_id": [r.performing_provider_id for r in rows],
            "procedure_code": [r.procedure_code for r in rows],
            "procedure_code_format": [r.procedure_code_format for r in rows],
            "procedure_billed_dttm": [r.procedure_billed_dttm for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "billing_provider_id": pl.String,
            "performing_provider_id": pl.String,
            "procedure_code": pl.String,
            "procedure_code_format": pl.String,
            "procedure_billed_dttm": UTC_DATETIME,
        },
    )
