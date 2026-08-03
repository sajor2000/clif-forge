"""Tier 5 ``code_status`` generator (U19; R12, KTD-6).

``code_status`` is a **patient-level** change-event table (its key is
``patient_id``). Every patient starts ``Full`` at admission. In trajectories the
spine marks as **expired**, a de-escalation to ``DNR/DNI`` and often ``AND``
(allow natural death / comfort care) is concentrated late in the stay, before
death — the R12 coupling to the terminal outcome. Survivors overwhelmingly stay
``Full``. Start times are strictly ordered per patient.

When the pack carries a fitted ``code_status`` block (MIMIC all-28 packs),
de-escalation rates are read from ``params``; otherwise documented priors apply
(R15). The spine is the only cross-table channel (KTD-6); output is reproducible
byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["CodeStatusEvent", "code_status_frame", "sample_code_status"]

#: Documented de-escalation rates near death (un-fitted).
_DNR_PROB_EXPIRED = 0.7
_COMFORT_PROB_EXPIRED = 0.5  # of those DNR, fraction that reach comfort care
_DNR_PROB_SURVIVOR = 0.05
#: Fractions of the stay at which late transitions are placed.
_DNR_AT_FRACTION = 0.7
_COMFORT_AT_FRACTION = 0.92

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CodeStatusEvent:
    """One code-status change event for a patient."""

    patient_id: str
    start_dttm: datetime
    code_status_category: str


def sample_code_status(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    patient_id: str = "P0",
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[CodeStatusEvent]:
    """Emit a patient's ordered code-status events (R12, R22)."""
    los_hours = spine.n_intervals * grid_step_hours(pack)
    events = [CodeStatusEvent(patient_id, admit_dttm, "Full")]  # everyone starts Full

    if los_hours <= 0:
        return events  # zero-length stay: only the admission Full status (keeps start times strict)

    block = pack.tables.get("code_status", {})
    params = block.get("params", {}) if isinstance(block, dict) else {}
    dnr_expired = float(params.get("dnr_prob_expired", _DNR_PROB_EXPIRED))
    comfort_expired = float(params.get("comfort_prob_expired", _COMFORT_PROB_EXPIRED))
    dnr_survivor = float(params.get("dnr_prob_survivor", _DNR_PROB_SURVIVOR))

    if spine.outcome == "expired":
        if rng.random() < dnr_expired:
            events.append(
                CodeStatusEvent(
                    patient_id,
                    admit_dttm + timedelta(hours=_DNR_AT_FRACTION * los_hours),
                    "DNR/DNI",
                )
            )
            if rng.random() < comfort_expired:
                events.append(
                    CodeStatusEvent(
                        patient_id,
                        admit_dttm + timedelta(hours=_COMFORT_AT_FRACTION * los_hours),
                        "AND",
                    )
                )
    elif rng.random() < dnr_survivor:
        events.append(
            CodeStatusEvent(
                patient_id,
                admit_dttm + timedelta(hours=_DNR_AT_FRACTION * los_hours),
                "DNR/DNI",
            )
        )

    return events


def code_status_frame(events: list[CodeStatusEvent]) -> pl.DataFrame:
    """Stack code-status events into one conformant frame."""
    return pl.DataFrame(
        {
            "patient_id": [e.patient_id for e in events],
            "start_dttm": [e.start_dttm for e in events],
            "code_status_name": [e.code_status_category for e in events],
            "code_status_category": [e.code_status_category for e in events],
        },
        schema={
            "patient_id": pl.String,
            "start_dttm": UTC_DATETIME,
            "code_status_name": pl.String,
            "code_status_category": pl.String,
        },
    )
