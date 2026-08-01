"""Tier 1 ``hospitalization`` table generator (U8; R8, R12, AE4).

Each hospitalization is one encounter driven by a latent spine: its length of
stay is the spine's horizon (``n_intervals`` on the fitted grid), and its
death/discharge disposition is the spine's terminal outcome. This is the first
generator to read the spine (KTD-6): it consumes only ``spine.outcome`` and
``spine.n_intervals``, never another table's output.

**AE4 (death/discharge consistency).** CLIF 2.1.0 records the moment of death on
``patient.death_dttm``, not on ``hospitalization`` — the encounter's death signal
is ``discharge_category == "Expired"``. So:

* spine outcome ``expired`` -> ``discharge_category = "Expired"`` and the record
  exposes ``death_dttm = discharge_dttm`` (the value the U21 orchestrator writes
  back to the owning ``patient`` row, closing the cross-table couple, R12/AE4).
* spine outcome ``alive`` -> ``discharge_category`` drawn from the pack marginal
  **with the death category removed and renormalized**, and ``death_dttm = None``.

``patient_id`` / ``hospitalization_id`` are caller-assigned (the orchestrator owns
the id scheme and one-to-many linking, R8); the sampled content is reproducible
byte-for-byte under a fixed ``rng`` (R22). ``age_at_admission`` is drawn from the
pack's quantile grid when present. Geographic zip/census codes are omitted, not
fabricated (R15; schema is permissive) — ``place_based_index`` carries deprivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame

__all__ = [
    "DEATH_DISCHARGE_CATEGORY",
    "HospitalizationRecord",
    "hospitalization_frame",
    "sample_hospitalization",
]

#: The one mCIDE discharge_category that means the patient died in the encounter.
#: "Hospice" is a live disposition, not death, so only this value is excluded
#: from the survivor discharge distribution.
DEATH_DISCHARGE_CATEGORY = "Expired"

#: Default admission instant when the caller does not supply one. The orchestrator
#: spreads real admit times across a calendar; a fixed tz-aware epoch keeps a
#: standalone sample reproducible and tz-aware UTC (R7, R22).
_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class HospitalizationRecord:
    """One encounter linked to a patient, with AE4-consistent disposition.

    ``death_dttm`` is ``discharge_dttm`` for a death and ``None`` for a survivor;
    it is the value the orchestrator writes to the owning ``patient.death_dttm``
    (it is not a ``hospitalization`` column in CLIF 2.1.0).
    """

    patient_id: str
    hospitalization_id: str
    hospitalization_joined_id: str
    admission_dttm: datetime
    discharge_dttm: datetime
    admission_type_category: str
    admission_type_name: str
    discharge_category: str
    discharge_name: str
    death_dttm: datetime | None
    age_at_admission: int | None = None  # populated only when the pack fits an age grid


def _hospitalization_params(pack: ParamPack) -> dict[str, Any]:
    block = pack.tables.get("hospitalization")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'hospitalization' block to sample from")
    params: dict[str, Any] = block["params"]
    return params


def _sample_age(quantiles: list[float], rng: np.random.Generator) -> int:
    """Draw an age by inverse-CDF interpolation over a quantile grid ([q0..qN])."""
    pos = float(rng.random()) * (len(quantiles) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(quantiles) - 1)
    value = quantiles[lo] + (pos - lo) * (quantiles[hi] - quantiles[lo])
    return int(round(value))


def sample_hospitalization(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str = "H0",
    patient_id: str = "P0",
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> HospitalizationRecord:
    """Sample one hospitalization from its spine and the pack (R8, R12, AE4, R22).

    Length of stay is ``spine.n_intervals`` on the fitted grid; disposition is
    driven by ``spine.outcome`` for AE4 consistency. When the spine carries a
    coupled ``admission_route`` (a per-stay pathway on the latent spine, KTD-6)
    it *is* the ``admission_type_category`` — the route values are mCIDE
    ``admission_type_category`` members — so admission type and the ADT arrival
    location agree per stay; the ``rng`` is then not drawn for it. Otherwise the
    admission type is drawn from ``rng`` off the pack marginal (ICU master / demo,
    byte-for-byte unchanged). Survivors then draw ``discharge_category``.
    """
    params = _hospitalization_params(pack)
    los_hours = spine.n_intervals * grid_step_hours(pack)
    discharge_dttm = admit_dttm + timedelta(hours=los_hours)

    if spine.admission_route:
        admission_type = spine.admission_route
    else:
        admission_type = categorical(params["admission_type_category_marginal"], rng)

    if spine.outcome == "expired":
        discharge_category = DEATH_DISCHARGE_CATEGORY
        death_dttm: datetime | None = discharge_dttm
    else:
        survivor_marginal = {
            cat: prob
            for cat, prob in params["discharge_category_marginal"].items()
            if cat != DEATH_DISCHARGE_CATEGORY
        }
        if not survivor_marginal:
            raise ValueError(
                "hospitalization discharge_category marginal has no non-death category "
                "to draw a survivor disposition from"
            )
        discharge_category = categorical(survivor_marginal, rng)
        death_dttm = None

    # Age is drawn last so the rng order for packs without an age grid (and thus
    # the byte-for-byte reproducibility of their output) is unchanged.
    age_quantiles = params.get("age_at_admission_quantiles")
    age = _sample_age(age_quantiles, rng) if age_quantiles else None

    return HospitalizationRecord(
        patient_id=patient_id,
        hospitalization_id=hospitalization_id,
        hospitalization_joined_id=hospitalization_id,
        admission_dttm=admit_dttm,
        discharge_dttm=discharge_dttm,
        admission_type_category=admission_type,
        admission_type_name=admission_type,
        discharge_category=discharge_category,
        discharge_name=discharge_category,
        death_dttm=death_dttm,
        age_at_admission=age,
    )


def hospitalization_frame(records: list[HospitalizationRecord]) -> pl.DataFrame:
    """Stack sampled encounters into one conformant ``hospitalization`` frame.

    ``age_at_admission`` is emitted only when the pack fit an age grid (any record
    carries it); otherwise the column is omitted entirely, so output for age-less
    packs stays byte-for-byte identical (R22).
    """
    data: dict[str, list[object]] = {
        "patient_id": [r.patient_id for r in records],
        "hospitalization_id": [r.hospitalization_id for r in records],
        "hospitalization_joined_id": [r.hospitalization_joined_id for r in records],
        "admission_dttm": [r.admission_dttm for r in records],
        "discharge_dttm": [r.discharge_dttm for r in records],
        "admission_type_name": [r.admission_type_name for r in records],
        "admission_type_category": [r.admission_type_category for r in records],
        "discharge_name": [r.discharge_name for r in records],
        "discharge_category": [r.discharge_category for r in records],
    }
    schema: dict[str, Any] = {
        "patient_id": pl.String,
        "hospitalization_id": pl.String,
        "hospitalization_joined_id": pl.String,
        "admission_dttm": UTC_DATETIME,
        "discharge_dttm": UTC_DATETIME,
        "admission_type_name": pl.String,
        "admission_type_category": pl.String,
        "discharge_name": pl.String,
        "discharge_category": pl.String,
    }
    if any(r.age_at_admission is not None for r in records):
        data["age_at_admission"] = [r.age_at_admission for r in records]
        schema["age_at_admission"] = pl.Int64
    return pl.DataFrame(data, schema=schema)
