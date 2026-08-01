"""Tier 1 ``patient`` table generator (U7; R5, R6, R8).

One synthetic patient's demographics are a draw of race / ethnicity / sex from
the parameter pack's category marginals. Demographics do **not** couple to the
latent acuity spine — KTD-6's ``(spine, pack, rng)`` channel is for per-encounter
clinical tables (vitals, respiratory support, …), whereas a patient is a stable
entity spanning one-or-more encounters — so this sampler takes only
``(pack, rng)``.

Faithful-omission notes (R15 — never invent un-fitted structure):

* ``*_name`` echoes the mCIDE ``*_category`` value. CLIF 2.1.0 ``patient`` is
  de-identified and carries **no person-name columns**, so there is nothing for
  Faker to fill; the ``*_name`` fields are the source-string columns behind each
  category, and echoing the human-readable category is a faithful stand-in.
* ``language_category`` and ``birth_date`` are not fitted by the U5 fit stage, so
  they are omitted rather than fabricated. mCIDE *does* enumerate the language
  values, but a permissible-value list is not a distribution: drawing uniformly
  across it would put English at a few percent, which is worse than admitting the
  field was never captured. The schema is permissive (``required=False``), so
  their absence still validates.
* ``death_dttm`` is emitted, but it is not sampled here — the orchestrator
  propagates it from the encounter that ended in death (U8, AE4), so the patient
  row cannot disagree with the hospitalization it came from.

``patient_id`` is assigned by the caller — the U21 orchestrator owns the id
scheme and the one-to-many ``patient_id -> hospitalization_id`` linking (KTD-6,
R8). Given a fixed ``rng`` the sampled *content* is reproducible byte-for-byte
(R22).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.sampling import categorical

__all__ = ["PatientRecord", "patient_frame", "sample_patient"]

#: The columns emitted, in order — all string. Optional CLIF columns U5 does not
#: fit (language/birth_date/death_dttm) are intentionally absent (schema permits).
_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "race_category",
    "race_name",
    "ethnicity_category",
    "ethnicity_name",
    "sex_category",
    "sex_name",
)


@dataclass(frozen=True)
class PatientRecord:
    """One patient's demographics (all mCIDE ``*_category`` values, R5)."""

    patient_id: str
    race_category: str
    race_name: str
    ethnicity_category: str
    ethnicity_name: str
    sex_category: str
    sex_name: str


def _patient_params(pack: ParamPack) -> dict[str, dict[str, float]]:
    block = pack.tables.get("patient")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'patient' block to sample from")
    params: dict[str, dict[str, float]] = block["params"]
    return params


def sample_patient(
    pack: ParamPack, rng: np.random.Generator, *, patient_id: str = "P0"
) -> PatientRecord:
    """Sample one patient's demographics from the pack marginals (R5, R6, R22).

    Draws race, then ethnicity, then sex from ``rng`` in that fixed order, so the
    same seed reproduces the same record. Every ``*_category`` is an exact,
    case-sensitive mCIDE member because it is drawn from the fitted marginal's own
    (mCIDE-conformant) keys; ``*_name`` echoes the category.
    """
    params = _patient_params(pack)
    race = categorical(params["race_category_marginal"], rng)
    ethnicity = categorical(params["ethnicity_category_marginal"], rng)
    sex = categorical(params["sex_category_marginal"], rng)
    return PatientRecord(
        patient_id=patient_id,
        race_category=race,
        race_name=race,
        ethnicity_category=ethnicity,
        ethnicity_name=ethnicity,
        sex_category=sex,
        sex_name=sex,
    )


def patient_frame(records: list[PatientRecord]) -> pl.DataFrame:
    """Stack sampled patient records into one conformant ``patient`` frame."""
    return pl.DataFrame(
        {
            "patient_id": [r.patient_id for r in records],
            "race_category": [r.race_category for r in records],
            "race_name": [r.race_name for r in records],
            "ethnicity_category": [r.ethnicity_category for r in records],
            "ethnicity_name": [r.ethnicity_name for r in records],
            "sex_category": [r.sex_category for r in records],
            "sex_name": [r.sex_name for r in records],
        },
        schema={name: pl.String for name in _COLUMNS},
    )
