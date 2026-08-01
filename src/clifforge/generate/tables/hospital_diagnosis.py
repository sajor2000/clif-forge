"""``hospital_diagnosis`` generator (prior-driven; R14, KTD-6).

Billing diagnoses are the coded record of what an encounter was *about*, so they
are read off the spine's organ-failure flags rather than sampled independently: a
stay that carried ``resp_flag`` gets acute respiratory failure, ``renal_flag``
gets acute kidney failure, and so on. That keeps the coded record consistent with
the vitals, labs, and support the same spine produced (KTD-6).

Codes are real **ICD-10-CM**. CLIF 2.1 types ``diagnosis_code`` as ``VARCHAR`` and
permits "Valid ICD-9-CM or ICD-10-CM"; ICD-10-CM is what US hospitals have coded
in since 2015, so it is the honest default. The flag -> code mapping is a
documented prior (there is no fitted diagnosis block), recorded in
``PROVENANCE.md``.

``diagnosis_primary`` and ``poa_present`` are the 0/1 integers CLIF requires, and
they carry real meaning rather than being filled in: exactly one row per
encounter is primary, and present-on-admission is true for what the patient
arrived with and false for organ failure that developed during the stay. The
spine knows the difference — a flag set in the first interval was present on
admission; one that first appears later was not — so POA is derived from *when*
each flag first fires instead of being coin-flipped.

Reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame

__all__ = ["DiagnosisRow", "hospital_diagnosis_frame", "sample_hospital_diagnosis"]

_CODE_FORMAT = "ICD10CM"

#: Spine organ-failure flag -> its real ICD-10-CM code.
_FLAG_CODES: tuple[tuple[str, str], ...] = (
    ("resp_flag", "J96.00"),  # Acute respiratory failure, unspecified w/ hypoxia or hypercapnia
    ("cv_flag", "R65.21"),  # Severe sepsis with septic shock
    ("renal_flag", "N17.9"),  # Acute kidney failure, unspecified
    ("neuro_flag", "G93.40"),  # Encephalopathy, unspecified
)

#: The admitting diagnosis every encounter carries. J18.9 = pneumonia, unspecified
#: organism: the most common single ICU admitting diagnosis.
_PRINCIPAL_CODE = "J18.9"

#: Chronic conditions coded on a share of encounters. These are comorbidities the
#: patient arrived with, so they are always present-on-admission and never primary.
_COMORBIDITY_CODES: tuple[tuple[str, float], ...] = (
    ("E11.9", 0.30),  # Type 2 diabetes mellitus without complications
    ("I10", 0.45),  # Essential (primary) hypertension
    ("N18.3", 0.15),  # Chronic kidney disease, stage 3
    ("J44.9", 0.18),  # COPD, unspecified
    ("I50.9", 0.20),  # Heart failure, unspecified
)


@dataclass(frozen=True)
class DiagnosisRow:
    """One coded billing diagnosis for a hospitalization."""

    hospitalization_id: str
    diagnosis_code: str
    diagnosis_code_format: str
    diagnosis_primary: int
    poa_present: int


def _first_true(flags: list[bool]) -> int | None:
    """Index of the first interval where a flag fires, or ``None`` if it never does."""
    for i, flag in enumerate(flags):
        if flag:
            return i
    return None


def sample_hospital_diagnosis(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = datetime(2020, 1, 1, tzinfo=UTC),
) -> list[DiagnosisRow]:
    """Emit the coded diagnosis list implied by this stay's spine (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    if spine.n_intervals <= 0:
        return []

    rows = [DiagnosisRow(hid, _PRINCIPAL_CODE, _CODE_FORMAT, 1, 1)]
    for flag_name, code in _FLAG_CODES:
        onset = _first_true(getattr(spine, flag_name))
        if onset is None:
            continue
        # Organ failure present in the first interval came in with the patient;
        # anything that starts later developed in hospital.
        rows.append(DiagnosisRow(hid, code, _CODE_FORMAT, 0, 1 if onset == 0 else 0))
    for code, prevalence in _COMORBIDITY_CODES:
        if rng.random() < prevalence:
            rows.append(DiagnosisRow(hid, code, _CODE_FORMAT, 0, 1))
    return rows


def hospital_diagnosis_frame(rows: list[DiagnosisRow]) -> pl.DataFrame:
    """Stack diagnosis rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "diagnosis_code": [r.diagnosis_code for r in rows],
            "diagnosis_code_format": [r.diagnosis_code_format for r in rows],
            "diagnosis_primary": [r.diagnosis_primary for r in rows],
            "poa_present": [r.poa_present for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "diagnosis_code": pl.String,
            "diagnosis_code_format": pl.String,
            "diagnosis_primary": pl.Int64,
            "poa_present": pl.Int64,
        },
    )
