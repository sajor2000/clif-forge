"""``patient_diagnosis`` generator (prior-driven; R14, KTD-6).

Where ``hospital_diagnosis`` is the *billed* record of one encounter,
``patient_diagnosis`` is the clinical one: the problem list and medical history a
patient carries, plus the encounter-level diagnoses documented during the stay.
CLIF 2.1 marks it ICD-10-CM only (``diagnosis_code_format`` permits ``ICD10CM``
alone, unlike ``hospital_diagnosis``, which also permits ICD-9), because clinical
documentation is not back-coded.

The three ``source_type`` values are used for what they mean rather than sampled:

* ``medical_history`` and ``problem_list`` carry chronic comorbidities. These
  predate the admission, so their ``start_dttm`` is *before* it, and they are
  ongoing — a ``NULL`` ``end_dttm``, which the DDL defines as the encoding for a
  condition that has not ended. Writing a discharge time there would assert that
  the patient's diabetes resolved when they went home.
* ``encounter_dx`` carries what was diagnosed during this stay, so it starts at
  admission and mirrors the spine's organ-failure flags — the same source the
  billing table reads, so the two agree by construction.

The comorbidity set and its prevalences are documented priors (no fitted
diagnosis block exists), recorded in ``PROVENANCE.md``. Reproducible under a
fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["PatientDiagnosisRow", "patient_diagnosis_frame", "sample_patient_diagnosis"]

_CODE_FORMAT = "ICD10CM"

_PROBLEM_LIST = "problem_list"
_MEDICAL_HISTORY = "medical_history"
_ENCOUNTER_DX = "encounter_dx"

#: Chronic conditions, with prevalence and how long before this admission they
#: were first documented. Real ICD-10-CM codes.
_CHRONIC: tuple[tuple[str, float, int], ...] = (
    ("E11.9", 0.30, 1460),  # Type 2 diabetes mellitus — documented years earlier
    ("I10", 0.45, 1825),  # Essential hypertension
    ("N18.3", 0.15, 900),  # Chronic kidney disease, stage 3
    ("J44.9", 0.18, 1200),  # COPD
    ("I50.9", 0.20, 730),  # Heart failure
    ("Z95.1", 0.06, 2555),  # Presence of aortocoronary bypass graft
)

#: Spine flag -> the ICD-10-CM code documented for it during the encounter. Kept
#: identical to hospital_diagnosis so the clinical and billed records agree.
_FLAG_CODES: tuple[tuple[str, str], ...] = (
    ("resp_flag", "J96.00"),
    ("cv_flag", "R65.21"),
    ("renal_flag", "N17.9"),
    ("neuro_flag", "G93.40"),
)

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class PatientDiagnosisRow:
    """One clinically documented diagnosis. ``end_dttm`` is None while ongoing."""

    patient_id: str
    hospitalization_id: str
    diagnosis_code: str
    diagnosis_code_format: str
    source_type: str
    start_dttm: datetime
    end_dttm: datetime | None


def sample_patient_diagnosis(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    patient_id: str | None = None,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[PatientDiagnosisRow]:
    """Emit this patient's problem list, history, and encounter diagnoses (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    pid = patient_id if patient_id is not None else hid
    if spine.n_intervals <= 0:
        return []

    rows: list[PatientDiagnosisRow] = []
    for code, prevalence, days_before in _CHRONIC:
        if rng.random() >= prevalence:
            continue
        # Jitter onset so a cohort does not share one documentation date.
        onset = admit_dttm - timedelta(days=days_before + float(rng.integers(0, 365)))
        source = _PROBLEM_LIST if rng.random() < 0.6 else _MEDICAL_HISTORY
        rows.append(PatientDiagnosisRow(pid, hid, code, _CODE_FORMAT, source, onset, None))

    discharge = admit_dttm + timedelta(hours=spine.n_intervals * grid_step_hours(pack))
    expired = spine.outcome == "expired"
    for flag_name, code in _FLAG_CODES:
        if not any(getattr(spine, flag_name)):
            continue
        # Acute organ failure resolves by discharge for a survivor; for a patient
        # who died it never did, so it stays open.
        rows.append(
            PatientDiagnosisRow(
                patient_id=pid,
                hospitalization_id=hid,
                diagnosis_code=code,
                diagnosis_code_format=_CODE_FORMAT,
                source_type=_ENCOUNTER_DX,
                start_dttm=admit_dttm,
                end_dttm=None if expired else discharge,
            )
        )
    return rows


def patient_diagnosis_frame(rows: list[PatientDiagnosisRow]) -> pl.DataFrame:
    """Stack clinical diagnoses into one conformant frame."""
    return pl.DataFrame(
        {
            "patient_id": [r.patient_id for r in rows],
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "diagnosis_code": [r.diagnosis_code for r in rows],
            "diagnosis_code_format": [r.diagnosis_code_format for r in rows],
            "source_type": [r.source_type for r in rows],
            "start_dttm": [r.start_dttm for r in rows],
            "end_dttm": [r.end_dttm for r in rows],
        },
        schema={
            "patient_id": pl.String,
            "hospitalization_id": pl.String,
            "diagnosis_code": pl.String,
            "diagnosis_code_format": pl.String,
            "source_type": pl.String,
            "start_dttm": UTC_DATETIME,
            "end_dttm": UTC_DATETIME,
        },
    )
