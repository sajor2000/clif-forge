"""``patient_diagnosis`` generator (prior-driven; R14, KTD-6).

Where ``hospital_diagnosis`` is the *billed* record of one encounter,
``patient_diagnosis`` is the clinical one: problem list / medical history plus
encounter-level diagnoses. Chronic and cancer load use the same MIMIC ICU
stay-prevalence priors and acuity scaling as ``hospital_diagnosis`` so the two
tables agree on case mix (KTD-6).

Reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables.hospital_diagnosis import (
    _CANCER_BASE_RATE,
    _CANCER_CODES,
    _DISEASE_PRIORS,
    _FLAG_CODES,
    _RESP_PHENOTYPE_CODES,
    _disease_rate,
    _emit_resp_failure,
    _emit_sepsis,
)

__all__ = ["PatientDiagnosisRow", "patient_diagnosis_frame", "sample_patient_diagnosis"]

_CODE_FORMAT = "ICD10CM"

_PROBLEM_LIST = "problem_list"
_MEDICAL_HISTORY = "medical_history"
_ENCOUNTER_DX = "encounter_dx"

#: Days before admission when a chronic code is typically first documented.
_CHRONIC_DAYS_BEFORE: dict[str, int] = {
    "I10": 1825,
    "E78.5": 1460,
    "E11.9": 1460,
    "I25.10": 1100,
    "I48.91": 900,
    "I50.9": 730,
    "J44.9": 1200,
    "N18.3": 900,
    "N18.4": 700,
    "N18.5": 500,
    "E66.9": 1500,
    "E66.2": 1825,
    "K74.60": 800,
    "K70.30": 800,
    "F32.9": 1000,
    "F41.9": 1000,
    "E03.9": 1600,
    "Z87.891": 2000,
    "K21.9": 1400,
    "D62": 30,
    "Z79.01": 600,
    "Z79.4": 800,
    "I11.0": 900,
    "I12.9": 900,
}

_RESP_PHENOTYPE_COMORBID: dict[str, tuple[tuple[str, int], ...]] = {
    "type2_ohs": (("E66.2", 1825),),
    "type2_hf": (("I50.9", 730),),
    "type2_copd": (("J44.9", 1200),),
}

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


def _chronic_onset(admit_dttm: datetime, code: str, rng: np.random.Generator) -> datetime:
    days = _CHRONIC_DAYS_BEFORE.get(code, 1000)
    return admit_dttm - timedelta(days=days + float(rng.integers(0, 365)))


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
    seen: set[str] = set()
    phenotype = getattr(spine, "resp_phenotype", "") or ""

    for code, days_before in _RESP_PHENOTYPE_COMORBID.get(phenotype, ()):
        onset = admit_dttm - timedelta(days=days_before + float(rng.integers(0, 365)))
        source = _PROBLEM_LIST if rng.random() < 0.6 else _MEDICAL_HISTORY
        rows.append(PatientDiagnosisRow(pid, hid, code, _CODE_FORMAT, source, onset, None))
        seen.add(code)

    for family, codes, base in _DISEASE_PRIORS:
        if family == "hf" and phenotype == "type2_hf":
            continue
        if family == "copd" and phenotype == "type2_copd":
            continue
        if family == "obesity" and phenotype == "type2_ohs":
            continue
        if any(c in seen for c in codes):
            continue
        if rng.random() >= _disease_rate(family, base, spine):
            continue
        code = codes[int(rng.integers(0, len(codes)))]
        onset = _chronic_onset(admit_dttm, code, rng)
        source = _PROBLEM_LIST if rng.random() < 0.6 else _MEDICAL_HISTORY
        rows.append(PatientDiagnosisRow(pid, hid, code, _CODE_FORMAT, source, onset, None))
        seen.add(code)

    cancer_p = _disease_rate("cancer", _CANCER_BASE_RATE, spine)
    if spine.outcome == "expired":
        cancer_p = min(0.55, cancer_p + 0.06)
    if rng.random() < cancer_p:
        code = _CANCER_CODES[int(rng.integers(0, len(_CANCER_CODES)))]
        if code not in seen:
            onset = _chronic_onset(admit_dttm, code, rng)
            source = _PROBLEM_LIST if rng.random() < 0.5 else _MEDICAL_HISTORY
            rows.append(PatientDiagnosisRow(pid, hid, code, _CODE_FORMAT, source, onset, None))
            seen.add(code)

    discharge = admit_dttm + timedelta(hours=spine.n_intervals * grid_step_hours(pack))
    expired = spine.outcome == "expired"
    for flag_name, code in _FLAG_CODES:
        if not any(getattr(spine, flag_name)):
            continue
        if flag_name == "resp_flag":
            if not _emit_resp_failure(spine, rng):
                continue
            code = _RESP_PHENOTYPE_CODES.get(phenotype, code)
        elif flag_name == "cv_flag":
            if not _emit_sepsis(spine, rng):
                continue
        if code in seen:
            continue
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
        seen.add(code)
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
