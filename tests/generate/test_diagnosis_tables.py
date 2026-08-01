"""Unit tests for hospital_diagnosis and patient_diagnosis (R22, KTD-6).

Both tables code the same encounter from different angles, so the tests below
concentrate on the couplings that make them meaningful: diagnoses must follow the
spine's organ-failure flags, present-on-admission must reflect *when* a failure
started, and an ongoing condition must be encoded as a null end rather than a
fabricated resolution date.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import hospital_diagnosis as hd
from clifforge.generate.tables import patient_diagnosis as pd
from clifforge.generate.tables.hospital_diagnosis import (
    hospital_diagnosis_frame,
    sample_hospital_diagnosis,
)
from clifforge.generate.tables.patient_diagnosis import (
    patient_diagnosis_frame,
    sample_patient_diagnosis,
)

ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(
    n: int,
    hid: str = "H0",
    *,
    resp: list[bool] | None = None,
    renal: bool = False,
    outcome: str = "alive",
) -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[3] * n,
        resp_flag=resp if resp is not None else [False] * n,
        cv_flag=[False] * n,
        renal_flag=[renal] * n,
        neuro_flag=[False] * n,
        outcome=outcome,
    )


# --- hospital_diagnosis ------------------------------------------------------ #
def test_hospital_diagnosis_is_deterministic() -> None:
    pack = _pack()
    a = sample_hospital_diagnosis(_spine(24), pack, np.random.default_rng(1), admit_dttm=ADMIT)
    b = sample_hospital_diagnosis(_spine(24), pack, np.random.default_rng(1), admit_dttm=ADMIT)
    assert a == b


def test_exactly_one_diagnosis_is_primary() -> None:
    pack = _pack()
    for seed in range(30):
        rows = sample_hospital_diagnosis(
            _spine(24, renal=True), pack, np.random.default_rng(seed), admit_dttm=ADMIT
        )
        assert sum(r.diagnosis_primary for r in rows) == 1


def test_organ_failure_flags_produce_their_codes() -> None:
    pack = _pack()
    rows = sample_hospital_diagnosis(
        _spine(24, renal=True), pack, np.random.default_rng(2), admit_dttm=ADMIT
    )
    assert "N17.9" in {r.diagnosis_code for r in rows}


def test_a_stay_without_a_flag_does_not_code_it() -> None:
    pack = _pack()
    for seed in range(20):
        rows = sample_hospital_diagnosis(
            _spine(24, renal=False), pack, np.random.default_rng(seed), admit_dttm=ADMIT
        )
        assert "N17.9" not in {r.diagnosis_code for r in rows}


def test_present_on_admission_tracks_when_the_failure_started() -> None:
    """Failure in the first interval came in with the patient; later failure did not."""
    pack = _pack()
    on_arrival = [True] * 24
    developed_later = [False] * 12 + [True] * 12

    poa_of = {}
    for resp in (on_arrival, developed_later):
        rows = sample_hospital_diagnosis(
            _spine(24, resp=resp), pack, np.random.default_rng(3), admit_dttm=ADMIT
        )
        poa_of[tuple(resp)] = next(r.poa_present for r in rows if r.diagnosis_code == "J96.00")

    assert poa_of[tuple(on_arrival)] == 1
    assert poa_of[tuple(developed_later)] == 0


def test_flags_are_binary_integers() -> None:
    pack = _pack()
    rows = sample_hospital_diagnosis(
        _spine(24, renal=True), pack, np.random.default_rng(4), admit_dttm=ADMIT
    )
    for row in rows:
        assert row.diagnosis_primary in (0, 1)
        assert row.poa_present in (0, 1)


def test_hospital_diagnosis_frame_passes_the_gate() -> None:
    pack, rng = _pack(), np.random.default_rng(5)
    rows: list = []
    for i in range(40):
        rows += sample_hospital_diagnosis(
            _spine(24, hid=f"H{i}", renal=True), pack, rng, admit_dttm=ADMIT
        )
    gate.validate(hospital_diagnosis_frame(rows), "hospital_diagnosis")


# --- patient_diagnosis ------------------------------------------------------- #
def test_patient_diagnosis_is_deterministic() -> None:
    pack = _pack()
    kwargs = {"patient_id": "P0", "hospitalization_id": "H0", "admit_dttm": ADMIT}
    a = sample_patient_diagnosis(_spine(24), pack, np.random.default_rng(6), **kwargs)
    b = sample_patient_diagnosis(_spine(24), pack, np.random.default_rng(6), **kwargs)
    assert a == b


def test_chronic_conditions_predate_the_admission_and_stay_open() -> None:
    """A comorbidity did not begin at admission and does not end at discharge."""
    pack, rng = _pack(), np.random.default_rng(7)
    rows: list = []
    for i in range(60):
        rows += sample_patient_diagnosis(
            _spine(24, hid=f"H{i}"), pack, rng, patient_id=f"P{i}", admit_dttm=ADMIT
        )
    chronic = [r for r in rows if r.source_type in (pd._PROBLEM_LIST, pd._MEDICAL_HISTORY)]
    assert chronic
    for row in chronic:
        assert row.start_dttm < ADMIT
        assert row.end_dttm is None


def test_encounter_diagnoses_start_at_admission() -> None:
    pack = _pack()
    rows = sample_patient_diagnosis(
        _spine(24, renal=True), pack, np.random.default_rng(8), admit_dttm=ADMIT
    )
    encounter = [r for r in rows if r.source_type == pd._ENCOUNTER_DX]
    assert encounter
    for row in encounter:
        assert row.start_dttm == ADMIT


def test_acute_failure_resolves_for_a_survivor_but_not_for_a_death() -> None:
    pack = _pack()
    ends = {}
    for outcome in ("alive", "expired"):
        rows = sample_patient_diagnosis(
            _spine(24, renal=True, outcome=outcome),
            pack,
            np.random.default_rng(9),
            admit_dttm=ADMIT,
        )
        ends[outcome] = [r.end_dttm for r in rows if r.source_type == pd._ENCOUNTER_DX]

    assert all(e is not None for e in ends["alive"])
    assert all(e is None for e in ends["expired"])


def test_code_format_is_icd10cm_only() -> None:
    """Unlike hospital_diagnosis, CLIF permits no ICD-9 here."""
    pack, rng = _pack(), np.random.default_rng(10)
    rows: list = []
    for i in range(40):
        rows += sample_patient_diagnosis(
            _spine(24, hid=f"H{i}", renal=True), pack, rng, patient_id=f"P{i}", admit_dttm=ADMIT
        )
    assert {r.diagnosis_code_format for r in rows} == {"ICD10CM"}


def test_both_tables_agree_on_the_code_for_a_flag() -> None:
    """The clinical and billed records must not disagree about what was diagnosed."""
    assert dict(pd._FLAG_CODES).items() <= dict(hd._FLAG_CODES).items()


def test_patient_diagnosis_frame_passes_the_gate() -> None:
    pack, rng = _pack(), np.random.default_rng(11)
    rows: list = []
    for i in range(40):
        rows += sample_patient_diagnosis(
            _spine(24, hid=f"H{i}", renal=True), pack, rng, patient_id=f"P{i}", admit_dttm=ADMIT
        )
    gate.validate(patient_diagnosis_frame(rows), "patient_diagnosis")


def test_empty_frames_are_typed() -> None:
    assert hospital_diagnosis_frame([]).height == 0
    assert patient_diagnosis_frame([]).height == 0
