"""Unit tests for the remaining canonical tables added in the re-foundation.

Covers ``intake_output``, ``microbiology_nonculture``, ``patient_procedures``,
``place_based_index`` and ``clinical_trial``. Each test targets the claim its
generator's docstring makes — the oliguria coupling, the ADI/SVI correlation, the
consent-before-randomization ordering — rather than just re-asserting that rows
come out.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import clinical_trial as ct
from clifforge.generate.tables import intake_output as io
from clifforge.generate.tables import microbiology_nonculture as mnc
from clifforge.generate.tables import place_based_index as pbi
from clifforge.generate.tables.clinical_trial import clinical_trial_frame, sample_clinical_trial
from clifforge.generate.tables.intake_output import intake_output_frame, sample_intake_output
from clifforge.generate.tables.microbiology_nonculture import (
    microbiology_nonculture_frame,
    sample_microbiology_nonculture,
)
from clifforge.generate.tables.patient_procedures import (
    patient_procedures_frame,
    sample_patient_procedures,
)
from clifforge.generate.tables.place_based_index import (
    place_based_index_frame,
    sample_place_based_index,
)
from clifforge.reference import categories, loader

ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(
    n: int, hid: str = "H0", *, level: int = 3, renal: bool = False, cv: bool = False
) -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[level] * n,
        resp_flag=[False] * n,
        cv_flag=[cv] * n,
        renal_flag=[renal] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


# --- intake_output ----------------------------------------------------------- #
def test_intake_output_is_deterministic() -> None:
    pack = _pack()
    a = sample_intake_output(_spine(24), pack, np.random.default_rng(1), admit_dttm=ADMIT)
    b = sample_intake_output(_spine(24), pack, np.random.default_rng(1), admit_dttm=ADMIT)
    assert a == b


def test_every_charting_point_records_both_intake_and_output() -> None:
    rows = sample_intake_output(_spine(24), _pack(), np.random.default_rng(2), admit_dttm=ADMIT)
    assert rows
    flags = [r.in_out_flag for r in rows]
    assert flags.count(io._INTAKE) == flags.count(io._OUTPUT)


def test_renal_failure_charts_oliguria() -> None:
    """Urine output must fall when the spine says the kidneys failed.

    Without this coupling the fluid record would contradict the creatinine in
    ``labs`` and the ``crrt_therapy`` rows generated from the same flag.
    """
    pack = _pack()
    normal = sample_intake_output(
        _spine(48, renal=False), pack, np.random.default_rng(3), admit_dttm=ADMIT
    )
    oliguric = sample_intake_output(
        _spine(48, renal=True), pack, np.random.default_rng(3), admit_dttm=ADMIT
    )

    def mean_urine(rows: list) -> float:
        urine = [r.amount for r in rows if r.fluid_name == io._URINE]
        return sum(urine) / len(urine)

    assert mean_urine(oliguric) < mean_urine(normal) / 2


def test_resuscitation_raises_intake() -> None:
    pack = _pack()
    baseline = sample_intake_output(
        _spine(48, cv=False), pack, np.random.default_rng(4), admit_dttm=ADMIT
    )
    shocked = sample_intake_output(
        _spine(48, cv=True), pack, np.random.default_rng(4), admit_dttm=ADMIT
    )

    def mean_intake(rows: list) -> float:
        intake = [r.amount for r in rows if r.in_out_flag == io._INTAKE]
        return sum(intake) / len(intake)

    assert mean_intake(shocked) > mean_intake(baseline)


def test_ward_time_is_not_charted() -> None:
    rows = sample_intake_output(
        _spine(24, level=1), _pack(), np.random.default_rng(5), admit_dttm=ADMIT
    )
    assert rows == []


def test_intake_output_frame_passes_the_gate() -> None:
    pack, rng = _pack(), np.random.default_rng(6)
    rows: list = []
    for i in range(20):
        rows += sample_intake_output(_spine(24, hid=f"H{i}"), pack, rng, admit_dttm=ADMIT)
    gate.validate(intake_output_frame(rows), "intake_output")


# --- microbiology_nonculture ------------------------------------------------- #
def _nonculture_cohort(n: int, seed: int = 0) -> list:
    pack, rng = _pack(), np.random.default_rng(seed)
    events: list = []
    for i in range(n):
        events += sample_microbiology_nonculture(
            _spine(72, hid=f"H{i}"), pack, rng, patient_id=f"P{i}", admit_dttm=ADMIT
        )
    return events


def test_nonculture_is_deterministic() -> None:
    assert _nonculture_cohort(30, seed=7) == _nonculture_cohort(30, seed=7)


def test_nonculture_categories_are_exact_mcide_members() -> None:
    events = _nonculture_cohort(80, seed=8)
    assert events
    for field, values in (
        ("organism_category", {e.organism_category for e in events}),
        ("fluid_category", {e.fluid_category for e in events}),
        ("result_category", {e.result_category for e in events}),
    ):
        assert values <= set(categories("microbiology_nonculture", field))


def test_organism_group_comes_from_the_vendored_crosswalk() -> None:
    """sars_cov2 rolls up to viral_other — not the identity mapping one might assume."""
    crosswalk = loader.crosswalk("microbiology_nonculture", "organism_category", "organism_group")
    assert crosswalk["sars_cov2"] == "viral_other"
    for event in _nonculture_cohort(80, seed=9):
        assert event.organism_group == crosswalk[event.organism_category]


def test_specimen_matches_the_target() -> None:
    """A C. difficile PCR is run on stool, not a nasopharyngeal swab."""
    assert mnc._TARGET_SPECIMEN["clostridium_difficile"][0] == "feces_stool"
    for event in _nonculture_cohort(80, seed=10):
        expected_fluid, _order_name = mnc._TARGET_SPECIMEN[event.organism_category]
        assert event.fluid_category == expected_fluid


def test_nonculture_timestamps_are_ordered() -> None:
    for event in _nonculture_cohort(80, seed=11):
        assert event.order_dttm <= event.collect_dttm <= event.result_dttm


def test_qualitative_assay_reports_no_reference_range() -> None:
    frame = microbiology_nonculture_frame(_nonculture_cohort(60, seed=12))
    for column in ("reference_low", "reference_high", "result_units"):
        assert frame[column].is_null().all()


def test_nonculture_frame_passes_the_gate() -> None:
    gate.validate(
        microbiology_nonculture_frame(_nonculture_cohort(40, seed=13)), "microbiology_nonculture"
    )


# --- patient_procedures ------------------------------------------------------ #
def _procedure_cohort(n: int, seed: int = 0, level: int = 4) -> list:
    pack, rng = _pack(), np.random.default_rng(seed)
    rows: list = []
    for i in range(n):
        rows += sample_patient_procedures(
            _spine(24, hid=f"H{i}", level=level), pack, rng, admit_dttm=ADMIT
        )
    return rows


def test_procedures_use_real_vendored_cpt_codes() -> None:
    rows = _procedure_cohort(400, seed=14)
    assert rows, "no procedures generated"
    valid = {c["procedure_code"] for c in loader.code_list("patient_procedures")}
    assert {r.procedure_code for r in rows} <= valid
    assert {r.procedure_code_format for r in rows} == {"CPT"}


def test_procedures_are_not_billed_to_low_acuity_stays() -> None:
    """The vendored list is transplant and thoracic surgery, not routine ICU care."""
    assert _procedure_cohort(400, seed=15, level=1) == []


def test_procedures_are_rare() -> None:
    rows = _procedure_cohort(500, seed=16)
    assert len(rows) < 500 * 0.15


def test_billing_lands_after_admission() -> None:
    for row in _procedure_cohort(400, seed=17):
        assert row.procedure_billed_dttm > ADMIT


def test_procedures_frame_passes_the_gate() -> None:
    gate.validate(patient_procedures_frame(_procedure_cohort(300, seed=18)), "patient_procedures")


# --- place_based_index ------------------------------------------------------- #
def _index_cohort(n: int, seed: int = 0) -> list:
    pack, rng = _pack(), np.random.default_rng(seed)
    rows: list = []
    for i in range(n):
        rows += sample_place_based_index(_spine(24, hid=f"H{i}"), pack, rng, admit_dttm=ADMIT)
    return rows


def test_each_hospitalization_gets_both_indices() -> None:
    rows = _index_cohort(50, seed=19)
    assert len(rows) == 100
    assert {r.index_name for r in rows} == {pbi._ADI_NAME, pbi._SVI_NAME}


def test_indices_stay_on_their_published_scales() -> None:
    """ADI national rank is a 1-100 percentile; SVI is a 0-1 proportion."""
    for row in _index_cohort(200, seed=20):
        if row.index_name == pbi._ADI_NAME:
            assert 1.0 <= row.index_value <= 100.0
        else:
            assert 0.0 <= row.index_value <= 1.0


def test_the_two_indices_describe_the_same_neighbourhood() -> None:
    """A deprived area must score high on both, or disparities analyses break."""
    frame = place_based_index_frame(_index_cohort(300, seed=21))
    wide = frame.pivot(values="index_value", index="hospitalization_id", on="index_name")
    correlation = wide.select(pl.corr(pbi._ADI_NAME, pbi._SVI_NAME)).item()
    assert correlation > 0.8


def test_place_based_index_frame_passes_the_gate() -> None:
    gate.validate(place_based_index_frame(_index_cohort(40, seed=22)), "place_based_index")


# --- clinical_trial ---------------------------------------------------------- #
def _trial_cohort(n: int, seed: int = 0, level: int = 4) -> list:
    pack, rng = _pack(), np.random.default_rng(seed)
    rows: list = []
    for i in range(n):
        rows += sample_clinical_trial(
            _spine(48, hid=f"H{i}", level=level), pack, rng, admit_dttm=ADMIT
        )
    return rows


def test_consent_precedes_randomization() -> None:
    rows = _trial_cohort(400, seed=23)
    assert rows, "no enrolments generated"
    for row in rows:
        assert row.consent_dttm < row.randomized_dttm


def test_withdrawal_follows_randomization_when_it_happens() -> None:
    withdrawn = [r for r in _trial_cohort(600, seed=24) if r.withdrawal_dttm is not None]
    assert withdrawn, "no withdrawals generated"
    for row in withdrawn:
        assert row.withdrawal_dttm > row.randomized_dttm


def test_most_enrolled_patients_do_not_withdraw() -> None:
    rows = _trial_cohort(600, seed=25)
    withdrawn = sum(r.withdrawal_dttm is not None for r in rows)
    assert withdrawn / len(rows) < 0.25


def test_only_high_acuity_stays_are_enrolled() -> None:
    assert _trial_cohort(400, seed=26, level=1) == []


def test_arms_belong_to_the_trial_they_are_recorded_against() -> None:
    for row in _trial_cohort(400, seed=27):
        name, arms = ct._TRIALS[row.trial_id]
        assert row.trial_name == name
        assert row.arm_id in arms


def test_trial_ids_are_not_real_nct_numbers() -> None:
    """A synthetic cohort must not appear to have enrolled in a real study."""
    for trial_id in ct._TRIALS:
        assert not trial_id.upper().startswith("NCT")


def test_clinical_trial_frame_passes_the_gate() -> None:
    gate.validate(clinical_trial_frame(_trial_cohort(300, seed=28)), "clinical_trial")


@pytest.mark.parametrize(
    "frame_fn",
    [
        intake_output_frame,
        microbiology_nonculture_frame,
        patient_procedures_frame,
        place_based_index_frame,
        clinical_trial_frame,
    ],
)
def test_empty_input_yields_an_empty_typed_frame(frame_fn) -> None:
    frame = frame_fn([])
    assert frame.height == 0
    assert "hospitalization_id" in frame.columns
