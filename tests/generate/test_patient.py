"""Unit tests for the Tier 1 patient generator (U7, R5/R6/R8).

Driven by a small hand-built pack (no real data, no run_fit) so the generator's
contract is checked in isolation: exact mCIDE categories, marginals tracking the
pack, unique ids, conformance-gate pass, and seed reproducibility.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.tables import patient
from clifforge.generate.tables.patient import PatientRecord, patient_frame, sample_patient
from clifforge.reference import categories

# Marginals whose keys are exact mCIDE members (a strict subset is fine — the
# gate only requires emitted values be members, not that every member appears).
_RACE = {"White": 0.6, "Black or African American": 0.25, "Asian": 0.1, "Unknown": 0.05}
_ETHNICITY = {"Hispanic": 0.1, "Non-Hispanic": 0.7, "Unknown": 0.2}
_SEX = {"Female": 0.52, "Male": 0.48}


def _patient_pack() -> ParamPack:
    params = {
        "race_category_marginal": _RACE,
        "ethnicity_category_marginal": _ETHNICITY,
        "sex_category_marginal": _SEX,
    }
    return ParamPack(
        manifest={},
        tables={"patient": {"n_records": 1000, "fitted": True, "params": params}},
    )


def _cohort(pack: ParamPack, rng: np.random.Generator, n: int) -> list[PatientRecord]:
    return [sample_patient(pack, rng, patient_id=f"P{i}") for i in range(n)]


def test_sample_patient_is_deterministic_under_fixed_seed() -> None:
    pack = _patient_pack()
    a = sample_patient(pack, np.random.default_rng(2024))
    b = sample_patient(pack, np.random.default_rng(2024))
    assert a == b


def test_frame_reproducible_byte_for_byte() -> None:
    pack = _patient_pack()
    a = patient_frame(_cohort(pack, np.random.default_rng(9), 50))
    b = patient_frame(_cohort(pack, np.random.default_rng(9), 50))
    assert a.equals(b)


def test_all_categories_are_exact_mcide_members() -> None:
    pack = _patient_pack()
    rng = np.random.default_rng(1)
    race_ok = set(categories("patient", "race_category"))
    eth_ok = set(categories("patient", "ethnicity_category"))
    sex_ok = set(categories("patient", "sex_category"))
    lang_ok = set(categories("patient", "language_category"))
    for _ in range(500):
        r = sample_patient(pack, rng)
        assert r.race_category in race_ok
        assert r.ethnicity_category in eth_ok
        assert r.sex_category in sex_ok
        assert r.language_category in lang_ok


def test_names_echo_categories() -> None:
    pack = _patient_pack()
    rng = np.random.default_rng(4)
    for _ in range(200):
        r = sample_patient(pack, rng)
        assert r.race_name == r.race_category
        assert r.ethnicity_name == r.ethnicity_category
        assert r.sex_name == r.sex_category
        assert r.language_name == r.language_category


def test_language_prior_is_english_skewed_not_uniform() -> None:
    """Uniform over ~45 languages would put English at ~2% — worse than omitting."""
    pack = _patient_pack()
    rng = np.random.default_rng(0)
    n = 2000
    english = sum(
        1 for _ in range(n) if sample_patient(pack, rng).language_category == "English"
    )
    assert english / n > 0.7


def test_category_marginals_match_pack() -> None:
    pack = _patient_pack()
    rng = np.random.default_rng(31)
    records = _cohort(pack, rng, 6000)
    races = [r.race_category for r in records]
    for cat, prob in _RACE.items():
        assert abs(races.count(cat) / len(races) - prob) < 0.03
    sexes = [r.sex_category for r in records]
    assert abs(sexes.count("Female") / len(sexes) - 0.52) < 0.03


def test_ids_are_unique_and_frame_passes_gate() -> None:
    pack = _patient_pack()
    frame = patient_frame(_cohort(pack, np.random.default_rng(0), 300))
    assert frame["patient_id"].n_unique() == 300
    # The conformance gate raises on any pandera violation; a clean return is pass.
    report = gate.validate(frame, "patient", run_secondary=False)
    assert report.pandera_passed


def test_empty_frame_has_correct_schema() -> None:
    frame = patient_frame([])
    assert frame.height == 0
    assert frame.columns == [
        "patient_id",
        "race_category",
        "race_name",
        "ethnicity_category",
        "ethnicity_name",
        "sex_category",
        "sex_name",
        "language_category",
        "language_name",
    ]
    assert all(dt == pl.String for dt in frame.dtypes)


def test_missing_patient_block_raises() -> None:
    empty = ParamPack(manifest={}, tables={})
    try:
        sample_patient(empty, np.random.default_rng(0))
    except ValueError as exc:
        assert "patient" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a pack with no patient block")


def test_module_exports() -> None:
    assert set(patient.__all__) == {"PatientRecord", "patient_frame", "sample_patient"}
