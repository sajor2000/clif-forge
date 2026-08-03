"""Unit tests for the derived microbiology_susceptibility generator (R5/R22)."""

from __future__ import annotations

import numpy as np
import pytest

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import microbiology_susceptibility as ms
from clifforge.generate.tables.microbiology_culture import (
    sample_microbiology_culture,
)
from clifforge.generate.tables.microbiology_susceptibility import (
    microbiology_susceptibility_frame,
    sample_microbiology_susceptibility,
)
from clifforge.reference import categories

_PARENT = "microbiology_culture"


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(n: int, hid: str = "H0") -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[3] * n,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def _cohort(n_stays: int, seed: int = 0, n_int: int = 200) -> tuple[list, list]:
    """Generate cultures and their derived susceptibilities across many stays."""
    pack, rng = _pack(), np.random.default_rng(seed)
    cultures, susceptibilities = [], []
    for i in range(n_stays):
        spine = _spine(n_int, hid=f"H{i}")
        events = sample_microbiology_culture(spine, pack, rng, patient_id=f"P{i}")
        cultures += events
        susceptibilities += sample_microbiology_susceptibility(
            {_PARENT: events}, pack, rng, hospitalization_id=f"H{i}"
        )
    return cultures, susceptibilities


def test_is_deterministic() -> None:
    a = _cohort(60, seed=11)[1]
    b = _cohort(60, seed=11)[1]
    assert a == b


def test_a_culture_that_grew_nothing_yields_no_panel() -> None:
    pack, rng = _pack(), np.random.default_rng(3)
    events = sample_microbiology_culture(_spine(200), pack, rng)
    no_growth = [e for e in events if e.organism_category is None]
    rows = sample_microbiology_susceptibility({_PARENT: no_growth}, pack, rng)
    assert rows == []


def test_every_isolate_that_grew_is_paneled() -> None:
    cultures, rows = _cohort(400, seed=5)
    grew = {c.organism_id for c in cultures if c.organism_category is not None}
    assert grew, "test cohort produced no positive cultures"
    assert {r.organism_id for r in rows} == grew


def test_organism_ids_are_never_orphaned() -> None:
    """Every susceptibility must join back to a culture that minted its id."""
    cultures, rows = _cohort(400, seed=7)
    minted = {c.organism_id for c in cultures if c.organism_id is not None}
    assert {r.organism_id for r in rows} <= minted


def test_categories_are_exact_mcide_members() -> None:
    _cultures, rows = _cohort(300, seed=9)
    assert rows, "test cohort produced no susceptibility rows"
    permitted_abx = set(categories("microbiology_susceptibility", "antimicrobial_category"))
    permitted_susc = set(categories("microbiology_susceptibility", "susceptibility_category"))
    assert {r.antimicrobial_category for r in rows} <= permitted_abx
    assert {r.susceptibility_category for r in rows} <= permitted_susc


def test_panel_matches_the_organism() -> None:
    """A gram-negative is not tested against the anti-staphylococcal panel."""
    cultures, rows = _cohort(400, seed=13)
    organism_of = {c.organism_id: c.organism_category for c in cultures}
    for row in rows:
        expected = ms._PANEL[organism_of[row.organism_id]]
        assert row.antimicrobial_category in expected


def test_one_isolate_is_tested_against_each_drug_once() -> None:
    cultures, rows = _cohort(400, seed=17)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.organism_id, row.antimicrobial_category)
        assert key not in seen
        seen.add(key)
    assert len(seen) == len(rows)
    assert cultures  # sanity: the cohort was non-empty


def test_resistance_differs_by_drug_as_the_priors_specify() -> None:
    """Oxacillin resistance (MRSA) must run far above vancomycin resistance (VRSA).

    Flattening these to one rate is the failure this guards: an antibiogram in
    which every drug resists equally cannot be used to study resistance at all.
    """
    rng = np.random.default_rng(0)
    n = 4000
    ox = sum(
        rng.random() < ms._resistance_prob("staphylococcus_aureus", "oxacillin") for _ in range(n)
    )
    vanc = sum(
        rng.random() < ms._resistance_prob("staphylococcus_aureus", "vancomycin") for _ in range(n)
    )
    assert ox > vanc * 5


@pytest.mark.parametrize(
    ("organism", "drug"),
    [("staphylococcus_aureus", "oxacillin"), ("escherichia_coli", "ampicillin")],
)
def test_overrides_beat_the_organism_baseline(organism: str, drug: str) -> None:
    assert ms._resistance_prob(organism, drug) > ms._BASELINE_RESISTANCE[organism]


def test_interpretation_string_agrees_with_the_category() -> None:
    _cultures, rows = _cohort(300, seed=21)
    frame = microbiology_susceptibility_frame(rows)
    for category, name in zip(
        frame["susceptibility_category"], frame["susceptibility_name"], strict=True
    ):
        assert name == ms._INTERPRETATION[category]


def test_frame_passes_the_conformance_gate() -> None:
    _cultures, rows = _cohort(200, seed=23)
    gate.validate(microbiology_susceptibility_frame(rows), "microbiology_susceptibility")


def test_empty_input_yields_an_empty_typed_frame() -> None:
    frame = microbiology_susceptibility_frame([])
    assert frame.height == 0
    assert "organism_id" in frame.columns


def test_module_exports() -> None:
    assert set(ms.__all__) == {
        "SusceptibilityRow",
        "microbiology_susceptibility_frame",
        "sample_microbiology_susceptibility",
    }
