"""Column-level coverage of the generated dataset against the canonical CLIF 2.1 DDL.

The conformance gate already rejects a table that *drops a required column* or
carries a non-canonical one. What it cannot see is the quieter failure: an
optional canonical column that no generator ever populates. Those are invisible
at runtime — the frame validates, the pipeline is green, and a consumer only
finds out when they select a column that is not there.

So this module makes the omissions explicit. ``KNOWN_GAPS`` lists every canonical
column the generators do not emit, with the reason, and the test fails on any gap
that is *not* listed as well as any listed gap that has since been filled. New
unemitted columns cannot accumulate silently, and filling one forces the list to
be updated.

The distinction that matters when reading the reasons below: a column omitted
because the fit stage has no distribution for it is a *faithful omission* (R15) —
fabricating one would be worse than leaving it out. A column omitted because it
is simply not modelled yet is a *gap*, and is labelled as such.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from clifforge.demo import demo_pack
from clifforge.generate.orchestrator import generate_dataset
from clifforge.reference import loader

#: table -> {column: why it is not emitted}.
KNOWN_GAPS: dict[str, dict[str, str]] = {
    "crrt_therapy": {
        "dialysis_machine_name": "device make/model free text; no vendored source to draw from",
    },
    "hospitalization": {
        # Geography is deliberately absent rather than merely unmodelled. Synthetic
        # codes would be either meaningless or, if drawn from real ones, a
        # re-identification surface on a dataset whose whole point is carrying none.
        # place_based_index carries the neighbourhood-deprivation signal instead.
        "zipcode_nine_digit": "deliberate: no synthetic geography (see place_based_index)",
        "zipcode_five_digit": "deliberate: no synthetic geography (see place_based_index)",
        "census_block_code": "deliberate: no synthetic geography (see place_based_index)",
        "census_block_group_code": "deliberate: no synthetic geography (see place_based_index)",
        "census_tract": "deliberate: no synthetic geography (see place_based_index)",
        "state_code": "deliberate: no synthetic geography (see place_based_index)",
        "county_code": "deliberate: no synthetic geography (see place_based_index)",
        "fips_version": "deliberate: no synthetic geography (see place_based_index)",
    },
    "labs": {
        # A wrong LOINC code is worse than an absent one: it resolves, silently,
        # to a different analyte than the row actually reports.
        "lab_loinc_code": "no vendored LOINC crosswalk; an invented code would resolve wrongly",
        "lab_specimen_name": "no consortium specimen vocabulary; not inventing one",
        "lab_specimen_category": "no consortium specimen vocabulary; not inventing one",
    },
    "microbiology_culture": {
        "lab_loinc_code": "no vendored LOINC crosswalk; an invented code would resolve wrongly",
    },
    "respiratory_support": {
        "vent_brand_name": "ventilator make/model free text; no consortium catalog",
        # Off-matrix set fields stay null until R10 DEVICE_SET_FIELDS expands to
        # modes that use them (e.g. Pressure Control → pressure_control_set).
        "pressure_control_set": "R10: not in DEVICE_SET_FIELDS for currently emitted modes",
        "flow_rate_set": "R10: not in DEVICE_SET_FIELDS for currently emitted modes",
        "peak_inspiratory_pressure_set": "R10: not in DEVICE_SET_FIELDS for currently emitted modes",
        "inspiratory_time_set": "R10: not in DEVICE_SET_FIELDS for currently emitted modes",
    },
}

#: The nine tables added in the canonical re-foundation are column-complete, and
#: are named here so a future edit cannot quietly reintroduce a gap in one.
FULLY_COVERED_TABLES = (
    "clinical_trial",
    "hospital_diagnosis",
    "intake_output",
    "medication_orders",
    "microbiology_nonculture",
    "microbiology_susceptibility",
    "patient_diagnosis",
    "patient_procedures",
    "place_based_index",
)


@pytest.fixture(scope="module")
def emitted() -> dict[str, set[str]]:
    """Columns actually emitted per table, from one generated dataset.

    Module-scoped so the parametrized checks below share a single generation
    rather than rebuilding a dataset per table; it therefore builds its own pack
    instead of taking the function-scoped ``pack`` fixture.
    """
    ds = generate_dataset(demo_pack(), n_patients=12, seed=1)
    return {name: set(frame.columns) for name, frame in ds.tables.items()}


def test_every_canonical_table_is_emitted(emitted: dict[str, set[str]]) -> None:
    assert set(emitted) == set(loader.dictionary_tables())


@pytest.mark.parametrize("table", sorted(loader.dictionary_tables()))
def test_no_unexpected_missing_columns(table: str, emitted: dict[str, set[str]]) -> None:
    """Every canonical column is either emitted or listed in KNOWN_GAPS with a reason."""
    canonical = {c["name"] for c in loader.table_columns(table)}
    missing = canonical - emitted[table]
    unexplained = missing - set(KNOWN_GAPS.get(table, {}))
    assert not unexplained, (
        f"{table} silently omits canonical column(s) {sorted(unexplained)} — "
        "emit them, or record them in KNOWN_GAPS with the reason."
    )


@pytest.mark.parametrize("table", sorted(KNOWN_GAPS))
def test_known_gaps_are_still_gaps(table: str, emitted: dict[str, set[str]]) -> None:
    """A gap that has been filled must be removed from KNOWN_GAPS."""
    filled = set(KNOWN_GAPS[table]) & emitted[table]
    assert not filled, f"{table} now emits {sorted(filled)} — delete it from KNOWN_GAPS."


@pytest.mark.parametrize("table", FULLY_COVERED_TABLES)
def test_canonically_complete_tables_stay_complete(
    table: str, emitted: dict[str, set[str]]
) -> None:
    canonical = {c["name"] for c in loader.table_columns(table)}
    assert canonical == emitted[table]


@pytest.mark.parametrize("table", sorted(loader.dictionary_tables()))
def test_no_table_invents_a_column(table: str, emitted: dict[str, set[str]]) -> None:
    """Nothing outside the canonical DDL may appear in an emitted CLIF table."""
    canonical = {c["name"] for c in loader.table_columns(table)}
    assert not emitted[table] - canonical


def test_known_gaps_reference_real_columns() -> None:
    """KNOWN_GAPS cannot list a column the canonical DDL does not define."""
    for table, gaps in KNOWN_GAPS.items():
        canonical = {c["name"] for c in loader.table_columns(table)}
        assert set(gaps) <= canonical, f"{table}: {sorted(set(gaps) - canonical)} are not canonical"


def test_provenance_documents_exactly_the_canonical_tables() -> None:
    """Every emitted table must declare how its content was produced (R4, R14).

    A table that reaches a consumer without a provenance row leaves them unable to
    tell fitted structure from a documented prior, which is the whole point of the
    document.
    """
    text = pathlib.Path("PROVENANCE.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `(\w+)`", text, re.M))
    assert documented == set(loader.dictionary_tables())


def test_every_gap_has_a_reason() -> None:
    for table, gaps in KNOWN_GAPS.items():
        for column, reason in gaps.items():
            assert reason.strip(), f"{table}.{column} is listed with no reason"
