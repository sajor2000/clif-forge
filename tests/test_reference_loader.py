"""Tests for the vendored CLIF 2.1.0 reference-data loader (U2).

These lock the contract the schema layer (U3) depends on: exact, case-sensitive
mCIDE category lists; numeric outlier bounds; and — critically — that a *missing*
table/field RAISES rather than returning an empty result (R4, R9).
"""

from __future__ import annotations

import math

import pytest

from clifforge import reference
from clifforge.reference import ReferenceDataError


def test_sex_categories_exact_and_case_sensitive() -> None:
    assert reference.categories("patient", "sex_category") == ["Male", "Female", "Unknown"]


def test_position_categories_exact() -> None:
    assert reference.categories("position", "position_category") == ["prone", "not_prone"]


def test_device_categories_preserve_case_and_spacing() -> None:
    devices = reference.categories("respiratory_support", "device_category")
    # Case- and space-sensitive: "High Flow NC" / "Room Air" must survive verbatim.
    assert "IMV" in devices
    assert "High Flow NC" in devices
    assert "Room Air" in devices
    # Lower-cased variants are NOT present — matching is exact downstream.
    assert "imv" not in devices


def test_lab_categories_nonempty_and_expected_member() -> None:
    labs = reference.categories("labs", "lab_category")
    assert len(labs) == 52
    assert "albumin" in labs
    # No blank rows leaked in.
    assert all(v.strip() == v and v for v in labs)


def test_language_categories_exclude_footnote_rows() -> None:
    """Upstream language CSV ends with a ``*Descriptions...`` prose row — not a value."""
    langs = reference.categories("patient", "language_category")
    assert "English" in langs
    assert "Unknown or NA" in langs
    assert all(not v.startswith("*") for v in langs)
    assert len(langs) == 45


def test_vitals_categories_present() -> None:
    vitals = reference.categories("vitals", "vital_category")
    assert "heart_rate" in vitals
    assert "spo2" in vitals


def test_missing_table_raises() -> None:
    with pytest.raises(ReferenceDataError):
        reference.categories("no_such_table", "whatever")


def test_missing_field_raises_not_empty() -> None:
    with pytest.raises(ReferenceDataError):
        reference.categories("patient", "no_such_field")


def test_bounds_return_numeric_tuple() -> None:
    lower, upper = reference.bounds("vitals", "heart_rate")
    assert (lower, upper) == (0.0, 300.0)
    assert isinstance(lower, float)
    assert isinstance(upper, float)


def test_bounds_float_valued_thresholds() -> None:
    assert reference.bounds("respiratory_support", "fio2_set") == (0.21, 1.0)


def test_bounds_labs() -> None:
    assert reference.bounds("labs", "albumin") == (0.0, 15.0)


def test_bounds_missing_table_raises() -> None:
    with pytest.raises(ReferenceDataError):
        reference.bounds("patient", "sex_category")


def test_bounds_missing_field_raises() -> None:
    with pytest.raises(ReferenceDataError):
        reference.bounds("vitals", "no_such_vital")


def test_bounds_rejects_nonstandard_ecmo_schema() -> None:
    # ecmo_mcs uses a multi-column ranged-text schema, not lower/upper limits.
    with pytest.raises(ReferenceDataError):
        reference.bounds("ecmo_mcs", "Impella_2.5")


def test_tables_and_fields_listing() -> None:
    tables = reference.tables()
    assert "patient" in tables
    assert "vitals" in tables
    assert "sex_category" in reference.mcide_fields("patient")


def test_mcide_fields_unknown_table_raises() -> None:
    with pytest.raises(ReferenceDataError):
        reference.mcide_fields("no_such_table")


def test_provenance_pins_version_and_commit() -> None:
    prov = reference.provenance()
    assert prov["clif_version"] == "2.1.0"
    assert prov["source_ref"] == "v2.1.0"
    assert len(prov["source_commit"]) == 40


def test_bounds_are_finite_for_standard_files() -> None:
    lower, upper = reference.bounds("crrt_therapy", "blood_flow_rate")
    assert math.isfinite(lower) and math.isfinite(upper)
