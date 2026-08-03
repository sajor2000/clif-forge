"""Maturity-tagged CLIF parquet filename layout."""

from __future__ import annotations

import pytest

from clifforge.generate.filenames import (
    CLIF_FILENAME_VERSION,
    DELIVERABLE_MATURITIES,
    deliverable_tables,
    is_deliverable_table,
    parse_table_from_stem,
    table_parquet_filename,
    table_parquet_stem,
)
from clifforge.reference import loader


def test_beta_and_concept_tables_are_tagged() -> None:
    assert table_parquet_filename("vitals") == "clif_vitals_2.1_beta.parquet"
    assert table_parquet_filename("provider") == "clif_provider_2.1_concept.parquet"
    assert CLIF_FILENAME_VERSION == "2.1"


def test_deliverable_tables_are_exactly_beta_and_concept() -> None:
    delivered = deliverable_tables()
    assert len(delivered) == 25
    assert all(is_deliverable_table(t) for t in delivered)
    omitted = set(loader.dictionary_tables()) - set(delivered)
    assert omitted == {"clinical_trial", "patient_diagnosis", "place_based_index"}
    for table in delivered:
        stem = table_parquet_stem(table)
        maturity = loader.table_maturity(table)
        assert maturity in DELIVERABLE_MATURITIES
        assert stem.endswith(f"_{maturity}")
        assert table_parquet_filename(table) == f"{stem}.parquet"


def test_untiered_tables_cannot_be_written_as_deliverable() -> None:
    for table in ("clinical_trial", "patient_diagnosis", "place_based_index"):
        assert not is_deliverable_table(table)
        assert table_parquet_stem(table).endswith("_untiered")
        with pytest.raises(ValueError, match="must not be written"):
            table_parquet_filename(table)


def test_round_trip_every_dictionary_table_stem() -> None:
    for table in loader.dictionary_tables():
        stem = table_parquet_stem(table)
        assert parse_table_from_stem(stem) == table
        assert stem.startswith("clif_")
        assert f"_{CLIF_FILENAME_VERSION}_" in stem


def test_truth_and_untagged_stems_are_not_tables() -> None:
    assert parse_table_from_stem("_truth") is None
    assert parse_table_from_stem("clif_vitals") is None  # missing maturity tag
