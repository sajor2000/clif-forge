"""Maturity-tagged CLIF parquet filename layout."""

from __future__ import annotations

from clifforge.generate.filenames import (
    CLIF_FILENAME_VERSION,
    parse_table_from_stem,
    table_parquet_filename,
    table_parquet_stem,
)
from clifforge.reference import loader


def test_beta_and_concept_tables_are_tagged() -> None:
    assert table_parquet_filename("vitals") == "clif_vitals_2.1_beta.parquet"
    assert table_parquet_filename("provider") == "clif_provider_2.1_concept.parquet"
    assert CLIF_FILENAME_VERSION == "2.1"


def test_round_trip_every_dictionary_table() -> None:
    for table in loader.dictionary_tables():
        stem = table_parquet_stem(table)
        assert parse_table_from_stem(stem) == table
        assert stem.startswith("clif_")
        assert f"_{CLIF_FILENAME_VERSION}_" in stem
        maturity = loader.table_maturity(table)
        assert maturity in {"beta", "concept", None}
        if maturity is None:
            assert stem.endswith("_untiered")
        else:
            assert stem.endswith(f"_{maturity}")


def test_truth_and_untagged_stems_are_not_tables() -> None:
    assert parse_table_from_stem("_truth") is None
    assert parse_table_from_stem("clif_vitals") is None  # missing maturity tag
