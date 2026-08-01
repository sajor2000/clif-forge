"""Cross-check the vendored dictionary against CLIF's *other* canonical artifact.

``scripts/vendor_dictionary.py`` vendors from the consortium's canonical DDL
(``CLIF/ddl/2.1/CLIF2.1_MYSQL_ddl.sql``). ``clifpy`` independently ships
``schemas/2.1/*.yaml`` for 18 of the 28 tables. Both are maintained by the
consortium, so where they overlap they should agree — and when they stop agreeing
we want a red test, not a silent choice.

This suite exists because the project previously vendored from the CLIF website's
Quarto prose page, which had drifted far enough from the DDL to describe four
tables with the wrong columns entirely. Nothing compared the sources, so the drift
was invisible until someone read the DDL.

Known, deliberate divergences are listed in :data:`ACCEPTED_DIVERGENCES` with the
reason. Anything else fails.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from clifforge.reference import loader

_SCHEMA_DIR = pathlib.Path(__import__("clifpy").__file__).parent / "schemas" / "2.1"

#: table -> (DDL-only columns, clifpy-only columns). Everywhere except ecmo_mcs the
#: DDL is a strict superset, so the clifpy-only side is empty.
ACCEPTED_DIVERGENCES: dict[str, tuple[set[str], set[str]]] = {
    # A genuine two-way conflict, not just a coverage gap: clifpy 2.1 models ECMO
    # with a generic metric/rate pair while the DDL models an explicit control
    # parameter plus configuration. The DDL is primary, so we follow it and pin the
    # conflict here rather than letting it pass unnoticed.
    "ecmo_mcs": (
        {
            "ecmo_configuration_category",
            "control_parameter_name",
            "control_parameter_category",
            "control_parameter_value",
            "sweep_set",
            "fdO2_set",
        },
        {"device_metric_name", "device_rate", "sweep", "fdO2"},
    ),
    "microbiology_culture": ({"organism_name", "lab_loinc_code"}, set()),
    "microbiology_nonculture": (
        {"reference_low", "reference_high", "result_units", "lab_loinc_code"},
        set(),
    ),
}

#: SQL spells "floating point" several ways. The DDL prefers FLOAT and clifpy
#: prefers DOUBLE for the same measurement columns; both resolve to a float column,
#: so comparing the raw tokens would flag a difference that does not exist.
_EQUIVALENT_TYPES = ({"FLOAT", "DOUBLE", "NUMERIC"}, {"INT", "INTEGER", "BIGINT"})


def _types_agree(a: str, b: str) -> bool:
    return a == b or any(a in family and b in family for family in _EQUIVALENT_TYPES)


def _clifpy_tables() -> list[str]:
    return sorted(p.stem.removesuffix("_schema") for p in _SCHEMA_DIR.glob("*_schema.yaml"))


def _clifpy_columns(table: str) -> list[str]:
    spec = yaml.safe_load((_SCHEMA_DIR / f"{table}_schema.yaml").read_text(encoding="utf-8"))
    return [c["name"] for c in spec["columns"]]


CLIFPY_TABLES = _clifpy_tables()


def test_clifpy_schemas_are_available() -> None:
    """Guard the fixture itself: an empty glob would make every check vacuous."""
    assert len(CLIFPY_TABLES) >= 18


@pytest.mark.parametrize("table", CLIFPY_TABLES)
def test_clifpy_table_exists_in_vendored_dictionary(table: str) -> None:
    assert table in loader.dictionary_tables()


@pytest.mark.parametrize("table", CLIFPY_TABLES)
def test_no_unexpected_column_divergence(table: str) -> None:
    """Column *sets* must match, modulo the accepted extras. Order is not canonical.

    The DDL and clifpy order ``patient``, ``patient_procedures``, and
    ``respiratory_support`` differently while listing identical columns, so
    ordering carries no meaning and is not asserted.
    """
    ours = {c["name"] for c in loader.table_columns(table)}
    theirs = set(_clifpy_columns(table))
    ddl_only, clifpy_only = ACCEPTED_DIVERGENCES.get(table, (set(), set()))

    assert (theirs - ours) == clifpy_only, f"{table}: clifpy has columns the DDL vendoring missed"
    assert (ours - theirs) == ddl_only, f"{table}: unexpected DDL-only columns"


@pytest.mark.parametrize("table", CLIFPY_TABLES)
def test_shared_columns_agree_on_type(table: str) -> None:
    spec = yaml.safe_load((_SCHEMA_DIR / f"{table}_schema.yaml").read_text(encoding="utf-8"))
    theirs = {c["name"]: c["data_type"].upper() for c in spec["columns"]}
    for column in loader.table_columns(table):
        expected = theirs.get(column["name"])
        if expected is not None:
            assert _types_agree(column["dtype"], expected), (
                f"{table}.{column['name']}: DDL {column['dtype']} vs clifpy {expected}"
            )
