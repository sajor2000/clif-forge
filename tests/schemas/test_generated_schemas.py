"""The committed schema modules must equal what the generator would emit today.

``scripts/gen_schemas.py`` renders ``schemas/<table>.py`` by mirroring
``base.build_column``. Nothing enforced that mirroring, so a hand-edit to a
committed module (the ``numeric_id_column`` id-type rule) once lived only in the
generated files: regenerating silently reverted it and re-broke integer ids. These
tests make that divergence a CI failure instead of a surprise at regeneration time.

The check compares *rendered source* rather than comparing ``DataFrameSchema``
objects. Object equality looked like the more direct assertion but is not usable
here: pandera stamps state onto a ``Check`` when it runs, so a module-level schema
that any earlier test validated against no longer compares equal to a freshly
built one. That made the object-equality version pass alone and fail in a full
run, which is worse than no test at all.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from clifforge.reference import loader
from clifforge.schemas import base, get_schema

TABLES = loader.dictionary_tables()

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _gen_schemas():  # type: ignore[no-untyped-def]
    """Import ``scripts/gen_schemas.py``, which is not an installed package."""
    spec = importlib.util.spec_from_file_location("_gen_schemas", _SCRIPTS / "gen_schemas.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN = _gen_schemas()


@pytest.mark.parametrize("table", TABLES)
def test_committed_schema_matches_what_the_generator_renders(table: str) -> None:
    committed = (GEN.SCHEMAS_DIR / f"{table}.py").read_text(encoding="utf-8")
    rendered = GEN._render(table)
    # gen_schemas ruff-formats what it renders, which re-wraps long factory calls
    # across lines. Collapsing all whitespace compares the code and ignores the
    # wrapping, so a reformat is not mistaken for drift.
    assert "".join(rendered.split()) == "".join(committed.split()), (
        f"schemas/{table}.py is stale — re-run scripts/gen_schemas.py"
    )


@pytest.mark.parametrize("table", TABLES)
def test_committed_schema_has_the_canonical_columns(table: str) -> None:
    """Whatever the rendering, the committed schema must cover the canonical columns."""
    expected = {c["name"] for c in loader.table_columns(table)}
    assert set(get_schema(table).columns) == expected


@pytest.mark.parametrize("table", TABLES)
def test_numeric_join_keys_are_not_dtype_pinned_to_string(table: str) -> None:
    """The three integer join keys must stay dtype-free so Int64 ids validate."""
    schema = get_schema(table)
    for name in base.NUMERIC_ID_COLUMNS:
        if name in schema.columns:
            column = schema.columns[name]
            assert column.dtype is None, f"{table}.{name} is pinned to {column.dtype}"
            assert column.required and not column.nullable


@pytest.mark.parametrize("table", TABLES)
def test_other_id_columns_stay_strings(table: str) -> None:
    schema = get_schema(table)
    for name, column in schema.columns.items():
        if name.endswith("_id") and name not in base.NUMERIC_ID_COLUMNS:
            assert str(column.dtype) == "String", f"{table}.{name} is {column.dtype}"


def test_maturity_tiers_are_beta_concept_or_untiered() -> None:
    tiers = {t: loader.table_maturity(t) for t in TABLES}
    assert set(tiers.values()) <= {"beta", "concept", None}
    # The website files these six under its "Beta tables" heading, but each carries
    # a Concept maturity badge; the badge is authoritative (see vendor_dictionary.py).
    for table in (
        "code_status",
        "crrt_therapy",
        "ecmo_mcs",
        "hospital_diagnosis",
        "microbiology_culture",
        "microbiology_nonculture",
    ):
        assert tiers[table] == "concept"
    # These three exist only in the canonical DDL; the website never documented them,
    # so no per-table tier exists and we must not invent one.
    for table in ("clinical_trial", "patient_diagnosis", "place_based_index"):
        assert tiers[table] is None
