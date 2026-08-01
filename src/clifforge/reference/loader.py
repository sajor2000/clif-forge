"""Read-only accessors for the vendored CLIF 2.1.0 reference data (U2).

The vendored ``data/manifest.json`` is self-describing: it maps every
``(table, field)`` to the relative path of the mCIDE category CSV that defines
the field's permissible values, and every table to its outlier-threshold CSV.
This module exposes that data as plain Python so schema construction (U3) and
sampling never parse filenames or guess at spellings.

Design contract (R4, R9): a *missing* table/field is a programming error, not an
empty result — every accessor RAISES ``ReferenceDataError`` rather than returning
an empty list or silently degrading. A schema built against a field the vendor
never captured must fail loudly.
"""

from __future__ import annotations

import csv
import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

_DATA_ROOT = Path(__file__).parent / "data"
_MANIFEST_PATH = _DATA_ROOT / "manifest.json"
_DICTIONARY_PATH = _DATA_ROOT / "dictionary.json"


class ReferenceDataError(LookupError):
    """A requested table, field, or reference file is absent from the vendor set."""


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.exists():
        raise ReferenceDataError(
            f"Reference manifest not found at {_MANIFEST_PATH}. "
            "Run `uv run python scripts/vendor_reference.py` to vendor CLIF 2.1.0 data."
        )
    with _MANIFEST_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@lru_cache(maxsize=1)
def _dictionary() -> dict[str, Any]:
    if not _DICTIONARY_PATH.exists():
        raise ReferenceDataError(
            f"Data dictionary not found at {_DICTIONARY_PATH}. "
            "Run `uv run python scripts/vendor_dictionary.py` to vendor the CLIF 2.1.0 dictionary."
        )
    with _DICTIONARY_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@cache
def _read_first_column(rel_path: str) -> tuple[str, ...]:
    """Return the first-column values (the category values) of a vendored CSV.

    Skips the header row and blank lines. Values are returned verbatim (CLIF
    mCIDE categories are case-sensitive and matched exactly downstream).
    """
    path = _DATA_ROOT / rel_path
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    values: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value:
                values.append(value)
    return tuple(values)


def provenance() -> dict[str, str]:
    """Return the source provenance recorded at vendor time (version, commit, ...)."""
    m = _manifest()
    return {
        "clif_version": m["clif_version"],
        "source_repo": m["source_repo"],
        "source_commit": m["source_commit"],
        "source_ref": m["source_ref"],
        "retrieved_at": m["retrieved_at"],
    }


def tables() -> list[str]:
    """All tables that have at least one vendored mCIDE category field."""
    return sorted(_manifest()["mcide"].keys())


def dictionary_tables() -> list[str]:
    """All CLIF 2.1.0 tables defined in the vendored data dictionary."""
    return sorted(_dictionary()["tables"].keys())


def _table_entry(table: str) -> dict[str, Any]:
    tables_map = _dictionary()["tables"]
    if table not in tables_map:
        raise ReferenceDataError(
            f"No dictionary entry for table {table!r}. "
            f"Known tables: {', '.join(sorted(tables_map))}"
        )
    entry: dict[str, Any] = tables_map[table]
    return entry


def table_columns(table: str) -> list[dict[str, str]]:
    """Return the column list for ``table`` as ``[{name, dtype, permissible}, ...]``.

    ``dtype`` is the canonical DDL type (VARCHAR / DATETIME / DATE / DOUBLE /
    FLOAT / INT / BOOLEAN); ``permissible`` is the canonical permissible-values
    text, which may be empty. Raises if the table is not in the dictionary.
    """
    return [dict(c) for c in _table_entry(table)["columns"]]


def table_maturity(table: str) -> str | None:
    """Return ``'beta'`` / ``'concept'``, or ``None`` for an untiered table.

    Maturity comes from the CLIF website's per-table badge — the only per-table
    tiering that exists. Tables the canonical DDL defines but the website never
    documented (``clinical_trial``, ``patient_diagnosis``, ``place_based_index``)
    are genuinely untiered, so this returns ``None`` rather than inventing a tier.
    """
    maturity = _table_entry(table).get("maturity")
    return str(maturity) if maturity is not None else None


def foreign_keys(table: str) -> list[dict[str, str]]:
    """Canonical ``FOREIGN KEY`` constraints for ``table`` as ``[{column, references}]``."""
    keys: list[dict[str, str]] = _table_entry(table).get("foreign_keys", [])
    return [dict(k) for k in keys]


def dictionary_provenance() -> dict[str, str]:
    """Source provenance for the vendored data dictionary (repo, commit, ...)."""
    d = _dictionary()
    return {
        "clif_version": d["clif_version"],
        "source_repo": d["source_repo"],
        "source_commit": d["source_commit"],
        "source_path": d["source_path"],
        "retrieved_at": d["retrieved_at"],
    }


def mcide_fields(table: str) -> list[str]:
    """All mCIDE category fields vendored for ``table`` (raises if table unknown)."""
    mcide = _manifest()["mcide"]
    if table not in mcide:
        raise ReferenceDataError(
            f"No mCIDE reference data for table {table!r}. Known tables: {', '.join(sorted(mcide))}"
        )
    return sorted(mcide[table].keys())


def categories(table: str, field: str) -> list[str]:
    """Return the exact, case-sensitive permissible values for ``table.field``.

    Raises ``ReferenceDataError`` if the table or field is not in the vendor set —
    never returns an empty list for a missing field.
    """
    mcide = _manifest()["mcide"]
    if table not in mcide:
        raise ReferenceDataError(
            f"No mCIDE reference data for table {table!r}. Known tables: {', '.join(sorted(mcide))}"
        )
    fields = mcide[table]
    if field not in fields:
        raise ReferenceDataError(
            f"No mCIDE field {field!r} for table {table!r}. "
            f"Known fields: {', '.join(sorted(fields))}"
        )
    return list(_read_first_column(fields[field]))


@cache
def _read_crosswalk(rel_path: str, to_column: str) -> dict[str, str]:
    path = _DATA_ROOT / rel_path
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ReferenceDataError(f"mCIDE file has no header: {path}")
        if to_column not in fieldnames:
            raise ReferenceDataError(
                f"mCIDE file {rel_path} has no column {to_column!r} "
                f"(columns: {', '.join(fieldnames)})"
            )
        key_col = fieldnames[0]
        mapping = {
            (row.get(key_col) or "").strip(): (row.get(to_column) or "").strip()
            for row in reader
            if (row.get(key_col) or "").strip()
        }
    return mapping


def crosswalk(table: str, field: str, to_column: str) -> dict[str, str]:
    """Map each permissible value of ``table.field`` to a companion column's value.

    Several mCIDE files define more than a flat value list: they carry the
    consortium's own roll-up alongside it, e.g.
    ``clif_medication_admin_continuous_med_categories.csv`` maps every
    ``med_category`` to its ``med_group``, and the action-category file maps every
    ``mar_action_category`` to ``administered`` / ``not_administered``. Reading
    those pairings straight out of the vendored file is what keeps a generated
    ``*_group`` column canonically correct instead of hand-maintained here.

    Raises if the file, field, or companion column is absent (R4).
    """
    mcide = _manifest()["mcide"]
    if table not in mcide or field not in mcide[table]:
        raise ReferenceDataError(f"No mCIDE field {field!r} for table {table!r}.")
    return dict(_read_crosswalk(mcide[table][field], to_column))


@cache
def _read_rows(rel_path: str) -> tuple[dict[str, str], ...]:
    path = _DATA_ROOT / rel_path
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        # Upstream headers carry stray whitespace (``procedure_code_format `` has a
        # trailing space), so keys and values are both stripped on read.
        rows = tuple(
            {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(fh)
        )
    return rows


def code_list(table: str) -> list[dict[str, str]]:
    """Return a vendored clinical *code table* for ``table`` as a list of row dicts.

    Distinct from :func:`categories`: a code list enumerates real billing/clinical
    codes (``patient_procedures`` ships CPT codes with their procedure names),
    where every row is a code rather than a permissible value for one column.
    Reading it here is what lets the generator emit genuine consortium-published
    codes instead of plausible-looking invented ones.
    """
    code_lists = _manifest().get("code_lists", {})
    if table not in code_lists:
        raise ReferenceDataError(
            f"No vendored code list for table {table!r}. "
            f"Known tables: {', '.join(sorted(code_lists)) or '(none)'}"
        )
    return [dict(row) for row in _read_rows(code_lists[table])]


def bounds(table: str, field: str) -> tuple[float, float]:
    """Return ``(lower_limit, upper_limit)`` plausibility bounds for ``table.field``.

    ``field`` is the key used within the table's outlier CSV — the category or
    variable name (e.g. ``"heart_rate"`` for vitals, ``"fio2_set"`` for
    respiratory_support). Raises ``ReferenceDataError`` if the table has no
    outlier file or the field is not listed in it.
    """
    outliers = _manifest()["outliers"]
    if table not in outliers:
        raise ReferenceDataError(
            f"No outlier-threshold reference data for table {table!r}. "
            f"Known tables: {', '.join(sorted(outliers))}"
        )
    rel_path = outliers[table]
    path = _DATA_ROOT / rel_path
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ReferenceDataError(f"Outlier file has no header: {path}")
        # Simple two-sided numeric bounds only. Some upstream files (ecmo_mcs)
        # use a multi-column ranged-text schema that does not reduce to a single
        # (lower, upper) pair — reject those loudly rather than fabricate inf.
        if "lower_limit" not in fieldnames or "upper_limit" not in fieldnames:
            raise ReferenceDataError(
                f"Outlier file for table {table!r} has a non-standard schema "
                f"(columns: {', '.join(fieldnames)}); it does not expose "
                "lower_limit/upper_limit numeric bounds."
            )
        key_col = fieldnames[0]
        for row in reader:
            if (row.get(key_col) or "").strip() == field:
                lower = _parse_limit(row.get("lower_limit"), default=float("-inf"))
                upper = _parse_limit(row.get("upper_limit"), default=float("inf"))
                return (lower, upper)
    raise ReferenceDataError(f"No outlier bounds for {field!r} in table {table!r} ({rel_path}).")


def outlier_keys(table: str) -> list[str]:
    """List the keys (categories / variable names) bounded for ``table``."""
    outliers = _manifest()["outliers"]
    if table not in outliers:
        raise ReferenceDataError(
            f"No outlier-threshold reference data for table {table!r}. "
            f"Known tables: {', '.join(sorted(outliers))}"
        )
    path = _DATA_ROOT / outliers[table]
    if not path.exists():
        raise ReferenceDataError(f"Vendored reference file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ReferenceDataError(f"Outlier file has no header: {path}")
        key_col = fieldnames[0]
        return [
            (row.get(key_col) or "").strip() for row in reader if (row.get(key_col) or "").strip()
        ]


def _parse_limit(raw: str | None, *, default: float) -> float:
    """Parse an outlier limit cell to float; blank/absent falls back to ``default``.

    ``default`` carries the sign of the unbounded side (``-inf`` for a missing
    lower limit, ``+inf`` for a missing upper limit).
    """
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ReferenceDataError(f"Non-numeric outlier limit {raw!r}") from exc
