"""CLIF parquet filename layout for generated datasets.

Every *deliverable* CLIF table is written as::

    clif_<table>_2.1_<maturity>.parquet

where ``<maturity>`` is the table's website maturity badge (``beta`` or
``concept`` only). Tables the canonical DDL defines but the website never
tiered (``clinical_trial``, ``patient_diagnosis``, ``place_based_index``) are
still generated in memory for tests, but are **omitted from disk** — deliverable
writers never emit ``*_untiered`` or non-2.1 CLIF table files.

The in-memory latent spine never takes the ``clif_`` prefix and is not written
to share packages.

The version token is major.minor (``2.1``), not the patch-level ``2.1.0``
release string — maturity is a 2.1-era property, and the filename should stay
stable across 2.1.x patch bumps of the vendored dictionary.
"""

from __future__ import annotations

import re
from pathlib import Path

from clifforge.reference import loader

__all__ = [
    "CLIF_FILENAME_VERSION",
    "DELIVERABLE_MATURITIES",
    "deliverable_tables",
    "is_deliverable_table",
    "parse_table_from_stem",
    "table_parquet_filename",
    "table_parquet_path",
    "table_parquet_stem",
]

#: Spec version embedded in every generated CLIF parquet filename.
CLIF_FILENAME_VERSION = "2.1"

#: Maturity tags allowed on deliverable CLIF parquet stems.
DELIVERABLE_MATURITIES = frozenset({"beta", "concept"})

#: ``clif_<table>_2.1_<maturity>`` — deliverable maturity is beta or concept;
#: ``untiered`` remains parseable for legacy artifacts only.
_STEM = re.compile(
    rf"^clif_(?P<table>.+)_{re.escape(CLIF_FILENAME_VERSION)}_"
    r"(?P<maturity>beta|concept|untiered)$"
)


def is_deliverable_table(table: str) -> bool:
    """True when the table has a real website beta/concept maturity badge."""
    return loader.table_maturity(table) in DELIVERABLE_MATURITIES


def deliverable_tables() -> list[str]:
    """Dictionary tables that may be written as deliverable CLIF parquet."""
    return [t for t in loader.dictionary_tables() if is_deliverable_table(t)]


def table_parquet_stem(table: str) -> str:
    """Return the parquet stem for a dictionary table (no ``.parquet`` suffix).

    Untiered tables still get an ``_untiered`` stem for naming helpers / legacy
    parse tests; deliverable writers must not call this for those tables
    (use :func:`is_deliverable_table` first).
    """
    maturity = loader.table_maturity(table) or "untiered"
    if maturity not in {"beta", "concept", "untiered"}:
        raise ValueError(f"unknown maturity {maturity!r} for table {table!r}")
    return f"clif_{table}_{CLIF_FILENAME_VERSION}_{maturity}"


def table_parquet_filename(table: str) -> str:
    """Return ``clif_<table>_2.1_<maturity>.parquet`` for a deliverable table.

    Raises ``ValueError`` for untiered tables so writers cannot accidentally
    emit non-beta/concept CLIF parquet.
    """
    if not is_deliverable_table(table):
        raise ValueError(
            f"table {table!r} has no beta/concept maturity badge and must not "
            "be written as a deliverable CLIF parquet"
        )
    return f"{table_parquet_stem(table)}.parquet"


def table_parquet_path(out_dir: str | Path, table: str) -> Path:
    """``out_dir / clif_<table>_2.1_<maturity>.parquet`` (deliverable tables only)."""
    return Path(out_dir) / table_parquet_filename(table)


def parse_table_from_stem(stem: str) -> str | None:
    """Extract the table name from a maturity-tagged stem, or ``None`` if it is not one."""
    match = _STEM.match(stem)
    return match.group("table") if match else None
