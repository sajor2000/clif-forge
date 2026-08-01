"""CLIF parquet filename layout for generated datasets.

Every real CLIF table is written as::

    clif_<table>_2.1_<maturity>.parquet

where ``<maturity>`` is the table's website maturity badge (``beta`` or
``concept``). Tables the canonical DDL defines but the website never tiered
(``clinical_trial``, ``patient_diagnosis``, ``place_based_index``) use
``untiered``. The latent spine stays ``_truth.parquet`` and never takes the
``clif_`` prefix.

The version token is major.minor (``2.1``), not the patch-level
``2.1.0`` release string — maturity is a 2.1-era property, and the filename
should stay stable across 2.1.x patch bumps of the vendored dictionary.
"""

from __future__ import annotations

import re
from pathlib import Path

from clifforge.reference import loader

__all__ = [
    "CLIF_FILENAME_VERSION",
    "parse_table_from_stem",
    "table_parquet_filename",
    "table_parquet_path",
    "table_parquet_stem",
]

#: Spec version embedded in every generated CLIF parquet filename.
CLIF_FILENAME_VERSION = "2.1"

#: ``clif_<table>_2.1_<maturity>`` — maturity is beta, concept, or untiered.
_STEM = re.compile(
    rf"^clif_(?P<table>.+)_{re.escape(CLIF_FILENAME_VERSION)}_"
    r"(?P<maturity>beta|concept|untiered)$"
)


def table_parquet_stem(table: str) -> str:
    """Return the parquet stem for a dictionary table (no ``.parquet`` suffix)."""
    maturity = loader.table_maturity(table) or "untiered"
    if maturity not in {"beta", "concept", "untiered"}:
        raise ValueError(f"unknown maturity {maturity!r} for table {table!r}")
    return f"clif_{table}_{CLIF_FILENAME_VERSION}_{maturity}"


def table_parquet_filename(table: str) -> str:
    """Return ``clif_<table>_2.1_<maturity>.parquet``."""
    return f"{table_parquet_stem(table)}.parquet"


def table_parquet_path(out_dir: str | Path, table: str) -> Path:
    """``out_dir / clif_<table>_2.1_<maturity>.parquet``."""
    return Path(out_dir) / table_parquet_filename(table)


def parse_table_from_stem(stem: str) -> str | None:
    """Extract the table name from a maturity-tagged stem, or ``None`` if it is not one."""
    match = _STEM.match(stem)
    return match.group("table") if match else None
