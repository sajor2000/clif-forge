"""Curated lab_category → LOINC (+ specimen) crosswalk for emitted CLIF labs.

Hand-reviewed official LOINC codes for consortium ``lab_category`` values.
Categories without an entry stay null — never invent a code. Specimen pairs are
a documented synthetic prior (not mCIDE).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

__all__ = ["lab_loinc_code", "lab_specimen", "micro_loinc_code"]

_DATA = Path(__file__).resolve().parents[1] / "reference" / "data" / "crosswalks"


@cache
def _lab_rows() -> dict[str, tuple[str, str, str]]:
    """category -> (loinc, specimen_category, specimen_name)."""
    path = _DATA / "lab_category_loinc.csv"
    out: dict[str, tuple[str, str, str]] = {}
    with path.open(encoding="utf-8") as fh:
        header = next(fh).strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            cat = parts[idx["lab_category"]]
            out[cat] = (
                parts[idx["lab_loinc_code"]],
                parts[idx["lab_specimen_category"]],
                parts[idx["lab_specimen_name"]],
            )
    return out


@cache
def _micro_rows() -> dict[str, str]:
    path = _DATA / "micro_assay_loinc.csv"
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def lab_loinc_code(lab_category: str) -> str | None:
    row = _lab_rows().get(lab_category)
    return row[0] if row else None


def lab_specimen(lab_category: str) -> tuple[str | None, str | None]:
    """Return ``(specimen_category, specimen_name)`` or ``(None, None)``."""
    row = _lab_rows().get(lab_category)
    if not row:
        return None, None
    return row[1], row[2]


def micro_loinc_code(assay_key: str) -> str | None:
    """LOINC for a microbiology assay key (fluid/method or organism target)."""
    return _micro_rows().get(assay_key)
