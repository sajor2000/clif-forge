#!/usr/bin/env python3
"""Emit a machine-readable CLIF 2.1.0 compliance matrix for one generated dataset.

Usage::

    uv run python scripts/audit_clif21.py [--n-patients 40] [--seed 7] [--out AUDIT.md]

Reports, per canonical column: ``emitted`` / ``gap`` / ``deliberate`` (from
``KNOWN_GAPS``), plus whether each mCIDE field's emitted values ⊆ the vendored
list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clifforge.demo import demo_pack
from clifforge.generate.orchestrator import generate_dataset
from clifforge.reference import categories, loader

# Imported from the coverage test so the matrix and the pytest gate stay in sync.
from tests.generate.test_canonical_coverage import KNOWN_GAPS


def _status(table: str, column: str, emitted: set[str]) -> str:
    if column in emitted:
        return "emitted"
    if column in KNOWN_GAPS.get(table, {}):
        return "deliberate"
    return "gap"


def _mcide_ok(table: str, field: str, frame_columns: set[str], frame) -> str:
    if field not in frame_columns:
        return "not_emitted"
    allowed = set(categories(table, field))
    series = frame[field].drop_nulls()
    if series.is_empty():
        return "empty"
    bad = set(series.cast(str).unique().to_list()) - allowed
    return "ok" if not bad else f"FAIL:{len(bad)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ds = generate_dataset(demo_pack(), n_patients=args.n_patients, seed=args.seed)
    lines = [
        "# CLIF 2.1.0 compliance matrix",
        "",
        f"Generated with demo_pack, n={args.n_patients}, seed={args.seed}.",
        "",
        "| Table | Column | Coverage | mCIDE |",
        "|-------|--------|----------|-------|",
    ]
    for table in loader.dictionary_tables():
        frame = ds.tables[table]
        emitted = set(frame.columns)
        mcide_fields = set()
        try:
            mcide_fields = set(loader.mcide_fields(table))
        except loader.ReferenceDataError:
            pass
        for col in loader.table_columns(table):
            name = col["name"]
            cov = _status(table, name, emitted)
            mcide = (
                _mcide_ok(table, name, emitted, frame)
                if name in mcide_fields
                else "—"
            )
            lines.append(f"| `{table}` | `{name}` | {cov} | {mcide} |")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
