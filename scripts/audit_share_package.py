#!/usr/bin/env python3
"""Strict 1:1 audit of a shareable CLIF 2.1 deliverable directory.

Fails unless the directory contains *exactly* the website-badged beta/concept
CLIF tables (``clif_<table>_2.1_{beta|concept}.parquet``) — no missing tables,
no extras (``_truth.parquet``, untagged ``clif_*.parquet``, untiered stems, etc.).

Usage::

    uv run python scripts/audit_share_package.py sample_dataset
    uv run python scripts/audit_share_package.py ~/Desktop/CLIFForge_datasets_for_dropbox/icu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clifforge.generate.filenames import deliverable_tables, table_parquet_filename
from clifforge.reference import categories, loader, resolve_mcide_column


def _expected_names() -> set[str]:
    return {table_parquet_filename(t) for t in deliverable_tables()}


def audit_layout(out_dir: Path) -> list[str]:
    """Return human-readable failure lines; empty means pass."""
    errors: list[str] = []
    if not out_dir.is_dir():
        return [f"{out_dir} is not a directory"]

    parquet = {p.name for p in out_dir.glob("*.parquet")}
    expected = _expected_names()
    missing = sorted(expected - parquet)
    extras = sorted(parquet - expected)
    if missing:
        errors.append(f"missing {len(missing)} deliverable table(s): {', '.join(missing)}")
    if extras:
        errors.append(
            f"extra parquet file(s) not allowed in share packages: {', '.join(extras)}"
        )

    # Non-parquet clutter is fine (README, manifest, spec) except forbidden names.
    for banned in ("_truth.parquet", "clif_truth.parquet"):
        if (out_dir / banned).exists():
            errors.append(f"forbidden generator-internal file present: {banned}")

    return errors


def audit_mcide(out_dir: Path) -> list[str]:
    """Fail if any mCIDE column emits values outside the vendored list."""
    import polars as pl

    errors: list[str] = []
    for table in deliverable_tables():
        path = out_dir / table_parquet_filename(table)
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        try:
            fields = loader.mcide_fields(table)
        except loader.ReferenceDataError:
            continue
        emitted = set(frame.columns)
        for field in fields:
            data_col = resolve_mcide_column(field, emitted)
            if data_col is None or data_col not in frame.columns:
                continue
            allowed = set(categories(table, field))
            series = frame[data_col].drop_nulls()
            if series.is_empty():
                continue
            bad = set(series.cast(str).unique().to_list()) - allowed
            if bad:
                sample = ", ".join(sorted(bad)[:5])
                errors.append(
                    f"{table}.{data_col}: {len(bad)} value(s) outside mCIDE "
                    f"(e.g. {sample})"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path, help="Dataset directory to audit")
    ap.add_argument(
        "--skip-mcide",
        action="store_true",
        help="Only check file layout (faster for huge Dropbox masters).",
    )
    args = ap.parse_args(argv)
    out = args.dir.expanduser()

    errors = audit_layout(out)
    if not errors and not args.skip_mcide:
        errors.extend(audit_mcide(out))

    expected_n = len(_expected_names())
    if errors:
        print(f"FAIL {out} — {len(errors)} issue(s) (expected {expected_n} tables):", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(f"PASS {out} — exactly {expected_n} CLIF 2.1 beta/concept tables, no extras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
