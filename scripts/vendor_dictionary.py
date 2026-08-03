"""Vendor the CLIF 2.1 schema from the consortium's canonical DDL (U3).

**Source of truth: ``CLIF/ddl/2.1/CLIF2.1_MYSQL_ddl.sql``.** That file is the
consortium's machine-readable schema — one ``CREATE TABLE`` per CLIF table, with a
data type and a JSON ``COMMENT`` (description + permissible values) per column, and
explicit ``FOREIGN KEY`` constraints. It is byte-identical at the pinned ``v2.1.0``
tag and on ``main``, so pinning costs nothing and buys reproducibility (R24).

This script previously parsed the CLIF website's human-facing Quarto page
(``data-dictionary-2.1.0.qmd``). That page is prose documentation, not a spec, and
it had drifted badly from the DDL: it omitted 3 whole tables, dropped 41 columns,
mistyped 45 more, and gave four tables (``hospital_diagnosis``,
``microbiology_susceptibility``, ``patient_procedures``, ``microbiology_nonculture``)
column lists that were wrong rather than merely incomplete — e.g. it keyed
``hospital_diagnosis`` on ``patient_id`` with a ``DOUBLE`` diagnosis code, where the
DDL keys it on ``hospitalization_id`` with a ``VARCHAR`` ICD code. Generators built
on the prose page emitted tables that do not exist in CLIF.

The Quarto page is still fetched, but **only** for per-table maturity badges: the
canonical repo's ``maturity.md`` tiers the project as a whole and never labels
individual tables, so the website badge is the only per-table beta/concept signal
that exists. Column names, types, permissible values, and foreign keys all come
from the DDL.

``clifpy`` (an installed dependency) ships ``schemas/2.1/*.yaml`` for 18 of the 28
tables and is the second canonical artifact. It is not merged in here — it is used
as an independent cross-check in ``tests/schemas/test_canonical_sources.py``, so
drift between the two canonical sources fails CI instead of passing silently.

Re-run with::

    uv run python scripts/vendor_dictionary.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

CLIF_REPO = "Common-Longitudinal-ICU-data-Format/CLIF"
# v2.1.0 release commit; ddl/2.1/CLIF2.1_MYSQL_ddl.sql is identical here and on main.
CLIF_COMMIT = "966bc5fb0dc0f5664405f833568886ad850d869d"
DDL_PATH = "ddl/2.1/CLIF2.1_MYSQL_ddl.sql"

WEBSITE_REPO = "clif-consortium/website"
WEBSITE_COMMIT = "888353c0521ac2ce2c380cf2ec94cedaabd36bfe"
QMD_PATH = "data-dictionary/data-dictionary-2.1.0.qmd"

CLIF_VERSION = "2.1.0"
RETRIEVED_AT = "2026-08-01"

DATA_ROOT = Path(__file__).resolve().parent.parent / "src" / "clifforge" / "reference" / "data"

#: Website section heading -> canonical DDL table name, where they differ.
TABLE_RENAME = {
    "microbiology_non_culture": "microbiology_nonculture",
    "procedures": "patient_procedures",
    "sensitivity": "microbiology_susceptibility",
}

#: How far past a ``## Table`` heading to look for that table's maturity badge.
_BADGE_SCAN_LINES = 6


def _fetch(repo: str, commit: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (pinned host)
        return str(resp.read().decode("utf-8"))


# --- canonical DDL ----------------------------------------------------------- #
_CREATE_TABLE = re.compile(r"CREATE TABLE\s+(\w+)\s*\((.*?)\n\);", re.S)
_COLUMN = re.compile(r"^(\w+)\s+(\w+)\s*(?:COMMENT\s*'(.*)')?,?$")
_FOREIGN_KEY = re.compile(r"FOREIGN KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\((\w+)\)")


def _permissible(comment: str | None) -> str:
    """Pull the ``permissible`` field out of a column's JSON COMMENT.

    The COMMENT is JSON but embedded in a single-quoted SQL string, so inner double
    quotes arrive escaped as ``\\"``. A column whose comment will not parse yields
    an empty string rather than aborting the vendor run.
    """
    if not comment:
        return ""
    try:
        return str(json.loads(comment.replace('\\"', '"')).get("permissible", ""))
    except json.JSONDecodeError:
        return ""


def _parse_ddl(text: str) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for match in _CREATE_TABLE.finditer(text):
        name, body = match.group(1), match.group(2)
        columns: list[dict[str, str]] = []
        foreign_keys: list[dict[str, str]] = []
        for raw in body.splitlines():
            line = raw.strip()
            if not line or line.startswith("--"):
                continue
            if fk := _FOREIGN_KEY.search(line):
                foreign_keys.append(
                    {"column": fk.group(1), "references": f"{fk.group(2)}.{fk.group(3)}"}
                )
                continue
            if col := _COLUMN.match(line):
                columns.append(
                    {
                        "name": col.group(1),
                        "dtype": col.group(2).upper(),
                        "permissible": _permissible(col.group(3)),
                    }
                )
        tables[name] = {"columns": columns, "foreign_keys": foreign_keys}
    return tables


# --- website maturity badges (the only per-table tiering that exists) --------- #
def _parse_maturity(text: str) -> dict[str, str]:
    lines = text.splitlines()
    maturity: dict[str, str] = {}
    for i, line in enumerate(lines):
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if not heading:
            continue
        raw = re.sub(r"\{#.*?\}", "", heading.group(1)).replace("*", "").strip()
        table = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        table = TABLE_RENAME.get(table, table)
        for probe in lines[i + 1 : i + 1 + _BADGE_SCAN_LINES]:
            if badge := re.search(r"Maturity-(\w+)-", probe):
                maturity[table] = badge.group(1).lower()
                break
    return maturity


def main() -> None:
    tables = _parse_ddl(_fetch(CLIF_REPO, CLIF_COMMIT, DDL_PATH))
    maturity = _parse_maturity(_fetch(WEBSITE_REPO, WEBSITE_COMMIT, QMD_PATH))
    for name, entry in tables.items():
        # A DDL table the website never documented has no tier; record it as such
        # rather than guessing one.
        entry["maturity"] = maturity.get(name)

    payload = {
        "clif_version": CLIF_VERSION,
        "source_repo": f"https://github.com/{CLIF_REPO}",
        "source_commit": CLIF_COMMIT,
        "source_path": DDL_PATH,
        "retrieved_at": RETRIEVED_AT,
        "maturity_source": {
            "repo": f"https://github.com/{WEBSITE_REPO}",
            "commit": WEBSITE_COMMIT,
            "path": QMD_PATH,
            "note": (
                "Per-table maturity badges only. The canonical CLIF repo's maturity.md "
                "tiers the project as a whole and does not label individual tables, so "
                "the website badge is the sole per-table beta/concept signal. Six tables "
                "carry a Concept badge while sitting under the website's 'Beta tables' "
                "heading; the badge is authoritative."
            ),
        },
        "note": (
            "Columns, data types, permissible values, and foreign keys are parsed from "
            "the consortium's canonical CLIF 2.1 DDL. clifpy's schemas/2.1/*.yaml is the "
            "second canonical artifact and is cross-checked in the test suite, not merged."
        ),
        "tables": {
            name: {
                "maturity": entry["maturity"],
                "foreign_keys": entry["foreign_keys"],
                "columns": entry["columns"],
            }
            for name, entry in sorted(tables.items())
        },
    }
    out = DATA_ROOT / "dictionary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total = sum(len(e["columns"]) for e in tables.values())
    print(f"Wrote {out.name} from the canonical DDL: {len(tables)} tables, {total} columns.")
    for name, entry in sorted(tables.items()):
        tier = entry["maturity"] or "untiered"
        fks = f"  fk->{len(entry['foreign_keys'])}" if entry["foreign_keys"] else ""
        print(f"  {tier:9} {name:30} {len(entry['columns']):2} cols{fks}")


if __name__ == "__main__":
    main()
