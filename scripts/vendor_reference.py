"""One-time fetch of the CLIF 2.1.0 mCIDE + outlier reference data (U2).

Vendors the permissible-value (mCIDE) category CSVs and the outlier-threshold
CSVs from the CLIF spec repo at a *pinned commit* into
``src/clifforge/reference/data/`` so the generator is fully offline and
reproducible (R24, R4). A ``manifest.json`` records the source repo, commit,
ref, retrieval date, and a self-describing ``(table, field) -> file`` map that
``reference/loader.py`` reads — so nothing downstream guesses filenames.

The plan targets CLIF **2.1.0**; this pins the ``v2.1.0`` release commit. Newer
releases (v2.1.1, v2.2.0, v3.0.0) exist and can be diffed later against the
recorded commit.

Re-run with::

    uv run python scripts/vendor_reference.py

It is network-touching and intended to be run rarely (when bumping the pinned
commit), then the resulting CSVs + manifest are committed.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

REPO = "Common-Longitudinal-ICU-data-Format/CLIF"
# v2.1.0 release commit (released 2025-12-12).
COMMIT = "966bc5fb0dc0f5664405f833568886ad850d869d"
REF = "v2.1.0"
CLIF_VERSION = "2.1.0"
RETRIEVED_AT = "2026-07-23"

# mCIDE table folders that are NOT the stable 2.1.0 per-table category sets.
EXCLUDED_MCIDE_DIRS = {"1_0_0", "2_2_0_WIP"}

# Files under mCIDE/ that are code *tables*, not permissible-value enumerations.
# `clif_patient_procedure_codes.csv` is a CPT code list
# (`procedure_code_format,procedure_code,proc_name`) whose first column is the
# literal "CPT" on every row — indexing it as a category set made
# `categories("patient_procedures", "procedure_code_format")` return ["CPT"] x 15.
# It is real reference data (the generator emits these exact codes), so it is
# vendored and recorded under `code_lists`, just not under the category map.
CODE_LIST_FILES = {
    "mCIDE/patient_procedures/clif_patient_procedure_codes.csv": "patient_procedures"
}

# Upstream misspellings normalized on vendoring so downstream code uses the
# canonical spelling.
FOLDER_NORMALIZE = {"postion": "position"}
FIELD_NORMALIZE = {"ethinicity_category": "ethnicity_category"}

# Outlier-threshold files (plausibility bounds for R9 in_range gating) mapped to
# the CLIF table they bound. Each file is `<key>,lower_limit,upper_limit`.
OUTLIER_FILES = {
    "outlier-handling/outlier_thresholds_adults_vitals.csv": "vitals",
    "outlier-handling/outlier_thresholds_labs.csv": "labs",
    "outlier-handling/outlier_thresholds_respiratory_support.csv": "respiratory_support",
    "outlier-handling/outlier_thresholds_crrt_modes.csv": "crrt_therapy",
    "outlier-handling/outlier_thresholds_ecmo_mcs.csv": "ecmo_mcs",
}

DATA_ROOT = Path(__file__).resolve().parent.parent / "src" / "clifforge" / "reference" / "data"


def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"


def _fetch_bytes(path: str) -> bytes:
    with urllib.request.urlopen(_raw_url(path), timeout=30) as resp:  # noqa: S310 (pinned host)
        data: bytes = resp.read()
    return data


def _fetch_tree() -> list[str]:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (pinned host)
        tree: dict[str, Any] = json.load(resp)
    return [entry["path"] for entry in tree["tree"]]


def _decode_clean(raw: bytes) -> str:
    """Decode UTF-8 (stripping an optional BOM) and drop leading blank lines.

    Some upstream mCIDE CSVs carry a BOM (outlier files) or a stray blank first
    line before the header (e.g. the med-action categories); both are normalized
    here so the header is always the first row.
    """
    return raw.decode("utf-8-sig").lstrip("\r\n")


def _first_column_header(text: str) -> str:
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    return header[0].strip()


def _select_mcide_paths(all_paths: list[str]) -> list[str]:
    selected = []
    for path in all_paths:
        if not (path.startswith("mCIDE/") and path.endswith(".csv")):
            continue
        parts = path.split("/")
        # Stable per-table files are exactly mCIDE/<table>/<file>.csv.
        if len(parts) != 3:
            continue
        if parts[1] in EXCLUDED_MCIDE_DIRS:
            continue
        selected.append(path)
    return selected


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    all_paths = _fetch_tree()

    mcide_map: dict[str, dict[str, str]] = {}
    code_list_map: dict[str, str] = {}
    for path in sorted(_select_mcide_paths(all_paths)):
        _, table_raw, filename = path.split("/")
        table = FOLDER_NORMALIZE.get(table_raw, table_raw)
        raw = _fetch_bytes(path)
        text = _decode_clean(raw)
        field_raw = _first_column_header(text)
        field = FIELD_NORMALIZE.get(field_raw, field_raw)

        rel = f"mcide/{table}/{filename}"
        dest = DATA_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        if path in CODE_LIST_FILES:
            code_list_map[CODE_LIST_FILES[path]] = rel
            print(f"  codes  {table} <- {path} (code table, not a category list)")
            continue
        mcide_map.setdefault(table, {})[field] = rel
        print(f"  mcide  {table}.{field} <- {path}")

    outlier_map: dict[str, str] = {}
    for path, table in OUTLIER_FILES.items():
        raw = _fetch_bytes(path)
        text = _decode_clean(raw)
        filename = path.split("/")[-1]
        rel = f"outliers/{filename}"
        dest = DATA_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        outlier_map[table] = rel
        print(f"  outlier {table} <- {path}")

    manifest = {
        "clif_version": CLIF_VERSION,
        "source_repo": f"https://github.com/{REPO}",
        "source_commit": COMMIT,
        "source_ref": REF,
        "retrieved_at": RETRIEVED_AT,
        "note": (
            "Targets CLIF 2.1.0 per the plan. Newer releases (v2.1.1, v2.2.0, "
            "v3.0.0) exist upstream and can be diffed against source_commit."
        ),
        "mcide": {t: dict(sorted(fields.items())) for t, fields in sorted(mcide_map.items())},
        "code_lists": dict(sorted(code_list_map.items())),
        "outliers": dict(sorted(outlier_map.items())),
    }
    (DATA_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nWrote manifest with {len(mcide_map)} mCIDE tables "
        f"and {len(outlier_map)} outlier tables."
    )


if __name__ == "__main__":
    main()
