#!/usr/bin/env python3
"""Emit the realism-source matrix for all 28 CLIF tables.

For each canonical table reports whether generation is driven by a MIMIC-fitted
pack block, a vendored clif-icu.com cohort dashboard prior, or derivation from
a parent table.

Usage::

    uv run python scripts/audit_realism_sources.py [--pack data/param_packs/mimic_all28]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clifforge.fit.param_pack import ParamPack
from clifforge.reference import dashboard_priors, loader

#: Tables local MIMIC Ext CLIF ships under ``~/Data/clif-mimic``.
MIMIC_TABLES: frozenset[str] = frozenset(
    {
        "adt",
        "code_status",
        "crrt_therapy",
        "ecmo_mcs",
        "hospital_diagnosis",
        "hospitalization",
        "labs",
        "medication_admin_continuous",
        "medication_admin_intermittent",
        "microbiology_culture",
        "patient",
        "patient_assessments",
        "patient_procedures",
        "position",
        "respiratory_support",
        "vitals",
    }
)

#: Tables with no MIMIC file — dashboard / literature priors.
DASHBOARD_TABLES: frozenset[str] = frozenset(
    {
        "clinical_trial",
        "intake_output",
        "invasive_hemodynamics",
        "key_icu_orders",
        "microbiology_nonculture",
        "patient_diagnosis",
        "place_based_index",
        "provider",
        "therapy_details",
        "transfusion",
    }
)

#: Folded from parent tables (no independent realism source).
DERIVED_TABLES: frozenset[str] = frozenset(
    {"medication_orders", "microbiology_susceptibility"}
)


def _source_for(table: str, pack: ParamPack | None) -> dict[str, object]:
    fitted = bool(
        pack is not None
        and table in pack.tables
        and pack.tables[table].get("fitted")
    )
    if table in DERIVED_TABLES:
        kind = "derived"
        detail = (
            "medication_orders ← med-admin tables"
            if table == "medication_orders"
            else "microbiology_susceptibility ← microbiology_culture"
        )
    elif fitted and table in MIMIC_TABLES:
        kind = "mimic_fit"
        detail = f"pack.tables[{table!r}].params"
    elif table in MIMIC_TABLES:
        kind = "mimic_pending"
        detail = "MIMIC parquet present; pack block not yet fitted (spine/prior path)"
    elif table in DASHBOARD_TABLES:
        kind = "dashboard_prior"
        detail = dashboard_priors.DASHBOARD_SOURCE_URL
    else:
        kind = "unknown"
        detail = ""
    return {"table": table, "source": kind, "detail": detail, "fitted_in_pack": fitted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        type=Path,
        default=None,
        help="Optional parameter pack to mark which tables are already fitted",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pack = ParamPack.load(str(args.pack)) if args.pack and args.pack.is_dir() else None
    rows = [_source_for(t, pack) for t in loader.dictionary_tables()]
    # Spine is not a CLIF table but drives many generators.
    spine_fitted = bool(pack and "spine" in pack.tables and pack.tables["spine"].get("fitted"))
    payload = {
        "dashboard": {
            "url": dashboard_priors.DASHBOARD_SOURCE_URL,
            "retrieved_at": dashboard_priors.DASHBOARD_RETRIEVED_AT,
        },
        "mimic_tables": sorted(MIMIC_TABLES),
        "spine_fitted": spine_fitted,
        "tables": rows,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
