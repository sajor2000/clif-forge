"""Shared generate-stage thresholds and the pack grid-step accessor.

Single source of truth for the two organ-support thresholds and the grid-step
helper that many sibling table generators consume. Defining them once keeps the
safety-relevant clinical thresholds from drifting across a dozen files (a
name-based grep for a threshold used to miss half its copies, since the same
value appeared under several private names).

The ordinal ``support_level`` spine ladder: 0 room-air, 1 low-flow O2,
2 high-flow/NIV, 3 IMV, 4 +vasopressor, 5 +CRRT/ECMO.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.schemas.base import NUMERIC_ID_COLUMNS

__all__ = [
    "ICU_MIN_SUPPORT_LEVEL",
    "IMV_MIN_SUPPORT_LEVEL",
    "NUMERIC_ID_COLUMNS",
    "PATIENT_ID_OFFSET",
    "UTC_DATETIME",
    "enforce_numeric_ids",
    "grid_step_hours",
    "pack_table_params",
]

#: THE ID-TYPE RULE (hardcoded, single source of truth — defined in
#: ``schemas.base`` and re-exported here). The analyst-facing join keys are emitted
#: as integers so they load as numbers — no leading zeros, no string coercion — in
#: Python, R, and Stata; every other id column (``device_id``, ``provider_id``,
#: ``med_order_id``, ``culture_id``, ``hospital_id``) stays a string. Applied to
#: every generated table (see :func:`enforce_numeric_ids`) and enforced by the
#: conformance gate (``schemas.base.numeric_id_column``).

#: ``patient_id`` is shifted into this disjoint high range so it never numerically
#: collides with ``hospitalization_id`` (both derive from the same encounter index).
PATIENT_ID_OFFSET = 1_000_000_000


def enforce_numeric_ids(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply THE id-type rule to a frame: cast the analyst-facing ids in
    :data:`NUMERIC_ID_COLUMNS` from their internal prefixed-string form
    (``H{i}`` / ``P{i}``) to 1-based ``Int64``, with ``patient_id`` shifted by
    :data:`PATIENT_ID_OFFSET`. Other id columns are left untouched (strings).
    Idempotent — a column already ``Int64`` is skipped — so it is safe to apply
    at every boundary (frame build, gate, write)."""
    exprs = []
    for col in ("hospitalization_id", "hospitalization_joined_id"):
        if col in frame.columns and frame.schema[col] == pl.String:
            exprs.append((pl.col(col).str.strip_prefix("H").cast(pl.Int64) + 1).alias(col))
    if "patient_id" in frame.columns and frame.schema["patient_id"] == pl.String:
        exprs.append(
            (
                pl.col("patient_id").str.strip_prefix("P").cast(pl.Int64) + 1 + PATIENT_ID_OFFSET
            ).alias("patient_id")
        )
    return frame.with_columns(exprs) if exprs else frame


#: The polars dtype for every tz-aware UTC datetime column (R7). Shared so a
#: generator's frame schema never drifts from the conformance gate's expectation.
UTC_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")

#: ``support_level >=`` this is ICU-level care (high-flow O2 / NIV and above).
#: Used to place adt ICU segments and to gate ICU-only observations.
ICU_MIN_SUPPORT_LEVEL = 2

#: ``support_level >=`` this is invasive mechanical ventilation. The same
#: threshold marks sedation presence (ventilated patients are sedated), enables
#: proning (intubated ARDS), and drives the IMV device — one fact, one constant.
IMV_MIN_SUPPORT_LEVEL = 3


def grid_step_hours(pack: ParamPack) -> float:
    """Hours per spine interval (from the pack's spine block; default 1.0).

    The AR(1)/hazard params are only valid at the grid the pack was fitted on, so
    every generator turns interval indices into real durations through this one
    accessor.
    """
    block = pack.tables.get("spine")
    if block is None or "params" not in block:
        return 1.0
    return float(block["params"].get("state_model", {}).get("grid_step_hours", 1.0))


def pack_table_params(pack: ParamPack, table: str) -> dict[str, Any]:
    """Return ``pack.tables[table]['params']`` or ``{}`` when absent / unfitted.

    Rejects non-dict table blocks and non-dict ``params`` so a corrupt pack
    falls back to dashboard priors instead of raising mid-generation.
    """
    block = pack.tables.get(table, {})
    if not isinstance(block, dict):
        return {}
    params = block.get("params", {})
    return params if isinstance(params, dict) else {}
