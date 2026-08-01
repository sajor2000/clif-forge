"""Tier 6 ``ecmo_mcs`` generator (U20; prior-driven, R14, KTD-6).

ECMO / mechanical circulatory support is confined to the sickest patients, so it
is emitted only during the top of the organ-support ladder
(``support_level >= IMV+2`` = the CRRT/ECMO tier). There is no fitted block, so
device parameters come from the consortium's own
``outlier-handling/outlier_thresholds_ecmo_mcs.csv`` (R15 — prior-driven, marked
in ``PROVENANCE.md``, not fitted). The spine supplies only acuity (KTD-6);
reproducible under a fixed ``rng`` (R22).

**Shape follows the canonical DDL, which disagrees with clifpy here.** The DDL
models the device's work rate as an explicit
``control_parameter_name``/``_category``/``_value`` triple plus
``ecmo_configuration_category``, while clifpy 2.1 models it as a generic
``device_metric_name``/``device_rate`` pair with ``sweep``/``fdO2``. The DDL is
this project's primary source, so it wins; the divergence is pinned in
``tests/schemas/test_canonical_sources.py`` so it cannot drift unnoticed.

``device_category`` is ``VV_ECMO`` — the exact token the consortium's ECMO
outlier-threshold table keys on. (The DDL's mCIDE link for these device groups
404s upstream, so that threshold table is the only published vocabulary.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.spine import SpineFrame

__all__ = ["EcmoRow", "ecmo_mcs_frame", "sample_ecmo_mcs"]

#: ECMO/MCS is the top of the support ladder (5 = +CRRT/ECMO).
_ECMO_MIN_SUPPORT_LEVEL = 5

_DEVICE_CATEGORY = "VV_ECMO"
_MCS_GROUP = "ECMO"
#: DDL permissible set for the cannulation strategy: vv, va, va_v, vv_a. Veno-venous
#: is the configuration that matches the respiratory-failure phenotype the spine
#: drives this table from.
_CONFIGURATION = "vv"

#: The work-rate control parameter for a centrifugal ECMO pump.
_CONTROL_PARAMETER_NAME = "Pump Speed"
_CONTROL_PARAMETER_CATEGORY = "rpm"

#: Ranges for VV_ECMO from outlier_thresholds_ecmo_mcs.csv. Draws are taken from
#: the clinically typical interior of each range, not its full width: the
#: thresholds mark what is *implausible*, so sampling edge-to-edge would make
#: every stay an outlier.
_RPM_RANGE = (2500.0, 3500.0)  # canonical bound 1000-5500 RPM
_FLOW_RANGE = (3.5, 5.0)  # canonical bound 1.0-10.0 L/min
_SWEEP_RANGE = (2.0, 6.0)  # canonical bound 0.5-20 L/min
_FDO2_RANGE = (0.6, 1.0)  # canonical bound 0.21-1.0 (fraction)

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class EcmoRow:
    """One ECMO/MCS charting row."""

    hospitalization_id: str
    recorded_dttm: datetime
    device_category: str
    mcs_group: str
    ecmo_configuration_category: str
    control_parameter_name: str
    control_parameter_category: str
    control_parameter_value: float
    flow: float
    sweep_set: float
    fdO2_set: float  # noqa: N815 — canonical CLIF column name


def sample_ecmo_mcs(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[EcmoRow]:
    """Emit ECMO/MCS rows during the highest-acuity (ECMO-tier) intervals (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    grid_step = grid_step_hours(pack)

    rows: list[EcmoRow] = []
    for idx, level in enumerate(spine.support_level):
        if level < _ECMO_MIN_SUPPORT_LEVEL:
            continue
        rows.append(
            EcmoRow(
                hospitalization_id=hid,
                recorded_dttm=admit_dttm + timedelta(hours=idx * grid_step),
                device_category=_DEVICE_CATEGORY,
                mcs_group=_MCS_GROUP,
                ecmo_configuration_category=_CONFIGURATION,
                control_parameter_name=_CONTROL_PARAMETER_NAME,
                control_parameter_category=_CONTROL_PARAMETER_CATEGORY,
                control_parameter_value=round(float(rng.uniform(*_RPM_RANGE)), 0),
                flow=round(float(rng.uniform(*_FLOW_RANGE)), 2),
                sweep_set=round(float(rng.uniform(*_SWEEP_RANGE)), 1),
                fdO2_set=round(float(rng.uniform(*_FDO2_RANGE)), 2),
            )
        )
    return rows


def ecmo_mcs_frame(rows: list[EcmoRow]) -> pl.DataFrame:
    """Stack ECMO/MCS rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "recorded_dttm": [r.recorded_dttm for r in rows],
            "device_name": [r.device_category for r in rows],
            "device_category": [r.device_category for r in rows],
            "mcs_group": [r.mcs_group for r in rows],
            "ecmo_configuration_category": [r.ecmo_configuration_category for r in rows],
            "control_parameter_name": [r.control_parameter_name for r in rows],
            "control_parameter_category": [r.control_parameter_category for r in rows],
            "control_parameter_value": [r.control_parameter_value for r in rows],
            "flow": [r.flow for r in rows],
            "sweep_set": [r.sweep_set for r in rows],
            "fdO2_set": [r.fdO2_set for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "recorded_dttm": UTC_DATETIME,
            "device_name": pl.String,
            "device_category": pl.String,
            "mcs_group": pl.String,
            "ecmo_configuration_category": pl.String,
            "control_parameter_name": pl.String,
            "control_parameter_category": pl.String,
            "control_parameter_value": pl.Float64,
            "flow": pl.Float64,
            "sweep_set": pl.Float64,
            "fdO2_set": pl.Float64,
        },
    )
