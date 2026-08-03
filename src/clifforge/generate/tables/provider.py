"""Tier 6 ``provider`` generator (U20; prior-driven, R14, KTD-6).

Every hospitalization has a care team. Without a fitted block, each stay gets a
documented attending + bedside nurse. With pack params, role marginal and
roles-per-stay drive the assignment set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours, pack_table_params
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference.dashboard_priors import absent_table_rates as _DASH_RATES

__all__ = ["ProviderRow", "provider_frame", "sample_provider"]

#: Documented care-team roles assigned for the whole stay.
_ROLES: tuple[str, ...] = ("Attending", "Nurse")
_DEFAULT_ROLES_PER_STAY = _DASH_RATES["provider_roles_per_stay"]

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class ProviderRow:
    """One provider assignment spanning the stay."""

    hospitalization_id: str
    provider_id: str
    start_dttm: datetime
    stop_dttm: datetime
    provider_role_category: str


def sample_provider(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[ProviderRow]:
    """Emit the stay's provider assignments (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    los_hours = spine.n_intervals * grid_step_hours(pack)
    discharge = admit_dttm + timedelta(hours=los_hours)
    params = pack_table_params(pack, "provider")
    role_marginal = params.get("provider_role_category_marginal")
    roles_per_stay = float(params.get("roles_per_stay", _DEFAULT_ROLES_PER_STAY))

    if isinstance(role_marginal, dict) and role_marginal:
        n_roles = max(1, int(round(roles_per_stay)))
        roles = [categorical(role_marginal, rng) for _ in range(n_roles)]
    else:
        del rng
        roles = list(_ROLES)

    return [
        ProviderRow(
            hospitalization_id=hid,
            provider_id=f"{hid}-{role}" if roles.count(role) == 1 else f"{hid}-{role}-{i}",
            start_dttm=admit_dttm,
            stop_dttm=discharge,
            provider_role_category=role,
        )
        for i, role in enumerate(roles)
    ]


def provider_frame(rows: list[ProviderRow]) -> pl.DataFrame:
    """Stack provider assignments into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "provider_id": [r.provider_id for r in rows],
            "start_dttm": [r.start_dttm for r in rows],
            "stop_dttm": [r.stop_dttm for r in rows],
            "provider_role_name": [r.provider_role_category for r in rows],
            "provider_role_category": [r.provider_role_category for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "provider_id": pl.String,
            "start_dttm": UTC_DATETIME,
            "stop_dttm": UTC_DATETIME,
            "provider_role_name": pl.String,
            "provider_role_category": pl.String,
        },
    )
