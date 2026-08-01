"""``place_based_index`` generator (prior-driven; R14, R19).

Place-based indices summarise the neighbourhood a patient lives in — the Area
Deprivation Index and the CDC/ATSDR Social Vulnerability Index are the two the
DDL names. Each hospitalization gets one row per index.

**Both indices are drawn on their real published scales, and the two are
correlated**, because a single neighbourhood produces both: ADI national rank is
a percentile 1-100, SVI is a 0-1 proportion, and a deprived area scores high on
both. Drawing them independently would let a dataset contain a hospitalization in
the least-deprived ADI decile and the most-vulnerable SVI decile at once, which
would quietly break the disparities analyses this table exists to support. So a
single latent deprivation draw generates both, with the SVI jittered around it
rather than resampled.

This carries no PHI risk: the values are drawn from the marginal scale, not from
any real geography, and nothing here ties back to the FIPS/zip columns on
``hospitalization`` (which this generator does not populate).

Scales and version labels are documented priors recorded in ``PROVENANCE.md``;
CLIF 2.1 gives this table no mCIDE. Reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame

__all__ = ["PlaceIndexRow", "place_based_index_frame", "sample_place_based_index"]

_ADI_NAME = "Area Deprivation Index"
_ADI_VERSION = "ADI 2020 national percentile"
_SVI_NAME = "Social Vulnerability Index"
_SVI_VERSION = "SVI 2020"

#: How far the SVI percentile wanders from the ADI percentile for the same
#: neighbourhood (they measure overlapping but not identical constructs).
_SVI_JITTER = 0.12

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class PlaceIndexRow:
    """One place-based index value for a hospitalization."""

    hospitalization_id: str
    index_name: str
    index_value: float
    index_version: str


def sample_place_based_index(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[PlaceIndexRow]:
    """Emit correlated ADI and SVI values for one hospitalization (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id

    # One latent neighbourhood deprivation level in [0, 1] drives both indices.
    deprivation = float(rng.random())
    svi = float(np.clip(deprivation + rng.normal(0.0, _SVI_JITTER), 0.0, 1.0))

    return [
        PlaceIndexRow(hid, _ADI_NAME, round(1.0 + deprivation * 99.0, 0), _ADI_VERSION),
        PlaceIndexRow(hid, _SVI_NAME, round(svi, 4), _SVI_VERSION),
    ]


def place_based_index_frame(rows: list[PlaceIndexRow]) -> pl.DataFrame:
    """Stack index values into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "index_name": [r.index_name for r in rows],
            "index_value": [r.index_value for r in rows],
            "index_version": [r.index_version for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "index_name": pl.String,
            "index_value": pl.Float64,
            "index_version": pl.String,
        },
    )
