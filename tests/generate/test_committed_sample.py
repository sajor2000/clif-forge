"""The committed sample reproduces byte-for-byte from its recipe (reproducibility audit)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.filenames import table_parquet_path
from clifforge.generate.orchestrator import generate_dataset
from clifforge.variants import load_spec, spec_to_pack

_SAMPLE = Path("sample_dataset")
_FULL_SAMPLE = Path("sample_full_hospital")
_BASE = Path("base_pack")

pytestmark = pytest.mark.skipif(
    not table_parquet_path(_SAMPLE, "hospitalization").exists() or not _BASE.exists(),
    reason="requires the committed sample and base pack",
)


def test_committed_sample_reproduces_from_its_recipe() -> None:
    # Rebuild the exact pack from the committed spec + base pack, then regenerate the
    # first encounters with the recorded seed. SeedSequence(seed).spawn assigns each
    # encounter a stable key, so H0..H29 must match the committed sample's first 30 —
    # proving the whole dataset is reproducible from spec + base pack + seed.
    spec = load_spec(_SAMPLE / "spec.toml")
    pack = spec_to_pack(spec, ParamPack.load(str(_BASE)))
    regen = (
        generate_dataset(pack, n_patients=30, seed=42)
        .tables["hospitalization"]
        .sort("hospitalization_id")
    )
    ids = list(range(1, 31))  # 1-based int hospitalization ids
    committed = (
        pl.read_parquet(table_parquet_path(_SAMPLE, "hospitalization"))
        .filter(pl.col("hospitalization_id").is_in(ids))
        .sort("hospitalization_id")
    )
    assert regen.equals(committed)


@pytest.mark.skipif(
    not table_parquet_path(_FULL_SAMPLE, "hospitalization").exists(),
    reason="requires the committed full-hospital sample",
)
def test_committed_full_hospital_sample_reproduces_from_its_recipe() -> None:
    # Same reproducibility contract as the ICU sample, but through the
    # ``mode = "full_hospital"`` spec path: regenerating the first 30 encounters from the committed
    # base pack + spec + seed must match the committed full-hospital sample.
    spec = load_spec(_FULL_SAMPLE / "spec.toml")
    pack = spec_to_pack(spec, ParamPack.load(str(_BASE)))
    regen = (
        generate_dataset(pack, n_patients=30, seed=42)
        .tables["hospitalization"]
        .sort("hospitalization_id")
    )
    ids = list(range(1, 31))  # 1-based int hospitalization ids
    committed = (
        pl.read_parquet(_FULLtable_parquet_path(_SAMPLE, "hospitalization"))
        .filter(pl.col("hospitalization_id").is_in(ids))
        .sort("hospitalization_id")
    )
    assert regen.equals(committed)
