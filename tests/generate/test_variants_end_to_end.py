"""End-to-end: derivatives are conformant, distinct, and reproducible (U6, R6).

Generates small cohorts from shipped presets against the committed shareable base
pack and asserts every table passes the CLIF 2.1 gate, two different presets yield
distinct datasets, and identical spec+seed reproduces byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.orchestrator import generate_dataset
from clifforge.variants import load_preset, spec_to_pack

_BASE = Path("base_pack")

pytestmark = pytest.mark.skipif(not _BASE.exists(), reason="requires the shipped base pack")


def _generate(preset: str, *, n: int = 150, seed: int = 1) -> dict[str, pl.DataFrame]:
    pack = spec_to_pack(load_preset(preset), ParamPack.load(str(_BASE)))
    return generate_dataset(pack, n_patients=n, seed=seed).tables


def test_two_presets_are_conformant_and_distinct() -> None:
    a = _generate("high-acuity")
    b = _generate("older-cohort")
    for ds in (a, b):
        for name, frame in ds.items():
            assert gate.validate(frame, name, run_secondary=False).pandera_passed, name
    # Always distinct: different specs -> different content.
    assert not a["hospitalization"].equals(b["hospitalization"])


def test_same_spec_and_seed_reproduces_byte_for_byte() -> None:
    a = _generate("high-acuity", seed=5)
    b = _generate("high-acuity", seed=5)
    for name in a:
        assert a[name].equals(b[name]), name


def test_presets_move_the_illness_rates() -> None:
    # high-acuity is sicker than older-cohort (which changes only demographics).
    hi = _generate("high-acuity", n=1000)
    base = _generate("older-cohort", n=1000)

    def imv(ds: dict[str, pl.DataFrame]) -> float:
        h = ds["hospitalization"].height
        return (
            ds["respiratory_support"]
            .filter(pl.col("device_category") == "IMV")["hospitalization_id"]
            .n_unique()
            / h
        )

    assert imv(hi) > imv(base) + 0.1  # meaningfully higher ventilation


def test_rare_support_emits_ecmo_at_small_n() -> None:
    """Teaching preset elevates ECMO so n=1000 is non-empty (unlike network median)."""
    ds = _generate("rare-support", n=1000, seed=11)
    assert ds["ecmo_mcs"].height > 0
    assert ds["crrt_therapy"].height > 0
    assert ds["ecmo_mcs"]["hospitalization_id"].n_unique() >= 1
