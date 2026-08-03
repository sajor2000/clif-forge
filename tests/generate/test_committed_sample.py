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
    not table_parquet_path(_SAMPLE, "hospitalization").exists(),
    reason="requires the committed sample",
)


#: Encounters compared. ``SeedSequence(seed).spawn`` assigns each encounter a
#: stable key regardless of how many are generated, so regenerating the first 30
#: must reproduce the committed sample's first 30 exactly.
_N = 30


def _compare(sample_dir: Path) -> None:
    """Assert every table of the committed sample reproduces from its recipe.

    Checking *every* table matters: an earlier version of this test compared only
    ``hospitalization``, so it kept passing while nine tables were added and five
    others changed shape underneath it. A reproducibility test that only looks at
    the one table nobody edits does not test reproducibility.
    """
    spec = load_spec(sample_dir / "spec.toml")
    base_path = Path(spec.base_pack) if spec.base_pack else _BASE
    pack = spec_to_pack(spec, ParamPack.load(str(base_path)))
    regenerated = generate_dataset(pack, n_patients=_N, seed=spec.seed).tables

    compared = 0
    for table, regen in regenerated.items():
        path = table_parquet_path(sample_dir, table)
        assert path.exists(), f"{sample_dir}/{path.name} is missing — regenerate the sample"

        # Slice the committed dataset down to the same encounters. The ids are read
        # off the regenerated frame rather than assumed, because patient and
        # hospitalization ids are numbered from different offsets. Tables keyed on
        # neither id (microbiology_susceptibility joins on organism_id alone) are
        # covered transitively by the parent whose ids they carry.
        key = next((c for c in ("hospitalization_id", "patient_id") if c in regen.columns), None)
        if key is None:
            continue
        ids = regen[key].unique().to_list()
        if not ids:
            continue  # table is empty at n=30; nothing to compare against
        committed = pl.read_parquet(path).filter(pl.col(key).is_in(ids))

        assert regen.columns == committed.columns, (
            f"{table}: committed sample has columns {committed.columns}, "
            f"generator now produces {regen.columns} — regenerate the sample"
        )
        order = regen.columns
        assert regen.sort(order).equals(committed.sort(order)), (
            f"{table} does not reproduce from its recipe — regenerate the sample"
        )
        compared += 1
    assert compared > 20, f"only {compared} tables compared; expected the full canonical set"


def test_committed_sample_reproduces_from_its_recipe() -> None:
    _compare(_SAMPLE)


@pytest.mark.skipif(
    not table_parquet_path(_FULL_SAMPLE, "hospitalization").exists(),
    reason="requires the committed full-hospital sample",
)
def test_committed_full_hospital_sample_reproduces_from_its_recipe() -> None:
    """Same contract as the ICU sample, through the ``mode = "full_hospital"`` spec path."""
    _compare(_FULL_SAMPLE)
