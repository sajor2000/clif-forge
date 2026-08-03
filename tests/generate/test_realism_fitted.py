"""Realism checks that synthetic rates stay near the reference when a pack is available.

These tests are **skip-friendly**: they require either a committed ``icu_all28``
pack or ``CLIF_SOURCE_DIR`` pointing at local source CLIF. They never load
real row-level data into assertions beyond aggregate comparisons via the
validation harness.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from clifforge.fit.param_pack import ParamPack

_PACK = Path("data/param_packs/icu_all28")
_SOURCE = Path(os.environ.get("CLIF_SOURCE_DIR", str(Path.home() / "Data" / "clif-source")))
_ROOT = Path(__file__).resolve().parents[2]
_PACK_READY = (_PACK / "manifest.json").is_file()


@pytest.mark.skipif(not _PACK_READY, reason="icu_all28 pack not present")
def test_icu_all28_pack_covers_expected_tables() -> None:
    pack = ParamPack.load(str(_PACK))
    expected = {
        "patient",
        "hospitalization",
        "vitals",
        "labs",
        "medication_admin_continuous",
        "spine",
        "code_status",
        "adt",
        "respiratory_support",
        "medication_admin_intermittent",
        "microbiology_culture",
        "crrt_therapy",
        "ecmo_mcs",
        "patient_assessments",
        "position",
        "hospital_diagnosis",
        "patient_procedures",
    }
    fitted = {name for name, block in pack.tables.items() if block.get("fitted")}
    missing = expected - fitted
    assert not missing, f"pack missing fitted blocks: {sorted(missing)}"
    fs = pack.manifest.get("fit_source") or {}
    assert isinstance(fs.get("dataset_id"), str) and bool(fs["dataset_id"])
    # Fitted knobs the validated ICU path consumes.
    spine = pack.tables["spine"]["params"]
    assert "terminal_deterioration_hours" in spine
    assert "admission_route_marginal" in spine  # retained for full-hospital transform
    adt = pack.tables["adt"]["params"]
    assert "arrival_location_marginal" in adt
    assert "direct_icu_frac" in adt
    rs = pack.tables["respiratory_support"]["params"]
    assert "niv" in rs


@pytest.mark.skipif(
    not (_PACK_READY and _SOURCE.is_dir()),
    reason="need icu_all28 pack and local source CLIF dir",
)
def test_validate_against_source_smoke(tmp_path: Path) -> None:
    """Generate via recalibrate_fitted_icu and compare key rates to the reference."""
    from clifforge.generate.orchestrator import generate_dataset, write_dataset
    from clifforge.generate.recalibrate import recalibrate_fitted_icu

    sys.path.insert(0, str(_ROOT))
    from scripts.validate_against_real import validate

    pack = recalibrate_fitted_icu(ParamPack.load(str(_PACK)))
    out = tmp_path / "synth"
    write_dataset(generate_dataset(pack, n_patients=200, seed=7), out)
    result = validate(out, _SOURCE)
    assert abs(result["mortality"]["synthetic"] - result["mortality"]["real"]) < 0.08
    assert abs(
        result["life_support"]["IMV"]["synthetic"] - result["life_support"]["IMV"]["real"]
    ) < 0.12
    # ADT front door: ED-dominant like reference ICU cohort.
    assert result["adt_arrival"]["synthetic"].get("ed", 0.0) > 0.5
    assert result["adt_arrival"]["synthetic"].get("icu", 0.0) > 0.05
    # LOS median (index 2 of p10/p25/p50/p75/p90) near the reference.
    assert abs(result["los_hours"]["synthetic"][2] - result["los_hours"]["real"][2]) < 40
    # Trajectory audit shape: vitals/labs/support vs real, both arms present.
    for key in ("map", "creatinine", "lactate", "imv_prevalence", "vaso_prevalence"):
        arm = result["trajectory"][key]
        assert "synthetic" in arm and "real" in arm
        assert len(arm["synthetic"]["decedents"]) == 4
    assert "vaso_given_imv" in result["coherence"]["synthetic"]
    assert "decedent_map_falls" in result["coherence"]["real"]
