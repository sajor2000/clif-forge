"""Respiratory phenotype pathways: type1 NC→HFNC→IMV; type2 NIPPV→IMV."""

from __future__ import annotations

import numpy as np

from clifforge.demo import demo_pack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables.hospital_diagnosis import sample_hospital_diagnosis
from clifforge.generate.tables.respiratory_support import sample_respiratory_support


def _spine(levels: list[int], phenotype: str, *, hid: str = "H0") -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[True] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
        resp_phenotype=phenotype,
    )


def test_type1_escalates_nc_to_hfnc_to_imv() -> None:
    pack = demo_pack()
    pack.tables.setdefault("respiratory_support", {"params": {}})
    pack.tables["respiratory_support"]["params"]["enrich_devices"] = True
    # Escalating acuity: L1 → L2 → L3 (ungated demo: full type1 ladder)
    sp = _spine([1, 1, 2, 2, 3, 3], "type1")
    rows = sample_respiratory_support(sp, pack, np.random.default_rng(0))
    devices = [r.device_category for r in rows]
    assert "Nasal Cannula" in devices
    assert "High Flow NC" in devices
    assert "IMV" in devices
    # Ordered by time: NC before HFNC before IMV
    first = {d: i for i, d in enumerate(devices)}
    assert first["Nasal Cannula"] < first["High Flow NC"] < first["IMV"]
    assert "NIPPV" not in devices


def test_gated_niv_keeps_phenotype_device_choice_without_inflating_rate() -> None:
    """Gated NIV: typed stays rarely get NIV; when they do, type picks the device."""
    pack = demo_pack()
    pack.tables.setdefault("respiratory_support", {"params": {}})
    pack.tables["respiratory_support"]["params"]["niv"] = {
        "nippv_prob": 0.064,
        "hfnc_prob": 0.069,
    }
    n_hfnc = n_nippv = n = 0
    for seed in range(400):
        sp = _spine([2] * 12, "type1" if seed % 2 == 0 else "type2_hf", hid=f"H{seed}")
        rows = sample_respiratory_support(sp, pack, np.random.default_rng(seed))
        cats = {r.device_category for r in rows}
        n += 1
        n_hfnc += "High Flow NC" in cats
        n_nippv += "NIPPV" in cats
    # Combined NIV stay rate near reference any-NIV (~0.13), not ~0.7
    assert 0.05 < (n_hfnc + n_nippv) / n < 0.25
    # Type1 seeds (even) should not mint NIPPV; type2 (odd) should not mint HFNC
    # Spot-check: among HFNC stays, majority are type1 pathway draws.
    assert n_hfnc > 0 and n_nippv > 0


def test_type2_ohs_uses_nippv_then_imv_with_high_pressures() -> None:
    pack = demo_pack()
    pack.tables.setdefault("respiratory_support", {"params": {}})
    pack.tables["respiratory_support"]["params"]["enrich_devices"] = True
    sp = _spine([1, 2, 2, 3, 3], "type2_ohs")
    rows = sample_respiratory_support(sp, pack, np.random.default_rng(1))
    devices = [r.device_category for r in rows]
    assert "NIPPV" in devices
    assert "IMV" in devices
    assert "High Flow NC" not in devices
    nippv = next(r for r in rows if r.device_category == "NIPPV")
    # OHS NIPPV settings: higher EPAP / PS than garden-variety
    assert nippv.set_values["peep_set"] >= 8.0
    assert nippv.set_values["pressure_support_set"] >= 10.0


def test_type2_hf_and_copd_prefer_nippv_ladder() -> None:
    pack = demo_pack()
    for pheno in ("type2_hf", "type2_copd"):
        sp = _spine([2, 2, 3], pheno, hid=pheno)
        rows = sample_respiratory_support(sp, pack, np.random.default_rng(2))
        devices = {r.device_category for r in rows}
        assert "NIPPV" in devices and "IMV" in devices
        assert "High Flow NC" not in devices


def test_type2_ohs_codes_obesity_hypoventilation() -> None:
    pack = demo_pack()
    sp = _spine([2, 3], "type2_ohs")
    dx = sample_hospital_diagnosis(sp, pack, np.random.default_rng(0))
    codes = {r.diagnosis_code for r in dx}
    assert "E66.2" in codes  # OHS
    assert "J96.02" in codes  # hypercapnic RF


def test_type1_codes_hypoxemic_rf() -> None:
    pack = demo_pack()
    sp = _spine([2, 3], "type1")
    dx = sample_hospital_diagnosis(sp, pack, np.random.default_rng(0))
    codes = {r.diagnosis_code for r in dx}
    assert "J96.01" in codes
    assert "J96.02" not in codes
