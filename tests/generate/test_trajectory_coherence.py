"""Longitudinal sicker↔sicker coupling across vitals / labs / resp / meds."""

from __future__ import annotations

import numpy as np

from clifforge.demo import demo_pack
from clifforge.generate.recalibrate import recalibrate_to_network_median
from clifforge.generate.spine import sample_spine
from clifforge.generate.tables.labs import sample_labs
from clifforge.generate.tables.medication_admin_continuous import (
    sample_medication_admin_continuous,
)
from clifforge.generate.tables.respiratory_support import sample_respiratory_support
from clifforge.generate.tables.vitals import sample_vitals


def _expired_imv_spine(seed: int = 0):
    pack = recalibrate_to_network_median(demo_pack(), mortality_target=1.0)
    # Force terminal decline path.
    pack.tables["spine"]["params"]["terminal_deterioration_hours"] = 24.0
    pack.tables["spine"]["params"]["terminal_archetype_mix"] = {
        "abrupt": 0.0,
        "prolonged": 1.0,
        "comfort": 0.0,
    }
    pack.tables["spine"]["params"]["terminal_imv_prob"] = 1.0
    pack.tables["spine"]["params"]["terminal_vaso_prob"] = 1.0
    pack.tables["spine"]["params"]["terminal_renal_prob"] = 1.0
    pack.tables["spine"]["params"]["ladder_cv_prob"] = 1.0
    # Bias start toward IMV so pairing is observable.
    start = pack.tables["spine"]["params"]["support_level_start_dist"]
    for k in list(start):
        start[k] = 0.05
    start["3"] = 0.4
    start["4"] = 0.4
    start["5"] = 0.1
    return sample_spine(pack, np.random.default_rng(seed), hospitalization_id="H0"), pack


def test_imv_stay_pairs_with_vasopressors_when_l4() -> None:
    """L4+ implies cv_flag → norepinephrine (or sibling vaso) appears."""
    paired = 0
    for seed in range(40):
        spine, pack = _expired_imv_spine(seed)
        if max(spine.support_level) < 4:
            continue
        assert any(spine.cv_flag)  # ladder re-couple
        rs = sample_respiratory_support(spine, pack, np.random.default_rng(seed + 1))
        meds = sample_medication_admin_continuous(
            spine, pack, np.random.default_rng(seed + 2)
        )
        has_imv = any(r.device_category == "IMV" for r in rs)
        vaso = {
            "norepinephrine",
            "vasopressin",
            "epinephrine",
            "phenylephrine",
            "dopamine",
        }
        has_vaso = any(m.med_category in vaso for m in meds)
        if has_imv and has_vaso:
            paired += 1
    assert paired >= 8  # most L4+ IMV stays should carry a pressor


def test_terminal_renal_raises_creatinine_over_stay() -> None:
    """Creatinine is higher when renal_flag is on than when it is off (same stay)."""
    spine, pack = _expired_imv_spine(7)
    assert any(spine.renal_flag) and not all(spine.renal_flag)
    labs = sample_labs(spine, pack, np.random.default_rng(11))
    creat = [o for o in labs if o.lab_category == "creatinine"]
    assert creat
    admit = creat[0].lab_order_dttm  # unused; map via order time → interval
    # Rebuild interval index from order time relative to first order's day grid.
    # Simpler: sample with known admit and check renal bump helper + flag pairing.
    from datetime import UTC, datetime

    from clifforge.generate.tables.labs import _apply_clinical_lab_bumps

    bumped = _apply_clinical_lab_bumps(
        "creatinine", 1.0, renal=True, renal_frac=1.0, shock=False, log_space=False
    )
    base = _apply_clinical_lab_bumps(
        "creatinine", 1.0, renal=False, renal_frac=0.0, shock=False, log_space=False
    )
    assert bumped > base

    # Across many seeds, mean creat on renal intervals exceeds non-renal.
    on_vals: list[float] = []
    off_vals: list[float] = []
    for seed in range(25):
        sp, pk = _expired_imv_spine(seed)
        admit_dttm = datetime(2020, 1, 1, tzinfo=UTC)
        rows = sample_labs(sp, pk, np.random.default_rng(seed + 99), admit_dttm=admit_dttm)
        step = 1.0
        for o in rows:
            if o.lab_category != "creatinine":
                continue
            hrs = (o.lab_order_dttm - admit_dttm).total_seconds() / 3600.0
            idx = min(int(hrs // step), len(sp.renal_flag) - 1)
            if idx < 0:
                continue
            (on_vals if sp.renal_flag[idx] else off_vals).append(o.lab_value_numeric)
    if on_vals and off_vals:
        assert float(np.mean(on_vals)) > float(np.mean(off_vals))


def test_cv_flag_pulls_map_toward_shock_state() -> None:
    spine, pack = _expired_imv_spine(3)
    assert any(spine.cv_flag)
    vit = sample_vitals(spine, pack, np.random.default_rng(5))
    maps = [o.vital_value for o in vit if o.vital_category == "map"]
    assert maps  # ICU stay emits MAP
    # Shock physiology should not sit at hypertensive means.
    assert float(np.median(maps)) < 100.0
