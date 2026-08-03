"""Unit tests for population recalibration (network-median transform + repairs).

Covers the transform math (escalation/start tempering, sojourn + mortality
scaling), the vitals-dispersion repair, outcome-coupled terminal deterioration in
the spine, and the length-aware gated generator paths (non-invasive level-2
escalation, per-stay vasopressor confinement, denser lab panels). Driven by
constructed packs/spines — no real data.
"""

from __future__ import annotations

import numpy as np

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.recalibrate import (
    recalibrate_to_full_hospital,
    recalibrate_to_network_median,
    repair_vitals_dispersion,
)
from clifforge.generate.spine import (
    FLAG_NAMES,
    SpineFrame,
    _apply_terminal_deterioration,
    _pick_terminal_archetype,
    sample_spine,
)
from clifforge.generate.tables.labs import _panel_intervals
from clifforge.generate.tables.medication_admin_continuous import (
    _VASOPRESSORS,
    sample_medication_admin_continuous,
)
from clifforge.generate.tables.respiratory_support import sample_respiratory_support


def _spine_params(*, expired_rate: float = 0.2) -> dict:
    """A minimal-but-complete spine param block for sampling/recalibration."""
    return {
        "state_model": {"grid_step_hours": 1.0, "horizon_intervals": 240},
        "support_level_start_dist": {"1": 0.4, "2": 0.1, "3": 0.2, "4": 0.29, "5": 0.01},
        "support_level_transition_matrix": {
            "1": {"discharge": 0.4, "2": 0.1, "4": 0.5},
            "2": {"discharge": 0.05, "1": 0.5, "3": 0.05, "4": 0.4},
            "3": {"discharge": 0.01, "1": 0.6, "4": 0.39},
            "4": {"discharge": 0.05, "1": 0.5, "3": 0.4, "5": 0.05},
            "5": {"discharge": 0.1, "4": 0.9},
        },
        "support_level_sojourn": {
            "1": {"family": "lognormal", "mean_hours": 10.0, "params": [1.0, 0.0, 5.0]},
            "2": {"family": "lognormal", "mean_hours": 9.0, "params": [1.0, 0.0, 4.5]},
            "3": {"family": "lognormal", "mean_hours": 5.0, "params": [1.0, 0.0, 2.8]},
            "4": {"family": "lognormal", "mean_hours": 5.0, "params": [0.8, 0.0, 3.3]},
            "5": {"family": "weibull", "mean_hours": 40.0, "params": [0.8, 0.0, 36.0]},
        },
        "flag_prevalence_by_level": {
            str(lvl): {name: 0.1 for name in FLAG_NAMES} for lvl in range(6)
        },
        "expired_rate_by_peak_level": {
            str(lvl): {"expired_rate": expired_rate, "n_hospitalizations": 100} for lvl in range(6)
        },
        "outcome_marginal": {"alive": 1 - expired_rate, "expired": expired_rate},
    }


def _pack(**kw) -> ParamPack:
    return ParamPack(manifest={}, tables={"spine": {"params": _spine_params(**kw)}})


def _vitals_pack() -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "spine": {"params": _spine_params()},
            "vitals": {
                "params": {
                    "map_ar1_by_state": {
                        "1": {"mean": 88.0, "phi": 0.0, "sigma": 6838.0},
                        "4": {"mean": 74.0, "phi": 0.0, "sigma": 205.0},
                    },
                    "spo2_ar1_by_state": {"1": {"mean": 108.0, "phi": 0.0, "sigma": 900.0}},
                }
            },
        },
    )


# --- transform math -------------------------------------------------------- #


def test_recalibrate_does_not_mutate_input() -> None:
    pack = _pack()
    before = pack.tables["spine"]["params"]["support_level_start_dist"]["4"]
    recalibrate_to_network_median(pack)
    assert pack.tables["spine"]["params"]["support_level_start_dist"]["4"] == before


def test_start_law_shaped_to_icu_conditioned_peak_target() -> None:
    out = recalibrate_to_network_median(_pack(), peak_imv_target=0.28).tables["spine"]["params"]
    start = out["support_level_start_dist"]
    # ICU-conditioned target: the non-ventilated ICU mass sits at level 2 (~1 - imv),
    # every stay peaks at the ICU floor or above, and no mass starts below level 2.
    assert abs(start["2"] - 0.72) < 0.02
    assert start.get("0", 0.0) == 0.0 and start.get("1", 0.0) == 0.0
    assert 0.0 < start["4"] < 0.29  # high-acuity mass spread realistically, not piled
    # Escalation to level 4 is tempered in every row that had it.
    row = out["support_level_transition_matrix"]["1"]
    assert row["4"] < 0.5 and row["1"] > 0.0


def test_network_median_peak_target_is_icu_conditioned() -> None:
    from clifforge.generate.recalibrate import _network_median_peak_target

    # A real profile with non-ICU L0/L1 mass and an L4-heavy high band.
    real = {"0": 0.01, "1": 0.34, "2": 0.07, "3": 0.19, "4": 0.34, "5": 0.05}
    target = _network_median_peak_target(real, imv_target=0.41)
    assert "0" not in target and "1" not in target  # ICU-conditioned: no sub-ICU peak
    assert abs(target["2"] - 0.59) < 1e-9  # non-ventilated ICU floor = 1 - imv
    assert abs(sum(target[k] for k in ("3", "4", "5")) - 0.41) < 1e-9  # reaches-IMV rate held
    assert target["4"] > target["3"] > target["5"]  # real high-acuity shape preserved


def test_sojourns_scaled_and_mortality_scaled() -> None:
    # sojourn_shape_scale=1.0 isolates the pure scale multiply (no shape tightening).
    out = recalibrate_to_network_median(_pack(expired_rate=0.2), sojourn_shape_scale=1.0).tables[
        "spine"
    ]["params"]
    # Level-1 sojourn scale (params[2]) multiplied by the default 3.2x, sigma untouched.
    assert out["support_level_sojourn"]["1"]["params"][2] == 5.0 * 3.2
    assert out["support_level_sojourn"]["1"]["params"][0] == 1.0
    # Peak mortality scaled by the default 0.66.
    assert abs(out["expired_rate_by_peak_level"]["4"]["expired_rate"] - 0.2 * 0.66) < 1e-9


def test_sojourn_shape_scale_tightens_lognorm_and_holds_median() -> None:
    import math

    from clifforge.generate.recalibrate import _SOJOURN_MEAN_COMPENSATION

    shape_scale = 0.5
    pack = _pack()
    before = pack.tables["spine"]["params"]["support_level_sojourn"]["1"]["params"]
    sigma0 = before[0]
    scale0 = before[2] * 3.2  # scale after the default 3.2x level-1 multiplier

    lvl1 = recalibrate_to_network_median(pack, sojourn_shape_scale=shape_scale).tables["spine"][
        "params"
    ]["support_level_sojourn"]["1"]["params"]
    # sigma (params[0]) is tightened by the knob — the tail-fattening lever.
    assert abs(lvl1[0] - sigma0 * shape_scale) < 1e-12
    # scale (params[2] == median, since loc == 0) is bumped by the documented partial
    # mean-compensation, so it grows but by less than a full mean-preserving bump.
    bump = sigma0**2 * (1.0 - shape_scale**2) / 2.0 * _SOJOURN_MEAN_COMPENSATION
    assert abs(lvl1[2] - scale0 * math.exp(bump)) < 1e-9
    assert scale0 < lvl1[2] < scale0 * math.exp(sigma0**2 * (1.0 - shape_scale**2) / 2.0)
    # input pack is not mutated (R22).
    assert pack.tables["spine"]["params"]["support_level_sojourn"]["1"]["params"][0] == sigma0


def test_mortality_target_solves_scale_to_hit_target() -> None:
    # With uniform per-level expired rates, expected cohort mortality equals the
    # scaled rate, so the solved scale lands every peak cell exactly on the target.
    target = 0.3
    out = recalibrate_to_network_median(_pack(expired_rate=0.2), mortality_target=target).tables[
        "spine"
    ]["params"]
    for cell in out["expired_rate_by_peak_level"].values():
        assert abs(cell["expired_rate"] - target) < 1e-3
    # The target path overrides the fixed mortality_scale (which would give 0.132).
    assert abs(out["expired_rate_by_peak_level"]["4"]["expired_rate"] - 0.2 * 0.66) > 0.1


def test_mortality_target_unreachable_returns_ceiling_without_diverging() -> None:
    # Zero base expired rates cannot produce any target > 0; the solver returns the
    # bracket ceiling rather than diverging, and no deaths are manufactured (0 * s = 0).
    out = recalibrate_to_network_median(_pack(expired_rate=0.0), mortality_target=0.3).tables[
        "spine"
    ]["params"]
    assert all(cell["expired_rate"] == 0.0 for cell in out["expired_rate_by_peak_level"].values())


def test_derivative_rate_overrides_propagate() -> None:
    # Users spin derivatives by overriding rates; the overrides must reach the pack.
    out = recalibrate_to_network_median(
        _pack(),
        peak_imv_target=0.60,
        crrt_prob=0.50,
        flag_target_prevalence={
            "resp_flag": 0.4,
            "cv_flag": 0.50,
            "renal_flag": 0.1,
            "neuro_flag": 0.1,
        },
    ).tables
    start = out["spine"]["params"]["support_level_start_dist"]
    assert sum(start.get(k, 0.0) for k in ("3", "4", "5")) > 0.5  # higher IMV target
    assert out["crrt_therapy"]["params"]["crrt_prob"] == 0.50
    assert out["spine"]["params"]["flag_target_prevalence"]["cv_flag"] == 0.50


def test_generator_paths_enabled() -> None:
    out = recalibrate_to_network_median(_pack()).tables
    niv = out["respiratory_support"]["params"]["niv"]
    assert niv["nippv_prob"] == 0.064 and niv["hfnc_prob"] == 0.069
    assert out["medication_admin_continuous"]["params"]["vasopressor_per_stay"] is True
    assert out["labs"]["params"]["ward_panel_interval_hours"] is not None
    assert out["spine"]["params"]["terminal_deterioration_hours"] == 24.0


def test_network_median_sets_no_admission_route_marginal() -> None:
    # Backward compatibility (R22): the ICU (network-median) transform must NOT set an
    # admission_route_marginal, so the spine never draws the coupled route and the ICU
    # master / demo output stays byte-for-byte identical (the route draw is skipped and
    # the rng order is unchanged). Only the full-hospital transform enables the route.
    out = recalibrate_to_network_median(_pack()).tables["spine"]["params"]
    assert "admission_route_marginal" not in out


# --- full-hospital transform ----------------------------------------------- #


def test_full_hospital_does_not_mutate_input() -> None:
    pack = _pack()
    before_start = dict(pack.tables["spine"]["params"]["support_level_start_dist"])
    before_soj = pack.tables["spine"]["params"]["support_level_sojourn"]["1"]["params"][2]
    recalibrate_to_full_hospital(pack)
    assert pack.tables["spine"]["params"]["support_level_start_dist"] == before_start
    assert pack.tables["spine"]["params"]["support_level_sojourn"]["1"]["params"][2] == before_soj
    assert "adt" not in pack.tables  # enrichment block added only on the copy


def test_full_hospital_enables_location_enrichment() -> None:
    # The defining difference from the ICU mode: ed/ward/stepdown/icu locations on.
    out = recalibrate_to_full_hospital(_pack()).tables
    assert out["adt"]["params"]["enrich_locations"] is True


def test_full_hospital_sets_coupled_admission_route_marginal() -> None:
    # The single coupled route on the spine drives BOTH the admission type and the
    # arrival location (KTD-6); the adt block no longer carries a separate arrival
    # marginal / direct-ICU fraction (the route supersedes them).
    from clifforge.reference import categories

    out = recalibrate_to_full_hospital(_pack()).tables
    route = out["spine"]["params"]["admission_route_marginal"]
    # Route values ARE mCIDE admission_type_category members, ED-dominant with a small
    # osh referral pathway.
    assert set(route) <= set(categories("hospitalization", "admission_type_category"))
    assert route["ed"] > max(route[k] for k in route if k != "ed")  # ED-dominant
    assert 0.08 <= route["direct"] <= 0.12
    assert 0.02 <= route["osh"] <= 0.04  # thin OSH->ICU referral share
    # The full-hospital adt block is now just location enrichment — no arrival marginal
    # or direct-ICU fraction (the coupled route decides the front door).
    assert out["adt"]["params"] == {"enrich_locations": True}
    # Override propagates onto the deep copy only.
    ov = recalibrate_to_full_hospital(
        _pack(), admission_route_marginal={"ed": 0.6, "osh": 0.4}
    ).tables["spine"]["params"]["admission_route_marginal"]
    assert ov == {"ed": 0.6, "osh": 0.4}


def test_full_hospital_does_not_override_hospitalization_admission_type_marginal() -> None:
    # The coupled route supersedes the hospitalization admission-type marginal, so the
    # full-hospital transform leaves any existing marginal on the hospitalization block
    # untouched (R22: only the spine route marginal is added, on the deep copy).
    pack = ParamPack(
        manifest={},
        tables={
            "spine": {"params": _spine_params()},
            "hospitalization": {"params": {"admission_type_category_marginal": {"ed": 1.0}}},
        },
    )
    out = recalibrate_to_full_hospital(pack).tables
    # input not mutated
    assert pack.tables["hospitalization"]["params"]["admission_type_category_marginal"] == {
        "ed": 1.0
    }
    # the transform did not rewrite the copy's marginal (the route drives admission type)
    assert out["hospitalization"]["params"]["admission_type_category_marginal"] == {"ed": 1.0}
    assert "admission_route_marginal" in out["spine"]["params"]


def test_full_hospital_start_law_is_ward_dominant() -> None:
    # Most start/peak mass sits below the stepdown tier (levels 0-1 = ward/ED), the
    # opposite of the ICU-conditioned mode, which puts all mass at level 2+.
    start = recalibrate_to_full_hospital(_pack()).tables["spine"]["params"][
        "support_level_start_dist"
    ]
    ward = start.get("0", 0.0) + start.get("1", 0.0)
    icu = sum(start.get(k, 0.0) for k in ("3", "4", "5"))
    assert ward > 0.7  # ward/ED dominant
    assert start["2"] < 0.1 < ward  # only a thin stepdown slice
    # A minority reach an ICU tier. The default start-law icu_target (0.08) is now set
    # below the ~0.156 real ICU fraction because the coupled osh route arrives ~3% of
    # stays at icu independent of spine escalation (see recalibrate docstring).
    assert 0.05 < icu < 0.2


def test_full_hospital_tempers_stepdown_and_high_escalation() -> None:
    out = recalibrate_to_full_hospital(_pack()).tables["spine"]["params"]
    row = out["support_level_transition_matrix"]["1"]
    # Escalation to the stepdown tier (L2) and the invasive tier (L4) are both damped
    # so peak tracks start; recovery (L1 self-loop) carries the shed mass.
    assert row.get("2", 0.0) < 0.1
    assert row.get("4", 0.0) < 0.05
    assert row["1"] > 0.0


def test_full_hospital_mortality_target_is_low_by_default() -> None:
    # Full-population mortality (~0.021) is far below the ICU cohort's; the default
    # target solves the peak-coupled scale down from the base expired rates.
    out = recalibrate_to_full_hospital(_pack(expired_rate=0.2)).tables["spine"]["params"]
    # Uniform base rates -> every peak cell lands on the 0.021 target.
    for cell in out["expired_rate_by_peak_level"].values():
        assert cell["expired_rate"] < 0.2  # scaled down, not up


def test_full_hospital_gated_generator_paths_carry_through() -> None:
    out = recalibrate_to_full_hospital(_pack()).tables
    assert out["respiratory_support"]["params"]["niv"]["nippv_prob"] < 0.06  # low-prevalence
    assert out["medication_admin_continuous"]["params"]["vasopressor_per_stay"] is True
    assert out["spine"]["params"]["terminal_deterioration_hours"] == 24.0


# --- vitals repair --------------------------------------------------------- #


def test_repair_vitals_sets_robust_sigma_and_clamps_mean() -> None:
    tables = _vitals_pack().tables
    repair_vitals_dispersion(tables)
    mp = tables["vitals"]["params"]["map_ar1_by_state"]
    assert mp["1"]["sigma"] < 30.0 and mp["4"]["sigma"] < 30.0  # no longer thousands
    # SpO2 mean of 108 is clamped into the physiologic bound (<= 100).
    assert tables["vitals"]["params"]["spo2_ar1_by_state"]["1"]["mean"] <= 100.0


def test_repair_vitals_leaves_physiologic_refit_sigma_untouched() -> None:
    # A pack from the robust re-fit carries physiologic, heteroscedastic per-state
    # sigma; repair must be a no-op so that state-dependent variance is preserved (KTD2).
    tables = {
        "vitals": {
            "params": {
                "map_ar1_by_state": {
                    "2": {"mean": 80.6, "phi": 0.66, "sigma": 11.18},
                    "5": {"mean": 73.4, "phi": 0.66, "sigma": 8.85},
                }
            }
        }
    }
    repair_vitals_dispersion(tables)
    mp = tables["vitals"]["params"]["map_ar1_by_state"]
    assert (
        mp["2"]["sigma"] == 11.18 and mp["5"]["sigma"] == 8.85
    )  # unchanged, still heteroscedastic


# --- terminal deterioration (heterogeneous archetypes) --------------------- #


def _blank_flags(n: int) -> dict[str, list[bool]]:
    return {name: [False] * n for name in FLAG_NAMES}


def test_comfort_archetype_withdraws_support() -> None:
    # Withdrawal: device acuity de-escalates; shock/renal only when Bernoulli says so.
    lvl = [4] * 30
    flags = _blank_flags(30)
    _apply_terminal_deterioration(
        lvl,
        flags,
        24.0,
        1.0,
        np.random.default_rng(0),
        mix={"comfort": 1.0},
        imv_prob=1.0,
        vaso_prob=1.0,
        renal_prob=1.0,
    )
    assert lvl[-1] < 4  # support withdrawn (acuity falls)
    assert lvl[-1] >= 3  # ventilated floor when do_imv
    assert flags["renal_flag"][-1] and flags["cv_flag"][-1]


def test_abrupt_archetype_is_a_short_steep_collapse() -> None:
    lvl = [2] * 30
    flags = _blank_flags(30)
    _apply_terminal_deterioration(
        lvl,
        flags,
        24.0,
        1.0,
        np.random.default_rng(0),
        mix={"abrupt": 1.0},
        imv_prob=1.0,
        vaso_prob=1.0,
        renal_prob=1.0,
    )
    changed = [i for i, v in enumerate(lvl) if v > 2]
    assert changed and min(changed) > 30 - 24  # only a short tail (window // 3) escalated
    assert max(lvl) >= 5  # to the acuity ceiling


def test_prolonged_archetype_ladders_the_organs() -> None:
    lvl = [2] * 30
    flags = _blank_flags(30)
    _apply_terminal_deterioration(
        lvl,
        flags,
        24.0,
        1.0,
        np.random.default_rng(0),
        mix={"prolonged": 1.0},
        imv_prob=1.0,
        vaso_prob=1.0,
        renal_prob=1.0,
    )
    assert flags["resp_flag"][-1] and flags["cv_flag"][-1] and flags["renal_flag"][-1]
    assert max(lvl) >= 4  # laddered escalation tops at high vent (L4), not the ceiling


def test_terminal_never_imv_death_can_stay_noninvasive() -> None:
    lvl = [2] * 30
    flags = _blank_flags(30)
    _apply_terminal_deterioration(
        lvl,
        flags,
        24.0,
        1.0,
        np.random.default_rng(1),
        mix={"prolonged": 1.0},
        imv_prob=0.0,
        vaso_prob=1.0,
        renal_prob=0.0,
    )
    assert max(lvl) == 2
    assert not any(flags["resp_flag"])


def test_terminal_archetypes_vary_across_stays() -> None:
    seen = {
        _pick_terminal_archetype(
            {"abrupt": 0.3, "prolonged": 0.5, "comfort": 0.2}, np.random.default_rng(s)
        )
        for s in range(200)
    }
    assert len(seen) >= 2  # dying courses are not a single stereotyped shape


def test_aggregate_escalation_dominates_but_not_uniform() -> None:
    # With MIMIC-like invent rates, a substantial but not universal share of
    # decedents end at high vent — not stereotyped 100%.
    pack = recalibrate_to_network_median(_pack(expired_rate=1.0), terminal_deterioration_hours=24.0)
    n, high = 200, 0
    for s in range(n):
        sp = sample_spine(pack, np.random.default_rng(s), hospitalization_id=f"H{s}")
        if max(sp.support_level[-24:]) >= 4:
            high += 1
    assert 0.25 < high / n < 0.95


def test_ladder_recouples_cv_and_renal_flags() -> None:
    from clifforge.generate.spine import _recouple_ladder_organs

    lvl = [2, 3, 4, 5]
    flags = _blank_flags(4)
    _recouple_ladder_organs(lvl, flags, np.random.default_rng(0), ladder_cv_prob=1.0)
    assert flags["cv_flag"] == [False, False, True, True]
    assert flags["renal_flag"] == [False, False, False, True]
    # resp stays decoupled — support_level alone drives IMV devices
    assert flags["resp_flag"] == [False, False, False, False]


def test_terminal_deterioration_is_deterministic() -> None:
    pack = recalibrate_to_network_median(_pack(expired_rate=1.0), terminal_deterioration_hours=24.0)
    a = sample_spine(pack, np.random.default_rng(3), hospitalization_id="Hd")
    b = sample_spine(pack, np.random.default_rng(3), hospitalization_id="Hd")
    assert a.support_level == b.support_level and a.cv_flag == b.cv_flag


# --- gated generator paths ------------------------------------------------- #


def _spine_frame(levels: list[int], *, resp: bool, cv: bool) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id="H0",
        support_level=levels,
        resp_flag=[resp] * n,
        cv_flag=[cv] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def test_l2_resp_noninvasive_reserves_imv_for_intubation_tier() -> None:
    # A level-2 stay in respiratory failure: default escalates to IMV, the gated
    # path keeps it non-invasive (no IMV device).
    sp = _spine_frame([2] * 20, resp=True, cv=False)
    default = ParamPack(
        manifest={}, tables={"spine": {"params": {"state_model": {"grid_step_hours": 1.0}}}}
    )
    gated = ParamPack(
        manifest={},
        tables={
            "spine": {"params": {"state_model": {"grid_step_hours": 1.0}}},
            "respiratory_support": {"params": {"l2_resp_noninvasive": True}},
        },
    )
    dev_default = {
        r.device_category for r in sample_respiratory_support(sp, default, np.random.default_rng(0))
    }
    dev_gated = {
        r.device_category for r in sample_respiratory_support(sp, gated, np.random.default_rng(0))
    }
    assert "IMV" in dev_default
    assert "IMV" not in dev_gated


def test_gated_niv_matches_target_prevalence_and_keeps_floor_low_flow() -> None:
    # The gated path assigns non-invasive support once per stay at the target
    # per-stay prevalence (not for every ICU-floor stay), keeps the floor baseline
    # low-flow, and never mints IMV at level 2.
    pack = ParamPack(
        manifest={},
        tables={
            "spine": {"params": {"state_model": {"grid_step_hours": 1.0}}},
            "respiratory_support": {"params": {"niv": {"nippv_prob": 0.10, "hfnc_prob": 0.15}}},
        },
    )
    nippv = hfnc = imv = low_flow = 0
    n = 800
    for s in range(n):
        sp = _spine_frame([2] * 8, resp=True, cv=False)
        devs = {
            r.device_category
            for r in sample_respiratory_support(sp, pack, np.random.default_rng(s))
        }
        nippv += "NIPPV" in devs
        hfnc += "High Flow NC" in devs
        imv += "IMV" in devs
        low_flow += bool({"Nasal Cannula", "Face Mask"} & devs)
    assert 0.07 < nippv / n < 0.13  # ~0.10 target prevalence
    assert 0.11 < hfnc / n < 0.19  # ~0.15 target prevalence
    assert imv == 0  # ICU floor never mints ventilation
    assert low_flow / n > 0.7  # most floor stays are low-flow, not non-invasive


def test_vasopressor_per_stay_confines_pressors_to_cv_stays() -> None:
    marginal = {"norepinephrine": 0.3, "propofol": 0.3, "sodium_chloride": 0.4}
    params = {"med_category_marginal": marginal, "vasopressor_per_stay": True}
    pack = ParamPack(
        manifest={},
        tables={
            "spine": {"params": {"state_model": {"grid_step_hours": 1.0}}},
            "medication_admin_continuous": {"params": params},
        },
    )
    non_cv = _spine_frame([3] * 60, resp=False, cv=False)
    rows = sample_medication_admin_continuous(non_cv, pack, np.random.default_rng(0))
    assert not any(r.med_category in _VASOPRESSORS for r in rows)  # no pressors without cv failure


def test_lab_panel_schedule_is_denser_with_pack_overrides() -> None:
    levels = [2] * 48  # 48h of ICU at grid_step 1.0
    daily = _panel_intervals(levels, 1.0)
    bid = _panel_intervals(levels, 1.0, icu_hours=12.0)
    ward = _panel_intervals([1] * 48, 1.0, ward_hours=24.0)
    assert len(bid) > len(daily)  # twice-daily yields more panels than daily
    assert len(_panel_intervals([1] * 48, 1.0)) == 0 and len(ward) > 0  # ward panels off by default
