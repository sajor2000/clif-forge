"""Unit tests for the latent state spine sampler (U6, KTD-6).

Driven by a small hand-built parameter pack (no real data, no run_fit) so the
spine's contract is checked in isolation: reproducibility, natural termination,
flag/acuity coherence, outcome coupled to peak acuity, and sampled marginals
tracking the pack.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate import spine
from clifforge.generate.spine import sample_spine, truth_frame

# A lognormal dwell (s=0.5, loc=0, scale=5) -> median 5h, right-skewed.
_LOGNORMAL_5H = {"family": "lognormal", "params": [0.5, 0.0, 5.0], "aic": 0.0, "mean_hours": 5.66}


def _spine_pack() -> ParamPack:
    params = {
        "support_level_states": [0, 1, 2],
        "support_level_start_dist": {"0": 0.6, "1": 0.4},
        "support_level_transition_matrix": {
            "0": {"1": 0.7, "discharge": 0.3},
            "1": {"2": 0.5, "discharge": 0.5},
            "2": {"discharge": 1.0},
        },
        "support_level_sojourn": {
            "0": _LOGNORMAL_5H,
            "1": _LOGNORMAL_5H,
            "2": _LOGNORMAL_5H,
        },
        "outcome_marginal": {"alive": 0.7, "expired": 0.3},
        "expired_rate_by_peak_level": {
            "0": {"expired_rate": 0.0, "n_hospitalizations": 100},
            "1": {"expired_rate": 0.1, "n_hospitalizations": 100},
            "2": {"expired_rate": 0.8, "n_hospitalizations": 100},
        },
        "flag_prevalence_by_level": {
            "0": {"resp_flag": 0.0, "cv_flag": 0.0, "renal_flag": 0.0, "neuro_flag": 0.0},
            "1": {"resp_flag": 0.5, "cv_flag": 0.0, "renal_flag": 0.0, "neuro_flag": 0.0},
            "2": {"resp_flag": 1.0, "cv_flag": 1.0, "renal_flag": 0.0, "neuro_flag": 0.0},
        },
        "state_model": {
            "state_model": "organ_support_ladder_v1",
            "grid_step_hours": 1.0,
            "horizon_intervals": 200,
            "rass_deep_sedation_max": -3.0,
            "gcs_low_max": 8.0,
        },
    }
    return ParamPack(
        manifest={},
        tables={"spine": {"n_records": 1000, "fitted": True, "params": params}},
    )


def test_sample_spine_is_deterministic_under_fixed_seed() -> None:
    pack = _spine_pack()
    a = sample_spine(pack, np.random.default_rng(2024))
    b = sample_spine(pack, np.random.default_rng(2024))
    assert a == b


def test_sample_spine_reproducible_byte_for_byte() -> None:
    # A stronger reproducibility check: the serialized long frames are identical.
    pack = _spine_pack()
    a = sample_spine(pack, np.random.default_rng(7)).to_polars()
    b = sample_spine(pack, np.random.default_rng(7)).to_polars()
    assert a.equals(b)


def test_trajectory_terminates_within_horizon() -> None:
    pack = _spine_pack()
    rng = np.random.default_rng(0)
    # Every trajectory has a discharge exit on every row, so it must terminate
    # with a non-empty, horizon-bounded timeline.
    for _ in range(200):
        frame = sample_spine(pack, rng)
        assert 1 <= frame.n_intervals <= 200
        assert frame.outcome in {"expired", "alive"}


def test_flags_are_consistent_with_acuity() -> None:
    # Level 2 has resp/cv prevalence 1.0, so every level-2 interval must carry
    # both flags on; level 0 has all-zero prevalence, so no flags there.
    pack = _spine_pack()
    rng = np.random.default_rng(5)
    for _ in range(200):
        f = sample_spine(pack, rng)
        for lvl, resp, cv, renal, neuro in zip(
            f.support_level, f.resp_flag, f.cv_flag, f.renal_flag, f.neuro_flag, strict=True
        ):
            if lvl == 2:
                assert resp and cv
            if lvl == 0:
                assert not (resp or cv or renal or neuro)


def test_flags_hold_constant_within_a_run() -> None:
    # Flags are drawn once per run and broadcast, so within a maximal constant
    # support-level run at level 1 the resp_flag must not flip interval-to-interval.
    pack = _spine_pack()
    rng = np.random.default_rng(9)
    flips_seen = False
    for _ in range(300):
        f = sample_spine(pack, rng)
        prev_level = None
        prev_resp = None
        for lvl, resp in zip(f.support_level, f.resp_flag, strict=True):
            if lvl == prev_level:
                # same run -> flag must be identical to the previous interval
                assert resp == prev_resp
            else:
                flips_seen = True
            prev_level, prev_resp = lvl, resp
    assert flips_seen  # sanity: runs of length > 1 actually occurred


def test_outcome_couples_to_peak_acuity() -> None:
    # Peak-level-2 trajectories expire ~80%; peak-level-0/1 far less. The gap must
    # be visible in the sampled marginals.
    pack = _spine_pack()
    rng = np.random.default_rng(123)
    peak2_expired, peak2_total = 0, 0
    low_expired, low_total = 0, 0
    for _ in range(1500):
        f = sample_spine(pack, rng)
        if f.peak_level == 2:
            peak2_total += 1
            peak2_expired += f.outcome == "expired"
        elif f.peak_level <= 1:
            low_total += 1
            low_expired += f.outcome == "expired"
    assert peak2_total > 50 and low_total > 50
    peak2_rate = peak2_expired / peak2_total
    low_rate = low_expired / low_total
    assert peak2_rate > 0.65  # near the configured 0.8
    assert low_rate < 0.2  # near the configured 0.0-0.1
    assert peak2_rate - low_rate > 0.4


def test_start_level_marginal_matches_pack() -> None:
    # The first interval's level must follow support_level_start_dist (0.6/0.4).
    pack = _spine_pack()
    rng = np.random.default_rng(31)
    starts = [sample_spine(pack, rng).support_level[0] for _ in range(2000)]
    frac0 = starts.count(0) / len(starts)
    assert abs(frac0 - 0.6) < 0.05


def test_dwell_lengths_are_non_degenerate() -> None:
    # A continuous (lognormal) sojourn family yields varied run lengths, not a
    # single constant dwell — evidence the semi-Markov dwell law is honored.
    pack = _spine_pack()
    rng = np.random.default_rng(17)
    run_lengths: list[int] = []
    for _ in range(300):
        f = sample_spine(pack, rng)
        length = 1
        for i in range(1, f.n_intervals):
            if f.support_level[i] == f.support_level[i - 1]:
                length += 1
            else:
                run_lengths.append(length)
                length = 1
        run_lengths.append(length)
    assert len(set(run_lengths)) > 3
    assert float(np.std(run_lengths)) > 0.0


def test_truth_frame_stacks_spines() -> None:
    pack = _spine_pack()
    rng = np.random.default_rng(1)
    spines = [sample_spine(pack, rng, hospitalization_id=f"H{i}") for i in range(5)]
    frame = truth_frame(spines)
    assert frame["hospitalization_id"].n_unique() == 5
    assert set(frame.columns) == {
        "hospitalization_id",
        "interval_idx",
        "support_level",
        "resp_flag",
        "cv_flag",
        "renal_flag",
        "neuro_flag",
        "outcome",
        "admission_route",
        "resp_phenotype",
    }
    assert truth_frame([]).height == 0


def test_admission_route_empty_without_marginal_and_rng_untouched() -> None:
    # Backward compatibility (R22): a pack with no admission_route_marginal must not
    # draw the route (route stays "") and must leave the rng draw order — and thus the
    # rest of the spine — byte-for-byte identical to a run where the route code did not
    # exist. We prove the rng is untouched by checking that adding the marginal (drawn
    # LAST) does not alter any earlier array.
    pack = _spine_pack()
    plain = sample_spine(pack, np.random.default_rng(42))
    assert plain.admission_route == ""
    assert plain.to_polars()["admission_route"].to_list() == [""] * plain.n_intervals

    routed_pack = _spine_pack()
    routed_pack.tables["spine"]["params"]["admission_route_marginal"] = {
        "ed": 0.7,
        "direct": 0.2,
        "osh": 0.1,
    }
    routed = sample_spine(routed_pack, np.random.default_rng(42))
    # The route is drawn LAST, so every earlier draw is unchanged: trajectory, flags,
    # and outcome are identical to the no-marginal run.
    assert routed.support_level == plain.support_level
    assert routed.resp_flag == plain.resp_flag
    assert routed.cv_flag == plain.cv_flag
    assert routed.outcome == plain.outcome
    # ...and the route itself is now a drawn mCIDE admission_type_category member.
    assert routed.admission_route in {"ed", "direct", "osh"}


def test_admission_route_marginal_matches_pack() -> None:
    # The drawn route follows admission_route_marginal within sampling error.
    pack = _spine_pack()
    pack.tables["spine"]["params"]["admission_route_marginal"] = {
        "ed": 0.76,
        "elective": 0.10,
        "direct": 0.10,
        "osh": 0.03,
        "facility": 0.005,
        "other": 0.005,
    }
    rng = np.random.default_rng(11)
    routes = [sample_spine(pack, rng).admission_route for _ in range(4000)]
    assert abs(routes.count("ed") / len(routes) - 0.76) < 0.03
    assert abs(routes.count("osh") / len(routes) - 0.03) < 0.02


def test_missing_spine_block_raises() -> None:
    empty = ParamPack(manifest={}, tables={})
    try:
        sample_spine(empty, np.random.default_rng(0))
    except ValueError as exc:
        assert "spine" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a pack with no spine block")


def test_module_exports() -> None:
    assert set(spine.__all__) == {
        "FLAG_NAMES",
        "RESP_PHENOTYPES",
        "SpineFrame",
        "sample_spine",
        "truth_frame",
    }
    assert isinstance(pl.DataFrame, type)
