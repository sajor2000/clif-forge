"""Population recalibration: reshape a source-cohort-fitted pack to a target ICU cohort.

The raw fit is faithful to its source cohort, but that cohort is one high-acuity
academic center: its fitted spine over-concentrates trajectories at ventilator acuity
(sampled peak level 4 ~0.59 vs ~0.34 in the real per-hospitalization peak the fit
records) and terminates them too quickly (sampled ICU LOS ~28h vs the ~47h real
median). That single distortion cascades: it inflates invasive-ventilation and
mortality prevalence, and — because every time-series table (vitals, labs,
medications, devices) is driven by the spine's interval count — it starves each
stay of the measurements a real multi-day ICU course accumulates, leaving stays
~13x too sparse.

:func:`recalibrate_to_network_median` applies one principled transform whose
targets are all *measured real quantities*, not free knobs:

* **Temper escalation and realign the peak-acuity shape.** High-acuity transition
  mass (to levels 3-5) is tempered toward recovery, and the peak-acuity
  distribution is shaped through the start law to an ICU-conditioned, real-shape
  target at the network-median ventilation rate (see :func:`_network_median_peak_target`)
  — so the fraction reaching invasive ventilation and peak-coupled mortality land
  at the network median *without shortening the stay* and with a realistically
  spread acuity mix, the opposite of trimming the trajectory to hit a rate.
* **Scale sojourns to real LOS.** Level dwell-times are scaled (on the fitted
  distribution's scale parameter, preserving family) so hospital and ICU
  length-of-stay medians match the real cohort — which restores per-stay
  measurement density as a side effect. A separate ``sojourn_shape_scale`` then
  shrinks each log-normal level's log-scale ``sigma`` at fixed sojourn mean, pulling
  the summed-LOS tails in toward the real cohort's spread without moving its median
  (multiplying the scale alone leaves ``sigma`` — and so the fat tails — untouched).
* **Scale peak mortality to the target network rate.**
* **Decouple the organ-failure axes** so cardiovascular / renal support
  prevalence (vasopressors, CRRT) track their own real rates rather than the
  respiratory acuity distribution, and **enable the length-aware generator
  paths** (non-invasive level-2 escalation, per-stay vasopressor confinement,
  denser whole-stay lab panels) that keep life-support *prevalence* a per-stay
  property once stays are realistically long.

The defaults target the CLIF network median (in-hospital mortality ~9.5%,
invasive ventilation ~41%, ICU LOS ~47h) rather than the source cohort's own higher-acuity
figures, so the output represents a generic US ICU and is statistically distinct
from its source cohort while remaining in the real cross-site envelope. All spine
edits operate on a deep copy; the input pack is never mutated (R22).
"""

from __future__ import annotations

import copy
import math
from typing import Any

from clifforge.fit.estimators import DISCHARGE_STATE
from clifforge.fit.param_pack import ParamPack
from clifforge.reference import bounds

__all__ = [
    "recalibrate_to_full_hospital",
    "recalibrate_to_network_median",
    "recalibrate_fitted_icu",
    "repair_vitals_dispersion",
]

#: High-acuity (invasive-ventilation) levels whose start/escalation mass is tempered.
_HIGH_LEVELS: tuple[str, ...] = ("3", "4", "5")

#: Robust within-physiologic-bounds dispersion (1.4826 x MAD) per vital, measured
#: on real CLIF. The fitted AR(1) ``sigma`` is corrupted by extreme charting
#: artifacts (e.g. MAP sigma ~4000 mmHg), so the generated walk otherwise pins
#: values to the outlier-clamp bounds; these aggregate real dispersions replace it.
#: (The estimator fix in ``fit._fit_ar1`` prevents this in future fits; this repairs
#: packs already fitted with the un-robust estimator.)
_ROBUST_VITAL_SD: dict[str, float] = {
    "heart_rate": 17.79,
    "sbp": 22.24,
    "dbp": 13.34,
    "map": 14.83,
    "respiratory_rate": 5.93,
    "spo2": 2.97,
    "temp_c": 0.44,
}

#: A fitted per-state σ above this multiple of the robust real dispersion is treated
#: as a corrupted (pre-robust-estimator) value and replaced; physiologic re-fit σ
#: sits at ~1× and is left untouched, preserving its heteroscedasticity.
_CORRUPT_SIGMA_FACTOR: float = 3.0

#: Fraction of the mean-preserving scale bump applied when tightening a log-normal
#: sojourn's shape (see :func:`_scale_sojourns`). At 1.0 the per-sojourn *mean* is
#: held exactly, but summing many tightened (less-skewed) sojourns then pulls the
#: LOS *median* up toward that fixed mean; under-compensating (< 1) trims the scale
#: back so the summed-LOS median holds while the tails still contract. Measured
#: against real CLIF: at full compensation the hospital-LOS median overshot
#: ~165h→~193h, and 0.7 restores it (~163h) with tails tracking real.
_SOJOURN_MEAN_COMPENSATION: float = 0.7


def _renormalize(dist: dict[str, float]) -> None:
    """In-place renormalize a probability dict to sum to 1 (no-op if empty/zero)."""
    total = sum(dist.values())
    if total > 0:
        for key in dist:
            dist[key] /= total


def _temper_transitions(
    matrix: dict[str, dict[str, float]], *, esc_keep: float, disch_damp: float
) -> None:
    """Temper high-acuity escalation and premature discharge.

    ``esc_keep`` is the fraction of each row's mass toward high levels that is
    kept; the rest flows to level 1 (0.7) as recovery/de-escalation and level 2
    (0.3), so the peak-acuity distribution is driven by the start law (shaped by
    :func:`_apply_peak_target`) rather than by escalation. ``disch_damp`` multiplies
    the discharge probability (< 1 lengthens stays); the freed mass returns to
    level 1. Rows are renormalized in place.
    """
    for row in matrix.values():
        moved = 0.0
        for level in _HIGH_LEVELS:
            if level in row:
                shed = row[level] * (1.0 - esc_keep)
                row[level] -= shed
                moved += shed
        row["1"] = row.get("1", 0.0) + moved * 0.7  # de-escalate toward recovery (L1)
        row["2"] = row.get("2", 0.0) + moved * 0.3
        if DISCHARGE_STATE in row:
            disch = row[DISCHARGE_STATE]
            freed = disch * (1.0 - disch_damp)
            row[DISCHARGE_STATE] = disch * disch_damp
            row["1"] = row.get("1", 0.0) + freed
        _renormalize(row)


def _temper_stepdown_escalation(
    matrix: dict[str, dict[str, float]], *, stepdown_keep: float
) -> None:
    """Temper escalation into the stepdown tier (level 2), routing the rest to recovery.

    :func:`_temper_transitions` only tempers the invasive-ventilation tier (levels
    3-5); its de-escalation even *adds* mass to level 2. In the ICU cohort that is
    the floor and is fine, but the full hospital population must keep most stays at
    ward acuity, so escalation into level 2 would otherwise pull the stepdown
    fraction far above its ~8% target. This keeps only ``stepdown_keep`` of each
    row's mass toward level 2, routing the rest to level 1 (recovery), so a stay's
    peak tier tracks its start tier for stepdown just as it does for the ICU tiers.
    Rows are renormalized in place; run **after** :func:`_temper_transitions`.
    """
    for row in matrix.values():
        if "2" in row:
            shed = row["2"] * (1.0 - stepdown_keep)
            row["2"] -= shed
            row["1"] = row.get("1", 0.0) + shed
        _renormalize(row)


def _real_peak_profile(spine_params: dict[str, Any]) -> dict[str, float]:
    """Real per-hospitalization peak-level distribution, from the fit's peak counts."""
    by_peak = spine_params.get("expired_rate_by_peak_level", {})
    counts = {k: float(v.get("n_hospitalizations", 0)) for k, v in by_peak.items()}
    total = sum(counts.values()) or 1.0
    return {k: n / total for k, n in counts.items()}


def _network_median_peak_target(real: dict[str, float], imv_target: float) -> dict[str, float]:
    """ICU-conditioned network-median peak-level target.

    The dataset is an **ICU** cohort, so every stay peaks at the ICU floor (level 2)
    or above — the full-cohort real profile's large level-0/1 mass is non-ICU floor
    stays and must not be targeted. This puts the non-ventilated ICU mass at level 2
    (``1 - imv_target``) and spreads the ventilated mass (``imv_target``, the
    network-median reaches-IMV rate) across levels 3-5 in the **real** high-acuity
    proportions — which corrects the earlier shape's under-representation of the
    highest level while holding the ventilation rate.
    """
    high = {k: v for k, v in real.items() if int(k) >= 3}
    hs = sum(high.values()) or 1.0
    target = {k: imv_target * v / hs for k, v in high.items()}
    target["2"] = 1.0 - imv_target  # ICU floor: non-ventilated ICU stays peak at L2
    return target


def _full_hospital_peak_target(
    real: dict[str, float], icu_target: float, stepdown_target: float
) -> dict[str, float]:
    """Full-hospital-population peak-level target (ward-dominant, not ICU-conditioned).

    Unlike :func:`_network_median_peak_target` (every stay is an ICU stay), the full
    hospital population is dominated by ward/ED stays that never reach an ICU tier:
    only ``icu_target`` of stays peak at invasive-ventilation acuity (level >= 3, an
    ``icu`` ADT location), ``stepdown_target`` peak at the high-flow/NIV tier (level
    2, ``stepdown``), and the rest stay at ward/ED acuity (levels 0-1). The
    reaches-ICU mass is spread across levels 3-5 in the **real** high-acuity
    proportions and the ward mass across levels 0-1 in the real low-acuity
    proportions (from the fit's per-peak hospitalization counts), so the within-tier
    acuity shape stays realistic while the tier fractions hit the measured full-pop
    targets.
    """
    high = {k: v for k, v in real.items() if int(k) >= 3}
    hs = sum(high.values()) or 1.0
    ward = {k: v for k, v in real.items() if int(k) <= 1}
    ws = sum(ward.values()) or 1.0
    ward_mass = max(0.0, 1.0 - icu_target - stepdown_target)
    target = {k: icu_target * v / hs for k, v in high.items()}
    target["2"] = stepdown_target
    for k, v in ward.items():
        target[k] = ward_mass * v / ws
    return target


def _apply_peak_target(start_dist: dict[str, float], target: dict[str, float]) -> None:
    """Shape the start distribution to the target peak profile (in place).

    With escalation heavily tempered the trajectory's peak tracks its start level,
    so setting the start law to the desired peak profile realigns the sampled
    peak-acuity distribution across levels in real proportions instead of piling
    it at level 2 (which the earlier level-2 routing produced).
    """
    start_dist.clear()
    for level, prob in target.items():
        start_dist[str(level)] = float(prob)
    _renormalize(start_dist)


def _scale_sojourns(
    sojourn: dict[str, dict[str, Any]],
    multipliers: dict[str, float],
    shape_scale: float = 1.0,
) -> None:
    """Scale each level's dwell-time and optionally tighten its multiplicative spread.

    Both fitted families (scipy ``lognorm`` / ``weibull_min``) carry their scale in
    ``params[2]`` with the location at ``params[1]``; the mean is proportional to
    that scale, so multiplying it scales the mean while preserving the fitted
    family and shape. ``mean_hours`` (metadata) is kept consistent.

    ``shape_scale`` (< 1) then tightens each **log-normal** level's tails by
    shrinking its log-scale ``sigma`` (``params[0]``) to ``shape_scale * sigma``.
    A lognorm's mean is ``scale * exp(sigma**2 / 2)``, so ``scale`` is grown back by
    ``exp(sigma**2 * (1 - shape_scale**2) / 2 * c)`` where ``c`` is
    :data:`_SOJOURN_MEAN_COMPENSATION`; ``c == 1`` holds the per-sojourn mean exactly
    but — because summing less-skewed sojourns lifts the median toward that mean —
    overshoots the *summed* LOS median, so ``c`` under-compensates to keep it fixed.
    Hospital LOS is a sum of per-level sojourns, so its median holds while its tails,
    set by the per-sojourn multiplicative spread, contract; a plain ``params[2]``
    multiply leaves ``sigma`` untouched and so cannot pull the tails in.
    ``shape_scale == 1.0`` is an exact no-op; Weibull levels keep their fitted shape.
    """
    for level, mult in multipliers.items():
        block = sojourn.get(level)
        if block is None:
            continue
        params = block.get("params")
        if isinstance(params, list) and len(params) >= 3:
            params[2] *= mult
        mean = block.get("mean_hours")
        if isinstance(mean, int | float):
            block["mean_hours"] = mean * mult
    if shape_scale == 1.0:
        return
    for block in sojourn.values():
        if not str(block.get("family", "")).startswith("lognorm"):
            continue
        params = block.get("params")
        if isinstance(params, list) and len(params) >= 3:
            sigma = float(params[0])
            params[0] = sigma * shape_scale
            bump = sigma**2 * (1.0 - shape_scale**2) / 2.0 * _SOJOURN_MEAN_COMPENSATION
            params[2] = float(params[2]) * math.exp(bump)


def _expected_mortality(
    peak_dist: dict[str, float],
    expired_by_peak: dict[str, dict[str, float]],
    marginal_expired: float,
    scale: float,
) -> float:
    """Analytic cohort mortality under a peak distribution and a mortality scale.

    Because escalation is heavily tempered, a stay's peak acuity tracks its start
    level, so ``peak_dist`` (the start law) approximates the sampled peak
    distribution. Each level contributes its (scaled, rate-capped) expired rate.
    """
    e = 0.0
    for level, prob in peak_dist.items():
        cell = expired_by_peak.get(str(level))
        rate = float(cell["expired_rate"]) if cell else marginal_expired
        e += prob * min(1.0, rate * scale)
    return e


def _solve_mortality_scale(
    peak_dist: dict[str, float],
    expired_by_peak: dict[str, dict[str, float]],
    marginal_expired: float,
    target: float,
) -> float:
    """Bisection for the mortality scale that lands cohort mortality on ``target``.

    Monotone in ``scale`` (each term is non-decreasing), so bisection converges.
    Falls back to the upper bracket when the target exceeds what capping allows.
    """
    lo, hi = 0.0, 100.0
    if _expected_mortality(peak_dist, expired_by_peak, marginal_expired, hi) < target:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _expected_mortality(peak_dist, expired_by_peak, marginal_expired, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def repair_vitals_dispersion(tables: dict[str, Any]) -> None:
    """Repair *corrupted* AR(1) ``sigma`` and clamp state means into physiologic range.

    A legacy pack fitted with the pre-robust estimator can carry an implausible
    ``sigma`` (e.g. a MAP σ of ~4000 mmHg) that pins the generated walk to the
    outlier bounds. This replaces **only** such corrupted values — a per-state σ
    above ``_CORRUPT_SIGMA_FACTOR`` × the robust real dispersion — with that robust
    value, and clamps impossible means (e.g. an SpO2 of 108) into bounds. A pack
    from the robust re-fit already carries physiologic, *heteroscedastic* per-state
    σ, so this is a no-op on it and its state-dependent variance is preserved
    (KTD2). Operates in place on a pack's ``tables`` dict.
    """
    block = tables.get("vitals")
    if not isinstance(block, dict) or "params" not in block:
        return
    params = block["params"]
    for vital, robust_sd in _ROBUST_VITAL_SD.items():
        by_state = params.get(f"{vital}_ar1_by_state")
        if not isinstance(by_state, dict):
            continue
        lower, upper = bounds("vitals", vital)
        for cell in by_state.values():
            if float(cell["sigma"]) > _CORRUPT_SIGMA_FACTOR * robust_sd:
                cell["sigma"] = robust_sd  # corrupted fit — replace
            cell["mean"] = min(max(float(cell["mean"]), lower), upper)


def recalibrate_to_network_median(
    pack: ParamPack,
    *,
    esc_keep: float = 0.04,
    peak_imv_target: float = 0.28,
    peak_target: dict[str, float] | None = None,
    disch_damp: float = 0.55,
    sojourn_multipliers: dict[str, float] | None = None,
    sojourn_shape_scale: float = 0.6,
    mortality_scale: float = 0.66,
    mortality_target: float | None = None,
    flag_target_prevalence: dict[str, float] | None = None,
    prone_prob_severe: float = 0.026,
    niv_nippv_prob: float = 0.064,
    niv_hfnc_prob: float = 0.069,
    vasopressor_cv_boost: float = 3.0,
    lab_panel_interval_hours: float = 12.0,
    lab_ward_panel_interval_hours: float = 18.0,
    terminal_deterioration_hours: float = 24.0,
    crrt_prob: float = 0.29,
    repair_vitals: bool = True,
) -> ParamPack:
    """Return a deep-copied pack recalibrated to the CLIF network-median ICU cohort.

    The defaults were tuned against real CLIF so the generated cohort's
    length-of-stay, organ-support prevalences, and per-stay measurement density
    land in the real cross-site range (see module docstring). Every argument is a
    documented lever over that transform; the input pack is not mutated.
    """
    tables = copy.deepcopy(dict(pack.tables))
    spine = tables["spine"]["params"]

    _temper_transitions(
        spine["support_level_transition_matrix"], esc_keep=esc_keep, disch_damp=disch_damp
    )
    # Realign the peak-acuity distribution to a realistic (real-shape) profile at the
    # network-median ventilation rate, driven through the start law (peak tracks start
    # once escalation is tempered) instead of piling peak mass at level 2.
    target = peak_target or _network_median_peak_target(_real_peak_profile(spine), peak_imv_target)
    _apply_peak_target(spine["support_level_start_dist"], target)
    _scale_sojourns(
        spine["support_level_sojourn"],
        sojourn_multipliers or {"1": 3.2, "2": 4.6, "3": 2.8, "4": 2.8},
        shape_scale=sojourn_shape_scale,
    )
    # Mortality: scale peak-coupled expired rates. ``mortality_target`` (an exact
    # cohort in-hospital mortality) takes precedence — the scale is solved
    # analytically against the peak-target distribution — otherwise the fixed
    # ``mortality_scale`` is applied.
    if mortality_target is not None:
        eff_mortality_scale = _solve_mortality_scale(
            target,
            spine["expired_rate_by_peak_level"],
            float(spine.get("outcome_marginal", {}).get("expired", 0.0)),
            mortality_target,
        )
    else:
        eff_mortality_scale = mortality_scale
    for cell in spine["expired_rate_by_peak_level"].values():
        cell["expired_rate"] = min(1.0, cell["expired_rate"] * eff_mortality_scale)

    # Outcome-coupled terminal deterioration: expiring stays decline into
    # multi-organ failure over their final window (see spine._apply_...).
    spine["terminal_deterioration_hours"] = terminal_deterioration_hours

    if repair_vitals:
        repair_vitals_dispersion(tables)

    # Decoupled organ axes: each failure flag drawn once per stay at a marginal
    # target, active across ICU time (see spine.sample_spine).
    spine["flag_target_prevalence"] = flag_target_prevalence or {
        "resp_flag": 0.5,
        "cv_flag": 0.27,
        "renal_flag": 0.05,
        "neuro_flag": 0.2,
    }

    # Length-aware generator paths: keep life-support *prevalence* a per-stay
    # property and restore lab volume once stays are realistically long.
    tables["adt"] = {"params": {"enrich_locations": False}}
    # Non-invasive support (NIPPV / High Flow NC) is a low-prevalence per-stay
    # property gated to real ICU rates, not minted for every ICU-floor stay; the
    # gated path also keeps IMV to the intubation tier (level >= 3), so the floor
    # never produces spurious ventilation.
    tables["respiratory_support"] = {
        "params": {
            "enrich_devices": True,
            "niv": {"nippv_prob": niv_nippv_prob, "hfnc_prob": niv_hfnc_prob},
        }
    }
    med = dict(tables.get("medication_admin_continuous", {}))
    med_params = dict(med.get("params", {}))
    med_params["vasopressor_per_stay"] = True
    med_params["vasopressor_cv_boost"] = vasopressor_cv_boost
    med_params["sedation_per_imv"] = True
    med_params["sedation_imv_prob"] = 0.85  # reference P(sedation|IMV)
    med["params"] = med_params
    tables["medication_admin_continuous"] = med
    labs = dict(tables.get("labs", {}))
    lab_params = dict(labs.get("params", {}))
    lab_params["panel_interval_hours"] = lab_panel_interval_hours
    lab_params["ward_panel_interval_hours"] = lab_ward_panel_interval_hours
    labs["params"] = lab_params
    tables["labs"] = labs
    # Renal failure drives creatinine/BUN up in many stays (incl. every terminal
    # decline); only a fraction are dialyzed, so CRRT is gated to its real rate.
    tables["crrt_therapy"] = {"params": {"crrt_prob": crrt_prob}}
    tables["position"] = {
        "params": {"prone_prob_severe": prone_prob_severe, "prone_prob_otherwise": 0.001}
    }

    return ParamPack(manifest=dict(pack.manifest), tables=tables)


#: Full-population **coupled admission-route** mix — the single per-stay pathway that
#: drives BOTH the hospitalization ``admission_type_category`` and the ADT arrival
#: location (KTD-6), so the two agree per stay. Drawn once per stay on the latent
#: spine (see ``spine.sample_spine``); the route values ARE mCIDE
#: ``admission_type_category`` members and map to arrival locations via
#: ``adt._ROUTE_TO_ARRIVAL`` (``ed``->ed, ``elective``->procedural, ``direct``->ward,
#: ``osh``->icu, ``facility``->ward, ``other``->ed). ED-dominant emergency flow with a
#: small elective (OR) and direct-admit share and a thin outside-hospital (``osh``)
#: referral pathway that arrives straight at the higher-level ICU. Normalized on draw.
_FULL_HOSPITAL_ADMISSION_ROUTE_MARGINAL: dict[str, float] = {
    "ed": 0.76,
    "elective": 0.10,
    "direct": 0.10,
    "osh": 0.03,
    "facility": 0.005,
    "other": 0.005,
}


def recalibrate_to_full_hospital(
    pack: ParamPack,
    *,
    esc_keep: float = 0.04,
    stepdown_keep: float = 0.05,
    icu_target: float = 0.08,
    stepdown_target: float = 0.05,
    peak_target: dict[str, float] | None = None,
    disch_damp: float = 1.0,
    sojourn_multipliers: dict[str, float] | None = None,
    sojourn_shape_scale: float = 0.68,
    mortality_scale: float = 1.0,
    mortality_target: float | None = 0.021,
    flag_target_prevalence: dict[str, float] | None = None,
    admission_route_marginal: dict[str, float] | None = None,
    prone_prob_severe: float = 0.026,
    niv_nippv_prob: float = 0.03,
    niv_hfnc_prob: float = 0.035,
    vasopressor_cv_boost: float = 3.0,
    lab_panel_interval_hours: float = 12.0,
    lab_ward_panel_interval_hours: float = 24.0,
    terminal_deterioration_hours: float = 24.0,
    crrt_prob: float = 0.29,
    repair_vitals: bool = True,
) -> ParamPack:
    """Return a deep-copied pack recalibrated to the **full hospital population**.

    This is the ward/ED/stepdown/ICU sibling of
    :func:`recalibrate_to_network_median`. Where that function conditions on an ICU
    stay (every trajectory peaks at the ICU floor or above), this one shapes a
    ward-dominant cohort in which most stays never leave ward/ED acuity, a minority
    pass through stepdown, and ~15-17% reach an ICU tier — the mix measured on the
    real full hospital population (ICU fraction ~0.156, stepdown ~0.08, hospital-LOS
    median ~67h, in-hospital mortality ~0.021).

    The transform is the same principled machine, retargeted:

    * **ADT location enrichment is enabled** (``enrich_locations``) so the
      ed / ward / stepdown / icu location mix appears; after the admission interval a
      stay's ADT tier is driven by its spine support level (2 -> stepdown, >= 3 ->
      icu, else ward).
    * **A single coupled admission route drives both the admission type and the
      arrival location** (``admission_route_marginal``, default
      :data:`_FULL_HOSPITAL_ADMISSION_ROUTE_MARGINAL`). The route is drawn once per
      stay on the latent spine (KTD-6: the spine is the only cross-table channel), and
      **both** the ``hospitalization`` generator (as ``admission_type_category``) and
      the ``adt`` generator (as the mapped arrival location) read it, so a stay's
      admission type and its front door always agree — unlike the earlier design, in
      which the two axes were calibrated from independent marginals. The route values
      are exact mCIDE ``admission_type_category`` members; they map to arrival
      locations as ``ed``->ed, ``elective``->procedural, ``direct``/``facility``->ward,
      ``other``->ed, and ``osh``->**icu** (an outside-hospital transfer arrives straight
      at the higher-level ICU — the modelled OSH->ICU referral pathway). Because this
      route supersedes them, the full-hospital adt block no longer sets
      ``arrival_location_marginal`` / ``direct_icu_frac`` and the hospitalization block
      no longer overrides ``admission_type_category_marginal`` (those code paths remain
      for other callers). Probe at n=6000, seed 5: admission-type ed 0.76 / direct 0.10
      / elective 0.10 / osh 0.03; osh->icu coupling ~100%; total ICU ~0.16, of which
      transfer ~0.12 and direct-arrive-icu ~0.04.
    * **The peak-acuity distribution is shaped ward-dominant.** Escalation to the
      invasive-ventilation tier (levels 3-5) *and* to the stepdown tier (level 2) is
      tempered so peak tracks start, and the start law is set to a full-population
      peak target (see :func:`_full_hospital_peak_target`) placing ``icu_target``
      start mass at the ICU tiers and ``stepdown_target`` at stepdown, the rest at
      ward/ED. These are **start-law** fractions; residual (tempered) escalation and
      the terminal-deterioration climb of decedents lift the realized peak fractions
      somewhat. ``icu_target`` (default 0.08, a **start-law** fraction that realizes as
      ~0.13 reaches-ICU after escalation/terminal climb) is deliberately below the
      ~0.156 real ICU fraction because the coupled ``osh`` route now arrives ~3% of
      stays straight at the ICU **independent of spine escalation** — the spine-driven
      ICU transfers (~0.12) plus the OSH direct-ICU arrivals (~0.03, not
      double-counted) sum to the measured total ICU fraction (~0.15-0.16).
    * **Sojourns are scaled to the short full-population LOS** — most stays are brief
      ward stays, so the multipliers are far smaller than the ICU mode's; the
      log-normal tails are tightened by ``sojourn_shape_scale`` as there.
    * **Mortality is solved to the low full-population rate** (``mortality_target``
      0.021, peak-coupled) rather than the ICU cohort's ~9.5%.
    * The corrected generator paths that still apply carry through: vitals-dispersion
      repair, terminal deterioration, per-stay non-invasive support gating (kept
      low-prevalence), decoupled organ flags, gated CRRT. Lab presence/quantiles ride
      through the base pack unchanged.

    Every argument is a documented keyword lever over that transform; the input pack
    is deep-copied and never mutated (R22). The default ``mortality_scale`` is a
    no-op because ``mortality_target`` is set by default (the target path takes
    precedence); pass ``mortality_target=None`` to fall back to the fixed scale.
    """
    tables = copy.deepcopy(dict(pack.tables))
    spine = tables["spine"]["params"]

    _temper_transitions(
        spine["support_level_transition_matrix"], esc_keep=esc_keep, disch_damp=disch_damp
    )
    # Also temper escalation into the stepdown tier (level 2) so the ward-dominant
    # cohort keeps its stepdown fraction near target; peak then tracks start for the
    # whole ward/stepdown/ICU split.
    _temper_stepdown_escalation(
        spine["support_level_transition_matrix"], stepdown_keep=stepdown_keep
    )
    # Shape the peak-acuity distribution ward-dominant through the start law (peak
    # tracks start once escalation is tempered).
    target = peak_target or _full_hospital_peak_target(
        _real_peak_profile(spine), icu_target, stepdown_target
    )
    _apply_peak_target(spine["support_level_start_dist"], target)
    _scale_sojourns(
        spine["support_level_sojourn"],
        sojourn_multipliers or {"0": 1.0, "1": 3.4, "2": 2.2, "3": 2.2, "4": 2.2},
        shape_scale=sojourn_shape_scale,
    )
    # Mortality: ``mortality_target`` (default 0.021) solves the peak-coupled scale
    # analytically against the peak-target distribution; otherwise the fixed
    # ``mortality_scale`` is applied.
    if mortality_target is not None:
        eff_mortality_scale = _solve_mortality_scale(
            target,
            spine["expired_rate_by_peak_level"],
            float(spine.get("outcome_marginal", {}).get("expired", 0.0)),
            mortality_target,
        )
    else:
        eff_mortality_scale = mortality_scale
    for cell in spine["expired_rate_by_peak_level"].values():
        cell["expired_rate"] = min(1.0, cell["expired_rate"] * eff_mortality_scale)

    spine["terminal_deterioration_hours"] = terminal_deterioration_hours

    if repair_vitals:
        repair_vitals_dispersion(tables)

    # Decoupled organ axes at full-population marginals (the ward-heavy cohort has a
    # lower organ-support prevalence than the ICU cohort, but the per-stay flag is
    # still gated to ICU time, so only reaches-ICU stays carry active support).
    spine["flag_target_prevalence"] = flag_target_prevalence or {
        "resp_flag": 0.5,
        "cv_flag": 0.27,
        "renal_flag": 0.05,
        "neuro_flag": 0.2,
    }

    # Coupled admission route (KTD-6): one per-stay draw on the spine that both the
    # hospitalization generator (as admission_type_category) and the adt generator (as
    # the mapped arrival location) read, so admission type and front door agree per
    # stay. This supersedes the previous independent admission_type/arrival marginals.
    spine["admission_route_marginal"] = dict(
        admission_route_marginal or _FULL_HOSPITAL_ADMISSION_ROUTE_MARGINAL
    )

    # Enable the ed/ward/stepdown/icu location mix — the defining difference from the
    # ICU mode, which sets this False (every stay an ICU stay). The admission (first)
    # location is now set by the coupled route (see spine.admission_route -> adt), so
    # no arrival_location_marginal / direct_icu_frac is needed: an osh route arrives
    # straight at icu, elective at procedural, direct/facility at ward, ed/other at ed.
    tables["adt"] = {"params": {"enrich_locations": True}}
    # Non-invasive support stays a low-prevalence per-stay property; the gated path
    # also keeps IMV to the intubation tier (level >= 3) so the ward/stepdown floor
    # never produces spurious ventilation.
    tables["respiratory_support"] = {
        "params": {
            "enrich_devices": True,
            "niv": {"nippv_prob": niv_nippv_prob, "hfnc_prob": niv_hfnc_prob},
        }
    }
    med = dict(tables.get("medication_admin_continuous", {}))
    med_params = dict(med.get("params", {}))
    med_params["vasopressor_per_stay"] = True
    med_params["vasopressor_cv_boost"] = vasopressor_cv_boost
    med_params["sedation_per_imv"] = True
    med_params["sedation_imv_prob"] = 0.85
    med["params"] = med_params
    tables["medication_admin_continuous"] = med
    labs = dict(tables.get("labs", {}))
    lab_params = dict(labs.get("params", {}))
    lab_params["panel_interval_hours"] = lab_panel_interval_hours
    lab_params["ward_panel_interval_hours"] = lab_ward_panel_interval_hours
    labs["params"] = lab_params
    tables["labs"] = labs
    tables["crrt_therapy"] = {"params": {"crrt_prob": crrt_prob}}
    tables["position"] = {
        "params": {"prone_prob_severe": prone_prob_severe, "prone_prob_otherwise": 0.001}
    }

    return ParamPack(manifest=dict(pack.manifest), tables=tables)


#: Reference ICU-cohort rates (ADT ``location_category == "icu"``) from the fitted
#: source CLIF extract.
_FITTED_ICU_IMV = 0.412
_FITTED_ICU_MORTALITY = 0.115
_FITTED_ICU_NIPPV = 0.064
_FITTED_ICU_HFNC = 0.069


def recalibrate_fitted_icu(pack: ParamPack) -> ParamPack:
    """Reshape a fitted all-28 pack with the **validated** ICU recalibrate path.

    Applies the same spine tempering / sojourn scaling / terminal deterioration /
    gated-NIV / CRRT machinery as :func:`recalibrate_to_network_median`, targeted
    at **reference ICU-cohort** rates (IMV ≈ 41%, mortality ≈ 11.5%, NIPPV/HFNC ≈
    6–7%). Then restores fitted ADT front-door knobs so ED→ICU arrivals use the
    validated ``arrival_location_marginal`` + ``direct_icu_frac`` path (and drops
    ``admission_route_marginal``, which would otherwise override arrivals — the
    source extract has no ``osh`` route, so the coupled route alone never emits
    direct-ICU).
    """
    orig_adt = copy.deepcopy(dict(pack.tables.get("adt", {}).get("params", {}) or {}))
    orig_rs = copy.deepcopy(
        dict(pack.tables.get("respiratory_support", {}).get("params", {}) or {})
    )
    orig_crrt = dict(pack.tables.get("crrt_therapy", {}).get("params", {}) or {})
    orig_pos = dict(pack.tables.get("position", {}).get("params", {}) or {})

    # Peak start near published IMV: terminal invents late IMV only for a fraction
    # of never-ventilated deaths, so little tempering is needed.
    out = recalibrate_to_network_median(
        pack,
        peak_imv_target=_FITTED_ICU_IMV - 0.05,
        mortality_target=_FITTED_ICU_MORTALITY,
        # ICU-conditional NIV rates (not all-hospital stay prevalence).
        niv_nippv_prob=_FITTED_ICU_NIPPV,
        niv_hfnc_prob=_FITTED_ICU_HFNC,
        prone_prob_severe=float(orig_pos.get("prone_prob_severe", 0.026)),
        # Reference vaso stay ~0.29 survivors / ~0.58 deaths — keep base cv low; terminal adds.
        flag_target_prevalence={
            "resp_flag": 0.5,
            "cv_flag": 0.10,
            "renal_flag": 0.055,
            "neuro_flag": 0.2,
        },
        # Lift P(CRRT|creat≥2) toward reference (~0.16); stay CRRT may land ~6–8%.
        crrt_prob=0.95,
    )
    tables = out.tables

    adt_params: dict[str, Any] = {"enrich_locations": True}
    for key in (
        "arrival_location_marginal",
        "direct_icu_frac",
        "hospital_type_marginal",
        "location_type_marginal",
        "location_category_marginal",
    ):
        if key in orig_adt:
            adt_params[key] = orig_adt[key]
    tables["adt"] = {
        "n_records": pack.tables.get("adt", {}).get("n_records", 0),
        "fitted": True,
        "params": adt_params,
    }
    spine_params = dict(tables["spine"]["params"])
    spine_params.pop("admission_route_marginal", None)
    # Reference-matched terminal mix + conditionals (±2 pp targets from audit).
    spine_params["terminal_archetype_mix"] = {
        "abrupt": 0.25,
        "prolonged": 0.50,
        "comfort": 0.25,
    }
    spine_params["terminal_imv_prob"] = 0.22
    # Steeper decedent physiology (clearer sicker→sicker story vs milder reference means).
    spine_params["terminal_vaso_prob"] = 0.35
    spine_params["terminal_renal_prob"] = 0.40
    spine_params["ladder_cv_prob"] = 0.72
    # Type-1 (NC→HFNC→IMV) vs type-2 OHS/HF/COPD (NIPPV→IMV) pathways.
    spine_params["resp_phenotype_marginal"] = {
        "type1": 0.40,
        "type2_ohs": 0.08,
        "type2_hf": 0.12,
        "type2_copd": 0.15,
        "unspecified": 0.25,
    }
    tables["spine"]["params"] = spine_params

    rs_params = dict(tables.get("respiratory_support", {}).get("params", {}))
    for key, val in orig_rs.items():
        if key in {"niv", "enrich_devices"}:
            continue
        rs_params.setdefault(key, val)
    rs_params["enrich_devices"] = True
    tables["respiratory_support"] = {
        "n_records": pack.tables.get("respiratory_support", {}).get("n_records", 0),
        "fitted": True,
        "params": rs_params,
    }

    crrt_params = dict(tables.get("crrt_therapy", {}).get("params", {}))
    for key, val in orig_crrt.items():
        if key == "crrt_prob":
            continue
        crrt_params.setdefault(key, val)
    tables["crrt_therapy"] = {
        "n_records": pack.tables.get("crrt_therapy", {}).get("n_records", 0),
        "fitted": True,
        "params": crrt_params,
    }

    return ParamPack(manifest=dict(out.manifest), tables=tables)
