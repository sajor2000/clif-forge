"""Latent state spine sampler — Tier 0 of the generate stage (U6, KTD-6).

Each synthetic hospitalization first gets an internal trajectory: an acuity
(organ-support) level over time, four organ-failure flags, and a terminal
outcome. This spine is the **sole cross-table channel** — every downstream table
generator reads ``(spine, param_pack, rng)`` and never another table's output,
which is what pairs vasopressors with hypotension, sedation with IMV, and prone
with severe hypoxemia while keeping the generators decoupled in code.

The spine is *not* a CLIF table. It is optionally retained as ``_truth.parquet``
for benchmarking (free ground-truth acuity/flag/outcome labels).

Everything here is sampled offline from the parameter pack the fit stage (U5)
emits — no real data is present. The pack's ``spine`` block supplies every input:

* ``support_level_start_dist`` — the initial-state law.
* ``support_level_transition_matrix`` — the embedded jump chain with an absorbing
  ``discharge`` exit, so trajectories terminate naturally.
* ``support_level_sojourn`` — per-level dwell-time family.
* ``expired_rate_by_peak_level`` / ``outcome_marginal`` — mortality coupled to
  peak acuity (falling back to the cohort marginal when a peak level was gated).
* ``flag_prevalence_by_level`` — per-level organ-failure flag prevalences.

A fixed ``numpy.random.Generator`` makes a sampled spine reproducible
byte-for-byte (R22): the trajectory, then per-run flags, then the outcome are
all drawn from that one generator in a fixed order.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.estimators import DISCHARGE_STATE
from clifforge.fit.param_pack import ParamPack
from clifforge.generate import semimarkov
from clifforge.generate._common import ICU_MIN_SUPPORT_LEVEL
from clifforge.generate.sampling import categorical
from clifforge.generate.semimarkov import SojournSampler

__all__ = [
    "FLAG_NAMES",
    "RESP_PHENOTYPES",
    "SpineFrame",
    "sample_spine",
    "truth_frame",
]

#: The four organ-failure flags carried per interval, in a stable order.
FLAG_NAMES: tuple[str, ...] = ("resp_flag", "cv_flag", "renal_flag", "neuro_flag")


@dataclass(frozen=True)
class SpineFrame:
    """One hospitalization's latent trajectory (per-interval arrays + outcome).

    ``support_level`` and the four flag lists are all length ``n_intervals`` and
    aligned interval-by-interval. ``outcome`` is ``"expired"`` or ``"alive"``.

    ``admission_route`` is a per-stay scalar (like ``outcome``): a coupled
    admission pathway drawn once per stay when the pack carries an
    ``admission_route_marginal``, otherwise ``""``. It is the sole cross-table
    channel (KTD-6) that keeps the hospitalization ``admission_type_category`` and
    the ADT arrival location in agreement per stay: both generators read it. Its
    values are mCIDE ``admission_type_category`` members (``ed``, ``elective``,
    ``direct``, ``osh``, ``facility``, ``other``).

    ``resp_phenotype`` is a per-stay respiratory-failure pathway (KTD-6):
    ``type1`` (hypoxemic: NC→HFNC→IMV), ``type2_ohs`` / ``type2_hf`` /
    ``type2_copd`` (hypercapnic / OHS / HF: NIPPV→IMV), or ``""`` / ``unspecified``
    (gated NIV mix). Downstream RS and diagnosis tables read it; they never
    invent the pathway themselves.
    """

    hospitalization_id: str
    support_level: list[int]
    resp_flag: list[bool]
    cv_flag: list[bool]
    renal_flag: list[bool]
    neuro_flag: list[bool]
    outcome: str
    admission_route: str = ""
    resp_phenotype: str = ""

    @property
    def n_intervals(self) -> int:
        return len(self.support_level)

    @property
    def peak_level(self) -> int:
        return max(self.support_level) if self.support_level else 0

    def to_polars(self) -> pl.DataFrame:
        """Long-format frame: one row per interval, plus scalar labels broadcast."""
        n = self.n_intervals
        return pl.DataFrame(
            {
                "hospitalization_id": [self.hospitalization_id] * n,
                "interval_idx": list(range(n)),
                "support_level": self.support_level,
                "resp_flag": self.resp_flag,
                "cv_flag": self.cv_flag,
                "renal_flag": self.renal_flag,
                "neuro_flag": self.neuro_flag,
                "outcome": [self.outcome] * n,
                "admission_route": [self.admission_route] * n,
                "resp_phenotype": [self.resp_phenotype] * n,
            }
        )


def _spine_params(pack: ParamPack) -> dict[str, Any]:
    block = pack.tables.get("spine")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'spine' block to sample from")
    params: dict[str, Any] = block["params"]
    return params


def _int_key_dist(dist: dict[str, float]) -> dict[Hashable, float]:
    return {int(level): float(prob) for level, prob in dist.items()}


def _transitions(matrix: dict[str, dict[str, float]]) -> dict[Hashable, dict[Hashable, float]]:
    """Parse the string-keyed pack matrix into int levels + the discharge exit."""
    out: dict[Hashable, dict[Hashable, float]] = {}
    for frm, row in matrix.items():
        parsed: dict[Hashable, float] = {}
        for to, prob in row.items():
            key: Hashable = DISCHARGE_STATE if to == DISCHARGE_STATE else int(to)
            parsed[key] = float(prob)
        out[int(frm)] = parsed
    return out


def _referenced_levels(
    start_dist: dict[Hashable, float], transitions: dict[Hashable, dict[Hashable, float]]
) -> set[int]:
    """Every integer support level the trajectory could dwell in."""
    levels: set[int] = {k for k in start_dist if isinstance(k, int)}
    for frm, row in transitions.items():
        if isinstance(frm, int):
            levels.add(frm)
        levels.update(to for to in row if isinstance(to, int))
    return levels


def _sojourn_samplers(
    sojourn_block: dict[str, dict[str, Any]], levels: set[int]
) -> dict[Hashable, SojournSampler]:
    """A dwell-time sampler for every referenced level.

    A level whose sojourn was gated out (n < 20) has no fitted family; it falls
    back to an exponential whose mean is the average of the fitted levels'
    means, so a reachable-but-unfit level can still be dwelt in rather than
    raising. If no level was fit at all, the fallback is a unit-mean exponential.
    """
    samplers: dict[Hashable, SojournSampler] = {}
    fitted_means: list[float] = []
    for level_str, fit in sojourn_block.items():
        samplers[int(level_str)] = semimarkov.make_sojourn_sampler(fit["family"], fit["params"])
        mean = fit.get("mean_hours")
        if isinstance(mean, int | float) and mean > 0:
            fitted_means.append(float(mean))

    fallback_mean = float(np.mean(fitted_means)) if fitted_means else 1.0
    fallback = semimarkov.make_sojourn_sampler("empirical_mean", [fallback_mean])
    for level in levels:
        samplers.setdefault(level, fallback)
    return samplers


def _visits_to_intervals(
    visits: list[semimarkov.Visit], grid_step_hours: float, horizon_intervals: int
) -> list[tuple[int, int]]:
    """Expand (level, hours) visits into per-run (level, interval_count) pairs.

    A sub-grid dwell still occupies at least one interval; the whole timeline is
    capped at ``horizon_intervals`` so rounding can never overrun the fit grid.
    The absorbing ``discharge`` visit (a non-int state) contributes nothing.
    """
    runs: list[tuple[int, int]] = []
    total = 0
    for visit in visits:
        if not isinstance(visit.state, int):  # the discharge terminal marker
            continue
        n_int = max(1, round(visit.duration / grid_step_hours))
        if total + n_int > horizon_intervals:
            n_int = horizon_intervals - total
        if n_int <= 0:
            break
        runs.append((visit.state, n_int))
        total += n_int
        if total >= horizon_intervals:
            break
    return runs


def sample_spine(
    pack: ParamPack, rng: np.random.Generator, *, hospitalization_id: str = "H0"
) -> SpineFrame:
    """Sample one hospitalization's latent spine from the pack (KTD-6, R22).

    Draws, in order from ``rng``: the support-level trajectory (semi-Markov),
    then each run's organ-failure flags (Bernoulli at the run's per-level
    prevalence, held constant across the run for within-run coherence), then the
    terminal outcome (Bernoulli at the mortality rate for the trajectory's peak
    acuity), and — **last, and only when the pack carries an
    ``admission_route_marginal``** — the coupled admission route. Placing the
    route draw last keeps every earlier draw (trajectory/flags/outcome) at its
    original position, so a pack without the marginal never draws it and its
    output is byte-for-byte unchanged (R22). Same seed in -> identical
    :class:`SpineFrame` out.
    """
    params = _spine_params(pack)
    state_model = params["state_model"]
    grid_step_hours = float(state_model["grid_step_hours"])
    horizon_intervals = int(state_model["horizon_intervals"])
    horizon_hours = horizon_intervals * grid_step_hours

    start_dist = _int_key_dist(params["support_level_start_dist"])
    transitions = _transitions(params["support_level_transition_matrix"])
    levels = _referenced_levels(start_dist, transitions)
    sojourns = _sojourn_samplers(params["support_level_sojourn"], levels)
    flag_prevalence: dict[str, dict[str, float]] = params["flag_prevalence_by_level"]

    visits = semimarkov.sample(
        transitions,
        sojourns,
        start_dist,
        {DISCHARGE_STATE},
        rng,
        horizon_hours,
    )
    runs = _visits_to_intervals(visits, grid_step_hours, horizon_intervals)

    support_level = [level for level, n_int in runs for _ in range(n_int)]
    flags: dict[str, list[bool]] = {name: [] for name in FLAG_NAMES}

    target = params.get("flag_target_prevalence")
    if target:
        # Decoupled organ axes: each failure flag is drawn once per stay at a
        # fitted *marginal* target (independent of the acuity-level distribution)
        # and active across the stay's ICU time. This lets cardiovascular /
        # renal failure — and thus vasopressor / CRRT prevalence — track their
        # real rates even when the respiratory acuity distribution is shifted, so
        # a single latent level no longer forces every organ support to move
        # together. A stay with no ICU interval carries no active flag.
        icu_mask = [level >= ICU_MIN_SUPPORT_LEVEL for level in support_level]
        for name in FLAG_NAMES:
            has = bool(rng.random() < float(target.get(name, 0.0)))
            flags[name] = [has and in_icu for in_icu in icu_mask]
    else:
        # Fitted per-level behaviour: one draw per flag per run at the level's
        # prevalence (the flag marginal is then hostage to the level mix).
        for level, n_int in runs:
            prevalence = flag_prevalence.get(str(level), {})
            run_flags = {
                name: bool(rng.random() < float(prevalence.get(name, 0.0))) for name in FLAG_NAMES
            }
            for name in FLAG_NAMES:
                flags[name].extend([run_flags[name]] * n_int)

    peak = max(support_level) if support_level else 0
    outcome = _sample_outcome(params, peak, rng)

    term_hours = params.get("terminal_deterioration_hours")
    if outcome == "expired" and term_hours:
        _apply_terminal_deterioration(
            support_level,
            flags,
            float(term_hours),
            grid_step_hours,
            rng,
            mix=params.get("terminal_archetype_mix"),
            imv_prob=float(params.get("terminal_imv_prob", _DEFAULT_TERMINAL_IMV_PROB)),
            vaso_prob=float(params.get("terminal_vaso_prob", _DEFAULT_TERMINAL_VASO_PROB)),
            renal_prob=float(params.get("terminal_renal_prob", _DEFAULT_TERMINAL_RENAL_PROB)),
        )

    # Soft ladder→organ re-couple: the support ladder's clinical meaning is
    # L4 = +vaso, L5 = +CRRT. Recalibrate's ``flag_target_prevalence`` deliberately
    # decouples marginal rates, but leaving L4/L5 stays without cv/renal flags makes
    # vitals/labs/meds/resp disagree longitudinally ("sicker" acuity without shock
    # physiology). Soft Bernoulli on L4→cv keeps MIMIC vaso|IMV (~0.60).
    _recouple_ladder_organs(
        support_level,
        flags,
        rng,
        ladder_cv_prob=float(params.get("ladder_cv_prob", _DEFAULT_LADDER_CV_PROB)),
    )

    # Coupled admission route + respiratory phenotype (drawn LAST so they never
    # shift the earlier draws). Absent marginals leave "" (backward compatible).
    route_marginal = params.get("admission_route_marginal")
    admission_route = categorical(route_marginal, rng) if route_marginal else ""
    pheno_marginal = params.get("resp_phenotype_marginal") or _DEFAULT_RESP_PHENOTYPE_MIX
    resp_phenotype = categorical(pheno_marginal, rng)

    return SpineFrame(
        hospitalization_id=hospitalization_id,
        support_level=support_level,
        resp_flag=flags["resp_flag"],
        cv_flag=flags["cv_flag"],
        renal_flag=flags["renal_flag"],
        neuro_flag=flags["neuro_flag"],
        outcome=outcome,
        admission_route=admission_route,
        resp_phenotype=resp_phenotype,
    )


#: Respiratory-failure pathways (per-stay). Typed pathways drive NC→HFNC→IMV
#: (type1) or NIPPV→IMV (type2_*); unspecified keeps the gated NIV mix.
RESP_PHENOTYPES: tuple[str, ...] = (
    "type1",
    "type2_ohs",
    "type2_hf",
    "type2_copd",
    "unspecified",
)
_DEFAULT_RESP_PHENOTYPE_MIX: dict[str, float] = {
    "type1": 0.40,
    "type2_ohs": 0.08,
    "type2_hf": 0.12,
    "type2_copd": 0.15,
    "unspecified": 0.25,
}


#: Terminal-decline archetypes and their default mix. Real ICU deaths are not a
#: single stereotyped course, so an expiring stay draws one of these shapes.
TERMINAL_ARCHETYPES: tuple[str, ...] = ("abrupt", "prolonged", "comfort")
_DEFAULT_TERMINAL_ARCHETYPE_MIX: dict[str, float] = {
    "abrupt": 0.3,
    "prolonged": 0.5,
    "comfort": 0.2,
}

#: Ladder rungs whose clinical meaning implies organ support (soft re-couple).
_VASO_MIN_SUPPORT_LEVEL = 4
_CRRT_MIN_SUPPORT_LEVEL = 5

#: MIMIC ICU conditionals (±2 pp targets for ``recalibrate_mimic_icu``).
#: ``terminal_imv_prob`` = P(invent late IMV | death ∧ never-IMV), not stay-level.
_DEFAULT_TERMINAL_IMV_PROB = 0.20
_DEFAULT_TERMINAL_VASO_PROB = 0.58
_DEFAULT_TERMINAL_RENAL_PROB = 0.40
#: Soft L4→cv re-couple rate so vaso|IMV lands near MIMIC (~0.60), not 1.0.
_DEFAULT_LADDER_CV_PROB = 0.55


def _recouple_ladder_organs(
    support_level: list[int],
    flags: dict[str, list[bool]],
    rng: np.random.Generator,
    *,
    ladder_cv_prob: float = _DEFAULT_LADDER_CV_PROB,
) -> None:
    """Soft-couple cv/renal flags to ladder rungs that clinically imply them.

    L4 = IMV+vaso, L5 = +CRRT. Coupling is a **stay-level** Bernoulli when any
    L4+ interval exists (per-interval draws made long L4 runs saturate to cv=1
    and blew past MIMIC vaso|IMV / vaso|death). L5→renal stays hard (rare).
    """
    has_l4 = any(level >= _VASO_MIN_SUPPORT_LEVEL for level in support_level)
    couple_cv = has_l4 and rng.random() < ladder_cv_prob
    for i, level in enumerate(support_level):
        if couple_cv and level >= _VASO_MIN_SUPPORT_LEVEL:
            flags["cv_flag"][i] = True
        if level >= _CRRT_MIN_SUPPORT_LEVEL:
            flags["renal_flag"][i] = True


def _pick_terminal_archetype(mix: dict[str, float], rng: np.random.Generator) -> str:
    """Draw one terminal archetype from the mix (renormalized over known archetypes)."""
    total = sum(mix.get(a, 0.0) for a in TERMINAL_ARCHETYPES) or 1.0
    r = float(rng.random())
    cum = 0.0
    for archetype in TERMINAL_ARCHETYPES:
        cum += mix.get(archetype, 0.0) / total
        if r < cum:
            return archetype
    return TERMINAL_ARCHETYPES[-1]


def _apply_terminal_deterioration(
    support_level: list[int],
    flags: dict[str, list[bool]],
    hours: float,
    grid_step_hours: float,
    rng: np.random.Generator,
    mix: dict[str, float] | None = None,
    *,
    imv_prob: float = _DEFAULT_TERMINAL_IMV_PROB,
    vaso_prob: float = _DEFAULT_TERMINAL_VASO_PROB,
    renal_prob: float = _DEFAULT_TERMINAL_RENAL_PROB,
) -> None:
    """Shape the final window of an expiring trajectory into a terminal decline.

    A peak-coupled outcome alone leaves the terminal intervals benign (a patient
    can peak mid-stay then de-escalate), so decedents look like survivors. This
    imprints one of several **archetypes** on the last ``hours`` of the stay, so
    dying courses vary across encounters while the deterioration still reaches
    every table through the spine's existing couplings (support-level state means
    drive vitals; the renal flag drives creatinine/BUN), never by reading another
    table (KTD-6):

    * ``abrupt`` — a short, steep collapse when vented; shock/renal by Bernoulli.
    * ``prolonged`` — laddered climb among IMV deaths; organs fail in sequence.
    * ``comfort`` — device withdrawal; optional shock physiology without inventing
      IMV on every death (MIMIC: ~33% of ICU deaths never receive IMV).

    Stay-level Bernoulli draws (``imv_prob`` / ``vaso_prob`` / ``renal_prob``)
    target MIMIC conditionals P(IMV|death)≈0.67, P(vaso|death)≈0.58.
    """
    n = len(support_level)
    if n == 0:
        return
    archetype = _pick_terminal_archetype(mix or _DEFAULT_TERMINAL_ARCHETYPE_MIX, rng)
    full_window = max(1, round(hours / grid_step_hours))
    n_term = max(1, full_window // 3) if archetype == "abrupt" else full_window
    start = max(0, n - n_term)
    span = n - start
    pre_peak = max(support_level[:start], default=max(support_level, default=0))
    already_imv = pre_peak >= 3
    # ``imv_prob`` is the *invent* rate among never-ventilated deaths. Already-IMV
    # decedents stay counted; together they target MIMIC P(IMV|expired)≈0.67.
    do_imv = already_imv or (rng.random() < imv_prob)
    do_vaso = rng.random() < vaso_prob
    do_renal = rng.random() < renal_prob

    for i in range(start, n):
        frac = (i - start + 1) / span  # 0..1 across the terminal window
        if archetype == "comfort":
            if do_imv:
                support_level[i] = max(3, 5 - int(round(2.0 * frac)))
                flags["resp_flag"][i] = True
            else:
                # Withdraw to ICU floor — no invented IMV on comfort deaths.
                support_level[i] = min(support_level[i], max(2, 3 - int(round(frac))))
            if do_vaso:
                flags["cv_flag"][i] = True
            if do_renal:
                flags["renal_flag"][i] = True
        elif archetype == "abrupt":
            if do_imv:
                support_level[i] = max(support_level[i], 4 + int(round(frac)))
                flags["resp_flag"][i] = True
            else:
                support_level[i] = max(support_level[i], 2)
            if do_vaso:
                flags["cv_flag"][i] = True
            if do_renal:
                flags["renal_flag"][i] = True
        else:  # prolonged
            if do_imv:
                support_level[i] = max(support_level[i], 3 + int(round(frac)))
                flags["resp_flag"][i] = True
            else:
                support_level[i] = max(support_level[i], 2)
            if do_vaso and frac >= 0.34:
                flags["cv_flag"][i] = True
            if do_renal and frac >= 0.67:
                flags["renal_flag"][i] = True


def _sample_outcome(params: dict[str, Any], peak_level: int, rng: np.random.Generator) -> str:
    """Expired/alive, coupled to peak acuity with a cohort-marginal fallback."""
    by_peak: dict[str, dict[str, float]] = params.get("expired_rate_by_peak_level", {})
    peak_cell = by_peak.get(str(peak_level))
    if peak_cell is not None:
        expired_rate = float(peak_cell["expired_rate"])
    else:
        expired_rate = float(params.get("outcome_marginal", {}).get("expired", 0.0))
    return "expired" if rng.random() < expired_rate else "alive"


def truth_frame(spines: list[SpineFrame]) -> pl.DataFrame:
    """Stack sampled spines into one long ``_truth`` frame for benchmarking."""
    if not spines:
        return pl.DataFrame(
            schema={
                "hospitalization_id": pl.String,
                "interval_idx": pl.Int64,
                "support_level": pl.Int64,
                "resp_flag": pl.Boolean,
                "cv_flag": pl.Boolean,
                "renal_flag": pl.Boolean,
                "neuro_flag": pl.Boolean,
                "outcome": pl.String,
                "admission_route": pl.String,
                "resp_phenotype": pl.String,
            }
        )
    return pl.concat([s.to_polars() for s in spines], how="vertical")
