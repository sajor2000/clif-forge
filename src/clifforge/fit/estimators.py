"""Aggregate estimators for the empirical-fidelity fit stage (U5).

Each estimator takes eager polars frames (``run_fit`` owns reading the real
data and collecting — KTD-1) and returns a ``(params, suppression)`` pair:

* ``params`` — a JSON-serializable dict of **aggregate** statistics only
  (marginals, transition probabilities, parametric-family parameters, a
  correlation matrix, hazards). Never a per-record value (R1).
* ``suppression`` — the ``cell_gate.SuppressionRecord`` audit list, so the
  pack manifest can report exactly which cells fell below the n>=20 floor (R2).

Every cell (a category, a transition pair, a per-state physiology fit, a lab,
a med) is routed through :func:`cell_gate.suppress` before it can enter the
pack — that is the single choke point enforcing the count floor.

The estimators are deliberately model-light and inspectable:

* transitions — the **embedded** (jump) chain over the organ-support ladder:
  self-transitions are removed, so the diagonal is zero and each row of the
  emitted matrix sums to 1 over the *other* states plus an absorbing
  ``discharge`` exit; the initial-state law is emitted alongside as
  ``support_level_start_dist`` (both consumed by the U6 spine sampler).
* sojourns — per-state dwell time, best parametric family chosen by AIC among
  exponential / gamma / lognormal / Weibull.
* AR1 — per (vital, support-level) first-order autoregression on a fixed grid.
* lab copula — Spearman correlation over co-measured labs, projected to the
  nearest positive-definite correlation matrix, plus per-lab log-marginals and
  presence rates.
* infusion hazards — per-drug start/stop hazard per interval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from clifforge.fit.cell_gate import SuppressionRecord, suppress

#: 1-D/2-D float64 array alias for the numeric estimator internals.
_F64 = NDArray[np.float64]

#: Fixed probability grid for the empirical (inverse-CDF) lab marginals: 101 points
#: 0.00, 0.01, ..., 1.00. A single log-normal cannot capture real lab shapes
#: (creatinine's CKD tail, lactate's right skew, troponin's bimodality); the fitted
#: quantile grid on this grid, driven through the copula's probability-integral
#: transform at generation, matches each lab's marginal exactly while preserving the
#: copula's rank-correlation. Shared with the labs generator (single source of truth).
LAB_QUANTILE_PROBS: _F64 = np.linspace(0.0, 1.0, 101)

#: Minimum record count for a lab to receive an empirical quantile marginal: >= 20
#: records per grid interval (100 intervals), mirroring the n >= 20 cell floor. Below
#: this the fine grid would approach the raw sorted values (a leakage risk the pack's
#: value-level scan rejects), so sparser labs fall back to the log-normal marginal.
_QUANTILE_MIN_RECORDS: int = 20 * (LAB_QUANTILE_PROBS.size - 1)

__all__ = [
    "DISCHARGE_STATE",
    "EstimatorResult",
    "LAB_QUANTILE_PROBS",
    "fit_categorical_marginals",
    "fit_continuous_marginals",
    "fit_transitions",
    "fit_sojourns",
    "fit_outcome_rates",
    "fit_flag_prevalence",
    "fit_ar1_by_state",
    "fit_lab_copula",
    "fit_infusion_hazards",
    "fit_stay_prevalence",
    "fit_stay_prevalence_by_category",
    "fit_code_status_rates",
    "fit_top_k_category",
    "fit_cultures_per_icu_day",
    "fit_prone_rates",
    "fit_age_quantiles",
    "fit_adt_arrival",
    "fit_admission_route_marginal",
    "nearest_positive_definite_correlation",
]

#: (params-block, suppression-audit) — the shape every estimator returns.
EstimatorResult = tuple[dict[str, object], list[SuppressionRecord]]

#: candidate positive-support families for sojourn dwell times, fit with the
#: location pinned at 0 (durations are non-negative).
_SOJOURN_FAMILIES: dict[str, stats.rv_continuous] = {
    "exponential": stats.expon,
    "gamma": stats.gamma,
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
}

#: AR1 coefficient is clamped just inside the unit circle for stationarity.
_PHI_CLAMP = 0.999


# --------------------------------------------------------------------------- #
# Marginals
# --------------------------------------------------------------------------- #
def fit_categorical_marginals(
    df: pl.DataFrame, fields: Sequence[str], *, min_n: int = 20
) -> EstimatorResult:
    """Per-field category proportions, each category gated at ``min_n`` (R2).

    A category observed fewer than ``min_n`` times is dropped from its field's
    marginal (recorded in the audit); the surviving proportions are then
    renormalized to sum to 1 so the emitted marginal is a proper distribution.
    """
    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []
    for field in fields:
        if field not in df.columns:
            continue
        vc = df.select(pl.col(field).drop_nulls()).to_series().value_counts(sort=True)
        counts = {row[0]: int(row[1]) for row in vc.iter_rows()}
        total = sum(counts.values())
        if total == 0:
            continue
        raw_props = {cat: n / total for cat, n in counts.items()}
        survived, records = suppress(counts, raw_props, min_n=min_n)
        audit.extend(
            SuppressionRecord(cell=(field, r.cell), n=r.n, fallback_kind=r.fallback_kind)
            for r in records
        )
        surviving_total = sum(survived.values())
        if surviving_total > 0:
            params[f"{field}_marginal"] = {
                cat: prop / surviving_total for cat, prop in survived.items()
            }
    return params, audit


def fit_continuous_marginals(
    df: pl.DataFrame, fields: Sequence[str], *, n_bins: int = 10, min_n: int = 20
) -> EstimatorResult:
    """Coarse quantile bin-edges per field (a compact, non-leaking marginal).

    Emits at most ``n_bins`` bins, and only when the whole field clears the
    ``min_n`` floor. The edge list has ``<= n_bins + 1`` entries regardless of
    ``n_records``, so it can never approach a raw column (R1/R2).
    """
    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []
    for field in fields:
        if field not in df.columns:
            continue
        col = df.select(pl.col(field).drop_nulls().cast(pl.Float64)).to_series()
        n = col.len()
        counts = {field: n}
        quantiles = [i / n_bins for i in range(n_bins + 1)]
        raw_edges = [col.quantile(q, interpolation="linear") for q in quantiles]
        edges = sorted({round(float(e), 6) for e in raw_edges if e is not None})
        survived, records = suppress(counts, {field: edges}, min_n=min_n)
        audit.extend(records)
        if field in survived and len(survived[field]) >= 2:
            params[f"{field}_quantile_bin_edges"] = survived[field]
    return params, audit


# --------------------------------------------------------------------------- #
# Semi-Markov spine: transitions + sojourns
# --------------------------------------------------------------------------- #
def _runs(state_timeline: pl.DataFrame) -> pl.DataFrame:
    """Run-length-encode the per-hospitalization support-level sequence.

    Returns one row per run with ``hospitalization_id``, ``run_idx`` (0-based
    within the hospitalization), ``support_level``, and ``start_interval``.
    Consecutive equal states collapse into one run — the basis for both the
    embedded transition chain and the sojourn dwell times.
    """
    tl = state_timeline.sort("hospitalization_id", "interval_idx")
    prev = pl.col("support_level").shift(1).over("hospitalization_id")
    changed = (pl.col("support_level") != prev) | prev.is_null()
    tl = tl.with_columns(changed.cast(pl.Int64).alias("_chg")).with_columns(
        pl.col("_chg").cum_sum().over("hospitalization_id").alias("run_idx")
    )
    return (
        tl.group_by("hospitalization_id", "run_idx")
        .agg(
            pl.col("support_level").first().alias("support_level"),
            pl.col("interval_idx").min().alias("start_interval"),
        )
        .sort("hospitalization_id", "run_idx")
    )


#: absorbing target that ends a hospitalization; a run with no following run is
#: a discharge (alive or dead — the terminal *outcome* is modelled separately by
#: ``fit_outcome_rates``). Emitted as a competing-risk column in every transition
#: row so the U6 spine sampler terminates naturally instead of running to horizon.
DISCHARGE_STATE = "discharge"


def fit_transitions(state_timeline: pl.DataFrame, *, min_n: int = 20) -> EstimatorResult:
    """Embedded (jump-chain) transition matrix over the support ladder (R2).

    Three parameters are emitted, all gated at ``min_n`` and all aggregate-only:

    * ``support_level_states`` — the sorted set of observed support levels.
    * ``support_level_start_dist`` — the distribution of the **first** run's
      support level per hospitalization; the U6 spine draws its initial state
      from this (R15: the sampler must not invent an initial condition).
    * ``support_level_transition_matrix`` — a nested ``{from: {to: prob}}`` map.
      ``from -> to`` counts observed jumps (``from != to`` by run construction);
      each row additionally carries a :data:`DISCHARGE_STATE` competing-risk
      entry counting the runs at ``from`` that were **terminal** (the
      hospitalization ended rather than jumping onward). The matrix has a **zero
      diagonal**, and each row sums to 1 over its reachable next-states plus
      discharge — so the trajectory always has an exit and terminates without a
      horizon cap. ``discharge`` is absorbing; it has no sojourn and no outgoing
      row.
    """
    runs = _runs(state_timeline)

    # --- initial-state distribution (first run per hospitalization) -----------
    first_levels = (
        runs.sort("hospitalization_id", "run_idx")
        .group_by("hospitalization_id")
        .agg(pl.col("support_level").first().alias("start_level"))
    )
    start_counts = {
        int(r["start_level"]): int(r["n"])
        for r in first_levels.group_by("start_level")
        .len()
        .rename({"len": "n"})
        .iter_rows(named=True)
    }
    start_survived, start_audit = suppress(start_counts, start_counts, min_n=min_n)
    start_total = sum(start_survived.values())
    start_dist = (
        {str(level): n / start_total for level, n in start_survived.items()}
        if start_total > 0
        else {}
    )

    # --- jumps + discharge competing risk -------------------------------------
    nxt = pl.col("support_level").shift(-1).over("hospitalization_id")
    with_next = runs.with_columns(nxt.alias("to_level"))
    jump_counts = (
        with_next.drop_nulls("to_level")
        .group_by("support_level", "to_level")
        .len()
        .rename({"len": "n"})
    )
    discharge_counts = (
        with_next.filter(pl.col("to_level").is_null())
        .group_by("support_level")
        .len()
        .rename({"len": "n"})
    )

    # Cell key = (from_level:int, to:str) where ``to`` is a level string or
    # DISCHARGE_STATE; gate every ordered pair (including the discharge column).
    counts: dict[tuple[int, str], int] = {
        (int(r["support_level"]), str(int(r["to_level"]))): int(r["n"])
        for r in jump_counts.iter_rows(named=True)
    }
    for r in discharge_counts.iter_rows(named=True):
        counts[(int(r["support_level"]), DISCHARGE_STATE)] = int(r["n"])
    survived, audit = suppress(counts, counts, min_n=min_n)

    # Row-normalize surviving counts into a nested {from: {to: prob}} matrix.
    row_totals: dict[int, int] = {}
    for (frm, _to), n in survived.items():
        row_totals[frm] = row_totals.get(frm, 0) + n
    matrix: dict[str, dict[str, float]] = {}
    for (frm, to), n in survived.items():
        if row_totals[frm] > 0:
            matrix.setdefault(str(frm), {})[to] = n / row_totals[frm]

    states = sorted({frm for frm, _to in counts})
    params: dict[str, object] = {
        "support_level_states": states,
        "support_level_start_dist": start_dist,
        "support_level_transition_matrix": matrix,
    }
    return params, audit + start_audit


def fit_sojourns(
    state_timeline: pl.DataFrame, *, grid_step_hours: float = 1.0, min_n: int = 20
) -> EstimatorResult:
    """Per-state dwell-time family chosen by AIC (R2).

    Dwell time of a run = (start of the next run - start of this run) in hours.
    Terminal runs (no following run — right-censored at discharge) are dropped
    to keep the fit a plain complete-data MLE; the count of dropped censored
    runs is not emitted (aggregate-only). Each state is gated at ``min_n``
    non-censored sojourns.
    """
    runs = _runs(state_timeline)
    nxt_start = pl.col("start_interval").shift(-1).over("hospitalization_id")
    durations = (
        runs.with_columns(nxt_start.alias("_next_start"))
        .drop_nulls("_next_start")
        .with_columns(
            (
                (pl.col("_next_start") - pl.col("start_interval")).cast(pl.Float64)
                * grid_step_hours
            ).alias("duration_hours")
        )
        .filter(pl.col("duration_hours") > 0)
        .select("support_level", "duration_hours")
    )

    by_state: dict[int, list[float]] = {}
    for row in durations.iter_rows(named=True):
        by_state.setdefault(int(row["support_level"]), []).append(float(row["duration_hours"]))

    counts = {state: len(vals) for state, vals in by_state.items()}
    fits = {
        state: _best_sojourn_family(np.asarray(vals, dtype=float))
        for state, vals in by_state.items()
    }
    survived, audit = suppress(counts, fits, min_n=min_n)
    params: dict[str, object] = {
        "support_level_sojourn": {str(state): fit for state, fit in survived.items()}
    }
    return params, audit


def _best_sojourn_family(durations: _F64) -> dict[str, object]:
    """Fit each candidate family (loc=0) and return the min-AIC choice."""
    best: dict[str, object] | None = None
    best_aic = np.inf
    for name, dist in _SOJOURN_FAMILIES.items():
        try:
            fitted = dist.fit(durations, floc=0.0)
            loglik = float(np.sum(dist.logpdf(durations, *fitted)))
        except Exception:  # noqa: BLE001 — a family that fails to fit is just skipped
            continue
        if not np.isfinite(loglik):
            continue
        k = len(fitted)
        aic = 2 * k - 2 * loglik
        if aic < best_aic:
            best_aic = aic
            best = {
                "family": name,
                "params": [round(float(p), 6) for p in fitted],
                "aic": round(float(aic), 4),
                "mean_hours": round(float(np.mean(durations)), 4),
            }
    return best or {
        "family": "empirical_mean",
        "params": [round(float(np.mean(durations)), 6)],
        "aic": None,
        "mean_hours": round(float(np.mean(durations)), 4),
    }


# --------------------------------------------------------------------------- #
# Spine attributes: terminal outcome + organ-failure flags
# --------------------------------------------------------------------------- #
def fit_outcome_rates(
    state_timeline: pl.DataFrame, outcomes: pl.DataFrame, *, min_n: int = 20
) -> EstimatorResult:
    """Terminal-outcome marginal + expired rate by peak support level (R2).

    The spine sampler (U6) draws each hospitalization's terminal outcome
    (survive/expire) from pack params, coupled to acuity. This emits both the
    overall outcome marginal and ``P(expired | peak support level reached)`` so
    the coupling is empirical, not assumed. ``outcomes`` carries
    ``hospitalization_id`` and ``outcome`` (``"alive"``/``"expired"``). The
    overall marginal is gated on the hospitalization count; each peak-level cell
    is gated on the hospitalizations that peaked at that level.
    """
    peak = state_timeline.group_by("hospitalization_id").agg(
        pl.col("support_level").max().alias("peak_level")
    )
    joined = peak.join(outcomes, on="hospitalization_id", how="inner")

    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []

    # Overall outcome marginal, gated as a single cell.
    n_total = joined.height
    n_expired = int(joined.filter(pl.col("outcome") == "expired").height)
    marginal = {"expired": n_expired / n_total, "alive": 1.0 - n_expired / n_total}
    surv_marg, marg_audit = suppress({"outcome": n_total}, {"outcome": marginal}, min_n=min_n)
    audit.extend(
        SuppressionRecord(cell=("outcome_marginal", r.cell), n=r.n, fallback_kind=r.fallback_kind)
        for r in marg_audit
    )
    if "outcome" in surv_marg:
        params["outcome_marginal"] = {
            k: round(float(v), 6) for k, v in surv_marg["outcome"].items()
        }

    # Expired rate conditioned on peak acuity, one gated cell per peak level.
    by_level = joined.group_by("peak_level").agg(
        pl.len().alias("n"),
        (pl.col("outcome") == "expired").sum().alias("n_expired"),
    )
    counts = {int(r["peak_level"]): int(r["n"]) for r in by_level.iter_rows(named=True)}
    rates = {
        int(r["peak_level"]): {
            "expired_rate": round(int(r["n_expired"]) / int(r["n"]), 6),
            "n_hospitalizations": int(r["n"]),
        }
        for r in by_level.iter_rows(named=True)
    }
    survived, level_audit = suppress(counts, rates, min_n=min_n)
    audit.extend(
        SuppressionRecord(
            cell=("expired_rate_by_peak_level", r.cell), n=r.n, fallback_kind=r.fallback_kind
        )
        for r in level_audit
    )
    if survived:
        params["expired_rate_by_peak_level"] = {
            str(level): rate for level, rate in survived.items()
        }
    return params, audit


def fit_flag_prevalence(state_timeline: pl.DataFrame, *, min_n: int = 20) -> EstimatorResult:
    """Per-support-level prevalence of each organ-failure flag (R2).

    The spine sampler (U6) draws organ-failure flags coupled to acuity from pack
    params; this emits ``P(flag | support level)`` for each of the four flags
    (respiratory / cardiovascular / renal / neuro) over the intervals observed
    at each level. Each level is gated on its interval count.
    """
    flags = ("resp_flag", "cv_flag", "renal_flag", "neuro_flag")
    present = [f for f in flags if f in state_timeline.columns]
    if not present:
        return {}, []

    by_level = state_timeline.group_by("support_level").agg(
        pl.len().alias("n"),
        *[pl.col(f).sum().alias(f) for f in present],
    )
    counts = {int(r["support_level"]): int(r["n"]) for r in by_level.iter_rows(named=True)}
    prevalence = {
        int(r["support_level"]): {f: round(int(r[f]) / int(r["n"]), 6) for f in present}
        for r in by_level.iter_rows(named=True)
    }
    survived, audit = suppress(counts, prevalence, min_n=min_n)
    audit = [
        SuppressionRecord(
            cell=("flag_prevalence_by_level", r.cell), n=r.n, fallback_kind=r.fallback_kind
        )
        for r in audit
    ]
    params: dict[str, object] = {}
    if survived:
        params["flag_prevalence_by_level"] = {str(level): prev for level, prev in survived.items()}
    return params, audit


# --------------------------------------------------------------------------- #
# Per-state physiology: AR1
# --------------------------------------------------------------------------- #
def fit_ar1_by_state(
    vitals_gridded: pl.DataFrame,
    state_timeline: pl.DataFrame,
    *,
    vitals: Sequence[str],
    min_n: int = 20,
) -> EstimatorResult:
    """First-order autoregression per (vital, support-level) (R2).

    ``vitals_gridded`` must carry columns ``hospitalization_id``,
    ``interval_idx``, ``vital_category``, ``value`` (one mean value per cell).
    The support level is forward-filled onto each vital interval via an as-of
    join; lag-1 pairs are formed only across **adjacent** grid intervals, and
    the pair is attributed to the state at the later interval. Each
    (vital, state) cell is gated at ``min_n`` lag-1 pairs.
    """
    state_sorted = state_timeline.sort("interval_idx")
    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []

    for vital in vitals:
        vf = (
            vitals_gridded.filter(pl.col("vital_category") == vital)
            .select("hospitalization_id", "interval_idx", "value")
            .sort("interval_idx")
        )
        if vf.height == 0:
            continue
        # Forward-fill support level onto each vital interval (as-of backward).
        with_state = vf.join_asof(
            state_sorted.select("hospitalization_id", "interval_idx", "support_level"),
            on="interval_idx",
            by="hospitalization_id",
            strategy="backward",
        ).drop_nulls("support_level")

        # Lag-1 pairs across adjacent intervals within a hospitalization.
        with_state = with_state.sort("hospitalization_id", "interval_idx")
        prev_val = pl.col("value").shift(1).over("hospitalization_id")
        prev_iv = pl.col("interval_idx").shift(1).over("hospitalization_id")
        pairs = (
            with_state.with_columns(prev_val.alias("prev_value"), prev_iv.alias("prev_interval"))
            .filter(pl.col("interval_idx") - pl.col("prev_interval") == 1)
            .select("support_level", "prev_value", "value")
            .drop_nulls()
        )

        by_state: dict[int, tuple[_F64, _F64]] = {}
        for state in pairs["support_level"].unique().to_list():
            sub = pairs.filter(pl.col("support_level") == state)
            by_state[int(state)] = (
                sub["prev_value"].to_numpy(),
                sub["value"].to_numpy(),
            )

        counts = {state: len(x) for state, (x, _y) in by_state.items()}
        fits = {state: _fit_ar1(x_prev, x_curr) for state, (x_prev, x_curr) in by_state.items()}
        survived, records = suppress(counts, fits, min_n=min_n)
        audit.extend(
            SuppressionRecord(cell=(vital, r.cell), n=r.n, fallback_kind=r.fallback_kind)
            for r in records
        )
        if survived:
            params[f"{vital}_ar1_by_state"] = {str(state): fit for state, fit in survived.items()}
    return params, audit


def _fit_ar1(x_prev: _F64, x_curr: _F64) -> dict[str, float]:
    """OLS fit of x_t = mean + phi*(x_{t-1}-mean) + eps; phi clamped stationary.

    Physiologic vital streams carry rare but extreme artifacts (device errors,
    charting mistakes) whose magnitude dwarfs the true signal. Ordinary
    mean/variance are not robust to them: a handful of 100000-mmHg readings
    inflate ``sigma`` by orders of magnitude, which then dominates the generated
    walk and pins values to the outlier-clamp bounds. Both series are trimmed to
    a wide robust band (median +- 5 scaled-MAD, ~3.4 sd for a normal, so genuine
    physiology is kept) before the OLS fit, so ``mean``/``phi``/``sigma`` reflect
    the real distribution rather than its artifacts.
    """
    combined = np.concatenate([x_prev, x_curr])
    med = float(np.median(combined))
    mad = float(np.median(np.abs(combined - med)))
    if mad > 0.0:
        band = 5.0 * 1.4826 * mad
        keep = (np.abs(x_prev - med) <= band) & (np.abs(x_curr - med) <= band)
        if int(keep.sum()) >= 2:
            x_prev, x_curr = x_prev[keep], x_curr[keep]
    mean = float(np.mean(np.concatenate([x_prev, x_curr])))
    cp = x_prev - mean
    cc = x_curr - mean
    denom = float(np.sum(cp * cp))
    phi = float(np.sum(cp * cc) / denom) if denom > 0 else 0.0
    phi = max(-_PHI_CLAMP, min(_PHI_CLAMP, phi))
    resid = cc - phi * cp
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    return {
        "phi": round(phi, 6),
        "sigma": round(sigma, 6),
        "mean": round(mean, 6),
    }


# --------------------------------------------------------------------------- #
# Lab copula
# --------------------------------------------------------------------------- #
def fit_lab_copula(
    labs_gridded: pl.DataFrame,
    *,
    n_hospitalizations: int,
    min_n: int = 20,
    icu_hospitalizations: set[str] | None = None,
) -> EstimatorResult:
    """Co-measurement Spearman correlation + per-lab log-marginals + presence.

    ``labs_gridded`` carries ``hospitalization_id``, ``interval_idx``,
    ``lab_category``, ``value``. Correlation is computed over lab pairs measured
    in the **same** (hospitalization, interval) window, then projected to the
    nearest positive-definite correlation matrix. Per-lab marginals are fit on
    ``log1p`` values (labs are heavy-tailed and non-negative), and a per-lab
    empirical quantile grid (``lab_quantiles``, on :data:`LAB_QUANTILE_PROBS`) is
    fit over the same values so the generator can match each lab's marginal shape
    exactly through an inverse-CDF map. Each lab is gated at ``min_n``
    observations; presence rate is the fraction of hospitalizations with at least
    one measurement.

    ``presence`` is conditioned on the **ICU-exposed** cohort when
    ``icu_hospitalizations`` is supplied — the population the generator targets.
    Computing it over all hospitalizations (many of them non-ICU floor stays with
    few labs) otherwise undershoots ICU presence several-fold (e.g. creatinine
    0.73 vs a real-ICU ~0.99). Marginals and correlation are left over the full
    gridded input; only presence is re-based.
    """
    labs = labs_gridded.drop_nulls("value")
    per_lab_counts = labs.group_by("lab_category").len().rename({"len": "n"})
    counts = {r["lab_category"]: int(r["n"]) for r in per_lab_counts.iter_rows(named=True)}

    marginals: dict[str, dict[str, float]] = {}
    values_by_lab: dict[str, _F64] = {}
    for lab, sub in labs.group_by("lab_category"):
        name = lab[0] if isinstance(lab, tuple) else lab
        vals = sub["value"].to_numpy()
        values_by_lab[name] = vals
        logv = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
        marginals[name] = {
            "log_mean": round(float(np.mean(logv)), 6),
            "log_sd": round(float(np.std(logv, ddof=1)) if logv.size > 1 else 0.0, 6),
        }

    survived_marginals, audit = suppress(counts, marginals, min_n=min_n)
    lab_order = sorted(survived_marginals)

    # Empirical inverse-CDF marginal per surviving lab: the fitted quantile grid on
    # ``LAB_QUANTILE_PROBS`` over the SAME null-dropped gridded values used for the
    # log-normal marginal. Driven through the copula's probability-integral transform
    # at generation, it matches each lab's real marginal exactly (the log-normal block
    # above is retained as a fallback for older/sparser labs). Non-decreasing by
    # construction. Emitted only for labs with enough records that the fine grid is a
    # genuine aggregate (>= 20 records per interval, mirroring the n>=20 cell floor);
    # a fine grid over sparse values approaches the raw sorted data, so sparse labs
    # keep the two-parameter log-normal marginal — this also keeps the pack past the
    # value-level leakage scan (array length must stay well under the record count).
    lab_quantiles: dict[str, list[float]] = {
        lab: [round(float(q), 4) for q in np.quantile(values_by_lab[lab], LAB_QUANTILE_PROBS)]
        for lab in lab_order
        if counts.get(lab, 0) >= _QUANTILE_MIN_RECORDS
    }

    if icu_hospitalizations is not None:
        presence_labs = labs.filter(pl.col("hospitalization_id").is_in(list(icu_hospitalizations)))
        presence_denom = len(icu_hospitalizations)
    else:
        presence_labs = labs
        presence_denom = n_hospitalizations

    presence = {}
    for lab in lab_order:
        n_hosp_with = (
            presence_labs.filter(pl.col("lab_category") == lab)
            .select(pl.col("hospitalization_id").n_unique())
            .item()
        )
        presence[lab] = round(n_hosp_with / presence_denom, 6) if presence_denom > 0 else 0.0

    correlation = _co_measurement_correlation(labs, lab_order)
    if icu_hospitalizations is not None:
        presence_cohort = list(icu_hospitalizations)
    else:
        presence_cohort = presence_labs.select("hospitalization_id").unique().to_series().to_list()
    presence_correlation = _presence_correlation(presence_labs, presence_cohort, lab_order)

    params: dict[str, object] = {
        "lab_order": lab_order,
        "lab_marginals": survived_marginals,
        "lab_quantiles": lab_quantiles,
        "lab_presence": presence,
        "lab_correlation": correlation,
        "lab_presence_correlation": presence_correlation,
    }
    return params, audit


def _co_measurement_correlation(labs: pl.DataFrame, lab_order: Sequence[str]) -> list[list[float]]:
    """Spearman correlation over co-measured labs -> nearest-PD matrix."""
    k = len(lab_order)
    if k == 0:
        return []
    if k == 1:
        return [[1.0]]

    wide = (
        labs.filter(pl.col("lab_category").is_in(list(lab_order)))
        .group_by("hospitalization_id", "interval_idx", "lab_category")
        .agg(pl.col("value").mean().alias("value"))
        .pivot(on="lab_category", index=["hospitalization_id", "interval_idx"], values="value")
    )
    # Rank each lab column (Spearman = Pearson on ranks), keeping NaN for gaps.
    cols = [c for c in lab_order if c in wide.columns]
    mat = wide.select(cols).to_numpy().astype(float)

    corr = np.eye(k)
    index = {name: i for i, name in enumerate(lab_order)}
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            xa = mat[:, a]
            xb = mat[:, b]
            both = ~np.isnan(xa) & ~np.isnan(xb)
            if both.sum() >= 3:
                rho, _ = stats.spearmanr(xa[both], xb[both])
                if np.isfinite(rho):
                    ia, ib = index[cols[a]], index[cols[b]]
                    corr[ia, ib] = corr[ib, ia] = float(rho)

    pd_corr = nearest_positive_definite_correlation(corr)
    return [[round(float(v), 6) for v in row] for row in pd_corr]


def _presence_correlation(
    presence_labs: pl.DataFrame, cohort_ids: Sequence[str], lab_order: Sequence[str]
) -> list[list[float]]:
    """Phi (Pearson-on-indicator) correlation of per-stay lab presence over the cohort.

    Real labs are ordered as panels — a basic metabolic panel yields sodium,
    potassium, chloride, ... together; an arterial blood gas yields po2/pco2/ph
    together — so their per-stay presence is strongly co-occurrent (measured
    Jaccard ~0.97 for arterial gases, ~1.0 for the metabolic panel). Drawing each
    lab's presence independently inflates any panel's *union* several-fold. This
    fitted correlation lets the generator draw presence through a Gaussian copula
    that preserves each lab's marginal presence while restoring co-occurrence.
    Constant-presence labs (always/never measured) carry no off-diagonal mass.
    Computed over the full cohort (absent stays contribute all-zero rows) and
    nearest-PD projected.
    """
    k = len(lab_order)
    if k == 0:
        return []
    if k == 1:
        return [[1.0]]
    n = len(cohort_ids)
    corr = np.eye(k)
    if n == 0:
        return [[round(float(v), 6) for v in row] for row in corr]

    # Stay x lab {0,1} indicator over the full cohort (left-join fills absent stays 0).
    seen = (
        presence_labs.filter(pl.col("lab_category").is_in(list(lab_order)))
        .select("hospitalization_id", "lab_category")
        .unique()
        .with_columns(pl.lit(1.0).alias("present"))
        .pivot(on="lab_category", index="hospitalization_id", values="present")
    )
    cohort = pl.DataFrame({"hospitalization_id": list(cohort_ids)})
    wide = cohort.join(seen, on="hospitalization_id", how="left").fill_null(0.0)
    present_cols = [c for c in lab_order if c in wide.columns]
    ind = wide.select(present_cols).to_numpy().astype(float)

    # Phi correlation = Pearson on indicators; skip constant (zero-variance) columns.
    index = {name: i for i, name in enumerate(lab_order)}
    active_local = [j for j in range(len(present_cols)) if ind[:, j].std() > 0.0]
    if len(active_local) >= 2:
        sub = ind[:, active_local]
        cmat = np.corrcoef(sub, rowvar=False)
        for aj, a in enumerate(active_local):
            for bj, b in enumerate(active_local):
                if a != b and np.isfinite(cmat[aj, bj]):
                    ia, ib = index[present_cols[a]], index[present_cols[b]]
                    corr[ia, ib] = float(cmat[aj, bj])

    pd_corr = nearest_positive_definite_correlation(corr)
    return [[round(float(v), 6) for v in row] for row in pd_corr]


def nearest_positive_definite_correlation(matrix: _F64) -> _F64:
    """Project a symmetric matrix to the nearest PD correlation matrix.

    Clips negative eigenvalues to a small positive floor, reconstructs, then
    rescales to unit diagonal. Sufficient for sampling a Gaussian copula; not
    the full Higham iteration, but PD and correlation-normalized.
    """
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, a_min=1e-6, a_max=None)
    rebuilt = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(rebuilt), a_min=1e-12, a_max=None))
    normalized = rebuilt / np.outer(d, d)
    np.fill_diagonal(normalized, 1.0)
    result: _F64 = ((normalized + normalized.T) / 2.0).astype(np.float64)
    return result


# --------------------------------------------------------------------------- #
# Infusion hazards
# --------------------------------------------------------------------------- #
def fit_infusion_hazards(
    mac_gridded: pl.DataFrame,
    *,
    min_n: int = 20,
) -> EstimatorResult:
    """Per-drug start/stop hazard per interval (R2).

    ``mac_gridded`` carries ``hospitalization_id``, ``interval_idx``,
    ``med_category`` for every interval a continuous drug is active. Start
    hazard = starts / (starts + off-intervals-at-risk) approximated as
    starts / hospitalization-exposures; stop hazard = stops / on-intervals.
    Each drug is gated at ``min_n`` on-intervals. Doses are intentionally not
    emitted here (dose marginals belong to the med marginal block).
    """
    active = mac_gridded.select("hospitalization_id", "interval_idx", "med_category").unique()
    active = active.sort("hospitalization_id", "med_category", "interval_idx")

    prev_iv = pl.col("interval_idx").shift(1).over(["hospitalization_id", "med_category"])
    marked = active.with_columns(
        ((pl.col("interval_idx") - prev_iv != 1) | prev_iv.is_null()).alias("_is_start")
    )

    per_drug = marked.group_by("med_category").agg(
        pl.len().alias("on_intervals"),
        pl.col("_is_start").sum().alias("starts"),
    )

    counts = {r["med_category"]: int(r["on_intervals"]) for r in per_drug.iter_rows(named=True)}
    hazards: dict[str, dict[str, float]] = {}
    for r in per_drug.iter_rows(named=True):
        drug = r["med_category"]
        on = int(r["on_intervals"])
        starts = int(r["starts"])
        # A "run" of consecutive on-intervals begins at each start; the stop
        # hazard is (#runs) / (on-intervals) = mean 1/duration per interval.
        stop_hazard = starts / on if on > 0 else 0.0
        hazards[drug] = {
            "stop_hazard": round(stop_hazard, 6),
            "mean_run_intervals": round(on / starts, 4) if starts > 0 else 0.0,
        }

    survived, audit = suppress(counts, hazards, min_n=min_n)
    params: dict[str, object] = {"infusion_hazards": survived}
    return params, audit


# --------------------------------------------------------------------------- #
# Prior-table estimators (source CLIF realism / all-28 pack)
# --------------------------------------------------------------------------- #
#: Code-status categories that count as a de-escalation from Full.
_DNR_LIKE: frozenset[str] = frozenset(
    {"DNR", "DNAR", "UDNR", "DNR/DNI", "DNAR/DNI", "DNI_only"}
)
_AND_LIKE: frozenset[str] = frozenset({"AND"})


def fit_age_quantiles(
    hospitalization: pl.DataFrame,
    *,
    column: str = "age_at_admission",
    n_bins: int = 10,
    min_n: int = 20,
) -> EstimatorResult:
    """Empirical age quantiles for hospitalization (``age_at_admission_quantiles``)."""
    if column not in hospitalization.columns:
        return {}, []
    col = hospitalization.select(pl.col(column).drop_nulls().cast(pl.Float64)).to_series()
    n = col.len()
    counts = {column: n}
    probs = [i / n_bins for i in range(n_bins + 1)]
    raw = [col.quantile(q, interpolation="linear") for q in probs]
    edges = [round(float(e), 4) for e in raw if e is not None]
    survived, audit = suppress(counts, {column: edges}, min_n=min_n)
    if column not in survived or len(survived[column]) < 2:
        return {}, audit
    return {"age_at_admission_quantiles": survived[column]}, audit


def fit_stay_prevalence(
    df: pl.DataFrame,
    n_hospitalizations: int,
    *,
    id_col: str = "hospitalization_id",
    min_n: int = 20,
) -> EstimatorResult:
    """Fraction of hospitalizations with ≥1 row in ``df`` (gated on stay count)."""
    if id_col not in df.columns or n_hospitalizations <= 0:
        return {}, []
    n_with = int(df.select(pl.col(id_col).n_unique()).item())
    counts = {"stay_prevalence": n_with}
    raw = {"stay_prevalence": n_with / n_hospitalizations}
    survived, audit = suppress(counts, raw, min_n=min_n)
    if "stay_prevalence" not in survived:
        return {}, audit
    return {"stay_prevalence": round(float(survived["stay_prevalence"]), 6)}, audit


def fit_stay_prevalence_by_category(
    df: pl.DataFrame,
    n_hospitalizations: int,
    *,
    category_col: str,
    categories: Sequence[str],
    id_col: str = "hospitalization_id",
    min_n: int = 20,
) -> EstimatorResult:
    """Per-category stay prevalence (e.g. IMV / NIPPV / HFNC among hospitalizations)."""
    if id_col not in df.columns or category_col not in df.columns or n_hospitalizations <= 0:
        return {}, []
    counts: dict[str, int] = {}
    raw: dict[str, float] = {}
    for cat in categories:
        n_with = int(
            df.filter(pl.col(category_col) == cat).select(pl.col(id_col).n_unique()).item()
        )
        counts[cat] = n_with
        raw[cat] = n_with / n_hospitalizations
    survived, audit = suppress(counts, raw, min_n=min_n)
    if not survived:
        return {}, audit
    return {
        "stay_prevalence_by_category": {
            cat: round(float(p), 6) for cat, p in survived.items()
        }
    }, audit


def fit_code_status_rates(
    code_status: pl.DataFrame,
    hospitalization: pl.DataFrame,
    *,
    min_n: int = 20,
) -> EstimatorResult:
    """Outcome-conditional DNR / comfort-care rates (patient-level table).

    Emits ``dnr_prob_expired``, ``comfort_prob_expired`` (among DNR decedents),
    and ``dnr_prob_survivor``, matching the code_status generator knobs.
    """
    if "patient_id" not in code_status.columns or "code_status_category" not in code_status.columns:
        return {}, []
    if "patient_id" not in hospitalization.columns or "discharge_category" not in hospitalization.columns:
        return {}, []

    flags = (
        code_status.group_by("patient_id")
        .agg(pl.col("code_status_category").alias("_cats"))
        .with_columns(
            pl.col("_cats")
            .list.eval(pl.element().is_in(list(_DNR_LIKE)))
            .list.any()
            .alias("has_dnr"),
            pl.col("_cats")
            .list.eval(pl.element().is_in(list(_AND_LIKE)))
            .list.any()
            .alias("has_and"),
        )
        .select("patient_id", "has_dnr", "has_and")
    )
    outcomes = hospitalization.group_by("patient_id").agg(
        (pl.col("discharge_category") == "Expired").any().alias("expired")
    )
    joined = flags.join(outcomes, on="patient_id", how="inner")
    expired = joined.filter(pl.col("expired"))
    survivors = joined.filter(~pl.col("expired"))

    params: dict[str, object] = {}
    audit: list[SuppressionRecord] = []

    def _rate(subset: pl.DataFrame, col: str, key: str) -> None:
        n = subset.height
        if n == 0:
            return
        rate = float(subset.select(pl.col(col).mean()).item() or 0.0)
        survived, records = suppress({key: n}, {key: rate}, min_n=min_n)
        audit.extend(records)
        if key in survived:
            params[key] = round(float(survived[key]), 6)

    _rate(expired, "has_dnr", "dnr_prob_expired")
    _rate(survivors, "has_dnr", "dnr_prob_survivor")

    dnr_expired = expired.filter(pl.col("has_dnr"))
    n_dnr = dnr_expired.height
    if n_dnr > 0:
        comfort = float(dnr_expired.select(pl.col("has_and").mean()).item() or 0.0)
        survived, records = suppress(
            {"comfort_prob_expired": n_dnr}, {"comfort_prob_expired": comfort}, min_n=min_n
        )
        audit.extend(records)
        if "comfort_prob_expired" in survived:
            params["comfort_prob_expired"] = round(float(survived["comfort_prob_expired"]), 6)

    # Category marginal for generators that sample terminal status directly.
    cat_params, cat_audit = fit_categorical_marginals(
        code_status, ["code_status_category"], min_n=min_n
    )
    params.update(cat_params)
    audit.extend(cat_audit)
    return params, audit


def fit_top_k_category(
    df: pl.DataFrame,
    field: str,
    *,
    k: int = 40,
    min_n: int = 20,
) -> EstimatorResult:
    """Top-``k`` category proportions; long tail dropped (survives leakage scan)."""
    if field not in df.columns:
        return {}, []
    vc = (
        df.select(pl.col(field).drop_nulls().cast(pl.String))
        .to_series()
        .value_counts(sort=True)
    )
    if vc.height == 0:
        return {}, []
    top = vc.head(k)
    counts = {str(row[0]): int(row[1]) for row in top.iter_rows()}
    total = sum(counts.values())
    if total == 0:
        return {}, []
    raw = {cat: n / total for cat, n in counts.items()}
    survived, audit = suppress(counts, raw, min_n=min_n)
    surviving_total = sum(survived.values())
    if surviving_total <= 0:
        return {}, audit
    return {
        f"{field}_marginal": {
            cat: prop / surviving_total for cat, prop in survived.items()
        }
    }, audit


def fit_cultures_per_icu_day(
    cultures: pl.DataFrame,
    adt: pl.DataFrame | None,
    *,
    min_n: int = 20,
) -> EstimatorResult:
    """Aggregate cultures / ICU-day from ADT ICU location windows."""
    n_cult = cultures.height
    if n_cult < min_n:
        audit = [
            SuppressionRecord(cell=("cultures_per_icu_day",), n=n_cult, fallback_kind="none")
        ]
        return {}, audit
    icu_days = 0.0
    if (
        adt is not None
        and "location_category" in adt.columns
        and "in_dttm" in adt.columns
        and "out_dttm" in adt.columns
    ):
        hours = (
            adt.filter(pl.col("location_category") == "icu")
            .select(((pl.col("out_dttm") - pl.col("in_dttm")).dt.total_hours()).sum())
            .item()
        )
        icu_days = float(hours or 0.0) / 24.0
    if icu_days <= 0:
        return {}, []
    rate = n_cult / icu_days
    return {"cultures_per_icu_day": round(rate, 6)}, []


def fit_adt_arrival(
    adt: pl.DataFrame,
    *,
    icu_location: str = "icu",
    min_n: int = 20,
) -> EstimatorResult:
    """Fit the validated ADT front-door knobs from real first-location segments.

    Emits the same parameter names the ADT generator already consumes
    (``arrival_location_marginal``, ``direct_icu_frac``) — the path exercised by
    full-hospital / network-median recalibration — so fitted ICU realism plugs into the
    proven arrival machinery rather than inventing a parallel one.

    * ``arrival_location_marginal`` — first ``location_category`` among stays that
      ever visit ICU (the synthetic ICU cohort's front door).
    * ``direct_icu_frac`` — among those ICU-reaching stays, fraction whose first
      segment is already ``icu``.
    """
    need = {"hospitalization_id", "location_category", "in_dttm"}
    if not need.issubset(adt.columns):
        return {}, []

    first = (
        adt.sort("in_dttm")
        .group_by("hospitalization_id")
        .agg(pl.col("location_category").first().alias("_arrival"))
    )
    icu_ids = (
        adt.filter(pl.col("location_category") == icu_location)
        .select("hospitalization_id")
        .unique()
    )
    cohort = first.join(icu_ids, on="hospitalization_id", how="inner")
    n = cohort.height
    if n < min_n:
        return {}, [
            SuppressionRecord(cell=("arrival_location_marginal",), n=n, fallback_kind="none")
        ]

    vc = cohort["_arrival"].value_counts(sort=True)
    counts = {str(row[0]): int(row[1]) for row in vc.iter_rows() if row[0] is not None}
    total = sum(counts.values())
    raw = {cat: c / total for cat, c in counts.items()}
    survived, audit = suppress(counts, raw, min_n=min_n)
    surviving_total = sum(survived.values())
    params: dict[str, object] = {}
    if surviving_total > 0:
        params["arrival_location_marginal"] = {
            cat: prop / surviving_total for cat, prop in survived.items()
        }

    n_direct = int(cohort.filter(pl.col("_arrival") == icu_location).height)
    s2, a2 = suppress(
        {"direct_icu_frac": n},
        {"direct_icu_frac": n_direct / n},
        min_n=min_n,
    )
    audit.extend(a2)
    if "direct_icu_frac" in s2:
        params["direct_icu_frac"] = round(float(s2["direct_icu_frac"]), 6)
    return params, audit


def fit_admission_route_marginal(
    hospitalization: pl.DataFrame,
    *,
    field: str = "admission_type_category",
    min_n: int = 20,
) -> EstimatorResult:
    """Fit the coupled spine ``admission_route_marginal`` from admission types.

    Same knob full-hospital recalibration sets: one draw per stay drives both
    hospitalization ``admission_type_category`` and the ADT front door via
    ``adt._ROUTE_TO_ARRIVAL``.
    """
    if field not in hospitalization.columns:
        return {}, []
    params, audit = fit_categorical_marginals(hospitalization, [field], min_n=min_n)
    marginal = params.get(f"{field}_marginal")
    if not isinstance(marginal, dict) or not marginal:
        return {}, audit
    return {"admission_route_marginal": dict(marginal)}, audit


def fit_prone_rates(
    position: pl.DataFrame,
    respiratory_support: pl.DataFrame | None,
    *,
    min_n: int = 20,
) -> EstimatorResult:
    """Prone chart probability overall and among IMV stays (position generator knobs).

    Emits ``prone_prob_otherwise`` (global chart rate) and, when IMV stays are
    available, ``prone_prob_severe`` as the prone-chart rate among IMV stays.
    """
    if "position_category" not in position.columns:
        return {}, []
    n = position.height
    prone_n = int(position.filter(pl.col("position_category") == "prone").height)
    counts = {"prone_prob_otherwise": n}
    raw = {"prone_prob_otherwise": prone_n / n if n else 0.0}
    survived, audit = suppress(counts, raw, min_n=min_n)
    params: dict[str, object] = {}
    if "prone_prob_otherwise" in survived:
        params["prone_prob_otherwise"] = round(float(survived["prone_prob_otherwise"]), 6)

    if (
        respiratory_support is not None
        and "device_category" in respiratory_support.columns
        and "hospitalization_id" in respiratory_support.columns
        and "hospitalization_id" in position.columns
    ):
        imv_ids = (
            respiratory_support.filter(pl.col("device_category") == "IMV")
            .select("hospitalization_id")
            .unique()
        )
        imv_pos = position.join(imv_ids, on="hospitalization_id", how="inner")
        n_imv = imv_pos.height
        if n_imv > 0:
            prone_imv = int(imv_pos.filter(pl.col("position_category") == "prone").height)
            s2, a2 = suppress(
                {"prone_prob_severe": n_imv},
                {"prone_prob_severe": prone_imv / n_imv},
                min_n=min_n,
            )
            audit.extend(a2)
            if "prone_prob_severe" in s2:
                params["prone_prob_severe"] = round(float(s2["prone_prob_severe"]), 6)
    return params, audit
