"""Tier 3 ``labs`` generator (U11; R9, R12, KTD-4, KTD-6).

Labs are drawn from a **hand-rolled Gaussian copula** (no SDV/copulas dependency,
R-stack constraint): the pack carries a positive-definite Spearman correlation
matrix over labs measured together (``lab_correlation``, indexed by ``lab_order``),
per-lab log-normal marginals fit on ``log1p`` values (``lab_marginals``), and
per-hospitalization presence rates (``lab_presence``).

Generation is **sample-then-mask**, never impute-then-sparsify, so no CLIF
missingness artifact is manufactured (KTD-4):

1. Draw the present-set once per hospitalization — each lab is present for the
   stay with its fitted ``lab_presence`` probability. This matches the fit
   definition (presence = fraction of hospitalizations with >=1 measurement).
   Presence is drawn through a Gaussian copula over the fitted
   ``lab_presence_correlation`` so co-ordered panels (metabolic panel, arterial
   blood gas) appear together and their union does not inflate; the copula
   preserves each lab's marginal presence. Packs without that matrix fall back
   to independent Bernoulli draws.
2. At each order time in the stay's ICU windows, draw the **full** correlated
   45-vector from the copula (``z = L @ N(0, I)`` with ``L`` the Cholesky factor
   of the correlation), map each component through its marginal, and emit a row
   **only** for labs in the present-set. When the pack carries ``lab_quantiles``,
   each component is mapped by the probability-integral transform — ``u = Φ(z)``
   then inverse-CDF via the fitted quantile grid (``np.interp(u, probs, grid)``) —
   so the lab's marginal matches real exactly (a single log-normal cannot capture
   creatinine's CKD tail, lactate's skew, or troponin's bimodality) while the
   copula's rank-correlation is preserved. Packs without it fall back to the
   log-normal marginal (``value = expm1(log_mean + log_sd * z)``). Neither map
   draws rng, so the stream is identical. Absent labs are native nulls (no row),
   never imputed.

Every value is clamped into the consortium outlier bounds (R9). Creatinine and
bun are shifted up in log space when the spine's renal-failure flag is set at the
order interval — a documented R12 clinical coupling (the copula itself is
acuity-agnostic, so this is an explicit coupling, not a fitted mechanism), kept
to classic renal markers rather than invented broadly (R15). The spine is the
only cross-table channel (KTD-6); this generator never reads another table.

``reference_unit`` and ``lab_order_category`` come from the vendored mCIDE
crosswalk on ``lab_category`` (exact consortium pairings). Collect/result times
follow the same order ≤ collect < result prior as microbiology cultures, with
shorter chem-panel turnaround. Specimen type and LOINC come from the curated
``lab_category`` crosswalk (:mod:`clifforge.generate.loinc`) — never invented
ad-hoc codes.

Output is reproducible byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.special import ndtr, ndtri

from clifforge.fit.estimators import LAB_QUANTILE_PROBS
from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import ICU_MIN_SUPPORT_LEVEL, UTC_DATETIME, grid_step_hours
from clifforge.generate.loinc import lab_loinc_code, lab_specimen
from clifforge.generate.spine import SpineFrame
from clifforge.reference import bounds, loader

__all__ = ["LabObservation", "labs_frame", "sample_labs"]


#: Target spacing between lab panels within ICU time (labs are ~daily in the ICU).
_LAB_PANEL_INTERVAL_HOURS = 24.0

#: Classic renal-failure markers shifted up when the spine renal flag is set, and
#: the additive shift in log1p space (~doubles creatinine) — a documented R12
#: clinical coupling, not a fitted quantity.
_RENAL_MARKERS = frozenset({"creatinine", "bun"})
_RENAL_LOG_SHIFT = 0.75
#: Value-space equivalent of the log1p-space renal shift, for the empirical-quantile
#: marginal path (which produces a value directly, not a log1p value): a multiplicative
#: bump ``exp(_RENAL_LOG_SHIFT)`` (~2.1), so creatinine/bun still rise with renal
#: failure and CRRT stays land in the top creat quartile (reference high_creat|CRRT≈0.97).
_RENAL_VALUE_FACTOR = float(np.exp(_RENAL_LOG_SHIFT))
#: Progressive renal derangement: consecutive renal-flag hours scale the bump up
#: toward full strength over ~12h so creat *rises* within a stay (sicker→sicker).
_RENAL_RAMP_HOURS = 12.0

#: Shock / hypoperfusion markers bumped when ``cv_flag`` is on (lactate rises with
#: pressors and falling MAP — longitudinal pairing, not a free invention).
_SHOCK_MARKERS = frozenset({"lactate"})
_SHOCK_LOG_SHIFT = 0.4
_SHOCK_VALUE_FACTOR = float(np.exp(_SHOCK_LOG_SHIFT))

#: Soft leukocytosis when respiratory failure / IMV-tier acuity is active
#: (spine-available infection proxy — no invented culture positivity).
#: Soft infection proxy: vaso-tier + respiratory failure flag (not all IMV).
_INFLAMMATION_MARKERS = frozenset({"wbc"})
_INFLAMMATION_LOG_SHIFT = 0.35
_INFLAMMATION_VALUE_FACTOR = float(np.exp(_INFLAMMATION_LOG_SHIFT))
_INFLAMMATION_MIN_SUPPORT = 4

#: Collect delay after order (minutes) and result delay after collect (hours) —
#: documented chem-panel priors, shorter than culture turnaround.
_COLLECT_DELAY_MINUTES = (5.0, 60.0)
_RESULT_DELAY_HOURS = (0.5, 6.0)

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


def _renal_run_hours(
    renal_flag: list[bool], interval_idx: int, grid_step: float
) -> float:
    """Consecutive renal-flag hours ending at ``interval_idx`` (for rising creat)."""
    run = 0
    for j in range(interval_idx, -1, -1):
        if not renal_flag[j]:
            break
        run += 1
    return run * grid_step


def _apply_clinical_lab_bumps(
    lab: str,
    value: float,
    *,
    renal: bool,
    renal_frac: float,
    shock: bool,
    inflammation: bool,
    log_space: bool,
) -> float:
    """Apply R12 organ-failure bumps; renal ramps with consecutive flag hours."""
    if renal and lab in _RENAL_MARKERS:
        strength = 0.35 + 0.65 * min(1.0, renal_frac)
        if log_space:
            value += _RENAL_LOG_SHIFT * strength
        else:
            value *= 1.0 + (_RENAL_VALUE_FACTOR - 1.0) * strength
    if shock and lab in _SHOCK_MARKERS:
        if log_space:
            value += _SHOCK_LOG_SHIFT
        else:
            value *= _SHOCK_VALUE_FACTOR
    elif lab in _SHOCK_MARKERS and not shock:
        # Soft cap: non-shock hyperlactatemia is uncommon (vaso|lactate ≈ reference).
        value = min(value, 3.0) if not log_space else min(value, float(np.log1p(3.0)))
    if inflammation and lab in _INFLAMMATION_MARKERS:
        if log_space:
            value += _INFLAMMATION_LOG_SHIFT
        else:
            value *= _INFLAMMATION_VALUE_FACTOR
    return value


@dataclass(frozen=True)
class LabObservation:
    """One observed lab result in the long ``labs`` table."""

    hospitalization_id: str
    lab_order_dttm: datetime
    lab_collect_dttm: datetime
    lab_result_dttm: datetime
    lab_name: str
    lab_category: str
    lab_value: str
    lab_value_numeric: float


def _labs_params(pack: ParamPack) -> dict[str, Any]:
    block = pack.tables.get("labs")
    if block is None or "params" not in block:
        raise ValueError("parameter pack has no fitted 'labs' block to sample from")
    params: dict[str, Any] = block["params"]
    return params


def _cholesky(correlation: list[list[float]]) -> npt.NDArray[np.float64]:
    """Cholesky factor of the copula correlation, jittered onto the PD cone.

    The fit projects to the nearest PD matrix, but round-tripping through JSON can
    leave it merely PSD; escalating diagonal jitter recovers a usable factor.
    """
    mat = np.asarray(correlation, dtype=float)
    eye = np.eye(mat.shape[0])
    for jitter in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
        try:
            return np.linalg.cholesky(mat + jitter * eye)
        except np.linalg.LinAlgError:
            continue
    raise ValueError("lab correlation matrix is not positive semi-definite")


def _panel_intervals(
    support_level: list[int],
    grid_step: float,
    *,
    icu_hours: float = _LAB_PANEL_INTERVAL_HOURS,
    ward_hours: float | None = None,
) -> list[int]:
    """Interval indices at which to draw a lab panel.

    By default panels are drawn only inside ICU time, spaced ``icu_hours`` apart
    (~daily). A derived pack can shorten ``icu_hours`` (real ICUs draw chem panels
    roughly twice daily) and set ``ward_hours`` to also draw sparser panels during
    non-ICU time — real labs span the whole hospital stay, not just the ICU
    window, so ICU-only daily panels under-produce lab volume several-fold. With
    the defaults (``ward_hours=None``) the schedule is unchanged.
    """
    panels: list[int] = []
    last: int | None = None
    for idx, level in enumerate(support_level):
        if level >= ICU_MIN_SUPPORT_LEVEL:
            interval = icu_hours
        elif ward_hours is not None:
            interval = ward_hours
        else:
            continue
        if last is None or (idx - last) * grid_step >= interval:
            panels.append(idx)
            last = idx
    return panels


def _clamp(value: float, lab: str) -> float:
    try:
        lower, upper = bounds("labs", lab)
    except (LookupError, ValueError):
        # reference.bounds raises ReferenceDataError (a LookupError) when a lab has
        # no fitted outlier bounds — leave the value as drawn rather than crash.
        return value
    return min(max(value, lower), upper)


def sample_labs(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[LabObservation]:
    """Emit one hospitalization's observed labs as a long list of rows (R9, R22).

    Draws the stay's present-set from ``lab_presence`` once, then a full correlated
    copula vector per ICU order time, emitting rows only for present labs.
    ``hospitalization_id`` defaults to the spine's own id.
    """
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    params = _labs_params(pack)
    order: list[str] = params["lab_order"]
    marginals: dict[str, dict[str, float]] = params["lab_marginals"]
    quantiles: dict[str, list[float]] | None = params.get("lab_quantiles")
    presence: dict[str, float] = params["lab_presence"]
    chol = _cholesky(params["lab_correlation"])
    grid_step = grid_step_hours(pack)
    n = len(order)

    # (1) present-set for the whole stay — one draw, matching the fit's
    #     per-hospitalization presence definition. When the pack carries a fitted
    #     presence correlation, draw through a Gaussian copula so co-ordered panels
    #     (a metabolic panel, an arterial blood gas) are present together and a
    #     panel's union does not inflate; the copula preserves each lab's marginal
    #     presence exactly. Older/demo packs without it fall back to independent
    #     Bernoulli draws.
    presence_vec = np.array([presence.get(lab, 0.0) for lab in order], dtype=float)
    pres_corr = params.get("lab_presence_correlation")
    if pres_corr:
        latent = _cholesky(pres_corr) @ rng.standard_normal(n)
        present_mask = latent < ndtri(presence_vec)  # P(latent < Φ⁻¹(p)) = p
    else:
        present_mask = rng.random(n) < presence_vec

    icu_hours = float(params.get("panel_interval_hours", _LAB_PANEL_INTERVAL_HOURS))
    ward_hours_raw = params.get("ward_panel_interval_hours")
    ward_hours = float(ward_hours_raw) if ward_hours_raw is not None else None

    observations: list[LabObservation] = []
    for interval_idx in _panel_intervals(
        spine.support_level, grid_step, icu_hours=icu_hours, ward_hours=ward_hours
    ):
        z = chol @ rng.standard_normal(n)  # (2) correlated latent draw per panel
        jitter = rng.random() * grid_step
        order_dttm = admit_dttm + timedelta(hours=interval_idx * grid_step + jitter)
        collect_dttm = order_dttm + timedelta(
            minutes=float(rng.uniform(*_COLLECT_DELAY_MINUTES))
        )
        result_dttm = collect_dttm + timedelta(hours=float(rng.uniform(*_RESULT_DELAY_HOURS)))
        renal = spine.renal_flag[interval_idx]
        shock = spine.cv_flag[interval_idx]
        inflammation = (
            spine.support_level[interval_idx] >= _INFLAMMATION_MIN_SUPPORT
            and interval_idx < len(spine.resp_flag)
            and spine.resp_flag[interval_idx]
        )
        renal_frac = (
            _renal_run_hours(spine.renal_flag, interval_idx, grid_step) / _RENAL_RAMP_HOURS
            if renal
            else 0.0
        )
        for i, lab in enumerate(order):
            if not present_mask[i]:
                continue
            grid = quantiles.get(lab) if quantiles is not None else None
            if grid is not None:
                # Empirical inverse-CDF marginal: push the latent through the
                # standard-normal CDF (probability-integral transform) then invert
                # via the fitted quantile grid, so the lab's marginal matches real
                # exactly while the copula's rank-correlation (carried by ``z``) is
                # preserved. ``ndtr`` and ``np.interp`` draw no rng — the stream and
                # draw counts are identical to the log-normal path.
                u = float(ndtr(float(z[i])))
                value = float(np.interp(u, LAB_QUANTILE_PROBS, grid))
                value = _apply_clinical_lab_bumps(
                    lab,
                    value,
                    renal=renal,
                    renal_frac=renal_frac,
                    shock=shock,
                    inflammation=inflammation,
                    log_space=False,
                )
            else:
                marg = marginals.get(lab)
                if marg is None:
                    continue
                log_val = marg["log_mean"] + marg["log_sd"] * float(z[i])
                log_val = _apply_clinical_lab_bumps(
                    lab,
                    log_val,
                    renal=renal,
                    renal_frac=renal_frac,
                    shock=shock,
                    inflammation=inflammation,
                    log_space=True,
                )
                value = float(np.expm1(log_val))
            value = _clamp(value, lab)
            value = round(value, 4)
            observations.append(
                LabObservation(
                    hospitalization_id=hid,
                    lab_order_dttm=order_dttm,
                    lab_collect_dttm=collect_dttm,
                    lab_result_dttm=result_dttm,
                    lab_name=lab,
                    lab_category=lab,
                    lab_value=f"{value:g}",
                    lab_value_numeric=value,
                )
            )

    observations.sort(key=lambda o: (o.lab_order_dttm, o.lab_category))
    return observations


def labs_frame(observations: list[LabObservation]) -> pl.DataFrame:
    """Stack observed labs into one conformant long ``labs`` frame.

    ``reference_unit`` / ``lab_order_category`` / ``lab_order_name`` are filled from
    the vendored mCIDE companion columns so every emitted category pairing is an
    exact consortium member (R5).
    """
    unit_by_lab = loader.crosswalk("labs", "lab_category", "reference_unit")
    order_cat_by_lab = loader.crosswalk("labs", "lab_category", "lab_order_category")
    order_name_by_cat = loader.crosswalk("labs", "lab_order_category", "description")

    order_categories = [order_cat_by_lab[o.lab_category] for o in observations]
    specimens = [lab_specimen(o.lab_category) for o in observations]
    return pl.DataFrame(
        {
            "hospitalization_id": [o.hospitalization_id for o in observations],
            "lab_order_dttm": [o.lab_order_dttm for o in observations],
            "lab_collect_dttm": [o.lab_collect_dttm for o in observations],
            "lab_result_dttm": [o.lab_result_dttm for o in observations],
            "lab_order_name": [
                order_name_by_cat.get(cat, cat) or cat for cat in order_categories
            ],
            "lab_order_category": order_categories,
            "lab_name": [o.lab_name for o in observations],
            "lab_category": [o.lab_category for o in observations],
            "lab_value": [o.lab_value for o in observations],
            "lab_value_numeric": [o.lab_value_numeric for o in observations],
            "reference_unit": [unit_by_lab[o.lab_category] for o in observations],
            "lab_specimen_name": [s[1] for s in specimens],
            "lab_specimen_category": [s[0] for s in specimens],
            "lab_loinc_code": [lab_loinc_code(o.lab_category) for o in observations],
        },
        schema={
            "hospitalization_id": pl.String,
            "lab_order_dttm": UTC_DATETIME,
            "lab_collect_dttm": UTC_DATETIME,
            "lab_result_dttm": UTC_DATETIME,
            "lab_order_name": pl.String,
            "lab_order_category": pl.String,
            "lab_name": pl.String,
            "lab_category": pl.String,
            "lab_value": pl.String,
            "lab_value_numeric": pl.Float64,
            "reference_unit": pl.String,
            "lab_specimen_name": pl.String,
            "lab_specimen_category": pl.String,
            "lab_loinc_code": pl.String,
        },
    )
