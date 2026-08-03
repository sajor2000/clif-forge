"""``clinical_trial`` generator (prior-driven; R14, KTD-6).

Trial enrolment is rare and selective: only a small fraction of ICU stays are
enrolled in an interventional trial, and enrolment is not random with respect to
acuity — a critical-care trial recruits the patients who meet its severity
criteria. So enrolment here is gated on the spine reaching mechanical
ventilation, and the two documented trial archetypes recruit accordingly.

The event sequence is the part worth getting right, because it is what makes the
table analysable:

* ``consent_dttm`` precedes ``randomized_dttm`` — you cannot randomize a patient
  who has not consented.
* Both land inside the enrolment window near admission, since ICU trials
  randomize early.
* ``withdrawal_dttm`` is **null for the great majority**, because most enrolled
  patients do not withdraw. Populating it for everyone would turn an exception
  into the norm; leaving it null is the honest encoding of "did not withdraw".

Trial names, ids, and arms are documented synthetic placeholders (R15, recorded
in ``PROVENANCE.md``) — deliberately not real NCT numbers, which would imply this
synthetic cohort was enrolled in a real study. Reproducible under a fixed ``rng``
(R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours, pack_table_params
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame
from clifforge.reference.dashboard_priors import absent_table_rates as _DASH_RATES

__all__ = ["ClinicalTrialRow", "clinical_trial_frame", "sample_clinical_trial"]

#: Trials recruit ventilated patients (support level 3+ on the ladder).
_MIN_SUPPORT_LEVEL = 3
#: Fraction of eligible stays enrolled (dashboard-prior).
_ENROLMENT_PROB = _DASH_RATES["clinical_trial"]

#: Synthetic trial registry: id -> (name, arms). Not real NCT identifiers.
_TRIALS: dict[str, tuple[str, tuple[str, ...]]] = {
    "SYN-ARDS-001": ("Synthetic Lung Protective Ventilation Trial", ("intervention", "control")),
    "SYN-SEPSIS-002": ("Synthetic Early Vasopressor Strategy Trial", ("intervention", "control")),
}
_TRIAL_MARGINAL = {"SYN-ARDS-001": 0.6, "SYN-SEPSIS-002": 0.4}

#: ICU trials randomize early; consent precedes randomization by minutes to hours.
_CONSENT_WINDOW_HOURS = 24.0
_CONSENT_TO_RANDOM_HOURS = (0.5, 6.0)

#: Most enrolled patients complete the trial.
_WITHDRAWAL_PROB = 0.08

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class ClinicalTrialRow:
    """One trial enrolment. ``withdrawal_dttm`` is None unless the patient withdrew."""

    hospitalization_id: str
    trial_id: str
    trial_name: str
    arm_id: str
    consent_dttm: datetime
    randomized_dttm: datetime
    withdrawal_dttm: datetime | None


def sample_clinical_trial(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[ClinicalTrialRow]:
    """Enrol a small, acuity-selected fraction of stays in a synthetic trial (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    los_hours = spine.n_intervals * grid_step_hours(pack)
    if los_hours <= 0 or spine.peak_level < _MIN_SUPPORT_LEVEL:
        return []
    params = pack_table_params(pack, "clinical_trial")
    # Peak gate above defines eligibility. Enrolment must be among-eligible:
    # prefer fitted eligible_conditional_prevalence; never apply unconditional
    # stay_prevalence here (that would double-gate). Dashboard prior is already
    # among ventilated by convention.
    enrol = float(params.get("eligible_conditional_prevalence", _ENROLMENT_PROB))
    if rng.random() >= enrol:
        return []

    # Always use synthetic trial IDs — never emit real NCT strings from a fit.
    trial_marginal = _TRIAL_MARGINAL
    trial_id = categorical(trial_marginal, rng)
    name, arms = _TRIALS[trial_id]
    withdrawal_prob = float(params.get("withdrawal_prob", _WITHDRAWAL_PROB))

    consent = admit_dttm + timedelta(
        hours=float(rng.random()) * min(los_hours, _CONSENT_WINDOW_HOURS)
    )
    randomized = consent + timedelta(hours=float(rng.uniform(*_CONSENT_TO_RANDOM_HOURS)))
    withdrawal = None
    if rng.random() < withdrawal_prob:
        # Withdrawal happens after randomization, within the remaining stay.
        remaining = max(1.0, los_hours - (randomized - admit_dttm).total_seconds() / 3600.0)
        withdrawal = randomized + timedelta(hours=float(rng.random()) * remaining)

    return [
        ClinicalTrialRow(
            hospitalization_id=hid,
            trial_id=trial_id,
            trial_name=name,
            arm_id=str(rng.choice(arms)),
            consent_dttm=consent,
            randomized_dttm=randomized,
            withdrawal_dttm=withdrawal,
        )
    ]


def clinical_trial_frame(rows: list[ClinicalTrialRow]) -> pl.DataFrame:
    """Stack trial enrolments into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "trial_id": [r.trial_id for r in rows],
            "trial_name": [r.trial_name for r in rows],
            "arm_id": [r.arm_id for r in rows],
            "consent_dttm": [r.consent_dttm for r in rows],
            "randomized_dttm": [r.randomized_dttm for r in rows],
            "withdrawal_dttm": [r.withdrawal_dttm for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "trial_id": pl.String,
            "trial_name": pl.String,
            "arm_id": pl.String,
            "consent_dttm": UTC_DATETIME,
            "randomized_dttm": UTC_DATETIME,
            "withdrawal_dttm": UTC_DATETIME,
        },
    )
