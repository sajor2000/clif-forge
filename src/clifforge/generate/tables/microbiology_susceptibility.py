"""``microbiology_susceptibility`` generator (derived, prior-driven; R5, R14).

Sensitivities are not an independent event stream — they are a property of an
isolate that a specific culture grew. This generator therefore reads the same
encounter's :class:`~clifforge.generate.tables.microbiology_culture.CultureEvent`
rows and emits a panel only for cultures that actually grew something, reusing
the parent's ``organism_id``. That id is the canonical join key: CLIF 2.1 defines
it as linking "a unique, non-missing ``organism_category``" in
``microbiology_culture`` to an antimicrobial and susceptibility here. A culture
that grew nothing has no isolate, so it contributes no rows at all.

**The panel depends on the organism, because in reality it does.** A lab does not
run the same antibiotics against E. coli and S. aureus, so each organism gets its
own documented panel (gram-negatives get the beta-lactam/aminoglycoside/quinolone
set, gram-positives the anti-staphylococcal set). Every antimicrobial below is an
exact mCIDE ``antimicrobial_category`` member, as is every emitted
``susceptibility_category`` (R5).

Resistance rates are documented per-organism priors (un-fitted, recorded in
``PROVENANCE.md``) rather than a single global rate — MRSA-style oxacillin
resistance and carbapenem resistance differ by an order of magnitude, and
flattening them would erase the main thing an antibiogram is used to study.

This table has no timestamp of its own in CLIF 2.1; it is reached through the
culture it hangs off. Reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.tables.microbiology_culture import CultureEvent

__all__ = [
    "SusceptibilityRow",
    "microbiology_susceptibility_frame",
    "sample_microbiology_susceptibility",
]

_PARENT = "microbiology_culture"

#: organism_category -> the antimicrobials a lab actually panels it against.
#: Every value is an exact mCIDE ``antimicrobial_category`` member.
_PANEL: dict[str, tuple[str, ...]] = {
    "staphylococcus_aureus": ("oxacillin", "vancomycin", "clindamycin", "linezolid", "daptomycin"),
    "escherichia_coli": (
        "ampicillin",
        "ceftriaxone",
        "cefepime",
        "meropenem",
        "ciprofloxacin",
        "gentamicin",
        "trimethoprim_sulfamethoxazole",
    ),
    "klebsiella_pneumoniae": (
        "ceftriaxone",
        "cefepime",
        "meropenem",
        "ciprofloxacin",
        "gentamicin",
        "trimethoprim_sulfamethoxazole",
    ),
    "pseudomonas_aeruginosa": (
        "cefepime",
        "ceftazidime",
        "piperacillin_tazobactam",
        "meropenem",
        "ciprofloxacin",
        "tobramycin",
    ),
    "enterococcus_faecalis": ("ampicillin", "vancomycin", "linezolid", "daptomycin"),
}

#: P(non-susceptible), per (organism, antimicrobial). Documented priors: the
#: default is the organism's baseline and the entries override the drugs whose
#: resistance is clinically distinctive (oxacillin -> MRSA, carbapenems -> CRE).
_BASELINE_RESISTANCE: dict[str, float] = {
    "staphylococcus_aureus": 0.10,
    "escherichia_coli": 0.15,
    "klebsiella_pneumoniae": 0.20,
    "pseudomonas_aeruginosa": 0.20,
    "enterococcus_faecalis": 0.10,
}
_RESISTANCE_OVERRIDE: dict[tuple[str, str], float] = {
    ("staphylococcus_aureus", "oxacillin"): 0.45,  # MRSA prevalence
    ("staphylococcus_aureus", "vancomycin"): 0.01,  # VRSA is vanishingly rare
    ("escherichia_coli", "ampicillin"): 0.55,
    ("escherichia_coli", "meropenem"): 0.02,
    ("klebsiella_pneumoniae", "meropenem"): 0.05,
    ("pseudomonas_aeruginosa", "meropenem"): 0.15,
    ("enterococcus_faecalis", "vancomycin"): 0.08,  # VRE
}

#: A small share of results are technically uninterpretable rather than a clean
#: susceptible/resistant call.
_INDETERMINATE_PROB = 0.02

_SUSCEPTIBLE = "susceptible"
_NON_SUSCEPTIBLE = "non_susceptible"
_INDETERMINATE = "indeterminate"

#: Reported interpretation strings that pair with each mCIDE category.
_INTERPRETATION = {
    _SUSCEPTIBLE: "Susceptible",
    _NON_SUSCEPTIBLE: "Resistant",
    _INDETERMINATE: "Indeterminate",
}

#: Minimum-inhibitory-concentration ladders (mcg/mL) reported as ``sensitivity_name``.
_MIC_SUSCEPTIBLE = ("<=0.25", "<=0.5", "<=1", "2")
_MIC_RESISTANT = (">=8", ">=16", ">=32", ">=64")

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class SusceptibilityRow:
    """One antimicrobial tested against one isolate."""

    organism_id: str
    antimicrobial_category: str
    sensitivity_name: str
    susceptibility_category: str


def _resistance_prob(organism: str, antimicrobial: str) -> float:
    return _RESISTANCE_OVERRIDE.get(
        (organism, antimicrobial), _BASELINE_RESISTANCE.get(organism, 0.15)
    )


def sample_microbiology_susceptibility(
    parents: dict[str, list[Any]],
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    patient_id: str | None = None,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[SusceptibilityRow]:
    """Emit sensitivity panels for every isolate this encounter's cultures grew (R5, R22).

    ``parents`` carries the same encounter's ``microbiology_culture`` events. The
    unused id/time kwargs keep the derived-sampler signature uniform: this table
    is keyed only on ``organism_id``, which the parent already minted.
    """
    cultures: list[CultureEvent] = parents[_PARENT]
    rows: list[SusceptibilityRow] = []
    for culture in cultures:
        organism = culture.organism_category
        if organism is None or culture.organism_id is None:
            continue  # nothing grew, so there is nothing to test
        for antimicrobial in _PANEL.get(organism, ()):
            if rng.random() < _INDETERMINATE_PROB:
                category, mic = _INDETERMINATE, "n/a"
            elif rng.random() < _resistance_prob(organism, antimicrobial):
                category = _NON_SUSCEPTIBLE
                mic = str(rng.choice(_MIC_RESISTANT))
            else:
                category = _SUSCEPTIBLE
                mic = str(rng.choice(_MIC_SUSCEPTIBLE))
            rows.append(SusceptibilityRow(culture.organism_id, antimicrobial, mic, category))
    return rows


def microbiology_susceptibility_frame(rows: list[SusceptibilityRow]) -> pl.DataFrame:
    """Stack susceptibility results into one conformant frame."""
    return pl.DataFrame(
        {
            "organism_id": [r.organism_id for r in rows],
            "antimicrobial_name": [r.antimicrobial_category for r in rows],
            "antimicrobial_category": [r.antimicrobial_category for r in rows],
            "sensitivity_name": [r.sensitivity_name for r in rows],
            "susceptibility_name": [_INTERPRETATION[r.susceptibility_category] for r in rows],
            "susceptibility_category": [r.susceptibility_category for r in rows],
        },
        schema={
            "organism_id": pl.String,
            "antimicrobial_name": pl.String,
            "antimicrobial_category": pl.String,
            "sensitivity_name": pl.String,
            "susceptibility_name": pl.String,
            "susceptibility_category": pl.String,
        },
    )
