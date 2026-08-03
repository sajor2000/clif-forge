"""Tier 5 ``microbiology_culture`` generator (U17; R5, KTD-6).

Cultures are **sparse** events (a handful per stay at most). No fitted block
exists, so a documented per-ICU-day culture rate (un-fitted, like the adt
constants) draws a small Poisson count per stay; each culture picks a fluid,
method, and organism from documented marginals — heavily weighted toward
``no_growth``, matching real ICU microbiology. Timestamps respect the clinical
order ``order_dttm <= collect_dttm < result_dttm`` (result lands after a
turnaround).

Organisms are drawn as an ``(organism_category, organism_group)`` **pair**, not
independently. The two mCIDE lists are separate files with no crosswalk between
them (543 species, 108 groups), so sampling them separately would routinely emit
a species filed under the wrong group — each value individually mCIDE-valid, the
row clinically nonsense. The pairs below are a documented prior recorded in
``PROVENANCE.md``; both members of every pair are exact mCIDE members (R5).

``organism_id`` is the canonical join key to ``microbiology_susceptibility``: the
DDL defines it as identifying "a unique, non-missing ``organism_category``" that
links a culture to its sensitivities. A culture that grew nothing therefore gets a
**null** ``organism_id`` and a null ``organism_category`` — there is no isolate to
identify and nothing to test — which is exactly why the schema must not treat
every ``*_id`` as non-nullable.

The spine supplies only the stay horizon (KTD-6). Output is reproducible
byte-for-byte under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, grid_step_hours
from clifforge.generate.loinc import micro_loinc_code
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame

__all__ = [
    "NO_GROWTH",
    "CultureEvent",
    "microbiology_culture_frame",
    "sample_microbiology_culture",
]

#: Expected cultures per ICU day (documented sparsity constant, un-fitted).
_CULTURES_PER_DAY = 0.4

_FLUID_MARGINAL = {
    "blood_buffy": 0.5,
    "respiratory_tract_lower": 0.2,
    "genito_urinary_tract": 0.2,
    "skin_unspecified": 0.1,
}
_METHOD_MARGINAL = {"culture": 0.8, "gram_stain": 0.2}

#: The mCIDE value meaning "nothing grew", in both the category and group lists.
NO_GROWTH = "no_growth"

#: ``organism_category -> organism_group``, for the isolates this generator emits.
#: Every pair is a real species filed under its real mCIDE group.
_ORGANISM_GROUP: dict[str, str] = {
    "staphylococcus_aureus": "staphylococcus_coag_pos",
    "escherichia_coli": "escherichia",
    "klebsiella_pneumoniae": "klebsiella",
    "pseudomonas_aeruginosa": "pseudomonas_wo_cepacia_maltophilia",
    "enterococcus_faecalis": "enterococcus",
}

#: Heavily weighted to no growth, matching realistic ICU culture yield.
_ORGANISM_MARGINAL = {
    NO_GROWTH: 0.65,
    "staphylococcus_aureus": 0.10,
    "escherichia_coli": 0.10,
    "klebsiella_pneumoniae": 0.08,
    "pseudomonas_aeruginosa": 0.05,
    "enterococcus_faecalis": 0.02,
}

_DEFAULT_ADMIT = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CultureEvent:
    """One microbiology culture (order/collect/result + categories).

    ``organism_category`` and ``organism_id`` are ``None`` when nothing grew.
    """

    patient_id: str
    hospitalization_id: str
    organism_id: str | None
    order_dttm: datetime
    collect_dttm: datetime
    result_dttm: datetime
    fluid_category: str
    method_category: str
    organism_category: str | None
    organism_group: str

    @property
    def grew_organism(self) -> bool:
        """True when the culture yielded an isolate, i.e. it can be sensitivity-tested."""
        return self.organism_category is not None


def sample_microbiology_culture(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    patient_id: str | None = None,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = _DEFAULT_ADMIT,
) -> list[CultureEvent]:
    """Emit one hospitalization's sparse culture events (R5, R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    pid = patient_id if patient_id is not None else hid
    los_hours = spine.n_intervals * grid_step_hours(pack)
    if los_hours <= 0:
        return []

    block = pack.tables.get("microbiology_culture", {})
    params = block.get("params", {}) if isinstance(block, dict) else {}
    rate = float(params.get("cultures_per_icu_day", _CULTURES_PER_DAY))
    fluid_m = params.get("fluid_category_marginal")
    method_m = params.get("method_category_marginal")
    organism_m = params.get("organism_category_marginal")
    fluid_marginal = fluid_m if isinstance(fluid_m, dict) and fluid_m else _FLUID_MARGINAL
    method_marginal = method_m if isinstance(method_m, dict) and method_m else _METHOD_MARGINAL
    organism_marginal = (
        organism_m if isinstance(organism_m, dict) and organism_m else _ORGANISM_MARGINAL
    )

    n_cultures = int(rng.poisson(rate * los_hours / 24.0))
    events: list[CultureEvent] = []
    for i in range(n_cultures):
        order = admit_dttm + timedelta(hours=float(rng.random()) * los_hours)
        collect = order + timedelta(minutes=float(rng.uniform(5.0, 60.0)))
        result = collect + timedelta(hours=float(rng.uniform(24.0, 72.0)))
        organism = categorical(organism_marginal, rng)
        grew = organism != NO_GROWTH and organism is not None
        if grew and organism not in _ORGANISM_GROUP:
            # Fitted isolate without a prior group map — still emit category.
            group = organism
        elif grew:
            group = _ORGANISM_GROUP[organism]
        else:
            group = NO_GROWTH
        events.append(
            CultureEvent(
                patient_id=pid,
                hospitalization_id=hid,
                organism_id=f"{hid}-ORG{i}" if grew else None,
                order_dttm=order,
                collect_dttm=collect,
                result_dttm=result,
                fluid_category=categorical(fluid_marginal, rng),
                method_category=categorical(method_marginal, rng),
                organism_category=organism if grew else None,
                organism_group=group,
            )
        )
    events.sort(key=lambda e: e.order_dttm)
    return events


def microbiology_culture_frame(events: list[CultureEvent]) -> pl.DataFrame:
    """Stack culture events into one conformant frame."""
    return pl.DataFrame(
        {
            "patient_id": [e.patient_id for e in events],
            "hospitalization_id": [e.hospitalization_id for e in events],
            "organism_id": [e.organism_id for e in events],
            "order_dttm": [e.order_dttm for e in events],
            "collect_dttm": [e.collect_dttm for e in events],
            "result_dttm": [e.result_dttm for e in events],
            "fluid_name": [e.fluid_category for e in events],
            "fluid_category": [e.fluid_category for e in events],
            "method_name": [e.method_category for e in events],
            "method_category": [e.method_category for e in events],
            "organism_name": [e.organism_category for e in events],
            "organism_category": [e.organism_category for e in events],
            "organism_group": [e.organism_group for e in events],
            "lab_loinc_code": [micro_loinc_code(e.fluid_category) for e in events],
        },
        schema={
            "patient_id": pl.String,
            "hospitalization_id": pl.String,
            "organism_id": pl.String,
            "order_dttm": UTC_DATETIME,
            "collect_dttm": UTC_DATETIME,
            "result_dttm": UTC_DATETIME,
            "fluid_name": pl.String,
            "fluid_category": pl.String,
            "method_name": pl.String,
            "method_category": pl.String,
            "organism_name": pl.String,
            "organism_category": pl.String,
            "organism_group": pl.String,
            "lab_loinc_code": pl.String,
        },
    )
