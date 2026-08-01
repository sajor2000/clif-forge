"""Unit tests for the Tier 5 microbiology_culture generator (U17, R5/R22)."""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import microbiology_culture as mc
from clifforge.generate.tables.microbiology_culture import (
    microbiology_culture_frame,
    sample_microbiology_culture,
)
from clifforge.reference import categories


def _pack(grid_step_hours: float = 1.0) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={"spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}}},
    )


def _spine(n: int, hid: str = "H0") -> SpineFrame:
    return SpineFrame(
        hospitalization_id=hid,
        support_level=[3] * n,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[False] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def _cohort(pack: ParamPack, rng: np.random.Generator, n_stays: int, n_int: int = 120) -> list:
    events = []
    for i in range(n_stays):
        events += sample_microbiology_culture(_spine(n_int, hid=f"H{i}"), pack, rng)
    return events


def test_is_deterministic() -> None:
    pack = _pack()
    sp = _spine(120)
    a = sample_microbiology_culture(sp, pack, np.random.default_rng(0))
    b = sample_microbiology_culture(sp, pack, np.random.default_rng(0))
    assert a == b


def test_result_after_collect_after_order() -> None:
    pack = _pack()
    for e in _cohort(pack, np.random.default_rng(1), 60):
        assert e.order_dttm <= e.collect_dttm < e.result_dttm


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    fluid_ok = set(categories("microbiology_culture", "fluid_category"))
    method_ok = set(categories("microbiology_culture", "method_category"))
    group_ok = set(categories("microbiology_culture", "organism_group"))
    organism_ok = set(categories("microbiology_culture", "organism_category"))
    events = _cohort(pack, np.random.default_rng(2), 60)
    assert events
    for e in events:
        assert e.fluid_category in fluid_ok
        assert e.method_category in method_ok
        assert e.organism_group in group_ok
        if e.organism_category is not None:
            assert e.organism_category in organism_ok


def test_organism_species_and_group_are_drawn_as_a_matched_pair() -> None:
    """A species must be filed under its own group, not an unrelated one.

    The two mCIDE lists have no crosswalk, so sampling them independently would
    yield rows that are individually valid and jointly nonsense (E. coli reported
    under the staphylococcus group).
    """
    pack = _pack()
    for e in _cohort(pack, np.random.default_rng(6), 200):
        if e.organism_category is None:
            assert e.organism_group == mc.NO_GROWTH
        else:
            assert e.organism_group == mc._ORGANISM_GROUP[e.organism_category]


def test_organism_id_is_present_exactly_when_something_grew() -> None:
    """``organism_id`` links a culture to its sensitivities; no isolate, no id."""
    pack = _pack()
    events = _cohort(pack, np.random.default_rng(7), 200)
    grew = [e for e in events if e.grew_organism]
    assert grew and len(grew) < len(events)  # both branches exercised
    for e in events:
        assert (e.organism_id is not None) == e.grew_organism
    assert len({e.organism_id for e in grew}) == len(grew)  # unique per isolate


def test_patient_id_is_carried_alongside_hospitalization_id() -> None:
    pack = _pack()
    events = sample_microbiology_culture(
        _spine(240), pack, np.random.default_rng(8), patient_id="P7", hospitalization_id="H7"
    )
    assert events
    assert all(e.patient_id == "P7" and e.hospitalization_id == "H7" for e in events)


def test_cultures_are_sparse() -> None:
    pack = _pack()
    rng = np.random.default_rng(3)
    n_stays = 500
    total = sum(
        len(sample_microbiology_culture(_spine(120, hid=f"H{i}"), pack, rng))
        for i in range(n_stays)
    )
    assert total / n_stays < 2.0  # sparse: ~0.4/day over a 5-day stay


def test_no_growth_is_the_dominant_result() -> None:
    pack = _pack()
    events = _cohort(pack, np.random.default_rng(4), 400)
    no_growth = sum(e.organism_group == mc.NO_GROWTH for e in events)
    assert no_growth / len(events) > 0.5  # realistic low yield


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    frame = microbiology_culture_frame(_cohort(pack, np.random.default_rng(5), 80))
    for col in ("order_dttm", "collect_dttm", "result_dttm"):
        dtype = frame.schema[col]
        assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    assert gate.validate(frame, "microbiology_culture", run_secondary=False).pandera_passed


def test_module_exports() -> None:
    assert set(mc.__all__) == {
        "NO_GROWTH",
        "CultureEvent",
        "microbiology_culture_frame",
        "sample_microbiology_culture",
    }
