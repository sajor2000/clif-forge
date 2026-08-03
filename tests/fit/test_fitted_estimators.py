"""Unit tests for reference prior-table estimators (all-28 realism)."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from clifforge.fit import estimators
from clifforge.fit.param_pack import ParamPack, scan_for_leakage


def test_fit_code_status_rates_gated() -> None:
    rows = []
    for i in range(40):
        rows.append({"patient_id": f"P{i}", "code_status_category": "Full"})
        if i < 25:
            rows.append({"patient_id": f"P{i}", "code_status_category": "DNR/DNI"})
        if i < 12:
            rows.append({"patient_id": f"P{i}", "code_status_category": "AND"})
    cs = pl.DataFrame(rows)
    hosp = pl.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(40)],
            "discharge_category": ["Expired"] * 20 + ["Home"] * 20,
        }
    )
    params, _audit = estimators.fit_code_status_rates(cs, hosp)
    assert "dnr_prob_expired" in params
    assert 0.0 <= float(params["dnr_prob_expired"]) <= 1.0


def test_fit_stay_prevalence_and_top_k() -> None:
    df = pl.DataFrame(
        {
            "hospitalization_id": [f"H{i % 30}" for i in range(100)],
            "med_category": ["insulin"] * 50 + ["vancomycin"] * 30 + ["rare"] * 20,
        }
    )
    prev, _audit = estimators.fit_stay_prevalence(df, n_hospitalizations=50)
    assert "stay_prevalence" in prev
    assert 0.0 < float(prev["stay_prevalence"]) <= 1.0
    top, _ = estimators.fit_top_k_category(df, "med_category", k=2)
    assert "med_category_marginal" in top
    assert len(top["med_category_marginal"]) <= 2


def test_fit_prone_and_cultures() -> None:
    pos = pl.DataFrame(
        {
            "hospitalization_id": [f"H{i}" for i in range(40)] * 5,
            "position_category": ["not_prone"] * 190 + ["prone"] * 10,
        }
    )
    rs = pl.DataFrame(
        {
            "hospitalization_id": [f"H{i}" for i in range(40)],
            "device_category": ["IMV"] * 40,
        }
    )
    params, _ = estimators.fit_prone_rates(pos, rs)
    assert "prone_prob_otherwise" in params

    cult = pl.DataFrame({"hospitalization_id": [f"H{i}" for i in range(50)]})
    start = datetime(2020, 1, 1)
    adt = pl.DataFrame(
        {
            "location_category": ["icu"] * 50,
            "in_dttm": [start] * 50,
            "out_dttm": [start + timedelta(days=2)] * 50,
        }
    )
    rate, _ = estimators.fit_cultures_per_icu_day(cult, adt)
    assert "cultures_per_icu_day" in rate
    assert float(rate["cultures_per_icu_day"]) > 0


def test_fit_adt_arrival_and_route() -> None:
    from datetime import datetime, timedelta

    start = datetime(2020, 1, 1)
    rows = []
    for i in range(40):
        hid = f"H{i}"
        # First location: mostly ED, some direct ICU.
        first = "icu" if i < 8 else "ed"
        rows.append(
            {
                "hospitalization_id": hid,
                "location_category": first,
                "in_dttm": start,
                "out_dttm": start + timedelta(hours=6),
            }
        )
        rows.append(
            {
                "hospitalization_id": hid,
                "location_category": "icu",
                "in_dttm": start + timedelta(hours=6),
                "out_dttm": start + timedelta(hours=30),
            }
        )
    adt = pl.DataFrame(rows)
    params, _ = estimators.fit_adt_arrival(adt)
    assert "arrival_location_marginal" in params
    assert "direct_icu_frac" in params
    assert abs(float(params["direct_icu_frac"]) - 0.2) < 0.05

    hosp = pl.DataFrame(
        {
            "admission_type_category": ["ed"] * 30 + ["elective"] * 10,
        }
    )
    route, _ = estimators.fit_admission_route_marginal(hosp)
    assert "admission_route_marginal" in route
    assert "ed" in route["admission_route_marginal"]


def test_new_estimator_params_pass_leakage_scan() -> None:
    pack = ParamPack(
        manifest={"pack_version": "1.0", "tables": {}},
        tables={
            "code_status": {
                "n_records": 100,
                "fitted": True,
                "params": {
                    "dnr_prob_expired": 0.4,
                    "comfort_prob_expired": 0.2,
                    "dnr_prob_survivor": 0.05,
                    "code_status_category_marginal": {"Full": 0.8, "DNR/DNI": 0.2},
                },
            },
            "hospital_diagnosis": {
                "n_records": 1000,
                "fitted": True,
                "params": {
                    "diagnosis_code_marginal": {f"A{i:02d}.9": 1.0 / 40 for i in range(40)}
                },
            },
            "adt": {
                "n_records": 100,
                "fitted": True,
                "params": {
                    "arrival_location_marginal": {"ed": 0.7, "icu": 0.2, "ward": 0.1},
                    "direct_icu_frac": 0.11,
                    "enrich_locations": True,
                },
            },
        },
    )
    assert scan_for_leakage(pack) == []
