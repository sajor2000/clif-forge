"""Full CLIF 2.1.0 mCIDE / bounds / FK audit over a generated dataset.

Proves the README claim that every emitted category cell is an exact,
case-sensitive member of the vendored mCIDE list (R5), that companion rollups
agree with ``loader.crosswalk``, that numeric outliers stay in bounds (R9), and
that derived-table join keys resolve without orphans (R8).
"""

from __future__ import annotations

from typing import Any

import pytest

from clifforge.demo import demo_pack
from clifforge.generate.orchestrator import generate_dataset
from clifforge.reference import bounds, categories, loader, resolve_mcide_column

#: (table, category_field, companion_column) triples whose row-level rollup must
#: agree with the vendored mCIDE companion column via ``loader.crosswalk``.
_CROSSWALKS: tuple[tuple[str, str, str], ...] = (
    ("labs", "lab_category", "reference_unit"),
    ("labs", "lab_category", "lab_order_category"),
    ("medication_admin_continuous", "med_category", "med_group"),
    ("medication_admin_continuous", "mar_action_category", "mar_action_group"),
    ("medication_admin_intermittent", "med_category", "med_group"),
    ("medication_admin_intermittent", "mar_action_category", "mar_action_group"),
    ("medication_orders", "med_category", "med_group"),
    ("microbiology_nonculture", "organism_category", "organism_group"),
    ("patient_assessments", "assessment_category", "assessment_group"),
)

#: Tables whose numeric columns are keyed in the outlier CSVs by column name
#: (not by a category value like vitals/labs).
_COLUMN_BOUNDED_TABLES = frozenset({"respiratory_support", "crrt_therapy"})


@pytest.fixture(scope="module")
def dataset() -> dict[str, Any]:
    return generate_dataset(demo_pack(), n_patients=40, seed=7).tables


def _mcide_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for table in loader.tables():
        for field in loader.mcide_fields(table):
            pairs.append((table, field))
    return pairs


def test_resolve_mcide_column_handles_hospital_type_skew() -> None:
    """DDL ``hospital_type`` carries mCIDE ``hospital_type_category`` values."""
    assert resolve_mcide_column("hospital_type_category", {"hospital_type"}) == "hospital_type"
    assert resolve_mcide_column("location_category", {"location_category"}) == "location_category"
    assert resolve_mcide_column("hospital_type_category", {"location_category"}) is None


@pytest.mark.parametrize(("table", "field"), _mcide_pairs())
def test_every_emitted_category_is_exact_mcide_member(
    table: str, field: str, dataset: dict[str, Any]
) -> None:
    frame = dataset[table]
    # mCIDE field key may differ from the DDL column (adt.hospital_type_category).
    data_col = resolve_mcide_column(field, frame.columns)
    if data_col is None:
        pytest.skip(f"{table}.{field} not emitted (documented gap or unused)")
    allowed = set(categories(table, field))
    series = frame[data_col].drop_nulls()
    if series.is_empty():
        return
    bad = set(series.cast(str).unique().to_list()) - allowed
    assert not bad, f"{table}.{data_col} ({field}) has non-mCIDE values: {sorted(bad)[:20]}"


@pytest.mark.parametrize(("table", "field", "companion"), _CROSSWALKS)
def test_companion_crosswalk_agrees_rowwise(
    table: str, field: str, companion: str, dataset: dict[str, Any]
) -> None:
    frame = dataset[table]
    if field not in frame.columns or companion not in frame.columns:
        pytest.skip(f"{table} missing {field}/{companion}")
    if table == "medication_orders":
        mapping = {
            **loader.crosswalk("medication_admin_continuous", field, companion),
            **loader.crosswalk("medication_admin_intermittent", field, companion),
        }
    else:
        mapping = loader.crosswalk(table, field, companion)
    mismatches: list[str] = []
    for row in frame.select(field, companion).drop_nulls().iter_rows(named=True):
        key = str(row[field])
        expected = mapping.get(key)
        if not expected:
            continue
        if str(row[companion]) != expected:
            mismatches.append(f"{key}: got {row[companion]!r}, want {expected!r}")
            if len(mismatches) >= 10:
                break
    assert not mismatches, f"{table}.{companion} vs {field}: {mismatches}"


def test_outlier_bounded_columns_stay_in_range(dataset: dict[str, Any]) -> None:
    """Every non-null numeric with vendored column-name bounds is inside them."""
    for table in _COLUMN_BOUNDED_TABLES:
        if table not in dataset:
            continue
        frame = dataset[table]
        for key in loader.outlier_keys(table):
            if key not in frame.columns:
                continue
            lo, hi = bounds(table, key)
            series = frame[key].drop_nulls()
            if series.is_empty():
                continue
            vals = series.to_list()
            oob = [v for v in vals if not (lo <= float(v) <= hi)]
            assert not oob, f"{table}.{key} out of [{lo}, {hi}]: {oob[:5]}"


def test_vitals_and_labs_values_within_category_bounds(dataset: dict[str, Any]) -> None:
    """Vitals/labs bound by the *category* value, not the column name."""
    for table, cat_col, val_col in (
        ("vitals", "vital_category", "vital_value"),
        ("labs", "lab_category", "lab_value_numeric"),
    ):
        frame = dataset[table]
        if cat_col not in frame.columns or val_col not in frame.columns:
            continue
        for row in frame.select(cat_col, val_col).drop_nulls().iter_rows(named=True):
            try:
                lo, hi = bounds(table, str(row[cat_col]))
            except loader.ReferenceDataError:
                continue
            assert lo <= float(row[val_col]) <= hi


def test_no_orphan_hospitalization_ids(dataset: dict[str, Any]) -> None:
    parents = set(dataset["hospitalization"]["hospitalization_id"].to_list())
    for table, frame in dataset.items():
        if table == "hospitalization" or "hospitalization_id" not in frame.columns:
            continue
        orphans = set(frame["hospitalization_id"].drop_nulls().to_list()) - parents
        assert not orphans, f"{table} has orphan hospitalization_id(s): {sorted(orphans)[:10]}"


def test_susceptibility_organisms_resolve_to_culture(dataset: dict[str, Any]) -> None:
    culture_ids = {
        oid
        for oid in dataset["microbiology_culture"]["organism_id"].drop_nulls().to_list()
    }
    sus = dataset["microbiology_susceptibility"]
    if sus.is_empty() or "organism_id" not in sus.columns:
        return
    orphans = set(sus["organism_id"].drop_nulls().to_list()) - culture_ids
    assert not orphans, f"susceptibility orphans: {sorted(orphans)[:10]}"


def test_medication_orders_correspond_to_administrations(dataset: dict[str, Any]) -> None:
    order_ids = set(dataset["medication_orders"]["med_order_id"].drop_nulls().to_list())
    admin_ids: set[object] = set()
    for table in ("medication_admin_continuous", "medication_admin_intermittent"):
        frame = dataset[table]
        if "med_order_id" in frame.columns:
            admin_ids |= set(frame["med_order_id"].drop_nulls().to_list())
    assert admin_ids <= order_ids, (
        f"admin med_order_id without order row: {sorted(admin_ids - order_ids)[:10]}"
    )
    assert order_ids <= admin_ids, (
        f"order without administrations: {sorted(order_ids - admin_ids)[:10]}"
    )
