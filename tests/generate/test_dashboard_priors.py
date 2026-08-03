"""Dashboard-prior wiring for MIMIC-absent tables."""

from __future__ import annotations

from clifforge.reference import dashboard_priors


def test_dashboard_priors_cited() -> None:
    assert "clif-icu.com" in dashboard_priors.DASHBOARD_SOURCE_URL
    assert dashboard_priors.DASHBOARD_RETRIEVED_AT
    assert dashboard_priors.advanced_support["imv"] == 0.306
    assert dashboard_priors.advanced_support["crrt"] == 0.034
    assert dashboard_priors.advanced_support["vasopressor"] == 0.253


def test_absent_generators_use_dashboard_constants() -> None:
    from clifforge.generate.tables import clinical_trial, key_icu_orders, microbiology_nonculture

    assert clinical_trial._ENROLMENT_PROB == dashboard_priors.absent_table_rates["clinical_trial"]
    assert key_icu_orders._REHAB_PROB == dashboard_priors.absent_table_rates["key_icu_orders"]
    assert (
        microbiology_nonculture._PANELS_PER_STAY
        == dashboard_priors.absent_table_rates["microbiology_nonculture"]
    )
