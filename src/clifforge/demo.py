"""A built-in, no-real-data demo parameter pack for out-of-the-box generation.

The realistic parameter packs are fitted over credentialed real CLIF data and are
not redistributable, so a fresh clone has nothing to generate from. This module
ships a **complete, hand-specified synthetic pack** — every fitted block the
generators read, populated with plausible constants and no real data — so anyone
can run ``clif-forge generate --demo`` immediately to produce CLIF 2.1-conformant
output for pipeline testing, CI fixtures, and agent/tool development.

The demo pack is deliberately *structural*, not statistically calibrated: it
exercises every table and coupling and passes the conformance gate, but its
distributions are illustrative. Realistic, network-representative calibration
requires fitting to real CLIF statistics (the ``fit`` stage) and is a separate,
data-governed step.
"""

from __future__ import annotations

from clifforge.fit.param_pack import ParamPack

__all__ = ["demo_pack"]


def demo_pack() -> ParamPack:
    """A minimal but complete parameter pack: every fitted block the generators
    read, with no real data. Used by ``clif-forge generate --demo`` and the test
    suite so the two never drift apart."""

    def six(mean: float) -> dict[str, dict[str, float]]:
        return {str(s): {"mean": mean - 3 * s, "phi": 0.4, "sigma": 3.0} for s in range(6)}

    return ParamPack(
        manifest={
            "clif_version": "2.1.0",
            "pack_version": "1.0",
            "fit_source": {"dataset_id": "synthetic-demo", "commit": "none"},
            "suppression_audit": [],
        },
        tables={
            "patient": {
                "params": {
                    "race_category_marginal": {"White": 0.6, "Unknown": 0.4},
                    "ethnicity_category_marginal": {"Non-Hispanic": 0.7, "Unknown": 0.3},
                    "sex_category_marginal": {"Female": 0.5, "Male": 0.5},
                }
            },
            "hospitalization": {
                "params": {
                    "admission_type_category_marginal": {"ed": 0.8, "direct": 0.2},
                    "discharge_category_marginal": {"Home": 0.7, "Expired": 0.3},
                    # Illustrative adult age deciles so demo/coverage emits
                    # age_at_admission the same way fitted packs do.
                    "age_at_admission_quantiles": [
                        25.0,
                        35.0,
                        45.0,
                        55.0,
                        62.0,
                        68.0,
                        74.0,
                        80.0,
                        86.0,
                        92.0,
                        100.0,
                    ],
                }
            },
            "vitals": {
                "params": {
                    f"{v}_ar1_by_state": six(base)
                    for v, base in {
                        "heart_rate": 90,
                        "sbp": 130,
                        "dbp": 75,
                        "map": 90,
                        "respiratory_rate": 20,
                        "spo2": 100,
                        "temp_c": 38,
                    }.items()
                }
            },
            "labs": {
                "params": {
                    "lab_order": ["creatinine", "bun", "sodium"],
                    "lab_correlation": [[1.0, 0.6, 0.0], [0.6, 1.0, -0.3], [0.0, -0.3, 1.0]],
                    "lab_marginals": {
                        "creatinine": {"log_mean": 0.79, "log_sd": 0.4},
                        "bun": {"log_mean": 2.8, "log_sd": 0.5},
                        "sodium": {"log_mean": 4.95, "log_sd": 0.03},
                    },
                    "lab_presence": {"creatinine": 0.8, "bun": 0.6, "sodium": 0.9},
                }
            },
            "medication_admin_continuous": {
                "params": {
                    "infusion_hazards": {
                        "norepinephrine": {"mean_run_intervals": 2.0, "stop_hazard": 0.4},
                        "propofol": {"mean_run_intervals": 2.0, "stop_hazard": 0.4},
                    }
                }
            },
            "spine": {
                "params": {
                    "state_model": {"grid_step_hours": 1.0, "horizon_intervals": 72},
                    "support_level_states": [0, 1, 2, 3, 4, 5],
                    "support_level_start_dist": {
                        "0": 0.3,
                        "1": 0.2,
                        "2": 0.2,
                        "3": 0.15,
                        "4": 0.1,
                        "5": 0.05,
                    },
                    "support_level_transition_matrix": {
                        "0": {"1": 0.2, "3": 0.3, "discharge": 0.5},
                        "1": {"2": 0.3, "0": 0.2, "discharge": 0.5},
                        "2": {"3": 0.4, "1": 0.2, "discharge": 0.4},
                        "3": {"4": 0.3, "2": 0.3, "discharge": 0.4},
                        "4": {"5": 0.3, "3": 0.3, "discharge": 0.4},
                        "5": {"4": 0.4, "discharge": 0.6},
                    },
                    "support_level_sojourn": {
                        # Short sojourns at low acuity; long enough at L4/L5 that the
                        # CRRT 12h sustained-renal gate fires in a typical demo run.
                        **{
                            str(s): {
                                "family": "empirical_mean",
                                "params": [2.0],
                                "mean_hours": 2.0,
                            }
                            for s in range(4)
                        },
                        "4": {
                            "family": "empirical_mean",
                            "params": [16.0],
                            "mean_hours": 16.0,
                        },
                        "5": {
                            "family": "empirical_mean",
                            "params": [20.0],
                            "mean_hours": 20.0,
                        },
                    },
                    "flag_prevalence_by_level": {
                        "0": {
                            "resp_flag": 0.0,
                            "cv_flag": 0.0,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.0,
                        },
                        "1": {
                            "resp_flag": 0.1,
                            "cv_flag": 0.0,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.1,
                        },
                        "2": {
                            "resp_flag": 0.4,
                            "cv_flag": 0.1,
                            "renal_flag": 0.0,
                            "neuro_flag": 0.2,
                        },
                        "3": {
                            "resp_flag": 0.7,
                            "cv_flag": 0.2,
                            "renal_flag": 0.2,
                            "neuro_flag": 0.3,
                        },
                        "4": {
                            "resp_flag": 0.7,
                            "cv_flag": 0.8,
                            "renal_flag": 0.7,
                            "neuro_flag": 0.3,
                        },
                        "5": {
                            "resp_flag": 0.8,
                            "cv_flag": 0.9,
                            "renal_flag": 1.0,
                            "neuro_flag": 0.4,
                        },
                    },
                    "outcome_marginal": {"alive": 0.8, "expired": 0.2},
                    "expired_rate_by_peak_level": {
                        str(s): {"expired_rate": 0.05 + 0.09 * s} for s in range(6)
                    },
                }
            },
            # Structural CRRT block so demo audits exercise the mCIDE mode column
            # (mode values are exact consortium members; rates stay prior-driven).
            "crrt_therapy": {
                "params": {
                    "crrt_prob": 1.0,
                    "crrt_mode_category_marginal": {
                        "cvvhdf": 0.6,
                        "cvvhd": 0.25,
                        "cvvh": 0.15,
                    },
                }
            },
        },
    )
