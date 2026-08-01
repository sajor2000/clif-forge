"""Unit tests for the Tier 3 labs generator (U11, R9/R12/KTD-4/R22).

Driven by a small hand-built copula pack plus directly-constructed ``SpineFrame``s
(no real data, no run_fit): generated pairwise correlations recover the pack
copula, values stay within outlier bounds, per-lab presence tracks the fitted
rates with no imputation of absent labs, creatinine rises under the renal-failure
flag, categories are exact mCIDE members, the frame passes the gate, and output
is seed-reproducible.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.spine import SpineFrame
from clifforge.generate.tables import labs
from clifforge.generate.tables.labs import LabObservation, labs_frame, sample_labs
from clifforge.reference import bounds, categories

_GRID = 24.0  # one lab panel per ICU interval, for dense sampling

# Three real mCIDE labs so category/bounds checks are meaningful. creatinine and
# bun are positively correlated (0.6); sodium is ~uncorrelated with creatinine.
_ORDER = ["creatinine", "bun", "sodium"]
_CORR = [
    [1.0, 0.6, 0.0],
    [0.6, 1.0, -0.3],
    [0.0, -0.3, 1.0],
]
_MARGINALS = {
    "creatinine": {"log_mean": 0.79, "log_sd": 0.4},
    "bun": {"log_mean": 2.8, "log_sd": 0.5},
    "sodium": {"log_mean": 4.95, "log_sd": 0.03},
}


def _pack(
    presence: dict[str, float] | None = None,
    correlation: list[list[float]] | None = None,
    grid_step_hours: float = _GRID,
) -> ParamPack:
    return ParamPack(
        manifest={},
        tables={
            "labs": {
                "n_records": 1000,
                "fitted": True,
                "params": {
                    "lab_order": _ORDER,
                    "lab_correlation": correlation or _CORR,
                    "lab_marginals": _MARGINALS,
                    "lab_presence": presence or {"creatinine": 1.0, "bun": 1.0, "sodium": 1.0},
                },
            },
            "spine": {"params": {"state_model": {"grid_step_hours": grid_step_hours}}},
        },
    )


def _spine(levels: list[int], hid: str = "H0", renal: bool = False) -> SpineFrame:
    n = len(levels)
    return SpineFrame(
        hospitalization_id=hid,
        support_level=levels,
        resp_flag=[False] * n,
        cv_flag=[False] * n,
        renal_flag=[renal] * n,
        neuro_flag=[False] * n,
        outcome="alive",
    )


def _spearman(x: list[float], y: list[float]) -> float:
    xr = np.argsort(np.argsort(np.asarray(x)))
    yr = np.argsort(np.argsort(np.asarray(y)))
    return float(np.corrcoef(xr, yr)[0, 1])


def _presence_pack(presence_correlation: list[list[float]] | None) -> ParamPack:
    pack = _pack(presence={"creatinine": 0.5, "bun": 0.5, "sodium": 0.5})
    if presence_correlation is not None:
        pack.tables["labs"]["params"]["lab_presence_correlation"] = presence_correlation
    return pack


def _present_flags(pack: ParamPack, n_stays: int) -> dict[str, list[int]]:
    """Per-stay present/absent (1/0) for each lab across many independent stays."""
    flags: dict[str, list[int]] = {lab: [] for lab in _ORDER}
    for s in range(n_stays):
        obs = sample_labs(_spine([3] * 6, hid=f"H{s}"), pack, np.random.default_rng(s))
        present = {o.lab_category for o in obs}
        for lab in _ORDER:
            flags[lab].append(1 if lab in present else 0)
    return flags


def test_presence_copula_co_occurs_paneled_labs() -> None:
    # creatinine + bun are a "panel" (presence correlation ~1); sodium independent.
    corr = [[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 1.0]]
    paneled = _present_flags(_presence_pack(corr), 400)
    independent = _present_flags(_presence_pack(None), 400)

    def co_occur(flags: dict[str, list[int]], a: str, b: str) -> float:
        return float(np.mean([x == y for x, y in zip(flags[a], flags[b], strict=True)]))

    # With the copula, creatinine and bun are present-together in nearly every stay;
    # independent Bernoulli draws (no correlation matrix) agree only ~half the time.
    assert co_occur(paneled, "creatinine", "bun") > 0.9
    assert co_occur(independent, "creatinine", "bun") < 0.75
    # Each lab's marginal presence is preserved by the copula (~0.5, the fitted rate).
    assert 0.4 < float(np.mean(paneled["creatinine"])) < 0.6


def test_is_deterministic_under_fixed_seed() -> None:
    pack = _pack()
    sp = _spine([3] * 10)
    a = sample_labs(sp, pack, np.random.default_rng(4))
    b = sample_labs(sp, pack, np.random.default_rng(4))
    assert a == b


def test_frame_reproducible_byte_for_byte() -> None:
    pack = _pack()
    sp = _spine([3] * 8, hid="H1")
    a = labs_frame(sample_labs(sp, pack, np.random.default_rng(5)))
    b = labs_frame(sample_labs(sp, pack, np.random.default_rng(5)))
    assert a.equals(b)


def test_generated_correlation_recovers_copula() -> None:
    pack = _pack()
    rng = np.random.default_rng(0)
    # Pair labs measured in the same panel (same hid + order time).
    panels: dict[tuple[str, object], dict[str, float]] = {}
    for i in range(40):
        for o in sample_labs(_spine([3] * 30, hid=f"H{i}"), pack, rng):
            panels.setdefault((o.hospitalization_id, o.lab_order_dttm), {})[o.lab_category] = (
                o.lab_value_numeric
            )
    creat = [p["creatinine"] for p in panels.values()]
    bun = [p["bun"] for p in panels.values()]
    sod = [p["sodium"] for p in panels.values()]
    assert _spearman(creat, bun) > 0.45  # strong positive, ~0.6 in the pack
    assert abs(_spearman(creat, sod)) < 0.15  # ~uncorrelated in the pack


def test_all_values_within_outlier_bounds() -> None:
    pack = _pack()
    rng = np.random.default_rng(1)
    for i in range(20):
        for o in sample_labs(_spine([3] * 20, hid=f"H{i}"), pack, rng):
            lo, hi = bounds("labs", o.lab_category)
            assert lo <= o.lab_value_numeric <= hi


def test_presence_rates_match_pack() -> None:
    pack = _pack(presence={"creatinine": 0.7, "bun": 0.4, "sodium": 0.9})
    rng = np.random.default_rng(2)
    n_stays = 3000
    stays_with = {"creatinine": 0, "bun": 0, "sodium": 0}
    for i in range(n_stays):
        seen = {o.lab_category for o in sample_labs(_spine([3], hid=f"H{i}"), pack, rng)}
        for lab in stays_with:
            if lab in seen:
                stays_with[lab] += 1
    for lab, rate in {"creatinine": 0.7, "bun": 0.4, "sodium": 0.9}.items():
        assert abs(stays_with[lab] / n_stays - rate) < 0.04


def test_masked_absent_lab_never_emitted() -> None:
    pack = _pack(presence={"creatinine": 1.0, "bun": 1.0, "sodium": 0.0})
    rng = np.random.default_rng(3)
    for i in range(50):
        for o in sample_labs(_spine([3] * 5, hid=f"H{i}"), pack, rng):
            assert o.lab_category != "sodium"  # sample-then-mask: absent -> no row


def test_creatinine_rises_under_renal_failure() -> None:
    pack = _pack()
    healthy = _spine([3] * 60, hid="Hok", renal=False)
    failing = _spine([3] * 60, hid="Hbad", renal=True)
    ok = [
        o.lab_value_numeric
        for o in sample_labs(healthy, pack, np.random.default_rng(1))
        if o.lab_category == "creatinine"
    ]
    bad = [
        o.lab_value_numeric
        for o in sample_labs(failing, pack, np.random.default_rng(1))
        if o.lab_category == "creatinine"
    ]
    sod_ok = [
        o.lab_value_numeric
        for o in sample_labs(healthy, pack, np.random.default_rng(1))
        if o.lab_category == "sodium"
    ]
    sod_bad = [
        o.lab_value_numeric
        for o in sample_labs(failing, pack, np.random.default_rng(1))
        if o.lab_category == "sodium"
    ]
    assert float(np.mean(bad)) > float(np.mean(ok))  # renal coupling raises creatinine
    assert abs(float(np.mean(sod_bad)) - float(np.mean(sod_ok))) < 1.0  # sodium uncoupled


# --------------------------------------------------------------------------- #
# Empirical-quantile (inverse-CDF) marginals
# --------------------------------------------------------------------------- #
# A distinctly non-log-normal creatinine marginal: a two-mode distribution with
# ~half the mass near 0.8 and ~half near 6.0 (mimicking a normal-renal cluster plus
# a CKD tail), which a single log-normal cannot represent. On the 101-point grid,
# probs < 0.50 map to 0.8, probs > 0.50 map to 6.0.
_BIMODAL_CREATININE = [0.8] * 50 + [6.0] * 51


def _quantile_pack(quantiles: dict[str, list[float]]) -> ParamPack:
    pack = _pack()
    pack.tables["labs"]["params"]["lab_quantiles"] = quantiles
    return pack


def test_lab_quantiles_marginal_is_recovered() -> None:
    pack = _quantile_pack({"creatinine": _BIMODAL_CREATININE})
    rng = np.random.default_rng(0)
    vals = np.asarray(
        [
            o.lab_value_numeric
            for i in range(60)
            for o in sample_labs(_spine([3] * 30, hid=f"H{i}"), pack, rng)
            if o.lab_category == "creatinine"
        ]
    )
    # Bimodal shape is reproduced: ~half the mass at each mode, and almost nothing in
    # the middle band a single log-normal would fill.
    assert 0.4 < float(np.mean(vals < 2.0)) < 0.6
    assert float(np.mean((vals > 2.0) & (vals < 5.0))) < 0.05
    # Empirical quantiles recover the fitted grid within tolerance (marginal matched).
    assert abs(float(np.quantile(vals, 0.25)) - 0.8) < 0.1
    assert abs(float(np.quantile(vals, 0.75)) - 6.0) < 0.1


def test_pack_without_quantiles_uses_lognormal_fallback() -> None:
    # Backward-compat: older/demo packs carry no ``lab_quantiles`` and must still
    # generate through the log-normal marginal path.
    pack = _pack()
    assert "lab_quantiles" not in pack.tables["labs"]["params"]
    obs = sample_labs(_spine([3] * 6), pack, np.random.default_rng(0))
    assert obs
    for o in obs:
        lo, hi = bounds("labs", o.lab_category)
        assert lo <= o.lab_value_numeric <= hi


def test_quantile_path_renal_coupling_raises_creatinine() -> None:
    pack = _quantile_pack({"creatinine": _BIMODAL_CREATININE})
    healthy = _spine([3] * 60, hid="Hok", renal=False)
    failing = _spine([3] * 60, hid="Hbad", renal=True)
    ok = [
        o.lab_value_numeric
        for o in sample_labs(healthy, pack, np.random.default_rng(1))
        if o.lab_category == "creatinine"
    ]
    bad = [
        o.lab_value_numeric
        for o in sample_labs(failing, pack, np.random.default_rng(1))
        if o.lab_category == "creatinine"
    ]
    assert float(np.mean(bad)) > float(np.mean(ok))  # value-space renal bump still lifts it


def test_quantile_path_preserves_rng_stream() -> None:
    # ``ndtr`` and ``np.interp`` draw no rng, so swapping the marginal map must not
    # shift the stream: the presence draw and per-panel jitter draws are unchanged, so
    # row count and order timestamps are byte-identical to the log-normal path.
    qpack = _quantile_pack(
        {
            "creatinine": _BIMODAL_CREATININE,
            "bun": [10.0] * 50 + [40.0] * 51,
            "sodium": [138.0] * 50 + [142.0] * 51,
        }
    )
    lpack = _pack()
    sp = _spine([3] * 12, hid="Hs")
    q = sample_labs(sp, qpack, np.random.default_rng(7))
    lognormal = sample_labs(sp, lpack, np.random.default_rng(7))
    assert [o.lab_order_dttm for o in q] == [o.lab_order_dttm for o in lognormal]
    assert [o.lab_category for o in q] == [o.lab_category for o in lognormal]


def test_no_icu_time_yields_no_labs() -> None:
    pack = _pack()
    ward_only = _spine([0, 1, 1, 0], hid="Hward")  # never reaches ICU threshold
    assert sample_labs(ward_only, pack, np.random.default_rng(0)) == []


def test_categories_are_exact_mcide_members() -> None:
    pack = _pack()
    ok = set(categories("labs", "lab_category"))
    seen = {o.lab_category for o in sample_labs(_spine([3] * 10), pack, np.random.default_rng(0))}
    assert seen and seen <= ok


def test_names_echo_categories() -> None:
    pack = _pack()
    for o in sample_labs(_spine([3] * 6), pack, np.random.default_rng(0)):
        assert o.lab_name == o.lab_category


def test_timing_order_is_order_le_collect_lt_result() -> None:
    pack = _pack()
    for o in sample_labs(_spine([3] * 8), pack, np.random.default_rng(0)):
        assert o.lab_order_dttm <= o.lab_collect_dttm < o.lab_result_dttm


def test_frame_crosswalks_are_exact_mcide_companions() -> None:
    from clifforge.reference import loader

    pack = _pack()
    frame = labs_frame(sample_labs(_spine([3] * 6), pack, np.random.default_rng(0)))
    unit = loader.crosswalk("labs", "lab_category", "reference_unit")
    order_cat = loader.crosswalk("labs", "lab_category", "lab_order_category")
    order_name = loader.crosswalk("labs", "lab_order_category", "description")
    ok_orders = set(categories("labs", "lab_order_category"))
    for row in frame.iter_rows(named=True):
        assert row["reference_unit"] == unit[row["lab_category"]]
        assert row["lab_order_category"] == order_cat[row["lab_category"]]
        assert row["lab_order_category"] in ok_orders
        assert row["lab_order_name"] == (
            order_name.get(row["lab_order_category"], row["lab_order_category"])
            or row["lab_order_category"]
        )


def test_frame_passes_gate_and_datetimes_are_tz_aware() -> None:
    pack = _pack()
    obs: list[LabObservation] = []
    for i in range(20):
        obs += sample_labs(_spine([2, 3, 4, 3, 2], hid=f"H{i}"), pack, np.random.default_rng(i))
    frame = labs_frame(obs)
    for col in ("lab_order_dttm", "lab_collect_dttm", "lab_result_dttm"):
        dtype = frame.schema[col]
        assert isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC"
    report = gate.validate(frame, "labs", run_secondary=False)
    assert report.pandera_passed


def test_missing_labs_block_raises() -> None:
    empty = ParamPack(manifest={}, tables={"spine": {"params": {"state_model": {}}}})
    try:
        sample_labs(_spine([3, 3]), empty, np.random.default_rng(0))
    except ValueError as exc:
        assert "labs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a pack with no labs block")


def test_module_exports() -> None:
    assert set(labs.__all__) == {"LabObservation", "labs_frame", "sample_labs"}
