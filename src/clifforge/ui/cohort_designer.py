"""CLIFForge Cohort Designer — a Streamlit front-end for designing and generating
synthetic CLIF datasets.

This is a *thin wrapper* over the real engine: the sidebar builds a
:class:`clifforge.variants.VariantSpec`, ``spec_to_pack`` recalibrates the shared
base pack to that recipe, and ``generate_dataset`` produces the tables — exactly
the pipeline ``clif-forge generate`` runs. Nothing here re-implements generation,
so the GUI and the CLI always agree.

Launch with ``clif-forge ui`` (or ``streamlit run`` on this file).
"""

from __future__ import annotations

import dataclasses
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from clifforge import __version__
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.filenames import is_deliverable_table, table_parquet_path
from clifforge.generate.orchestrator import TRUTH_FILENAME, generate_dataset
from clifforge.manifest import write_manifest
from clifforge.preview import PREVIEW_SAMPLE, cohort_profile
from clifforge.variants import (
    VariantSpec,
    default_base_pack_path,
    list_presets,
    load_preset,
    spec_to_pack,
)

#: Encounters used for the interactive preview — small enough to regenerate on
#: each change in a second or two; distributions converge at full size.
PREVIEW_N = PREVIEW_SAMPLE
#: Above this, in-app generation is slow (generation is a per-encounter Python
#: loop); steer the user to the CLI / streaming path instead.
MAX_UI_N = 25_000
#: clif-icu.com brand accent teal, used for the preview chart bars.
TEAL = "#14867e"

st.set_page_config(page_title="CLIFForge — Cohort Designer", page_icon="🫀", layout="wide")


# --------------------------------------------------------------------------- #
# Engine access (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _base_pack() -> ParamPack:
    """Load the shipped aggregate base pack once per session."""
    return ParamPack.load(default_base_pack_path())


def _build_spec(w: dict[str, Any]) -> VariantSpec:
    """Assemble a VariantSpec from the sidebar widget values."""
    race_target = None
    if w["race_custom"]:
        total = w["race_white"] + w["race_black"] + w["race_other"]
        if total > 0:
            race_target = {
                "White": w["race_white"] / total,
                "Black or African American": w["race_black"] / total,
                "Other": w["race_other"] / total,
            }
    return VariantSpec(
        name=w["name"] or "my-cohort",
        n=int(w["n"]),
        seed=int(w["seed"]),
        mode=w["mode"],
        age_shift=float(w["age_shift"]),
        hispanic_frac=(float(w["hispanic_frac"]) if w["hispanic_custom"] else None),
        race_target=race_target,
        imv=float(w["imv"]),
        mortality_scale=float(w["mortality_scale"]),
        vaso_frac=float(w["vaso_frac"]),
        crrt_prob=float(w["crrt_prob"]),
        prone_severe=float(w["prone_severe"]),
    )


def _recipe_toml(spec: VariantSpec) -> str:
    """Render a faithful, minimal TOML recipe for ``spec``.

    Only fields that differ from the master defaults are emitted (omitted fields
    fall back to the same defaults on load), so the shown recipe reproduces the
    previewed cohort exactly — the GUI and ``clif-forge generate --spec`` agree.
    """
    d = VariantSpec()
    lines = [
        f'name = "{spec.name}"',
        f'mode = "{spec.mode}"',
        f"n = {spec.n}",
        f"seed = {spec.seed}",
    ]

    demo: list[str] = []
    if spec.age_shift != d.age_shift:
        demo.append(f"age_shift = {spec.age_shift}")
    if spec.hispanic_frac is not None:
        demo.append(f"hispanic_frac = {spec.hispanic_frac}")
    if spec.race_target:
        inner = ", ".join(f'"{k}" = {v:.3f}' for k, v in spec.race_target.items())
        demo.append(f"race_target = {{ {inner} }}")
    if demo:
        lines += ["", "[demographics]", *demo]

    rates: list[str] = []
    if spec.mode == "icu":  # full_hospital ignores these except crrt_prob
        for field in ("imv", "mortality_scale", "vaso_frac", "prone_severe"):
            if getattr(spec, field) != getattr(d, field):
                rates.append(f"{field} = {getattr(spec, field)}")
    if spec.crrt_prob != d.crrt_prob:
        rates.append(f"crrt_prob = {spec.crrt_prob}")
    if rates:
        lines += ["", "[rates]", *rates]
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def _preview(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Generate a small sample for the given recipe and return preview artifacts.

    Keyed (via ``spec_dict``) so moving a slider back to a prior value is instant.
    """
    spec = dataclasses.replace(VariantSpec(**spec_dict), n=PREVIEW_N)
    pack = spec_to_pack(spec, _base_pack())
    ds = generate_dataset(pack, n_patients=PREVIEW_N, seed=spec.seed)
    stats = cohort_profile(ds)

    h = ds.tables["hospitalization"]
    los = ((h["discharge_dttm"] - h["admission_dttm"]).dt.total_seconds() / 3600.0).to_numpy()
    los = los[np.isfinite(los)]
    counts, edges = np.histogram(np.clip(los, 0, 720), bins=24)
    los_hist = pd.DataFrame({"LOS (h)": np.round(edges[:-1]).astype(int), "count": counts})

    peak = (
        ds.truth.group_by("hospitalization_id")
        .agg(pl.col("support_level").max().alias("peak"))["peak"]
        .to_numpy()
    )
    levels = ["0 room-air", "1 low-O₂", "2 HFNC/NIV", "3 IMV", "4 +vaso", "5 +CRRT/ECMO"]
    lvl_counts = [int((peak == i).sum()) for i in range(6)]
    support = pd.DataFrame({"peak support": levels, "encounters": lvl_counts})

    patient = ds.tables["patient"]
    demo_cols = [c for c in ("patient_id", "sex_category", "race_category") if c in patient.columns]
    sample = h.select(
        [c for c in ("patient_id", "age_at_admission", "discharge_category") if c in h.columns]
    ).head(12)
    if demo_cols:
        sample = sample.join(patient.select(demo_cols), on="patient_id", how="left")
    return {
        "stats": stats,
        "los_hist": los_hist,
        "support": support,
        "sample": sample.to_pandas(),
    }


def _generate_download(spec: VariantSpec) -> tuple[bytes, dict[str, Any]]:
    """Generate the full cohort, write parquet + manifest, return a zip + manifest."""
    ds = generate_dataset(spec_to_pack(spec, _base_pack()), n_patients=spec.n, seed=spec.seed)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for name, frame in ds.tables.items():
            if not is_deliverable_table(name):
                continue
            frame.write_parquet(table_parquet_path(out, name))
        ds.truth.write_parquet(out / TRUTH_FILENAME)
        manifest = write_manifest(out, spec=dataclasses.asdict(spec), seed=spec.seed)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out.iterdir()):
                zf.write(f, arcname=f"{spec.name}/{f.name}")
    return buf.getvalue(), manifest


# --------------------------------------------------------------------------- #
# Sidebar — the levers
# --------------------------------------------------------------------------- #
st.sidebar.title("🫀 Cohort Designer")
st.sidebar.caption(f"CLIFForge v{__version__} · synthetic CLIF 2.1")

presets = ["(start from defaults)", *list_presets()]
chosen_preset = st.sidebar.selectbox("Start from a preset", presets, index=0)
base_spec = load_preset(chosen_preset) if chosen_preset in list_presets() else VariantSpec()

w: dict[str, Any] = {}
w["name"] = st.sidebar.text_input("Dataset name", value=base_spec.name)
w["mode"] = st.sidebar.radio(
    "Population",
    options=["icu", "full_hospital"],
    format_func=lambda m: "ICU cohort" if m == "icu" else "Whole-hospital",
    index=0 if base_spec.mode == "icu" else 1,
    horizontal=True,
)
w["n"] = st.sidebar.slider(
    "Size (encounters)", min_value=100, max_value=MAX_UI_N, value=min(base_spec.n, 5_000), step=100
)
w["seed"] = st.sidebar.number_input("Seed", min_value=0, value=int(base_spec.seed), step=1)

with st.sidebar.expander("Demographics", expanded=False):
    w["age_shift"] = st.slider("Age shift (years)", -15.0, 15.0, float(base_spec.age_shift), 0.5)
    w["hispanic_custom"] = st.checkbox(
        "Set Hispanic fraction", value=base_spec.hispanic_frac is not None
    )
    w["hispanic_frac"] = st.slider(
        "Hispanic fraction",
        0.0,
        1.0,
        float(base_spec.hispanic_frac or 0.2),
        0.01,
        disabled=not w["hispanic_custom"],
    )
    w["race_custom"] = st.checkbox("Set race mix", value=base_spec.race_target is not None)
    rt = base_spec.race_target or {}
    w["race_white"] = st.slider(
        "White", 0.0, 1.0, float(rt.get("White", 0.42)), 0.01, disabled=not w["race_custom"]
    )
    w["race_black"] = st.slider(
        "Black",
        0.0,
        1.0,
        float(rt.get("Black or African American", 0.30)),
        0.01,
        disabled=not w["race_custom"],
    )
    w["race_other"] = st.slider("Other", 0.0, 1.0, 0.28, 0.01, disabled=not w["race_custom"])

icu_mode = w["mode"] == "icu"
with st.sidebar.expander("Illness rates", expanded=icu_mode):
    if not icu_mode:
        st.caption(
            "Illness-rate levers apply to the ICU cohort. Whole-hospital uses its own "
            "tuned patient-flow defaults; only CRRT and demographics apply."
        )
    w["imv"] = st.slider(
        "Invasive ventilation", 0.0, 1.0, float(base_spec.imv), 0.01, disabled=not icu_mode
    )
    w["mortality_scale"] = st.slider(
        "Mortality ×", 0.1, 3.0, float(base_spec.mortality_scale), 0.05, disabled=not icu_mode
    )
    w["vaso_frac"] = st.slider(
        "Vasopressors (CV failure)",
        0.0,
        1.0,
        float(base_spec.vaso_frac),
        0.01,
        disabled=not icu_mode,
    )
    w["crrt_prob"] = st.slider(
        "CRRT (among renal failure)", 0.0, 1.0, float(base_spec.crrt_prob), 0.01
    )
    w["prone_severe"] = st.slider(
        "Proning (severe hypoxemia)",
        0.0,
        1.0,
        float(base_spec.prone_severe),
        0.005,
        disabled=not icu_mode,
    )

spec = _build_spec(w)


# --------------------------------------------------------------------------- #
# Main — recipe, preview, generate
# --------------------------------------------------------------------------- #
st.title("Design a synthetic CLIF cohort")
st.markdown(
    "Move the levers on the left, preview the cohort you'd get, then generate and "
    "download it. Every recipe is **distinct** but still clinically realistic and "
    "**CLIF 2.1-conformant** — and reproducible from the spec below."
)

st.header("Your recipe")
st.markdown(
    f"**{spec.name}** · {'ICU cohort' if icu_mode else 'Whole-hospital'} · "
    f"n = {spec.n:,} · seed {spec.seed}"
)
with st.expander("Reproduce on the command line — same recipe, same output"):
    st.markdown(f"Save this as `{spec.name}.toml`:")
    # Plain (no syntax highlighting) so every token keeps full dark-on-light
    # contrast; the block stays one-click copyable.
    st.code(_recipe_toml(spec), language=None)
    st.markdown("Then generate the identical dataset:")
    st.code(f"clif-forge generate --spec {spec.name}.toml --out ./{spec.name}", language=None)

st.divider()

st.header("Preview")
st.caption(
    f"A {PREVIEW_N}-encounter sample of this exact recipe. "
    "Full-size output converges to these shapes."
)
with st.spinner("Sampling this cohort…"):
    pv = _preview(dataclasses.asdict(spec))
s = pv["stats"]

m = st.columns(6)
m[0].metric("Encounters (sample)", f"{int(s['n'])}")
m[1].metric("Mortality", f"{s.get('mortality', 0) * 100:.1f}%")
m[2].metric("LOS median", f"{s['los_median_h']:.0f} h")
m[3].metric("Invasive vent", f"{s['imv'] * 100:.0f}%")
m[4].metric("Reached ICU", f"{s['icu'] * 100:.0f}%")
m[5].metric("Vasopressors", f"{s['vaso'] * 100:.0f}%")

c1, c2 = st.columns(2)
with c1:
    st.caption("Hospital length-of-stay (hours)")
    st.bar_chart(pv["los_hist"], x="LOS (h)", y="count", color=TEAL, height=240)
with c2:
    st.caption("Peak organ-support level")
    st.bar_chart(pv["support"], x="peak support", y="encounters", color=TEAL, height=240)

st.caption("Sample encounters")
st.dataframe(pv["sample"], width="stretch", hide_index=True)

st.divider()

st.header("Generate & download")
st.markdown(
    f"Generates the full **{spec.n:,}-encounter** dataset — one "
    "`clif_<table>_2.1_<beta|concept>.parquet` per CLIF table (share layout: "
    "exactly the 25 website-badged tables; no `_truth.parquet`) and a "
    "`manifest.json` (recipe, seed, per-table content hashes)."
)
if spec.n > 15_000:
    st.info(
        f"n = {spec.n:,} may take a few minutes in the browser. For very large cohorts, "
        f"use the CLI: `clif-forge generate --spec {spec.name}.toml --n-patients {spec.n}`."
    )
if st.button("Generate full dataset", type="primary"):
    with st.spinner(f"Generating {spec.n:,} encounters…"):
        payload, manifest = _generate_download(spec)
    st.success(f"Generated {len(manifest['tables'])} tables.")
    rows = pd.DataFrame(
        [{"table": t, "rows": v["rows"]} for t, v in sorted(manifest["tables"].items())]
    )
    st.dataframe(rows, width="stretch", hide_index=True, height=240)
    st.download_button(
        "⬇ Download dataset (.zip)",
        data=payload,
        file_name=f"{spec.name}.zip",
        mime="application/zip",
        type="primary",
    )
