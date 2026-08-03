"""Declarative variant specs: turn a small TOML file into a generation pack.

A *variant* is a derivative of the master synthetic dataset differing on three
axes — **size**, **demographics**, and **illness rates** — expressed as a TOML
config and mapped to the existing derive/recalibrate parameters. Every field
defaults to the master's value, so a minimal spec reproduces the master.

Read with the stdlib :mod:`tomllib` (no new dependency). A spec generates against
a **base pack**: with ``real_dir`` supplied the full population derivation runs
(age/med fit from real data, credentialed path); without it the pack's existing
demographic blocks are re-weighted in place (the no-credential path a shareable
base pack enables).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.populations import (
    CHICAGO_ETHNICITY_TARGET,
    apply_population_overrides,
    derive_chicago_population,
)
from clifforge.generate.recalibrate import (
    recalibrate_fitted_icu,
    recalibrate_to_full_hospital,
    recalibrate_to_network_median,
)

__all__ = [
    "VariantSpec",
    "default_base_pack_path",
    "list_presets",
    "load_preset",
    "load_spec",
    "spec_to_pack",
]


def _resolve_data_dir(name: str) -> Path:
    """Locate a shipped data directory (``presets`` / ``base_pack``).

    Prefers the copy bundled inside the installed wheel (``clifforge/_data/<name>``,
    put there by hatch ``force-include``) so a ``pip install`` works standalone;
    falls back to the repo-root copy for an editable/development checkout.
    """
    packaged = Path(__file__).resolve().parent / "_data" / name
    return packaged if packaged.is_dir() else Path(__file__).resolve().parents[2] / name


#: Shipped example variant specs; a preset *is* a spec.
_PRESET_DIR = _resolve_data_dir("presets")


def default_base_pack_path() -> str:
    """The base pack to use when ``--base-pack`` is not given.

    A local ``./base_pack`` (a clone, or a pack you built) wins; otherwise the
    shareable base pack shipped inside the package, so a ``pip install`` can
    generate calibrated datasets without cloning the repo.
    """
    if Path("base_pack").is_dir():
        return "base_pack"
    return str(_resolve_data_dir("base_pack"))


_KNOWN_TOP = {"name", "n", "seed", "base_pack", "mode", "demographics", "rates"}
#: Population modes: ICU network-median, full hospital, or fitted empirical ICU.
_KNOWN_MODES = {"icu", "full_hospital", "fitted_icu"}
_KNOWN_DEMOGRAPHICS = {"age_shift", "hispanic_frac", "race_target"}
_KNOWN_RATES = {
    "imv",
    "mortality_scale",
    "vaso_frac",
    "crrt_prob",
    "prone_severe",
    "ecmo_stay",
}


@dataclass(frozen=True)
class VariantSpec:
    """A validated derivative spec; defaults are the master's network-median values."""

    name: str = "variant"
    n: int = 85_248
    seed: int = 2025
    base_pack: str | None = None
    #: "icu" (network-median), "full_hospital" (ward/ED/ICU mix), or "fitted_icu"
    #: (fitted all-28 pack + validated reference ICU recalibrate).
    mode: str = "icu"
    # demographics — age_shift is *relative to the base pack* (0 = the base's age
    # distribution); the master's own shift is already baked into the base pack.
    age_shift: float = 0.0
    hispanic_frac: float | None = None
    race_target: dict[str, float] | None = None
    # illness rates (network-median master defaults)
    imv: float = 0.28
    mortality_scale: float = 0.66
    vaso_frac: float = 0.27
    crrt_prob: float = 0.29
    prone_severe: float = 0.026
    #: Optional ECMO stay prevalence override (None = leave pack / network default).
    ecmo_stay: float | None = None


class SpecError(ValueError):
    """A variant spec failed validation."""


def _reject_unknown(section: dict[str, Any], known: set[str], where: str) -> None:
    extra = set(section) - known
    if extra:
        raise SpecError(f"unknown key(s) in {where}: {sorted(extra)}")


def _unit(value: float, name: str, *, lo: float = 0.0, hi: float = 1.0) -> float:
    if not lo <= value <= hi:
        raise SpecError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def _validate(spec: VariantSpec) -> None:
    if spec.n <= 0:
        raise SpecError(f"n must be positive, got {spec.n}")
    if spec.mode not in _KNOWN_MODES:
        raise SpecError(f"mode must be one of {sorted(_KNOWN_MODES)}, got {spec.mode!r}")
    _unit(spec.imv, "rates.imv")
    _unit(spec.vaso_frac, "rates.vaso_frac")
    _unit(spec.crrt_prob, "rates.crrt_prob")
    _unit(spec.prone_severe, "rates.prone_severe")
    if spec.ecmo_stay is not None:
        _unit(spec.ecmo_stay, "rates.ecmo_stay")
    if spec.mortality_scale <= 0:
        raise SpecError(f"rates.mortality_scale must be positive, got {spec.mortality_scale}")
    if spec.hispanic_frac is not None:
        _unit(spec.hispanic_frac, "demographics.hispanic_frac")


def _parse(data: dict[str, Any]) -> VariantSpec:
    _reject_unknown(data, _KNOWN_TOP, "top-level")
    demo = data.get("demographics", {})
    rates = data.get("rates", {})
    _reject_unknown(demo, _KNOWN_DEMOGRAPHICS, "[demographics]")
    _reject_unknown(rates, _KNOWN_RATES, "[rates]")
    defaults = VariantSpec()
    spec = VariantSpec(
        name=str(data.get("name", defaults.name)),
        n=int(data.get("n", defaults.n)),
        seed=int(data.get("seed", defaults.seed)),
        base_pack=data.get("base_pack"),
        mode=str(data.get("mode", defaults.mode)),
        age_shift=float(demo.get("age_shift", defaults.age_shift)),
        hispanic_frac=demo.get("hispanic_frac"),
        race_target=demo.get("race_target"),
        imv=float(rates.get("imv", defaults.imv)),
        mortality_scale=float(rates.get("mortality_scale", defaults.mortality_scale)),
        vaso_frac=float(rates.get("vaso_frac", defaults.vaso_frac)),
        crrt_prob=float(rates.get("crrt_prob", defaults.crrt_prob)),
        prone_severe=float(rates.get("prone_severe", defaults.prone_severe)),
        ecmo_stay=(
            float(rates["ecmo_stay"]) if "ecmo_stay" in rates else defaults.ecmo_stay
        ),
    )
    _validate(spec)
    return spec


def load_spec(path: str | Path) -> VariantSpec:
    """Load and validate a variant spec from a TOML file."""
    with Path(path).open("rb") as fh:
        return _parse(tomllib.load(fh))


def list_presets() -> list[str]:
    """Names of the shipped example variant specs."""
    if not _PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in _PRESET_DIR.glob("*.toml"))


def load_preset(name: str) -> VariantSpec:
    """Load a shipped preset by name (a preset is just a variant spec)."""
    path = _PRESET_DIR / f"{name}.toml"
    if not path.exists():
        raise SpecError(f"unknown preset {name!r}; available: {list_presets()}")
    return load_spec(path)


def spec_to_pack(
    spec: VariantSpec, base_pack: ParamPack, real_dir: str | Path | None = None
) -> ParamPack:
    """Build the generation pack for ``spec`` from ``base_pack`` (input not mutated).

    ``real_dir`` supplied -> full population derivation (credentialed path);
    ``real_dir`` None -> demographic overrides on the already-complete base pack
    (the no-credential path). Then network-median recalibration with the spec's
    rate overrides.
    """
    ethnicity = None
    if spec.hispanic_frac is not None:
        non_hisp = max(0.0, 1.0 - spec.hispanic_frac - 0.02)
        ethnicity = {"Hispanic": spec.hispanic_frac, "Non-Hispanic": non_hisp, "Unknown": 0.02}

    if real_dir is not None:
        derived = derive_chicago_population(
            base_pack,
            real_dir,
            age_shift_years=spec.age_shift,
            race_target=spec.race_target,
            ethnicity_target=ethnicity or CHICAGO_ETHNICITY_TARGET,
        )
    else:
        derived = apply_population_overrides(
            base_pack,
            age_shift_years=spec.age_shift,
            race_target=spec.race_target,
            ethnicity_target=ethnicity,
        )

    flags = {"resp_flag": 0.5, "cv_flag": spec.vaso_frac, "renal_flag": 0.05, "neuro_flag": 0.2}
    if spec.mode == "full_hospital":
        # Full hospital population (ward/ED/stepdown/ICU with realistic arrival flow).
        # The full-hospital transform carries its own tuned acuity/LOS/flow defaults;
        # the spec's ICU-specific rate overrides (imv, mortality_scale) do not apply.
        return recalibrate_to_full_hospital(derived, crrt_prob=spec.crrt_prob)
    if spec.mode == "fitted_icu":
        # Fitted all-28 pack + validated ICU recalibrate (IMV/mort/NIV/ADT arrivals).
        out = recalibrate_fitted_icu(derived)
        if spec.ecmo_stay is not None:
            tables = dict(out.tables)
            ecmo = dict(tables.get("ecmo_mcs", {}))
            ecmo_params = dict(ecmo.get("params", {}))
            ecmo_params["stay_prevalence"] = float(spec.ecmo_stay)
            ecmo_params["min_support_level"] = 4
            ecmo["params"] = ecmo_params
            tables["ecmo_mcs"] = ecmo
            return ParamPack(manifest=dict(out.manifest), tables=tables)
        return out
    return recalibrate_to_network_median(
        derived,
        peak_imv_target=spec.imv,
        mortality_scale=spec.mortality_scale,
        flag_target_prevalence=flags,
        crrt_prob=spec.crrt_prob,
        prone_prob_severe=spec.prone_severe,
        ecmo_stay=spec.ecmo_stay,
    )
