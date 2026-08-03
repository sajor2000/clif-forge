"""``hospital_diagnosis`` generator (prior-driven; R14, KTD-6).

Billing diagnoses are the coded record of what an encounter was *about*. Acute
organ-failure codes are read off the spine's flags (KTD-6); chronic comorbidity /
cancer load is drawn from **MIMIC ICU stay-prevalence priors**, scaled up for
higher peak acuity and expired outcomes so sicker stays carry more chronics —
matching the real case-mix pattern (not independent coin-flips).

Codes are **ICD-10-CM**. A fitted ``diagnosis_code_marginal`` (when present) only
pads the list toward MIMIC code-density (~18/stay); it never replaces the
calibrated comorbidity / cancer layer.

Reproducible under a fixed ``rng`` (R22).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from clifforge.fit.param_pack import ParamPack
from clifforge.generate.sampling import categorical
from clifforge.generate.spine import SpineFrame

__all__ = ["DiagnosisRow", "hospital_diagnosis_frame", "sample_hospital_diagnosis"]

_CODE_FORMAT = "ICD10CM"

#: Spine organ-failure flag -> ICD-10-CM (fallback when phenotype does not refine).
_FLAG_CODES: tuple[tuple[str, str], ...] = (
    ("resp_flag", "J96.00"),
    ("cv_flag", "R65.21"),
    ("renal_flag", "N17.9"),
    ("neuro_flag", "G93.40"),
)

_RESP_PHENOTYPE_CODES: dict[str, str] = {
    "type1": "J96.01",
    "type2_ohs": "J96.02",
    "type2_hf": "J96.02",
    "type2_copd": "J96.02",
}

_RESP_PHENOTYPE_COMORBID: dict[str, tuple[str, ...]] = {
    "type2_ohs": ("E66.2",),
    "type2_hf": ("I50.9",),
    "type2_copd": ("J44.9",),
}

#: ICU principal-dx mix (stay share) — pneumonia is common but not universal.
_PRINCIPAL_MIX: dict[str, float] = {
    "J18.9": 0.09,  # pneumonia
    "A41.9": 0.10,  # sepsis, unspecified organism
    "I50.9": 0.07,  # heart failure
    "J96.01": 0.07,  # hypoxemic respiratory failure
    "I21.4": 0.07,  # NSTEMI
    "N17.9": 0.06,  # AKI
    "J44.1": 0.02,  # COPD with acute exacerbation
    "I63.9": 0.05,  # cerebral infarction
    "J69.0": 0.04,  # aspiration pneumonia
    "K92.2": 0.04,  # GI hemorrhage
    "R57.0": 0.04,  # cardiogenic shock
    "J96.02": 0.04,  # hypercapnic RF
    "E87.1": 0.04,  # hypo-osmolality
    "I48.91": 0.04,  # AF
    "S06.9X9A": 0.03,  # TBI
    "K70.30": 0.03,  # alcoholic cirrhosis with ascites
    "E11.65": 0.03,  # T2DM with hyperglycemia
    "N39.0": 0.04,  # UTI
    "R69": 0.04,  # illness, unspecified
}

#: MIMIC ICU stay-level disease priors: (family, codes to pick among, base stay rate).
#: One draw per family so HTN/CKD staging codes do not stack into near-100% rates.
_DISEASE_PRIORS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("htn", ("I10",), 0.60),
    ("lipid", ("E78.5",), 0.38),
    ("dm", ("E11.9",), 0.29),
    ("cad", ("I25.10",), 0.31),
    ("afib", ("I48.91",), 0.27),
    ("hf", ("I50.9",), 0.16),
    ("copd", ("J44.9",), 0.0),  # coded via type2_copd phenotype force (~MIMIC 0.15)
    ("ckd", ("N18.3", "N18.4", "N18.5"), 0.18),
    ("obesity", ("E66.9",), 0.05),
    ("liver", ("K74.60", "K70.30"), 0.05),
    ("depression", ("F32.9",), 0.10),
    ("anxiety", ("F41.9",), 0.08),
    ("thyroid", ("E03.9",), 0.10),
    ("tobacco_hx", ("Z87.891",), 0.12),
    ("gerd", ("K21.9",), 0.12),
    ("anemia", ("D62",), 0.08),
    ("anticoag", ("Z79.01",), 0.08),
    ("insulin", ("Z79.4",), 0.06),
)

#: Flat comorbidity list kept for filler categorical draws (legacy import surface).
_COMORBIDITY_CODES: tuple[tuple[str, float], ...] = tuple(
    (codes[0], rate) for _, codes, rate in _DISEASE_PRIORS
)

#: Any-cancer stay rate ~0.21 in MIMIC ICU; draw once then pick a site / history code.
_CANCER_BASE_RATE = 0.18
_CANCER_CODES: tuple[str, ...] = (
    "C34.90",
    "C50.919",
    "C61",
    "C18.9",
    "C25.9",
    "C80.1",
    "C78.00",
    "C79.51",
    "Z85.3",
    "Z85.118",
    "Z85.038",
)

#: Non-disease secondaries used to pad toward MIMIC code density (~18/stay).
_SAFE_FILLERS: dict[str, float] = {
    "Z87.891": 0.12,
    "K21.9": 0.10,
    "F32.9": 0.08,
    "F41.9": 0.07,
    "E03.9": 0.08,
    "Z20.822": 0.06,
    "Z79.01": 0.06,
    "Z79.4": 0.05,
    "D62": 0.06,
    "E87.1": 0.05,
    "E87.5": 0.04,
    "E87.6": 0.04,
    "R00.0": 0.04,
    "R00.1": 0.03,
    "R06.02": 0.04,
    "R50.9": 0.03,
    "R53.1": 0.03,
    "Y92.230": 0.02,
    "Z51.11": 0.03,
    "Z99.11": 0.02,
    "Z99.2": 0.02,
    "Z66": 0.02,
    "Z23": 0.03,
    "T45.1X5A": 0.02,
    "T50.901A": 0.02,
    "W19.XXXA": 0.02,
}
_RESP_FAILURE_EMIT_TYPED = 0.55
_RESP_FAILURE_EMIT_IF_IMV = 0.32
_RESP_FAILURE_EMIT_FLOOR = 0.10
_SEPSIS_EMIT_IF_SHOCK_TIER = 0.55
_SEPSIS_EMIT_FLOOR = 0.18

#: Filler must not invent chronic/acute disease families already governed by priors.
_FILLER_BLOCK_PREFIXES: tuple[str, ...] = (
    "I10",
    "I11",
    "I12",
    "I13",
    "401",
    "402",
    "403",
    "404",
    "E11",
    "E10",
    "250",
    "I50",
    "428",
    "I25",
    "I21",
    "414",
    "410",
    "I48",
    "4273",
    "N18",
    "585",
    "J44",
    "J45",
    "496",
    "E66",
    "278",
    "C",
    "Z85",
    "K70",
    "K74",
    "571",
    "J18",
    "J69",
    "J96",
    "51881",
    "5185",
    "A41",
    "R65",
    "038",
    "78552",
    "N17",
    "584",
)


def _filler_blocked(code: str) -> bool:
    n = str(code).replace(".", "").upper()
    return any(n.startswith(p) for p in _FILLER_BLOCK_PREFIXES)


@dataclass(frozen=True)
class DiagnosisRow:
    """One coded billing diagnosis for a hospitalization."""

    hospitalization_id: str
    diagnosis_code: str
    diagnosis_code_format: str
    diagnosis_primary: int
    poa_present: int


def _first_true(flags: list[bool]) -> int | None:
    """Index of the first interval where a flag fires, or ``None`` if it never does."""
    for i, flag in enumerate(flags):
        if flag:
            return i
    return None


def _disease_rate(family: str, base: float, spine: SpineFrame) -> float:
    """Light additive acuity bumps (MIMIC: expired / high-peak carry more chronics)."""
    p = base
    if spine.outcome == "expired":
        p += 0.07
    if spine.peak_level >= 5:
        p += 0.06
    elif spine.peak_level >= 4:
        p += 0.03
    if family == "ckd" and any(spine.renal_flag):
        p += 0.14
    if family == "hf" and any(spine.cv_flag):
        p += 0.03
    return min(p, 0.82)


def _acuity_comorbidity_boost(spine: SpineFrame) -> float:
    """Legacy scalar used by patient_diagnosis cancer / shared helpers."""
    boost = 1.0
    if spine.peak_level >= 5:
        boost += 0.20
    elif spine.peak_level >= 4:
        boost += 0.10
    if spine.outcome == "expired":
        boost += 0.15
    if any(spine.renal_flag):
        boost += 0.10
    return min(boost, 1.45)


def _target_code_count(spine: SpineFrame, rng: np.random.Generator) -> int:
    """MIMIC ICU mean ~18 codes/stay; sicker stays run denser."""
    base = int(rng.integers(14, 20))
    if spine.outcome == "expired":
        base += int(rng.integers(3, 7))
    if spine.peak_level >= 5:
        base += int(rng.integers(2, 5))
    elif spine.peak_level >= 4:
        base += int(rng.integers(1, 3))
    return min(base, 32)


def _pick_principal(spine: SpineFrame, rng: np.random.Generator) -> str:
    """Principal dx biased by phenotype / organ flags, else ICU mix.

    Acute organ-failure principals are only offered when that flag is present
    (and, for respiratory failure, present on admission) so a later-onset
    failure is never the admitting diagnosis.
    """
    phenotype = getattr(spine, "resp_phenotype", "") or ""
    resp_onset = _first_true(spine.resp_flag)
    mix = dict(_PRINCIPAL_MIX)
    if not any(spine.renal_flag):
        mix.pop("N17.9", None)
    if not any(spine.cv_flag):
        mix.pop("A41.9", None)
        mix.pop("R57.0", None)
    if resp_onset != 0:
        mix.pop("J96.01", None)
        mix.pop("J96.02", None)

    if phenotype == "type2_hf" and rng.random() < 0.45:
        return "I50.9"
    if phenotype == "type2_copd" and rng.random() < 0.25:
        return "J44.1"
    if phenotype == "type2_ohs" and resp_onset == 0 and rng.random() < 0.35:
        return "J96.02"
    if phenotype == "type1" and resp_onset == 0 and rng.random() < 0.35:
        return "J96.01" if rng.random() < 0.5 else "J18.9"
    if any(spine.cv_flag) and spine.peak_level >= 4 and rng.random() < 0.40:
        return "A41.9"
    if any(spine.renal_flag) and rng.random() < 0.25:
        return "N17.9"
    if not mix:
        return "R69"
    return categorical(mix, rng)


def _emit_resp_failure(spine: SpineFrame, rng: np.random.Generator) -> bool:
    if not any(spine.resp_flag):
        return False
    phenotype = getattr(spine, "resp_phenotype", "") or ""
    if phenotype.startswith("type"):
        return rng.random() < _RESP_FAILURE_EMIT_TYPED
    if spine.peak_level >= 3:
        return rng.random() < _RESP_FAILURE_EMIT_IF_IMV
    return rng.random() < _RESP_FAILURE_EMIT_FLOOR


def _emit_sepsis(spine: SpineFrame, rng: np.random.Generator) -> bool:
    if not any(spine.cv_flag):
        return False
    if spine.peak_level >= 4 or spine.outcome == "expired":
        return rng.random() < _SEPSIS_EMIT_IF_SHOCK_TIER
    return rng.random() < _SEPSIS_EMIT_FLOOR


def sample_hospital_diagnosis(
    spine: SpineFrame,
    pack: ParamPack,
    rng: np.random.Generator,
    *,
    hospitalization_id: str | None = None,
    admit_dttm: datetime = datetime(2020, 1, 1, tzinfo=UTC),
) -> list[DiagnosisRow]:
    """Emit the coded diagnosis list implied by this stay's spine (R22)."""
    hid = hospitalization_id if hospitalization_id is not None else spine.hospitalization_id
    if spine.n_intervals <= 0:
        return []

    principal = _pick_principal(spine, rng)
    rows: list[DiagnosisRow] = [DiagnosisRow(hid, principal, _CODE_FORMAT, 1, 1)]
    seen = {principal}
    phenotype = getattr(spine, "resp_phenotype", "") or ""

    for flag_name, code in _FLAG_CODES:
        onset = _first_true(getattr(spine, flag_name))
        if onset is None:
            continue
        if flag_name == "resp_flag":
            if not _emit_resp_failure(spine, rng):
                continue
            code = _RESP_PHENOTYPE_CODES.get(phenotype, code)
        elif flag_name == "cv_flag":
            if not _emit_sepsis(spine, rng):
                continue
        if code in seen:
            continue
        seen.add(code)
        rows.append(DiagnosisRow(hid, code, _CODE_FORMAT, 0, 1 if onset == 0 else 0))

    for code in _RESP_PHENOTYPE_COMORBID.get(phenotype, ()):
        if code not in seen:
            seen.add(code)
            rows.append(DiagnosisRow(hid, code, _CODE_FORMAT, 0, 1))

    for family, codes, base in _DISEASE_PRIORS:
        # Phenotype already forced the pathway comorbidity — skip the family draw.
        if family == "hf" and phenotype == "type2_hf":
            continue
        if family == "copd" and phenotype == "type2_copd":
            continue
        if family == "obesity" and phenotype == "type2_ohs":
            continue
        if any(c in seen for c in codes):
            continue
        if rng.random() >= _disease_rate(family, base, spine):
            continue
        code = codes[int(rng.integers(0, len(codes)))]
        seen.add(code)
        rows.append(DiagnosisRow(hid, code, _CODE_FORMAT, 0, 1))

    cancer_p = _disease_rate("cancer", _CANCER_BASE_RATE, spine)
    if spine.outcome == "expired":
        cancer_p = min(0.55, cancer_p + 0.06)  # MIMIC: expired cancer ~0.31
    if rng.random() < cancer_p:
        code = _CANCER_CODES[int(rng.integers(0, len(_CANCER_CODES)))]
        if code not in seen:
            seen.add(code)
            rows.append(DiagnosisRow(hid, code, _CODE_FORMAT, 0, 1))

    block = pack.tables.get("hospital_diagnosis", {})
    params = block.get("params", {}) if isinstance(block, dict) else {}
    code_marginal = params.get("diagnosis_code_marginal")
    target = _target_code_count(spine, rng)
    filler: dict[str, float] | None = (
        code_marginal if isinstance(code_marginal, dict) and code_marginal else None
    )
    # Acute / disease codes that must not be invented by filler without a flag.
    blocked = {"N17.9", "N179", "R65.21", "R6521", "J96.00", "J9600", "J96.01", "J9601", "J96.02", "J9602"}
    guard = 0
    while len(rows) < target and guard < 120:
        guard += 1
        # Prefer fitted marginal when it is a safe (non-disease) code; else safe pool.
        if filler is not None and rng.random() < 0.55:
            code = categorical(filler, rng)
            if code in seen or code in blocked or _filler_blocked(code):
                code = categorical(_SAFE_FILLERS, rng)
        else:
            code = categorical(_SAFE_FILLERS, rng)
        if code in seen:
            continue
        seen.add(code)
        rows.append(DiagnosisRow(hid, code, _CODE_FORMAT, 0, 1))

    return rows


def hospital_diagnosis_frame(rows: list[DiagnosisRow]) -> pl.DataFrame:
    """Stack diagnosis rows into one conformant frame."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r.hospitalization_id for r in rows],
            "diagnosis_code": [r.diagnosis_code for r in rows],
            "diagnosis_code_format": [r.diagnosis_code_format for r in rows],
            "diagnosis_primary": [r.diagnosis_primary for r in rows],
            "poa_present": [r.poa_present for r in rows],
        },
        schema={
            "hospitalization_id": pl.String,
            "diagnosis_code": pl.String,
            "diagnosis_code_format": pl.String,
            "diagnosis_primary": pl.Int64,
            "poa_present": pl.Int64,
        },
    )
