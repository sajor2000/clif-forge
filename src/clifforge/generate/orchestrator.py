"""U21 orchestrator — assemble a full multi-table synthetic CLIF 2.1 dataset.

The orchestrator owns everything the per-table generators deliberately do *not*:
the id scheme, patient<->hospitalization linking, admit-time spread across a
calendar, death propagation to ``patient.death_dttm`` (AE4), per-table conformance
gating, and reproducibility.

**Reproducibility (R22, AE6).** One ``SeedSequence(seed)`` is spawned into one
independent child stream per encounter (child ``i`` is stable regardless of
``n_patients``), so a given ``(seed, n_patients)`` produces identical output.
Within an encounter a single ``Generator`` is threaded through spine -> every
table in a fixed order, so the whole encounter is deterministic. Parquet output
is byte-identical **within a fixed environment**; across a polars/arrow upgrade
the frame *contents* stay identical (the honest invariant) while the on-disk
bytes may change.

**Encounter model.** One patient : one hospitalization (``P{i}`` / ``H{i}``). A
fitted encounters-per-patient distribution does not exist in the pack, so a 1:1
mapping is the honest default (R15) rather than an invented multi-encounter model.

**Gating (R25).** Every assembled table is run through the conformance gate; a
``ConformanceError`` propagates to the caller (the CLI turns it into a nonzero
exit). ``code_status`` is patient-level (keyed on ``patient_id``); every other
table hangs off ``hospitalization_id``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, enforce_numeric_ids
from clifforge.generate.spine import sample_spine, truth_frame
from clifforge.generate.tables.adt import adt_frame, sample_adt
from clifforge.generate.tables.code_status import code_status_frame, sample_code_status
from clifforge.generate.tables.crrt_therapy import crrt_therapy_frame, sample_crrt_therapy
from clifforge.generate.tables.ecmo_mcs import ecmo_mcs_frame, sample_ecmo_mcs
from clifforge.generate.tables.hospitalization import (
    hospitalization_frame,
    sample_hospitalization,
)
from clifforge.generate.tables.invasive_hemodynamics import (
    invasive_hemodynamics_frame,
    sample_invasive_hemodynamics,
)
from clifforge.generate.tables.key_icu_orders import key_icu_orders_frame, sample_key_icu_orders
from clifforge.generate.tables.labs import labs_frame, sample_labs
from clifforge.generate.tables.medication_admin_continuous import (
    medication_admin_continuous_frame,
    sample_medication_admin_continuous,
)
from clifforge.generate.tables.medication_admin_intermittent import (
    medication_admin_intermittent_frame,
    sample_medication_admin_intermittent,
)
from clifforge.generate.tables.microbiology_culture import (
    microbiology_culture_frame,
    sample_microbiology_culture,
)
from clifforge.generate.tables.patient import patient_frame, sample_patient
from clifforge.generate.tables.patient_assessments import (
    patient_assessments_frame,
    sample_patient_assessments,
)
from clifforge.generate.tables.position import position_frame, sample_position
from clifforge.generate.tables.provider import provider_frame, sample_provider
from clifforge.generate.tables.respiratory_support import (
    respiratory_support_frame,
    sample_respiratory_support,
)
from clifforge.generate.tables.therapy_details import sample_therapy_details, therapy_details_frame
from clifforge.generate.tables.transfusion import sample_transfusion, transfusion_frame
from clifforge.generate.tables.vitals import sample_vitals, vitals_frame

from clifforge.generate.filenames import table_parquet_filename, table_parquet_path

__all__ = ["TRUTH_FILENAME", "GeneratedDataset", "generate_dataset", "write_dataset"]

#: Filename for the retained latent spine. It deliberately does **not** carry the
#: ``clif_`` prefix: the spine is not a CLIF table, and every ``clif_*.parquet`` in
#: an output directory must be a real CLIF 2.1 table
#: (``clif_<table>_2.1_<beta|concept>.parquet``).
TRUTH_FILENAME = "_truth.parquet"

#: Admissions are spread across a two-year calendar at second resolution, so
#: timestamps are realistic and collisions stay vanishingly rare even at large n
#: (hour resolution gave only ~17.5k slots and collided within a few hundred stays).
_CALENDAR_START = datetime(2018, 1, 1, tzinfo=UTC)
_CALENDAR_SPAN_SECONDS = 730 * 24 * 3600

#: The spine-driven tables, as one list: name, sampler, frame assembler, and which
#: id the sampler is keyed on. `acc`, the per-encounter loop, and the frame
#: assembly all derive from this, so adding a table means editing exactly one
#: place instead of three hand-synced lists.
#:
#: **Order is load-bearing.** Each encounter threads a single Generator through
#: these samplers in sequence, so reordering changes every downstream draw and
#: breaks byte-reproducibility (R22/AE6) against previously generated datasets.
_TABLE_REGISTRY: tuple[tuple[str, Any, Any, str], ...] = (
    ("adt", sample_adt, adt_frame, "hospitalization_id"),
    ("vitals", sample_vitals, vitals_frame, "hospitalization_id"),
    ("labs", sample_labs, labs_frame, "hospitalization_id"),
    (
        "respiratory_support",
        sample_respiratory_support,
        respiratory_support_frame,
        "hospitalization_id",
    ),
    (
        "medication_admin_continuous",
        sample_medication_admin_continuous,
        medication_admin_continuous_frame,
        "hospitalization_id",
    ),
    (
        "medication_admin_intermittent",
        sample_medication_admin_intermittent,
        medication_admin_intermittent_frame,
        "hospitalization_id",
    ),
    (
        "patient_assessments",
        sample_patient_assessments,
        patient_assessments_frame,
        "hospitalization_id",
    ),
    ("position", sample_position, position_frame, "hospitalization_id"),
    (
        "microbiology_culture",
        sample_microbiology_culture,
        microbiology_culture_frame,
        "hospitalization_id",
    ),
    ("crrt_therapy", sample_crrt_therapy, crrt_therapy_frame, "hospitalization_id"),
    ("code_status", sample_code_status, code_status_frame, "patient_id"),
    ("ecmo_mcs", sample_ecmo_mcs, ecmo_mcs_frame, "hospitalization_id"),
    (
        "invasive_hemodynamics",
        sample_invasive_hemodynamics,
        invasive_hemodynamics_frame,
        "hospitalization_id",
    ),
    ("transfusion", sample_transfusion, transfusion_frame, "hospitalization_id"),
    ("key_icu_orders", sample_key_icu_orders, key_icu_orders_frame, "hospitalization_id"),
    ("therapy_details", sample_therapy_details, therapy_details_frame, "hospitalization_id"),
    ("provider", sample_provider, provider_frame, "hospitalization_id"),
)


@dataclass(frozen=True)
class GeneratedDataset:
    """A complete synthetic dataset: table name -> frame, plus the truth spine."""

    tables: dict[str, pl.DataFrame]
    truth: pl.DataFrame


def _admit_dttm(rng: np.random.Generator) -> datetime:
    return _CALENDAR_START + timedelta(seconds=int(rng.integers(0, _CALENDAR_SPAN_SECONDS)))


def generate_dataset(
    pack: ParamPack,
    *,
    n_patients: int,
    seed: int = 42,
    id_offset: int = 0,
) -> GeneratedDataset:
    """Generate and gate a full multi-table synthetic dataset (R22, R25, AE6).

    ``id_offset`` shifts the numeric suffix of the emitted ``P{i}`` / ``H{i}``
    identifiers, so a large cohort can be produced in independently-seeded chunks
    (each with its own ``seed`` and a disjoint ``id_offset`` range) without
    identifier collisions — the whole cohort need never be held in memory at once.
    """
    if n_patients <= 0:
        raise ValueError("n_patients must be a positive integer")
    if id_offset < 0:
        raise ValueError("id_offset must be non-negative")

    child_seeds = np.random.SeedSequence(seed).spawn(n_patients)
    tables, spines = _generate_frames(pack, child_seeds, id_offset)
    return GeneratedDataset(tables=tables, truth=enforce_numeric_ids(truth_frame(spines)))


def _generate_frames(
    pack: ParamPack, child_seeds: list[np.random.SeedSequence], id_start: int
) -> tuple[dict[str, pl.DataFrame], list[Any]]:
    """Build (and gate) every table for the encounters in ``child_seeds``.

    Encounter ``j`` uses ``child_seeds[j]`` and id ``id_start + j`` (``P{i}`` /
    ``H{i}``). Because ``child_seeds`` comes from ``SeedSequence(seed).spawn`` and
    child ``i`` is stable regardless of how the range is sliced, generating in one
    call or in batches with matching ``id_start`` offsets yields identical output —
    this is what lets :func:`generate_streaming` bound memory without changing a byte.
    """
    patients, patient_deaths, hospitalizations, spines = [], [], [], []
    acc: dict[str, list[Any]] = {name: [] for name, *_ in _TABLE_REGISTRY}

    for j, child in enumerate(child_seeds):
        rng = np.random.default_rng(child)
        pid, hid = f"P{id_start + j}", f"H{id_start + j}"
        admit = _admit_dttm(rng)

        spine = sample_spine(pack, rng, hospitalization_id=hid)
        spines.append(spine)
        patients.append(sample_patient(pack, rng, patient_id=pid))
        hosp = sample_hospitalization(
            spine, pack, rng, hospitalization_id=hid, patient_id=pid, admit_dttm=admit
        )
        hospitalizations.append(hosp)
        patient_deaths.append(hosp.death_dttm)  # AE4: death lands on the patient row

        for name, sample_fn, _frame_fn, id_kwarg in _TABLE_REGISTRY:
            ident = pid if id_kwarg == "patient_id" else hid
            acc[name] += sample_fn(spine, pack, rng, **{id_kwarg: ident}, admit_dttm=admit)

    patient_df = patient_frame(patients).with_columns(
        pl.Series("death_dttm", patient_deaths, dtype=UTC_DATETIME)
    )
    tables: dict[str, pl.DataFrame] = {
        "patient": patient_df,
        "hospitalization": hospitalization_frame(hospitalizations),
    }
    for name, _sample_fn, frame_fn, _id_kwarg in _TABLE_REGISTRY:
        tables[name] = frame_fn(acc[name])

    # Emit integer patient/hospitalization ids (Python/R/Stata-friendly), then gate.
    tables = {name: enforce_numeric_ids(frame) for name, frame in tables.items()}
    for name, frame in tables.items():
        gate.validate(frame, name, run_secondary=False)  # raises ConformanceError (R25)
    return tables, spines


def write_dataset(
    dataset: GeneratedDataset,
    out_dir: str | Path,
    *,
    write_truth: bool = True,
) -> list[Path]:
    """Write each CLIF table to ``clif_<table>_2.1_<maturity>.parquet`` (+ ``_truth.parquet``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in dataset.tables.items():
        path = table_parquet_path(out, name)
        frame.write_parquet(path)
        written.append(path)
    if write_truth:
        path = out / TRUTH_FILENAME
        dataset.truth.write_parquet(path)
        written.append(path)
    return written


def generate_streaming(
    pack: ParamPack,
    out_dir: str | Path,
    *,
    n_patients: int,
    seed: int = 42,
    id_offset: int = 0,
    chunk_size: int = 10_000,
    write_truth: bool = True,
) -> list[Path]:
    """Generate a large cohort with bounded memory, writing directly to ``out_dir``.

    Identical output to :func:`generate_dataset` + :func:`write_dataset` for the
    same ``(seed, n_patients, id_offset)`` — every encounter keeps its stable child
    seed and id regardless of ``chunk_size`` (see :func:`_generate_frames`) — but
    only ``chunk_size`` encounters are ever held in memory at once. Each batch is
    gated and written to a per-table part file; the parts are then streamed into one
    ``clif_<table>.parquet`` each and removed. ``chunk_size`` is the memory dial:
    smaller uses less RAM (and runs a touch slower). Returns the written paths.
    """
    if n_patients <= 0:
        raise ValueError("n_patients must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = out / "_parts"

    child_seeds = np.random.SeedSequence(seed).spawn(n_patients)
    table_names: list[str] = ["patient", "hospitalization", *[n for n, *_ in _TABLE_REGISTRY]]
    if write_truth:
        table_names.append("truth")

    n_chunks = (n_patients + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, n_patients)
        tables, spines = _generate_frames(pack, child_seeds[lo:hi], id_offset + lo)
        batch = (
            {**tables, "truth": enforce_numeric_ids(truth_frame(spines))} if write_truth else tables
        )
        for name, frame in batch.items():
            (parts / name).mkdir(parents=True, exist_ok=True)
            frame.write_parquet(parts / name / f"part_{c:05d}.parquet")

    written: list[Path] = []
    for name in table_names:
        part_files = sorted((parts / name).glob("part_*.parquet"))
        dest = out / (TRUTH_FILENAME if name == "truth" else table_parquet_filename(name))
        pl.scan_parquet(part_files).sink_parquet(dest)  # streamed concat, bounded memory
        written.append(dest)
    shutil.rmtree(parts, ignore_errors=True)
    return written
