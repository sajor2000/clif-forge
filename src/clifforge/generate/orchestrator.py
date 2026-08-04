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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from clifforge.conformance import gate
from clifforge.fit.param_pack import ParamPack
from clifforge.generate._common import UTC_DATETIME, enforce_numeric_ids
from clifforge.generate.filenames import (
    is_deliverable_table,
    table_parquet_filename,
    table_parquet_path,
)
from clifforge.generate.spine import sample_spine, truth_frame
from clifforge.generate.tables.adt import adt_frame, sample_adt
from clifforge.generate.tables.clinical_trial import clinical_trial_frame, sample_clinical_trial
from clifforge.generate.tables.code_status import code_status_frame, sample_code_status
from clifforge.generate.tables.crrt_therapy import crrt_therapy_frame, sample_crrt_therapy
from clifforge.generate.tables.ecmo_mcs import ecmo_mcs_frame, sample_ecmo_mcs
from clifforge.generate.tables.hospital_diagnosis import (
    hospital_diagnosis_frame,
    sample_hospital_diagnosis,
)
from clifforge.generate.tables.hospitalization import (
    hospitalization_frame,
    sample_hospitalization,
)
from clifforge.generate.tables.intake_output import intake_output_frame, sample_intake_output
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
from clifforge.generate.tables.medication_orders import (
    medication_orders_frame,
    sample_medication_orders,
)
from clifforge.generate.tables.microbiology_culture import (
    microbiology_culture_frame,
    sample_microbiology_culture,
)
from clifforge.generate.tables.microbiology_nonculture import (
    microbiology_nonculture_frame,
    sample_microbiology_nonculture,
)
from clifforge.generate.tables.microbiology_susceptibility import (
    microbiology_susceptibility_frame,
    sample_microbiology_susceptibility,
)
from clifforge.generate.tables.patient import patient_frame, sample_patient
from clifforge.generate.tables.patient_assessments import (
    patient_assessments_frame,
    sample_patient_assessments,
)
from clifforge.generate.tables.patient_diagnosis import (
    patient_diagnosis_frame,
    sample_patient_diagnosis,
)
from clifforge.generate.tables.patient_procedures import (
    patient_procedures_frame,
    sample_patient_procedures,
)
from clifforge.generate.tables.place_based_index import (
    place_based_index_frame,
    sample_place_based_index,
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

__all__ = ["TRUTH_FILENAME", "GeneratedDataset", "generate_dataset", "write_dataset"]

#: Historical / banned share-package name for a spine dump. Generators never write
#: this file; :mod:`scripts.audit_share_package` rejects it if present. Kept so
#: callers and tests can assert the forbidden name explicitly.
TRUTH_FILENAME = "_truth.parquet"

#: Admissions are spread across a two-year calendar at second resolution, so
#: timestamps are realistic and collisions stay vanishingly rare even at large n
#: (hour resolution gave only ~17.5k slots and collided within a few hundred stays).
_CALENDAR_START = datetime(2018, 1, 1, tzinfo=UTC)
_CALENDAR_SPAN_SECONDS = 730 * 24 * 3600

#: The spine-driven tables, as one list: name, sampler, frame assembler, and which
#: id(s) the sampler is keyed on (a single kwarg name, or a tuple for the canonical
#: tables that carry both ``patient_id`` and ``hospitalization_id``). `acc`, the
#: per-encounter loop, and the frame assembly all derive from this, so adding a
#: table means editing exactly one place instead of three hand-synced lists.
#:
#: **Order is load-bearing.** Each encounter threads a single Generator through
#: these samplers in sequence, so reordering changes every downstream draw and
#: breaks byte-reproducibility (R22/AE6) against previously generated datasets.
_TABLE_REGISTRY: tuple[tuple[str, Any, Any, str | tuple[str, ...]], ...] = (
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
        ("patient_id", "hospitalization_id"),
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
    (
        "hospital_diagnosis",
        sample_hospital_diagnosis,
        hospital_diagnosis_frame,
        "hospitalization_id",
    ),
    (
        "patient_diagnosis",
        sample_patient_diagnosis,
        patient_diagnosis_frame,
        ("patient_id", "hospitalization_id"),
    ),
    (
        "microbiology_nonculture",
        sample_microbiology_nonculture,
        microbiology_nonculture_frame,
        ("patient_id", "hospitalization_id"),
    ),
    ("intake_output", sample_intake_output, intake_output_frame, "hospitalization_id"),
    (
        "patient_procedures",
        sample_patient_procedures,
        patient_procedures_frame,
        "hospitalization_id",
    ),
    (
        "place_based_index",
        sample_place_based_index,
        place_based_index_frame,
        "hospitalization_id",
    ),
    ("clinical_trial", sample_clinical_trial, clinical_trial_frame, "hospitalization_id"),
)

#: Tables that hang off *another table's rows* rather than off the spine: name,
#: the parent table(s) they read, sampler, frame assembler.
#:
#: These exist because CLIF defines them that way, not as a shortcut. A
#: susceptibility result is a property of an isolate that a specific culture grew,
#: and a medication order is what the administrations were given under — both join
#: on an id the parent minted (``organism_id``, ``med_order_id``), so deriving
#: them is the only way the join can mean anything.
#:
#: This does not weaken the "generators read the spine and the pack, never another
#: table" rule: the sampler receives the parent's in-memory event objects for the
#: *same encounter*, never a generated frame, and still never reaches across
#: encounters or reads output files.
#:
#: **Order is load-bearing**, for the same reason as ``_TABLE_REGISTRY``: these run
#: after the spine-driven tables, in this order, on the same encounter Generator.
_DERIVED_REGISTRY: tuple[tuple[str, tuple[str, ...], Any, Any], ...] = (
    (
        "microbiology_susceptibility",
        ("microbiology_culture",),
        sample_microbiology_susceptibility,
        microbiology_susceptibility_frame,
    ),
    (
        "medication_orders",
        ("medication_admin_continuous", "medication_admin_intermittent"),
        sample_medication_orders,
        medication_orders_frame,
    ),
)


@dataclass(frozen=True)
class GeneratedDataset:
    """A complete synthetic dataset: table name -> frame, plus in-memory spine.

    ``truth`` is the stacked latent spine used during generation (not a CLIF
    table and not written to disk by :func:`write_dataset`).
    """

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
    acc.update({name: [] for name, *_ in _DERIVED_REGISTRY})

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

        ids = {"patient_id": pid, "hospitalization_id": hid}
        encounter: dict[str, list[Any]] = {}
        for name, sample_fn, _frame_fn, id_kwarg in _TABLE_REGISTRY:
            keys = (id_kwarg,) if isinstance(id_kwarg, str) else id_kwarg
            rows = sample_fn(spine, pack, rng, **{k: ids[k] for k in keys}, admit_dttm=admit)
            encounter[name] = rows
            acc[name] += rows

        for name, parents, sample_fn, _frame_fn in _DERIVED_REGISTRY:
            acc[name] += sample_fn(
                {parent: encounter[parent] for parent in parents},
                pack,
                rng,
                patient_id=pid,
                hospitalization_id=hid,
                admit_dttm=admit,
            )

    # AE4: death lands on the patient row. birth_date is derived from the
    # encounter's age_at_admission so the two tables cannot disagree (R15).
    birth_dates: list[date | None] = []
    for hosp in hospitalizations:
        if hosp.age_at_admission is None:
            birth_dates.append(None)
        else:
            birth_dates.append(
                (hosp.admission_dttm - timedelta(days=int(hosp.age_at_admission) * 365)).date()
            )
    patient_df = patient_frame(patients).with_columns(
        pl.Series("death_dttm", patient_deaths, dtype=UTC_DATETIME),
        pl.Series("birth_date", birth_dates, dtype=pl.Date),
    )
    tables: dict[str, pl.DataFrame] = {
        "patient": patient_df,
        "hospitalization": hospitalization_frame(hospitalizations),
    }
    for name, _sample_fn, frame_fn, _id_kwarg in _TABLE_REGISTRY:
        tables[name] = frame_fn(acc[name])
    for name, _parents, _sample_fn, frame_fn in _DERIVED_REGISTRY:
        tables[name] = frame_fn(acc[name])

    # Emit integer patient/hospitalization ids (Python/R/Stata-friendly), then gate.
    tables = {name: enforce_numeric_ids(frame) for name, frame in tables.items()}
    for name, frame in tables.items():
        gate.validate(frame, name, run_secondary=False)  # raises ConformanceError (R25)
    return tables, spines


def write_dataset(
    dataset: GeneratedDataset,
    out_dir: str | Path,
) -> list[Path]:
    """Write deliverable CLIF tables as ``clif_<table>_2.1_{beta|concept}.parquet``.

    Tables without a website beta/concept badge are generated in memory but
    omitted from disk. The latent spine is never written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in dataset.tables.items():
        if not is_deliverable_table(name):
            continue
        path = table_parquet_path(out, name)
        frame.write_parquet(path)
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
) -> list[Path]:
    """Generate a large cohort with bounded memory, writing directly to ``out_dir``.

    Identical output to :func:`generate_dataset` + :func:`write_dataset` for the
    same ``(seed, n_patients, id_offset)`` — every encounter keeps its stable child
    seed and id regardless of ``chunk_size`` (see :func:`_generate_frames`) — but
    only ``chunk_size`` encounters are ever held in memory at once. Each batch is
    gated and written to a per-table part file; the parts are then streamed into one
    ``clif_<table>_2.1_{beta|concept}.parquet`` each and removed. Untiered tables
    are skipped at write time. ``chunk_size`` is the memory dial: smaller uses less
    RAM (and runs a touch slower). Returns the written paths.
    """
    if n_patients <= 0:
        raise ValueError("n_patients must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = out / "_parts"

    child_seeds = np.random.SeedSequence(seed).spawn(n_patients)
    table_names: list[str] = [
        "patient",
        "hospitalization",
        *[n for n, *_ in _TABLE_REGISTRY if is_deliverable_table(n)],
        *[n for n, *_ in _DERIVED_REGISTRY if is_deliverable_table(n)],
    ]
    # patient + hospitalization are always beta; still filter for safety
    table_names = [n for n in table_names if is_deliverable_table(n)]

    n_chunks = (n_patients + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, n_patients)
        tables, _spines = _generate_frames(pack, child_seeds[lo:hi], id_offset + lo)
        for name, frame in tables.items():
            if not is_deliverable_table(name):
                continue
            (parts / name).mkdir(parents=True, exist_ok=True)
            frame.write_parquet(parts / name / f"part_{c:05d}.parquet")

    written: list[Path] = []
    for name in table_names:
        part_files = sorted((parts / name).glob("part_*.parquet"))
        dest = out / table_parquet_filename(name)
        pl.scan_parquet(part_files).sink_parquet(dest)  # streamed concat, bounded memory
        written.append(dest)
    shutil.rmtree(parts, ignore_errors=True)
    return written
