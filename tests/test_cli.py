"""Tests for the CLI + U21 orchestrator (R22, R23, R25, AE6).

The scaffold parser tests (U1) plus the end-to-end pipeline: a fully
self-contained synthetic parameter pack (no real data) drives spine -> all 28 tables
-> gate -> parquet, so CI exercises seeded determinism / byte-identical output
(AE6), CLIF ``--out`` naming (R23), nonzero exit on any validation failure (R25),
and the ``fit`` subcommand wiring.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from clifforge.cli import build_parser, main
from clifforge.conformance.gate import ConformanceError
from clifforge.fit.param_pack import ParamPack
from clifforge.generate.filenames import parse_table_from_stem, table_parquet_filename
from clifforge.generate.orchestrator import (
    TRUTH_FILENAME,
    GeneratedDataset,
    generate_dataset,
    generate_streaming,
    write_dataset,
)
from clifforge.reference import loader


# --- U1 scaffold parser tests ------------------------------------------------ #
def test_parser_program_name() -> None:
    assert build_parser().prog == "clif-forge"


def test_no_command_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "generate" in out and "fit" in out


def test_presets_lists_shipped_names(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["presets"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert "high-acuity" in out
    assert "rare-support" in out


def test_generate_requires_out_unless_preview(capsys: pytest.CaptureFixture[str]) -> None:
    # --out is optional at the parser level (so --preview can dry-run), but a real
    # generation still requires it.
    rc = main(["generate", "--demo", "--n-patients", "3"])
    assert rc == 1
    assert "--out" in capsys.readouterr().err


def test_generate_parses_flags() -> None:
    args = build_parser().parse_args(
        ["generate", "--n-patients", "100", "--seed", "42", "--out", "./out"]
    )
    assert args.command == "generate"
    assert args.n_patients == 100
    assert args.seed == 42
    assert args.pack is None and args.demo is False  # no pack by default; opt in via --pack/--demo


def test_generate_demo_works_without_a_pack(tmp_path) -> None:
    # A fresh clone has no fitted pack; --demo must produce a conformant dataset
    # out of the box (the external self-service path).
    out = tmp_path / "demo_out"
    rc = main(["generate", "--n-patients", "8", "--seed", "1", "--demo", "--out", str(out)])
    assert rc == 0
    assert (out / table_parquet_filename("hospitalization")).exists()
    assert (out / table_parquet_filename("vitals")).exists()


def test_generate_without_pack_or_demo_errors(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["generate", "--n-patients", "5", "--out", str(tmp_path / "x")])
    assert rc == 1
    assert "--demo" in capsys.readouterr().err


def test_generate_preview_prints_profile_and_writes_nothing(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    # --preview is a dry run: it prints the expected cohort profile and writes no
    # files (it returns before the --out check, so no --out is needed).
    rc = main(["generate", "--preset", "high-acuity", "--preview"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "Expected cohort profile" in printed
    assert "mortality" in printed
    assert "reached ICU" in printed


def test_init_writes_a_generatable_recipe(tmp_path, monkeypatch) -> None:
    # The init wizard writes a TOML that `generate --spec` can actually load and run.
    answers = iter(["study", "icu", "600", "9", "n", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    recipe = tmp_path / "study.toml"
    assert main(["init", "--out", str(recipe)]) == 0
    assert recipe.exists()

    out = tmp_path / "ds"
    rc = main(["generate", "--spec", str(recipe), "--n-patients", "6", "--out", str(out)])
    assert rc == 0
    assert (out / table_parquet_filename("hospitalization")).exists()


def test_ui_command_parses_with_default_port() -> None:
    args = build_parser().parse_args(["ui"])
    assert args.command == "ui"
    assert args.port == 8501


def test_ui_errors_cleanly_when_streamlit_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The UI is an optional extra; without it, `clif-forge ui` must give a clear
    # install hint and a nonzero exit rather than a traceback.
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    rc = main(["ui"])
    assert rc == 1
    assert "clifforge[ui]" in capsys.readouterr().err


def test_cohort_profile_keys_and_formatting(pack: ParamPack) -> None:
    from clifforge.preview import cohort_profile, format_profile

    ds = generate_dataset(pack, n_patients=20, seed=1)
    prof = cohort_profile(ds)
    assert set(prof) >= {"n", "los_median_h", "imv", "icu", "vaso", "crrt"}
    assert prof["n"] == 20
    assert all(0.0 <= prof[k] <= 1.0 for k in ("imv", "icu", "vaso", "crrt"))
    text = format_profile(prof)
    assert "in-hospital mortality" in text and "reached ICU" in text


def test_rng_fixture_is_seed_reproducible(rng: np.random.Generator, seed: int) -> None:
    first = rng.integers(0, 1_000_000, size=5)
    fresh = np.random.Generator(np.random.PCG64(seed))
    assert (first == fresh.integers(0, 1_000_000, size=5)).all()


# --- orchestrator ------------------------------------------------------------ #
def test_generate_dataset_produces_every_canonical_table(pack: ParamPack) -> None:
    """Every table the canonical CLIF 2.1 DDL defines must be emitted — no more, no less.

    Asserted against the vendored dictionary rather than a literal count, so
    adding a table upstream fails here instead of silently going unemitted.
    """
    ds = generate_dataset(pack, n_patients=12, seed=1)
    assert isinstance(ds, GeneratedDataset)
    assert set(ds.tables) == set(loader.dictionary_tables())
    assert ds.truth.height > 0


def test_generate_dataset_is_deterministic(pack: ParamPack) -> None:
    a = generate_dataset(pack, n_patients=10, seed=7)
    b = generate_dataset(pack, n_patients=10, seed=7)
    for name in a.tables:
        assert a.tables[name].equals(b.tables[name]), f"{name} differs across identical seeds"
    assert a.truth.equals(b.truth)


def test_id_offset_shifts_identifiers_for_chunked_generation(pack: ParamPack) -> None:
    # Chunked large-cohort generation relies on id_offset producing correctly
    # shifted, collision-free identifiers, and staying reproducible per chunk.
    base = generate_dataset(pack, n_patients=5, seed=3)
    shifted = generate_dataset(pack, n_patients=5, seed=3, id_offset=100)
    # Ids are emitted as 1-based integers (analyst-friendly, no leading zeros).
    assert base.tables["hospitalization"]["hospitalization_id"].to_list() == [
        i + 1 for i in range(5)
    ]
    assert shifted.tables["hospitalization"]["hospitalization_id"].to_list() == [
        100 + i + 1 for i in range(5)
    ]
    # Disjoint id ranges across chunks (no collisions when concatenated).
    base_ids = set(base.tables["hospitalization"]["hospitalization_id"].to_list())
    shifted_ids = set(shifted.tables["hospitalization"]["hospitalization_id"].to_list())
    assert base_ids.isdisjoint(shifted_ids)
    # Reproducible: identical (seed, id_offset) reproduces the chunk byte-for-byte.
    again = generate_dataset(pack, n_patients=5, seed=3, id_offset=100)
    assert shifted.tables["hospitalization"].equals(again.tables["hospitalization"])


def test_id_offset_rejects_negative(pack: ParamPack) -> None:
    with pytest.raises(ValueError, match="id_offset"):
        generate_dataset(pack, n_patients=3, seed=1, id_offset=-1)


def test_streaming_matches_single_call_regardless_of_chunk_size(pack: ParamPack, tmp_path) -> None:
    # The RAM dial must not change a byte: streaming in batches of any size must
    # produce the same data as one in-memory generate_dataset + write_dataset.
    single = tmp_path / "single"
    write_dataset(generate_dataset(pack, n_patients=50, seed=9), single)
    for chunk in (7, 20, 1000):  # smaller than, straddling, and larger than n
        streamed = tmp_path / f"stream_{chunk}"
        generate_streaming(pack, streamed, n_patients=50, seed=9, chunk_size=chunk)
        for f in sorted(single.glob("clif_*.parquet")):
            a = pl.read_parquet(f).sort(pl.all())
            b = pl.read_parquet(streamed / f.name).sort(pl.all())
            assert a.equals(b), f"{f.name} differs at chunk_size={chunk}"


def test_streaming_leaves_no_parts_dir(pack: ParamPack, tmp_path) -> None:
    out = tmp_path / "out"
    generate_streaming(pack, out, n_patients=30, seed=1, chunk_size=8)
    assert not (out / "_parts").exists()  # intermediate parts cleaned up
    assert (out / table_parquet_filename("hospitalization")).exists()


def test_streaming_rejects_bad_chunk_size(pack: ParamPack, tmp_path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        generate_streaming(pack, tmp_path / "o", n_patients=5, seed=1, chunk_size=0)


def test_cli_large_cohort_streams_and_caps_threads(tmp_path, monkeypatch) -> None:
    # A cohort larger than --chunk-size routes through streaming; --max-threads is
    # applied to the environment before the heavy imports.
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    out = tmp_path / "ds"
    rc = main(
        [
            "generate",
            "--demo",
            "--n-patients",
            "40",
            "--seed",
            "2",
            "--chunk-size",
            "10",
            "--max-threads",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert os.environ["POLARS_MAX_THREADS"] == "2"
    hosp = pl.read_parquet(out / table_parquet_filename("hospitalization"))
    assert hosp["hospitalization_id"].n_unique() == 40  # all encounters written


def test_seed_actually_changes_output(pack: ParamPack) -> None:
    # Guards against the seed being ignored (e.g. a hardcoded SeedSequence(42)) —
    # the same-seed determinism tests would still pass in that regression.
    a = generate_dataset(pack, n_patients=10, seed=1)
    b = generate_dataset(pack, n_patients=10, seed=2)
    assert not a.truth.equals(b.truth)
    assert not a.tables["hospitalization"].equals(b.tables["hospitalization"])


def test_first_encounters_stable_across_n_patients(pack: ParamPack) -> None:
    # SeedSequence.spawn(n) assigns child i a stable key regardless of n, so the
    # first k encounters must be identical whether we ask for k or more — the
    # property that would make generation safely resumable/extendable.
    small = generate_dataset(pack, n_patients=5, seed=9)
    large = generate_dataset(pack, n_patients=20, seed=9)
    first5 = set(small.tables["patient"]["patient_id"].to_list())  # the first 5 int patient ids
    small_p = small.tables["patient"].sort("patient_id")
    large_p = large.tables["patient"].filter(pl.col("patient_id").is_in(first5)).sort("patient_id")
    assert small_p.equals(large_p)


def test_ae4_death_propagates_to_patient(pack: ParamPack) -> None:
    ds = generate_dataset(pack, n_patients=60, seed=3)
    deaths = ds.tables["patient"].filter(pl.col("death_dttm").is_not_null()).height
    expired = ds.tables["hospitalization"].filter(pl.col("discharge_category") == "Expired").height
    assert deaths == expired  # AE4: every expired encounter marks its patient row


def test_zero_orphans_across_all_tables(pack: ParamPack) -> None:
    ds = generate_dataset(pack, n_patients=40, seed=5)
    hosp_ids = set(ds.tables["hospitalization"]["hospitalization_id"].to_list())
    patient_ids = set(ds.tables["patient"]["patient_id"].to_list())
    assert set(ds.tables["hospitalization"]["patient_id"].to_list()) <= patient_ids
    for name, frame in ds.tables.items():
        if name != "hospitalization" and "hospitalization_id" in frame.columns:
            assert set(frame["hospitalization_id"].to_list()) <= hosp_ids, f"orphan in {name}"
    # code_status is the one table keyed on patient_id, not hospitalization_id.
    code_status = ds.tables["code_status"]
    assert "patient_id" in code_status.columns
    assert set(code_status["patient_id"].to_list()) <= patient_ids, "orphan in code_status"


def test_high_acuity_tables_are_actually_populated(pack: ParamPack) -> None:
    # ecmo/crrt/hemodynamics fire only at rare high-acuity states; assert they
    # are non-empty at n large enough to reach those states, so their row-build
    # logic is exercised rather than passing vacuously on empty frames.
    ds = generate_dataset(pack, n_patients=200, seed=11)
    for name in ("ecmo_mcs", "crrt_therapy", "invasive_hemodynamics", "transfusion"):
        assert ds.tables[name].height > 0, f"{name} never populated — coupling logic untested"


def test_n_patients_must_be_positive(pack: ParamPack) -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_dataset(pack, n_patients=0, seed=1)


def test_conformance_failure_is_detected(pack: ParamPack) -> None:
    from clifforge.conformance import gate

    ds = generate_dataset(pack, n_patients=5, seed=1)
    corrupt = ds.tables["patient"].with_columns(pl.lit("NotAnMcideRace").alias("race_category"))
    with pytest.raises(ConformanceError):
        gate.validate(corrupt, "patient", run_secondary=False)


def test_write_dataset_omits_spine(pack: ParamPack, tmp_path) -> None:
    from clifforge.generate.filenames import deliverable_tables

    ds = generate_dataset(pack, n_patients=4, seed=1)
    written = write_dataset(ds, tmp_path)
    assert not (tmp_path / "_truth.parquet").exists()
    assert len(written) == len(deliverable_tables())
    assert all(p.suffix == ".parquet" for p in written)
    assert all("_2.1_beta.parquet" in p.name or "_2.1_concept.parquet" in p.name for p in written)
    assert not any("untiered" in p.name for p in written)


# --- CLI end-to-end ---------------------------------------------------------- #
def test_cli_generate_writes_clif_layout(pack: ParamPack, tmp_path) -> None:
    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out = tmp_path / "out"
    code = main(
        [
            "generate",
            "--n-patients",
            "8",
            "--seed",
            "42",
            "--out",
            str(out),
            "--pack",
            str(pack_dir),
        ]
    )
    assert code == 0
    assert (out / table_parquet_filename("patient")).exists()
    assert (out / table_parquet_filename("hospitalization")).exists()
    assert not (out / "_truth.parquet").exists()
    written = {p.name for p in out.glob("clif_*.parquet")}
    assert table_parquet_filename("vitals") in written
    assert table_parquet_filename("provider") in written


def test_clif_prefix_is_reserved_for_real_clif_tables(pack: ParamPack, tmp_path) -> None:
    """Nothing outside the CLIF 2.1.0 dictionary may claim a ``clif_`` filename.

    The latent spine stays in memory only — never ``clif_truth`` or ``_truth``.
    """
    write_dataset(generate_dataset(pack, n_patients=4, seed=1), tmp_path)
    emitted = {parse_table_from_stem(p.stem) for p in tmp_path.glob("clif_*.parquet")}
    assert None not in emitted
    assert emitted <= set(loader.dictionary_tables())
    # Beta and concept tables both appear, under their maturity-tagged stems.
    stems = {p.stem for p in tmp_path.glob("clif_*.parquet")}
    assert table_parquet_filename("vitals").removesuffix(".parquet") in stems
    assert table_parquet_filename("provider").removesuffix(".parquet") in stems
    assert not (tmp_path / TRUTH_FILENAME).exists()
    assert not (tmp_path / "clif_truth.parquet").exists()
    assert not any("untiered" in p.name for p in tmp_path.glob("*.parquet"))


def test_cli_ae6_two_runs_byte_identical(pack: ParamPack, tmp_path) -> None:
    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    args = ["generate", "--n-patients", "10", "--seed", "99", "--pack", str(pack_dir), "--out"]
    assert main([*args, str(out_a)]) == 0
    assert main([*args, str(out_b)]) == 0
    for pa in sorted(out_a.glob("clif_*.parquet")):
        assert (out_b / pa.name).read_bytes() == pa.read_bytes(), f"{pa.name} not byte-identical"


def _null_id_wrapper(frame_fn, id_col):
    """Wrap a frame assembler so it nulls a required id column (a real gate failure)."""

    def corrupt(records):
        return frame_fn(records).with_columns(pl.lit(None, dtype=pl.String).alias(id_col))

    return corrupt


@pytest.mark.parametrize(
    ("table", "id_col"),
    [
        ("patient", "patient_id"),  # assembled directly by the orchestrator
        # Nulled on a late registry table so a dropped gate call downstream can't
        # hide behind patient's own failure. It must be a *backbone* key: canonical
        # 2.1 leaves provider_id, device_id, med_order_id and friends nullable, so
        # nulling one of those is valid CLIF and correctly does not trip the gate.
        ("provider", "hospitalization_id"),
    ],
)
def test_cli_generate_nonzero_on_conformance_failure(
    monkeypatch, tmp_path, pack, table, id_col
) -> None:
    # Corrupt a table's assembler (null a required id column) so the real gate
    # rejects it; the CLI must surface that as a nonzero exit (R25), not bad data.
    # Parametrized over an early and a late table so a dropped gate call for a
    # downstream table can't hide behind patient's own failure.
    import clifforge.generate.orchestrator as orch

    if table == "patient":
        monkeypatch.setattr(orch, "patient_frame", _null_id_wrapper(orch.patient_frame, id_col))
    else:
        # _TABLE_REGISTRY captures frame functions at import, so patching the
        # module attribute would not reach it — patch the registry entry itself.
        patched = tuple(
            (
                name,
                sample_fn,
                _null_id_wrapper(frame_fn, id_col) if name == table else frame_fn,
                key,
            )
            for name, sample_fn, frame_fn, key in orch._TABLE_REGISTRY
        )
        monkeypatch.setattr(orch, "_TABLE_REGISTRY", patched)

    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out = tmp_path / "o"
    code = main(["generate", "--n-patients", "5", "--out", str(out), "--pack", str(pack_dir)])
    assert code == 1  # R25: any validation failure -> nonzero exit
    assert not out.exists()  # gate precedes write: no partial output for any table


def test_cli_generate_nonzero_on_missing_pack(tmp_path) -> None:
    code = main(
        [
            "generate",
            "--n-patients",
            "3",
            "--out",
            str(tmp_path / "o"),
            "--pack",
            str(tmp_path / "does_not_exist"),
        ]
    )
    assert code == 1


def test_cli_fit_invokes_run_fit(monkeypatch, tmp_path) -> None:
    called = {}

    def _fake_run_fit(real_dir, out_dir, **_k):
        called["real_dir"] = str(real_dir)
        called["out_dir"] = str(out_dir)

    monkeypatch.setattr("clifforge.fit.run_fit.run_fit", _fake_run_fit)
    code = main(["fit", "--real-dir", str(tmp_path / "real"), "--out", str(tmp_path / "pack")])
    assert code == 0
    assert called["real_dir"].endswith("real") and called["out_dir"].endswith("pack")


def test_cli_fit_nonzero_on_run_fit_error(monkeypatch, tmp_path) -> None:
    def _boom(*_a, **_k):
        raise FileNotFoundError("real-dir not found")

    monkeypatch.setattr("clifforge.fit.run_fit.run_fit", _boom)
    code = main(["fit", "--real-dir", str(tmp_path / "real"), "--out", str(tmp_path / "pack")])
    assert code == 1  # fit's error path mirrors generate's R25-style clean exit


def test_cli_generate_clean_exit_when_out_is_a_file(pack: ParamPack, tmp_path) -> None:
    # --out pointing at an existing regular file makes mkdir raise FileExistsError;
    # the CLI must report it cleanly (nonzero), not crash with a traceback.
    pack_dir = tmp_path / "pack"
    pack.write(pack_dir)
    out_file = tmp_path / "out_is_a_file"
    out_file.write_text("i am not a directory")
    code = main(["generate", "--n-patients", "3", "--out", str(out_file), "--pack", str(pack_dir)])
    assert code == 1


def test_generate_from_spec_writes_dataset_and_manifest(tmp_path, pack: ParamPack) -> None:
    # A variant spec generates against a base pack (here the demo pack on disk) and
    # writes a manifest — the no-credential derivative path.
    base_dir = tmp_path / "base"
    pack.write(base_dir)
    spec = tmp_path / "v.toml"
    spec.write_text('name = "cli-variant"\n[rates]\ncrrt_prob = 0.4\n', encoding="utf-8")
    out = tmp_path / "out"
    rc = main(
        [
            "generate",
            "--spec",
            str(spec),
            "--base-pack",
            str(base_dir),
            "--n-patients",
            "6",
            "--seed",
            "3",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / table_parquet_filename("hospitalization")).exists()
    assert (out / "manifest.json").exists()


def test_generate_bad_spec_path_exits_nonzero(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "generate",
            "--spec",
            str(tmp_path / "missing.toml"),
            "--n-patients",
            "5",
            "--out",
            str(tmp_path / "x"),
        ]
    )
    assert rc == 1
