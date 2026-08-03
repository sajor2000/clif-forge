"""CLIFForge command-line interface.

Two subcommands:

* ``generate`` — sample a synthetic CLIF 2.1 dataset offline from a parameter
  pack (implemented in U21).
* ``fit`` — the one-time fit stage that builds a parameter pack over real
  real CLIF (implemented in U5).

At the scaffold stage both are argument-parsing stubs; they parse and validate
their flags but do not yet run a pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from clifforge import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``clif-forge`` argument parser (generate + fit)."""
    parser = argparse.ArgumentParser(
        prog="clif-forge",
        description="CLIFForge — generate fully synthetic CLIF 2.1 ICU datasets.",
    )
    parser.add_argument("--version", action="version", version=f"clif-forge {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{generate,init,ui,fit,presets}")

    generate = sub.add_parser(
        "generate", help="Generate a synthetic CLIF 2.1 dataset (offline, no real data)."
    )
    generate.add_argument(
        "--n-patients",
        type=int,
        default=None,
        help="Number of synthetic patients (overrides a spec's n).",
    )
    generate.add_argument(
        "--seed", type=int, default=None, help="Seed for byte-identical reproducible output."
    )
    generate.add_argument(
        "--out",
        default=None,
        help="Output directory for the generated dataset (required unless --preview).",
    )
    generate.add_argument(
        "--pack",
        default=None,
        help="Directory of a fitted parameter pack to sample from.",
    )
    generate.add_argument(
        "--demo",
        action="store_true",
        help="Use the built-in synthetic demo pack (no real data or fit required). "
        "Structurally valid CLIF 2.1 output for testing; not statistically calibrated.",
    )
    generate.add_argument(
        "--spec", default=None, help="A variant spec (TOML) describing a derivative to generate."
    )
    generate.add_argument(
        "--preset", default=None, help="A shipped preset variant name (see clifforge.variants)."
    )
    generate.add_argument(
        "--base-pack",
        default=None,
        help="Base pack a --spec/--preset derives from. Default: a local ./base_pack "
        "if present, else the shareable base pack shipped inside the package.",
    )
    generate.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Encounters generated per in-memory batch (the RAM dial). Cohorts larger "
        "than this stream to disk in batches so peak memory stays bounded; smaller "
        "values use less RAM on modest machines (a touch slower). Output is identical "
        "regardless of chunk size. Default 10000.",
    )
    generate.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Cap CPU threads (the compute dial). Fewer threads use less CPU on shared "
        "or low-core machines. Default: all available cores.",
    )
    generate.add_argument(
        "--preview",
        action="store_true",
        help="Dry run: print the expected cohort profile (mortality, length-of-stay, "
        "ventilation, organ support) from a small sample and exit without writing. "
        "Tune a recipe cheaply, then re-run without --preview to generate.",
    )
    generate.add_argument(
        "--write-truth",
        action="store_true",
        help="Also write generator-internal _truth.parquet (not a CLIF table). "
        "Omit for shareable Dropbox/repo packages.",
    )

    init = sub.add_parser(
        "init",
        help="Interactively build a variant recipe (TOML) — the no-guesswork way to "
        "start a custom cohort.",
    )
    init.add_argument(
        "--out",
        default=None,
        help="Path to write the recipe (default: <name>.toml in the current directory).",
    )

    fit = sub.add_parser(
        "fit", help="Fit a parameter pack over real CLIF (one-time, requires real data)."
    )
    fit.add_argument("--real-dir", required=True, help="Directory of real CLIF parquet files.")
    fit.add_argument(
        "--out", required=True, help="Output directory for the versioned parameter pack."
    )

    ui = sub.add_parser("ui", help="Launch the Cohort Designer web app (requires the 'ui' extra).")
    ui.add_argument(
        "--port", type=int, default=8501, help="Port for the Streamlit server (default 8501)."
    )

    sub.add_parser(
        "presets",
        help="List shipped variant preset names (usable with generate --preset).",
    )

    return parser


def _run_generate(args: argparse.Namespace) -> int:
    """Generate + gate + write a synthetic dataset; nonzero on any failure (R25)."""
    import dataclasses

    from clifforge.conformance.gate import ConformanceError
    from clifforge.demo import demo_pack
    from clifforge.fit.param_pack import ParamPack
    from clifforge.generate.orchestrator import (
        generate_dataset,
        generate_streaming,
        write_dataset,
    )
    from clifforge.manifest import write_manifest
    from clifforge.variants import (
        default_base_pack_path,
        load_preset,
        load_spec,
        spec_to_pack,
    )

    use_spec = args.spec is not None or args.preset is not None
    if not use_spec and not args.demo and args.pack is None:
        print(
            "clif-forge generate: provide --spec/--preset (a variant recipe), --pack <dir> "
            "(a fitted pack), or --demo (built-in synthetic pack, no real data required).",
            file=sys.stderr,
        )
        return 1

    spec = None
    try:
        # Resolve the pack + seed first (all that --preview needs).
        if use_spec:
            spec = load_spec(args.spec) if args.spec is not None else load_preset(args.preset)
            base = ParamPack.load(args.base_pack or default_base_pack_path())
            pack = spec_to_pack(spec, base)  # no real_dir -> demographic-override path
            seed = args.seed if args.seed is not None else spec.seed
        else:
            pack = demo_pack() if args.demo else ParamPack.load(args.pack)
            seed = args.seed if args.seed is not None else 42

        if getattr(args, "preview", False):
            # Dry run: sample a small cohort, print the expected profile, write nothing.
            from clifforge.preview import PREVIEW_SAMPLE, cohort_profile, format_profile

            ds = generate_dataset(pack, n_patients=PREVIEW_SAMPLE, seed=seed)
            print(f"Expected cohort profile ({PREVIEW_SAMPLE}-encounter sample of this recipe):")
            print(format_profile(cohort_profile(ds)))
            print("\nRe-run without --preview to generate the full dataset.")
            return 0

        # A real generation needs a cohort size and an output directory.
        if spec is not None:  # set exactly when --spec/--preset was given
            n_patients = args.n_patients if args.n_patients is not None else spec.n
        elif args.n_patients is None:
            print(
                "clif-forge generate: --n-patients is required without --spec/--preset.",
                file=sys.stderr,
            )
            return 1
        else:
            n_patients = args.n_patients
        if args.out is None:
            print("clif-forge generate: --out is required (unless --preview).", file=sys.stderr)
            return 1

        chunk_size = getattr(args, "chunk_size", 10_000)
        write_truth = bool(getattr(args, "write_truth", False))
        if n_patients > chunk_size:
            # Stream in bounded-memory batches (identical output, lower peak RAM).
            written = generate_streaming(
                pack,
                args.out,
                n_patients=n_patients,
                seed=seed,
                chunk_size=chunk_size,
                write_truth=write_truth,
            )
        else:
            written = write_dataset(
                generate_dataset(pack, n_patients=n_patients, seed=seed),
                args.out,
                write_truth=write_truth,
            )
        manifest_spec = dataclasses.asdict(spec) if spec is not None else "master"
        write_manifest(args.out, spec=manifest_spec, seed=seed)
    except ConformanceError as exc:
        print(f"clif-forge generate: conformance failure -> {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary: any failure is a clean nonzero exit
        # A malformed-but-loadable pack (KeyError), a version mismatch, or an
        # unwritable --out (OSError/FileExistsError) must report cleanly, not crash.
        print(f"clif-forge generate: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"clif-forge generate: wrote {len(written)} files to {args.out}")
    return 0


def _run_fit(args: argparse.Namespace) -> int:
    """Fit a parameter pack over real CLIF (U5)."""
    from clifforge.fit.run_fit import run_fit

    try:
        run_fit(args.real_dir, args.out)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: any failure is a clean nonzero exit
        print(f"clif-forge fit: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"clif-forge fit: wrote parameter pack to {args.out}")
    return 0


def _prompt(label: str, default: str) -> str:
    """Prompt with a shown default; blank input accepts the default."""
    reply = input(f"{label} [{default}]: ").strip()
    return reply or default


def _run_init(args: argparse.Namespace) -> int:
    """Interactively build a variant recipe (TOML) and write it to disk."""
    from pathlib import Path

    print("Build a CLIF cohort recipe. Press Enter to accept each [default].\n")
    try:
        name = _prompt("Dataset name", "my-cohort")
        mode = ""
        while mode not in ("icu", "full_hospital"):
            mode = _prompt("Population — icu or full_hospital", "icu").lower()
            if mode not in ("icu", "full_hospital"):
                print("  Please type 'icu' or 'full_hospital'.")
        n = _prompt("Size (encounters)", "5000")
        seed = _prompt("Seed", "2025")

        lines = [f'name = "{name}"', f'mode = "{mode}"', f"n = {int(n)}", f"seed = {int(seed)}"]

        if _prompt("Customize demographics? y/n", "n").lower().startswith("y"):
            age_shift = _prompt("  Age shift (years, +older / -younger)", "0")
            lines += ["", "[demographics]", f"age_shift = {float(age_shift)}"]

        if mode == "icu" and _prompt("Customize illness rates? y/n", "n").lower().startswith("y"):
            imv = _prompt("  Invasive ventilation (0-1)", "0.28")
            mortality = _prompt("  Mortality multiplier", "0.66")
            lines += ["", "[rates]", f"imv = {float(imv)}", f"mortality_scale = {float(mortality)}"]
    except (KeyboardInterrupt, EOFError):
        print("\ninit cancelled.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"clif-forge init: invalid number: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else Path(f"{name}.toml")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}. Preview it, then generate:")
    print(f"    clif-forge generate --spec {out} --preview")
    print(f"    clif-forge generate --spec {out} --out ./{name}")
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit Cohort Designer app."""
    import importlib.util
    import subprocess
    from importlib.resources import as_file, files

    if importlib.util.find_spec("streamlit") is None:
        print(
            "clif-forge ui: Streamlit is not installed. Install the UI extra:\n"
            '    pip install "clifforge[ui]"',
            file=sys.stderr,
        )
        return 1
    # Brand the app in the clif-icu.com palette (deep-teal CTA #0d5f59, ink text
    # #14242c, surface #f7f8f7) instead of Streamlit's default red, via env vars
    # so the theme travels with the launcher regardless of the working directory.
    # A complete palette is set so Streamlit does not derive a bad text color.
    # WCAG AA: white text on #0d5f59 is ~7.3:1; #14242c on #ffffff is ~15:1.
    env = {
        **os.environ,
        "STREAMLIT_THEME_BASE": "light",
        "STREAMLIT_THEME_PRIMARY_COLOR": "#0d5f59",
        "STREAMLIT_THEME_TEXT_COLOR": "#14242c",
        "STREAMLIT_THEME_BACKGROUND_COLOR": "#ffffff",
        "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR": "#f7f8f7",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    with as_file(files("clifforge.ui") / "cohort_designer.py") as app_path:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(args.port),
        ]
        return subprocess.call(cmd, env=env)


def _run_presets(_args: argparse.Namespace) -> int:
    """Print shipped preset names one per line."""
    from clifforge.variants import list_presets

    names = list_presets()
    if not names:
        print("clif-forge presets: no shipped presets found", file=sys.stderr)
        return 1
    for name in names:
        print(name)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        # Cap threads before polars/numpy are imported (they read the count at import).
        max_threads = getattr(args, "max_threads", None)
        if max_threads is not None:
            if max_threads < 1:
                print("clif-forge generate: --max-threads must be >= 1", file=sys.stderr)
                return 1
            os.environ["POLARS_MAX_THREADS"] = str(max_threads)
            os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
        return _run_generate(args)
    if args.command == "fit":
        return _run_fit(args)
    if args.command == "init":
        return _run_init(args)
    if args.command == "ui":
        return _run_ui(args)
    if args.command == "presets":
        return _run_presets(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
