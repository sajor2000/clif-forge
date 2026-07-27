# Contributing to CLIFForge

CLIFForge is built for the CLIF consortium, and contributions from member sites
are welcome — whether that's a bug report, a new preset, a calibration
improvement, or support for a new CLIF version.

## Ways to contribute

- **Report a problem or request a feature** — open a [GitHub issue](https://github.com/sajor2000/clif-forge/issues).
  For data-quality issues, please say which dataset/preset and CLIF version, and
  attach the `manifest.json` from the affected output.
- **Share a preset or spec** — if you've built a recipe that models a useful
  cohort (a specific case mix, an illness-rate profile), open a PR adding it under
  `presets/` with a one-line description.
- **Improve calibration or add a CLIF version** — see the notes below.

## Ground rules (important for a consortium tool)

- **Never commit real patient data.** No row-level records, no fitted pack derived
  from credentialed data, and nothing under a data-use agreement. Only the
  aggregate, non-derivable base pack and vendored CLIF reference files are
  committable. The fit stage runs locally against your own staged CLIF data and is
  not part of the public distribution.
- **Keep dependencies permissive.** MIT / BSD / Apache-2.0 only for runtime
  dependencies (heavier tooling goes in optional extras).
- **Everything must pass the conformance gate.** Generated tables are validated
  against CLIF schemas (categories, bounds, tz-aware datetimes, referential
  integrity) before anything is written.

## Development setup

```bash
git clone https://github.com/sajor2000/clif-forge.git
cd clif_synthetic_2.1
uv sync --extra dev          # or: pip install -e ".[dev,eval]"
uv run pytest -q             # run the test suite
uv run ruff check src tests  # lint
uv run ruff format --check src tests
uv run mypy src              # type-check
```

Please keep PRs green on `pytest`, `ruff`, and `mypy`, and add tests for
behavior-bearing changes.

## Adding a new CLIF version

The generator is schema-driven — tables, mCIDE categories, and outlier bounds
come from vendored CLIF reference files, not hard-coded logic. Supporting a new
version (e.g. CLIF 3.0) is a data/refit step: vendor the version's data dictionary
and mCIDE snapshot, refit the base pack against aggregate statistics for that
version, and record the target version in the output `manifest.json`. Open an
issue first so the work can be coordinated with the consortium.

## Questions

Open an issue or reach out through the CLIF consortium channels.
