---
title: "mypy untyped-decorator failure in CI from an optional-extra dependency"
date: 2026-07-28
category: build-errors
module: clifforge.ui
problem_type: build_error
component: tooling
symptoms:
  - "CI `mypy src/` fails with: error: Untyped decorator makes function \"_base_pack\" untyped [untyped-decorator]"
  - "Passes locally but fails only in the CI quality job"
  - "Only the @st.cache_resource / @st.cache_data decorated functions are flagged"
root_cause: incomplete_setup
resolution_type: config_change
severity: medium
tags: [mypy, ci, streamlit, optional-dependencies, strict-mode, type-checking]
---

# mypy untyped-decorator failure in CI from an optional-extra dependency

## Problem

A Streamlit UI module type-checked cleanly with `mypy` locally but failed in CI
with `untyped-decorator` errors on its `@st.cache_*`-decorated functions. The PR
merged with a red `quality` check because the repo has no branch protection, so
`main` briefly carried a failing CI run.

## Symptoms

- CI step `uv run mypy src/` exits 1 with:
  `error: Untyped decorator makes function "_base_pack" untyped [untyped-decorator]`
  (and the same for `_preview`).
- The exact same command passes locally.
- Only the two functions wrapped in `@st.cache_resource` / `@st.cache_data` are flagged;
  the rest of the module checks fine.

## What Didn't Work

- Assuming the local `mypy` run was representative — it was not. Local passed because
  the developer's venv had the optional `ui` extra installed (real `streamlit` types),
  masking the failure that only appears where `streamlit` is absent.
- The existing `ignore_missing_imports = true` override for `streamlit.*` did not help:
  it silences *missing-import* errors, but the decorators still resolve to `Any`, and
  strict mode rejects an `Any`-typed decorator separately.

## Solution

Root-cause the environment gap first: CI installs only `dev` + `eval`
(`uv sync --extra dev --extra eval`), **not** the `ui` extra, so `streamlit` is
absent in the type-check job and mypy treats it as `Any`. Under strict mode,
`disallow_untyped_decorators` then flags any `@Any`-typed decorator.

Reproduce CI exactly with the locked deps (not a fresh unpinned install, which can
pull a newer numpy and surface unrelated errors):

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/ci-locked
uv sync --extra dev --extra eval          # no `ui` extra — like CI
uv run python -c "import importlib.util; print(bool(importlib.util.find_spec('streamlit')))"  # -> False
uv run mypy src/                          # reproduces the CI failure
```

Fix: scope the one strict flag off for the optional UI module only, in the
repo-root `pyproject.toml`:

```toml
# The `ui` extra is not installed in the lint/type CI job, so streamlit resolves to
# Any there and its `@st.cache_*` decorators read as untyped under strict mode. The
# UI module is a thin front-end (not the generation core), so relax just that one
# strict check for it — the module still type-checks everywhere else.
[[tool.mypy.overrides]]
module = ["clifforge.ui.*"]
disallow_untyped_decorators = false
```

Verified against the CI-identical locked env: `uv run mypy src/` →
`Success: no issues found`. Fixed in PR #29.

## Why This Works

`untyped-decorator` fires only under `disallow_untyped_decorators` (part of mypy
`strict`). The decorator is "untyped" in CI purely because `streamlit` is unavailable
there and collapses to `Any`. The scoped override turns off *only* that one check for
`clifforge.ui.*`, so:

- In CI (streamlit absent) the `@st.cache_*` decorators no longer error.
- Locally (streamlit present) the override is a no-op — there is no untyped decorator to flag.
- Every other strict check on the UI module, and all strict checks elsewhere, stay on.

An alternative fix is to add `--extra ui` to the CI type-check install so the module
is checked against real `streamlit` types. That was rejected here to avoid pulling
the heavier UI dependency tree into every CI run; the scoped override keeps CI lean.

## Prevention

- **Type-check the same dependency set in CI that you claim to support locally.**
  When a module depends on an *optional extra*, either (a) install that extra in the
  type-check job, or (b) scope the affected strict flag off for that module. Don't
  rely on a developer venv that happens to have the extra installed.
- **Reproduce CI in a locked env before diagnosing**, e.g.
  `UV_PROJECT_ENVIRONMENT=/tmp/x uv sync --extra dev --extra eval && uv run mypy src/`.
  A fresh `uv pip install` resolves newer, unpinned versions and can surface unrelated
  errors that send you down the wrong path.
- **Know the distinction:** `ignore_missing_imports` silences *missing-import* errors;
  it does **not** prevent `untyped-decorator` (or other `Any`-propagation) errors from
  an absent dependency. They need separate handling.
- **Don't merge on a red check.** This repo has no branch protection, so a failing
  `quality` run reached `main`; enabling required status checks would have blocked it.

## Related Issues

- Fixed in PR #29 (sajor2000/clif-forge).
- Same `pyproject.toml` also carries the paired `ignore_missing_imports` override for
  `pandas.*` / `streamlit.*` — necessary but not sufficient on its own (see above).
