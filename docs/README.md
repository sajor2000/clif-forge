# docs/

Supporting documentation for CLIFForge. Start with the root
[`README.md`](../README.md) for what the project is and how to use it; the files
here go deeper.

| File | What it is | For whom |
|---|---|---|
| [`../PROVENANCE.md`](../PROVENANCE.md) | **Table catalog** — all 28 CLIF 2.1 tables + spine; fitted / prior / derived; fitted ICU realism changelog | Anyone auditing how a table was produced |
| [`../CONCEPTS.md`](../CONCEPTS.md) | Shared domain vocabulary (param pack, pack-prefer, stay prevalence, realism envelope) | Contributors reading code or solutions docs |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | The deterministic, seeded pipeline (real CLIF → pack → dataset), and how any artifact regenerates | Anyone reproducing or auditing a dataset |
| [`CONSORTIUM_ANNOUNCEMENT.md`](CONSORTIUM_ANNOUNCEMENT.md) | Copy-paste Slack + email blurbs announcing CLIFForge | The project owner, when sharing it |
| [`solutions/`](solutions) | Compounded learnings (e.g. stay-prevalence vs intensity, CI envelope locks) | Maintainers debugging realism / CI |
| [`plans/`](plans) | Historical design plans (founding generator; realism gaps; master + derivatives) | Contributors and maintainers |

## About `plans/`

These are **historical design records**, not user documentation — they capture the
decisions and requirements behind the build (methodology, conformance rules,
realism targets). They are kept for provenance and for contributors; a researcher
using the datasets never needs them. Prefer [`PROVENANCE.md`](../PROVENANCE.md)
for the current table map.

- `2026-07-23-001-…-generator-plan.md` — the founding design (empirical-fidelity fit-then-sample, conformance harness, table tiers).
- `2026-07-25-001-…-close-realism-gaps-plan.md` — tightening vitals, labs, terminal dynamics, and peak-acuity shape.
- `2026-07-25-002-…-master-dataset-and-derivatives-plan.md` — one master + shareable, always-distinct derivative generation.
