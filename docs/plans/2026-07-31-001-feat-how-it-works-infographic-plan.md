# How-It-Works Infographic — Design & Plan

**Date:** 2026-07-31
**Scope:** Add one visual "How it works" section to the landing page (`site/index.html`)
that explains the generation *method* and *how to use it*, using only the site's
existing design system. No new dependencies, no JS, web-only.

## Goal

The landing page explains realism and usage in prose, but has no diagram. The hero's
**"See how it works"** button currently jumps to *"Two ways to use it"* — a usage
section, not a how-it-works explainer. This slice adds a visual methods + usage
infographic and makes it the true target of that button.

## Design (approved)

A single new `<section id="how">` inserted **immediately after the hero**, titled
**"How it works"**, built from the existing CSS primitives (`.wrap`, card tokens,
`--teal`/`--amber`, mono section labels, serif headings, light/dark variables).

### Part A — The method as a pipeline (centerpiece)

Left→right flow of four stages joined by connector arrows; stacks vertically on
mobile. Copy is drawn only from the existing README/site (no new clinical claims):

1. **Real CLIF aggregates** — marginals, couplings, and state transitions fitted from
   *aggregate* statistics only; never row-level records.
2. **Parameter pack** — a versioned, shareable pack (~84 KB); no real data inside.
3. **Latent acuity spine** — each synthetic stay follows an internal acuity
   trajectory, keeping vitals / labs / organ support coupled (vasopressors ↔ hypotension).
4. **Sampled CLIF tables** — offline sampling emits 19 CLIF 2.1-conformant Parquet
   tables; provably synthetic.

Two phase brackets under the arrows: **fit** (stages 1→2) and **sample, offline**
(stages 2→4). A footnote line: *privacy by construction — no synthetic patient
traces back to a real one.*

### Part B — How to use it (3-step ribbon)

Thin numbered strip (no code blocks — those already live in "Two ways"):
**1 Choose** ready-made data or a TOML recipe → **2 Generate / grab**
(`clif-forge generate …` or `git clone`) → **3 Inspect & validate** (Parquet per
table + the synthetic-vs-real report, linking `validation.html`).

### Placement & hero link

- Insert `#how` as the first `<section>` after `</header>` (the hero).
- Repoint the hero button `href="#two-ways"` → `href="#how"`.
- Resulting narrative: **see the method → two ways to use it → why it's realistic →
  levers → stats**.

## Fidelity requirements

- **Full polish:** connector arrows via inline SVG that recolor for light/dark via
  `currentColor` / CSS variables; static (no animation), so nothing to disable under
  `prefers-reduced-motion`.
- Responsive: the 4-stage pipeline and 3-step ribbon collapse to single-column
  under the existing `max-width: 760px` breakpoint.
- Accessible: real heading hierarchy (`h2` section, `h3` stages), SVG connectors
  marked `aria-hidden`, sufficient contrast in both themes.

## Non-goals / guardrails

- No changes to any other section; no new JS; no external assets/fonts.
- No new clinical or statistical claims — copy paraphrases existing site/README text.
- The Pages deploy (`.github/workflows/pages.yml`, `paths: site/**`) is unchanged;
  this doc under `docs/` does not affect deploy.

## Verification

1. Open `site/index.html` in a browser; confirm the new section renders after the
   hero in both light and dark (`prefers-color-scheme` + `data-theme` overrides).
2. Confirm the hero "See how it works" button scrolls to `#how`.
3. Narrow the viewport below 760px; confirm pipeline and ribbon stack cleanly.
4. HTML validity: no unclosed tags; existing sections unchanged (diff is additive
   plus the one `href` edit).
