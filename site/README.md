# site/ — the CLIFForge website

Self-contained, dependency-free static pages for the project's public home
(planned: **clif-icu.com**). No build step, no external assets — every page
inlines its own CSS/JS and works opened directly in a browser.

| File | What it is |
|---|---|
| `index.html` | Landing page — the two-way pitch (use the data / pull the levers), why it's realistic, the levers (including presets / rare-support), and the CLIF-version roadmap. |
| `validation.html` | The synthetic-vs-real validation report — ICU and whole-hospital audits against real CLIF, with the deterioration-toward-death trajectory charts and a note on the always-on network-median CI envelope. |

## Hosting

Point a static host at this folder (or copy it to the web root). Options:

- **clif-icu.com** — the planned home; deploy this folder there when the domain is live.
- **GitHub Pages** — workflow deploys `site/` on pushes to `main`
  (https://sajor2000.github.io/clif-forge/).
- **Local** — just open `index.html` in a browser.

Both pages are light/dark theme-aware and responsive.
