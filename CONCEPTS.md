# Concepts

Shared domain vocabulary for this project — entities, named processes, and
status concepts with project-specific meaning. Seeded with core domain
vocabulary from the realism-lock learning, then accretes as ce-compound and
ce-compound-refresh process learnings; direct edits are fine. Glossary only,
not a spec or catch-all.

## Param pack

Versioned directory of fitted (or prior) table parameter blocks consumed by
generators. A shareable pack ships in-repo for CI; site packs may remain
DUA-local.

## Pack-prefer

Generator pattern: read fitted table params when present, else fall back to
documented dashboard priors. Fit writes blocks only when source tables exist
(hybrid).

## Stay prevalence

Fraction of hospitalizations with ≥1 row in a table. Used as a Bernoulli gate,
not as a Poisson intensity among positive stays.

## Intensity among positive stays

Expected event count among stays that already passed the stay-prevalence gate
(e.g. panels or transfusions per positive stay). After the gate, Poisson draws
must be zero-truncated so gated stays emit at least one event.

## Network-median recalibrate

Pure transform that pins consortium ICU rates (mortality, organ support, rare
ECMO stay prevalence, lab cadence, and related levers) onto a deep-copied pack
before generation or CI envelope checks.

## Realism envelope lock

Always-on CI check that generates a fixed-seed cohort from a recalibrated
shareable pack and asserts both published rate bands and pinned measured means
so widening floors alone cannot green-wash generator drift.

## Rare-support preset

Shipped teaching variant that elevates ECMO/CRRT rates for demonstration; not
the default network-median path.
