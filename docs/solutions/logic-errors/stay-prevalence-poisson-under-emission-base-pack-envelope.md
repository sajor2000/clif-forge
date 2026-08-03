---
title: "Stay-prevalence gates need zero-truncated intensity and measured CI anchors"
date: 2026-08-03
category: logic-errors
module: clifforge.generate
problem_type: logic_error
component: testing_framework
symptoms:
  - "Fitted stay_prevalence Bernoulli gate followed by unrestricted Poisson under-emitted positive stays"
  - "Network-median ECMO flooded every L5 window when stay_prevalence was unset"
  - "Unconditional clinical-trial stay_prevalence after peak gate double-filtered enrollment"
  - "Widening network_median_envelope.json lo/hi alone could green-wash generator drift"
  - "Missing base_pack/ skipped the always-on realism gate instead of failing CI"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags:
  - stay-prevalence
  - poisson-truncation
  - base-pack
  - realism-envelope
  - network-median
  - ecmo
  - pack-prefer
  - ci-gate
---

# Stay-prevalence gates need zero-truncated intensity and measured CI anchors

## Problem

Hybrid pack-prefer fit for dashboard-prior CLIF tables plus always-on `base_pack`
CI envelope locks still allowed silent realism breakage: stay-level Bernoulli
gates paired with unrestricted Poisson intensity under-emitted events, rare
ECMO flooded without an explicit prevalence, and envelope floors alone could
pass while seed-99 means drifted.

## Symptoms

- Transfusion / microbiology positive stays gated in, then Poisson returned 0
- ECMO rows on nearly every high-acuity stay at n≈900 after recalibrate
- Clinical-trial enrollment far below fitted rates among ventilated stays
- ABG co-occurrence and other envelope asserts too weak or skippable
- Local/CI runs without `base_pack/` skipped the realism module

## What Didn't Work

- Treating `stay_prevalence` as both gate and Poisson λ
- Applying unconditional prevalence after structural acuity gates (double-gate)
- Leaving `ecmo_mcs` without `stay_prevalence` so every L5 interval emitted
- Publishing wider `lo`/`hi` bands without pinning measured seed-99 means
- Letting fitted order marginals replace the PT/OT evaluation→treat ladder
- Flagging all IMV time as inflammatory for WBC bumps

## Solution

**Gate then intensity.** After a stay-prevalence Bernoulli succeeds, draw
zero-truncated Poisson among positives (`max(1, n)`). Fit writes separate
intensity (`events_per_positive_stay` / `panels_per_stay = height / n_pos`).

**Conditional enrollment.** Clinical trial uses
`eligible_conditional_prevalence` after the peak≥3 gate — never unconditional
`stay_prevalence`.

**Rare-event recalibrate.** `recalibrate_to_network_median` always sets
`ecmo_mcs.stay_prevalence ≈ 0.0009` (override via `ecmo_stay` / `rare-support`).

**Dual-layer CI lock.** `tests/eval/test_base_pack_envelope.py` asserts bands
and `measured_*` anchors with slack; hard-fails in CI if `base_pack/` is missing;
ABG Jaccard floor 0.70 with no skip.

**Preserve clinical structure.** Key ICU orders keep PT/OT eval→treat;
WBC inflammation requires support≥4 **and** `resp_flag`; vitals hemo noise uses
`rng.spawn`; `clif-forge presets` lists shipped recipes.

Pending merge on branch `cursor/rename-truth-spine` as of this writing.

## Why This Works

Stay prevalence is incidence; intensity is burden among positives. Structural
eligibility must not be multiplied by an unconditional rate. Measured anchors
make envelope edits require a real re-measurement, so CI cannot green-wash
generator regressions by only widening floors.

## Prevention

- New gated tables: gate on prevalence, zero-truncate Poisson among positives
- Prefer `*_conditional_prevalence` naming after spine eligibility gates
- Rare interval emitters always get explicit `stay_prevalence` in recalibrate
- When editing `network_median_envelope.json`, refresh `measured` from seed-99
- Keep `test_base_pack_envelope.py` in default pytest; ship `base_pack/` in CI
- Fitted marginals only soft-weight within structural families (PT/OT ladder)

## Related Issues

- Tangential CI tooling: [mypy untyped-decorator optional-extra CI](../build-errors/mypy-untyped-decorator-optional-extra-ci.md) (low overlap)
