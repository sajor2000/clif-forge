# Presets — example CLIF cohort recipes

Each `.toml` here is a small **variant recipe**: a starting point you can generate
from directly, or copy and tweak. Every field defaults to the network-median
master, so a preset only lists what it changes. Generate one with:

```bash
clif-forge generate --preset high-acuity --n-patients 5000 --out ./my-dataset
# or preview the expected cohort first (writes nothing):
clif-forge generate --preset high-acuity --preview
```

| Preset | What it models | Changes from the master |
|---|---|---|
| `high-acuity` | A sicker ICU cohort | invasive ventilation 0.55, vasopressors 0.45, mortality ×1.4 |
| `older-cohort` | An older population | age shifted +10 years |
| `sepsis-heavy` | A sepsis-weighted cohort | vasopressors 0.50, CRRT 0.06, mortality ×1.3 |
| `rare-support` | Teaching rare events | ECMO stay 0.08, CRRT|renal 0.55, IMV 0.50 — **not** network rates |

To build your own, start from any preset (or a blank spec) and change the
`[demographics]` / `[rates]` fields — see **What's tweakable** in the root
[`README.md`](../README.md#whats-tweakable), or run `clif-forge init` for an
interactive builder.
