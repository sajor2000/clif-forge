# CLIFForge — consortium announcement (copy-paste ready)

Two versions below: a short Slack post and a longer email. Fill in the
`[full-datasets link]` once the masters are hosted (Dropbox / clif-icu.com).

---

## Slack / short post

**📦 CLIFForge — fully synthetic CLIF datasets you can actually share**

We built a generator for **fully synthetic, openly redistributable CLIF datasets** —
no credential, no DUA, safe to post publicly. It's for all the things a real
credentialed dataset can't do: ETL smoke-testing, CI fixtures, teaching, demos,
and model/agent development.

Two ways to use it:
1. **Grab the ready-made data** — realistic, CLIF 2.1-conformant samples are in the
   repo; full-size 85k-ICU and 365k whole-hospital masters at [full-datasets link].
2. **Make your own** — `pip install`, tweak a short recipe, and generate a *distinct*
   but still-realistic cohort. You control the levers: population shape (ICU vs.
   whole-hospital), size, demographics, illness rates, and compute footprint.

It's not random data — distributions, trajectories, and organ-support couplings are
fit to real aggregate CLIF statistics, so the output lands in the real statistical
region while being provably synthetic (no real record leaves the fit stage).

CLIF 2.1 today; the engine is built to add 3.0 and future versions.

👉 Repo: https://github.com/sajor2000/clif-forge
Feedback and presets welcome.

---

## Email / longer

**Subject: CLIFForge — shareable synthetic CLIF datasets (use ours, or generate your own)**

Hi all,

I'd like to share **CLIFForge**, a tool for generating **fully synthetic,
openly redistributable CLIF datasets**. The goal is to give the consortium data
that is safe to share publicly — no credential, no data-use agreement — for the
work a real credentialed dataset can't support: public ETL testing, CI fixtures,
teaching material, reproducible demos, and model or agent development.

**There are two ways to use it:**

1. **Use the ready-made datasets, as-is.** Realistic, CLIF 2.1-conformant samples
   (an ICU cohort and a whole-hospital population) are committed in the repo — clone
   and go. Full-size masters (an 85k-encounter ICU cohort and a 365k whole-hospital
   population) are available at [full-datasets link].

2. **Pull the levers and generate your own.** Each dataset is a *recipe* you can
   change. Install the package, edit a short TOML spec, and generate a cohort that is
   distinct from everyone else's but still looks like the real thing. You control the
   levers: population shape (ICU cohort vs. whole-hospital with realistic patient
   flow), cohort size, demographics (age, ethnicity, race mix), illness rates
   (ventilation, mortality, vasopressors, CRRT, proning), and compute footprint (it
   runs on an ordinary laptop; RAM and CPU are dials). No two recipes produce the same
   data, and every dataset is reproducible from its recipe.

**Why it looks real, not random.** CLIFForge fits distributions, couplings, and
patient trajectories to *aggregate* CLIF statistics, then samples offline from a
versioned parameter pack. Skewed lab distributions, realistic missingness,
length-of-stay tails, and organ-support couplings (vasopressors with hypotension,
sedation with ventilation) all track the real cohort — while remaining provably
synthetic: no row-level record leaves the fit stage, and no synthetic patient traces
back to a real one (verified with distance-to-closest-record and identifiability
metrics).

**Versions.** CLIF 2.1 is available today. The engine is schema-driven, so CLIF 3.0
and later versions can be added as a refit step — the aim is one place to pull
synthetic data for whichever CLIF version a study needs, eventually hosted at
clif-icu.com.

Everything — the method, the conformance guarantees, and a synthetic-vs-real
validation report — is documented in the repo:

👉 **https://github.com/sajor2000/clif-forge**

It's an expansion of the consortium's `synthetic_clif`; the main addition is
empirical fidelity (fitting to real aggregate statistics rather than hand-specified
priors). Feedback, issues, and preset contributions are very welcome.

Best,
J.C. Rojas
