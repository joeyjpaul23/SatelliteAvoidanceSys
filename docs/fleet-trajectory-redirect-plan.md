# Project Redirect: Fleet-Wide, Fuel-Optimal Trajectory Coordination

Written 2026-08-27. Captures the pivot from "predict collision risk for one conjunction event"
(the Kelvins benchmark work now in `archive/kelvins-agent-b/`) toward "given risk data across a whole constellation,
compute the fuel-optimal set of trajectory adjustments for the fleet." This doc is the plan, not
the implementation — treat it as the thing to argue with before code gets written.

## What's changing, what's not

The CDM/TCA/Pc work in `archive/kelvins-agent-b/methods/` (naive, GBM, Bayesian LSTM, probabilistic programming,
SGP4-residual hybrid, physics-informed GAN, transformer) and the Kessler integration
(`docs/kessler-report-and-integration-plan.md`) don't get thrown away — they become the **risk
layer**: the thing that answers "what's the Pc and TCA for this pair of objects." What's new is a
layer on top that takes *many* simultaneous risk estimates across a fleet and outputs maneuver
recommendations optimized jointly for fuel and orbital "space" (not creating new conjunctions,
respecting station-keeping boxes), rather than one operator reacting to one CDM at a time.

This is also a more literal fit for the project's original name than the current benchmark work is.
The thesis in `docs/PINN-GNN_Satellite_Conjunction_Thesis.pdf` is about physics-informed GNNs; a
fleet of satellites with pairwise conjunction risk is naturally a graph (nodes = satellites, edges =
active conjunctions weighted by Pc/TCA), and that graph changes shape as conjunctions appear and
resolve. A GNN operating on that graph — with a physics-informed loss enforcing fuel/dynamics
constraints — is a plausible modeling choice for the optimizer itself, not just a Pc predictor. Worth
treating as the candidate novel-methods slot (parallel to methods 7/8 in `archive/kelvins-agent-b/CONTRACT.md`)
rather than the whole MVP.

## The MVP, as scoped

1. **Ingest** conjunction data (CDM-equivalent: object pair, Pc, TCA, miss distance, covariance) for
   a defined set of satellites — "consensus data" in the sense of one clean, validated input stream,
   even if built from more than one upstream source.
2. **Score** each active conjunction for Pc/TCA using the existing method library (start with
   whichever of the eight methods in `agent-b` is most reliable today — likely naive persistence or
   GBM as the trustworthy baseline, not the exploratory transformer/GAN methods).
3. **Optimize**: given all active conjunctions across the fleet at once, compute a maneuver
   (delta-v, timing) per satellite that drives every pairwise Pc below a threshold at its TCA, at
   minimum total fuel cost, without moving any satellite outside its allowed station-keeping box or
   creating a new conjunction with a satellite it wasn't previously close to.
4. **Output**: a maneuver plan (per satellite: burn vector, timing, resulting Pc) — this is the
   deliverable, not a live operational system.

## Data sources: the actual constraint on this MVP

This is the part most likely to break the plan if it's not addressed early, so it's worth being
blunt about it. Real, live CDMs are not a broadly public dataset — they're not like TLEs.

- **Space-Track.org / 18th SDS** issues CDMs to the specific owner/operator pair involved in a
  conjunction, not to the public. You'd need to be a registered operator (or partner with one) to
  receive live CDMs this way. ([space-track.org/documentation](https://www.space-track.org/documentation))
- **TraCSS** (NOAA's Traffic Coordination System for Space, the planned successor/supplement to
  Space-Track for civil conjunction data) is real and has a published CDM spec
  ([TraCSS CDM Spec v2.1](https://space.commerce.gov/wp-content/uploads/2025/07/TraCSS-_CDM_Spec_Version_2.1.pdf)),
  but as of the most recent reporting it's still in beta with a small number of onboarded operators
  (SpaceX was reported as roughly its 10th beta user) — not an open public API yet.
  ([Office of Space Commerce](https://space.commerce.gov/traffic-coordination-system-for-space-tracss/))
- **Starlink's Space Safety Platform** (docs.space-safety.starlink.com) is the richest data model of
  any of these: CDMs from both operator-submitted ephemeris and their own Stargaze optical SSA system,
  plus hourly-updated ephemeris/GNSS states for their own fleet, plus the ability to submit your own
  trajectories for real conjunction screening. **Correction to what I said earlier** — this is not
  self-serve/public. Access requires an onboarding email from SpaceX and mTLS client-certificate
  issuance (you submit a CSR, they approve and sign it) — i.e. it's vetted access, similar in kind to
  TraCSS's beta gating, just via a different door. Whether an independent/research project can get
  through that door is unconfirmed — worth asking, but not something to plan the MVP timeline around.
- **LeoLabs** sells an automated collision-avoidance service built on its own radar tracking plus
  other commercial data sources — a real "consensus" product, but commercial/paid, not something an
  MVP can assume free access to.
- **SpaceMap** offers a commercial CDM API positioned as a multi-source aggregator/space traffic
  management layer — same caveat, worth a look but likely paid.
- **The ESA Kelvins Collision Avoidance Challenge dataset** (already the basis of `archive/kelvins-agent-b/`) is
  real historical CDM data, but static and anonymized — excellent for training/validating the risk
  layer and for building/testing the optimizer's logic, useless as a live feed.

Practical read: a genuinely live, multi-source "consensus CDM" pipeline is a partnerships problem
(operator credentials, or a paid LeoLabs/SpaceMap subscription) as much as an engineering one, and
probably isn't achievable for a first MVP. Two honest paths forward — worth deciding explicitly
rather than defaulting into one:

- **Path A — historical/replay MVP.** Build the whole pipeline (ingest → risk → optimize) against
  the Kelvins dataset and/or a synthetic constellation propagated from public CelesTrak TLEs via
  SGP4, computing conjunctions yourself with the existing Pc calculators. This proves the
  optimization layer works and is fully buildable with public data today. It's not "live," but it's
  honest about what's actually accessible.
- **Path B — one real live source, revised.** Starlink's Space Safety CDM/ephemeris API is not
  actually a no-approval-needed source (see correction above), so this path now means: apply for
  Space Safety Platform access as a parallel, non-blocking ask (cheap to try, would meaningfully
  upgrade data fidelity if granted), while the MVP itself doesn't depend on the answer.

Either path defers the "multiple sources reconciled into one consensus feed" idea rather than
building it first — reconciling disagreeing CDMs from independent sources (different covariance
estimates for the same conjunction) is itself a nontrivial statistical problem, and better tackled
after the optimizer has something reliable to consume.

## Fleet choice and data-source choice are two separate decisions

Worth pulling these apart explicitly, because "focus on Starlink" answers one of them very well and
doesn't touch the other.

**Which satellites to optimize** — Starlink is a genuinely strong choice here, independent of API
access. It's the largest, best-documented operational constellation that exists (thousands of active
satellites), it has a real, well-understood shell/orbital-plane structure and public station-keeping
conventions to optimize against instead of made-up ones, and it's exactly the shape of "large
collection of satellites" the redirect is asking to solve for.

**Where the conjunction/risk data comes from** — this does not have to be the gated Starlink API.
CelesTrak publishes free, current, no-registration-required TLE and SupGP (SpaceX-submitted
high-precision supplemental) element sets for the entire Starlink constellation
([celestrak.org/NORAD/elements/supplemental](https://celestrak.org/NORAD/elements/supplemental/)),
and the same for the rest of the tracked catalog (debris, other operators) it might conjunct with.
That's enough to self-generate conjunction screening — propagate everything with SGP4, find close
approaches, estimate Pc with the calculators already built in `archive/kelvins-agent-b/methods/*/pc_calculator.py`
(with an assumed/typical covariance where real operator covariance isn't public) — which is standard
practice in conjunction-assessment research when operator-grade covariance isn't available. This path
requires nobody's approval and is buildable starting today.

**Recommendation:** treat "Starlink is the fleet" and "apply for Starlink Space Safety API access" as
two independent moves. Build the MVP on Starlink-as-fleet using self-generated TLE/SupGP-based
screening as the guaranteed data path; send the Space Safety Platform access request in parallel as a
free option on better data later, without letting it gate anything.

## What Starlink's own docs reveal about their algorithm (worth mining, not "images")

Joey's suggestion: since the actual API is gated, pull what's usable from the public documentation
site itself rather than the API. That's a good instinct — the docs pages are public and readable
without a key, and a couple of them turn out to specify the exact algorithm parameters their real
system uses, which is far more useful for this project than screenshots or a link crawl would be:

- **Screening box** (the threshold for flagging a close approach as worth generating a CDM for):
  2 km radial, 44 km along-track, 51 km out-of-plane, in the RTN frame, checked bidirectionally
  between the two objects. This is a concrete, drop-in parameter for the self-built intra-fleet
  screening pipeline — using the same box makes its output directly comparable to what the real
  platform would have flagged.
  ([Architecture and Internals](https://docs.space-safety.starlink.com/docs/tutorial-basics/architecture/))
- **Collision probability method**: the "two-dimensional Alfano collision probability method" (after
  S. Alfano) — a published, standard Pc algorithm in the conjunction-assessment literature, not a
  proprietary black box. Implementing the same method in the self-built Pc calculator (rather than,
  or in addition to, the methods already in `archive/kelvins-agent-b/`) means results are computed the same way a
  real operator-grade system computes them, not just "a reasonable approximation."
  ([Architecture and Internals](https://docs.space-safety.starlink.com/docs/tutorial-basics/architecture/))
- **Trajectory submission format**: CCSDS OEM v3.0 (state vectors + covariance, not TLEs) is what
  their API actually expects if trajectory submission access is ever granted — worth keeping the
  self-built pipeline able to *export* to that format even though it consumes TLE/SupGP data
  internally, so a future access grant is a format-conversion step, not a rewrite.
  ([Quick-Start Guide](https://docs.space-safety.starlink.com/docs/quick-start/))
- **Data model**: five core entities — Operators, Objects, Trajectories (state + covariance),
  CDMs (one close approach snapshot), and Conjunction Events (a close approach's full CDM history
  over time) — mirrors the standard CCSDS conjunction-assessment concepts and is a reasonable schema
  to mirror internally for consistency with the field.

Worth doing more of this kind of targeted extraction from their docs (API Patterns, Data Model
category, Objects/Trajectories pages) as the screening pipeline gets built, specifically for
algorithm parameters and schema shape — separate from actually needing API access, which stays a
low-probability, non-blocking side request per the discussion above.

## Don't replicate the Starlink platform — use what already exists

Question that came up: given the Space Safety Platform is gated to (likely) real operators, is it
worth having Claude build a from-scratch replica of it? Short answer: no, not as a whole platform —
but it's worth being precise about which piece of it is actually needed, because that piece is
smaller than "replicate the platform" and mostly already exists publicly.

Break the platform into three layers and be honest about each:

- **Tracking/observation infrastructure** (Stargaze's own optical telescopes, plus operators
  submitting their private ephemeris directly) — genuinely not reproducible without owning real
  sensors or striking data-sharing deals. Not worth attempting; not needed for the MVP either.
- **Orbit determination** (fusing observations into precise state vectors) — already covered at
  adequate fidelity by CelesTrak's free TLE/SupGP data (see above).
- **Conjunction screening + Pc computation** (the part that actually matters for this project) —
  this is public, documented orbital-mechanics math, and there are already several public/open-source
  implementations. Writing this from scratch would be re-deriving things other people have already
  built and validated:
  - **[SOCRATES Plus](https://celestrak.org/SOCRATES/)** (CelesTrak) — a free, public, no-application
    conjunction screening service. Runs 3x/day, screens active satellites against the full public
    catalog via TLEs, reports miss distance/max Pc/TCA. Important gap: it explicitly **excludes
    intra-fleet conjunctions** for constellations like Starlink, on the assumption operators track
    those internally — so it won't give Starlink-vs-Starlink conjunctions, which is the core of what
    a fleet-coordination MVP needs. Useful as a free external-threat feed and a sanity-check against
    your own screening output, not sufficient alone. Web report only, no API — would need to be
    scraped if automated.
  - **[NASA CARA_Analysis_Tools](https://github.com/nasa/CARA_Analysis_Tools)** — NASA's actual
    operational conjunction-assessment codebase, open source (NASA Open Source Agreement), actively
    maintained (commits as recent as January 2026). Mostly MATLAB, so treat it as a validated
    reference methodology to check your own Pc math against, not a drop-in Python dependency.
  - **[caspy](https://github.com/ut-astria/caspy)** (UT Austin ASTRIA group) — Python, GPL-3.0,
    reads ephemeris (CCSDS OEM, or converted TLE data) and performs screening, writing results out in
    CDM format. Closer fit to this project's Python stack than CARA; GPL-3.0 carries the same
    copyleft caveat already flagged for Kessler in `docs/kessler-report-and-integration-plan.md` —
    check the license implications before depending on it directly versus using it as a reference.
  - **AstroLibrary** (2024 paper, "A library for real-time conjunction assessment and optimal
    collision avoidance") — worth reading closely since it spans *both* halves of this redirect
    (screening and avoidance-maneuver optimization), not just the screening half the other tools
    cover.

**Bottom line:** don't build a replica of the platform. Do build the specific piece nothing public
gives you for free — intra-fleet screening across the chosen fleet, using CelesTrak TLE/SupGP data
with your own SGP4 propagation and an all-pairs (or spatially-partitioned) closest-approach search —
and validate your Pc math against NASA CARA's published methods and/or caspy's implementation rather
than deriving it from nothing. That intra-fleet screening step is genuinely necessary work (it's
listed in the build order above), but it's a well-precedented, moderate-scope task, not a platform
rebuild.

## The optimization layer itself

This is the actually-new engineering. Framing: minimize total fleet delta-v (or a weighted
per-satellite fuel budget) subject to (a) post-maneuver Pc below threshold for every conjunction at
its TCA, (b) each satellite's maneuver staying inside its station-keeping/mission box, (c) no
maneuver creating a new conjunction that wasn't there before. This is a constrained optimization
over a coupled system — moving satellite A to dodge B might put A too close to C — which is exactly
why it needs to be solved jointly across the fleet rather than satellite-by-satellite.

Relevant existing work to build from rather than reinvent:

- Convex formulations of collision-avoidance maneuver optimization under uncertainty — e.g.
  ["Convex optimization of collision avoidance maneuvers in the presence of uncertainty"](https://www.sciencedirect.com/science/article/abs/pii/S0094576522002570)
  and ["Collision Avoidance Maneuvers Optimization in the Presence of Multiple Encounters" (JGCD)](https://arc.aiaa.org/doi/abs/10.2514/1.G008332)
  — directly relevant to the "multiple simultaneous conjunctions per satellite" case.
- MDP/RL framings of maneuver planning + collision avoidance —
  ["Markov Decision Processes for Satellite Maneuver Planning and Collision Avoidance" (arXiv 2501.02667)](https://arxiv.org/abs/2501.02667)
  and an RL-based autonomous mission-planning framework for on-orbit collision avoidance
  ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0273117725014322)) — an
  alternative to convex optimization if the fleet-scale coupling makes a single convex program too
  unwieldy.
- Multi-spacecraft cooperative trajectory optimization
  ([Schaub group, Colorado](https://hanspeterschaub.info/Papers/grads/ChandrakanthVenigall.pdf)) —
  closest in spirit to "optimize a whole fleet jointly" rather than one satellite at a time.

Recommendation: start with the convex formulation for a small fleet (tens of satellites) as the
MVP — it's the most tractable to implement and verify correctness on, and gives a clean baseline
before reaching for RL or a GNN-based learned optimizer.

## Suggested build order

1. Nail down which data path (A or B above) the MVP targets — this changes what "week 1" looks like.
2. Reuse the strongest existing Pc/TCA method from `archive/kelvins-agent-b/` as the risk layer (don't rebuild it;
   pick the one with the best Kelvins score and treat it as done for now).
3. Build a synthetic or replayed fleet scenario: N satellites, their current states, and a set of
   simultaneous active conjunctions among them (from Kelvins replay or live Starlink data).
4. Implement the convex fuel-optimal deconfliction solver against that scenario; validate against
   hand-checkable small cases (2-3 satellites) before scaling up.
5. Only after (4) works: revisit the GNN framing as a learned alternative to the solver, and revisit
   multi-source consensus as a data-layer upgrade.

## Starlink's role, clarified (2026-08-27)

Starlink is the MVP's test fleet, not a target customer/vendor to build toward. The goal is a
prototype that demonstrates the general capability the redirected project is meant to prove out
(consolidated risk data → fleet-wide fuel-optimal maneuver plan), using Starlink because it's the
most realistic large, well-documented constellation with genuinely public orbital data (via
CelesTrak TLE/SupGP) to build against — not because the end product is meant to be Starlink-specific.
Practical implication: keep Starlink-specific details (shell inclinations, station-keeping
conventions, satellite count) as configuration/test fixtures rather than hardcoded assumptions in the
optimizer itself, so the same pipeline could point at a different or synthetic fleet later without a
rewrite. That's a "replicates what the project is meant to accomplish" bar, not a "ships for
Starlink" bar.

## Open questions for the human to decide

- Scale: is "a large collection of satellites" a single operator's constellation (tens to
  low-thousands, e.g. a Starlink-shell-sized problem) or the whole tracked catalog? This changes the
  optimization method (convex QP is fine at fleet scale, not at catalog scale). Starlink-as-test-fleet
  suggests the former for the MVP.
- Does "optimized for fuel and space" mean per-satellite fuel budgets are fixed and the objective is
  purely feasibility (find *any* maneuver set that clears all conjunctions within budget), or is
  minimizing total fuel the actual objective? These lead to different solver formulations
  (feasibility problem vs. cost-minimization).
