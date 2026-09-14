# Session report — 2026-09-12

What was built, what was measured, what was wrong on the first attempt, and
what is still open. Written so that someone who was not here can check every
claim.

---

## 1. What this session set out to do

The starting point was `docs/fleet-optimization-tradeoff-research-handoff.md`,
which identified a gap: AEGIS could plan fleet maneuvers, but could not say
what it cost to plan them so that no *new* conjunction was created. It listed
the missing pieces explicitly — induced-conjunction counts, a measured
fuel-versus-induced-risk frontier, a comparison against greedy and fuel-only
baselines, and "proof or empirical evidence that the final re-screen covers all
relevant fleet pairs and time epochs."

All of those now exist. Several of them turned out to require correcting the
existing model first.

---

## 2. What was built

| Package | Lines | What it is |
|---|---:|---|
| `aegis.fleetopt` | 6 769 | The optimization core: three-axis B-plane dynamics, certified induced-conjunction constraints, reachability bound, LP/MILP assembly and solving, certification, Farkas analysis, graph metrics, Pareto frontier, ten planners |
| `aegis.scenarios` | 1 357 | Ten deterministic, structurally-verified scenario families (nine synthetic, one replayed from committed TLE fixtures) with stable digests |
| `aegis.experiments` | 1 657 | The benchmark sweep, honest metrics, SGP4-measured validation, CLI |
| `aegis.ml` | 3 773 | Differentiable physics, graph tensorization, dataset generation, the PIGNN, training, calibration, certified inference |
| `aegis.store` | 1 276 | Content-addressed artifact store (local/S3/GCS), SQLite/Postgres experiment store, Parquet datasets |
| tests | 3 527 | Contract tests written from the specification by testers who could not edit the source |

The contract is `aegis/specs/step14_fleet_optimization.md` (943 lines).

**No pre-existing module was modified.** The 224 tests that passed at the start
of the session still pass; the suite is now **762 passing** in 19 s. The
working tree also carries an unrelated, uncommitted SOCRATES ingest feature
that predates this session and touches `api/`, `constants.py`,
`ingest/celestrak.py`, `pipeline/`, `screening/engine.py` and three existing
test files — none of its diffs reference anything built here, and it was left
exactly as found.

---

## 3. The mathematics, and why each piece is there

### 3.1 The displacement operator and the B-plane

Everything the optimizer knows about dynamics is the exact Clohessy-Wiltshire
position-from-impulse block `Phi(n, sigma)`. Its `(2,2)` entry carries the
secular term `-3 sigma`: only a tangential impulse buys unbounded displacement,
which is why lead time rather than propellant is the scarce resource.

The post-maneuver miss distance is `||P (d + B x)||` where `P` projects onto
the plane normal to the relative velocity — displacement along the relative
velocity moves the *time* of closest approach, not its *distance*.

**This is a correctness fix, not a refinement.** The existing planner estimates
the secondary's contribution as `(d·t_a)(t_a·t_b)` where the correct
coefficient is `d·t_b`. Measured against SGP4 on cross-plane intra-fleet
conjunctions, the scalar form has the **wrong sign**: it predicts a burn
increases separation by 0.21 km when it actually decreases it by 0.67 km.
Median error 128 % against 1.0 % for the B-plane form. Co-planar pairs agree to
0.01 %, exactly as the analysis predicts. See
`docs/model-fidelity-validation.md` §2.

### 3.2 Certified induced-conjunction constraints

A latent constraint says "do not bring this pair closer than `s_min`". Placed
at local minima of the nominal separation, in B-plane form, with an inflation
covering what a grid sample cannot see:

```
gamma = 0.5 * A * h^2 + Lambda * h,    A <= 2*mu*G / r^3,    Lambda = sqrt(51)*(D_a + D_b)
```

`A` is the **tidal** relative acceleration of a pair already inside the gate.
Using relative speed instead — the obvious Lipschitz bound — gives 285 km at a
30 s step against a 2 km exclusion radius, which makes the constraint set
vacuous. The tidal bound gives 0.55 km. A factor of 500.

### 3.3 The reachability bound (the completeness result)

`||delta r_i(t)|| <= D_i * max_tau ||Phi(n_i, t - tau)||`, so a pair whose
nominal separation always exceeds `s_min + rho_a + rho_b` cannot be brought
together by *any* admissible plan. Screening with that inflated gate therefore
enumerates every inducible pair — the completeness the handoff asked for.

Evaluating the bound needed care: `||Phi||_2` is **not** monotone in lead time
(its growth rate `3 - 4cos(n sigma)` goes negative just after each whole
orbit). The first draft asserted monotonicity; a 400-point sweep disproved it.
The closed-form monotone envelope `sqrt((3 n sigma + 4)^2 + 34)/n` replaced it.

Verified over 3 000 random admissible plans: zero violations, bound 91 % tight.

### 3.4 Exact penalty, and the tradeoff that isn't

Above `lambda* = max dual`, the soft-penalty and hard-constrained problems
coincide. Measured: at `0.5 lambda*` the optimizer leaves 1.34 km of slack and
spends no fuel; at `2 lambda*` it leaves exactly zero and matches the
hard-constrained delta-v to 1e-7. The threshold was `1.66e-5`, against the
codebase's default penalty of `1e4` — so AEGIS was already in the exact regime
and the apparent tradeoff was never real.

### 3.5 The frontier is convex and piecewise linear

`V*(epsilon)` is convex, piecewise linear and non-increasing, with slopes equal
to minus the induced-budget dual. Verified on 6 scenarios: convexity residual
≤ 9e-19, slope-versus-dual agreement ≤ 1.5e-13, breakpoints recovered where
the dual steps down.

---

## 4. Seven things that were wrong first, and how they were found

Recorded because the corrections are the evidence that the numbers were
measured rather than assumed.

1. **`||Phi||_2` is not monotone.** Asserted in the first draft of the spec; a
   sweep disproved it. Replaced with a proven closed-form envelope.
2. **The legacy burn mapping is not dynamically consistent with any optimizer.**
   A constant phase offset has no secular drift; measured against SGP4 it was
   wrong by up to 25 % at a different epoch. Replaced by a Gauss variational
   update plus a mean-element refit — 0.2 % for along-track burns.
3. **The grid-inflation bound was 285 km.** Using relative speed rather than
   tidal acceleration made the induced-conjunction constraint vacuous.
4. **Excluding resolve pairs from the latent set removed the rows that
   mattered.** An in-plane fleet-mate 12 km away is inside the 44 km screening
   box, so it is screened as a (negligible-Pc) conjunction — and dropping the
   pair let the optimizer slide straight into it. Now only the epochs a resolve
   row already governs are skipped.
5. **An empty plan reported `feasible=True` with zero conjunctions.** A planner
   with no usable burn slot looked identical to one with nothing to do.
6. **`fuel-only` reported zero induced conjunctions by construction**, because
   it was scored against its own (empty) latent set. Every planner is now
   scored against the full set the context knows about.
7. **The lazy-constraint loop checked the wrong residual.** It tested the
   physical separation `||d + Bx|| - f` where the program enforces the
   linearized row `u·(d + Bx) - f`. By Cauchy-Schwarz the linear form is never
   larger, so the loop terminated on solutions the full program would have
   rejected — breaking exactness on 6 of 27 scenarios. Fixed, and
   `linear_residual_km` now exists so the distinction cannot be lost again.

Two more were found by the verification agents rather than by me:

10. **The benchmark crashed writing its human-readable report**, on the exact
    command the README printed, because making the exclusion radius derived
    left it defaulting to `None` and the renderer formatted it with `:g`.
    Found by a fresh-eyes agent following the README verbatim.
11. **The premium's "infeasible" flag meant the wrong thing.** It was driven by
    "did `fleet-safe` resolve every conjunction", which marked a whole run
    infeasible when a single repeat encounter was out of reach — hiding
    premiums that were perfectly well defined. It now means what it says: no
    zero-induced plan exists at the linearization.

Two more were structural rather than bugs:

12. **A penalty formulation does not minimise delta-v when anything is
   unresolvable.** At 1e4 per km of shortfall against a delta-v cost of order
   1e-3, the optimizer spends its entire budget shaving a micrometre. Reported
   delta-v was not a minimum of anything, and premiums came out *negative*.
   Every planner is now two-stage: minimise shortfall, then minimise delta-v
   underneath it, pinning each row's shortfall individually.
13. **The safety premium is not a bound unless the comparison is constructed to
   make it one.** `fleet-safe` carries mandatory slack on its latent rows, so
   its objective differs from `fuel-only`'s and adding constraints does not
   force a worse optimum. `fixed_linearization_premium` solves both at one
   linearization with latent slack pinned to zero and resolve slack capped at
   the fuel-only solution — and then the premium is provably non-negative,
   which it now measures as.

---

## 5. Results

All numbers below are from 80 deterministic `induced-cascade` scenarios ---
one maneuverable satellite forced to act by an external hazard, with in-plane
fleet-mates close enough that the avoidance displacement can reach them. That
is the geometry the research question is about, and the one the other eight
families do not produce. Neighbour spacing is sampled per seed over
3--15 km, so the seed sweep is also a sweep over how tightly the fleet is
packed.

The induced-conjunction exclusion radius is **derived**, not chosen: the
separation at which collision probability reaches the watch threshold under
the assessed covariances, times a 1.5 margin. Median across these scenarios:
**3.137 km**.

### 5.1 The planner comparison

Induced conjunctions are counted two ways and both are reported. *Predicted*
is the optimizer's own constraint residual. *Measured* is a full SGP4
re-screen of the maneuvered catalog, counting pairs that reach WATCH and were
not there before. Only the second is evidence.

| planner | median Δv (mm/s) | induced predicted | **induced measured** | scenarios with induced | certified safe |
|---|---:|---:|---:|---:|---:|
| `no-maneuver` | 0.0 | 0 | **0** | 0 | — |
| `legacy-lp` | 1094.1 | 425 | **176** | 71 / 80 | 0.0 % |
| `greedy-pairwise` | 899.2 | 229 | **97** | 64 / 80 | 0.0 % |
| `fuel-only` | 715.7 | 200 | **71** | 55 / 80 | 100 % |
| **`fleet-safe`** | 843.5 | **0** | **5** | **5 / 80** | 98.8 % |
| `lexicographic` | 843.0 | **0** | **5** | 5 / 80 | 98.8 % |

Reading the table:

- **The constraint does what it is for.** Predicted induced conjunctions go
  from 200 (`fuel-only`) to **zero**. That is the quantity the program
  controls, and it controls it exactly.
- **Against SGP4 the reduction is 93 %**, from 71 measured induced
  conjunctions to 5, and from 55 affected scenarios to 5. Against the legacy
  scalar-model planner it is 97 %, from 176 to 5.
- **It is not zero**, and that is the honest part. Five survive because the
  optimizer's latent set is an enumeration under a derived radius and a
  thinning cap, while the re-screen is a probability test on the full catalog.
  The two criteria are different objects. Reporting only the predicted count
  would have shown a perfect score.
- **Coordination is doing the work, not fuel.** `greedy-pairwise` spends more
  delta-v than `fuel-only` and induces more; `legacy-lp` spends the most and
  induces the most. None of the uncoordinated planners passes certification.

### 5.2 The safety premium

Measured at a fixed linearization, with latent slack pinned to zero and
resolve slack capped at the fuel-only solution --- the construction under
which the premium is provably a lower bound.

| statistic | value |
|---|---|
| scenarios where it is defined | 70 / 80 |
| **no zero-induced plan exists** | 10 / 80 (12.5 %) |
| median | **4.41 %** |
| p90 | **16.00 %** |
| p99 | 42.62 % |
| worst | 54.69 % |
| zero-premium fraction | 20 % |
| **negative (construction forbids it)** | **0** |

The median sits inside the 1.1--3.8 % range Pavanello et al. report for a
single spacecraft paying for stronger probability margins across five, seven
and ten simultaneous encounters --- from an entirely different formulation,
measuring a different thing. That is corroboration, not confirmation, but it
is the closest published quantity.

### 5.3 The premium is governed by packing, not by conjunction count

| in-plane neighbour spacing | n | median | p90 | max | infeasible | induced measured (`fleet-safe`) |
|---|---:|---:|---:|---:|---:|---:|
| 3--5 km | 9 | **29.67 %** | 35.70 % | 37.20 % | 6 | 2 |
| 5--7 km | 12 | 15.42 % | 18.64 % | 54.69 % | 0 | 0 |
| 7--9 km | 15 | 10.00 % | 12.69 % | 13.22 % | 0 | 1 |
| 9--12 km | 21 | 3.28 % | 6.10 % | 6.95 % | 0 | 2 |
| 12--15 km | 23 | **0.00 %** | 2.25 % | 11.30 % | 4 | 0 |

Monotone across a factor of five in spacing and a factor of thirty in premium.
This is the shape the handoff document hypothesised --- "sparse scenarios will
often have a small safety premium... while tightly coupled encounter clusters
will produce much larger premiums or become infeasible" --- and it holds, with
the transition to infeasibility arriving below about 5 km.

**A hypothesis that did not survive.** The handoff proposed stratifying by
conjunction-graph structure, and this project added a spectral predictor for
it: the *coupling number*, `cond(B Bᵀ)` of the row-normalised projected
sensitivities. Its rank correlation with the premium is **0.074** --- nothing.
The premium is set by how far apart the neighbours are, not by the conditioning
of the conjunction network. The coupling number is still computed and reported;
it is simply not the predictor it was proposed as.

### 5.4 Structural results, verified

| claim | verification | result |
|---|---|---|
| Prop. 4 — reachability bound | 3 000 random admissible plans | 0 violations, bound 91 % tight |
| Prop. 2 — conservative linearization | 20 000 random draws | 0 violations |
| Prop. 6 — exact penalty threshold | λ* = 1.66e-5; solve at 0.25×, 0.5×, 2×, 10× | slack positive below, exactly zero above, delta-v matches hard-constrained solve to 1e-7 |
| Prop. 7 — frontier is convex, piecewise linear | 6 scenarios | convexity residual ≤ 9e-19, slope-versus-dual agreement ≤ 1.5e-13, breakpoints recovered |
| Lazy-constraint exactness | 27 problems × 4 guess kinds | max relative objective gap 1.4e-16 |

### 5.5 Certified acceleration, and a negative result

The lazy-constraint closure returns the exact optimum for any guess, and the
speedup grows with the problem:

| latent rows | rows used | full solve | lazy solve | speedup |
|---:|---:|---:|---:|---:|
| 68 | 1 | 4.5 ms | 2.5 ms | 1.8× |
| 358 | 2 | 18.5 ms | 5.4 ms | 3.5× |
| 820 | 2 | 48.5 ms | 8.3 ms | 5.9× |
| 1716 | 2 | 145.8 ms | 16.1 ms | **9.0×** |

Objective gap exactly zero at every size. The active set is tiny — typically
**two** binding rows out of 1716 — which is exactly why the reduction pays.

The physics-informed GNN trained to supply that guess **did not beat starting
from an empty set**: median end-to-end speedup 0.77× against roughly 4× for the
empty guess, because the active set is far smaller than the network's output
and inference costs as much as the solve. The full accounting, including the
training and calibration numbers, is in `docs/pignn-acceleration-results.md`.
The mechanism works; the model, on this corpus, does not earn its place.

---

## 6. Independent verification

The five load-bearing propositions were handed to an agent with no prior
exposure to the project, no ability to modify any file, and instructions to
write its own scripts with its own seeds and geometries rather than re-run the
project's tests.

| Claim | Method | Result |
|---|---|---|
| Prop. 4 — reachability bound | 3 000 random admissible plans across three mean-motion regimes including the rectilinear limit, both envelope modes, 6 000 checks | **0 violations**; tightest realized/bound ratio **0.9909** |
| Prop. 2 — conservative linearization | 8 000 random `MissSensitivity` objects, each at a random `x` and at an `x` engineered onto the linear boundary; 16 000 checks | **0 implication failures** |
| Prop. 6 — exact penalty threshold | own feasible problem; 0.25×, 0.5×, 2×, 10× λ* | positive slack below, zero slack above, delta-v gap **0.000e+00** |
| Lazy-constraint exactness | 20 own problems (own seeds 2000–2019, own spacings, 15–199 latent rows) × empty / random / adversarial guesses = 60 comparisons | max relative objective gap **0.000e+00**, bit-identical; `rows_used` genuinely differed by guess |
| Premium non-negativity | 30 own cases, own seeds 5000–5014 plus a fixed-spacing probe | 26 defined, 4 honestly infeasible, **0 negative**; independently reproduced the spacing transition |

### A second verifier, told only "see if it actually works"

A separate agent with no context was handed the repository and asked to behave
like a careful new user. It ran the suite (762 tests, clean, 19 s), discovered
the CLI without being told where it was, exercised every subcommand, and then
went after the guardrails and the docstrings rather than taking either on
trust.

- **Guardrails held under three separate bypass attempts**: no environment
  variable; environment variable without the acknowledge flag; and a direct
  Python call passing a hand-rolled duck-typed object with
  `acknowledge_synthetic=True`. All three refused, because the check is a real
  `isinstance` against the actual class plus two independent gates. The cloud
  store raised `StoreUnavailableError` with an actionable message rather than
  degrading, and only fell back when explicitly asked, recording a visible note.
- **It checked the docstrings numerically instead of believing them.**
  Proposition 2: 0 soundness violations over 20 000 trials, and *strictly
  inner in 57.3 %* — confirming it really is a restriction and not an equality
  wearing a restriction's name. The envelope `M(n,sigma)`: 0 violations over
  4 000 samples across five orbits, with the exact norm confirmed non-monotone
  exactly as the module says a first draft got wrong.
- **It found a real bug I had introduced.** Making the exclusion radius derived
  left `induced_exclusion_km` defaulting to `None`, and the markdown renderer
  formatted it unconditionally with `:g` — so `benchmark`, run with the exact
  command line the README printed, wrote `report.json` correctly and then
  crashed writing `report.md`. Fixed, along with three documentation defects it
  found: a duplicated heading, a frontier example whose seed happens to be
  degenerate, and a missing row in the docs index.

Its verdict on the headline claim was that it would believe it, having checked
the load-bearing parts by hand rather than by reading.

Its summary: *"Genuine, and unusually candid... this is not a case of tests
merely agreeing with the code, since I built my own geometries and
reimplemented the checks from the norm definitions, not the solver's own
rows."* It identified the learned component as the weakest part of the work,
which is also what the project's own measurements say.

It also found a real limitation worth recording: **`induced-cascade` scenarios
are infeasible at zero slack even for the resolve rows alone.** The family
generates a repeating crossing pair with four or five TCAs that the budget
cannot jointly satisfy, so a zero-slack feasible problem has to be constructed
by hand. That does not affect the premium measurement — which compares two
planners under identical resolve constraints — but it means the family is not
a source of fully-resolvable problems, and anything needing one must build it.

---

## 7. Second session: the acceleration and solution search

A follow-up session attacked every open problem empirically. The full record is
`docs/acceleration-and-solution-search.md`; the outcome in brief.

**A profile reframed the work.** The LP the first session spent its effort
accelerating is **1 % of the runtime**. Screening is 99 % — 14.7 s of initial
screen and 11.2 s of re-screen against a 0.24 s solve.

**Three exact optimisations, all verified output-identical:**

| fix | speedup | verification |
|---|---:|---|
| explicit cross products + memoised rotation per (object, epoch) | **2.1×** | conjunction lists bit-identical on 7 scenarios |
| re-screen only pairs touching a burn (exact: nothing else moved) | **2.5× – 14×** | induced counts identical in every case |
| lazy row closure from an **empty** guess, as the default | up to **5×** | Δv relative difference 0.000e+00 over 12 scenarios |

End to end, ~**3.8×** on the largest case.

**The induced-conjunction gap closed completely.** The first session reported 0
predicted against 5 SGP4-measured induced conjunctions. Diagnosis: the
first-order model was accurate to **six metres**; the constraint was satisfied
at every epoch where a row existed; and the true minimum sat two kilometres
lower at an epoch with **no row**. For a co-orbital pair the nominal separation
is flat — 9.345 km at every sampled epoch — so nominal local minima are
arbitrary, and the maneuver's own drift decides where the real minimum goes.

The fix is a lazy loop one level down: solve, evaluate each pair's *perturbed*
minimum cheaply, add one row there, re-solve. Result over 80 scenarios:

| planner | median Δv | induced predicted | **induced SGP4-measured** |
|---|---:|---:|---:|
| `legacy-lp` | 1094 mm/s | 425 | **176** |
| `greedy-pairwise` | 899 mm/s | 229 | **97** |
| `fuel-only` | 716 mm/s | 200 | **71** |
| **`fleet-safe`** | 868 mm/s | **0** | **0** |

**Zero**, across every scenario, for +2.9 % Δv and no change to the premium
distribution (median 4.41 %, p90 16.0 %). Repeating the sweep across all nine
families — 144 scenarios — every planner that carries the induced-conjunction
rows (`fleet-safe`, `milp-ops`, `lexicographic`) reaches **zero measured
induced conjunctions**, against 13 for `fuel-only`, 18 for `greedy-pairwise`
and 37 for `legacy-lp`. Getting `lexicographic` there required one further fix:
its stage-two shortfall budget was measured on the enumerated rows, so the
epoch refinement's new rows made it infeasible and it fell back to an
unrefined plan. See §4.4 of `docs/acceleration-and-solution-search.md`.

**Four ideas that did not work**, reported at the same length as the wins in
the results document: a certified analytic per-row prune (sound — zero binding
rows lost over 8 896 — and still slower than predicting nothing); per-pair
probability-derived floors (identical results to the last digit, because every
latent pair here shares one covariance regime); a linearisation margin (5 → 3
induced across a 0–1000 m sweep, at the cost of premium and feasibility); and
dense enumeration (correct but more expensive than the lazy epoch loop).

**The model bake-off.** Twenty-plus architectures were then trained on the two
targets where learning could still plausibly pay — pre-screening triage and the
safety premium. Full record in `docs/model-bakeoff-results.md`.

*Triage was rejected.* Every model family reached ROC-AUC 1.0 and specificity
1.0 at recall 1.0 — closed-form rule, logistic regression, gradient boosting,
random forest, k-NN, MLP, deep ensemble, MC-dropout, Bayes-by-backprop,
Bayesian LSTM, DeepSets, Set Transformer, 1D CNN. Five unrelated families
hitting a ceiling is a warning, and leave-one-family-out confirmed it: hold out
`induced-cascade` and the rule skips **100 of 100 scenarios that needed a
maneuver**, recall 0.000. The label is constant within nine of ten generator
families, so the task is family recognition. Since the one intolerable error is
a false negative, **no triage gate was implemented.** Bayesian uncertainty did
not earn its cost either: it correctly ranks the borderline cases, but those
"failures" are threshold-transfer noise that a better-chosen threshold removes
for free.

*One predictor was adopted.* The **penetration index** — the fuel-optimal
plan's worst latent-row violation divided by that row's reachable change —
predicts the premium at Spearman **0.989** (independently re-measured; the
bake-off got 0.943), and is zero **if and only if** the premium is zero, on
70/70 and 71/71 scenarios in two separate runs. It replaces the coupling
number, which measured 0.074 and is constant on this data. It beat thirteen
hand-built scalars and a random forest on sixty-six features, costs one pass
over the latent rows, and is physically readable: *what fraction of the
fleet's reach is already spoken for by the cheapest plan's worst induced
conjunction.*

**The pattern worth recording.** Every fix that worked was algorithmic and
exact — stop recomputing something, restrict to what can have changed, look in
the right place. The one genuine predictive win was a closed-form scalar, not a
model. Wherever this pipeline was slow or wrong, there turned out to be a cheap
way to *know* rather than guess.

---

## 8. Third session: the last induced conjunction

Two planners reached zero SGP4-measured induced conjunctions across all nine
families; `lexicographic` kept exactly one, in `induced-cascade` seed 3. Routing
its second stage through the shared lazy path — the change that fixed
`milp-ops` — did not fix it.

The cause is specific to the lexicographic construction. Stage one prices slack
at `1e12` and reads off the shortfall the geometry allows (1.06553 km here);
stage two pins that level and minimises Δv under it. Stage two's polish loop
then does its job, finds four pairs whose perturbed separation bottoms out at an
unenumerated epoch, and adds a row at each — and those rows carry shortfall of
their own, so the total rises above the pinned level and the solve returns
`infeasible`:

```
polish round 1: the perturbed separation bottoms out away from every
  enumerated epoch for 4 pair(s); added a row at each true minimum and re-solved
two-stage polish returned 'infeasible'; keeping the lazy-loop solution,
  whose delta-v is not a minimum
```

It then falls back to the unrefined plan, which violates the rows it just
found. Pair `13000:13002` closes to **1.0031 km** against a 4.3624 km floor at
t+3120 s; the linear model predicted 1.0019 km at the same epoch, a **1.2 metre**
discrepancy. The model was never wrong — the plan was never constrained there,
and the mechanism that should have constrained it was disabled by a budget
measured before those rows existed. `fleet-safe` pins no budget, absorbs the new
rows, and re-solves; that is the whole difference between 0 and 1.

**Fix.** Close the epoch set on stage one *before* reading the shortfall off it,
so the level stage two is pinned to was measured against the rows stage two will
face. The general statement, recorded as claim C15: *an attainment level
measured on one relaxation is not transferable to a tighter one*, and with
lazily generated rows the final set is known only after generation closes.

**A second defect found alongside it.** `solve_lazy_two_stage` grew the row set
internally but returned only the solution, so all three callers certified
against the *originally enumerated* rows. A plan violating a row the refinement
itself had added could be reported `linearized_safe` — which seed 3 did, with
`certificate_safe=True` and zero violations next to a real measured induced
conjunction. The function now returns the closed row set and every caller
certifies against it. No plan changes; what the certificate may claim does.

**Then a verifier broke it.** An independent agent, able to read but not edit,
was asked to refute "zero induced conjunctions." It did, by moving the Δv
budget off the 2000 mm/s the sweep had used. At 500 and 1000 mm/s — the latter
being the library's own `DEFAULT_DV_BUDGET_KM_S` — `chain` seed 6 induced a
conjunction for **all three** coordinated planners, every one reporting
`certificate_safe=True`.

The mechanism was not the enumeration gap fixed above. Pair `16000:16001`
crosses at **7.24 km/s** and sweeps **217 km** per 30 s step; its true minimum
of **0.143 km** sits between samples reading **76.3 km** and **140.8 km**, so
the sampled separation is monotone increasing straight through a minimum three
orders of magnitude below either neighbour. The linear model was accurate to a
median of **2.2 metres** — it was the *search* that was wrong.

Nor could the tidal inflation cover it. γ = ½Ah² + Λh bounds the curvature of
the relative trajectory, which is the right object only when the relative
velocity at the minimum is near zero — the co-orbital geometry the method was
developed on. Measured per family, the honest inter-sample bound ½·v_rel·h
exceeds the floor by **20× to 53×** everywhere, so a 30 s grid was certifying
nothing at all. Zero had been measured, not guaranteed, and 6 of 900
configurations were where the luck ran out.

The fix is to bracket by **range-rate sign change** — `ṙ = (δ·δ̇)/|δ|` changes
sign at every minimum however deep — and refine each bracket by cubic Hermite
interpolation, placing rows at the refined between-sample epoch. That needed
the exact derivative of Φ (`cw_impulse_rate_matrix`, checked against central
differences to `6.1e-09`), a matching `satellite_displacement_rates`, and the
SGP4 propagator retained on the context for off-grid states.

**A second verifier then broke the strengthened claim too**, on a new axis: a
560-configuration sweep over budgets from 50 to 10 000 mm/s, seeds to 60, burn
slots, refinement iterations, `axes=1`, the cone cost model and two probability
thresholds found exactly one failure — `latent_step_s = 120`, `chain` seed 1,
all three planners, 10 of 11 again claiming `certificate_safe=True`.

The range-rate search had not failed; the pair was never offered to it.
Candidate pairs were derived from the rows the enumeration placed, and admission
tested `separation <= gate` **at samples** — so a pair whose whole approach falls
between two samples has no sample inside the gate, gets no row, and never
becomes a candidate. Fixing the search was necessary and not sufficient.

This is where the rejected ½·v_rel·h term belongs, and the placement is the
result. On the constraint *floor* it is vacuous — 108 km against a 2.9 km
exclusion radius is an empty feasible set, which is what motivated the tidal
bound in the first place. On the *gate* it is nearly free: it admits a few more
candidate pairs, costing enumeration time and nothing else, because a pair that
turns out not to bind makes no plan worse. Admission now tests
`separation <= gate + ½·v_rel·h` with `v_rel` taken per pair, and candidate
membership is read from what the gate admitted rather than from what got a row.

That exposed one more defect: `lexicographic` was stacking a total-shortfall
`induced_budget` on top of the per-row `slack_caps` the shared path already
applies. The two, measured on different iterates, conflicted; stage two came
back `infeasible` and the planner reported 5.34 km of attainable shortfall
where `fleet-safe` found 0.34 km on identical geometry. The redundant total cap
was removed — the per-row cap is the stronger reading of "safety first" — and
the two stages are now iterated to a fixed point over the row set.

**A third sweep, adding the enumeration step as an axis**, settles where this
actually stands. 3066 non-vacuous planner checks — five non-vacuous families ×
16 seeds × 4 budgets × 3 enumeration steps × 3 planners, plus 366 checks
varying burn slots, refinement iterations, `axes=1`, the cone cost model,
`target_pc` and the risk level:

| enumeration step | measured induced | of which certified "safe" |
|---|---:|---:|
| **30 s (the default)** | **0 / 900** | — |
| 60 s | 4 / 900 | 2 |
| 120 s | 24 / 900 | 17 |
| all other axes, at 30 s | **0 / 366** | — |

So `latent_step_s` is a **correctness** parameter, not a performance knob, and
it is the only one in the module that is — the other six swept parameters move
the answer not at all. The gate widening improved the coarse-grid cases without
rescuing them (`chain` seed 1 at 120 s went from failing at all four budgets to
one) and no claim is made that a 120 s grid is sound. The enumeration now sets
`grid_coarser_than_validated` and emits a note naming these numbers, so a coarse
run is reported as unvalidated rather than passing for the validated one.

Three rounds, three root causes, one pattern: **every one was a place where a
continuous quantity was being decided from samples of it** — the constraint
epochs, the minimum search, and the candidate gate each assumed that what
happens between two samples is bounded by what happens at them. For slow
relative motion that is true, which is why the co-orbital family the method was
developed on never exposed any of it.

**At the validated 30 s step** (9 families × 16 seeds × 4 budgets × 3 planners,
1548 total): **6 measured induced conjunctions → 0**, and the 6
certificates that had claimed safety → 0. 593 of 899 plans changed; median Δv
change **+0.000 %**, p90 +16.2 %. The cost is concentrated entirely in the
cases that were wrong.

**Result** — 144 scenarios, nine families, 2000 mm/s budget:

| planner | median Δv (mm/s) | induced predicted | **induced measured** | scenarios affected |
|---|---:|---:|---:|---:|
| `legacy-lp` | 77.9 | 168 | **37** | 16 / 144 |
| `greedy-pairwise` | 12.5 | 79 | **18** | 13 / 144 |
| `fuel-only` | 33.1 | 68 | **13** | 10 / 144 |
| **`lexicographic`** | 510.6 | 6 | **0** | **0 / 144** |
| **`fleet-safe`** | 610.5 | 6 | **0** | **0 / 144** |
| **`milp-ops`** | 513.1 | 6 | **0** | **0 / 144** |

Premium over all families: 43 defined, median 0.00 %, p90 5.30 %, p99 14.46 %,
max 15.97 %, **zero negative**, 22.2 % with no zero-induced plan. Median plan
time 5.4 ms, worst 1.27 s. 762 tests pass.

**Four of the nine families never exercise the planner.** `clique`,
`debris-shower`, `intra-plane` and `replay-tle` produce no conjunction reaching
MONITOR at any budget tested, so no burn is ever commanded. "Nine families" is
five doing the work and four passing vacuously, and the claim should be read
that way. `intra-plane` is documented as exercising the 2D-Pc guard and, across
41 seeds, never does — worth fixing in the generator.

**Sparse LP assembly**, the last scaling limit left standing. Dense assembly
reached 4.7 GB at 11,936 rows. The slack block is diagonal — one slack column
per resolve/latent row — so each latent row holds `n_lifted + 1` nonzeros while
`n_cols` grows with the latent count: storing them densely is quadratic in the
one dimension that grows. At 20,840 rows the matrix is 0.23 % dense. Assembly
now accumulates coordinate triplets and returns CSR above 4M elements, dense
below. Memory 3,575 MB → **84 MB (42.5×)**; matrix entrywise identical, solution
vector bit-identical, and 56 of 56 benchmark rows reproduced to
`0.000e+00` mm/s.

The ten rows that still show a safe certificate beside a measured induced
conjunction are all `fuel-only`, which carries no latent rows by construction —
its certificate is silent on induced risk because it was never asked. That is
the ablation working, not a gap.

---

## 9. What is still open

- **The measured-versus-predicted induced gap.** The constraint drives its own
  predicted count to zero; a full SGP4 re-screen is the arbiter, and closing
  the gap required deriving the exclusion radius from the probability threshold
  rather than choosing it geometrically. That derivation is now the default,
  but the two criteria remain different objects and the gap should be
  re-measured on any new covariance model.
- **Operator-grade ephemerides.** Everything here inherits TLE-grade
  covariance. The premium numbers would change under real operator covariances,
  probably downward, since the required miss distances would shrink.
- **`intra-plane` never exercises the planner.** The generator is documented as
  testing the 2D-Pc guard and, across 41 seeds and every budget tested, never
  produces a conjunction reaching MONITOR. `clique`, `debris-shower` and
  `replay-tle` are vacuous the same way. Four of the nine families pass
  trivially; the generator should be fixed so they do not.
- **The refinement loop is not proven to converge.** Rows are added at
  perturbed minima until none is found or a round cap is hit, and the cap is
  reachable — perturbed minima are a property of the plan, and the plan moves
  when a row is added. It terminates in every case measured, and the
  certificate is evaluated on the closed row set so a non-converged plan is
  reported as unsafe rather than as safe. But "always terminates at a feasible
  point" is not established.
- **The learned component.** It does not currently beat an empty guess. The
  training corpus is small and the label extremely imbalanced (two binding rows
  in seventeen hundred). Reported as measured, not as settled.
- **MILP scale.** `milp-ops` is exercised but not stress-tested; the
  operational-cost weighting is a placeholder pending a real disruption model.
- **Cloud backends.** S3, GCS and Postgres adapters are implemented and were
  verified to raise `StoreUnavailableError` rather than degrade silently when
  the optional package or the service is missing. They could not be exercised
  against real services here: that needs credentials and
  `boto3` / `google-cloud-storage` / `psycopg`, none of which are installed.
- **The paper is not compiled.** `paper/aegis-certified-fleet-cam.tex` and its
  two `\input` files were checked structurally — balanced environments,
  balanced braces, matched `tabular` column counts — but no TeX toolchain
  exists on this machine (`pdflatex`, `xelatex`, `lualatex`, `tectonic` and
  `latexmk` are all absent), so it has never been run through a compiler.
- **Two paywalled prior-art items are unverified.** The SDEICA paper
  (*Applied Sciences* 16(8):3707) and the *Chinese Journal of Aeronautics*
  multi-debris paper are cited from abstracts only. Claim C1's "partially
  novel" verdict depends on what they actually contain and must be confirmed
  before any external submission.
- **Six lint errors remain**, all in files that predate this work
  (`core/timebase.py`, `pipeline/__init__.py`, `propagation/covariance.py`,
  `risk/assessor.py`). Every package added here — `fleetopt`, `experiments`,
  `scenarios`, `ml`, `store` — is clean.
- **Nothing is committed.** All of this work is uncommitted in the working
  tree.
