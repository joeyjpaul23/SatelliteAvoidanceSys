# Where the time actually goes, and what fixed it

Work on the problems the first session left open. Every number here was
measured; the null results are reported at the same length as the wins,
because several of the most promising ideas produced nothing.

Two sections are worth reading even if the rest is not. §3.4 is the last
scaling limit — memory, not time — closed exactly. §4.5 is the one that
mattered: an independent verifier refuted the headline result, and the cause
turned out to be a condition on the inflation bound that nothing in the
formulation had ever checked.

---

## 1. The profile that reframed everything

The first session built a certified lazy-constraint loop to shrink the linear
program, trained a graph network to guess which constraints mattered, and
reported that the network lost to an empty guess. The natural next step looked
like "find a better guess". It was not.

Profiling a 26-object scenario over four orbits:

| stage | time | share |
|---|---:|---:|
| `screen` (initial screening) | **14 699 ms** | 56 % |
| re-screen (the SGP4 measurement) | **11 178 ms** | 43 % |
| latent-row enumeration | 1 000 ms | 4 % |
| LP assembly | 33 ms | 0.1 % |
| **LP solve** | **240 ms** | **0.9 %** |

Every acceleration idea in the first session targeted that last row. The LP is
**1 % of the runtime.** Screening is 99 %.

---

## 2. What the search over row-selection strategies actually showed

Before the profile, four strategies were compared on row selection, each
returning a **bit-identical objective** (gap 0.0e+00):

| strategy | rows used (of 1716) | median speedup |
|---|---:|---:|
| **empty guess** — predict nothing | **0–9** | **1.3× – 10.6×** |
| trained graph network | 119–199 | 0.16× – 1.91× |
| certified analytic per-row pruning | 1286 | 0.65× – 0.98× |
| pruning + analytic a-posteriori guess | 37–105 | 0.43× – 0.97× |

The **certified per-row prune** is worth describing because it is sound and
still lost. A latent row can only ever be violated if the budget can move it
past its nominal margin, and that is checkable in closed form:
`margin > Σ_i D_i · max_k |coeff_{i,k}|` ⟹ the row can never bind. Verified
over 8 896 rows: **zero binding rows lost**. It prunes 15 % at a 2 m/s budget
and 65 % at 50 mm/s — and is still slower than starting empty, because
computing the bound for every row costs more than the rows save, and because
the empty start already converges to 0–9 rows in one or two rounds.

**The conclusion is the interesting part.** The active set holds two to eleven
rows out of seventeen hundred. Any predictor must be *cheaper than free* to
beat "do nothing", and nothing is. For this problem class the optimal
initialization is the empty one.

---

## 3. What was fixed

### 3.1 The screening hot path — 2.1×, bit-identical

`cProfile` put 12.6 s of an 18 s screen inside `_cluster_entered_box`, and
within that, `rtn_to_eci_matrix` called **174 057 times** — once per pair per
epoch. Two exact changes:

- **Explicit cross products.** `numpy.cross` dispatches through `moveaxis` and
  `normalize_axis_tuple`; 6.8 s of the 18 s was that machinery rather than
  arithmetic. Writing the three components out is bit-identical (verified over
  20 000 random pairs, zero mismatches) and removes the dispatch.
- **Memoise the rotation per (object, epoch).** The matrix depends only on one
  object's own state, but every pair that object belongs to recomputed it. In a
  26-object catalog each object is in 25 pairs.

Measured, with the conjunction list checked for bit-identity every time:

| scenario | objects | before | after | speedup | identical |
|---|---:|---:|---:|---:|---|
| induced-cascade, n=24 | 26 | 10 258 ms | 4 909 ms | **2.09×** | yes |
| induced-cascade, n=16 | 18 | 4 225 ms | 2 464 ms | 1.71× | yes |
| debris-shower | 9 | 943 ms | 717 ms | 1.32× | yes |
| star / clique / chain | 4–5 | — | — | ~1.0× | yes |

The speedup scales with catalog size, which is the right shape.

### 3.2 Incremental re-screening — up to 14×, exact

After a maneuver **only the maneuvered satellites moved**, so a pair with no
maneuvered member has geometry identical to the baseline and cannot have
*become* a conjunction. Restricting the re-screen to pairs touching a burn is
therefore exact for the induced count, not an approximation. `screen` already
had a `keep_pair` hook.

| scenario | objects | maneuvering | full re-screen | incremental | speedup | induced counts agree |
|---|---:|---:|---:|---:|---:|---|
| n=24 | 26 | 6 | 5 039 ms | 2 010 ms | **2.51×** | yes |
| n=16 | 18 | 6 | 2 621 ms | 1 330 ms | 1.97× | yes |
| n=24, fuel-only | 26 | 2 | 11 178 ms | 774 ms | **14.4×** | yes |

The gain scales with how few satellites maneuver, which is the common case.

A contract test caught two wrong versions of this on the way. Filtering the
*baseline* makes every untouched baseline pair look like it disappeared.
Filtering the *difference* as well discards genuinely induced pairs whenever
the caller supplies an unrestricted assessment. Neither filter is needed: the
restricted `screen` already guarantees only changeable pairs appear.

### 3.3 The empty-guess lazy solve, as the default — up to 5×, bit-identical Δv

Every solving planner now closes its induced-conjunction row set lazily from an
empty guess rather than assembling all of them. Verified against solving with
all rows present, across 12 scenarios from 88 to 1698 latent rows:

| latent rows | Δv all rows | Δv lazy | relative difference | speedup | rows used |
|---:|---:|---:|---:|---:|---:|
| 362 | 1525.360 | 1525.360 | 0.0e+00 | 1.23× | 9 |
| 815 | 1550.967 | 1550.967 | 0.0e+00 | 1.80× | 11 |
| 1698 | 1550.958 | 1550.958 | 0.0e+00 | **3.07×** | 11 |
| 1696 | 1351.699 | 1351.699 | 0.0e+00 | **4.97×** | 4 |

Maximum relative Δv difference across all twelve: **0.000e+00**. No held-back
row was violated in any case.

**End to end** the pipeline on the largest case went from roughly 26 s of
screening plus re-screening to about 6.9 s — **~3.8×** — with every output
bit-identical or provably equivalent.

---

### 3.4 Sparse LP assembly — up to 42× less memory, bit-identical

The one scaling limit left standing was memory, not time: dense assembly
reached **4.7 GB at 11,936 rows** and the risk filter was the only thing
keeping real problems away from it.

The reason it blows up is structural. The slack block is **diagonal** — each
resolve and latent row touches exactly one slack column — so a latent row
carries `n_lifted + 1` nonzeros. But `n_cols` grows with the number of latent
rows, so storing those rows densely is *quadratic in the one dimension that
actually grows*. Measured at 20,840 rows: 1,003,860 nonzeros in a
20,840 × 20,964 matrix, or **0.23 % dense**.

Assembly now accumulates coordinate triplets and materialises a CSR matrix
once the dense form would exceed 4M elements (32 MB); below that it returns a
dense array exactly as before, so anything small enough to inspect by hand is
unchanged. `scipy.optimize.linprog` and `LinearConstraint` both accept either.

| latent rows | shape | nonzeros | dense peak | sparse peak | ratio |
|---:|---|---:|---:|---:|---:|
| 4,164 | 4,184 × 4,308 | 201,252 | 160.2 MB | **16.8 MB** | 9.5× |
| 20,820 | 20,840 × 20,964 | 1,003,860 | 3,575.1 MB | **84.0 MB** | **42.5×** |

The ratio grows with the problem because dense cost is quadratic and sparse
cost is linear.

Exactness, since this touches the certified core:

- the two matrices are **entrywise identical**, max |difference| `0.000e+00`;
- `solve_lp` returns the same status and the same objective, relative gap
  `0.000e+00`, with the **solution vector bit-identical** (max |Δz| `0.000e+00`);
- a re-run of an 8-scenario benchmark reproduced **56 of 56** (scenario,
  planner) rows with max |Δv difference| `0.000e+00` mm/s and no change to any
  induced count.

## 4. What the failure search actually found

### 4.1 Per-pair probability-derived floors: no measurable effect

A geometric exclusion radius and a probability threshold are different objects,
so deriving each latent row's floor from *that pair's own* projected covariance
at the watch threshold should be better than one global median-derived radius.
Implemented (`pair_required_miss_km`) and measured over 80 scenarios: the
planner comparison and the premium distribution came back **identical to the
last digit**.

The reason is specific and worth knowing: in this family every latent pair is a
fleet satellite at the same altitude under the same covariance model, so each
pair's derived floor is within a few per cent of the global median. The
machinery is kept because it is the right construction and will matter for a
mixed-altitude or mixed-covariance catalog, but on this evidence it buys
nothing here.

### 4.2 A linearization margin: does not fix what it was introduced for

Plans satisfied their tightest latent row by **ten metres**, and an SGP4
re-screen then found a conjunction there anyway — which looks exactly like an
optimizer sitting on a boundary known only to linearisation accuracy. Sweeping
a margin on every latent floor over 80 scenarios:

| margin | induced measured (`fleet-safe`) | premium median | premium p90 | infeasible |
|---:|---:|---:|---:|---:|
| 0 m | 5 | 4.41 % | 16.00 % | 10 |
| 100 m | 6 | 4.53 % | 16.41 % | 10 |
| 200 m | 6 | 4.56 % | 17.59 % | 11 |
| 500 m | 4 | 4.83 % | 17.31 % | 14 |
| 1000 m | 3 | 5.87 % | 28.41 % | 16 |

It barely moves the count, costs a third of a percent of premium per 500 m, and
costs feasibility. The knob is kept and **defaults to zero**, because the
diagnosis below showed the cause was elsewhere.

### 4.3 The actual cause: enumeration, not accuracy

Measuring the failing cases directly against SGP4:

| case | floor | model, after plan | **true, after plan** | model error | true minimum over the window |
|---|---:|---:|---:|---:|---:|
| seed 16 | 4.332 | 4.464 | 4.471 | **+7 m** | **2.572 km at t+6.38 h** |
| seed 22 | 4.347 | 4.357 | 4.363 | **+6 m** | **2.788 km at t+6.38 h** |
| seed 59 | 4.307 | 4.792 | 4.798 | **+6 m** | **2.741 km at t+6.38 h** |

The first-order model is accurate to **six metres**. The constraint is
satisfied at every epoch where a row exists. The true minimum is two kilometres
lower, at an epoch where **no row exists**.

The enumeration placed rows at local minima of the *nominal* separation. For a
co-orbital pair the nominal separation is **flat** — 9.345 km at every sampled
epoch — so there are no meaningful local minima and four arbitrary epochs were
chosen. The maneuver itself creates the time-variation, by introducing secular
along-track drift, so the perturbed minimum is nowhere near any nominal one.

Two fixes were tried.

**Dense enumeration** — a row at every gate-admissible epoch — is affordable on
the solve side precisely because of §3.3, but it multiplies enumeration cost by
the number of epochs, and enumeration was already 4 % of runtime. It trades a
cheap stage for an expensive one.

**Lazy epoch generation** is the better answer, and it is the same idea as the
lazy row loop applied one level down. Once `x` is known, evaluating where the
*perturbed* separation bottoms out is cheap — a few 3x3 products per epoch via
`satellite_displacements`, no SGP4 and no constraint assembly. So: solve, find
each pair's true perturbed minimum, add one row there, re-solve. The epochs
that matter are discovered from the plan instead of guessed before it.

### 4.4 A pinned budget outlives the row set it was measured on

Once lazy epoch generation was in place, `fleet-safe` and `milp-ops` reached
zero measured induced conjunctions across all nine families, but
`lexicographic` kept exactly one, in `induced-cascade` seed 3. Routing its
second stage through the same shared lazy path did not fix it. The cause turns
out to be specific to the lexicographic construction, and it is a trap any
two-stage safety-then-fuel formulation can fall into.

`lexicographic` is two solves. Stage one prices slack at `1e12` and reads off
the total shortfall the geometry allows — for this scenario **1.06553 km**.
Stage two pins `induced_budget` to that value and minimises Δv underneath it.
The polish loop inside stage two then does its job: it finds four pairs whose
perturbed separation bottoms out at an epoch no row covered, and adds a row at
each. Those four rows carry shortfall of their own, total shortfall rises above
the pinned 1.06553 km, and the stage-two solve comes back `infeasible`:

```
polish round 1: the perturbed separation bottoms out away from every
  enumerated epoch for 4 pair(s); added a row at each true minimum and re-solved
two-stage polish returned 'infeasible'; keeping the lazy-loop solution,
  whose delta-v is not a minimum
```

The loop then falls back to the unrefined solution — which violates the rows it
just discovered. Measured against SGP4, pair `13000:13002` closes to
**1.0031 km** against a 4.3624 km floor, at t+3120 s. The linear model predicted
1.0019 km at the same epoch: a **1.2 metre** discrepancy. The model was never
wrong. The plan was simply never constrained there, and the mechanism that
should have constrained it was disabled by a budget measured before the rows
existed.

`fleet-safe` pins no budget, so its polish loop absorbs the new rows and
re-solves. That is the whole difference between 0 and 1.

The fix is to close the epoch set on stage one, before reading the shortfall
off it, so the budget stage two is pinned to was measured against the rows
stage two will actually face. After it, `lexicographic` on seed 3 converges to
the same 3334.2 mm/s plan as `fleet-safe` — the cheapest plan achieving the best
attainable safety, which is what the formulation was always supposed to return
— and induces nothing.

The general statement is worth keeping separate from the bug: **a constraint
budget derived from one relaxation is not transferable to a tighter one.** Any
scheme that measures an attainable safety level and then pins it must measure
it against the final row set, not an intermediate one. With lazily generated
rows, "final" is only known after the generation closes.

A second, quieter defect surfaced alongside it. `solve_lazy_two_stage` grew the
row set internally but returned only the solution, so every caller certified
against the *originally enumerated* rows. A plan violating a row the refinement
itself had added could therefore be reported `linearized_safe` — which is what
seed 3 did, with `certificate_safe=True` and 0 violations next to a real,
SGP4-measured induced conjunction. The function now returns the closed row set
and all three callers certify against it. This does not change any plan; it
changes what the certificate is allowed to claim, which for a certificate is
the whole point.

### 4.5 The one that mattered: a minimum the grid cannot see

An independent verifier, allowed to read but not edit, was asked to refute
"zero measured induced conjunctions." It found a counterexample by moving the
**Δv budget** off the 2000 mm/s the sweep had been run at. At 500 and
1000 mm/s — the latter being `DEFAULT_DV_BUDGET_KM_S`, the library's own
default — `chain` seed 6 produced a measured induced conjunction for **all
three** coordinated planners, each reporting `certificate_safe=True`.

The mechanism is not the one §4.3 fixed, and it is worse.

Pair `16000:16001` crosses at **7.24 km/s**. Over one 30 s latent step it
sweeps **217 km**. The true minimum is **0.1429 km** — and the two grid samples
bracketing it read **76.3 km** and **140.8 km**:

```
grid sample 00:28:33 : separation  76.3455 km
   true minimum 00:28:43.57 :      0.1429 km      <-- invisible
grid sample 00:29:03 : separation 140.8490 km
```

The sampled separation is **monotone increasing straight through a minimum
three orders of magnitude below either neighbour**. No test on sampled
separations can find it — not `argmin`, not a local-minimum scan, not the
decrease-then-increase test. The linear model was not at fault: it agreed with
SGP4 to a median of **2.2 metres** over the window.

The inflation γ cannot cover it either, and this is the part worth stating
carefully. γ = ½·A·h² + Λ·h bounds the **curvature** of the relative
trajectory. That is the right object only when the relative velocity at the
minimum is near zero — a co-orbital pair, which is exactly the
`induced-cascade` geometry the method was developed on. For a crossing, the
honest inter-sample bound is the ½·v_rel·h term the design had rejected as
vacuous, and at 7.24 km/s that is **108 km** against a 2.88 km floor.

**The tidal bound quietly assumed low relative velocity, and nothing checked
it.** Measuring ½·v_rel·h against the floor for every candidate pair shows the
assumption failing everywhere, not just where it happened to bite:

| family | worst ½·v_rel·h | floor | ratio |
|---|---:|---:|---:|
| `isolated-pair` | 73.6 km | 3.68 km | 20× |
| `chain` | 111.2 km | 2.88 km | 39× |
| `induced-cascade` | 142.5 km | 3.45 km | 41× |
| `star` | 185.2 km | 3.52 km | **53×** |

A 30 s grid was giving **no guarantee at all** in any family. Zero was being
measured, not certified — and 6 configurations out of 900 were where the luck
ran out.

**The fix is the range rate.** `ṙ = (δ·δ̇)/|δ|` changes sign at every minimum
however deep, so brackets come from a sign change in `ṙ` rather than from
comparing separations, and each bracket is refined by cubic Hermite
interpolation of the relative position from its endpoints and their
derivatives. This required the exact derivative of Φ — `cw_impulse_rate_matrix`,
verified against central differences to `6.1e-09` over 2000 random (n, σ) — and
a matching `satellite_displacement_rates`. Rows then go at the refined,
**between-sample** epoch, which needs SGP4 states off the grid, so the
propagator is retained on the context. The search is now independent of how
fast the pair moves relative to the grid step.

### 4.5b Where the ½·v·h term actually belongs

A second verifier was given the strengthened claim and told to find a new axis
of attack. It swept 560 configurations — budgets from 50 to 10 000 mm/s, seeds
out to 60, `burn_slots` 2 and 12, `scp_iterations` 1 and 2, `axes=1`, the cone
cost model, `target_pc=1e-4`, `plan_risk_level=ACT` — and broke it on exactly
one: **`latent_step_s = 120`**, `chain` seed 1, all three planners, 10 of 11
again reporting `certificate_safe=True`.

The range-rate search had not failed. The pair was never **offered** to it:

```
latent_step_s=30  : candidate_pairs = [11000:11001, 11001:11002, 11002:11003]  measured = []
latent_step_s=120 : candidate_pairs = [            11001:11002, 11002:11003]  measured = ['11000:11001']
                    11000:11001 is NOT a candidate pair
```

Candidate pairs were derived from the rows the enumeration happened to place,
and the enumeration admitted a pair by testing `separation <= gate` **at
samples**. A pair whose entire approach falls between two samples has no sample
inside the gate, gets no row, is therefore not a candidate, and the
perturbed-minimum search — which can only refine pairs it is handed — never
looks at it. Fixing the search was necessary and not sufficient; the pair has to
survive admission first.

This is where the rejected ½·v_rel·h term belongs, and the placement is the
whole point:

- **On the constraint floor it is vacuous.** 108 km against a 2.9 km exclusion
  radius is not a constraint, it is an empty feasible set. That is the measured
  finding that produced the tidal bound in the first place.
- **On the gate it is nearly free.** Widening admission by 863 km at a 120 s
  step admits a few more candidate pairs. It costs enumeration time and
  nothing else — no plan is made worse by considering a pair that turns out
  not to bind.

So admission now tests `separation <= gate + ½·v_rel·h`, with `v_rel` taken from
**that pair's own** relative speed rather than a catalog-wide 19 km/s maximum,
which keeps it as tight as soundness allows. Candidate membership comes from
what the gate admitted, not from what got a row — they are different questions,
and conflating them is what let a coarse grid drop a 7 km/s crossing outright.

With it, `chain` seed 1 at a 120 s step goes from a measured induced
conjunction to none, and the search is genuinely step-independent rather than
step-independent down to 30 s.

### 4.6 Two shortfall budgets that fought each other

Fixing the search exposed a second defect in `lexicographic`, on `star` seed 10.
It was setting `induced_budget` — a cap on **total** shortfall from its own
stage 1 — on top of the per-row `slack_caps` that `solve_two_stage` already
applies. Two budgets measured on different iterates, stacked:

```
stage 1 reached 5.33598 km of total shortfall; stage 2 minimised delta-v ... row by row
polish round 2: ... added a row at each true minimum and re-solved
two-stage polish returned 'infeasible'; keeping the lazy-loop solution,
  whose delta-v is not a minimum
```

The planner then reported a *worse* attainable safety than `fleet-safe` on
identical geometry — 5.34 km of shortfall against 0.34 km — and honestly
minimised Δv against the wrong number. The row-by-row cap is also the stronger
reading of "safety first": bounding only the total lets the optimizer trade a
resolved conjunction for a slightly worse one elsewhere at no cost. So the
total pin was removed and the shared path left to do it. Stage one is still run
explicitly, to report the attainable shortfall, and the two stages are now
iterated to a **fixed point over the row set**, since perturbed minima are a
property of the plan and each stage lands on a different one.

### 4.7 Where it actually stands, including where it does not hold

Three rounds of this — each fix broken by an axis the previous sweep had not
varied — is itself the result worth reporting. So the final claim is stated
over the axes that broke the earlier ones, and the limits are stated with it.

**3066 non-vacuous planner checks**: five non-vacuous families × 16 seeds ×
4 Δv budgets × 3 enumeration steps × 3 planners, plus 366 checks varying burn
slots, refinement iterations, `axes=1`, the cone cost model, `target_pc` and
the risk level.

| enumeration step | measured induced | of which certified "safe" |
|---|---:|---:|
| **30 s (the default)** | **0 / 900** | — |
| 60 s | 4 / 900 | 2 |
| 120 s | 24 / 900 | 17 |
| all other axes, at 30 s | **0 / 366** | — |

**`latent_step_s` is a correctness parameter, not a performance knob**, and it
is the only one in the module that is. Coarsening it does not merely place
fewer rows — it changes which approaches are *visible at all*, because both
admission and row placement start from sampled separations. The other six
parameters swept move the answer not at all.

The gate widening of §4.5b improved the coarse-grid cases without rescuing
them: `chain` seed 1 at 120 s went from failing at all four budgets to one. It
did not make a 120 s grid sound, and no claim is made that it did. The
enumeration now sets `grid_coarser_than_validated` and emits a note naming
these numbers, so a coarse run is reported as unvalidated rather than passing
for the validated one.

What the search cost overall:

| | before | after |
|---|---:|---:|
| **measured induced, at the validated step** | **6** | **0** |
| of those, certificate claimed `safe` | **6 of 6** | — |
| plans changed | — | 593 / 899 |
| median Δv change | — | **+0.000 %** |
| p90 Δv change | — | +16.2 % |

The median plan is unchanged; the cost falls entirely on the cases that were
wrong.

Three rounds, three different root causes, one pattern: **every one was a place
where a continuous quantity was being decided from samples of it.** The
constraint epochs (§4.3), the minimum search (§4.5), and the candidate gate
(§4.5b) each assumed that what happens between two samples is bounded by what
happens at them. For slow relative motion that is true, which is why the
co-orbital family the method was developed on never exposed any of it.

---

## 5. The result

First the family the method was developed against, then everything else. These
80 `induced-cascade` scenarios were the original measurement, taken before the
range-rate fix of §4.5; they are kept because they are where the
epoch-generation argument was established, and the co-orbital geometry they use
is the one case where the sampled grid was adequate. The all-family,
all-budget numbers that supersede them are below.

80 `induced-cascade` scenarios, exclusion radius derived per pair, lazy epoch
generation on:

| planner | median Δv (mm/s) | induced predicted | **induced SGP4-measured** | scenarios affected | certified safe |
|---|---:|---:|---:|---:|---:|
| `legacy-lp` | 1094.1 | 425 | **176** | 71 / 80 | 0.0 % |
| `greedy-pairwise` | 899.2 | 229 | **97** | 64 / 80 | 0.0 % |
| `fuel-only` | 715.7 | 200 | **71** | 55 / 80 | 100 % |
| **`fleet-safe`** | 867.7 | **0** | **0** | **0 / 80** | 95.0 % |

**Zero measured induced conjunctions, across every scenario**, confirmed by a
full SGP4 re-screen of the maneuvered catalog — against 71 for the same
optimizer without the induced-conjunction constraints, and 176 for the legacy
scalar-model planner.

The cost is small and bounded:

| quantity | before | after |
|---|---:|---:|
| measured induced (`fleet-safe`) | 5 | **0** |
| median Δv | 843.5 mm/s | 867.7 mm/s (+2.9 %) |
| safety premium, median | 4.41 % | 4.41 % |
| safety premium, p90 | 16.00 % | 16.00 % |
| latent rows enumerated (median) | 62 | 62 |
| rows actually used in the solve | 11 | **3** |
| context build | 133 ms | **92 ms** |
| plan time | 10.8 ms | 81.5 ms |

The premium distribution is unchanged because it is measured on the enumerated
rows at a fixed linearization; the extra rows the epoch loop discovers affect
the plan, not the comparison. Plan time rose eightfold and remains under a tenth
of the screening it sits inside.

The one regression worth naming: certified-safe fell from 98.8 % to 95.0 %. The
newly discovered rows are genuinely binding in four scenarios and the
certificate now says so, where before it was silent about a constraint nobody
had written down. That is the certificate working.

### Across all nine families

Repeating the full sweep — 144 scenarios, nine families, 2000 mm/s:

| planner | median Δv (mm/s) | induced predicted | **induced SGP4-measured** | scenarios affected |
|---|---:|---:|---:|---:|
| `legacy-lp` | 77.9 | 168 | **37** | 16 / 144 |
| `greedy-pairwise` | 12.5 | 79 | **18** | 13 / 144 |
| `fuel-only` | 33.1 | 68 | **13** | 10 / 144 |
| **`lexicographic`** | 510.6 | 6 | **0** | **0 / 144** |
| **`fleet-safe`** | 610.5 | 6 | **0** | **0 / 144** |
| **`milp-ops`** | 513.1 | 6 | **0** | **0 / 144** |

And, more to the point, zero across the whole **900-configuration** sweep that
varies the Δv budget as well — 500, 1000, 2000 and 4000 mm/s — which is where
the previous version failed. Safety premium over all families: 44 defined,
median 0.00 %, p90 5.30 %, p99 14.46 %, max 15.97 %, **zero negative**, 22.2 %
infeasible.

The ten rows that still show a safe certificate beside a measured induced
conjunction are all `fuel-only`, which carries no latent rows by construction —
its certificate is silent on induced risk because it was never asked. That is
the ablation working as intended, not a certificate gap.

**Four of the nine families never exercise the planner at all.** `clique`,
`debris-shower`, `intra-plane` and `replay-tle` produce no conjunction reaching
MONITOR at any budget tested, so `context.sensitivities` is empty and no burn
is ever commanded. "Nine families" is therefore five families doing the work
and four passing vacuously; the claim should be read that way. `intra-plane` is
documented as exercising the 2D-Pc guard and, across 41 seeds, never does.

### The monotone premium law survives

| in-plane neighbour spacing | n | median premium | p90 | infeasible | induced measured |
|---|---:|---:|---:|---:|---:|
| 3–5 km | 9 | **29.67 %** | 35.70 % | 6 | **0** |
| 5–7 km | 12 | 15.42 % | 18.64 % | 0 | **0** |
| 7–9 km | 15 | 10.00 % | 12.69 % | 0 | **0** |
| 9–12 km | 21 | 3.28 % | 6.10 % | 0 | **0** |
| 12–15 km | 23 | **0.00 %** | 2.25 % | 4 | **0** |

---

## 6. What this says about where to use learning

The search covered four row-selection strategies, a dense reformulation, a
probability-derived floor, a linearisation margin, a lazy epoch loop, sparse
assembly, and a range-rate minimum search. The ones that worked are all
**algorithmic and exact**: memoise a rotation, restrict a re-screen to the pairs
that can have changed, store a matrix that is 0.2 % dense as though it were,
discover the epochs that matter from the solution instead of guessing them, and
bracket a minimum by the quantity that actually changes sign at one.

None of them is a prediction problem. The pattern is consistent: wherever this
pipeline was slow, the fix was to stop computing something redundant, and
wherever it was wrong, the fix was to look in the right place rather than to
look harder. A model's job would be to guess, and in every case there turned
out to be a cheap way to know.

That leaves triage — deciding *before* screening whether a scenario needs work
at all, where 53 % of a benchmark sweep needed no maneuver and paid full
screening cost anyway. That is a genuine prediction problem with a 15–25 s
prize, and it is the one target where a model is not competing against a closed
form. See `docs/model-bakeoff-results.md`.
