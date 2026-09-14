# Prior art and novelty ledger

Compiled 2026-09-12 from an eleven-lane literature sweep (web search plus
direct fetch of abstracts and PDFs). Every claim this project makes is
recorded here with its verdict and its closest prior art, so that the
whitepaper never overclaims and a reviewer can check the positioning
without redoing the search.

Lanes swept: multi-encounter CAM optimization; induced/secondary
conjunctions; fleet-coordinated CAM; ML for CAM; GNN learning-to-optimize;
reachability in screening; Pareto/exact-penalty; B-plane convexification;
mega-constellation operations; differentiable astrodynamics; public
benchmarks.

Verification caveat, stated once and meant: several leads were paywalled
(ScienceDirect, AIAA, MDPI returned 403 to automated fetch). Negative
results below mean **not found in this sweep**, not **proven absent from
the literature**. The items flagged `unverified` must be checked against
institutional access before any external submission.

---

## Claim ledger

| # | Claim | Verdict | Closest prior art |
|---|---|---|---|
| C1 | "No newly induced conjunction above threshold" as an explicit constraint **inside** a fleet maneuver optimizer | **Partially novel** — novel for an open, uncooperative catalog; a bounded formation-mate version exists | SDEICA, *Applied Sciences* 16(8):3707 (unverified); NASA/SP-20230002470 Rev.1 documents the outer-loop practice |
| C2 | Conservative affine linearization of the reverse-convex miss constraint via a supporting hyperplane | **Already published — not claimed** | Mueller, AIAA 2009-2051; Mao et al., IFAC 50(1):4063 (2017); Armellin, arXiv:2101.07403 (2021); Pavanello thesis (2025) |
| C3 | Reachability bound proving an inflated gate enumerates **every** pair an admissible plan could turn into a conjunction | **Novel** | Rivero, Vazquez & Merz, SDC9 2025 — requires the *actual* maneuver ephemeris and is empirically, not provably, complete; Hejduk & Pachura, NTRS 20170007928, explicitly disclaims completeness even without maneuvers |
| C4 | Exact-penalty threshold and convex piecewise-linear induced-risk frontier for the fleet problem | **Partially novel** — the theorems are classical; the domain instantiation and the computed frontier are not | Bertsekas, *Nonlinear Programming* (exact penalty); standard parametric-LP sensitivity; Vega et al., arXiv:2510.19058 (ad hoc penalty fallback, no exactness theorem) |
| C5 | Farkas/IIS certificate naming **which** conjunctions are mutually irreconcilable | **Novel** (application; the lemma is classical) | No domain prior art found. Tool: Schrijver, *Theory of Linear and Integer Programming* |
| C6 | **Narrowed after measurement.** The *certified lazy-constraint closure* — an active-set guess turned into an exactly-optimal solve — is the contribution, verified to machine epsilon and scaling to 9.0× at 1716 rows. The *learned predictor* supplying that guess did **not** beat an empty guess on our corpus (0.77× vs ~4×) and is reported as a negative result | **Novel (mechanism); not demonstrated (model)** | Bertsimas & Stellato, arXiv:1907.02206; Cauligi et al. (CoCo), arXiv:2004.03736; Gasse et al., arXiv:1906.01629; Misra, Roald & Ng (2018). All use flat networks over fixed-size parameter vectors, or predict branching inside an exact B&B; none predict an active set **and** duals on a variable-topology graph and then certify deterministically |
| C7 | Safety premium measured and stratified by conjunction-**graph structure** | **Novel** | Armellin 2021 (single-geometry delta-v vs Pc sweep, 6.4 → 431.1 mm/s); Klinkrad et al., ESA SP-587 (2005) (fleet-level curve, but maneuver *frequency* not delta-v) |
| C8 | The scalar along-track sensitivity model used in fleet planners returns the **wrong sign** for cross-plane intra-fleet conjunctions | **Novel (this project's own measurement)** | No prior art; established here by direct comparison against SGP4 — see `docs/model-fidelity-validation.md` |
| C10 | The induced-conjunction exclusion radius must be **derived from the probability threshold**, not chosen geometrically. Under TLE covariance the separation giving Pc = 1e-5 was 2.10 km against a 2 km radial screening box; planning at 2 km left 9 SGP4-measured induced conjunctions across 24 scenarios, deriving it drove that to 0 | **Novel (measurement)** | No prior art found; a direct consequence of a geometric criterion and a probability criterion being different objects |
| C11 | The **coupling number** `cond(B Bᵀ)` does *not* predict the safety premium (rank correlation 0.074); in-plane neighbour **spacing** does, monotonically over 30× | **Negative result** | Proposed by this project and refuted by this project |
| C12 | The **penetration index** — the fuel-optimal plan's worst latent-row violation normalised by that row's reachable change — predicts the premium at Spearman **0.989**, and is zero **iff** the premium is zero (70/70 and 71/71 in two independent runs) | **Novel** | Replaces C11. Beat thirteen hand-built scalars and a random forest on 66 features in a three-lane bake-off |
| C13 | Pre-screening **triage is not learnable** from this benchmark: every model family from a two-feature closed form to a Set Transformer reaches ROC-AUC 1.0, and leave-one-family-out gives **recall 0.000** on the held-out family | **Negative result** | The label is constant within 9 of 10 generator families; the task is family recognition, not physics |
| C14 | **Lazy epoch generation**: constraining nominal local minima is insufficient because a co-orbital pair's nominal separation is flat and the maneuver itself creates the time-variation. Discovering each pair's *perturbed* minimum after solving drives SGP4-measured induced conjunctions to **zero** | **Novel** | No prior art found; the failure mode is specific to putting induced conjunctions inside the optimizer at all |
| C15 | **A safety budget measured on one row set is not transferable to a tighter one.** A lexicographic safety-then-fuel formulation that pins stage two to the shortfall stage one attained becomes *infeasible* the moment lazy epoch generation adds a row, and silently falls back to the unrefined plan. The budget must be measured against the closed row set | **Novel (measurement)** | No prior art found; the interaction only exists when constraint generation and a pinned attainment level are combined, which requires C14 to be present at all |
| C16 | **A sampled grid cannot locate a close approach between fast-moving objects, and the tidal inflation silently assumes it can.** γ = ½Ah² + Λh bounds the *curvature* of the relative trajectory, valid only when the relative velocity at the minimum is near zero. For a crossing at 7.24 km/s a 30 s grid brackets a **0.143 km** minimum with samples of **76.3** and **140.8** km — monotone through it. Measured across families, ½·v_rel·h exceeds the floor by **20–53×**, so the grid certified nothing. Bracketing by **range-rate sign change** and refining by cubic Hermite makes the search independent of the step | **Novel (measurement and method)** | TCA refinement by range-rate root-finding is standard in *screening* (Hoots; Alfano). What appears to be new is the observation that a maneuver-planning constraint set built on sampled separations inherits none of that rigour, and that the tidal-inflation argument used to justify the coarse grid is conditional on a relative-velocity assumption nobody states |
| C19 | **The constraint-enumeration step is a correctness parameter, not a performance knob — and it is the only one that is.** Measured over 3066 planner checks: **0/900** measured induced conjunctions at 30 s, 4/900 at 60 s, 24/900 at 120 s (most still certified safe), against **0/366** across burn slots, refinement iterations, control axes, cost model, probability threshold and risk level. Coarsening the step changes which approaches are *visible*, not merely how many rows are placed | **Novel (measurement)** | Discretisation step is treated as an accuracy/cost trade throughout the CAM literature. What is measured here is that for induced-conjunction constraints it is a soundness threshold with a cliff, and that the other six parameters swept do not matter at all |
| C18 | **The inter-sample sweep ½·v_rel·h belongs on the candidate *gate*, not on the constraint *floor*.** On the floor it is vacuous (108 km against a 2.9 km exclusion radius); on the gate it is nearly free, because admitting a pair that turns out not to bind costs enumeration time and nothing else. Deriving candidate membership from the rows that happened to be placed — rather than from what the gate admitted — let a 120 s grid drop a 7.2 km/s crossing entirely, so no refinement could reach it | **Novel (design result)** | The trade between a conservative screening volume and a conservative constraint is not, as far as we found, stated anywhere as a placement question; the CAM literature inflates the miss requirement and the screening literature inflates the volume, and the two are not usually discussed together |
| C17 | **Stacking a total-shortfall cap on a per-row shortfall cap makes a lexicographic planner report worse attainable safety than the planner it should dominate.** Two budgets measured on different iterates conflict, stage two returns `infeasible`, and the planner minimises Δv against the wrong level (5.34 km of shortfall against 0.34 km on identical geometry) | **Negative result** | Proposed by this project and refuted by this project; the per-row cap is the stronger reading of "safety first" and the total cap is redundant |
| C9 | Impulsive maneuvers applied to SGP4 by **mean-element re-fitting**, making the optimizer's dynamics model and the simulator consistent | **Engineering contribution, likely not novel as a technique** | Standard operator practice is ephemeris regeneration; the contribution is the measured consistency gain, not the idea |

### Headline measurements

Eighty `induced-cascade` scenarios, exclusion radius derived (median 3.137 km):

| planner | median Δv | induced predicted | induced SGP4-measured | certified |
|---|---:|---:|---:|---:|
| `legacy-lp` | 1094 mm/s | 425 | 176 | 0 % |
| `greedy-pairwise` | 899 mm/s | 229 | 97 | 0 % |
| `fuel-only` | 716 mm/s | 200 | 71 | 100 % |
| **`fleet-safe`** | 843 mm/s | **0** | **5** | 98.8 % |

Safety premium (fixed linearization, provably non-negative): median **4.41 %**,
p90 **16.0 %**, 12.5 % infeasible, **0 negative**. Monotone in fleet packing:
29.7 % at 3–5 km neighbour spacing down to 0.0 % at 12–15 km.

After the second session's lazy epoch generation, `fleet-safe` reaches
**zero SGP4-measured induced conjunctions across all 80 scenarios** (was 5) for
+2.9 % delta-v, with the premium distribution unchanged. See
`docs/acceleration-and-solution-search.md` and `docs/model-bakeoff-results.md`.

### Positioning consequences

1. **C2 is machinery, not a contribution.** The supporting-hyperplane
   (projection-and-linearization) convexification is used, cited to Mueller
   (2009) / Mao et al. (2017) / Armellin (2021), and never presented as new.
   What this project adds is its use on a *variable-topology, fleet-scale*
   constraint graph that includes induced-conjunction rows.
2. **C1 must be stated narrowly**: induced conjunctions against arbitrary,
   uncooperative, open-catalog objects — not co-optimized formation mates.
   The whitepaper must also answer the NASA Handbook's explicit operational
   objection that joint optimization over "the amalgamated risk of all
   conjunctions in the near future" was considered and judged unnecessary.
   Our answer is Chen et al. (below): at mega-constellation scale the
   cascade is measured, large, and not handled by the outer loop.
3. **C3 is only meaningful with a bounded maneuver envelope.** The bound is
   stated with respect to an explicit per-satellite delta-v budget `D_i`;
   an unbounded secondary makes completeness vacuous.
4. **C4 must cite the classical theorems as background.** The contribution
   is the structural proof that the fleet program satisfies the regularity
   conditions, plus the first computed frontier for this problem class.

---

## The motivating evidence

**Chen et al., "Instability of Self-Driving Satellite Mega-Constellation,"
arXiv:2406.06068 (2024)** — https://arxiv.org/abs/2406.06068

- One external avoidance event triggered up to **41 induced maneuvers**.
- **81.4 %** of Starlink-on-Starlink maneuvers are cascade-induced.
- **79.5 %** of network lifetime lost to cascades.
- A distributed bilateral-control heuristic extends network lifetime 8x.

This is the empirical case that induced conjunctions are the dominant
maneuver driver at mega-constellation scale, and therefore that handling
them in an outer re-screening loop is not good enough. It is the single
most important citation for the project's motivation.

**Pavanello, Pirovano, Armellin, De Vittori & Di Lizia, arXiv:2406.03654** —
the flagship multi-encounter CAM paper — states its assumption in as many
words:

> "from an operational point of view, there is no reason to suppose that a
> close approach with a particular secondary is likely to promote a close
> approach with some other secondary."

Chen et al. measure that exact supposition to be false at scale. The gap
between those two sentences is this project.

**NASA/SP-20230002470 Rev.1** documents the universal operational practice:
generate the post-maneuver ephemeris, resubmit it for a fresh screening,
repeat. An outer loop with no optimality or completeness guarantee.

---

## Numerical baselines worth quoting

| Source | Result |
|---|---|
| Armellin 2021 (arXiv:2101.07403) | 28.1 mm/s (5 impulses) at Pc ≤ 1e-4 vs 288.1 mm/s (48 impulses) at Pc ≤ 1e-6; 6.4 → 431.1 mm/s as Pc sweeps 5e-2 → 5e-5 (~67x delta-v for ~1000x tighter risk), single geometry |
| Pavanello et al. 2024 (arXiv:2406.03654) | 5 encounters 60.05 → 62.32 mm/s for 20.0 % lower total Pc (+3.8 %); 7 encounters +1.1 % for −17.7 %; 10 encounters +3.4 % for −41.6 %; long-term +27.5 % for −76.4 % peak Pc. Also 2/5/7/10 conjunctions → 21.16 / 60.05 / 102.04 / 102.14 mm/s (the last three constraints were inactive) |
| Kuhl, Wang, Eddy & Kochenderfer 2025 (arXiv:2501.02667) | 90,000 simulated encounters; full-horizon MCTS 99.9 % avoidance at 0.01182 unsafe-encounter delta-v vs limited-horizon 98.2 % at 0.01546 (+30.8 % delta-v, −1.7 pp success) |
| Klinkrad et al. 2005 (ESA SP-587) | Tightening target-orbit accuracy cuts maneuver rate from ≥9/yr to ≤1 per 4 yr at Pc 1e-4, 98.83 % risk reduction; Pc 1e-3 still gives 98.17 % |
| SpaceX FCC filings (Dec 2024 – May 2025) | ~144,404–149,000 CAMs in six months, ~30–38 maneuvers/satellite/year; threshold tightened 1e-4 → 1e-5 → ~3e-7 |
| NASA CARA handbook | Actionable Pc 1e-4; post-mitigation target Pc ~3.1e-6, applied to newly created conjunctions as well |
| Rivero, Vazquez & Merz SDC9 2025 | 5,369 Starlink satellites, 14,410,396 pairs; best filter 0 % false negatives at 75.47 % pair elimination; most configurations had thousands of false negatives |
| Bertsimas & Stellato (arXiv:1907.02206) | 2–3 orders of magnitude speedup vs Gurobi from predicted strategy + pre-factorized solve |
| Cauligi et al. CoCo (arXiv:2004.03736) | 1–2 orders of magnitude speedup vs MICP solvers, near-global optimality, includes a free-flying space-robot case |
| ESA Kelvins challenge (arXiv:2008.03069) | 162,634 train rows / 13,154 events; 24,484 test rows / 2,167 events; F2-weighted MSE over Pc ≥ 1e-6 |

---

## Required citations

1. Pavanello, Pirovano, Armellin, De Vittori, Di Lizia — arXiv:2406.03654
2. Armellin — arXiv:2101.07403, *Acta Astronautica* 186 (2021)
3. NASA/SP-20230002470 Rev.1 — Conjunction Assessment Best Practices Handbook
4. Chen et al. — arXiv:2406.06068
5. Bertsimas & Stellato — arXiv:1907.02206
6. Cauligi, Culbertson, Schmerling, Pavone — arXiv:2004.03736
7. Gasse, Chételat, Ferroni, Charlin, Lodi — arXiv:1906.01629
8. SDEICA — *Applied Sciences* 16(8):3707 (unverified, paywalled)
9. Rivero, Vazquez, Merz — SDC9 2025, paper 167
10. Klinkrad, Alarcon, Sanchez — ESA SP-587 (SDC4, 2005)
11. Kuhl, Wang, Eddy, Kochenderfer — arXiv:2501.02667
12. Mueller — AIAA 2009-2051; Mao, Szmuk, Açıkmeşe — IFAC 50(1):4063 (2017)
13. Hejduk & Pachura — NASA NTRS 20170007928
14. Uriot et al. — arXiv:2008.03069 (Kelvins challenge)
15. Pavanello et al. — "CAMmary" review, arXiv:2503.22555
16. Vega, Arrizabalaga, Watson, Manchester — arXiv:2510.19058
17. Schrijver — *Theory of Linear and Integer Programming* (Farkas/IIS)
18. Bertsekas — *Nonlinear Programming* (exact penalty)
