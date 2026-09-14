# Fleet Optimization Tradeoff Research Handoff

Saved 2026-09-12 for continuation by another agent.

## Research Goal

AEGIS's intended research contribution is not another conjunction feed. SOCRATES can provide many external-threat conjunction candidates. The unique contribution should be a coordinated fleet-wide maneuver planner:

> Given a network of simultaneous conjunctions, compute one coordinated set of maneuvers that minimizes total fleet delta-v while resolving all modeled risks, respecting mission constraints, and avoiding newly created conjunctions.

SOCRATES is an input and external validation source. AEGIS is the decision and optimization layer. Local screening may still be needed for intra-constellation events, including Starlink-to-Starlink encounters reportedly omitted by SOCRATES.

## Correct Optimization Framing

Minimizing fuel does not logically require worse induced-collision risk if safety is modeled as a hard constraint:

```text
minimize    sum_i ||delta-v_i||

subject to  post-maneuver Pc_j <= target Pc for every conjunction j
            no new conjunction above the chosen threshold
            each spacecraft remains inside its mission/station-keeping box
            per-spacecraft maneuver and fuel limits are respected
```

The tradeoff appears when:

- Safety is represented only as a soft penalty.
- No zero-induced-conjunction solution exists inside the fuel and mission constraints.
- The optimizer considers only the original conjunction set and does not re-screen the modified trajectories.
- Approximation, uncertainty, or convergence to a local rather than global optimum changes the realized result.

If delta-v alone were optimized without safety constraints, the mathematical optimum would be no maneuver. If induced conjunction count alone were optimized without a fuel objective, the solution could spend excessive fuel. For the research, safety should be a hard or lexicographically primary requirement, with delta-v minimized among feasible safe plans.

## Published Numerical Evidence

### Multiple-encounter low-thrust optimization

Source: Zeno Pavanello, Laura Pirovano, Roberto Armellin, Andrea De Vittori, and Pierluigi Di Lizia, "Collision Avoidance Maneuvers Optimization in the Presence of Multiple Encounters," arXiv:2406.03654.

- Paper page: https://arxiv.org/abs/2406.03654
- Public PDF: https://arxiv.org/pdf/2406.03654
- Relevant results: Tables 3 and 9.

The paper does not report newly induced conjunction counts. It compares solutions that reach different total or peak collision-probability margins. The results provide an empirical sensitivity estimate, not a universal Pareto frontier.

| Scenario | Delta-v change | Collision-probability change |
|---|---:|---:|
| Five encounters, more conservative solution | +3.8% | 20.0% lower total Pc |
| Seven encounters, more conservative solution | +1.1% | 17.7% lower total Pc |
| Ten encounters, more conservative solution | +3.4% | 41.6% lower total Pc |
| Long-term encounter, seven-component uncertainty model | +27.5% | 76.4% lower peak Pc |

Underlying values:

- Five encounters: 60.05 to 62.32 mm/s; total Pc 1.001e-6 to 0.801e-6.
- Seven encounters: 102.04 to 103.18 mm/s; total Pc 1.001e-6 to 0.824e-6.
- Ten encounters: 102.14 to 105.58 mm/s; total Pc 0.998e-6 to 0.583e-6.
- Long-term, seven mixands: 367.77 to 469.08 mm/s; peak instantaneous Pc 0.829e-6 to 0.196e-6.

The last comparison reached a different local solution, so its 27.5% fuel premium is not a clean exchange rate. The short-term cases suggest that a large extra safety margin can sometimes cost only about 1-4% more delta-v.

The same paper's Starlink-like short-term case also shows that not every additional conjunction increases fuel cost:

- 2 conjunctions: 21.16 mm/s.
- 5 conjunctions: 60.05 mm/s.
- 7 conjunctions: 102.04 mm/s.
- 10 conjunctions: 102.14 mm/s.

Going from seven to ten conjunctions added only 0.10 mm/s, about 0.1%, because the final three constraints were not active in the optimal solution. This supports graph/constraint analysis that identifies which conjunctions actually drive a fleet maneuver.

### Decision timing, fuel, and avoidance success

Source: William Kuhl, Jun Wang, Duncan Eddy, and Mykel Kochenderfer, "Markov Decision Processes for Satellite Maneuver Planning and Collision Avoidance," arXiv:2501.02667.

- Paper page: https://arxiv.org/abs/2501.02667
- Public PDF: https://arxiv.org/pdf/2501.02667
- Relevant result: Table 4, based on 90,000 simulated encounters.

| Policy | Safe-encounter delta-v | Unsafe-encounter delta-v | Avoidance success |
|---|---:|---:|---:|
| Full-horizon MCTS, stochastic depth | 0.01095 | 0.01182 | 99.9% |
| Limited-horizon MCTS, stochastic depth | 0.01043 | 0.01546 | 98.2% |

Relative effect of the limited-horizon policy:

- Saved 4.7% delta-v on safe/false-alarm encounters.
- Used 30.8% more delta-v on genuinely unsafe encounters.
- Lost 1.7 percentage points of avoidance success.
- Increased modeled failure frequency from 0.1% to 1.8%, an 18x relative increase.

An extreme late-decision rule had 93.9% success versus 100% for the earliest rule, while its unsafe-encounter delta-v increased from 0.01173 to 0.03656, a 212% increase. Its safe-encounter delta-v also increased 66%, from 0.01109 to 0.01842. This result is specific to the paper's simulated update process and along-track-only maneuvers, but it demonstrates that waiting to avoid unnecessary action can make eventual maneuvers much more expensive while reducing safety.

## Current AEGIS Behavior and Gap

Relevant implementation:

- `aegis/src/aegis/maneuver/planner.py`
- `aegis/src/aegis/maneuver/rescreen.py`
- `aegis/src/aegis/core/maneuver.py`

Current behavior:

- `plan_maneuvers` solves a fleet linear program whose objective is absolute delta-v plus a configurable penalty on unresolved-conjunction slack.
- It considers along-track burns at a small set of candidate epochs.
- `rescreen_until_stable` applies the proposed burns, re-screens the catalog, and repeats until no unseen WATCH/ACT pair appears or the iteration limit is reached.
- `ManeuverPlan.converged` records only whether a fixed point was reached.

Missing research metrics:

- Count of newly induced conjunctions after each plan.
- Maximum and aggregate Pc of newly induced conjunctions.
- Extra delta-v required to eliminate those new conjunctions.
- Comparison against independent or greedy pair-by-pair planning.
- A measured fuel-versus-induced-risk Pareto frontier.
- Proof or empirical evidence that the final re-screen covers all relevant fleet pairs and time epochs.

AEGIS therefore cannot yet provide an honest project-specific number for how much fuel zero-induced-conjunction planning costs. Producing that number should be a central experiment, not an assumption.

## Recommended Next Experiment

For the same fixed fleet scenarios and input conjunction graph, compare at least these planners:

1. **Greedy pairwise baseline:** resolve each conjunction independently in TCA order without global re-optimization.
2. **Fuel-only original-set optimizer:** minimize total delta-v subject only to resolving the initially supplied conjunctions.
3. **Fleet-safe optimizer:** minimize total delta-v subject to resolving initial conjunctions and producing zero new above-threshold conjunctions after full re-screening.
4. **Risk-first/lexicographic optimizer:** first minimize unresolved and induced risk, then minimize delta-v among equally safe plans.

Run each across many deterministic synthetic and historical/replay scenarios. At minimum, report:

- Total delta-v and estimated propellant.
- Original conjunctions resolved/unresolved.
- Newly induced conjunction count.
- Maximum new Pc and aggregate new Pc.
- Mission-box violations.
- Maneuvering spacecraft and burn counts.
- Solver time and re-screen iterations.
- Feasibility rate.

The central result is the safety premium:

```text
safety premium (%) =
  (delta-v_zero-induced - delta-v_fuel-only)
  / delta-v_fuel-only * 100
```

Report its median, 90th percentile, 99th percentile, worst case, and infeasible fraction rather than only its mean. Stratify results by conjunction-graph structure, such as number of active edges, largest connected component, node degree, and TCA overlap. The working hypothesis is that sparse scenarios will often have a small safety premium, potentially comparable to the published 1-4% examples, while tightly coupled encounter clusters will produce much larger premiums or become infeasible.

## Existing Project Documents

- `docs/starlink-full-catalog-screening-report.md`: SOCRATES ingestion, catalog coverage, caching, and limitations.
- `docs/fleet-trajectory-redirect-plan.md`: original pivot to fleet-wide fuel-optimal trajectory coordination and prior architecture discussion.

## Repository State Warning

At save time the `main` worktree contained substantial pre-existing modified and untracked files. They were not changed, staged, reverted, or committed as part of this research handoff. The only intended new file from this save is this document.
