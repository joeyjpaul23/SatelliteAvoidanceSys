# Step 14 contract: certified fleet maneuver optimization (AEGIS-FO)

Tester writes tests from this document. **Builder must not edit `tests/`.**
This contract introduces four new subpackages and does not modify the
semantics of any existing module. `aegis.maneuver.plan_maneuvers` stays
exactly as it is and becomes the `legacy-lp` baseline in the planner
comparison.

---

## 0. Research question this contract exists to answer

> Given a network of simultaneous conjunctions across a maneuverable fleet,
> compute one coordinated set of impulsive maneuvers that minimizes total
> fleet delta-v while (a) resolving every modeled conjunction to a target
> collision probability and (b) creating **zero** new conjunctions above
> threshold — and measure honestly what (b) costs.

The number we must be able to report, per scenario and as a distribution,
is the **safety premium**

```
pi = (dV*_zero-induced  -  dV*_fuel-only) / dV*_fuel-only
```

Everything in this contract exists to make `pi` computable, certified, and
explainable.

---

## 1. Notation and units

Units follow `aegis.constants`: kilometres, km/s, seconds, UTC datetimes.

| Symbol | Meaning |
|---|---|
| `F` | maneuverable fleet satellites |
| `i` | satellite index, `k` burn-slot index |
| `tau[i,k]` | burn epoch of slot `k` of satellite `i` |
| `dv[i,k]` | impulse 3-vector in satellite `i`'s RTN frame at `tau[i,k]`, km/s |
| `x` | stacked decision vector, length `3 * sum_i K_i` |
| `n_i` | satellite `i` mean motion, rad/s |
| `J` | indexed set of **resolve** conjunctions (must be made safe) |
| `L` | indexed set of **latent** pair/epoch constraints (must not be created) |
| `d_j` | nominal relative position at conjunction `j`, ECI km (secondary minus primary) |
| `w_j` | nominal relative velocity at conjunction `j`, ECI km/s |
| `rho_j` | required miss distance for `j`, km (from `aegis.maneuver.miss`) |
| `s_min` | induced-conjunction exclusion radius, km |
| `D_i` | per-satellite total delta-v budget, km/s |

---

## 2. `aegis.fleetopt.dynamics`

### 2.1 `cw_impulse_matrix(n_rad_s, lead_s) -> ndarray (3,3)`

The exact Clohessy-Wiltshire position-from-impulse block. With
`s = sin(n*sigma)`, `c = cos(n*sigma)`, `sigma = lead_s`:

```
Phi(n, sigma) =
  [ s/n            2(1-c)/n        0     ]
  [ 2(c-1)/n       (4s - 3 n sigma)/n    0 ]
  [ 0              0               s/n   ]
```

Rows are (radial, transverse, normal) **displacement**; columns are
(radial, transverse, normal) **impulse**. Requirements:

- `Phi[1,1]` must equal `aegis.maneuver.cw.along_track_response_km(n, sigma, 1.0)`
  to within 1e-12 for every `sigma` tested.
- Column 2 of `Phi` must equal
  `clohessy_wiltshire_state(n, sigma, zeros(3), [0,1,0])[0]`; column 1 the
  same with `[1,0,0]`; column 3 with `[0,0,1]` — all to 1e-12.
- `lead_s < 0` returns the zero matrix (an impulse cannot act before it is
  applied). `lead_s == 0` returns the zero matrix.
- `abs(n_rad_s) < 1e-15` degenerates to `sigma * I3` (rectilinear limit).

### 2.2 `cw_impulse_norm` and `cw_impulse_norm_bound`

`cw_impulse_norm(n_rad_s, lead_s)` is the exact spectral norm
`||Phi(n, sigma)||_2` and must agree with `numpy.linalg.norm(Phi, 2)` to
1e-10.

It is **not** monotone in `sigma`: the growth rate of the dominant
transverse term is `3 - 4 cos(n sigma)`, negative whenever
`cos(n sigma) > 3/4`, so the norm dips just after every whole orbit of lead
time. Tester must assert that a non-monotonicity exists (a 60000-point
sweep over 6 orbits contains at least one strictly decreasing step) so the
bound below is never "simplified" back into a pointwise evaluation.

`cw_impulse_norm_bound(n_rad_s, lead_s, *, mode="analytic"|"sampled")`
returns a monotone upper envelope of `max_{0<=u<=sigma} ||Phi(n,u)||_2`.

- `mode="analytic"` returns the closed form

  ```
  M(n, sigma) = sqrt( (3 n sigma + 4)^2 + 34 ) / n
  ```

  justified by `||Phi||_2 <= ||Phi||_F` with `2 sin^2 <= 2`,
  `8 (1-cos)^2 <= 32`, `|4 sin(n sigma) - 3 n sigma| <= 4 + 3 n sigma`.
- `mode="sampled"` returns a sampled running maximum inflated by
  `CW_NORM_LIPSCHITZ * delta / 2`, where
  `CW_NORM_LIPSCHITZ = sqrt(51)` bounds `d/d(sigma) ||Phi||` (the
  derivative matrix has squared Frobenius norm `10c^2 - 24c + 17 <= 51`),
  capped by the analytic value.

Tester requirements, for every mean motion in
`{0.0008, 0.0011, 0.00125, 0.0015}` rad/s sampled over at least 4 orbits:

- both modes dominate the true running maximum `np.maximum.accumulate` of
  the exact norm (tolerance 1e-9);
- the analytic mode is non-decreasing in `sigma` to 1e-12;
- the analytic mode over-estimates the exact norm by at most `12 / n`
  seconds;
- `lead_s <= 0` returns exactly 0 in both modes.

### 2.3 `BurnGrid`

```python
@dataclass(frozen=True)
class BurnGrid:
    satellite_ids: tuple[str, ...]
    epochs: dict[str, tuple[datetime, ...]]     # per satellite, ascending
    mean_motion: dict[str, float]               # rad/s
    axes: int = 3                               # 1 => along-track only
```

- `n_vars` = `axes * sum(len(epochs[s]) for s in satellite_ids)`.
- `index(satellite_id, slot, axis) -> int` gives the column in `x`.
  Ordering is satellite-major, then slot, then axis (R, T, N).
- `slot_epochs(satellite_id)` returns the tuple.
- Construction rejects: unknown satellite in `epochs`, non-ascending
  epochs, `axes not in (1, 3)`, non-positive mean motion.
- `build_burn_grid(objects, earliest_tca, *, burn_slots, min_lead_orbits, axes, now=None)`
  reproduces the existing planner's slot layout: working backwards from the
  satellite's earliest involved TCA, the first slot is `min_lead_orbits`
  periods before TCA and subsequent slots step back half an orbit. Slots
  strictly before `now` are dropped; if all are dropped the satellite is
  excluded and the reason is recorded.

### 2.4 `displacement_operator(grid, satellite_id, n_rad_s, epoch) -> ndarray (3, n_vars)`

The matrix `Psi_i(t)` with `delta_r_i(t) = Psi_i(t) @ x` in satellite `i`'s
**RTN frame at time `t`**. Block `k` is `Phi(n_i, t - tau[i,k])` (zero when
`tau[i,k] > t`), placed in the columns owned by `(i, k)`. Columns belonging
to other satellites are zero. When `grid.axes == 1` only the transverse
column of each `Phi` block is kept.

Requirement: for a single satellite with one slot and `axes=1`, the
`[1, :]` row times a scalar `dv` must reproduce `along_track_response_km`.

### 2.5 `eci_displacement_operator(grid, satellite_id, n_rad_s, epoch, state) -> ndarray (3, n_vars)`

`R(t) @ displacement_operator(...)` where `R = state.rtn_to_eci`. Must
satisfy `||eci_op @ x|| == ||rtn_op @ x||` to 1e-12 for random `x`
(rotation preserves norm).

---

## 3. `aegis.fleetopt.bplane`

### 3.1 `bplane_projector(relative_velocity_km_s) -> ndarray (3,3)`

`P = I3 - w_hat @ w_hat.T`. Requirements: symmetric, idempotent
(`P @ P == P` to 1e-12), `P @ w == 0` to 1e-12, eigenvalues `(0, 1, 1)`,
and a zero relative velocity returns `I3`.

### 3.2 `MissSensitivity`

```python
@dataclass(frozen=True)
class MissSensitivity:
    conjunction_id: str
    miss_vector_km: np.ndarray        # (3,) P @ d_j, ECI
    sensitivity: np.ndarray           # (3, n_vars) P @ B_j
    required_miss_km: float
    nominal_miss_km: float            # ||P @ d_j||
    relative_speed_km_s: float
```

`build_miss_sensitivity(conjunction, assessment, grid, mean_motion, required_miss_km)`
assembles `B_j = Psi_secondary(t_j) - Psi_primary(t_j)` in ECI (each term
zero for a non-maneuverable or ungridded object) and returns the projected
quantities.

**Proposition 1 (first-order miss distance).** The post-maneuver minimum
separation near `t_j` is

```
m_j(x) = || P_j ( d_j + B_j x ) ||  +  O(||displacement||^2)
```

Tester requirement (this is the headline physics check): construct a
two-satellite along-track conjunction, choose a `dv`, and compare
`m_j(x)` against the **SGP4 ground truth** obtained by
`apply_plan` + `screen`. Relative error must be below 15 % for
displacements up to 5 km and below 5 % for displacements up to 1 km.

**Proposition 2 (conservative affine restriction).** For any unit `u`,

```
u.T @ P_j @ (d_j + B_j x) >= rho_j    ==>    || P_j (d_j + B_j x) || >= rho_j
```

by Cauchy-Schwarz. Therefore every point feasible for the linear program is
safe under the linearized dynamics — the linearization is an **inner**
approximation, never an outer one. Tester requirement: for 1000 random
`(u, d, B, x)` draws, whenever the linear form holds the norm form holds.

### 3.3 `linearize(sensitivity, direction) -> (row, rhs)`

Returns `row = direction @ sensitivity.sensitivity` (length `n_vars`) and
`rhs = required_miss_km - direction @ miss_vector_km`, so the constraint is
`row @ x >= rhs`. `direction` is normalized internally; a zero direction
raises `FleetOptError`.

### 3.4 `refine_direction(sensitivity, x) -> ndarray (3,)`

`P_j (d_j + B_j x)` normalized. Returns the nominal direction when the
perturbed miss vector is numerically zero.

**Proposition 3 (monotone SCP).** If `x*` solves the LP linearized at
directions `u`, and `u'_j = refine_direction(j, x*)`, then `x*` is still
feasible for the LP linearized at `u'`, hence the optimal cost is
non-increasing across SCP iterations and the sequence converges. Tester
requirement: over at least 20 random multi-conjunction problems, the
recorded SCP cost sequence must be non-increasing to within 1e-9 and every
iterate must satisfy the true norm constraints.

---

## 4. `aegis.fleetopt.reachability`

### 4.1 `reach_radius_km(n_rad_s, budget_km_s, earliest_burn_epoch, epoch) -> float`

```
rho_i(t) = D_i * max_{tau in [tau_min, t]} || Phi(n_i, t - tau) ||_2
```

Implemented as `D_i * cw_impulse_norm_bound(n_i, t - tau_min, mode=...)`.
Because `||Phi||_2` is **not** monotone in lead time, the maximum over
`tau` is taken by the envelope function, not by evaluating the norm at
`tau_min`. Returns 0 for `t <= tau_min`. The default mode is `"analytic"`;
`"sampled"` is available when gate tightness matters more than evaluation
cost.

**Proposition 4 (displacement bound).** For any admissible plan with
`sum_k ||dv[i,k]|| <= D_i`, `||delta_r_i(t)|| <= rho_i(t)`. Proof: triangle
inequality plus submultiplicativity of the spectral norm.

Tester requirement: for 200 random admissible plans, the realized
`||Psi_i(t) x||` never exceeds `reach_radius_km` (1e-9 tolerance).

### 4.2 `certified_gate_km(pair_state, s_min, rho_a, rho_b) -> float`

`s_min + rho_a(t) + rho_b(t)`.

**Proposition 5 (candidate-set completeness).** If the nominal separation
of a pair satisfies `||d_ab(t)|| > certified_gate_km(t)` for every `t` in
the window, no admissible plan can bring that pair within `s_min`.
Therefore screening the nominal catalog with the inflated, time-varying
gate enumerates **every** pair any admissible plan could turn into a
conjunction.

Tester requirement: construct a pair outside the gate and verify that a
brute-force sweep over 500 random admissible plans never drives it below
`s_min`.

### 4.3 `a_posteriori_gate_km(...)`

Same formula with `D_i` replaced by the **realized** `sum_k ||dv[i,k]||`
of a solved plan, times a configurable margin (default 1.5). Used to
re-certify a finished plan with a far smaller gate than the a-priori
budget implies. Must be `<=` the a-priori gate whenever realized usage is
below budget.

### 4.4 `asymptotic_gate_km(budget_km_s, lead_s)`

The quotable closed form `3 * D * lead_s` (+ `s_min` at the call site).
Must upper-bound `reach_radius_km` for `n * lead_s >= 2*pi` (one full
orbit of lead) for all tested mean motions.

---

## 5. `aegis.fleetopt.latent`

### 5.1 `LatentConstraint`

```python
@dataclass(frozen=True)
class LatentConstraint:
    pair_id: str
    epoch: datetime
    separation_km: float            # nominal ||d_ab(t)||
    direction: np.ndarray           # (3,) nominal d_ab(t) / ||d_ab(t)||
    sensitivity: np.ndarray         # (3, n_vars)
    floor_km: float                 # s_min + grid inflation gamma
    kind: str                       # "grid" | "bplane_minimum"
```

### 5.2 `enumerate_latent_constraints(...)`

Inputs: the burn grid, a nominal `PropagationGrid`, a `ReachabilityModel`,
`s_min`, the resolve-pair exclusion set, and a mode.

Modes are `"none"`, `"certified_minima"` (the default), `"dense_grid"`, and
`"both"`.

**`certified_minima`** emits one B-plane-projected row at each local minimum
of the nominal separation that falls inside the reachability gate, with

```
floor = s_min + tidal_inflation_km(step, gate, displacement_rate)
```

**`dense_grid`** emits one unprojected row per sample inside the gate with
`floor = s_min + 0.5 * MAX_RELATIVE_SPEED_KM_S * step`. It is a diagnostic,
not a planning mode, and the report must say so when the inflation exceeds
ten times `s_min`.

### 5.3 `tidal_inflation_km(step_s, gate_km, *, radius_km, displacement_rate_km_s)`

```
gamma = 0.5 * A * step^2 + Lambda * step
A     = 2 * MU_EARTH_KM3_S2 * gate_km / radius_km^3
Lambda = CW_NORM_LIPSCHITZ * (D_a + D_b)
```

`A` is the **tidal** relative acceleration of a pair already inside the gate,
not the full two-body acceleration difference: two objects within a hundred
kilometres of each other feel nearly the same gravity, and only the residual
bends their relative trajectory. `Lambda` bounds how fast the
maneuver-induced displacement changes within one step.

Tester requirements:

- `tidal_inflation_km(30, 100, displacement_rate_km_s=sqrt(51)*2e-3)` must be
  below 1.0 km, and the crude `grid_inflation_km(30)` must exceed 200 km —
  the two bounds differ by more than two orders of magnitude and the test
  must pin that, because the crude bound makes the constraint set vacuous.
- Monotone non-decreasing in `step_s` and in `gate_km`.
- Zero at `step_s == 0`.
- Negative inputs raise `FleetOptError`.

**Soundness check (the important one).** Take a latent pair and a plan
feasible for the `certified_minima` rows. Propagate the maneuvered catalog
with SGP4 on a grid ten times finer than the enumeration step and assert the
true separation never drops below `s_min`. Run this over at least 30
pair/plan combinations spanning both co-planar and cross-plane geometries.

---

## 6. `aegis.fleetopt.norms`

### 6.1 `l1_cost_rows(grid, axis_weights) -> CostModel`

Plus/minus split: each `x` column becomes two non-negative columns.
`axis_weights = (w_R, w_T, w_N)` defaults to `(3.0, 1.0, 5.0)` — radial and
cross-track burns are operationally penalized because radial response is
bounded and cross-track burns break slot geometry (see
`Maneuver.is_along_track_only` in `aegis.core.maneuver`). Weight 1.0 on all
axes recovers a pure L1.

### 6.2 `polyhedral_cone(order) -> ndarray (L, 3)`

Near-uniform unit directions from a `order`-times subdivided icosahedron
(`order=0` gives the 12 vertices, `order=1` gives 42, `order=2` gives 162).
Requirements: all rows unit norm to 1e-12; the **covering radius**
`max_v min_l angle(v, g_l)` over 5000 random unit `v` must be below
25 deg for `order=1` and below 13 deg for `order=2`.

### 6.3 `cone_cost_rows(grid, directions) -> CostModel`

One auxiliary magnitude variable `t[i,k]` per burn slot with rows
`g_l . dv[i,k] - t[i,k] <= 0`. The objective is `sum t`.

**Approximation quality.** `max_l g_l . v` underestimates `||v||_2` by at
most `1 - cos(covering_radius)`. Tester requirement: for 2000 random
vectors the ratio `max_l(g_l . v) / ||v||` lies in
`[cos(covering_radius), 1]`.

---

## 7. `aegis.fleetopt.problem`

### 7.1 `FleetProblem`

A pure-data description of the assembled program. No solver calls.

```python
@dataclass
class FleetProblem:
    grid: BurnGrid
    cost: CostModel                     # objective over the lifted variables
    resolve: list[MissSensitivity]
    latent: list[LatentConstraint]
    budgets: dict[str, float]           # per-satellite D_i
    per_burn_cap_km_s: float
    station_keeping_box_km: tuple[float, float]
    resolve_slack_penalty: float
    latent_slack_penalty: float
    induced_budget: float | None        # the epsilon in V*(epsilon); None = unconstrained
    ops_cost_per_satellite: float = 0.0 # MILP only
    ops_cost_per_burn: float = 0.0      # MILP only
    integer: bool = False
```

### 7.2 `assemble(problem, directions) -> LinearProgramData`

Returns a plain dataclass with `c, A_ub, b_ub, bounds, integrality,
row_labels, col_labels`. Requirements:

- Every row carries a label of the form `resolve:<id>`,
  `latent:<pair>@<iso>`, `budget:<sat>`, `cap:<sat>:<slot>:<axis>`,
  `sk:<sat>:<axis>:<sign>`, `cone:<sat>:<slot>:<facet>`,
  `induced-budget`, `link:<sat>` (MILP).
- Slack columns exist for every resolve and latent row. Slacks are
  **mandatory** — an infeasible geometry must return a plan with
  unresolved entries, never raise (this preserves the existing contract
  in `aegis.maneuver.planner`).
- `col_labels` are stable and machine-parseable.
- Assembly is pure: calling it twice with the same inputs returns
  numerically identical arrays.

---

## 8. `aegis.fleetopt.solver`

### 8.1 `solve_lp(data) -> LpSolution`

Wraps `scipy.optimize.linprog(method="highs")` with `highs-ds` and
`interior-point` fallbacks (mirroring the existing planner). Returns

```python
@dataclass
class LpSolution:
    status: str                 # "optimal" | "infeasible" | "unbounded" | "failed"
    objective: float
    x: np.ndarray               # lifted variables
    duals: dict[str, float]     # row label -> optimal multiplier (>= 0)
    solve_time_s: float
    backend: str
```

Duals come from `result.ineqlin.marginals` (sign-normalized so that
relaxing a `>=` row by one unit reduces the objective by `duals[label]`,
i.e. reported duals are non-negative). When a backend provides no duals
the dict is empty and `LpSolution.has_duals` is False.

### 8.2 `solve_milp(data) -> LpSolution`

`scipy.optimize.milp`. Duals are unavailable; `duals == {}` and
`has_duals is False`. `status` must still be one of the four strings.

### 8.3 `sequential_solve(problem, *, max_iterations=8, tol_km=1e-6) -> ScpResult`

The certified SCP of Proposition 3:

1. `directions[j] = nominal miss direction` (or a supplied warm start).
2. Assemble, solve, record cost.
3. `directions[j] = refine_direction(j, x)`.
4. Stop when the max direction change is below `tol_km`-equivalent angle,
   or the cost improvement is below tolerance, or iteration cap is hit.

`ScpResult` records `iterations`, `cost_history`, `max_direction_change`,
`converged`, the final `LpSolution`, and `true_residuals` (the exact
`||P(d + Bx)|| - rho` for every resolve row, and `||d + Bx|| - s_min` for
every latent row).

Requirements:

- `cost_history` must be non-increasing to 1e-9.
- Every iterate must satisfy `true_residuals >= -1e-9` on rows whose slack
  is zero.
- With `max_iterations=1` the result must still be safe (Proposition 2).

### 8.4 `lazy_solve(problem, *, active_guess, max_rounds=12) -> LazyResult`

Cutting-plane loop for learned active-set acceleration:

1. Solve with only the rows in `active_guess` (plus all structural rows).
2. Evaluate **every** omitted row at the solution.
3. Add violated rows (beyond 1e-9) and re-solve.
4. Terminate when no omitted row is violated.

**Guarantee.** The returned solution is optimal for the full problem. Proof:
at termination the solution is feasible for the full problem, and its cost
is a lower bound on the full optimum because the solved program is a
relaxation. Tester requirement: across at least 50 random problems,
`lazy_solve` with an arbitrary (even empty or adversarial) `active_guess`
must return the same objective as the full solve to 1e-7, and must report
`rounds <= max_rounds`.

---

## 9. `aegis.fleetopt.certify`

### 9.1 `verify_plan(problem, x) -> Certificate`

Independent re-evaluation that never reuses the solver's own rows:

```python
@dataclass
class Certificate:
    resolve_violations: list[tuple[str, float]]
    latent_violations: list[tuple[str, float]]
    budget_violations: list[tuple[str, float]]
    cap_violations: list[tuple[str, float]]
    station_keeping_violations: list[tuple[str, float]]
    linearized_safe: bool
    worst_margin_km: float
```

`linearized_safe` is True only when every list is empty.

### 9.2 `exact_penalty_threshold(solution) -> float`

`lambda* = max over resolve/latent row duals`.

**Proposition 6 (exact penalty).** For the LP `min c'x s.t. Ax >= b, x in X`
with optimal dual `y*`, and its penalized form
`min c'x + lambda 1'sigma s.t. Ax + sigma >= b, sigma >= 0`, every optimal
solution of the penalized form has `sigma = 0` whenever `lambda > ||y*||_inf`
and the hard problem is feasible. Hence the apparent delta-v-versus-risk
tradeoff under a soft penalty is an artifact of choosing
`lambda <= ||y*||_inf`.

Tester requirement: on a feasible problem, solving with
`lambda = 0.5 * lambda*` must produce non-zero slack, and with
`lambda = 2 * lambda*` must produce zero slack and the same objective as
the hard-constrained solve (1e-7).

### 9.3 `irreconcilable_subset(problem, directions, *, max_rows=40) -> Farkas`

When the hard-constrained problem is infeasible, find a minimal infeasible
subsystem by deletion filtering over the resolve and latent rows, and
return

```python
@dataclass
class Farkas:
    infeasible: bool
    resolve_rows: list[str]
    latent_rows: list[str]
    explanation: str            # "resolving A and B necessarily induces C"
    multipliers: dict[str, float]   # Farkas certificate y, z when available
```

Requirement: every row in the returned subset must be necessary — removing
any one of them must make the subsystem feasible.

---

## 10. `aegis.fleetopt.graph`

### 10.1 `build_conjunction_graph(assessed, objects) -> ConjunctionGraph`

Nodes are objects, edges are conjunctions, with edge attributes carrying
the assessment. Uses no third-party graph library.

### 10.2 `GraphMetrics`

```
edges, active_edges, nodes, maneuverable_nodes,
largest_component, component_count, max_degree, mean_degree,
tca_span_s, tca_overlap_fraction, intra_fleet_fraction,
coupling_number, conflict_dimension
```

- `coupling_number` is `cond(B B^T)` of the stacked, row-normalized
  **projected** resolve sensitivities (`+inf` when rank-deficient, reported
  as a large finite sentinel `1e18`). It is the structural predictor we
  will correlate with the safety premium.
- `conflict_dimension` is `rank([B_J; B_L]) - rank(B_J)`: how many extra
  independent directions the induced constraints pin down.
- `tca_overlap_fraction` is the fraction of conjunction pairs whose TCAs
  fall within one orbital period of each other.

All metrics must be finite or the documented sentinel, never NaN.

---

## 11. `aegis.fleetopt.planners`

### 11.1 Common interface

```python
class Planner(Protocol):
    name: str
    def plan(self, request: PlanRequest) -> FleetPlan: ...
```

`PlanRequest` carries the assessed catalog, the object list, `now`,
`target_pc`, `s_min`, budgets, burn-grid settings, and an optional
`LearnedHints`. `FleetPlan` **extends** the existing
`aegis.core.maneuver.ManeuverPlan` contract: it holds a `ManeuverPlan` in
`.maneuver_plan` (so every existing consumer keeps working) plus the
research fields:

```python
@dataclass
class FleetPlan:
    planner: str
    maneuver_plan: ManeuverPlan
    problem_summary: dict
    scp: ScpResult | None
    certificate: Certificate | None
    graph_metrics: GraphMetrics | None
    induced_measured: InducedReport | None
    duals: dict[str, float]
    exact_penalty_threshold: float | None
    farkas: Farkas | None
    solver_time_s: float
    notes: list[str]
```

### 11.2 Required planners

| name | behaviour |
|---|---|
| `no-maneuver` | zero burns; establishes the unmitigated risk baseline |
| `legacy-lp` | delegates to `aegis.maneuver.plan_maneuvers` unchanged |
| `greedy-pairwise` | resolve conjunctions one at a time in TCA order, each as an independent single-pair LP, summing burns; no coordination, no induced constraints |
| `fuel-only` | fleet SCP with resolve constraints only (`latent mode "none"`) |
| `fleet-safe` | fleet SCP with resolve **and** certified latent constraints |
| `lexicographic` | stage 1 minimizes total slack; stage 2 minimizes delta-v subject to slack <= stage-1 optimum (+1e-9) |
| `milp-ops` | `fleet-safe` plus binaries and `ops_cost_per_satellite` / `ops_cost_per_burn` |
| `pignn-active-set` | `fleet-safe` restricted to GNN-predicted active rows, closed by `lazy_solve`; must return the same objective as `fleet-safe` to 1e-7 |
| `pignn-warm-start` | `fleet-safe` with GNN-predicted initial SCP directions; must be safe regardless of prediction quality |
| `pignn-direct` | GNN output used verbatim, no LP; reports its own violation rate honestly |

Every planner must accept a scenario with zero conjunctions and return an
empty plan without raising. Every planner except `pignn-direct` must
produce a `Certificate` with `linearized_safe is True` whenever it reports
all conjunctions resolved.

### 11.3 `apply_plan(objects, fleet_plan, assessed) -> list[SpaceObject]`

Generalizes `aegis.maneuver.rescreen.apply_along_track_burns` to
three-axis burns. The along-track component maps to the same mean-anomaly
shift as today (so existing behaviour is preserved bit-for-bit when the
plan is along-track only). Radial and cross-track components map to
eccentricity/argument-of-perigee and inclination/RAAN changes through the
Gauss variational equations:

```
da     = 2 * dv_T / n                     (a-change from tangential impulse)
de     = (1/(n a)) * [ dv_R sin u + 2 dv_T cos u ]      (circular-orbit limit)
di     = dv_N cos(u) / (n a)
dRAAN  = dv_N sin(u) / (n a sin i)
```

with `u` the argument of latitude at the burn. Requirement: an
along-track-only plan must produce byte-identical `SpaceObject` copies to
`apply_along_track_burns` for the same plan.

---

## 12. `aegis.fleetopt.pareto`

### 12.1 `induced_frontier(problem, *, epsilons=None) -> Frontier`

Solve `V*(epsilon)` over a sequence of induced-risk budgets.

**Proposition 7 (frontier structure).** `V*(epsilon)` is convex,
piecewise-linear and non-increasing in `epsilon`, with left derivative
equal to minus the optimal multiplier of the `induced-budget` row. The
frontier is therefore exactly representable by finitely many breakpoints.

`Frontier` records `epsilons`, `costs`, `multipliers`, `breakpoints`, and
`is_convex` (checked numerically, tolerance 1e-7 relative).

Tester requirement: for at least 20 random problems the sampled frontier
must be non-increasing and convex to 1e-6 relative, and the slope between
consecutive samples must match the reported multipliers to 1e-4 relative
wherever the multiplier is non-zero.

### 12.2 `safety_premium(fuel_only, fleet_safe) -> PremiumRecord`

```
pi = (dv_safe - dv_fuel) / dv_fuel      when dv_fuel > 0
```

`PremiumRecord` also carries `dv_fuel_mm_s`, `dv_safe_mm_s`, both induced
counts, feasibility of each, and `infeasible` when `fleet-safe` cannot
resolve everything. `dv_fuel == 0` yields `premium = None` with
`reason="fuel_only_requires_no_maneuver"` rather than an infinity.

### 12.3 `premium_statistics(records) -> dict`

Median, p90, p99, worst, mean, count, `infeasible_fraction`,
`undefined_fraction`. Must never return NaN; empty input returns zeros and
`count=0`.

---

## 13. `aegis.scenarios`

### 13.1 `ScenarioSpec` / `Scenario`

Deterministic and reproducible: `Scenario` carries `scenario_id`, the
`SpaceObject` list, window start/duration, a `family` label, the `seed`,
and a `provenance` dict. **Synthetic scenarios must carry
`data_source="SYNTHETIC"`** so the existing mixed-source guards and the
`AEGIS_ALLOW_SYNTHETIC` policy still apply; the generator requires the
same `SyntheticAuthorization` object `aegis.ingest.synthetic` uses.

### 13.2 Required families

| family | structure being probed |
|---|---|
| `isolated-pair` | one conjunction, one maneuverable satellite |
| `chain` | conjunctions sharing satellites in a path graph |
| `star` | one satellite in `k` simultaneous conjunctions |
| `clique` | a tightly coupled cluster, all TCAs within one orbit |
| `intra-plane` | same-plane, low relative velocity (tests the 2D-Pc guard) |
| `crossing-planes` | high relative velocity, different RAAN |
| `debris-shower` | many non-maneuverable secondaries on one plane |
| `dense-shell` | large catalog, realistic Starlink-like shell |
| `replay-tle` | built from the committed TLE fixtures, no synthesis |

Every family must accept `seed` and produce byte-identical output for the
same seed. `generate(family, seed)` must be pure.

### 13.3 `scenario_digest(scenario) -> str`

SHA-256 over a canonical serialization, used as the cache and storage key.
Must be stable across processes and Python versions (no `hash()`, no dict
ordering dependence).

---

## 14. `aegis.store`

Offline-first, cloud-optional, never silently fake.

### 14.1 `ArtifactStore` protocol

`put(key, data, *, content_type) -> ArtifactRef`, `get(key) -> bytes`,
`exists(key)`, `list(prefix)`, `uri(key)`.

Implementations: `LocalArtifactStore` (content-addressed under a root,
always available), `S3ArtifactStore` (lazy `boto3` import),
`GCSArtifactStore` (lazy `google-cloud-storage` import). Requirements:

- `open_artifact_store()` reads `AEGIS_STORE_BACKEND`
  (`local` | `s3` | `gcs`), `AEGIS_STORE_ROOT`, `AEGIS_S3_BUCKET`,
  `AEGIS_GCS_BUCKET`, `AEGIS_STORE_PREFIX`.
- A missing cloud SDK or missing credentials raises
  `StoreUnavailableError` with an actionable message; it must **not**
  silently downgrade to local. `open_artifact_store(fallback=True)`
  downgrades explicitly and records a note.
- Keys are validated: no leading slash, no `..`, no control characters.
- `put` is idempotent for identical content (same digest, no rewrite).

### 14.2 `ExperimentStore`

`sqlite3` (stdlib) with an optional Postgres DSN via lazy `psycopg`.
Schema (migrations applied on open, version-tracked):

```
runs(run_id PK, created_at, git_sha, config_json, code_version, notes)
scenarios(scenario_digest PK, family, seed, n_objects, n_conjunctions, spec_json)
results(result_id PK, run_id, scenario_digest, planner, metrics_json,
        total_dv_mm_s, induced_count, resolved, unresolved, feasible,
        solver_time_s, created_at)
premiums(result_id PK, run_id, scenario_digest, premium, dv_fuel_mm_s,
         dv_safe_mm_s, infeasible)
artifacts(artifact_id PK, run_id, key, uri, digest, content_type, bytes)
model_versions(model_id PK, created_at, architecture_json, metrics_json,
               artifact_key, train_run_id)
```

Requirements: `record_result` is idempotent on
`(run_id, scenario_digest, planner)`; concurrent writers do not corrupt
(WAL mode, retry on lock); all timestamps ISO-8601 UTC; queries for the
premium distribution and per-family breakdown are provided as methods.

### 14.3 `DatasetWriter`

Parquet via `pyarrow`, sharded, with a JSON manifest listing shard keys,
row counts, schema version, and the scenario digests contained. Must
round-trip through the artifact store.

---

## 15. `aegis.ml` — the physics-informed conjunction GNN

### 15.1 `aegis.ml.torchphysics`

Differentiable re-implementations that must agree with the numpy/scipy
originals:

- `cw_impulse_matrix_torch(n, sigma)` — max abs error < 1e-10 vs
  `fleetopt.dynamics.cw_impulse_matrix`.
- `bplane_projector_torch(w)` — < 1e-12.
- `collision_probability_torch(sigma_major, sigma_minor, miss_x, miss_z, hbr)`
  — a differentiable Gauss-Chebyshev Alfano integral. **Max relative error
  below 1e-6 against `aegis.risk.alfano.collision_probability` across at
  least 500 sampled geometries spanning `Pc` from 1e-12 to 1e-2.** Must be
  finite and have finite gradients everywhere tested (no NaN from
  `erf` saturation).
- `required_miss_torch` — differentiable surrogate for
  `required_miss_distance_km`, with documented error.

### 15.2 `aegis.ml.features`

`tensorize(scenario, problem, labels=None) -> GraphSample`.

Node features (per object, at least): normalized mean motion, altitude,
eccentricity, inclination sin/cos, maneuverable flag, remaining budget,
slot count, lead time to its earliest TCA, degree, is-debris flag.
Edge features (per conjunction or latent constraint): log10 Pc, miss
distance, required miss, shortfall, relative speed, lead time, sigma major
/ minor, HBR, `cos` of the angle between each member's velocity and the
B-plane, the row norm of the projected sensitivity for each endpoint, edge
kind one-hot (`resolve` / `latent-grid` / `latent-bplane`), intra-fleet
flag. Every feature must be finite; normalization statistics are stored
with the dataset, never recomputed at inference.

### 15.3 `aegis.ml.model` — `ConjunctionPIGNN`

Pure PyTorch (no `torch_geometric`). Encoder MLPs for node and edge
features, `n_layers` rounds of attention-weighted message passing with
residual connections and layer norm, then heads:

| head | output | loss |
|---|---|---|
| `edge_active` | per-edge logit | BCE with positive weighting |
| `edge_dual` | per-edge scalar | Huber on `log1p(dual / scale)` |
| `node_dv` | per-node `3 * K_max` | Huber, masked to real slots |
| `global_feasible` | scalar logit | BCE |
| `global_cost` | scalar | Huber on `log1p` |
| `global_premium` | scalar | Huber |

Plus the **physics loss**, computed by pushing `node_dv` through
`torchphysics` and the stored sensitivity tensors:

```
L_phys = sum_j relu(rho_j - u_j' P_j (d_j + B_j x_hat))^2
       + sum_p relu(floor_p - u_p' (d_p + B_p x_hat))^2
       + w_dv * ||x_hat||_1
```

Requirements: deterministic forward pass given a seed; the model must
handle a graph with a single node and zero edges; parameter count reported;
`state_dict` round-trips; a `ModelConfig` dataclass fully determines the
architecture and is stored with the checkpoint.

### 15.4 `aegis.ml.train`

Training must run to completion on CPU within a few minutes for the
committed small dataset. Records per-epoch metrics, writes a checkpoint
and a metrics JSON to the artifact store, and registers a
`model_versions` row. Requirements: deterministic with a fixed seed (same
loss to 1e-5 on repeat); early stopping; the physics loss weight is
annealed; a held-out split by **scenario digest** (never by edge) so there
is no leakage.

### 15.5 `aegis.ml.infer`

`LearnedHints` with `active_edges`, `edge_probabilities`,
`initial_directions`, `predicted_dv`. Loading a checkpoint must verify the
feature schema version and raise on mismatch. Inference must work with no
checkpoint present by returning `None` hints, and every `pignn-*` planner
must then fall back to its uncertified-free exact equivalent with a note —
**never** silently produce a worse plan.

### 15.6 Calibration

`calibrate_threshold(model, dataset) -> float` picks the active-set
probability threshold achieving a target recall (default 0.999) on the
validation split, because a missed active row costs an extra lazy round
but never safety. Report the achieved recall and the average fraction of
rows retained.

---

## 16. `aegis.experiments`

### 16.1 `run_benchmark(config) -> BenchmarkReport`

Runs every configured planner over every configured scenario, records
metrics into the `ExperimentStore`, writes artifacts, and returns a report.
Requirements: resumable (skips `(run_id, scenario, planner)` already
recorded); per-result timeout with the failure recorded rather than
crashing the sweep; a deterministic `run_id` derived from the config digest
plus an explicit timestamp argument.

### 16.2 Metrics per `(scenario, planner)`

Exactly the handoff list, plus the new quantities:

```
total_dv_mm_s, total_propellant_g,
conjunctions_total, resolved, unresolved, worst_shortfall_km,
induced_count_predicted, induced_count_measured,
induced_max_pc, induced_aggregate_pc,
station_keeping_violations, maneuvering_satellites, total_burns,
solver_time_s, scp_iterations, lazy_rounds, rescreen_iterations,
feasible, certificate_safe, linearization_error_km_p95,
exact_penalty_threshold, coupling_number, premium
```

`induced_count_measured` comes from a **full re-screen** of the maneuvered
catalog with the original screening box, counting WATCH-or-above pairs
that were not present before. This is the ground truth that the predicted
count is validated against.

### 16.3 `aegis.experiments.cli`

`python -m aegis.experiments --help` must work. Subcommands at least:
`benchmark`, `frontier`, `scenario-list`, `report`, `validate`. Each writes
machine-readable JSON and a human-readable markdown summary.

### 16.4 Validation suite

`validate_linearization(scenario, plan)` applies the plan with
`apply_plan`, re-screens with SGP4, and compares the predicted
post-maneuver miss distance against the measured one for every resolve
row. Reports the error distribution. **A research claim about the
optimizer is only reported alongside this number.**

---

## 17. Honesty requirements (non-negotiable)

These mirror `docs/LIMITATIONS.md` and extend it:

1. Any reported `Pc` inherits TLE-grade covariance; the reports must repeat
   that the covariance source is `SYNTHETIC_TLE`.
2. `induced_count_predicted` and `induced_count_measured` are always
   reported together. A planner is never described as achieving zero
   induced conjunctions on the basis of its own constraints alone.
3. The linearization error distribution is reported with every delta-v
   claim.
4. When the GNN is used, the certified objective and the uncertified
   objective are both reported, and any violation by `pignn-direct` is
   printed rather than hidden.
5. No synthetic scenario may be produced without `SyntheticAuthorization`.
6. Cloud storage failures are surfaced, never papered over.
