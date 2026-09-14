"""Solving the fleet program: LP and MILP backends, the certified sequential
refinement, and the lazy-constraint loop.

Three algorithms live here, and each one exists to deliver a guarantee.

**1. Backends.** :func:`solve_lp` wraps ``scipy.optimize.linprog`` with the
HiGHS simplex first because it returns dual multipliers, which everything
downstream -- the exact-penalty threshold, the safety-premium frontier slope,
the Farkas explanation, and the GNN's regression target -- depends on.
``highs-ipm`` and ``highs-ds`` follow as fallbacks. :func:`solve_milp` wraps
``scipy.optimize.milp``; mixed-integer programs have no useful duals, and the
result says so rather than returning silently empty numbers.

**2. Certified sequential refinement** (:func:`sequential_solve`). The
keep-out constraint :math:`\\lVert P_j(d_j + B_j x)\\rVert \\ge \\rho_j` is
reverse-convex. Replacing it with the supporting-hyperplane restriction
:math:`u_j^{\\top}P_j(d_j + B_j x) \\ge \\rho_j` is a standard convexification
-- introduced for ellipsoidal keep-out zones by Mueller (AIAA 2009-2051),
formalised as projection-and-linearisation by Mao, Szmuk & Acikmese (IFAC
2017), and applied to collision avoidance by Armellin (Acta Astronautica 186,
2021). It is cited here, not claimed.

What this implementation adds is the *bookkeeping* that makes the restriction
trustworthy at fleet scale:

* Because the restriction is an **inner** approximation (Cauchy-Schwarz), any
  feasible point of the LP satisfies the true constraint under the linearised
  dynamics -- for *any* direction, including a badly chosen or
  machine-learned one. Safety is therefore a property of the formulation, not
  of the direction supplier.
* Updating :math:`u_j` to the realised direction keeps the previous solution
  feasible, so the cost sequence is **non-increasing** and converges
  (Proposition 3 in :mod:`aegis.fleetopt.bplane`). The cost history is
  recorded and asserted, so a regression in the refinement shows up as a
  non-monotone sequence rather than as a quietly worse plan.

**3. Lazy constraints** (:func:`lazy_solve`). Solve with a subset of the
safety rows, check every omitted row, add the violated ones, repeat. At
termination the point is feasible for the full problem, and its cost is a
lower bound on the full optimum because every intermediate program is a
relaxation -- so the result is **exactly optimal for the full problem**,
whatever subset was guessed. That is what lets a learned active-set predictor
accelerate the solve without being trusted: a bad prediction costs extra
rounds, never correctness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, linprog, milp

from .bplane import MissSensitivity, refine_direction
from .errors import SolverBackendError
from .latent import LatentConstraint
from .problem import FleetProblem, LinearProgramData, assemble

__all__ = [
    "LpSolution",
    "ScpResult",
    "LazyResult",
    "solve_lp",
    "solve_milp",
    "solve_problem",
    "sequential_solve",
    "lazy_solve",
    "true_residuals",
]

#: Backends tried in order. HiGHS dual simplex first because it reports duals.
_LP_BACKENDS = ("highs", "highs-ds", "highs-ipm")

_FEASIBILITY_TOL = 1e-9


@dataclass
class LpSolution:
    """One solve, with duals when the backend provides them."""

    status: str
    objective: float
    z: np.ndarray
    duals: dict[str, float] = field(default_factory=dict)
    solve_time_s: float = 0.0
    backend: str = ""
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "optimal"

    @property
    def has_duals(self) -> bool:
        return bool(self.duals)

    def slack(self, data: LinearProgramData, prefix: str) -> dict[str, float]:
        """Slack values keyed by the bare row identity.

        ``prefix`` is ``"resolve"`` or ``"latent"``. The returned keys drop both
        the ``slack:`` marker and the category, so a resolve key is the bare
        conjunction id and a latent key is the bare ``latent:<pair>@...``
        label -- the same keys :func:`true_residuals` uses, so the two can be
        joined without string surgery at the call site.
        """
        if prefix not in ("resolve", "latent"):
            raise SolverBackendError(f"unknown slack prefix {prefix!r}")
        # Latent slack columns are labelled "slack:latent:<pair>@<iso>:<kind>"
        # and the key must keep the "latent:" part, because that is what
        # LatentConstraint.label is and what true_residuals keys on. Resolve
        # columns are "slack:resolve:<id>" and the key is the bare id. Matching
        # on a bare "slack:" prefix for the latent case -- which an earlier
        # version did -- silently swept up every resolve column as well, and
        # the reported induced shortfall was the sum of both.
        selector = f"slack:{prefix}:"
        strip = len("slack:") if prefix == "latent" else len(selector)
        return {
            label[strip:]: float(self.z[index])
            for index, label in enumerate(data.col_labels)
            if label.startswith(selector)
        }


def _normalize_duals(
    marginals: np.ndarray | None, row_labels: list[str]
) -> dict[str, float]:
    """Report duals as non-negative marginal values of tightening a row.

    ``scipy``'s HiGHS wrapper reports ``ineqlin.marginals`` for rows written as
    ``A_ub x <= b_ub``, with a non-positive sign convention. Safety rows were
    negated during assembly so that a ``>=`` requirement became ``<=``; the net
    effect is that ``-marginal`` is the increase in objective per unit increase
    in the requirement, which is the number an operator wants to read.
    """
    if marginals is None:
        return {}
    values = np.asarray(marginals, dtype=float)
    if values.shape[0] != len(row_labels):
        return {}
    return {
        label: float(-value) if abs(value) > 0.0 else 0.0
        for label, value in zip(row_labels, values)
    }


def solve_lp(data: LinearProgramData) -> LpSolution:
    """Solve a continuous program, preferring a backend that returns duals."""
    started = time.perf_counter()
    # linprog wants a sequence of (lo, hi) pairs with None for infinity; only
    # milp accepts a Bounds object. Passing the wrong one fails with a
    # TypeError that reads like a data problem, so it is worth being explicit.
    bounds = data.bounds
    last_message = ""
    for backend in _LP_BACKENDS:
        try:
            result = linprog(
                data.c,
                A_ub=data.a_ub if data.n_rows else None,
                b_ub=data.b_ub if data.n_rows else None,
                bounds=bounds,
                method=backend,
            )
        except (ValueError, TypeError) as error:
            last_message = f"{backend}: {error}"
            continue
        elapsed = time.perf_counter() - started
        if result.status == 0 and result.x is not None:
            marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
            return LpSolution(
                status="optimal",
                objective=float(result.fun),
                z=np.asarray(result.x, dtype=float),
                duals=_normalize_duals(marginals, data.row_labels),
                solve_time_s=elapsed,
                backend=backend,
                message=str(getattr(result, "message", "")),
            )
        if result.status == 2:
            return LpSolution(
                status="infeasible",
                objective=float("inf"),
                z=np.zeros(data.n_cols),
                solve_time_s=elapsed,
                backend=backend,
                message=str(getattr(result, "message", "")),
            )
        if result.status == 3:
            return LpSolution(
                status="unbounded",
                objective=float("-inf"),
                z=np.zeros(data.n_cols),
                solve_time_s=elapsed,
                backend=backend,
                message=str(getattr(result, "message", "")),
            )
        last_message = f"{backend}: {getattr(result, 'message', 'no message')}"

    return LpSolution(
        status="failed",
        objective=float("nan"),
        z=np.zeros(data.n_cols),
        solve_time_s=time.perf_counter() - started,
        backend="none",
        message=last_message or "every linprog backend refused the problem",
    )


def solve_milp(data: LinearProgramData) -> LpSolution:
    """Solve a mixed-integer program. Duals are not available and not faked."""
    started = time.perf_counter()
    constraints = (
        [LinearConstraint(data.a_ub, -np.inf, data.b_ub)] if data.n_rows else []
    )
    try:
        result = milp(
            c=data.c,
            constraints=constraints,
            integrality=data.integrality,
            bounds=Bounds(lb=data.lower, ub=data.upper),
        )
    except (ValueError, TypeError) as error:
        raise SolverBackendError(f"scipy.optimize.milp refused the problem: {error}") from error

    elapsed = time.perf_counter() - started
    if result.status == 0 and result.x is not None:
        return LpSolution(
            status="optimal",
            objective=float(result.fun),
            z=np.asarray(result.x, dtype=float),
            solve_time_s=elapsed,
            backend="highs-milp",
            message=str(getattr(result, "message", "")),
        )
    status = {2: "infeasible", 3: "unbounded"}.get(int(result.status), "failed")
    return LpSolution(
        status=status,
        objective=float("inf") if status == "infeasible" else float("nan"),
        z=np.zeros(data.n_cols),
        solve_time_s=elapsed,
        backend="highs-milp",
        message=str(getattr(result, "message", "")),
    )


def solve_problem(
    problem: FleetProblem,
    directions: dict[str, np.ndarray] | None = None,
) -> tuple[LpSolution, LinearProgramData]:
    """Assemble and solve once, choosing the backend from ``problem.integer``."""
    data = assemble(problem, directions)
    solution = solve_milp(data) if data.is_integer else solve_lp(data)
    return solution, data


def true_residuals(
    resolve: list[MissSensitivity],
    latent: list[LatentConstraint],
    x: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Exact first-order residuals, computed from the norms, not the rows.

    This never reuses the linearized rows the solver saw. It is the
    independent check that the restriction did what Proposition 2 promises.
    """
    resolve_residuals = {s.conjunction_id: s.residual_km(x) for s in resolve}
    latent_residuals = {c.label: c.residual_km(x) for c in latent}
    return resolve_residuals, latent_residuals


@dataclass
class ScpResult:
    """Outcome of the certified sequential refinement."""

    solution: LpSolution
    data: LinearProgramData
    x: np.ndarray
    directions: dict[str, np.ndarray]
    cost_history: list[float] = field(default_factory=list)
    direction_change_history: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    monotone: bool = True
    resolve_residuals: dict[str, float] = field(default_factory=dict)
    latent_residuals: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def objective(self) -> float:
        return self.solution.objective

    @property
    def worst_resolve_residual_km(self) -> float:
        return min(self.resolve_residuals.values()) if self.resolve_residuals else 0.0

    @property
    def worst_latent_residual_km(self) -> float:
        return min(self.latent_residuals.values()) if self.latent_residuals else 0.0

    def summary(self) -> dict:
        return {
            "status": self.solution.status,
            "objective": self.objective,
            "iterations": self.iterations,
            "converged": self.converged,
            "monotone": self.monotone,
            "cost_history": [round(value, 12) for value in self.cost_history],
            "max_direction_change": (
                max(self.direction_change_history) if self.direction_change_history else 0.0
            ),
            "worst_resolve_residual_km": self.worst_resolve_residual_km,
            "worst_latent_residual_km": self.worst_latent_residual_km,
            "backend": self.solution.backend,
            "notes": list(self.notes),
        }


def sequential_solve(
    problem: FleetProblem,
    *,
    max_iterations: int = 8,
    direction_tol: float = 1e-8,
    cost_tol: float = 1e-12,
    initial_directions: dict[str, np.ndarray] | None = None,
) -> ScpResult:
    """Refine the linearization directions until the cost stops improving.

    Every iterate is feasible for the true reverse-convex constraint, so the
    loop can be stopped at any point -- including after one iteration -- and
    still return a safe plan. See Propositions 2 and 3.
    """
    if max_iterations < 1:
        raise SolverBackendError("max_iterations must be at least 1")

    directions = dict(problem.nominal_directions())
    if initial_directions:
        for conjunction_id, vector in initial_directions.items():
            if conjunction_id in directions:
                candidate = np.asarray(vector, dtype=float).reshape(3)
                norm = float(np.linalg.norm(candidate))
                if norm > 1e-12:
                    directions[conjunction_id] = candidate / norm

    cost_history: list[float] = []
    change_history: list[float] = []
    notes: list[str] = []
    solution: LpSolution | None = None
    data: LinearProgramData | None = None
    x = np.zeros(problem.n_vars)
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        solution, data = solve_problem(problem, directions)
        if not solution.ok:
            notes.append(
                f"iteration {iterations} returned status {solution.status!r}: {solution.message}"
            )
            break
        x = data.decision_vector(solution.z)
        cost_history.append(float(solution.objective))

        updated: dict[str, np.ndarray] = {}
        max_change = 0.0
        for sensitivity in problem.resolve:
            new_direction = refine_direction(sensitivity, x)
            previous = directions[sensitivity.conjunction_id]
            max_change = max(max_change, float(np.linalg.norm(new_direction - previous)))
            updated[sensitivity.conjunction_id] = new_direction
        change_history.append(max_change)

        cost_settled = (
            len(cost_history) >= 2
            and abs(cost_history[-2] - cost_history[-1]) <= cost_tol * max(1.0, abs(cost_history[-1]))
        )
        if max_change <= direction_tol or cost_settled:
            converged = True
            directions = updated
            break
        directions = updated

    if solution is None or data is None:
        raise SolverBackendError("sequential_solve produced no solve at all")

    monotone = all(
        later <= earlier + 1e-9 * max(1.0, abs(earlier))
        for earlier, later in zip(cost_history, cost_history[1:])
    )
    if not monotone:
        notes.append(
            "cost history is not non-increasing, which contradicts Proposition 3 -- "
            "treat the result as suspect and investigate the refinement"
        )

    resolve_residuals, latent_residuals = true_residuals(problem.resolve, problem.latent, x)
    return ScpResult(
        solution=solution,
        data=data,
        x=x,
        directions=directions,
        cost_history=cost_history,
        direction_change_history=change_history,
        iterations=iterations,
        converged=converged,
        monotone=monotone,
        resolve_residuals=resolve_residuals,
        latent_residuals=latent_residuals,
        notes=notes,
    )


@dataclass
class LazyResult:
    """Outcome of the cutting-plane loop over held-back safety rows."""

    scp: ScpResult
    rounds: int
    rows_used: int
    rows_total: int
    rows_added_per_round: list[int] = field(default_factory=list)
    active_labels: list[str] = field(default_factory=list)
    closed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def objective(self) -> float:
        return self.scp.objective

    @property
    def reduction_fraction(self) -> float:
        if self.rows_total == 0:
            return 0.0
        return 1.0 - self.rows_used / self.rows_total

    def summary(self) -> dict:
        return {
            "rounds": self.rounds,
            "rows_used": self.rows_used,
            "rows_total": self.rows_total,
            "reduction_fraction": round(self.reduction_fraction, 6),
            "rows_added_per_round": list(self.rows_added_per_round),
            "closed": self.closed,
            "objective": self.objective,
            "notes": list(self.notes),
        }


def lazy_solve(
    problem: FleetProblem,
    *,
    active_guess: set[str] | None = None,
    max_rounds: int = 12,
    max_iterations: int = 8,
    initial_directions: dict[str, np.ndarray] | None = None,
    violation_tol: float = _FEASIBILITY_TOL,
) -> LazyResult:
    """Solve with a guessed subset of latent rows, then close the gap exactly.

    ``active_guess`` holds latent row labels believed to bind. Resolve rows are
    never held back: they are the conjunctions we were asked to fix, they are
    few, and omitting one would change the problem rather than relax it.

    The returned objective equals the full problem's optimum whenever the loop
    closes, for any guess. If the round limit is hit first, ``closed`` is False
    and the violated rows are listed in ``notes`` -- the result is then a
    relaxation and must not be reported as optimal.
    """
    all_latent = list(problem.latent)
    by_label = {constraint.label: constraint for constraint in all_latent}
    guess = set(active_guess or set())
    unknown = guess - set(by_label)
    active = [by_label[label] for label in sorted(guess & set(by_label))]

    notes: list[str] = []
    if unknown:
        notes.append(
            f"{len(unknown)} guessed row labels are not in this problem and were ignored"
        )

    added_per_round: list[int] = []
    scp: ScpResult | None = None
    closed = False
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        scp = sequential_solve(
            problem.with_latent(active),
            max_iterations=max_iterations,
            initial_directions=initial_directions,
        )
        if not scp.solution.ok:
            notes.append(f"round {rounds} solve status {scp.solution.status!r}")
            break

        held_back = [c for c in all_latent if c.label not in {a.label for a in active}]
        # The row the program enforces, not the physical separation. The
        # linearization is conservative, so a point can satisfy
        # ||d + Bx|| >= floor while violating u.(d + Bx) >= floor. Checking
        # the norm terminates the loop early on a solution the full program
        # would have rejected, and the "same objective" guarantee quietly
        # stops holding -- measured on 6 of 27 scenarios before this was fixed.
        violated = [c for c in held_back if c.linear_residual_km(scp.x) < -violation_tol]
        added_per_round.append(len(violated))
        if not violated:
            closed = True
            break
        violated.sort(key=lambda c: c.linear_residual_km(scp.x))
        active.extend(violated)

    if scp is None:
        raise SolverBackendError("lazy_solve produced no solve at all")
    if not closed:
        remaining = [
            c.label
            for c in all_latent
            if c.label not in {a.label for a in active}
            and c.linear_residual_km(scp.x) < -violation_tol
        ]
        notes.append(
            f"lazy loop did not close in {max_rounds} rounds; "
            f"{len(remaining)} rows remain violated, so this result is a relaxation "
            "and its objective is a lower bound, not the optimum"
        )

    return LazyResult(
        scp=scp,
        rounds=rounds,
        rows_used=len(active),
        rows_total=len(all_latent),
        rows_added_per_round=added_per_round,
        active_labels=[constraint.label for constraint in active],
        closed=closed,
        notes=notes,
    )
