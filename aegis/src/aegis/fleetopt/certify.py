"""Independent verification of a plan, and explanations when none exists.

Nothing in this module reuses the rows the solver saw. Every check is
recomputed from the underlying geometry, because a certificate that trusts
the assembler cannot catch an assembler bug -- and an assembler bug is
exactly the failure mode that would make an unsafe plan look safe.

Three things live here.

**Verification** (:func:`verify_plan`). Re-evaluates every physical
requirement from its norm form: the true reverse-convex miss constraint
:math:`\\lVert P_j(d_j + B_j x)\\rVert \\ge \\rho_j`, the true latent
separation, the per-satellite budget, the per-component cap, and the station
keeping box. A plan is ``linearized_safe`` only when all of them hold. Note
the word *linearized*: this certifies the plan against the linear dynamics
model, not against SGP4. The SGP4 check is a separate, empirical measurement
(:mod:`aegis.experiments`), and the two numbers are always reported together.

**The exact-penalty threshold** (:func:`exact_penalty_threshold`). The fleet
program carries mandatory slack on every safety row, penalised at
``lambda``. A natural reading of that is "fuel versus risk, traded off at
rate lambda". The reading is wrong, and LP duality says exactly why:

    **Proposition 6.** For ``min c'x s.t. Ax >= b, x in X`` with optimal dual
    ``y*``, and its penalised form ``min c'x + lambda 1'sigma`` subject to
    ``Ax + sigma >= b, sigma >= 0``, every optimal solution of the penalised
    form has ``sigma = 0`` whenever the hard problem is feasible and
    ``lambda > ||y*||_inf``.

    *Proof.* The dual of the penalised problem is the dual of the hard
    problem with the extra constraint ``y <= lambda 1``. If
    ``lambda > ||y*||_inf`` then ``y*`` is strictly interior to that box, so
    the dual optimum -- and hence the primal value -- is unchanged; and any
    ``sigma > 0`` costs ``lambda`` per unit while buying at most
    ``||y*||_inf < lambda`` per unit of objective relief, so it is strictly
    suboptimal.

The exact-penalty result itself is classical (see Bertsekas, *Nonlinear
Programming*). What is worth saying in this domain is the consequence: the
apparent delta-v-versus-risk tradeoff observed under a soft penalty is an
artifact of choosing ``lambda <= ||y*||_inf``, not a property of the physics.
Above the threshold the two formulations coincide. The real tradeoff, the one
that does not go away, is between delta-v and *induced* risk when no
zero-induced plan exists inside the budget -- and that one is measured by
:mod:`aegis.fleetopt.pareto`.

**Irreconcilability** (:func:`irreconcilable_subset`). When the hard-
constrained problem is genuinely infeasible, "infeasible" is a useless thing
to tell an operator. A deletion filter finds a minimal infeasible subsystem:
a set of rows such that the system is infeasible with all of them and
feasible without any one of them. Rendered in English that reads "resolving A
and B at the same time necessarily brings C within the exclusion radius",
which is actionable. Farkas' lemma is the mathematical basis (Schrijver,
*Theory of Linear and Integer Programming*); the contribution here is the
mapping back to named conjunctions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bplane import MissSensitivity
from .errors import CertificationError
from .latent import LatentConstraint
from .problem import FleetProblem, assemble, station_keeping_rows
from .solver import LpSolution, solve_lp

__all__ = [
    "Certificate",
    "verify_plan",
    "exact_penalty_threshold",
    "Farkas",
    "irreconcilable_subset",
    "binding_rows",
]

_TOL_KM = 1e-9
_TOL_KM_S = 1e-12


@dataclass
class Certificate:
    """Independent re-evaluation of every requirement at a candidate plan."""

    resolve_violations: list[tuple[str, float]] = field(default_factory=list)
    latent_violations: list[tuple[str, float]] = field(default_factory=list)
    budget_violations: list[tuple[str, float]] = field(default_factory=list)
    cap_violations: list[tuple[str, float]] = field(default_factory=list)
    station_keeping_violations: list[tuple[str, float]] = field(default_factory=list)
    resolve_margins: dict[str, float] = field(default_factory=dict)
    latent_margins: dict[str, float] = field(default_factory=dict)
    slack_absorbed: dict[str, float] = field(default_factory=dict)
    tolerance_km: float = _TOL_KM

    @property
    def linearized_safe(self) -> bool:
        """True only when nothing at all is violated.

        Rows whose shortfall was deliberately absorbed by slack are *not*
        counted as violations -- they are reported separately in
        ``slack_absorbed`` and in the plan's unresolved list. A plan that
        leaves conjunctions unresolved is honest, not unsafe; a plan that
        claims to resolve one and does not is unsafe.
        """
        return not (
            self.resolve_violations
            or self.latent_violations
            or self.budget_violations
            or self.cap_violations
            or self.station_keeping_violations
        )

    @property
    def worst_margin_km(self) -> float:
        margins = list(self.resolve_margins.values()) + list(self.latent_margins.values())
        return min(margins) if margins else 0.0

    @property
    def violation_count(self) -> int:
        return (
            len(self.resolve_violations)
            + len(self.latent_violations)
            + len(self.budget_violations)
            + len(self.cap_violations)
            + len(self.station_keeping_violations)
        )

    def summary(self) -> dict:
        return {
            "linearized_safe": self.linearized_safe,
            "violations": self.violation_count,
            "resolve_violations": [(k, round(v, 9)) for k, v in self.resolve_violations],
            "latent_violations": [(k, round(v, 9)) for k, v in self.latent_violations[:20]],
            "budget_violations": [(k, round(v, 12)) for k, v in self.budget_violations],
            "cap_violations": [(k, round(v, 12)) for k, v in self.cap_violations],
            "station_keeping_violations": [
                (k, round(v, 9)) for k, v in self.station_keeping_violations
            ],
            "worst_margin_km": round(self.worst_margin_km, 9),
            "slack_absorbed_count": len(self.slack_absorbed),
        }


def verify_plan(
    problem: FleetProblem,
    x: np.ndarray,
    *,
    slack: dict[str, float] | None = None,
    tolerance_km: float = _TOL_KM,
) -> Certificate:
    """Re-check every requirement from first principles at ``x``."""
    grid = problem.grid
    x = np.asarray(x, dtype=float).reshape(grid.n_vars)
    slack = dict(slack or {})
    certificate = Certificate(tolerance_km=tolerance_km)

    for sensitivity in problem.resolve:
        margin = sensitivity.residual_km(x)
        certificate.resolve_margins[sensitivity.conjunction_id] = margin
        absorbed = slack.get(sensitivity.conjunction_id, 0.0)
        if absorbed > tolerance_km:
            certificate.slack_absorbed[sensitivity.conjunction_id] = absorbed
            continue
        if margin < -tolerance_km:
            certificate.resolve_violations.append((sensitivity.conjunction_id, margin))

    for constraint in problem.latent:
        margin = constraint.residual_km(x)
        certificate.latent_margins[constraint.label] = margin
        absorbed = slack.get(constraint.label, 0.0)
        if absorbed > tolerance_km:
            certificate.slack_absorbed[constraint.label] = absorbed
            continue
        if margin < -tolerance_km:
            certificate.latent_violations.append((constraint.label, margin))

    cap = float(problem.per_burn_cap_km_s)
    for sat_id in grid.satellite_ids:
        total = 0.0
        for slot in range(len(grid.slot_epochs(sat_id))):
            component = np.zeros(3)
            for axis in (range(3) if grid.axes == 3 else (1,)):
                value = float(x[grid.index(sat_id, slot, axis)])
                component[axis] = value
                if abs(value) > cap + _TOL_KM_S:
                    certificate.cap_violations.append(
                        (f"cap:{sat_id}:{slot}:{'RTN'[axis]}", abs(value) - cap)
                    )
            total += float(np.sum(np.abs(component)))
        budget = problem.budget(sat_id)
        if total > budget + _TOL_KM_S:
            certificate.budget_violations.append((f"budget:{sat_id}", total - budget))

    if problem.enforce_station_keeping:
        box = (
            float(problem.station_keeping_box_km[0]),
            float(problem.station_keeping_box_km[1]),
        )
        for sat_id in grid.satellite_ids:
            operator, axis_labels = station_keeping_rows(
                grid, sat_id, evaluation_epoch=problem.station_keeping_epoch.get(sat_id)
            )
            displacement = operator @ x
            for position, axis_label in enumerate(axis_labels):
                excess = abs(float(displacement[position])) - box[position]
                if excess > tolerance_km:
                    certificate.station_keeping_violations.append(
                        (f"sk:{sat_id}:{axis_label}", excess)
                    )

    return certificate


def binding_rows(solution: LpSolution, *, tolerance: float = 1e-9) -> dict[str, float]:
    """Rows with a strictly positive dual -- the active set at the optimum.

    This is the regression and classification target the learned predictor in
    :mod:`aegis.ml` is trained against, and it is the set
    :func:`aegis.fleetopt.solver.lazy_solve` is trying to guess.
    """
    return {
        label: value
        for label, value in solution.duals.items()
        if value > tolerance
    }


def exact_penalty_threshold(solution: LpSolution) -> float:
    """``lambda* = max`` over safety-row duals; see Proposition 6.

    Returns 0.0 when no safety row binds (any positive penalty is then exact)
    and ``nan`` when the backend supplied no duals, which is the honest answer
    for a mixed-integer solve.
    """
    if not solution.has_duals:
        return float("nan")
    safety = [
        value
        for label, value in solution.duals.items()
        if label.startswith("resolve:") or label.startswith("latent:")
    ]
    if not safety:
        return 0.0
    return float(max(0.0, max(safety)))


@dataclass
class Farkas:
    """A minimal explanation of why no feasible plan exists."""

    infeasible: bool
    resolve_rows: list[str] = field(default_factory=list)
    latent_rows: list[str] = field(default_factory=list)
    explanation: str = ""
    multipliers: dict[str, float] = field(default_factory=dict)
    rows_tested: int = 0
    minimal: bool = False

    @property
    def size(self) -> int:
        return len(self.resolve_rows) + len(self.latent_rows)

    def summary(self) -> dict:
        return {
            "infeasible": self.infeasible,
            "size": self.size,
            "resolve_rows": list(self.resolve_rows),
            "latent_rows": list(self.latent_rows[:20]),
            "explanation": self.explanation,
            "minimal": self.minimal,
            "rows_tested": self.rows_tested,
        }


def _hard_feasible(
    problem: FleetProblem,
    resolve: list[MissSensitivity],
    latent: list[LatentConstraint],
    directions: dict[str, np.ndarray],
) -> bool:
    """Is the subsystem feasible with slack forced to zero?

    Slack columns are pinned to zero by their upper bound rather than removed,
    which keeps the column layout identical across every deletion-filter probe
    and so keeps the probes comparable.
    """
    candidate = problem.with_latent(latent)
    candidate.resolve = list(resolve)
    data = assemble(candidate, directions)
    upper = data.upper.copy()
    for index, label in enumerate(data.col_labels):
        if label.startswith("slack:"):
            upper[index] = 0.0
    probe = type(data)(
        c=np.zeros_like(data.c),
        a_ub=data.a_ub,
        b_ub=data.b_ub,
        lower=data.lower,
        upper=upper,
        integrality=np.zeros_like(data.integrality),
        row_labels=data.row_labels,
        col_labels=data.col_labels,
        n_lifted=data.n_lifted,
        n_resolve=data.n_resolve,
        n_latent=data.n_latent,
        cost_model=data.cost_model,
    )
    return solve_lp(probe).status == "optimal"


def irreconcilable_subset(
    problem: FleetProblem,
    directions: dict[str, np.ndarray] | None = None,
    *,
    max_rows: int = 40,
) -> Farkas:
    """Minimal infeasible subsystem of the hard-constrained problem.

    Deletion filtering: start from the full safety row set, try removing each
    row in turn, and keep the removal whenever the remainder stays infeasible.
    What survives is minimal -- removing any single remaining row makes the
    system feasible -- which is what makes the explanation trustworthy rather
    than merely suggestive.

    Cost is one LP per candidate row, so the search is capped at ``max_rows``
    rows; beyond that the rows are pre-ranked by nominal shortfall and only
    the worst are filtered, and ``minimal`` is reported False.
    """
    directions = dict(directions or problem.nominal_directions())

    if _hard_feasible(problem, problem.resolve, problem.latent, directions):
        return Farkas(
            infeasible=False,
            explanation="a zero-slack plan exists; nothing is irreconcilable",
        )

    resolve = sorted(problem.resolve, key=lambda s: -s.shortfall_km)
    latent = sorted(problem.latent, key=lambda c: c.slack_km)
    truncated = False
    if len(resolve) + len(latent) > max_rows:
        truncated = True
        keep_resolve = min(len(resolve), max(1, max_rows // 2))
        resolve = resolve[:keep_resolve]
        latent = latent[: max(0, max_rows - keep_resolve)]

    if not _hard_feasible(problem, resolve, latent, directions):
        working_resolve = list(resolve)
        working_latent = list(latent)
    else:
        working_resolve = list(problem.resolve)
        working_latent = list(problem.latent)
        truncated = False

    tested = 0
    index = 0
    while index < len(working_resolve):
        trial = working_resolve[:index] + working_resolve[index + 1 :]
        tested += 1
        if not _hard_feasible(problem, trial, working_latent, directions):
            working_resolve = trial
        else:
            index += 1

    index = 0
    while index < len(working_latent):
        trial = working_latent[:index] + working_latent[index + 1 :]
        tested += 1
        if not _hard_feasible(problem, working_resolve, trial, directions):
            working_latent = trial
        else:
            index += 1

    resolve_names = [s.conjunction_id for s in working_resolve]
    latent_names = [c.label for c in working_latent]
    if latent_names and resolve_names:
        explanation = (
            f"resolving {', '.join(resolve_names)} together necessarily brings "
            f"{len(latent_names)} latent pair(s) inside the exclusion radius "
            f"(first: {latent_names[0]})"
        )
    elif resolve_names:
        explanation = (
            f"{', '.join(resolve_names)} cannot be resolved within the delta-v, "
            "per-burn, and station-keeping limits, even with no induced-conjunction "
            "constraints active"
        )
    elif latent_names:
        explanation = (
            "the nominal geometry already violates the exclusion radius for "
            f"{len(latent_names)} latent pair(s); no maneuver is responsible"
        )
    else:
        raise CertificationError(
            "the full system is infeasible but the deletion filter removed every row; "
            "this means a structural row (budget, cap, or station keeping) is the "
            "cause and the safety rows are innocent"
        )

    return Farkas(
        infeasible=True,
        resolve_rows=resolve_names,
        latent_rows=latent_names,
        explanation=explanation,
        rows_tested=tested,
        minimal=not truncated,
    )
