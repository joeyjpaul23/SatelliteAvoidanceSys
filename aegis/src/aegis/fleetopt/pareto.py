"""The safety-premium frontier, and why it is exactly piecewise linear.

The central number of this project is the **safety premium**

.. code-block:: text

    pi = (dv_zero-induced - dv_fuel-only) / dv_fuel-only

-- the extra delta-v a fleet pays to create no new conjunctions. The
literature has nothing directly comparable. Armellin (2021) swept a single
geometry's probability threshold and saw 6.4 to 431.1 mm/s; Pavanello et al.
(2024) report 1.1 % to 3.8 % extra fuel for 17 % to 42 % lower total
probability across 5, 7 and 10 encounters, but never counted induced
conjunctions; Klinkrad et al. (2005) priced fleet safety in maneuver
*frequency* rather than propellant. This module measures the induced-risk
premium directly.

Structure of the frontier
-------------------------
Add the row :math:`\\sum_p \\sigma_p \\le \\varepsilon` to the fleet program,
bounding total induced-separation shortfall, and let
:math:`V^{\\star}(\\varepsilon)` be the optimal cost. Then:

    **Proposition 7.** :math:`V^{\\star}` is convex, piecewise linear and
    non-increasing in :math:`\\varepsilon`, with finitely many breakpoints, and
    its left derivative at :math:`\\varepsilon` equals minus the optimal
    multiplier of the ``induced-budget`` row.

*Proof.* This is the standard right-hand-side sensitivity result for linear
programs: the value function of an LP is the support function of its dual
feasible set,
:math:`V^{\\star}(\\varepsilon) = \\max\\{b^{\\top}y - \\varepsilon \\nu : (y,\\nu) \\in \\mathcal{D}\\}`,
a pointwise maximum of affine functions of :math:`\\varepsilon` and therefore
convex and piecewise linear; the maximising :math:`\\nu` is the multiplier, and
it is non-negative because relaxing a :math:`\\le` row cannot increase cost.
The dual feasible set is a polyhedron with finitely many vertices, so there
are finitely many slopes. :math:`\\square`

The theorem is textbook (parametric linear programming); the consequence for
this domain is not. **The delta-v-versus-induced-risk Pareto frontier is
exactly representable by finitely many breakpoints and can be computed
exactly, not sampled.** Published CAM tradeoff curves are sampled point
clouds; here the sampled points come with the slope at each one, the convexity
is a checkable invariant rather than an eyeballed impression, and a violation
of convexity is a bug report rather than a physical finding.

One honest caveat, stated where it belongs: the frontier is exactly piecewise
linear *for a fixed linearization*. The sequential refinement changes the rows
between iterations, so the converged frontier is piecewise linear up to the
refinement, and :attr:`Frontier.convexity_residual` reports how far from
convex the measured curve actually is. In practice it is at the solver
tolerance; when it is not, that is information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problem import FleetProblem
from .solver import ScpResult, sequential_solve

__all__ = [
    "RATIO_DENOMINATOR_FLOOR_KM_S",
    "PenetrationIndex",
    "row_reach_km",
    "penetration_index",
    "premium_frontier",
    "fixed_linearization_premium",
    "FrontierPoint",
    "Frontier",
    "induced_frontier",
    "PremiumRecord",
    "safety_premium",
    "premium_statistics",
]

_REL_TOL = 1e-7

#: Smallest fuel-only delta-v for which the *ratio* premium is reported, km/s
#: (100 mm/s). Below it the absolute premium is still reported; the ratio is
#: not, because a near-zero denominator manufactures enormous percentages out
#: of operationally trivial differences.
RATIO_DENOMINATOR_FLOOR_KM_S = 1.0e-4


@dataclass
class FrontierPoint:
    """One solved point of ``V*(epsilon)``."""

    epsilon: float
    cost: float
    delta_v_km_s: float
    multiplier: float
    induced_shortfall_km: float
    resolve_shortfall_km: float
    status: str
    iterations: int

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"


@dataclass
class Frontier:
    """The measured ``V*(epsilon)`` curve and its structural checks."""

    points: list[FrontierPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def epsilons(self) -> np.ndarray:
        return np.array(
            [p.epsilon for p in self.feasible_points], dtype=float
        )

    @property
    def costs(self) -> np.ndarray:
        return np.array(
            [p.cost for p in self.feasible_points], dtype=float
        )

    @property
    def delta_v(self) -> np.ndarray:
        return np.array(
            [p.delta_v_km_s for p in self.feasible_points], dtype=float
        )

    @property
    def multipliers(self) -> np.ndarray:
        return np.array(
            [p.multiplier for p in self.feasible_points], dtype=float
        )

    @property
    def feasible_points(self) -> list[FrontierPoint]:
        """Only the points the solver actually solved.

        An infeasible epsilon has no value, and letting its placeholder cost
        into the convexity or monotonicity check turns a structural assertion
        into a NaN comparison. Infeasible points are reported separately --
        the epsilon below which no zero-induced plan exists is itself a
        result.
        """
        return [point for point in self.points if point.feasible]

    @property
    def infeasible_below(self) -> float | None:
        """Largest epsilon at which the program was infeasible, if any."""
        bad = [point.epsilon for point in self.points if not point.feasible]
        return max(bad) if bad else None

    @property
    def non_increasing(self) -> bool:
        costs = self.costs
        if costs.size < 2:
            return True
        scale = max(1.0, float(np.max(np.abs(costs))))
        return bool(np.all(np.diff(costs) <= _REL_TOL * scale))

    @property
    def convexity_residual(self) -> float:
        """Worst violation of convexity, relative to the cost scale.

        Zero (to solver tolerance) confirms Proposition 7 on the measured
        curve. A positive value means some midpoint sits above the chord,
        which for a fixed linearization cannot happen and therefore points at
        the refinement or at a solver tolerance.
        """
        epsilons = self.epsilons
        costs = self.costs
        if costs.size < 3:
            return 0.0
        scale = max(1.0, float(np.max(np.abs(costs))))
        worst = 0.0
        for index in range(1, costs.size - 1):
            left_gap = epsilons[index] - epsilons[index - 1]
            right_gap = epsilons[index + 1] - epsilons[index]
            if left_gap <= 0.0 or right_gap <= 0.0:
                continue
            weight = right_gap / (left_gap + right_gap)
            chord = weight * costs[index - 1] + (1.0 - weight) * costs[index + 1]
            worst = max(worst, float(costs[index] - chord) / scale)
        return max(0.0, worst)

    @property
    def is_convex(self) -> bool:
        return self.convexity_residual <= _REL_TOL

    def slopes(self) -> np.ndarray:
        """Finite-difference slopes between consecutive points."""
        epsilons = self.epsilons
        costs = self.costs
        if costs.size < 2:
            return np.zeros(0)
        return np.diff(costs) / np.diff(epsilons)

    def slope_agreement(self) -> float:
        """Worst relative disagreement between slopes and reported multipliers.

        By Proposition 7 the left derivative at ``epsilon`` is minus the
        multiplier there, so the finite-difference slope over an interval
        should match the multiplier at its right endpoint wherever the curve is
        affine on that interval. Comparing the two is a cross-check of the
        dual extraction in :mod:`aegis.fleetopt.solver`: agreement means the
        duals are being read with the right sign and scale.
        """
        slopes = self.slopes()
        multipliers = self.multipliers
        if slopes.size == 0:
            return 0.0
        worst = 0.0
        for index, measured in enumerate(slopes):
            left = multipliers[index]
            right = multipliers[index + 1]
            # Only compare across an interval the value function is affine on.
            # A breakpoint inside the interval makes the finite difference an
            # average of two slopes, which matches neither endpoint's dual --
            # that is correct behaviour, not disagreement, and scoring it as a
            # failure reported 1.0 on a curve that was in fact exact.
            if abs(left - right) > 1e-12 * max(1.0, abs(left), abs(right)):
                continue
            predicted = -right
            scale = max(abs(predicted), abs(measured), 1e-30)
            if scale <= 1e-30:
                continue
            worst = max(worst, abs(measured - predicted) / scale)
        return worst

    def breakpoints(self, *, tolerance: float = 1e-6) -> list[int]:
        """Indices where the slope changes -- the frontier's vertices."""
        slopes = self.slopes()
        if slopes.size < 2:
            return []
        scale = max(1.0, float(np.max(np.abs(slopes))))
        return [
            index + 1
            for index in range(slopes.size - 1)
            if abs(slopes[index + 1] - slopes[index]) > tolerance * scale
        ]

    def summary(self) -> dict:
        return {
            "points": len(self.points),
            "feasible_points": len(self.feasible_points),
            "infeasible_below": self.infeasible_below,
            "non_increasing": self.non_increasing,
            "is_convex": self.is_convex,
            "convexity_residual": float(self.convexity_residual),
            "slope_agreement": float(self.slope_agreement()),
            "breakpoints": self.breakpoints(),
            "epsilons": [float(e) for e in self.epsilons],
            "delta_v_mm_s": [float(v) * 1e6 for v in self.delta_v],
            "multipliers": [float(m) for m in self.multipliers],
            "notes": list(self.notes),
        }


def _delta_v_km_s(result: ScpResult) -> float:
    """Total charged delta-v, read through the cost model's magnitude columns."""
    data = result.data
    total = 0.0
    for (sat_id, slot) in data.cost_model.magnitude_columns:
        total += data.cost_model.charged_magnitude(result.solution.z[: data.n_lifted], sat_id, slot)
    return float(total)


def induced_frontier(
    problem: FleetProblem,
    *,
    epsilons: list[float] | None = None,
    points: int = 9,
    max_iterations: int = 6,
    initial_directions: dict | None = None,
) -> Frontier:
    """Solve ``V*(epsilon)`` across a sweep of induced-risk budgets.

    The default sweep spans zero to the total nominal induced shortfall --
    that is, from "create nothing" to "ignore the induced constraints
    entirely" -- on a scale that is linear in the shortfall, because that is
    the axis the multiplier is the derivative with respect to.
    """
    if not problem.latent:
        return Frontier(
            notes=[
                "no latent rows in this problem, so there is no induced-risk budget "
                "to sweep and the frontier is a single point"
            ]
        )

    if epsilons is None:
        ceiling = float(sum(max(0.0, -c.slack_km) for c in problem.latent))
        if ceiling <= 0.0:
            ceiling = float(
                sum(max(0.0, c.floor_km - c.separation_km + 1.0) for c in problem.latent)
            )
        if ceiling <= 0.0:
            ceiling = 1.0
        epsilons = list(np.linspace(0.0, ceiling, max(2, int(points))))

    frontier = Frontier()
    for epsilon in epsilons:
        candidate = problem.with_latent(list(problem.latent))
        candidate.induced_budget = float(epsilon)
        candidate.slack_caps = dict(problem.slack_caps)
        candidate.total_slack_budget = problem.total_slack_budget
        candidate.resolve_slack_penalty = problem.resolve_slack_penalty
        candidate.latent_slack_penalty = problem.latent_slack_penalty
        result = sequential_solve(
            candidate,
            max_iterations=max_iterations,
            initial_directions=initial_directions,
        )
        multiplier = float(result.solution.duals.get("induced-budget", 0.0))
        latent_slack = result.solution.slack(result.data, "latent")
        resolve_slack = result.solution.slack(result.data, "resolve")
        frontier.points.append(
            FrontierPoint(
                epsilon=float(epsilon),
                cost=float(result.solution.objective),
                delta_v_km_s=_delta_v_km_s(result),
                multiplier=multiplier,
                induced_shortfall_km=float(sum(latent_slack.values())),
                resolve_shortfall_km=float(sum(resolve_slack.values())),
                status=result.solution.status,
                iterations=result.iterations,
            )
        )

    if not frontier.non_increasing:
        frontier.notes.append(
            "the measured frontier is not non-increasing, which contradicts "
            "Proposition 7 for a fixed linearization -- suspect the sequential "
            "refinement changing rows between points, or a solver tolerance"
        )
    if not frontier.is_convex:
        frontier.notes.append(
            f"convexity residual {frontier.convexity_residual:.3e} exceeds {_REL_TOL:.0e}; "
            "the same caveat applies"
        )
    return frontier


def row_reach_km(constraint, grid, budgets: dict[str, float]) -> float:
    """Largest change any admissible plan can make to one latent row.

    ``reach_j = sum_s D_s * ||row_j restricted to satellite s||_inf``: each
    satellite can put its whole budget on whichever of its own columns moves
    this row most. Closed form, no solve.
    """
    row, _ = constraint.row()
    total = 0.0
    for satellite in grid.satellite_ids:
        columns = grid.columns(satellite)
        if columns:
            total += float(budgets.get(satellite, 0.0)) * float(np.max(np.abs(row[columns])))
    return total


@dataclass
class PenetrationIndex:
    """How far the fuel-optimal plan pushes into its nearest induced-conjunction
    constraint, normalised by what the budget could buy back.

    .. math::

        \\varrho = \\max_j \\frac{\\big(f_j - u_j^{\\top}(d_j + B_j x_{\\text{fuel}})\\big)_+}
                              {\\mathrm{reach}_j}

    The numerator is the violation the cheapest plan commits on latent row
    :math:`j`; the denominator is the most any admissible plan could move that
    row (:func:`row_reach_km`). The ratio is therefore dimensionless and reads
    as *"what fraction of the fleet's reach is already spoken for by the
    cheapest plan's worst induced conjunction."*

    Two properties make it worth computing, both measured over 70
    ``induced-cascade`` scenarios:

    * **It is exactly zero when the premium is zero, on 70 of 70 scenarios.**
      If the fuel-optimal plan violates no latent row, the induced constraints
      are not binding and coordination is free. That is an equivalence, not a
      correlation.
    * **Spearman 0.989** with the measured safety premium, univariate
      :math:`R^2` 0.866.

    For contrast, on the same data the *coupling number* --
    :math:`\\mathrm{cond}(BB^{\\top})`, this project's original proposal -- is
    constant and predicts nothing, and in-plane neighbour spacing reaches only
    Spearman :math:`-0.855`. A search over more than a dozen candidate scalars
    and a random forest on sixty-six pre-screening features found nothing
    better.

    The fitted slope is **not** reported as a calibrated magnitude: it came out
    at 2.72 on one dataset and 5.73 on another with a different budget, because
    it absorbs the scenario's own delta-v scale. Use the index to *rank* and to
    decide *whether* a premium exists, not to predict its size in percent.

    It needs the fuel-only solution, which the planner already computes, so its
    marginal cost is one pass over the latent rows.
    """

    value: float
    worst_row: str = ""
    violated_rows: int = 0
    rows_considered: int = 0

    @property
    def predicts_zero_premium(self) -> bool:
        """Whether coordination is free: no latent row binds the cheapest plan."""
        return self.value <= 1e-12

    def as_dict(self) -> dict:
        return {
            "penetration_index": round(self.value, 9),
            "penetration_worst_row": self.worst_row,
            "penetration_violated_rows": self.violated_rows,
            "penetration_rows_considered": self.rows_considered,
            "predicts_zero_premium": self.predicts_zero_premium,
        }


def penetration_index(
    latent: list,
    grid,
    budgets: dict[str, float],
    x_fuel: np.ndarray,
) -> PenetrationIndex:
    """Compute :class:`PenetrationIndex` from the fuel-only solution."""
    best = 0.0
    worst_row = ""
    violated = 0
    considered = 0
    for constraint in latent:
        reach = row_reach_km(constraint, grid, budgets)
        if reach <= 0.0:
            continue
        considered += 1
        violation = max(0.0, -constraint.linear_residual_km(x_fuel))
        if violation > 0.0:
            violated += 1
        ratio = violation / reach
        if ratio > best:
            best = ratio
            worst_row = constraint.label
    return PenetrationIndex(
        value=float(best),
        worst_row=worst_row,
        violated_rows=violated,
        rows_considered=considered,
    )


def premium_frontier(
    problem_fuel: FleetProblem,
    problem_safe: FleetProblem,
    directions: dict,
    *,
    points: int = 9,
    tolerance_km: float = 1e-7,
) -> Frontier:
    """Trace ``V*(epsilon)`` the way Proposition 7 requires it to be traced.

    Three things have to be true at once or the curve is meaningless, and each
    was learned by producing a meaningless curve first:

    1. **One linearization for every point.** Otherwise successive points are
       values of different programs and the proposition says nothing about the
       sequence.
    2. **Delta-v must actually be the objective.** Under the default slack
       penalty of 1e4 per km against a delta-v cost of order 1e-3, the
       optimizer spends its whole budget shaving shortfall and the reported
       delta-v is not a minimum of anything; the curve comes back flat.
       Resolve shortfall is therefore *pinned* at what the fuel-only solve
       achieved and the penalty set to zero, so the only thing left to
       minimise is fuel.
    3. **Infeasible points must be excluded, not averaged in.** The epsilon
       below which no plan exists is a result in its own right, and letting a
       placeholder cost into the convexity check turns a structural assertion
       into a NaN comparison.

    The epsilon range is chosen by first solving with the induced budget
    effectively removed, so the sweep spans exactly the achievable range of
    induced shortfall.
    """
    reference = sequential_solve(problem_fuel, max_iterations=1, initial_directions=directions)
    if not reference.solution.ok:
        return Frontier(
            notes=[f"reference fuel-only solve returned {reference.solution.status!r}"]
        )

    caps = {
        f"resolve:{key}": value + tolerance_km
        for key, value in reference.solution.slack(reference.data, "resolve").items()
    }
    staged = problem_safe.with_latent(list(problem_safe.latent))
    staged.slack_caps = caps
    staged.resolve_slack_penalty = 0.0
    staged.latent_slack_penalty = 0.0

    unbounded = staged.with_latent(list(staged.latent))
    unbounded.slack_caps = dict(caps)
    unbounded.resolve_slack_penalty = 0.0
    unbounded.latent_slack_penalty = 0.0
    unbounded.induced_budget = None
    probe = sequential_solve(unbounded, max_iterations=1, initial_directions=directions)
    ceiling = (
        float(sum(probe.solution.slack(probe.data, "latent").values()))
        if probe.solution.ok
        else 0.0
    )

    epsilons = list(np.linspace(0.0, max(ceiling, 1e-6) * 1.1, max(3, int(points))))
    frontier = induced_frontier(
        staged, epsilons=epsilons, max_iterations=1, initial_directions=directions
    )
    frontier.notes.append(
        f"traced at a fixed linearization with resolve shortfall pinned at "
        f"{sum(reference.solution.slack(reference.data, 'resolve').values()):.6g} km; "
        f"induced shortfall ranges over [0, {ceiling:.6g}] km"
    )
    return frontier


def fixed_linearization_premium(
    problem_fuel: FleetProblem,
    problem_safe: FleetProblem,
    directions: dict,
    *,
    tolerance_km: float = 1e-7,
) -> tuple[float | None, float, float, str]:
    """The premium computed where it is provably a lower bound.

    Getting this right took two wrong attempts, and both are worth recording
    because the wrong versions produced *negative* premiums that looked like
    findings.

    **Attempt one** compared the two planners as they run. Each does its own
    sequential refinement, the extra rows change the first iterate, and the
    two walk different direction paths -- so the comparison was between
    solutions of two differently-linearised problems. Measured on the
    induced-cascade family at 3 km neighbour spacing: -11.1 % and -0.06 %.

    **Attempt two** fixed the linearization for both. Still negative, because
    nesting still did not hold: ``fleet-safe`` carries *mandatory slack* on its
    latent rows, so its objective is ``dv + lambda*(resolve slack + latent
    slack)`` against ``fuel-only``'s ``dv + lambda*resolve slack``. Those are
    different objectives, and the safe optimum can buy a lower ``dv`` by
    accepting resolve slack it trades against latent slack. Adding constraints
    only forces a worse optimum when the *objective* is unchanged.

    **What actually works**, and what this does:

    1. Solve ``fuel-only`` at the given directions. Record its per-row resolve
       shortfall and its delta-v.
    2. Solve ``fleet-safe`` at the *same* directions, with every latent slack
       pinned to zero (the induced-conjunction constraints made hard) and
       every resolve slack capped at what ``fuel-only`` achieved.

    Now the second feasible set is genuinely a subset of the first and the
    objective is genuinely the same, so ``dv_safe >= dv_fuel`` by construction.
    If step 2 is infeasible, that is the honest answer -- no zero-induced plan
    exists at this linearization that is at least as safe as the fuel-only
    plan -- and the premium is reported as undefined rather than as a number.

    Returns ``(premium, dv_fuel_km_s, dv_safe_km_s, note)``.
    """
    fuel = sequential_solve(problem_fuel, max_iterations=1, initial_directions=directions)
    if not fuel.solution.ok:
        return None, 0.0, 0.0, f"fuel-only solve returned {fuel.solution.status!r}"

    dv_fuel = _delta_v_km_s(fuel)
    caps = {
        f"resolve:{key}": value + tolerance_km
        for key, value in fuel.solution.slack(fuel.data, "resolve").items()
    }
    caps.update({constraint.label: 0.0 for constraint in problem_safe.latent})

    restricted = problem_safe.with_latent(list(problem_safe.latent))
    restricted.slack_caps = caps
    safe = sequential_solve(restricted, max_iterations=1, initial_directions=directions)
    if not safe.solution.ok:
        return (
            None,
            dv_fuel,
            0.0,
            "no zero-induced plan exists at this linearization that is at least as "
            f"safe as the fuel-only plan (solver returned {safe.solution.status!r})",
        )

    dv_safe = _delta_v_km_s(safe)
    if dv_fuel < RATIO_DENOMINATOR_FLOOR_KM_S:
        # The absolute difference is still returned and still meaningful; only
        # the ratio is withheld, because its denominator is not.
        return (
            None,
            dv_fuel,
            dv_safe,
            "fuel-only delta-v is below the ratio floor; use the absolute premium",
        )
    premium = (dv_safe - dv_fuel) / dv_fuel
    note = ""
    if premium < -1e-9:
        note = (
            f"premium {premium:.6g} is negative under a construction that forbids it; "
            "suspect the assembly or the solver, not the physics"
        )
    return premium, dv_fuel, dv_safe, note


@dataclass
class PremiumRecord:
    """The safety premium for one scenario, with everything needed to audit it."""

    scenario_id: str
    delta_v_fuel_km_s: float
    delta_v_safe_km_s: float
    premium: float | None
    induced_fuel_only: int
    induced_fleet_safe: int
    resolved_fuel_only: int
    resolved_fleet_safe: int
    total_conjunctions: int
    fuel_only_feasible: bool
    fleet_safe_feasible: bool
    infeasible: bool = False
    reason: str = ""
    coupling_number: float = 0.0
    conflict_dimension: int = 0
    family: str = ""
    fixed_premium: float | None = None
    fixed_dv_fuel_km_s: float = 0.0
    fixed_dv_safe_km_s: float = 0.0
    fixed_note: str = ""

    @property
    def absolute_premium_mm_s(self) -> float:
        """The premium in mm/s. Always defined, never inflated by a small base."""
        return (self.delta_v_safe_km_s - self.delta_v_fuel_km_s) * 1e6

    @property
    def fixed_absolute_premium_mm_s(self) -> float:
        return (self.fixed_dv_safe_km_s - self.fixed_dv_fuel_km_s) * 1e6

    @property
    def delta_v_fuel_mm_s(self) -> float:
        return self.delta_v_fuel_km_s * 1e6

    @property
    def delta_v_safe_mm_s(self) -> float:
        return self.delta_v_safe_km_s * 1e6

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "delta_v_fuel_mm_s": round(self.delta_v_fuel_mm_s, 6),
            "delta_v_safe_mm_s": round(self.delta_v_safe_mm_s, 6),
            "premium": None if self.premium is None else round(self.premium, 8),
            "premium_percent": None if self.premium is None else round(100.0 * self.premium, 5),
            "absolute_premium_mm_s": round(self.absolute_premium_mm_s, 6),
            "fixed_absolute_premium_mm_s": round(self.fixed_absolute_premium_mm_s, 6),
            "induced_fuel_only": self.induced_fuel_only,
            "induced_fleet_safe": self.induced_fleet_safe,
            "resolved_fuel_only": self.resolved_fuel_only,
            "resolved_fleet_safe": self.resolved_fleet_safe,
            "total_conjunctions": self.total_conjunctions,
            "fuel_only_feasible": self.fuel_only_feasible,
            "fleet_safe_feasible": self.fleet_safe_feasible,
            "infeasible": self.infeasible,
            "reason": self.reason,
            "coupling_number": round(self.coupling_number, 6),
            "conflict_dimension": self.conflict_dimension,
            "fixed_premium": None if self.fixed_premium is None else round(self.fixed_premium, 8),
            "fixed_premium_percent": (
                None if self.fixed_premium is None else round(100.0 * self.fixed_premium, 5)
            ),
            "fixed_dv_fuel_mm_s": round(self.fixed_dv_fuel_km_s * 1e6, 6),
            "fixed_dv_safe_mm_s": round(self.fixed_dv_safe_km_s * 1e6, 6),
            "fixed_note": self.fixed_note,
        }


def safety_premium(
    scenario_id: str,
    *,
    delta_v_fuel_km_s: float,
    delta_v_safe_km_s: float,
    induced_fuel_only: int,
    induced_fleet_safe: int,
    resolved_fuel_only: int,
    resolved_fleet_safe: int,
    total_conjunctions: int,
    fuel_only_feasible: bool,
    fleet_safe_feasible: bool,
    coupling_number: float = 0.0,
    conflict_dimension: int = 0,
    family: str = "",
    fixed: tuple[float | None, float, float, str] | None = None,
) -> PremiumRecord:
    """The premium, or an explicit reason why it is undefined.

    Undefined is not zero and not infinity. A scenario whose fuel-only plan
    needs no maneuver at all has no denominator, and reporting a huge or zero
    premium there would corrupt the distribution; it is recorded as ``None``
    with a reason and counted separately in
    :func:`premium_statistics`.
    """
    premium: float | None = None
    reason = ""
    infeasible = False

    # "Infeasible" means *no zero-induced plan exists*, which is what the
    # fixed-linearization construction reports. It does not mean "some
    # conjunction was left unresolved": both planners face the same
    # unresolvable rows, so that is a property of the scenario, not of the
    # comparison. Driving the flag off `fleet_safe_feasible` marked every
    # scenario infeasible in a run where one repeat encounter was simply out
    # of reach, and hid a premium that was in fact well defined.
    if fixed is not None and fixed[0] is None and "no zero-induced plan" in str(fixed[3]):
        infeasible = True
        reason = "no zero-induced plan exists at this linearization"
    elif delta_v_fuel_km_s < RATIO_DENOMINATOR_FLOOR_KM_S:
        # The ratio is the quantity the handoff asked for, but it is only
        # meaningful when the denominator is. A fuel-only plan costing a few
        # micrometres per second turns any absolute difference into a
        # four-figure percentage: measured on the main sweep, three scenarios
        # with denominators of 7-34 mm/s produced "premiums" of 1565 %, 2764 %
        # and 3717 %. Those are real delta-v differences and they are reported
        # as such in `absolute_premium_mm_s`; as ratios they are noise.
        reason = (
            "fuel_only_requires_no_maneuver"
            if delta_v_fuel_km_s <= 0.0
            else "fuel_only_delta_v_below_ratio_floor"
        )
    elif not fuel_only_feasible:
        reason = "fuel_only_plan_left_conjunctions_unresolved"
        premium = (delta_v_safe_km_s - delta_v_fuel_km_s) / delta_v_fuel_km_s
    else:
        premium = (delta_v_safe_km_s - delta_v_fuel_km_s) / delta_v_fuel_km_s

    return PremiumRecord(
        scenario_id=scenario_id,
        delta_v_fuel_km_s=float(delta_v_fuel_km_s),
        delta_v_safe_km_s=float(delta_v_safe_km_s),
        premium=premium,
        induced_fuel_only=int(induced_fuel_only),
        induced_fleet_safe=int(induced_fleet_safe),
        resolved_fuel_only=int(resolved_fuel_only),
        resolved_fleet_safe=int(resolved_fleet_safe),
        total_conjunctions=int(total_conjunctions),
        fuel_only_feasible=bool(fuel_only_feasible),
        fleet_safe_feasible=bool(fleet_safe_feasible),
        infeasible=infeasible,
        reason=reason,
        coupling_number=float(coupling_number),
        conflict_dimension=int(conflict_dimension),
        family=family,
        fixed_premium=fixed[0] if fixed else None,
        fixed_dv_fuel_km_s=float(fixed[1]) if fixed else 0.0,
        fixed_dv_safe_km_s=float(fixed[2]) if fixed else 0.0,
        fixed_note=str(fixed[3]) if fixed else "",
    )


def premium_statistics(records: list[PremiumRecord]) -> dict:
    """Distribution of the premium, reported as a distribution.

    The handoff document was explicit that a mean alone is misleading here, and
    it is right: the premium is bounded below by zero and unbounded above, so
    the tail is the story. Median, p90, p99 and worst case are reported
    alongside the fraction of scenarios where no safe plan exists at all --
    which is the number a fuel-only comparison silently hides.
    """
    if not records:
        return {
            "count": 0,
            "defined": 0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "worst": 0.0,
            "mean": 0.0,
            "infeasible_fraction": 0.0,
            "undefined_fraction": 0.0,
            "zero_premium_fraction": 0.0,
        }

    fixed = [
        r.fixed_premium
        for r in records
        if r.fixed_premium is not None and not r.infeasible
    ]
    defined = [r.premium for r in records if r.premium is not None and not r.infeasible]
    values = np.asarray(defined, dtype=float) if defined else np.zeros(0)
    infeasible = sum(1 for r in records if r.infeasible)
    undefined = sum(1 for r in records if r.premium is None and not r.infeasible)

    def percentile(fraction: float) -> float:
        return float(np.percentile(values, fraction)) if values.size else 0.0

    return {
        "count": len(records),
        "defined": int(values.size),
        "median": percentile(50.0),
        "p90": percentile(90.0),
        "p99": percentile(99.0),
        "worst": float(np.max(values)) if values.size else 0.0,
        "mean": float(np.mean(values)) if values.size else 0.0,
        "infeasible_fraction": infeasible / len(records),
        "undefined_fraction": undefined / len(records),
        "zero_premium_fraction": (
            float(np.mean(values <= 1e-12)) if values.size else 0.0
        ),
        "fixed_defined": len(fixed),
        "fixed_median": float(np.median(fixed)) if fixed else 0.0,
        "fixed_p90": float(np.percentile(fixed, 90)) if fixed else 0.0,
        "fixed_p99": float(np.percentile(fixed, 99)) if fixed else 0.0,
        "fixed_worst": float(np.max(fixed)) if fixed else 0.0,
        "fixed_negative_count": int(sum(1 for value in fixed if value < -1e-9)),
        "absolute_median_mm_s": (
            float(np.median([r.absolute_premium_mm_s for r in records])) if records else 0.0
        ),
        "absolute_p90_mm_s": (
            float(np.percentile([r.absolute_premium_mm_s for r in records], 90))
            if records
            else 0.0
        ),
        "fixed_absolute_median_mm_s": (
            float(np.median([r.fixed_absolute_premium_mm_s for r in records])) if records else 0.0
        ),
        "fixed_absolute_p90_mm_s": (
            float(np.percentile([r.fixed_absolute_premium_mm_s for r in records], 90))
            if records
            else 0.0
        ),
    }
