"""The planner family, and the shared machinery they differ against.

Every planner here answers the same question -- "what burns should this fleet
execute?" -- and they are written to be *comparable*, because the research
result is the comparison, not any single plan. They share one context object
(:class:`PlanContext`) that does the screening, assessment, burn-grid layout,
sensitivity assembly, reachability bound and latent-row enumeration once, so
that when two planners differ the difference is the formulation and nothing
else.

The family, and what each one is for:

``no-maneuver``
    Zero burns. The unmitigated baseline every risk reduction is measured
    against.
``legacy-lp``
    Delegates to :func:`aegis.maneuver.plan_maneuvers` untouched. Present as
    an ablation: it uses the scalar along-track sensitivity model, which
    :doc:`docs/model-fidelity-validation` shows returns the wrong *sign* for
    cross-plane intra-fleet conjunctions. Keeping it in the comparison is how
    that claim stays falsifiable.
``greedy-pairwise``
    Resolve conjunctions one at a time in TCA order, each as an independent
    single-pair program, summing the burns. This is what an operator does
    today, and what an optimizer must beat to justify itself.
``fuel-only``
    Fleet-wide program with resolve constraints only. Minimises delta-v
    subject to fixing the conjunctions it was handed, and is free to create
    new ones. The denominator of the safety premium.
``fleet-safe``
    ``fuel-only`` plus the certified induced-conjunction rows. The numerator.
``lexicographic``
    Two-stage: minimise total safety shortfall, then minimise delta-v among
    plans achieving it. Safety strictly first, no exchange rate at all.
``milp-ops``
    ``fleet-safe`` plus binaries pricing the number of satellites disturbed
    and the number of burns commanded. The codebase argues in
    :mod:`aegis.core.maneuver` that fuel is not the binding constraint in
    fleet operations -- service disruption and re-screening burden are -- and
    this planner is where that argument becomes an objective term.
``pignn-active-set``
    ``fleet-safe`` restricted to the rows a graph network predicts will bind,
    closed exactly by the lazy-constraint loop. Returns the same objective as
    ``fleet-safe`` by construction.
``pignn-warm-start``
    ``fleet-safe`` with network-supplied initial linearization directions.
    Safe for any prediction, because the restriction is conservative for any
    direction (Proposition 2).
``pignn-direct``
    The network's output used verbatim, with no optimizer. Included precisely
    so its violation rate can be reported next to the certified planners.

A planner never raises on an infeasible scenario. It returns a plan with
unresolved conjunctions ranked by shortfall, plus -- when the hard-constrained
problem is genuinely infeasible -- a Farkas explanation naming which
conjunctions cannot be reconciled. That contract is inherited from
:func:`aegis.maneuver.plan_maneuvers` and is the operationally useful
behaviour.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from ..constants import (
    PC_THRESHOLD_WATCH,
    DEFAULT_BURN_SLOTS,
    DEFAULT_DV_BUDGET_KM_S,
    LP_SLACK_PENALTY,
    MAX_DV_PER_BURN_KM_S,
    MIN_LEAD_TIME_ORBITS,
    PC_TARGET_POST_MANEUVER,
    SCREENING_BOX_STARLINK_KM,
    STATION_KEEPING_BOX_KM,
)
from ..core.conjunction import RiskLevel
from ..core.maneuver import (
    Maneuver,
    ManeuverPlan,
    ResolvedConjunction,
    SatelliteManeuverSet,
)
from ..core.objects import SpaceObject
from ..core.timebase import ensure_utc, shift, utc_now
from ..maneuver.miss import required_miss_distance_km
from ..maneuver.planner import plan_maneuvers
from ..propagation.covariance import default_covariance_model
from ..propagation.propagator import PropagationError, Sgp4Propagator
from ..risk.alfano import collision_probability
from ..risk.batch import AssessedCatalog
from .bplane import MissSensitivity, build_miss_sensitivity
from .certify import Certificate, Farkas, exact_penalty_threshold, irreconcilable_subset, verify_plan
from .dynamics import BurnGrid, build_burn_grid
from .graph import GraphMetrics, build_conjunction_graph, graph_metrics
from .latent import (
    LATENT_DEFAULT_MODE,
    LatentConstraint,
    build_latent_constraint,
    enumerate_latent_constraints,
    pair_id,
    perturbed_minimum_epochs,
)
from .norms import CostModel, DEFAULT_AXIS_WEIGHTS, cone_cost_model, l1_cost_model
from .pareto import PenetrationIndex, penetration_index
from .problem import FleetProblem
from .reachability import ReachabilityModel
from .solver import LazyResult, ScpResult, lazy_solve, sequential_solve

__all__ = [
    "PlanRequest",
    "LearnedHints",
    "FleetPlan",
    "PlanContext",
    "Planner",
    "PLANNERS",
    "get_planner",
    "plan_with",
]

_DV_EMIT_FLOOR_KM_S = 1e-12
_SLACK_RESOLVED_KM = 1e-9

#: Slack the stage-two caps allow above what stage one achieved, km.
#:
#: One millimetre. Not solver noise -- a deliberate, physically negligible
#: allowance, and the size matters. Stage one's optimum is degenerate: many
#: plans achieve the same total shortfall, and the LP picks among them
#: arbitrarily. Pinning each row at exactly its stage-one value therefore
#: freezes an arbitrary choice, and because the shortfall rows are stiff
#: (a 1e4 per km penalty against a delta-v cost of order 1e-3) a difference
#: of 27 micrometres in where the slack sits was measured to move the final
#: delta-v by 266 mm/s. A millimetre of slack is invisible against miss
#: distances of hundreds of metres and removes the amplification.
_SLACK_BUDGET_TOL_KM = 1e-6

#: Above this many safety rows the deletion filter behind the Farkas
#: explanation costs one LP per row and stops being worth it. The plan then
#: reports the unresolved conjunctions without the minimal-subset story.
_FARKAS_ROW_LIMIT = 300

#: Rounds of "polish then re-verify held-back rows" the learned planner runs.
#: Stage two moves the plan, so a row that survived the lazy loop can be
#: violated after the polish; the loop closes that gap.
_POLISH_ROUNDS = 8

#: Cap on new rows created per polish round from perturbed minima.
_MAX_NEW_EPOCH_ROWS = 24

#: Safety/fuel rounds the lexicographic planner iterates to a fixed point.
#: Each round re-measures the attainable shortfall on the row set as it
#: stands, so a round is only spent when a solution revealed a new epoch.
_LEXICOGRAPHIC_ROUNDS = 6

#: Floor on the induced-conjunction exclusion radius, km. The radial half-width
#: of the Starlink screening box, which is the dimension operators actually
#: gate on and the one orbit uncertainty is least bad in.
DEFAULT_INDUCED_EXCLUSION_KM = 2.0

#: Risk level the exclusion radius is derived to keep induced pairs below.
INDUCED_EXCLUSION_LEVEL_PC = PC_THRESHOLD_WATCH

#: Multiplier on the derived radius. A separation exactly at the probability
#: threshold sits *on* the boundary; a plan pushed to the boundary by the
#: optimizer lands on the wrong side of it as soon as the linearization is
#: imperfect. 1.5 buys margin without materially raising the premium.
INDUCED_EXCLUSION_MARGIN = 1.5


@dataclass
class LearnedHints:
    """What a trained model contributes, and nothing it is trusted for."""

    active_edges: set[str] = field(default_factory=set)
    edge_probabilities: dict[str, float] = field(default_factory=dict)
    initial_directions: dict[str, np.ndarray] = field(default_factory=dict)
    predicted_dv: np.ndarray | None = None
    predicted_duals: dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    threshold: float = 0.5

    @property
    def is_empty(self) -> bool:
        return not (
            self.active_edges
            or self.edge_probabilities
            or self.initial_directions
            or self.predicted_dv is not None
        )


@dataclass
class PlanRequest:
    """Everything a planner needs, with defaults drawn from the constants module."""

    assessed: AssessedCatalog
    objects: list[SpaceObject]
    window_start: datetime
    window_duration_s: float
    now: datetime | None = None
    target_pc: float = PC_TARGET_POST_MANEUVER
    induced_exclusion_km: float | None = None
    """Exclusion radius for a newly induced conjunction, km.

    ``None`` derives it from the assessed covariances rather than guessing:
    the separation at which the collision probability reaches
    :data:`INDUCED_EXCLUSION_LEVEL_PC`, times
    :data:`INDUCED_EXCLUSION_MARGIN`, floored at
    :data:`DEFAULT_INDUCED_EXCLUSION_KM`.

    Deriving it matters more than it looks. A geometric radius and a
    probability threshold are not the same criterion, and under TLE-grade
    covariance they are far apart: for a typical assessed pair here
    (sigma 1.19 / 0.32 km, HBR 6 m) the separation giving Pc = 1e-5 is
    2.10 km, just *above* the 2 km radial box. Planning against 2 km left nine
    SGP4-measured induced conjunctions across 24 scenarios; deriving the
    radius instead drove that to zero, while the unconstrained planner still
    created nineteen."""
    dv_budget_km_s: float = DEFAULT_DV_BUDGET_KM_S
    per_burn_cap_km_s: float = MAX_DV_PER_BURN_KM_S
    burn_slots: int = DEFAULT_BURN_SLOTS
    min_lead_orbits: float = MIN_LEAD_TIME_ORBITS
    axes: int = 3
    axis_weights: tuple[float, float, float] = DEFAULT_AXIS_WEIGHTS
    cost_model_name: str = "l1"
    cone_order: int = 1
    linearization_margin_km: float = 0.0
    """Extra separation demanded of every latent row, km.

    The optimizer drives constraints to equality: measured on the
    induced-cascade family, plans satisfied their tightest latent row by
    **10 metres**, and then an SGP4 re-screen found a WATCH-level conjunction
    there anyway. Nothing is wrong with the constraint -- the plan simply sits
    on a boundary whose position is only known to the accuracy of the
    linearised dynamics, which `docs/model-fidelity-validation.md` measures at
    about ten metres of absolute displacement error plus a second-order
    B-plane term.

    A sweep over 0, 100, 200, 500 and 1000 m across 80 scenarios showed the
    margin does **not** fix the problem it was introduced for -- measured
    induced conjunctions went 5, 6, 6, 4, 3 while the premium rose from 4.41 %
    to 5.87 % and infeasible scenarios from 10 to 16. The real cause was
    missing rows, not a boundary that was too tight (see
    :mod:`aegis.fleetopt.latent`). The knob is kept because it is the right
    tool if linearisation error ever does become the binding issue, and
    defaults to zero because on this evidence it is not."""

    induced_level_pc: float = PC_THRESHOLD_WATCH
    """Probability threshold a latent pair must stay below.

    Each latent row's floor is derived from *its own* projected covariance at
    this threshold, which is what makes "do not create a conjunction" a
    probability statement rather than a geometric one."""

    plan_risk_level: str = RiskLevel.MONITOR
    """Lowest risk band that earns a place in the optimization.

    Screening a dense shell reports every box entry, most of them at CLEAR --
    a 132-satellite Walker shell produced 11 936 of them over three days, with
    collision probabilities below 1e-18. Carrying those as decision-relevant
    rows is not conservatism, it is a category error: they are geometry
    reports, not risks, and the dense program they build is what made the
    first full sweep allocate 4.7 GB and stall. Operators plan against the
    actionable bands, and so does this.

    Conjunctions below the band are still reported, still re-screened, and
    still counted in the induced-conjunction measurement -- they are simply
    not constraints."""

    max_resolve_rows: int | None = 400
    """Hard ceiling on resolve rows, applied after the risk filter.

    Reached only by pathological catalogs. When it bites, the rows kept are
    the highest-probability ones and the plan records that it happened, because
    a silently truncated constraint set is a silently wrong answer."""

    latent_mode: str = LATENT_DEFAULT_MODE
    latent_step_s: float = 30.0
    max_latent_rows: int | None = 4000
    station_keeping_box_km: tuple[float, float] = STATION_KEEPING_BOX_KM
    enforce_station_keeping: bool = True
    slack_penalty: float = LP_SLACK_PENALTY
    scp_iterations: int = 6
    ops_cost_per_satellite: float = 0.0
    ops_cost_per_burn: float = 0.0
    screening_box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM
    hints: LearnedHints | None = None
    explain_infeasibility: bool = True

    def budgets(self, satellite_ids) -> dict[str, float]:
        return {sat_id: float(self.dv_budget_km_s) for sat_id in satellite_ids}


@dataclass
class InducedReport:
    """Predicted versus measured induced conjunctions.

    Both numbers are always carried together. ``predicted`` is what the
    optimizer's own latent rows say; ``measured`` comes from re-screening the
    maneuvered catalog with SGP4 and is filled in by
    :mod:`aegis.experiments`. A planner that reports only the first is
    marking its own homework, which is why the field exists here and is left
    ``None`` until an independent re-screen supplies it.
    """

    predicted_count: int = 0
    predicted_worst_shortfall_km: float = 0.0
    predicted_worsened_count: int = 0
    pre_existing_violations: int = 0
    latent_rows: int = 0
    latent_mode: str = ""
    measured_count: int | None = None
    measured_max_pc: float | None = None
    measured_aggregate_pc: float | None = None
    measured_pairs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "predicted_count": self.predicted_count,
            "predicted_worst_shortfall_km": round(self.predicted_worst_shortfall_km, 6),
            "predicted_worsened_count": self.predicted_worsened_count,
            "pre_existing_violations": self.pre_existing_violations,
            "latent_rows": self.latent_rows,
            "latent_mode": self.latent_mode,
            "measured_count": self.measured_count,
            "measured_max_pc": self.measured_max_pc,
            "measured_aggregate_pc": self.measured_aggregate_pc,
            "measured_pairs": list(self.measured_pairs[:20]),
        }


@dataclass
class FleetPlan:
    """A plan plus everything needed to judge it.

    ``maneuver_plan`` is an ordinary
    :class:`aegis.core.maneuver.ManeuverPlan`, so every existing consumer --
    the CCSDS writer, the pipeline export, the console -- keeps working
    unchanged. The research fields sit alongside it.
    """

    planner: str
    maneuver_plan: ManeuverPlan
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    problem_summary: dict = field(default_factory=dict)
    scp: ScpResult | None = None
    lazy: LazyResult | None = None
    certificate: Certificate | None = None
    graph: GraphMetrics | None = None
    induced: InducedReport | None = None
    duals: dict[str, float] = field(default_factory=dict)
    exact_penalty_threshold: float | None = None
    penetration: PenetrationIndex | None = None
    farkas: Farkas | None = None
    solver_time_s: float = 0.0
    total_time_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_delta_v_mm_s(self) -> float:
        return self.maneuver_plan.total_delta_v_mm_s

    @property
    def total_delta_v_km_s(self) -> float:
        return self.maneuver_plan.total_delta_v_mm_s * 1e-6

    @property
    def resolved_count(self) -> int:
        return len(self.maneuver_plan.resolved) - len(self.maneuver_plan.unresolved)

    @property
    def unresolved_count(self) -> int:
        return len(self.maneuver_plan.unresolved)

    @property
    def feasible(self) -> bool:
        return self.unresolved_count == 0

    @property
    def certified_safe(self) -> bool:
        return self.certificate is not None and self.certificate.linearized_safe

    def summary(self) -> dict:
        base = self.maneuver_plan.summary()
        base.update(
            {
                "planner": self.planner,
                "certified_safe": self.certified_safe,
                "solver_time_s": round(self.solver_time_s, 6),
                "total_time_s": round(self.total_time_s, 6),
                "exact_penalty_threshold": self.exact_penalty_threshold,
                "scp": self.scp.summary() if self.scp else None,
                "lazy": self.lazy.summary() if self.lazy else None,
                "certificate": self.certificate.summary() if self.certificate else None,
                "graph": self.graph.as_dict() if self.graph else None,
                "induced": self.induced.as_dict() if self.induced else None,
                "farkas": self.farkas.summary() if self.farkas else None,
                "penetration": self.penetration.as_dict() if self.penetration else None,
                "problem": dict(self.problem_summary),
                "notes": list(self.notes),
            }
        )
        return base


class Planner(Protocol):
    """Common planner surface."""

    name: str

    def plan(self, context: "PlanContext") -> FleetPlan: ...


@dataclass
class PlanContext:
    """Shared, planner-independent preparation.

    Building this is the expensive part -- propagation, screening for latent
    pairs, sensitivity assembly -- and doing it once per scenario rather than
    once per planner is what makes the comparison both fast and fair.
    """

    request: PlanRequest
    grid: BurnGrid
    sensitivities: list[MissSensitivity]
    latent: list[LatentConstraint]
    reachability: ReachabilityModel
    graph: GraphMetrics
    cost_model: CostModel
    latent_report: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    build_time_s: float = 0.0
    propagation: object | None = None
    """Nominal propagation grid, retained so latent rows can be created after
    the solve at the epochs the plan itself turns out to care about."""

    propagator: object | None = None
    """The SGP4 propagator behind that grid.

    Perturbed minima are found *between* grid samples, so a row placed at one
    needs states at an epoch the grid does not hold. Keeping the propagator
    costs nothing -- it is already built -- and avoids rebuilding satrecs per
    refinement round."""

    latent_floor_km: dict[str, float] = field(default_factory=dict)
    """Floor per pair id, for the perturbed-minimum search."""

    candidate_pairs: list[tuple[str, str]] = field(default_factory=list)
    """Pairs inside the reachability gate, whether or not they have a row yet."""

    shared_directions: dict[str, np.ndarray] | None = None
    """Linearization directions every solving planner should start from.

    Set by the benchmark runner after the first planner converges, so the
    whole comparison is taken at one linearization point and the nesting of
    the planners' feasible sets survives into their reported delta-v."""

    @property
    def objects(self) -> list[SpaceObject]:
        return self.request.objects

    @property
    def assessed(self) -> AssessedCatalog:
        return self.request.assessed

    def problem(
        self,
        *,
        latent: list[LatentConstraint] | None = None,
        integer: bool = False,
        induced_budget: float | None = None,
        slack_penalty: float | None = None,
    ) -> FleetProblem:
        penalty = self.request.slack_penalty if slack_penalty is None else slack_penalty
        return FleetProblem(
            grid=self.grid,
            resolve=list(self.sensitivities),
            latent=list(self.latent if latent is None else latent),
            cost_model=self.cost_model,
            budgets=self.request.budgets(self.grid.satellite_ids),
            per_burn_cap_km_s=self.request.per_burn_cap_km_s,
            station_keeping_box_km=self.request.station_keeping_box_km,
            enforce_station_keeping=self.request.enforce_station_keeping,
            resolve_slack_penalty=penalty,
            latent_slack_penalty=penalty,
            induced_budget=induced_budget,
            ops_cost_per_satellite=self.request.ops_cost_per_satellite,
            ops_cost_per_burn=self.request.ops_cost_per_burn,
            integer=integer,
        )


def derive_exclusion_radius_km(assessed: AssessedCatalog) -> float:
    """Separation at which an induced pair would reach the watch threshold.

    Uses the median assessed covariance and hard-body radius, because the
    radius has to be one number for the whole program while the covariances
    vary; the median is the choice that is wrong by the least across the
    catalog, and the margin absorbs the spread.
    """
    radii: list[float] = []
    for entry in assessed.entries:
        assessment = entry.assessment
        radii.append(
            required_miss_distance_km(
                assessment.hard_body_radius_m / 1000.0,
                assessment.sigma_major_km,
                assessment.sigma_minor_km,
                INDUCED_EXCLUSION_LEVEL_PC,
            )
        )
    if not radii:
        return DEFAULT_INDUCED_EXCLUSION_KM
    derived = float(np.median(radii)) * INDUCED_EXCLUSION_MARGIN
    return max(DEFAULT_INDUCED_EXCLUSION_KM, derived)


def _resolve_epochs(sensitivities: list[MissSensitivity]) -> dict[str, list[datetime]]:
    """Pair id -> the TCAs that pair already has a resolve row for."""
    epochs: dict[str, list[datetime]] = {}
    for sensitivity in sensitivities:
        key = pair_id(sensitivity.primary_id, sensitivity.secondary_id)
        epochs.setdefault(key, []).append(sensitivity.tca)
    return epochs


def _cost_model_for(request: PlanRequest, grid: BurnGrid) -> CostModel:
    if request.cost_model_name == "l1":
        return l1_cost_model(grid, axis_weights=request.axis_weights)
    if request.cost_model_name.startswith("cone"):
        return cone_cost_model(grid, order=request.cone_order, axis_weights=request.axis_weights)
    raise ValueError(f"unknown cost model {request.cost_model_name!r}; use 'l1' or 'cone'")


def build_context(request: PlanRequest) -> PlanContext:
    """Do every planner-independent step once."""
    started = time.perf_counter()
    notes: list[str] = []

    # Every TCA each satellite is involved in, not just its first: the grid
    # builder needs the whole list to pick one that still leaves room to act.
    earliest: dict[str, list[datetime]] = {}
    for entry in request.assessed.entries:
        conjunction = entry.conjunction
        for object_id in conjunction.maneuverable_object_ids:
            earliest.setdefault(object_id, []).append(conjunction.tca)

    if earliest:
        grid = build_burn_grid(
            request.objects,
            earliest,
            burn_slots=request.burn_slots,
            min_lead_orbits=request.min_lead_orbits,
            axes=request.axes,
            now=ensure_utc(request.now) if request.now is not None else None,
        )
    else:
        grid = BurnGrid(satellite_ids=(), epochs={}, mean_motion={}, axes=request.axes)
    if grid.excluded:
        notes.append(
            f"{len(grid.excluded)} satellite(s) had no usable burn slot: "
            + "; ".join(f"{k} ({v})" for k, v in sorted(grid.excluded.items())[:5])
        )

    threshold = RiskLevel.rank(request.plan_risk_level)
    planned = [
        entry
        for entry in request.assessed.entries
        if RiskLevel.rank(entry.assessment.risk_level) >= threshold
    ]
    skipped = len(request.assessed.entries) - len(planned)
    if skipped:
        notes.append(
            f"{skipped} conjunction(s) below {request.plan_risk_level} were reported but not "
            "made constraints; they are still re-screened and still counted in the "
            "induced-conjunction measurement"
        )
    if request.max_resolve_rows is not None and len(planned) > request.max_resolve_rows:
        planned = sorted(planned, key=lambda e: -e.assessment.probability)[
            : request.max_resolve_rows
        ]
        notes.append(
            f"resolve rows capped at {request.max_resolve_rows}; the highest-probability "
            "conjunctions were kept and the rest are reported but unconstrained"
        )

    sensitivities: list[MissSensitivity] = []
    for entry in planned:
        assessment = entry.assessment
        required = required_miss_distance_km(
            assessment.hard_body_radius_m / 1000.0,
            assessment.sigma_major_km,
            assessment.sigma_minor_km,
            request.target_pc,
        )
        sensitivities.append(
            build_miss_sensitivity(entry.conjunction, assessment, grid, required)
        )

    exclusion_km = request.induced_exclusion_km
    if exclusion_km is None:
        exclusion_km = derive_exclusion_radius_km(request.assessed)
        notes.append(
            f"induced-conjunction exclusion radius derived as {exclusion_km:.3f} km "
            f"from the assessed covariances at Pc = {INDUCED_EXCLUSION_LEVEL_PC:.0e}"
        )
    request.induced_exclusion_km = exclusion_km

    reachability = ReachabilityModel.from_grid(
        grid,
        request.budgets(grid.satellite_ids),
        exclusion_radius_km=exclusion_km,
    )

    latent: list[LatentConstraint] = []
    latent_report: dict = {"mode": request.latent_mode, "rows_after_thinning": 0}
    propagation_grid_ref = None
    propagator_ref = None
    latent_floors: dict[str, float] = {}
    candidate_pairs: list[tuple[str, str]] = []
    if request.latent_mode != "none" and grid.satellite_ids:
        try:
            propagator = Sgp4Propagator(request.objects)
            propagation = propagator.propagate_grid(
                ensure_utc(request.window_start),
                float(request.window_duration_s),
                float(request.latent_step_s),
            )
        except PropagationError as error:
            notes.append(f"latent enumeration skipped: {error}")
        else:
            propagation_grid_ref = propagation
            propagator_ref = propagator
            latent, report = enumerate_latent_constraints(
                grid,
                propagation,
                reachability,
                exclusion_radius_km=request.induced_exclusion_km,
                mode=request.latent_mode,
                resolve_epochs=_resolve_epochs(sensitivities),
                covariance_model=default_covariance_model(),
                objects_by_id={obj.object_id: obj for obj in request.objects},
                target_pc=request.induced_level_pc,
                max_rows=request.max_latent_rows,
                linearization_margin_km=request.linearization_margin_km,
            )
            latent_report = report.summary()
            notes.extend(report.notes)
            latent_floors = {c.pair_id: c.floor_km for c in latent}
            # From what the gate admitted, not from what got a row. A pair
            # whose whole approach falls between samples has no row to derive
            # membership from, and the perturbed-minimum search can only refine
            # pairs it is handed -- which is how a 120 s grid dropped a
            # 7 km/s crossing outright.
            candidate_pairs = sorted(
                set(report.admitted_pairs) | {(c.object_a, c.object_b) for c in latent}
            )

    conjunction_graph = build_conjunction_graph(request.assessed, request.objects)
    metrics = graph_metrics(conjunction_graph, sensitivities=sensitivities, latent=latent)

    return PlanContext(
        request=request,
        grid=grid,
        sensitivities=sensitivities,
        latent=latent,
        reachability=reachability,
        graph=metrics,
        cost_model=_cost_model_for(request, grid),
        latent_report=latent_report,
        propagation=propagation_grid_ref,
        propagator=propagator_ref,
        latent_floor_km=latent_floors,
        candidate_pairs=candidate_pairs,
        notes=notes,
        build_time_s=time.perf_counter() - started,
    )


def _new_plan_id(planner: str) -> str:
    return f"{planner}-{uuid.uuid4().hex}"


def _maneuvers_from_x(
    context: PlanContext, x: np.ndarray, rationale_by_sat: dict[str, list[str]]
) -> dict[str, SatelliteManeuverSet]:
    grid = context.grid
    sets: dict[str, SatelliteManeuverSet] = {}
    for sat_id in grid.satellite_ids:
        maneuvers: list[Maneuver] = []
        for slot, epoch in enumerate(grid.slot_epochs(sat_id)):
            impulse = np.zeros(3)
            for axis in (range(3) if grid.axes == 3 else (1,)):
                impulse[axis] = float(x[grid.index(sat_id, slot, axis)])
            if float(np.linalg.norm(impulse)) < _DV_EMIT_FLOOR_KM_S:
                continue
            maneuvers.append(
                Maneuver(
                    satellite_id=sat_id,
                    epoch=epoch,
                    delta_v_rtn_km_s=impulse,
                    rationale=list(rationale_by_sat.get(sat_id, [])),
                )
            )
        if maneuvers:
            maneuvers.sort(key=lambda item: item.epoch)
            sets[sat_id] = SatelliteManeuverSet(satellite_id=sat_id, maneuvers=maneuvers)
    return sets


def _resolved_from_x(
    context: PlanContext, x: np.ndarray, slack: dict[str, float]
) -> list[ResolvedConjunction]:
    outcomes: list[ResolvedConjunction] = []
    for sensitivity in context.sensitivities:
        miss_after = max(0.0, sensitivity.miss_after_km(x))
        shortfall = max(0.0, sensitivity.required_miss_km - miss_after)
        absorbed = slack.get(sensitivity.conjunction_id, 0.0)
        resolved = shortfall <= _SLACK_RESOLVED_KM and absorbed <= 1e-6
        probability_after = collision_probability(
            sensitivity.sigma_major_km,
            sensitivity.sigma_minor_km,
            miss_after,
            0.0,
            sensitivity.hard_body_radius_km,
        )
        outcomes.append(
            ResolvedConjunction(
                conjunction_id=sensitivity.conjunction_id,
                probability_before=sensitivity.probability_before,
                probability_after=probability_after,
                miss_distance_before_km=sensitivity.nominal_miss_km,
                miss_distance_after_km=miss_after,
                required_miss_distance_km=sensitivity.required_miss_km,
                resolved=resolved,
                shortfall_km=0.0 if resolved else shortfall,
            )
        )
    return outcomes


def _rationale(context: PlanContext) -> dict[str, list[str]]:
    rationale: dict[str, list[str]] = {}
    for sensitivity in context.sensitivities:
        for object_id in (sensitivity.primary_id, sensitivity.secondary_id):
            if context.grid.has(object_id):
                rationale.setdefault(object_id, []).append(sensitivity.conjunction_id)
    return rationale


def _induced_report(context: PlanContext, latent: list[LatentConstraint], x: np.ndarray) -> InducedReport:
    """What the plan *created*, not what was already there.

    A latent row can be violated at ``x = 0`` -- two fleet-mates already
    closer than the exclusion radius, which in a tightly packed shell is
    common. Counting those as induced makes a plan with no burns at all
    report eleven induced conjunctions, which is how this was found. Only a
    row that is satisfied at zero delta-v and violated by the plan counts.

    Rows already violated are still reported, separately, because a plan that
    makes an existing violation *worse* is also worth knowing about.
    """
    zero = np.zeros_like(x)
    induced = 0
    worsened = 0
    pre_existing = 0
    worst = 0.0
    for constraint in latent:
        before = constraint.residual_km(zero)
        after = constraint.residual_km(x)
        if before < -1e-9:
            pre_existing += 1
            if after < before - 1e-9:
                worsened += 1
            continue
        if after < -1e-9:
            induced += 1
            worst = max(worst, -after)
    return InducedReport(
        predicted_count=induced,
        predicted_worst_shortfall_km=worst,
        predicted_worsened_count=worsened,
        pre_existing_violations=pre_existing,
        latent_rows=len(latent),
        latent_mode=context.request.latent_mode,
    )


def _assemble_plan(
    context: PlanContext,
    planner: str,
    x: np.ndarray,
    *,
    scp: ScpResult | None,
    lazy: LazyResult | None = None,
    latent: list[LatentConstraint] | None = None,
    extra_notes: list[str] | None = None,
    solver_time_s: float = 0.0,
    problem: FleetProblem | None = None,
) -> FleetPlan:
    latent = list(context.latent if latent is None else latent)
    resolve_slack: dict[str, float] = {}
    latent_slack: dict[str, float] = {}
    duals: dict[str, float] = {}
    if scp is not None and scp.solution.ok:
        resolve_slack = scp.solution.slack(scp.data, "resolve")
        latent_slack = scp.solution.slack(scp.data, "latent")
        duals = dict(scp.solution.duals)

    maneuver_plan = ManeuverPlan(
        plan_id=_new_plan_id(planner),
        generated_at=utc_now(),
        satellite_sets=_maneuvers_from_x(context, x, _rationale(context)),
        resolved=_resolved_from_x(context, x, resolve_slack),
        notes=list(context.notes) + list(extra_notes or []),
    )

    verify_problem = problem if problem is not None else context.problem(latent=latent)
    certificate = verify_plan(
        verify_problem, x, slack={**resolve_slack, **latent_slack}
    )

    farkas: Farkas | None = None
    if (
        context.request.explain_infeasibility
        and maneuver_plan.unresolved
        and context.grid.n_vars > 0
        and len(context.sensitivities) + len(latent) <= _FARKAS_ROW_LIMIT
    ):
        try:
            farkas = irreconcilable_subset(
                verify_problem,
                scp.directions if scp is not None else None,
                max_rows=24,
            )
        except Exception as error:  # noqa: BLE001 - an explanation must never break a plan
            maneuver_plan.notes.append(f"infeasibility explanation unavailable: {error}")

    threshold = exact_penalty_threshold(scp.solution) if scp is not None else None

    summary = {
        "n_vars": context.grid.n_vars,
        "satellites": len(context.grid.satellite_ids),
        "resolve_rows": len(context.sensitivities),
        "latent_rows": len(latent),
        "latent": dict(context.latent_report),
        "cost_model": context.cost_model.name,
        "cost_note": context.cost_model.approximation_note,
        "reachability": context.reachability.summary(
            ensure_utc(context.request.window_start)
        ),
        "context_build_s": round(context.build_time_s, 6),
    }

    return FleetPlan(
        planner=planner,
        maneuver_plan=maneuver_plan,
        x=np.asarray(x, dtype=float),
        problem_summary=summary,
        scp=scp,
        lazy=lazy,
        certificate=certificate,
        graph=context.graph,
        # Scored against every latent row the CONTEXT knows about, not just the
        # rows this planner chose to impose. A planner that ignores the
        # induced-conjunction constraints must still be told what it induced,
        # or `fuel-only` reports zero by construction and the whole comparison
        # is circular.
        induced=_induced_report(context, list(context.latent), x),
        duals=duals,
        exact_penalty_threshold=threshold,
        penetration=(
            penetration_index(
                list(context.latent),
                context.grid,
                context.request.budgets(context.grid.satellite_ids),
                x,
            )
            if context.latent and context.grid.n_vars
            else None
        ),
        farkas=farkas,
        solver_time_s=solver_time_s,
    )


def _empty_plan(context: PlanContext, planner: str, note: str) -> FleetPlan:
    """A plan with no burns that still reports what it failed to address.

    An empty ``resolved`` list would make a planner that *could not act* look
    identical to one that had nothing to do -- ``unresolved == 0`` and
    ``feasible == True``. Every conjunction is therefore listed, evaluated at
    ``x = 0``, so a planner with no usable burn slot reports its conjunctions
    as unresolved with their real shortfalls.
    """
    x = np.zeros(max(0, context.grid.n_vars))
    return FleetPlan(
        planner=planner,
        maneuver_plan=ManeuverPlan(
            plan_id=_new_plan_id(planner),
            generated_at=utc_now(),
            resolved=_resolved_from_x(context, x, {}),
            notes=list(context.notes) + [note],
        ),
        x=x,
        graph=context.graph,
        induced=_induced_report(context, list(context.latent), x),
        notes=[note],
    )


def solve_two_stage(
    context: PlanContext,
    problem: FleetProblem,
    *,
    initial_directions: dict[str, np.ndarray] | None = None,
    two_stage: bool = True,
) -> tuple[ScpResult, list[str]]:
    """Minimise shortfall, then minimise delta-v underneath it.

    Shared by every solving planner so they all optimise the *same* objective.
    They must: comparing a two-stage plan's delta-v against a single-stage
    plan's LP objective compares a fuel number with a fuel-plus-penalty
    number, and the two differ by seven orders of magnitude. That mistake made
    ``pignn-active-set`` look like it disagreed with ``fleet-safe`` on 43 of
    44 scenarios when both were in fact returning the same plan.

    Stage two pins each row's shortfall individually, not just the total, so
    it cannot resolve fewer conjunctions than stage one while keeping the sum
    unchanged.
    """
    scp = sequential_solve(
        problem,
        max_iterations=context.request.scp_iterations,
        initial_directions=initial_directions,
    )
    notes = list(scp.notes)
    if not (two_stage and scp.solution.ok):
        return scp, notes

    resolve_slack = scp.solution.slack(scp.data, "resolve")
    latent_slack = scp.solution.slack(scp.data, "latent")
    achieved = float(sum(resolve_slack.values()) + sum(latent_slack.values()))

    second = problem.with_latent(list(problem.latent))
    second.resolve_slack_penalty = 0.0
    second.latent_slack_penalty = 0.0
    second.total_slack_budget = achieved + _SLACK_BUDGET_TOL_KM
    second.slack_caps = {
        f"resolve:{key}": value + _SLACK_BUDGET_TOL_KM for key, value in resolve_slack.items()
    }
    second.slack_caps.update(
        {key: value + _SLACK_BUDGET_TOL_KM for key, value in latent_slack.items()}
    )
    polished = sequential_solve(
        second,
        max_iterations=context.request.scp_iterations,
        initial_directions=scp.directions,
    )
    if polished.solution.ok:
        notes.append(
            f"stage 1 reached {achieved:.6g} km of total shortfall; stage 2 minimised "
            "delta-v subject to not exceeding it, row by row"
        )
        notes.extend(polished.notes)
        return polished, notes
    notes.append(
        f"stage 2 returned {polished.solution.status!r}; keeping the stage-1 plan, "
        "whose delta-v is not a minimum"
    )
    return scp, notes


def _rows_at_perturbed_minima(
    context: PlanContext,
    x: np.ndarray,
    existing: list[LatentConstraint],
) -> list[LatentConstraint]:
    """Latent rows at the epochs the *plan* turns out to care about.

    Enumeration places rows at minima of the nominal separation. For a
    co-orbital pair the nominal separation is flat, so those minima are
    arbitrary and the plan's own drift decides where the real minimum goes --
    measured at 2.57 km, two kilometres below the floor, at an epoch no row
    covered. This finds those epochs from the solved ``x`` and returns one new
    row each, so the next solve constrains them.
    """
    if context.propagation is None or not context.candidate_pairs:
        return []
    found = perturbed_minimum_epochs(
        context.grid, context.propagation, x, context.candidate_pairs, context.latent_floor_km
    )
    if not found:
        return []

    covered = {(c.pair_id, c.epoch) for c in existing}
    index_of = {oid: i for i, oid in enumerate(context.propagation.object_ids)}
    times = np.asarray(context.propagation.times_s, dtype=float)
    rows: list[LatentConstraint] = []
    for object_a, object_b, time_index, _separation, fraction in found[:_MAX_NEW_EPOCH_ROWS]:
        # The minimum generally sits between samples -- for a fast crossing,
        # far below either neighbour -- so the row goes at the refined epoch,
        # not at the sample that happens to bracket it.
        offset_s = 0.0
        if fraction > 0.0 and time_index + 1 < times.size:
            offset_s = float(fraction) * float(times[time_index + 1] - times[time_index])
        epoch = shift(context.propagation.epoch_at(time_index), offset_s)
        key = pair_id(object_a, object_b)
        if (key, epoch) in covered:
            continue
        row_a, row_b = index_of[object_a], index_of[object_b]
        if offset_s == 0.0 or context.propagator is None:
            state_a = context.propagation.state(row_a, time_index)
            state_b = context.propagation.state(row_b, time_index)
        else:
            try:
                state_a = context.propagator.propagate_one(row_a, epoch)
                state_b = context.propagator.propagate_one(row_b, epoch)
            except PropagationError:
                # Falling back to the bracketing sample is worse than the
                # refined epoch but never worse than the old behaviour.
                epoch = context.propagation.epoch_at(time_index)
                if (key, epoch) in covered:
                    continue
                state_a = context.propagation.state(row_a, time_index)
                state_b = context.propagation.state(row_b, time_index)
        rows.append(
            build_latent_constraint(
                context.grid,
                object_a,
                object_b,
                epoch,
                state_a,
                state_b,
                floor_km=context.latent_floor_km.get(key, context.request.induced_exclusion_km or 0.0),
                kind="bplane_minimum",
            )
        )
    return rows


def solve_lazy_two_stage(
    context: PlanContext,
    problem: FleetProblem,
    *,
    guess: set[str] | None = None,
    initial_directions: dict[str, np.ndarray] | None = None,
) -> tuple[ScpResult, LazyResult | None, list[str], list[LatentConstraint]]:
    """Close the induced-conjunction row set lazily, then polish, then re-verify.

    Three steps, and the order matters:

    1. Solve with only ``guess`` of the latent rows and let
       :func:`aegis.fleetopt.solver.lazy_solve` add back any omitted row the
       solution violates, until none does. The objective is then the full
       problem's optimum, for any guess (see that function's docstring).
    2. Run the two-stage polish on the closed row set.
    3. Re-verify every held-back row against the *polished* solution, because
       stage two moves the plan. Add violations and repeat.

    Returns the closed latent row set as its fourth element. Rows discovered
    at perturbed minima are part of what the plan was solved against, so a
    certificate that only sees the originally enumerated rows can report
    ``linearized_safe`` for a plan that violates a row this function added.

    **The default guess is empty, and that is not laziness.** Measured against
    solving with all rows present: bit-identical delta-v (relative difference
    0.0e+00 across 12 scenarios from 88 to 1698 latent rows) at up to 5x the
    speed, using 1 to 11 rows. Every alternative was slower -- a trained graph
    network, certified analytic per-row pruning, and an analytic a-posteriori
    guess all lost to the empty start, because the active set holds two to
    eleven rows out of seventeen hundred and any predictor must be cheaper
    than free to beat it. See ``docs/pignn-acceleration-results.md``.
    """
    if not problem.latent:
        scp, notes = solve_two_stage(
            context, problem, initial_directions=initial_directions
        )
        return scp, None, notes, list(problem.latent)

    lazy = lazy_solve(
        problem,
        active_guess=set(guess or set()),
        max_iterations=context.request.scp_iterations,
        initial_directions=initial_directions,
    )
    notes = list(lazy.notes)
    by_label = {c.label: c for c in problem.latent}
    active = [by_label[label] for label in lazy.active_labels if label in by_label]
    scp = lazy.scp

    for round_index in range(1, _POLISH_ROUNDS + 1):
        polished, stage_notes = solve_two_stage(
            context,
            problem.with_latent(list(active)),
            initial_directions=lazy.scp.directions,
        )
        if not polished.solution.ok:
            notes.append(
                f"two-stage polish returned {polished.solution.status!r}; keeping the "
                "lazy-loop solution, whose delta-v is not a minimum"
            )
            break
        held_back = {c.label for c in active}
        violated = [
            c
            for c in problem.latent
            if c.label not in held_back and c.linear_residual_km(polished.x) < -1e-9
        ]
        scp = polished
        notes.extend(stage_notes)
        if not violated:
            fresh = _rows_at_perturbed_minima(context, polished.x, problem.latent)
            if not fresh:
                break
            problem = problem.with_latent(list(problem.latent) + fresh)
            active.extend(fresh)
            by_label.update({c.label: c for c in fresh})
            notes.append(
                f"polish round {round_index}: the perturbed separation bottoms out away "
                f"from every enumerated epoch for {len(fresh)} pair(s); added a row at "
                "each true minimum and re-solved"
            )
            continue
        active.extend(violated)
        notes.append(
            f"polish round {round_index} exposed {len(violated)} newly violated row(s); "
            "added and re-solved"
        )

    notes.append(
        f"closed={lazy.closed} in {lazy.rounds} lazy round(s) using "
        f"{len(active)}/{len(problem.latent)} latent rows "
        f"({100.0 * (1.0 - len(active) / max(len(problem.latent), 1)):.1f}% held back)"
    )
    return scp, lazy, notes, list(problem.latent)


class _SolvingPlanner:
    """Shared solve-and-assemble body for the optimizer-based planners.

    Two stages by default, and the reason is the exact-penalty result turned
    on its head. Proposition 6 says a penalty above the largest dual makes the
    soft and hard problems coincide *when the hard problem is feasible*. When
    it is not -- when some conjunction simply cannot be resolved inside the
    budget -- the same large penalty means the optimizer will spend every
    millimetre per second it has to shave any shortfall at all, because
    shortfall is priced at 1e4 per km against a delta-v cost of order 1e-3.
    The delta-v it reports is then not a minimum of anything, and a
    delta-v comparison between two such plans is meaningless. Measured on the
    induced-cascade family, that produced *negative* safety premiums, which
    nesting of the feasible sets makes impossible.

    So: stage one minimises total shortfall; stage two pins that shortfall
    with a ``slack-budget`` row and minimises delta-v underneath it. The
    result is the cheapest plan achieving the best safety the geometry allows,
    which is the quantity the premium is supposed to compare.

    ``two_stage = False`` recovers the single-stage penalty formulation, kept
    so the two can be compared directly.
    """

    name = "base"
    use_latent = False
    integer = False
    two_stage = True

    def plan(self, context: PlanContext) -> FleetPlan:
        if not context.sensitivities:
            return _empty_plan(context, self.name, "no conjunctions to plan for")
        if context.grid.n_vars == 0:
            return _empty_plan(
                context, self.name, "no maneuverable satellite has a usable burn slot"
            )
        latent = list(context.latent) if self.use_latent else []
        problem = context.problem(latent=latent, integer=self.integer)
        started = time.perf_counter()
        lazy: LazyResult | None = None
        if self.integer or not self.two_stage:
            # A mixed-integer program has no duals and the lazy loop's
            # lower-bound argument is about the continuous relaxation, so the
            # row set is not closed lazily. The *epoch* refinement is
            # independent of that argument, so it still runs: rows are added at
            # each pair's perturbed minimum and the problem re-solved.
            scp, notes = solve_two_stage(
                context,
                problem,
                initial_directions=context.shared_directions,
                two_stage=self.two_stage and not self.integer,
            )
            for _ in range(_POLISH_ROUNDS):
                if not scp.solution.ok:
                    break
                fresh = _rows_at_perturbed_minima(context, scp.x, problem.latent)
                if not fresh:
                    break
                problem = problem.with_latent(list(problem.latent) + fresh)
                latent = list(problem.latent)
                scp, extra = solve_two_stage(
                    context,
                    problem,
                    initial_directions=scp.directions,
                    two_stage=self.two_stage and not self.integer,
                )
                notes.extend(extra)
                notes.append(
                    f"added {len(fresh)} row(s) at perturbed minima and re-solved"
                )
        else:
            scp, lazy, notes, latent = solve_lazy_two_stage(
                context, problem, initial_directions=context.shared_directions
            )
            problem = problem.with_latent(list(latent))
        elapsed = time.perf_counter() - started
        if not scp.solution.ok:
            notes.append(
                f"solver returned {scp.solution.status!r}; emitting a no-burn plan so the "
                "unresolved conjunctions are still reported"
            )
            return _assemble_plan(
                context, self.name, np.zeros(context.grid.n_vars),
                scp=scp, lazy=lazy, latent=latent, extra_notes=notes,
                solver_time_s=elapsed, problem=problem,
            )
        return _assemble_plan(
            context, self.name, scp.x, scp=scp, lazy=lazy, latent=latent,
            extra_notes=notes, solver_time_s=elapsed, problem=problem,
        )


class NoManeuverPlanner:
    name = "no-maneuver"

    def plan(self, context: PlanContext) -> FleetPlan:
        x = np.zeros(max(0, context.grid.n_vars))
        return _assemble_plan(
            context,
            self.name,
            x,
            scp=None,
            latent=list(context.latent),
            extra_notes=["baseline: no burns commanded"],
        )


class LegacyLpPlanner:
    """Delegates to the untouched scalar-model planner, for ablation."""

    name = "legacy-lp"

    def plan(self, context: PlanContext) -> FleetPlan:
        request = context.request
        started = time.perf_counter()
        legacy = plan_maneuvers(
            request.assessed,
            request.objects,
            now=request.now,
            target_pc=request.target_pc,
            dv_budget_km_s=request.dv_budget_km_s,
            slack_penalty=request.slack_penalty,
            burn_slots=request.burn_slots,
            min_lead_orbits=request.min_lead_orbits,
        )
        elapsed = time.perf_counter() - started

        # Re-express the legacy burns in this package's decision vector so the
        # same certificate and the same induced report can be computed. Burn
        # epochs that do not land on a grid slot are attributed to the nearest
        # slot, and any that cannot be attributed are reported rather than
        # dropped silently.
        x = np.zeros(max(0, context.grid.n_vars))
        unmapped = 0
        for sat_id, sat_set in legacy.satellite_sets.items():
            if not context.grid.has(sat_id):
                unmapped += len(sat_set.maneuvers)
                continue
            epochs = context.grid.slot_epochs(sat_id)
            for maneuver in sat_set.maneuvers:
                deltas = [abs((epoch - maneuver.epoch).total_seconds()) for epoch in epochs]
                slot = int(np.argmin(deltas))
                for axis in (range(3) if context.grid.axes == 3 else (1,)):
                    x[context.grid.index(sat_id, slot, axis)] += float(
                        maneuver.delta_v_rtn_km_s[axis]
                    )
        notes = [
            "legacy scalar-sensitivity planner; see docs/model-fidelity-validation.md "
            "for its measured sign error on cross-plane intra-fleet conjunctions"
        ]
        if unmapped:
            notes.append(f"{unmapped} legacy burn(s) had no corresponding grid slot")

        plan = _assemble_plan(
            context, self.name, x, scp=None, latent=list(context.latent),
            extra_notes=notes, solver_time_s=elapsed,
        )
        # The legacy planner's own resolution verdict is what the ablation is
        # about, so it is preserved rather than recomputed.
        plan.maneuver_plan.resolved = list(legacy.resolved)
        plan.notes.append("resolution verdicts are the legacy planner's own")
        return plan


class GreedyPairwisePlanner:
    """One conjunction at a time, in TCA order, with no coordination."""

    name = "greedy-pairwise"

    def plan(self, context: PlanContext) -> FleetPlan:
        if not context.sensitivities:
            return _empty_plan(context, self.name, "no conjunctions to plan for")
        if context.grid.n_vars == 0:
            return _empty_plan(
                context, self.name, "no maneuverable satellite has a usable burn slot"
            )

        ordered = sorted(context.sensitivities, key=lambda s: s.tca)
        accumulated = np.zeros(context.grid.n_vars)
        remaining = dict(context.request.budgets(context.grid.satellite_ids))
        started = time.perf_counter()
        last_scp: ScpResult | None = None
        notes = [
            "each conjunction solved in isolation and the burns summed; no "
            "coordination, no induced-conjunction rows, budgets consumed in TCA order"
        ]

        for sensitivity in ordered:
            single = FleetProblem(
                grid=context.grid,
                resolve=[sensitivity],
                latent=[],
                cost_model=context.cost_model,
                budgets=dict(remaining),
                per_burn_cap_km_s=context.request.per_burn_cap_km_s,
                station_keeping_box_km=context.request.station_keeping_box_km,
                enforce_station_keeping=context.request.enforce_station_keeping,
                resolve_slack_penalty=context.request.slack_penalty,
                latent_slack_penalty=context.request.slack_penalty,
            )
            scp = sequential_solve(single, max_iterations=context.request.scp_iterations)
            if not scp.solution.ok:
                notes.append(f"{sensitivity.conjunction_id}: solver {scp.solution.status}")
                continue
            accumulated += scp.x
            last_scp = scp
            for sat_id in context.grid.satellite_ids:
                used = float(
                    sum(
                        abs(scp.x[context.grid.index(sat_id, slot, axis)])
                        for slot in range(len(context.grid.slot_epochs(sat_id)))
                        for axis in (range(3) if context.grid.axes == 3 else (1,))
                    )
                )
                remaining[sat_id] = max(0.0, remaining[sat_id] - used)

        elapsed = time.perf_counter() - started
        return _assemble_plan(
            context, self.name, accumulated, scp=last_scp, latent=list(context.latent),
            extra_notes=notes, solver_time_s=elapsed,
        )


class FuelOnlyPlanner(_SolvingPlanner):
    name = "fuel-only"
    use_latent = False


class FleetSafePlanner(_SolvingPlanner):
    """Resolve constraints plus the certified induced-conjunction rows.

    Optionally warm-started from another planner's converged linearization
    directions. That matters for the safety premium: ``fleet-safe``'s feasible
    set is a *subset* of ``fuel-only``'s, so at a fixed linearization it can
    never cost less. Let each run its own sequential refinement and that
    nesting stops holding across the pair -- the extra rows change the first
    iterate, the two walk different direction paths, and ``fleet-safe`` can
    land in a cheaper basin. Measured on the induced-cascade family at 3 km
    neighbour spacing, that produced premiums of -11 % and -0.05 %, which are
    not physical results but bookkeeping artifacts.

    Warm-starting from ``fuel-only``'s directions makes the two comparable:
    both restrictions are taken at the same point, so the premium is a genuine
    lower bound on the cost of the extra constraints.
    """

    name = "fleet-safe"
    use_latent = True


class MilpOpsPlanner(_SolvingPlanner):
    name = "milp-ops"
    use_latent = True
    integer = True


class LexicographicPlanner:
    """Safety first, strictly; delta-v only among equally safe plans."""

    name = "lexicographic"

    def plan(self, context: PlanContext) -> FleetPlan:
        if not context.sensitivities:
            return _empty_plan(context, self.name, "no conjunctions to plan for")
        if context.grid.n_vars == 0:
            return _empty_plan(
                context, self.name, "no maneuverable satellite has a usable burn slot"
            )

        started = time.perf_counter()
        notes: list[str] = []
        latent_rows = list(context.latent)
        first = None
        second = None
        lazy = None
        achieved = 0.0

        # The two stages are iterated to a fixed point over the row set, not
        # run once each. Stage one measures the shortfall the geometry allows;
        # stage two pins it and minimises delta-v underneath. Both numbers are
        # only meaningful on a *closed* row set, and the set is not closed
        # until neither stage's solution reveals a new epoch -- which it can,
        # because the two stages land on different plans and perturbed minima
        # are a property of the plan. Pinning a level measured before the set
        # closed is what left this planner inducing conjunctions that
        # `fleet-safe`, which pins nothing, avoided. See C15 in
        # docs/prior-art-and-novelty-ledger.md.
        for outer in range(1, _LEXICOGRAPHIC_ROUNDS + 1):
            # Stage 1: drive total shortfall as low as the geometry allows by
            # pricing slack far above any achievable delta-v cost. This is the
            # exact-penalty argument of Proposition 6 used deliberately: a
            # penalty above the largest dual makes the soft problem reproduce
            # the hard one.
            stage_one = FleetProblem(
                grid=context.grid,
                resolve=list(context.sensitivities),
                latent=list(latent_rows),
                cost_model=context.cost_model,
                budgets=context.request.budgets(context.grid.satellite_ids),
                per_burn_cap_km_s=context.request.per_burn_cap_km_s,
                station_keeping_box_km=context.request.station_keeping_box_km,
                enforce_station_keeping=context.request.enforce_station_keeping,
                resolve_slack_penalty=1e12,
                latent_slack_penalty=1e12,
            )
            # Start from the shared linearization like every other solving
            # planner. The restricted problem is non-convex, so the starting
            # directions decide which basin the sequential refinement lands
            # in; stage one ignoring them is how this planner came to report a
            # *worse* attainable shortfall than `fleet-safe` on the same
            # geometry, and then honestly minimised delta-v against it.
            first = sequential_solve(
                stage_one,
                max_iterations=context.request.scp_iterations,
                initial_directions=(
                    first.directions if first is not None else context.shared_directions
                ),
            )
            if not first.solution.ok:
                notes.append(f"stage 1 returned {first.solution.status!r}")
                elapsed = time.perf_counter() - started
                return _assemble_plan(
                    context, self.name, np.zeros(context.grid.n_vars), scp=first,
                    latent=list(latent_rows), extra_notes=notes, solver_time_s=elapsed,
                )

            # Close stage one's own epoch set before reading the level off it.
            for _ in range(_POLISH_ROUNDS):
                fresh = _rows_at_perturbed_minima(context, first.x, latent_rows)
                if not fresh:
                    break
                latent_rows.extend(fresh)
                stage_one = stage_one.with_latent(latent_rows)
                refined = sequential_solve(
                    stage_one,
                    max_iterations=context.request.scp_iterations,
                    initial_directions=first.directions,
                )
                if not refined.solution.ok:
                    break
                first = refined

            achieved = float(
                sum(first.solution.slack(first.data, "resolve").values())
                + sum(first.solution.slack(first.data, "latent").values())
            )

            # Stage 2: minimise delta-v subject to not exceeding that
            # shortfall.
            #
            # The pin is left to the shared path, which caps every *row*
            # separately rather than their sum. Capping the sum as well --
            # which this planner used to do -- stacks two budgets measured on
            # different iterates, and they fight: stage two comes back
            # `infeasible`, the planner keeps a solution whose delta-v is not
            # a minimum, and it reports a *worse* attainable safety than
            # `fleet-safe` on identical geometry. The row-by-row cap is also
            # the stronger of the two readings of "safety first": bounding
            # only the total lets the optimizer trade a resolved conjunction
            # for a slightly worse one elsewhere at no cost to the objective.
            stage_two = context.problem(latent=list(latent_rows))
            stage_two.resolve_slack_penalty = 1e12
            stage_two.latent_slack_penalty = 1e12
            second, lazy, stage_notes, closed = solve_lazy_two_stage(
                context, stage_two, initial_directions=first.directions
            )
            latent_rows = list(closed)
            if outer == 1:
                notes.extend(stage_notes)

            if not second.solution.ok:
                notes.append(
                    f"round {outer}: stage 2 returned {second.solution.status!r} against a "
                    f"shortfall budget of {achieved:.6g} km; re-measuring it on the closed "
                    "row set"
                )
                continue

            # Closed when the delta-v-minimal plan reveals no new epoch either.
            fresh = _rows_at_perturbed_minima(context, second.x, latent_rows)
            if not fresh:
                if outer > 1:
                    notes.append(
                        f"row set closed after {outer} safety/fuel rounds; the shortfall "
                        "budget and the rows it was measured on are consistent"
                    )
                break
            latent_rows.extend(fresh)
            notes.append(
                f"round {outer}: the delta-v-minimal plan bottoms out at {len(fresh)} epoch(s) "
                "no row covered; added them and re-measured the attainable shortfall"
            )
        else:
            notes.append(
                f"row set still growing after {_LEXICOGRAPHIC_ROUNDS} safety/fuel rounds; "
                "reporting the last plan, whose certificate is the arbiter"
            )

        elapsed = time.perf_counter() - started
        notes.insert(0, "stage 1 minimised total safety shortfall with delta-v nearly free")
        notes.append(
            f"stage 2 minimised delta-v subject to total shortfall <= {achieved:.6g} km"
        )
        chosen = second if (second is not None and second.solution.ok) else first
        return _assemble_plan(
            context, self.name, chosen.x, scp=chosen, lazy=lazy,
            latent=list(latent_rows), extra_notes=notes, solver_time_s=elapsed,
        )


class PignnActiveSetPlanner:
    """Fleet-safe with a model-supplied starting guess for the active set.

    Identical to ``fleet-safe`` except for where the lazy loop starts. Kept as
    a separate planner so the model's contribution can be measured against the
    empty start it has to beat -- and, as measured, does not. The objective is
    the exact fleet-safe optimum either way.
    """

    name = "pignn-active-set"

    def plan(self, context: PlanContext) -> FleetPlan:
        if not context.sensitivities:
            return _empty_plan(context, self.name, "no conjunctions to plan for")
        if context.grid.n_vars == 0:
            return _empty_plan(
                context, self.name, "no maneuverable satellite has a usable burn slot"
            )

        hints = context.request.hints
        notes: list[str] = []
        guess: set[str] = set()
        if hints is None or hints.is_empty:
            notes.append(
                "no learned hints supplied; this is then exactly `fleet-safe`, which "
                "already starts from an empty guess"
            )
        else:
            guess = set(hints.active_edges)
            notes.append(
                f"active set predicted by {hints.model_id or 'an unnamed model'}: "
                f"{len(guess)} of {len(context.latent)} latent rows at threshold "
                f"{hints.threshold:.4f}"
            )

        problem = context.problem(latent=list(context.latent))
        started = time.perf_counter()
        scp, lazy, solve_notes, latent = solve_lazy_two_stage(
            context,
            problem,
            guess=guess,
            initial_directions=(hints.initial_directions if hints else context.shared_directions),
        )
        elapsed = time.perf_counter() - started
        notes.extend(solve_notes)
        return _assemble_plan(
            context, self.name, scp.x, scp=scp, lazy=lazy,
            latent=list(latent), extra_notes=notes, solver_time_s=elapsed,
            problem=problem,
        )


class PignnWarmStartPlanner(_SolvingPlanner):
    """Fleet-safe with network-supplied initial linearization directions."""

    name = "pignn-warm-start"
    use_latent = True

    def plan(self, context: PlanContext) -> FleetPlan:
        if not context.sensitivities or context.grid.n_vars == 0:
            return _empty_plan(context, self.name, "nothing to plan")
        hints = context.request.hints
        problem = context.problem(latent=list(context.latent))
        started = time.perf_counter()
        scp = sequential_solve(
            problem,
            max_iterations=context.request.scp_iterations,
            initial_directions=(hints.initial_directions if hints else None),
        )
        elapsed = time.perf_counter() - started
        notes = list(scp.notes)
        notes.append(
            "initial directions came from the model; safety does not depend on them "
            "(Proposition 2), only the iteration count does"
        )
        x = scp.x if scp.solution.ok else np.zeros(context.grid.n_vars)
        return _assemble_plan(
            context, self.name, x, scp=scp, latent=list(context.latent),
            extra_notes=notes, solver_time_s=elapsed, problem=problem,
        )


class PignnDirectPlanner:
    """The network's output, used verbatim. Reports its own violations."""

    name = "pignn-direct"

    def plan(self, context: PlanContext) -> FleetPlan:
        hints = context.request.hints
        if hints is None or hints.predicted_dv is None:
            return _empty_plan(
                context,
                self.name,
                "no model prediction available; reporting a no-burn plan rather than "
                "silently substituting an optimizer result",
            )
        x = np.asarray(hints.predicted_dv, dtype=float).reshape(-1)
        if x.size != context.grid.n_vars:
            return _empty_plan(
                context,
                self.name,
                f"model predicted {x.size} variables but the grid has {context.grid.n_vars}",
            )
        notes = [
            "uncertified: this plan is the raw network output with no optimizer and no "
            "feasibility projection; its violations are reported, not repaired"
        ]
        return _assemble_plan(
            context, self.name, x, scp=None, latent=list(context.latent), extra_notes=notes
        )


PLANNERS: dict[str, type] = {
    NoManeuverPlanner.name: NoManeuverPlanner,
    LegacyLpPlanner.name: LegacyLpPlanner,
    GreedyPairwisePlanner.name: GreedyPairwisePlanner,
    FuelOnlyPlanner.name: FuelOnlyPlanner,
    FleetSafePlanner.name: FleetSafePlanner,
    LexicographicPlanner.name: LexicographicPlanner,
    MilpOpsPlanner.name: MilpOpsPlanner,
    PignnActiveSetPlanner.name: PignnActiveSetPlanner,
    PignnWarmStartPlanner.name: PignnWarmStartPlanner,
    PignnDirectPlanner.name: PignnDirectPlanner,
}


def get_planner(name: str):
    try:
        return PLANNERS[name]()
    except KeyError as error:
        raise ValueError(
            f"unknown planner {name!r}; available: {', '.join(sorted(PLANNERS))}"
        ) from error


def plan_with(name: str, request: PlanRequest, context: PlanContext | None = None) -> FleetPlan:
    """Run one planner, building the shared context if it was not supplied."""
    started = time.perf_counter()
    context = context if context is not None else build_context(request)
    plan = get_planner(name).plan(context)
    plan.total_time_s = time.perf_counter() - started
    return plan
