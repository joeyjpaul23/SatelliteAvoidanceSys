"""Certified fleet-wide collision-avoidance maneuver optimization.

The research layer on top of AEGIS's screening and risk engine. Where
:mod:`aegis.maneuver` plans along-track burns for a fleet with a scalar
sensitivity model, this package solves the coordinated problem properly:
three-axis impulses, B-plane miss sensitivities, explicit
induced-conjunction constraints, a reachability bound that certifies the
candidate set, and a measured safety premium.

Start at :func:`aegis.fleetopt.planners.plan_with`. The contract is
``aegis/specs/step14_fleet_optimization.md``; the claim ledger and prior art
are in ``docs/prior-art-and-novelty-ledger.md``; the measured model fidelity
is in ``docs/model-fidelity-validation.md``.
"""

from __future__ import annotations

from .apply import ApplyReport, apply_fleet_burns, apply_impulse, fit_mean_elements
from .bplane import (
    MissSensitivity,
    bplane_projector,
    build_miss_sensitivity,
    linearize,
    refine_direction,
)
from .certify import (
    Certificate,
    Farkas,
    binding_rows,
    exact_penalty_threshold,
    irreconcilable_subset,
    verify_plan,
)
from .dynamics import (
    BurnGrid,
    build_burn_grid,
    cw_impulse_matrix,
    cw_impulse_norm,
    cw_impulse_norm_bound,
    displacement_operator,
    eci_displacement_operator,
)
from .errors import (
    AssemblyError,
    CertificationError,
    FleetOptError,
    GridError,
    SolverBackendError,
)
from .graph import (
    ConjunctionGraph,
    GraphMetrics,
    build_conjunction_graph,
    conflict_dimension,
    coupling_number,
    graph_metrics,
)
from .latent import (
    LatentConstraint,
    LatentEnumerationReport,
    enumerate_latent_constraints,
    grid_inflation_km,
    pair_id,
)
from .norms import (
    CostModel,
    cone_cost_model,
    covering_radius_deg,
    l1_cost_model,
    polyhedral_cone,
)
from .pareto import (
    Frontier,
    FrontierPoint,
    PenetrationIndex,
    PremiumRecord,
    penetration_index,
    row_reach_km,
    induced_frontier,
    premium_statistics,
    safety_premium,
)
from .planners import (
    PLANNERS,
    FleetPlan,
    InducedReport,
    LearnedHints,
    PlanContext,
    PlanRequest,
    build_context,
    get_planner,
    plan_with,
)
from .problem import FleetProblem, LinearProgramData, assemble
from .reachability import (
    ReachabilityModel,
    a_posteriori_gate_km,
    asymptotic_gate_km,
    certified_gate_km,
    reach_radius_km,
)
from .solver import (
    LazyResult,
    LpSolution,
    ScpResult,
    lazy_solve,
    sequential_solve,
    solve_lp,
    solve_milp,
    solve_problem,
)

__all__ = [
    "ApplyReport", "apply_fleet_burns", "apply_impulse", "fit_mean_elements",
    "MissSensitivity", "bplane_projector", "build_miss_sensitivity", "linearize",
    "refine_direction",
    "Certificate", "Farkas", "binding_rows", "exact_penalty_threshold",
    "irreconcilable_subset", "verify_plan",
    "BurnGrid", "build_burn_grid", "cw_impulse_matrix", "cw_impulse_norm",
    "cw_impulse_norm_bound", "displacement_operator", "eci_displacement_operator",
    "AssemblyError", "CertificationError", "FleetOptError", "GridError",
    "SolverBackendError",
    "ConjunctionGraph", "GraphMetrics", "build_conjunction_graph",
    "conflict_dimension", "coupling_number", "graph_metrics",
    "LatentConstraint", "LatentEnumerationReport", "enumerate_latent_constraints",
    "grid_inflation_km", "pair_id",
    "CostModel", "cone_cost_model", "covering_radius_deg", "l1_cost_model",
    "polyhedral_cone",
    "Frontier", "FrontierPoint", "PenetrationIndex", "PremiumRecord",
    "induced_frontier", "penetration_index", "row_reach_km",
    "premium_statistics", "safety_premium",
    "PLANNERS", "FleetPlan", "InducedReport", "LearnedHints", "PlanContext",
    "PlanRequest", "build_context", "get_planner", "plan_with",
    "FleetProblem", "LinearProgramData", "assemble",
    "ReachabilityModel", "a_posteriori_gate_km", "asymptotic_gate_km",
    "certified_gate_km", "reach_radius_km",
    "LazyResult", "LpSolution", "ScpResult", "lazy_solve", "sequential_solve",
    "solve_lp", "solve_milp", "solve_problem",
]
