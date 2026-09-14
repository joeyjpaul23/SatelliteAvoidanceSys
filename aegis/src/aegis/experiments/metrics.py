"""Per-(scenario, planner) measurements, and the ones that must be measured
rather than claimed.

Two numbers in here are the whole point of the project, and they are
deliberately kept apart:

``induced_count_predicted``
    What the optimizer's own latent rows say. It is the quantity the
    constraint controls, and a planner reporting only this is marking its own
    homework.

``induced_count_measured``
    What a full SGP4 re-screen of the maneuvered catalog actually finds: pairs
    at or above ``WATCH`` that were not present before the burns. This is
    ground truth, and it is what validates -- or refutes -- the constraint.

They are always reported together. The gap between them is the honest measure
of how much the linearised model can be trusted, and
``docs/model-fidelity-validation.md`` exists because of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.conjunction import RiskLevel
from ..core.objects import SpaceObject
from ..fleetopt.apply import apply_fleet_burns
from ..fleetopt.planners import FleetPlan
from ..risk.batch import AssessedCatalog, assess_catalog
from ..screening import screen

__all__ = [
    "InducedMeasurement",
    "LinearizationError",
    "PlannerMetrics",
    "pair_key",
    "measure_induced",
    "measure_linearization",
    "collect_metrics",
]


def pair_key(object_a: str, object_b: str) -> str:
    return f"{min(object_a, object_b)}:{max(object_a, object_b)}"


@dataclass
class InducedMeasurement:
    """What a full re-screen found that was not there before."""

    measured_count: int = 0
    measured_pairs: list[str] = field(default_factory=list)
    max_probability: float = 0.0
    aggregate_probability: float = 0.0
    worsened_count: int = 0
    worsened_pairs: list[str] = field(default_factory=list)
    rescreen_conjunctions: int = 0
    rescreen_iterations: int = 0
    rescreen_pairs_kept: int = 0
    apply_converged: bool = True
    apply_worst_position_error_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "induced_count_measured": self.measured_count,
            "induced_pairs_measured": list(self.measured_pairs[:20]),
            "induced_max_pc": self.max_probability,
            "induced_aggregate_pc": self.aggregate_probability,
            "worsened_count": self.worsened_count,
            "rescreen_conjunctions": self.rescreen_conjunctions,
            "rescreen_iterations": self.rescreen_iterations,
            "rescreen_maneuvered_satellites": self.rescreen_pairs_kept,
            "apply_converged": self.apply_converged,
            "apply_worst_position_error_m": round(self.apply_worst_position_error_m, 6),
            "notes": list(self.notes),
        }


@dataclass
class LinearizationError:
    """Predicted versus SGP4-measured post-maneuver miss distance."""

    samples: int = 0
    median_km: float = 0.0
    p95_km: float = 0.0
    max_km: float = 0.0
    median_relative: float = 0.0
    p95_relative: float = 0.0
    worst_conjunction: str = ""

    def as_dict(self) -> dict:
        return {
            "linearization_samples": self.samples,
            "linearization_error_km_median": round(self.median_km, 9),
            "linearization_error_km_p95": round(self.p95_km, 9),
            "linearization_error_km_max": round(self.max_km, 9),
            "linearization_relative_median": round(self.median_relative, 9),
            "linearization_relative_p95": round(self.p95_relative, 9),
            "linearization_worst_conjunction": self.worst_conjunction,
        }


def _watch_pairs(assessed: AssessedCatalog, level: str = RiskLevel.WATCH) -> dict[str, float]:
    """Pair key -> highest probability, for every pair at or above ``level``."""
    found: dict[str, float] = {}
    for entry in assessed.above(level):
        key = pair_key(
            entry.conjunction.primary.object_id, entry.conjunction.secondary.object_id
        )
        found[key] = max(found.get(key, 0.0), float(entry.assessment.probability))
    return found


def measure_induced(
    objects: list[SpaceObject],
    plan: FleetPlan,
    baseline: AssessedCatalog,
    window_start: datetime,
    window_duration_s: float,
    *,
    step_s: float,
    box_km: tuple[float, float, float],
    level: str = RiskLevel.WATCH,
) -> InducedMeasurement:
    """Apply the plan, re-screen with SGP4, and count what is new.

    "New" means a pair at or above ``level`` that was not at or above
    ``level`` before. A pair that was already above threshold and got *worse*
    is counted separately as ``worsened`` rather than as induced -- it is a
    different failure, and conflating the two would let a planner hide a
    doubled probability inside an unchanged count.
    """
    measurement = InducedMeasurement()
    if plan.maneuver_plan.total_burns == 0:
        measurement.notes.append("no burns commanded; nothing to re-screen")
        measurement.rescreen_conjunctions = len(baseline.entries)
        return measurement

    maneuvered, report = apply_fleet_burns(objects, plan.maneuver_plan)
    measurement.apply_converged = report.converged
    measurement.apply_worst_position_error_m = report.worst_position_error_km * 1000.0
    measurement.notes.extend(report.notes[:5])

    # Only the maneuvered satellites moved, so a pair with no maneuvered member
    # has geometry identical to the baseline and cannot have *become* a
    # conjunction. Restricting the re-screen to pairs touching a burn is
    # therefore exact for the induced count, not an approximation -- and it is
    # the difference between re-screening 325 pairs and 115 of them in a
    # 26-object catalog where five satellites maneuver. Screening is 99 % of
    # this pipeline's runtime, so this is where the time actually is.
    moved = {
        sat_id
        for sat_id, sat_set in plan.maneuver_plan.satellite_sets.items()
        if sat_set.burn_count
    }
    measurement.rescreen_pairs_kept = len(moved)

    def touches_a_burn(object_a: SpaceObject, object_b: SpaceObject) -> bool:
        return object_a.object_id in moved or object_b.object_id in moved

    conjunctions = screen(
        maneuvered,
        window_start,
        window_duration_s,
        step_s=step_s,
        box_km=box_km,
        keep_pair=touches_a_burn if moved else None,
    )
    assessed = assess_catalog(conjunctions, objects=maneuvered)
    measurement.rescreen_conjunctions = len(assessed.entries)
    # One re-screen, by design. The legacy loop in aegis.maneuver.rescreen
    # iterates plan/apply/re-screen until the WATCH pair set stops growing;
    # this planner puts the induced-conjunction condition inside the program
    # instead, so a single re-screen is a measurement rather than a step of
    # a fixed-point search. The count is reported so the two can be compared.
    measurement.rescreen_iterations = 1

    before = _watch_pairs(baseline, level)
    after = _watch_pairs(assessed, level)

    # No filtering of the difference is needed or wanted. The `keep_pair`
    # restriction on the re-screen above already guarantees that `after`
    # contains only pairs touching a burn, so `after - before` is restricted
    # automatically. Filtering `before` instead makes every untouched baseline
    # pair look like it disappeared, and filtering the difference as well
    # discards genuinely-induced pairs whenever the caller supplies an
    # unrestricted assessment -- a contract test caught both mistakes.
    induced = sorted(set(after) - set(before))
    measurement.measured_pairs = induced
    measurement.measured_count = len(induced)
    if induced:
        probabilities = [after[key] for key in induced]
        measurement.max_probability = float(max(probabilities))
        measurement.aggregate_probability = float(
            1.0 - np.prod([1.0 - min(p, 1.0) for p in probabilities])
        )

    worsened = sorted(
        key for key in set(after) & set(before) if after[key] > before[key] * 1.0000001
    )
    measurement.worsened_pairs = worsened
    measurement.worsened_count = len(worsened)
    return measurement


def _true_miss_km(
    maneuvered: list[SpaceObject],
    primary_id: str,
    secondary_id: str,
    tca: datetime,
    *,
    half_window_s: float = 900.0,
    samples: int = 601,
) -> float | None:
    """Minimum separation near ``tca`` from SGP4, by direct sampling.

    Sampling rather than reusing ``refine_tca`` because the maneuvered pair may
    no longer enter any screening box, and the question here is the separation
    itself, not whether it qualifies as a conjunction.
    """
    from ..core.timebase import shift
    from ..propagation.propagator import PropagationError, Sgp4Propagator

    try:
        propagator = Sgp4Propagator(maneuvered)
    except PropagationError:
        return None
    try:
        index_a = propagator.object_ids.index(primary_id)
        index_b = propagator.object_ids.index(secondary_id)
    except ValueError:
        return None

    best = float("inf")
    for offset in np.linspace(-half_window_s, half_window_s, samples):
        try:
            state_a = propagator.propagate_one(index_a, shift(tca, float(offset)))
            state_b = propagator.propagate_one(index_b, shift(tca, float(offset)))
        except PropagationError:
            continue
        separation = float(
            np.linalg.norm(
                np.asarray(state_b.position_km) - np.asarray(state_a.position_km)
            )
        )
        best = min(best, separation)
    return None if not np.isfinite(best) else best


def measure_linearization(
    objects: list[SpaceObject],
    plan: FleetPlan,
    context_sensitivities,
    *,
    max_conjunctions: int = 12,
) -> LinearizationError:
    """Compare each predicted post-maneuver miss against SGP4 ground truth.

    This is the number every delta-v claim is reported next to. Without it,
    "the optimizer resolved the conjunction" means only "the optimizer's own
    linear model says so".
    """
    error = LinearizationError()
    if plan.maneuver_plan.total_burns == 0 or not context_sensitivities:
        return error

    maneuvered, _ = apply_fleet_burns(objects, plan.maneuver_plan)
    ranked = sorted(context_sensitivities, key=lambda s: s.nominal_miss_km)[:max_conjunctions]

    absolute: list[float] = []
    relative: list[float] = []
    worst_name = ""
    worst_value = -1.0
    for sensitivity in ranked:
        predicted = sensitivity.miss_after_km(plan.x)
        truth = _true_miss_km(
            maneuvered, sensitivity.primary_id, sensitivity.secondary_id, sensitivity.tca
        )
        if truth is None:
            continue
        gap = abs(predicted - truth)
        absolute.append(gap)
        relative.append(gap / max(truth, 1e-9))
        if gap > worst_value:
            worst_value = gap
            worst_name = sensitivity.conjunction_id

    if not absolute:
        return error
    values = np.asarray(absolute, dtype=float)
    ratios = np.asarray(relative, dtype=float)
    error.samples = int(values.size)
    error.median_km = float(np.median(values))
    error.p95_km = float(np.percentile(values, 95))
    error.max_km = float(np.max(values))
    error.median_relative = float(np.median(ratios))
    error.p95_relative = float(np.percentile(ratios, 95))
    error.worst_conjunction = worst_name
    return error


@dataclass
class PlannerMetrics:
    """Everything recorded for one (scenario, planner) cell."""

    scenario_id: str
    scenario_digest: str
    family: str
    planner: str
    values: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        base = {
            "scenario_id": self.scenario_id,
            "scenario_digest": self.scenario_digest,
            "family": self.family,
            "planner": self.planner,
        }
        base.update(self.values)
        return base


def collect_metrics(
    scenario_id: str,
    scenario_digest: str,
    family: str,
    plan: FleetPlan,
    *,
    induced: InducedMeasurement | None = None,
    linearization: LinearizationError | None = None,
) -> PlannerMetrics:
    """Assemble the metric row the contract's section 16.2 asks for."""
    maneuver_plan = plan.maneuver_plan
    unresolved = maneuver_plan.unresolved
    values: dict = {
        "total_dv_mm_s": round(maneuver_plan.total_delta_v_mm_s, 6),
        "total_propellant_g": round(maneuver_plan.total_propellant_g, 6),
        "conjunctions_total": len(maneuver_plan.resolved),
        "resolved": len(maneuver_plan.resolved) - len(unresolved),
        "unresolved": len(unresolved),
        "worst_shortfall_km": round(
            max((item.shortfall_km for item in unresolved), default=0.0), 6
        ),
        "maneuvering_satellites": maneuver_plan.maneuvering_satellite_count,
        "total_burns": maneuver_plan.total_burns,
        "solver_time_s": round(plan.solver_time_s, 6),
        "total_time_s": round(plan.total_time_s, 6),
        "feasible": plan.feasible,
        "certificate_safe": plan.certified_safe,
        "exact_penalty_threshold": plan.exact_penalty_threshold,
        "scp_iterations": plan.scp.iterations if plan.scp else 0,
        "scp_monotone": plan.scp.monotone if plan.scp else True,
        "scp_converged": plan.scp.converged if plan.scp else False,
        "lazy_rounds": plan.lazy.rounds if plan.lazy else 0,
        "lazy_rows_used": plan.lazy.rows_used if plan.lazy else 0,
        "lazy_rows_total": plan.lazy.rows_total if plan.lazy else 0,
        "lazy_closed": plan.lazy.closed if plan.lazy else None,
        "induced_count_predicted": plan.induced.predicted_count if plan.induced else 0,
        "induced_predicted_worst_shortfall_km": (
            round(plan.induced.predicted_worst_shortfall_km, 6) if plan.induced else 0.0
        ),
        "latent_rows": plan.induced.latent_rows if plan.induced else 0,
        "station_keeping_violations": (
            len(plan.certificate.station_keeping_violations) if plan.certificate else 0
        ),
        "certificate_violations": (
            plan.certificate.violation_count if plan.certificate else 0
        ),
        "farkas_size": plan.farkas.size if plan.farkas else 0,
        "farkas_explanation": plan.farkas.explanation if plan.farkas else "",
        "penetration_index": plan.penetration.value if plan.penetration else None,
        "predicts_zero_premium": (
            plan.penetration.predicts_zero_premium if plan.penetration else None
        ),
        # Populated by the benchmark runner for the planners it compares.
        # The premium is a property of a PAIR of plans, not of one plan, so a
        # per-planner row carries it only as a convenience for joins and is
        # None wherever the pair was not formed.
        "premium": None,
        "rescreen_iterations": 0,
    }
    if plan.graph is not None:
        graph = plan.graph.as_dict()
        values.update(
            {
                "graph_edges": graph["edges"],
                "graph_max_degree": graph["max_degree"],
                "graph_components": graph["component_count"],
                "graph_largest_component": graph["largest_component"],
                "graph_tca_overlap": graph["tca_overlap_fraction"],
                "graph_intra_fleet_fraction": graph["intra_fleet_fraction"],
                "coupling_number": graph["coupling_number"],
                "conflict_dimension": graph["conflict_dimension"],
                "controllable_edges": graph["controllable_edges"],
            }
        )
    if induced is not None:
        values.update(induced.as_dict())
    if linearization is not None:
        values.update(linearization.as_dict())

    return PlannerMetrics(
        scenario_id=scenario_id,
        scenario_digest=scenario_digest,
        family=family,
        planner=plan.planner,
        values=values,
    )
