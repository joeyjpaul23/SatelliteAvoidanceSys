"""The benchmark sweep: every planner, every scenario, one comparable table.

Design rules, each of which exists because of a way a sweep can lie:

* **One context per scenario.** Screening, assessment, sensitivity assembly
  and latent enumeration happen once and every planner sees the identical
  inputs. Rebuilding per planner would let a difference in the *problem* show
  up as a difference in the *planner*.
* **Ground truth is measured, not inferred.** Induced conjunctions come from a
  full SGP4 re-screen of the maneuvered catalog, not from the optimizer's own
  constraint residuals, and both numbers are recorded.
* **Failures are recorded, not swallowed.** A planner that raises, or exceeds
  its time budget, produces a row saying so. A sweep that quietly drops its
  hard cases reports a flattering distribution.
* **Resumable and idempotent.** Results key on
  ``(run_id, scenario_digest, planner)``, so an interrupted overnight sweep
  resumes without double-counting.
* **Deterministic run ids.** The id is a digest of the configuration plus an
  explicit timestamp argument, never a wall-clock call, so a rerun of the same
  configuration lands in the same place on purpose rather than by accident.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from ..constants import (
    DEFAULT_BURN_SLOTS,
    DEFAULT_DV_BUDGET_KM_S,
    MIN_LEAD_TIME_ORBITS,
    PC_TARGET_POST_MANEUVER,
    SCREENING_STEP_S,
)
from ..core.conjunction import RiskLevel
from ..fleetopt.latent import LATENT_DEFAULT_MODE
from ..fleetopt.pareto import (
    PremiumRecord,
    fixed_linearization_premium,
    premium_statistics,
    safety_premium,
)
from ..fleetopt.planners import (
    PlanRequest,
    build_context,
    get_planner,
)
from ..ingest.synthetic import SyntheticAuthorization
from ..risk.batch import assess_catalog
from ..scenarios import Scenario, expand_suite, generate, scenario_digest
from ..screening import screen
from ..store import ExperimentStore, StoreConfig, open_artifact_store
from .metrics import PlannerMetrics, collect_metrics, measure_induced, measure_linearization

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "ScenarioOutcome",
    "run_benchmark",
    "run_scenario",
]

#: Planners run by default. ``pignn-*`` are excluded unless a checkpoint is
#: supplied, because without one they would silently duplicate ``fleet-safe``
#: and inflate the table with three identical columns.
DEFAULT_PLANNERS = (
    "no-maneuver",
    "legacy-lp",
    "greedy-pairwise",
    "fuel-only",
    "fleet-safe",
    "lexicographic",
    "milp-ops",
)

LEARNED_PLANNERS = ("pignn-active-set", "pignn-warm-start", "pignn-direct")


@dataclass
class BenchmarkConfig:
    """Everything that defines a sweep, and therefore its run id."""

    suite: str = "smoke"
    planners: tuple[str, ...] = DEFAULT_PLANNERS
    scenarios: tuple[tuple[str, int], ...] | None = None
    target_pc: float = PC_TARGET_POST_MANEUVER
    induced_exclusion_km: float | None = None
    dv_budget_km_s: float = DEFAULT_DV_BUDGET_KM_S
    burn_slots: int = DEFAULT_BURN_SLOTS
    min_lead_orbits: float = MIN_LEAD_TIME_ORBITS
    axes: int = 3
    cost_model_name: str = "l1"
    latent_mode: str = LATENT_DEFAULT_MODE
    latent_step_s: float = 30.0
    max_latent_rows: int | None = 2000
    scp_iterations: int = 6
    screening_step_s: float = SCREENING_STEP_S
    ops_cost_per_satellite: float = 0.0
    ops_cost_per_burn: float = 0.0
    plan_risk_level: str = RiskLevel.MONITOR
    max_resolve_rows: int | None = 400
    lead_pad_orbits: float = 3.0
    share_linearization: bool = True
    measure_induced: bool = True
    measure_linearization: bool = True
    per_planner_timeout_s: float = 600.0
    induced_level: str = RiskLevel.WATCH
    model_checkpoint: str | None = None
    hint_threshold: float | None = None
    notes: str = ""

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def run_id(self, stamp: str) -> str:
        return f"run-{stamp}-{self.digest()}"


@dataclass
class ScenarioOutcome:
    """Everything produced for one scenario across every planner."""

    scenario: Scenario
    digest: str
    conjunctions: int
    metrics: list[PlannerMetrics] = field(default_factory=list)
    premium: PremiumRecord | None = None
    errors: dict[str, str] = field(default_factory=dict)
    build_time_s: float = 0.0

    def by_planner(self) -> dict[str, PlannerMetrics]:
        return {item.planner: item for item in self.metrics}


@dataclass
class BenchmarkReport:
    """The sweep's results, ready to serialise or render."""

    run_id: str
    config: BenchmarkConfig
    started_at: str
    finished_at: str = ""
    outcomes: list[ScenarioOutcome] = field(default_factory=list)
    premiums: list[PremiumRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def scenario_count(self) -> int:
        return len(self.outcomes)

    @property
    def row_count(self) -> int:
        return sum(len(outcome.metrics) for outcome in self.outcomes)

    @property
    def error_count(self) -> int:
        return sum(len(outcome.errors) for outcome in self.outcomes)

    def premium_summary(self) -> dict:
        return premium_statistics(self.premiums)

    def planner_table(self) -> dict[str, dict]:
        """Aggregate per planner, over every scenario it produced a row for."""
        import numpy as np

        grouped: dict[str, list[PlannerMetrics]] = {}
        for outcome in self.outcomes:
            for item in outcome.metrics:
                grouped.setdefault(item.planner, []).append(item)

        table: dict[str, dict] = {}
        for planner, rows in sorted(grouped.items()):
            def column(name: str) -> list[float]:
                return [
                    float(r.values[name])
                    for r in rows
                    if isinstance(r.values.get(name), (int, float))
                ]

            dv = column("total_dv_mm_s")
            induced_predicted = column("induced_count_predicted")
            induced_measured = column("induced_count_measured")
            solver = column("solver_time_s")
            unresolved = column("unresolved")
            table[planner] = {
                "scenarios": len(rows),
                "median_dv_mm_s": float(np.median(dv)) if dv else 0.0,
                "mean_dv_mm_s": float(np.mean(dv)) if dv else 0.0,
                "total_dv_mm_s": float(np.sum(dv)) if dv else 0.0,
                "induced_predicted_total": int(sum(induced_predicted)),
                "induced_measured_total": int(sum(induced_measured)),
                "scenarios_with_induced_measured": int(
                    sum(1 for value in induced_measured if value > 0)
                ),
                "feasible_fraction": float(
                    np.mean([1.0 if r.values.get("feasible") else 0.0 for r in rows])
                ),
                "certified_fraction": float(
                    np.mean([1.0 if r.values.get("certificate_safe") else 0.0 for r in rows])
                ),
                "unresolved_total": int(sum(unresolved)),
                "median_solver_time_s": float(np.median(solver)) if solver else 0.0,
            }
        return table

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scenario_count": self.scenario_count,
            "row_count": self.row_count,
            "error_count": self.error_count,
            "premium_summary": self.premium_summary(),
            "planner_table": self.planner_table(),
            "rows": [item.as_dict() for outcome in self.outcomes for item in outcome.metrics],
            "premiums": [record.as_dict() for record in self.premiums],
            "errors": [
                {"scenario": outcome.scenario.scenario_id, "planner": planner, "error": message}
                for outcome in self.outcomes
                for planner, message in outcome.errors.items()
            ],
            "notes": list(self.notes),
        }


def _hints_for(config: BenchmarkConfig):
    """Load the hints provider if a checkpoint was supplied.

    Returns ``None`` -- and the report says so -- rather than fabricating
    hints. A ``pignn-*`` planner with no hints is honest: it falls back to the
    exact solve and records that it did. A planner with invented hints is not.

    The provider is a callable, not a fixed set of hints, because the
    prediction depends on the scenario's own graph.
    """
    if not config.model_checkpoint:
        return None
    try:
        from ..ml.infer import load_hints_provider
    except ImportError:
        return None
    return load_hints_provider(config.model_checkpoint, threshold=config.hint_threshold)


def _lead_padded_window(
    scenario: Scenario, lead_pad_orbits: float
) -> tuple[datetime, float]:
    """Start the planning window early enough that burns have somewhere to go.

    A scenario generator places a conjunction inside the window it designed;
    the *planner* needs that conjunction to sit at least ``min_lead_orbits``
    plus a few half-orbits after the planning epoch, or every candidate burn
    slot falls in the past and the satellite is excluded. Measured on the
    first sweep, the generated families put their TCAs 0.01 to 1.01 orbits
    after their own window start, so every solving planner returned an empty
    grid and the whole comparison was vacuous.

    Padding the window backwards -- rather than moving the conjunction -- keeps
    the scenario's geometry exactly as generated and exactly as its structural
    claims describe it. SGP4 propagates backwards from the element epoch
    without complaint.
    """
    from ..core.timebase import shift

    periods = [
        obj.elements.period_s
        for obj in scenario.objects
        if obj.elements is not None and obj.elements.mean_motion_rev_per_day > 0.0
    ]
    if not periods or lead_pad_orbits <= 0.0:
        return scenario.window_start, scenario.window_duration_s
    pad_s = float(lead_pad_orbits) * float(np.median(periods))
    return shift(scenario.window_start, -pad_s), scenario.window_duration_s + pad_s


def run_scenario(
    scenario: Scenario,
    config: BenchmarkConfig,
    *,
    hints_provider=None,
) -> ScenarioOutcome:
    """Run every configured planner against one scenario."""
    started = time.perf_counter()
    window_start, window_duration_s = _lead_padded_window(scenario, config.lead_pad_orbits)
    conjunctions = screen(
        scenario.objects,
        window_start,
        window_duration_s,
        step_s=config.screening_step_s,
        box_km=scenario.screening_box_km,
    )
    assessed = assess_catalog(conjunctions, objects=scenario.objects)

    request = PlanRequest(
        assessed=assessed,
        objects=scenario.objects,
        window_start=window_start,
        window_duration_s=window_duration_s,
        now=window_start,
        target_pc=config.target_pc,
        plan_risk_level=config.plan_risk_level,
        max_resolve_rows=config.max_resolve_rows,
        induced_exclusion_km=config.induced_exclusion_km,
        dv_budget_km_s=config.dv_budget_km_s,
        burn_slots=config.burn_slots,
        min_lead_orbits=config.min_lead_orbits,
        axes=config.axes,
        cost_model_name=config.cost_model_name,
        latent_mode=config.latent_mode,
        latent_step_s=config.latent_step_s,
        max_latent_rows=config.max_latent_rows,
        scp_iterations=config.scp_iterations,
        ops_cost_per_satellite=config.ops_cost_per_satellite,
        ops_cost_per_burn=config.ops_cost_per_burn,
        screening_box_km=scenario.screening_box_km,
    )
    context = build_context(request)
    if hints_provider is not None:
        try:
            request.hints = hints_provider(context, scenario)
        except Exception as error:  # noqa: BLE001 - a hint must never break a sweep
            outcome_note = f"{type(error).__name__}: {error}"
            request.hints = None
    digest = scenario_digest(scenario)
    outcome = ScenarioOutcome(
        scenario=scenario,
        digest=digest,
        conjunctions=len(assessed.entries),
        build_time_s=time.perf_counter() - started,
    )

    # Converge the reference linearization once, then hand it to every
    # solving planner, so differences in reported delta-v come from the
    # formulations rather than from the sequential refinement wandering into
    # different basins. See FleetSafePlanner's docstring for the measurement
    # that made this necessary.
    if config.share_linearization and context.sensitivities and context.grid.n_vars:
        try:
            reference = get_planner("fuel-only").plan(context)
            if reference.scp is not None and reference.scp.solution.ok:
                context.shared_directions = dict(reference.scp.directions)
        except Exception as error:  # noqa: BLE001 - a warm start is an optimisation
            outcome_note = f"shared linearization unavailable: {type(error).__name__}: {error}"
            outcome.errors["shared-linearization"] = outcome_note

    plans: dict[str, object] = {}
    for name in config.planners:
        planner_started = time.perf_counter()
        try:
            plan = get_planner(name).plan(context)
            plan.total_time_s = time.perf_counter() - planner_started
        except Exception as error:  # noqa: BLE001 - one planner must not kill the sweep
            outcome.errors[name] = f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=4)}"
            continue

        elapsed = time.perf_counter() - planner_started
        if elapsed > config.per_planner_timeout_s:
            outcome.errors[name] = (
                f"exceeded the {config.per_planner_timeout_s:.0f}s budget "
                f"({elapsed:.1f}s); the result is still recorded"
            )

        induced = None
        if config.measure_induced:
            try:
                induced = measure_induced(
                    scenario.objects,
                    plan,
                    assessed,
                    window_start,
                    window_duration_s,
                    step_s=config.screening_step_s,
                    box_km=scenario.screening_box_km,
                    level=config.induced_level,
                )
            except Exception as error:  # noqa: BLE001 - measurement failure is data
                outcome.errors[f"{name}:induced"] = f"{type(error).__name__}: {error}"

        linearization = None
        if config.measure_linearization:
            try:
                linearization = measure_linearization(
                    scenario.objects, plan, context.sensitivities
                )
            except Exception as error:  # noqa: BLE001
                outcome.errors[f"{name}:linearization"] = f"{type(error).__name__}: {error}"

        plans[name] = plan
        outcome.metrics.append(
            collect_metrics(
                scenario.scenario_id,
                digest,
                scenario.family,
                plan,
                induced=induced,
                linearization=linearization,
            )
        )

    fuel = plans.get("fuel-only")
    safe = plans.get("fleet-safe")
    if fuel is not None and safe is not None:
        fixed = None
        if context.shared_directions and context.grid.n_vars:
            try:
                fixed = fixed_linearization_premium(
                    context.problem(latent=[]),
                    context.problem(latent=list(context.latent)),
                    context.shared_directions,
                )
            except Exception as error:  # noqa: BLE001 - a bound is a bonus, not a blocker
                outcome.errors["fixed-premium"] = f"{type(error).__name__}: {error}"
        by_planner = outcome.by_planner()
        fuel_row = by_planner.get("fuel-only")
        safe_row = by_planner.get("fleet-safe")
        outcome.premium = safety_premium(
            scenario.scenario_id,
            delta_v_fuel_km_s=fuel.total_delta_v_km_s,
            delta_v_safe_km_s=safe.total_delta_v_km_s,
            induced_fuel_only=int(fuel_row.values.get("induced_count_measured", 0) or 0)
            if fuel_row
            else 0,
            induced_fleet_safe=int(safe_row.values.get("induced_count_measured", 0) or 0)
            if safe_row
            else 0,
            resolved_fuel_only=fuel.resolved_count,
            resolved_fleet_safe=safe.resolved_count,
            total_conjunctions=len(assessed.entries),
            fuel_only_feasible=fuel.feasible,
            fleet_safe_feasible=safe.feasible,
            coupling_number=context.graph.coupling_number,
            conflict_dimension=context.graph.conflict_dimension,
            family=scenario.family,
            fixed=fixed,
        )
        # Stamp the premium onto the two rows it was computed from, so a
        # per-planner join can find it without re-deriving the pair.
        for planner_name in ("fuel-only", "fleet-safe"):
            row = outcome.by_planner().get(planner_name)
            if row is not None:
                row.values["premium"] = outcome.premium.premium
    return outcome


def run_benchmark(
    config: BenchmarkConfig,
    *,
    stamp: str,
    authorization: SyntheticAuthorization | None = None,
    store_path: str | None = None,
    artifact_store=None,
    progress=None,
) -> BenchmarkReport:
    """Run the whole sweep, recording as it goes so it can be resumed."""
    run_id = config.run_id(stamp)
    report = BenchmarkReport(run_id=run_id, config=config, started_at=stamp)

    if config.scenarios is not None:
        scenarios = [
            generate(family, seed, authorization=authorization)
            for family, seed in config.scenarios
        ]
    else:
        scenarios = expand_suite(config.suite, authorization=authorization)

    hints = _hints_for(config)
    if config.model_checkpoint and hints is None:
        report.notes.append(
            f"model checkpoint {config.model_checkpoint!r} produced no hints; "
            "learned planners will fall back to the exact solve and are reported as such"
        )

    settings = StoreConfig.from_env()
    database = store_path or settings.experiment_db
    store: ExperimentStore | None = None
    try:
        store = ExperimentStore.open(database)
        store.record_run(
            run_id=run_id,
            config=asdict(config),
            notes=config.notes or None,
        )
    except Exception as error:  # noqa: BLE001 - the sweep is worth more than the log
        report.notes.append(f"experiment store unavailable ({error}); results are in-memory only")
        store = None

    try:
        for index, scenario in enumerate(scenarios, start=1):
            if progress is not None:
                progress(index, len(scenarios), scenario.scenario_id)
            outcome = run_scenario(scenario, config, hints_provider=hints)
            report.outcomes.append(outcome)
            if outcome.premium is not None:
                report.premiums.append(outcome.premium)

            if store is None:
                continue
            store.record_scenario(
                scenario_digest=outcome.digest,
                family=scenario.family,
                seed=scenario.seed,
                n_objects=len(scenario.objects),
                n_conjunctions=outcome.conjunctions,
                spec=dict(scenario.provenance),
            )
            for item in outcome.metrics:
                store.record_result(
                    run_id=run_id,
                    scenario_digest=outcome.digest,
                    planner=item.planner,
                    metrics=item.values,
                    total_dv_mm_s=float(item.values.get("total_dv_mm_s", 0.0)),
                    induced_count=int(item.values.get("induced_count_measured", 0) or 0),
                    resolved=int(item.values.get("resolved", 0)),
                    unresolved=int(item.values.get("unresolved", 0)),
                    feasible=bool(item.values.get("feasible", False)),
                    solver_time_s=float(item.values.get("solver_time_s", 0.0)),
                )
            if outcome.premium is not None:
                store.record_premium(
                    result_id=f"{run_id}:{outcome.digest}",
                    run_id=run_id,
                    scenario_digest=outcome.digest,
                    premium=outcome.premium.premium,
                    dv_fuel_mm_s=outcome.premium.delta_v_fuel_mm_s,
                    dv_safe_mm_s=outcome.premium.delta_v_safe_mm_s,
                    infeasible=outcome.premium.infeasible,
                )
    finally:
        if store is not None:
            store.close()

    report.finished_at = stamp
    if artifact_store is None:
        try:
            artifact_store = open_artifact_store(settings)
        except Exception as error:  # noqa: BLE001
            report.notes.append(f"artifact store unavailable ({error}); no artifacts written")
            artifact_store = None
    if artifact_store is not None:
        payload = json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str)
        try:
            artifact_store.put(
                f"benchmarks/{run_id}/report.json",
                payload.encode("utf-8"),
                content_type="application/json",
            )
        except Exception as error:  # noqa: BLE001
            report.notes.append(f"report artifact not written ({error})")

    return report
