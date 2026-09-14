"""Fleet-optimization integration tests (Step 14).

Covers ``aegis/specs/step14_fleet_optimization.md`` sections 11
(``aegis.fleetopt.planners``), 13 (``aegis.scenarios``) and 16
(``aegis.experiments``), end to end but deliberately small and fast: one
resolvable two-satellite geometry stands in for a "real" scenario wherever
the property under test does not require the scenario generator itself, so
this file never depends on scipy converging on a big, randomly-coupled
program.

Writes against the public ``aegis.fleetopt``, ``aegis.scenarios``,
``aegis.experiments`` and ``aegis.ingest`` surfaces. Does not import any
private (leading-underscore) helper from those packages.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from aegis.constants import (
    DEFAULT_DV_BUDGET_KM_S,
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    REV_PER_DAY_TO_RAD_PER_S,
)
from aegis.core.conjunction import Conjunction, RiskAssessment, RiskLevel
from aegis.core.maneuver import Maneuver, ManeuverPlan, SatelliteManeuverSet
from aegis.core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from aegis.core.state import StateVector
from aegis.core.timebase import shift
from aegis.experiments import (
    BenchmarkConfig,
    collect_metrics,
    measure_induced,
    measure_linearization,
    run_benchmark,
)
from aegis.fleetopt import (
    PLANNERS,
    FleetPlan,
    LearnedHints,
    PlanRequest,
    build_context,
    get_planner,
    plan_with,
)
from aegis.fleetopt.apply import ApplyReport
from aegis.ingest import SyntheticAuthorization, SyntheticNotAuthorizedError
from aegis.risk.batch import AssessedCatalog, assess_catalog
from aegis.scenarios import FAMILIES, generate, scenario_digest
from aegis.screening import screen

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)
_FLEET = Operator(identifier="FLEET", name="Fleet", maneuverable=True)
_HAZARD = Operator(identifier="HAZARD", name="Hazard", maneuverable=False)


# ---------------------------------------------------------------------------
# Shared geometry: a small, deterministic, resolvable two-satellite scenario.
# Built directly with OrbitalElements (as test_honest_maneuver.py does)
# rather than through aegis.scenarios, so section-11 planner tests do not
# depend on the scenario generator that section 13 is separately testing.
# ---------------------------------------------------------------------------


def _mean_motion_rev_per_day(altitude_km: float = 550.0) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    mean_motion_rad_s = math.sqrt(MU_EARTH_KM3_S2 / semi_major_km**3)
    return mean_motion_rad_s / REV_PER_DAY_TO_RAD_PER_S


def _mean_motion_rad_s(altitude_km: float = 550.0) -> float:
    return _mean_motion_rev_per_day(altitude_km) * REV_PER_DAY_TO_RAD_PER_S


def _circular(
    object_id: str,
    *,
    mean_anomaly_deg: float = 0.0,
    operator: Operator,
    hard_body_radius_m: float = 5.0,
) -> SpaceObject:
    return SpaceObject(
        object_id=object_id,
        name=f"SAT-{object_id}",
        object_type=ObjectType.PAYLOAD,
        elements=OrbitalElements(
            epoch=_EPOCH,
            mean_motion_rev_per_day=_mean_motion_rev_per_day(),
            eccentricity=0.0,
            inclination_deg=53.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=mean_anomaly_deg,
        ),
        data_source="SYNTHETIC",
        operator=operator,
        hard_body_radius_m=hard_body_radius_m,
    )


def _along_track_pair(miss_km: float = 0.05) -> list[SpaceObject]:
    """Two maneuverable satellites with a close along-track conjunction."""
    phase_deg = math.degrees(miss_km / (R_EARTH_KM + 550.0))
    return [
        _circular("7001", mean_anomaly_deg=0.0, operator=_FLEET),
        _circular("7002", mean_anomaly_deg=phase_deg, operator=_FLEET),
    ]


def _resolvable_request(
    *,
    lead_orbits: float = 2.0,
    miss_km: float = 0.05,
    hints: LearnedHints | None = None,
) -> PlanRequest:
    """A tiny fleet problem with one resolvable, fully-controllable conjunction."""
    objects = _along_track_pair(miss_km)
    period_s = objects[0].elements.period_s
    start = objects[0].elements.epoch
    duration_s = 1.5 * period_s

    conjunctions = screen(objects, start, duration_s)
    assert conjunctions, "constructed along-track pair must screen"
    assessed = assess_catalog(conjunctions, objects=objects)
    assert assessed.entries

    now = shift(start, -lead_orbits * period_s)
    return PlanRequest(
        assessed=assessed,
        objects=objects,
        window_start=start,
        window_duration_s=duration_s,
        now=now,
        dv_budget_km_s=DEFAULT_DV_BUDGET_KM_S,
    )


def _empty_request() -> PlanRequest:
    """Zero-conjunction request: an empty, homogeneous-source catalog."""
    objects = _along_track_pair()
    return PlanRequest(
        assessed=AssessedCatalog(source="SYNTHETIC", entries=[]),
        objects=objects,
        window_start=objects[0].elements.epoch,
        window_duration_s=3600.0,
        now=objects[0].elements.epoch,
    )


# ---------------------------------------------------------------------------
# Section 11.2 -- every planner, on an empty scenario.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("planner_name", sorted(PLANNERS))
def test_every_planner_accepts_zero_conjunctions_without_raising(planner_name: str) -> None:
    request = _empty_request()
    plan = plan_with(planner_name, request)

    assert isinstance(plan, FleetPlan)
    assert plan.maneuver_plan.total_burns == 0
    assert plan.maneuver_plan.resolved == []
    # An empty plan is feasible by construction (there is nothing unresolved),
    # which is distinct from the "no usable burn slot" case exercised below.
    assert plan.feasible is True


@pytest.mark.parametrize("planner_name", sorted(PLANNERS))
def test_every_planner_wraps_a_real_maneuver_plan(planner_name: str) -> None:
    """FleetPlan.maneuver_plan must be a real ManeuverPlan so existing consumers work."""
    request = _resolvable_request()
    plan = plan_with(planner_name, request)

    assert isinstance(plan.maneuver_plan, ManeuverPlan)
    summary = plan.maneuver_plan.summary()
    assert isinstance(summary, dict)
    # summary() must actually reflect the underlying plan, not a stub.
    assert summary["total_burns"] == plan.maneuver_plan.total_burns


# ---------------------------------------------------------------------------
# Section 11.2 -- the "no usable burn slot" bug class.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "planner_name",
    ["fuel-only", "fleet-safe", "lexicographic", "milp-ops", "pignn-active-set", "pignn-warm-start"],
)
def test_no_usable_burn_slot_reports_unresolved_not_empty_feasible(planner_name: str) -> None:
    """A planner that cannot act must still list its conjunctions as unresolved.

    Forcing ``now`` to fall after every candidate burn slot drops every slot
    from the grid (see BurnGrid/build_burn_grid), so ``grid.n_vars == 0`` while
    the assessed conjunction itself is still real and still violated. Before
    this contract, that state was indistinguishable from "nothing to do".
    """
    request = _resolvable_request()
    # now far in the future: every burn slot (computed backwards from the
    # TCA) now falls in the past and is dropped.
    request = replace(request, now=shift(request.window_start, 50.0 * 24.0 * 3600.0))

    context = build_context(request)
    assert context.grid.n_vars == 0, "test setup must actually exclude every burn slot"
    assert context.sensitivities, "test setup must keep the conjunction itself"

    plan = get_planner(planner_name).plan(context)

    assert plan.maneuver_plan.resolved != [], (
        "an unusable plan must still list its conjunctions, not report an "
        "empty resolved list"
    )
    assert plan.feasible is False
    assert plan.unresolved_count > 0
    for outcome in plan.maneuver_plan.unresolved:
        assert outcome.shortfall_km > 0.0


# ---------------------------------------------------------------------------
# Section 11.2 -- fleet-safe's certificate.
# ---------------------------------------------------------------------------


def test_fleet_safe_certificate_linearized_safe_when_fully_resolved() -> None:
    request = _resolvable_request()
    plan = plan_with("fleet-safe", request)

    assert plan.feasible is True, "test geometry must be resolvable within budget"
    assert plan.certificate is not None
    assert plan.certificate.linearized_safe is True


# ---------------------------------------------------------------------------
# Section 11.2 -- pignn-active-set matches fleet-safe's objective exactly.
# ---------------------------------------------------------------------------


def test_pignn_active_set_matches_fleet_safe_objective_with_no_hints() -> None:
    request = _resolvable_request()
    fleet_safe = plan_with("fleet-safe", request)
    active_set = plan_with("pignn-active-set", request)

    assert fleet_safe.scp is not None and fleet_safe.scp.solution.ok
    assert active_set.scp is not None and active_set.scp.solution.ok
    assert active_set.lazy is not None
    assert active_set.lazy.closed is True

    assert active_set.scp.objective == pytest.approx(
        fleet_safe.scp.objective, abs=1e-7, rel=0.0
    )


# ---------------------------------------------------------------------------
# Section 11.2 -- pignn-warm-start is safe for garbage directions (Prop 2).
# ---------------------------------------------------------------------------


def test_pignn_warm_start_is_safe_for_garbage_initial_directions() -> None:
    baseline_context = build_context(_resolvable_request())
    conjunction_ids = [s.conjunction_id for s in baseline_context.sensitivities]
    assert conjunction_ids

    rng = np.random.default_rng(20240914)
    garbage_directions: dict[str, np.ndarray] = {}
    for conjunction_id in conjunction_ids:
        vector = rng.normal(size=3)
        garbage_directions[conjunction_id] = vector / np.linalg.norm(vector)

    # Sanity: these are not the nominal directions -- the test would be
    # vacuous if the "garbage" happened to equal the sensible default.
    nominal = {s.conjunction_id: s.nominal_direction for s in baseline_context.sensitivities}
    assert any(
        float(np.linalg.norm(garbage_directions[cid] - nominal[cid])) > 1e-3
        for cid in conjunction_ids
    )

    hints = LearnedHints(initial_directions=garbage_directions, model_id="garbage-test")
    request = replace(_resolvable_request(), hints=hints)
    plan = plan_with("pignn-warm-start", request)

    assert plan.scp is not None and plan.scp.solution.ok, "solver must still succeed"
    assert plan.certificate is not None
    assert plan.certificate.linearized_safe is True, (
        "Proposition 2: the restriction is conservative for any linearization "
        "direction, so safety must not depend on prediction quality"
    )


# ---------------------------------------------------------------------------
# Section 11.2 -- pignn-direct with no prediction.
# ---------------------------------------------------------------------------


def test_pignn_direct_with_no_prediction_is_empty_not_a_silent_optimizer() -> None:
    request = _resolvable_request()  # hints=None by default
    plan = plan_with("pignn-direct", request)

    assert plan.maneuver_plan.total_burns == 0
    assert any("no model prediction" in note for note in plan.notes)

    # Confirm it did not quietly reuse an optimizer result: fleet-safe on the
    # identical request produces real burns for this geometry.
    fleet_safe = plan_with("fleet-safe", request)
    assert fleet_safe.maneuver_plan.total_burns > 0


def test_pignn_direct_reports_its_own_violations_when_prediction_is_wrong() -> None:
    """A wrong prediction is used verbatim and its violation is surfaced, not hidden."""
    context = build_context(_resolvable_request())
    n_vars = context.grid.n_vars
    assert n_vars > 0
    # A prediction of pure zeros resolves nothing (miss unchanged), so the
    # certificate must show the conjunction as violated/unresolved rather
    # than silently reporting success.
    hints = LearnedHints(predicted_dv=np.zeros(n_vars))
    request = replace(_resolvable_request(), hints=hints)
    plan = plan_with("pignn-direct", request)

    assert plan.maneuver_plan.total_burns == 0  # zero impulse emits no maneuvers
    assert plan.certificate is not None
    assert plan.certificate.linearized_safe is False


# ---------------------------------------------------------------------------
# Section 13 -- aegis.scenarios
# ---------------------------------------------------------------------------


_SEEDS = (0, 1, 2)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_generates_at_three_seeds(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    authorization = SyntheticAuthorization(acknowledge_synthetic=True)
    for seed in _SEEDS:
        scenario = generate(family, seed, authorization=authorization)
        assert scenario.family == family
        assert scenario.seed == seed
        assert len(scenario.objects) >= 2


@pytest.mark.parametrize("family", FAMILIES)
def test_same_seed_gives_identical_digest_in_process(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    authorization = SyntheticAuthorization(acknowledge_synthetic=True)
    first = generate(family, 3, authorization=authorization)
    second = generate(family, 3, authorization=authorization)
    assert scenario_digest(first) == scenario_digest(second)


_DIGEST_SUBPROCESS_FAMILIES = ("isolated-pair", "replay-tle")


@pytest.mark.parametrize("family", _DIGEST_SUBPROCESS_FAMILIES)
def test_digest_is_identical_across_a_fresh_subprocess(family: str) -> None:
    """scenario_digest must not depend on any in-process state (hash seed, etc.)."""
    seed = 11
    script = (
        "import json, os\n"
        "os.environ['AEGIS_ALLOW_SYNTHETIC'] = '1'\n"
        "from aegis.ingest import SyntheticAuthorization\n"
        "from aegis.scenarios import generate, scenario_digest\n"
        f"scenario = generate({family!r}, {seed}, "
        "authorization=SyntheticAuthorization(acknowledge_synthetic=True))\n"
        "print(scenario_digest(scenario))\n"
    )
    env = {"PYTHONHASHSEED": "random"}
    import os

    env.update(os.environ)
    src_root = str((__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
    env["PYTHONPATH"] = src_root

    def run_once() -> str:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    digest_a = run_once()
    digest_b = run_once()
    assert digest_a == digest_b
    assert len(digest_a) == 64  # SHA-256 hex


def test_synthetic_families_refuse_without_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
    for family in FAMILIES:
        if family == "replay-tle":
            continue
        with pytest.raises(SyntheticNotAuthorizedError):
            generate(family, 0, authorization=None)


def test_synthetic_families_refuse_without_env_var_even_with_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
    authorization = SyntheticAuthorization(acknowledge_synthetic=True)
    with pytest.raises(SyntheticNotAuthorizedError):
        generate("isolated-pair", 0, authorization=authorization)


def test_synthetic_families_refuse_with_env_var_but_no_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    with pytest.raises(SyntheticNotAuthorizedError):
        generate("isolated-pair", 0, authorization=None)


def test_replay_tle_needs_no_authorization_and_no_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
    scenario = generate("replay-tle", 0, authorization=None)
    assert scenario.family == "replay-tle"
    assert len(scenario.objects) >= 2


# ---------------------------------------------------------------------------
# Section 13.2 -- structural claims per family.
# ---------------------------------------------------------------------------


def test_isolated_pair_has_exactly_one_conjunction_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    scenario = generate(
        "isolated-pair", 4, authorization=SyntheticAuthorization(acknowledge_synthetic=True)
    )
    conjunctions = screen(
        scenario.objects, scenario.window_start, scenario.window_duration_s,
        box_km=scenario.screening_box_km,
    )
    assert len(conjunctions) == 1


def test_star_hub_reaches_max_degree_at_least_three(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    scenario = generate(
        "star", 5, authorization=SyntheticAuthorization(acknowledge_synthetic=True)
    )
    conjunctions = screen(
        scenario.objects, scenario.window_start, scenario.window_duration_s,
        box_km=scenario.screening_box_km,
    )
    hub_id = scenario.objects[0].object_id
    degree = sum(
        1 for c in conjunctions if hub_id in (c.primary.object_id, c.secondary.object_id)
    )
    assert degree >= 3


def test_crossing_planes_relative_speed_above_5_km_s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    scenario = generate(
        "crossing-planes", 6, authorization=SyntheticAuthorization(acknowledge_synthetic=True)
    )
    assert scenario.provenance["relative_speed_km_s"] > 5.0


def test_intra_plane_relative_speed_below_1_km_s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    scenario = generate(
        "intra-plane", 7, authorization=SyntheticAuthorization(acknowledge_synthetic=True)
    )
    speeds = scenario.provenance["relative_speed_km_s"]
    assert speeds
    assert all(speed < 1.0 for speed in speeds)


def test_induced_cascade_produces_at_least_one_latent_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    scenario = generate(
        "induced-cascade", 8, authorization=SyntheticAuthorization(acknowledge_synthetic=True)
    )
    period_s = max(
        obj.elements.period_s for obj in scenario.objects if obj.elements is not None
    )
    padded_start = shift(scenario.window_start, -3.0 * period_s)
    padded_duration = scenario.window_duration_s + 3.0 * period_s

    conjunctions = screen(
        scenario.objects, padded_start, padded_duration, box_km=scenario.screening_box_km
    )
    assessed = assess_catalog(conjunctions, objects=scenario.objects)
    request = PlanRequest(
        assessed=assessed,
        objects=scenario.objects,
        window_start=padded_start,
        window_duration_s=padded_duration,
        now=padded_start,
        dv_budget_km_s=DEFAULT_DV_BUDGET_KM_S,
    )
    context = build_context(request)
    assert len(context.latent) >= 1


# ---------------------------------------------------------------------------
# Section 16 -- aegis.experiments
# ---------------------------------------------------------------------------


def test_benchmark_config_run_id_is_deterministic_for_fixed_config_and_stamp() -> None:
    config_a = BenchmarkConfig(suite="smoke", planners=("no-maneuver", "fuel-only"))
    config_b = BenchmarkConfig(suite="smoke", planners=("no-maneuver", "fuel-only"))
    stamp = "2024-01-01T00:00:00Z"

    assert config_a.run_id(stamp) == config_b.run_id(stamp)
    assert config_a.run_id(stamp) == config_a.run_id(stamp)

    config_c = BenchmarkConfig(suite="smoke", planners=("no-maneuver",))
    assert config_c.run_id(stamp) != config_a.run_id(stamp)

    other_stamp = "2024-02-02T00:00:00Z"
    assert config_a.run_id(stamp) != config_a.run_id(other_stamp)


def test_run_benchmark_over_two_scenarios_records_rows_and_isolates_a_raising_planner(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    monkeypatch.setenv("AEGIS_STORE_ROOT", str(tmp_path / "artifact-store"))

    config = BenchmarkConfig(
        planners=("no-maneuver", "fuel-only", "not-a-real-planner"),
        scenarios=(("isolated-pair", 21), ("isolated-pair", 22)),
        measure_induced=False,
        measure_linearization=False,
        share_linearization=False,
    )
    report = run_benchmark(
        config,
        stamp="2024-03-03T00:00:00Z",
        authorization=SyntheticAuthorization(acknowledge_synthetic=True),
        store_path=str(tmp_path / "experiments.sqlite3"),
    )

    assert report.scenario_count == 2
    # Two real planners over two scenarios; the sweep must not crash even
    # though the third planner name does not exist.
    assert report.row_count == 4
    for outcome in report.outcomes:
        assert "not-a-real-planner" in outcome.errors
        assert outcome.errors["not-a-real-planner"]  # non-empty message
        planners_recorded = {item.planner for item in outcome.metrics}
        assert planners_recorded == {"no-maneuver", "fuel-only"}


_SECTION_16_2_METRIC_KEYS = (
    "total_dv_mm_s",
    "total_propellant_g",
    "conjunctions_total",
    "resolved",
    "unresolved",
    "worst_shortfall_km",
    "induced_count_predicted",
    "induced_count_measured",
    "induced_max_pc",
    "induced_aggregate_pc",
    "station_keeping_violations",
    "maneuvering_satellites",
    "total_burns",
    "solver_time_s",
    "scp_iterations",
    "lazy_rounds",
    "rescreen_iterations",
    "feasible",
    "certificate_safe",
    "linearization_error_km_p95",
    "exact_penalty_threshold",
    "coupling_number",
    "premium",
)


def test_collect_metrics_emits_every_contract_key() -> None:
    """Section 16.2: exactly this list of keys must appear per (scenario, planner).

    NOTE: at time of writing this test documents two keys the implementation
    does not emit (see the tester's report) -- ``rescreen_iterations`` and
    ``premium`` -- and is left failing rather than adjusted to match.
    """
    request = _resolvable_request()
    context = build_context(request)
    plan = plan_with("fleet-safe", request)

    induced = measure_induced(
        request.objects,
        plan,
        request.assessed,
        request.window_start,
        request.window_duration_s,
        step_s=30.0,
        box_km=(2.0, 44.0, 51.0),
    )
    linearization = measure_linearization(request.objects, plan, context.sensitivities)

    metrics = collect_metrics(
        "scenario-1", "digest-1", "isolated-pair", plan,
        induced=induced, linearization=linearization,
    )

    missing = [key for key in _SECTION_16_2_METRIC_KEYS if key not in metrics.values]
    assert not missing, f"collect_metrics is missing contract keys: {missing}"


# ---------------------------------------------------------------------------
# Section 16 -- measure_induced's induced-vs-worsened distinction.
# ---------------------------------------------------------------------------


def _state(position_km: list[float], velocity_km_s: list[float]) -> StateVector:
    return StateVector(
        epoch=_EPOCH,
        position_km=np.asarray(position_km, dtype=float),
        velocity_km_s=np.asarray(velocity_km_s, dtype=float),
    )


def _fake_conjunction(conjunction_id: str, primary: SpaceObject, secondary: SpaceObject) -> Conjunction:
    return Conjunction(
        conjunction_id=conjunction_id,
        primary=primary,
        secondary=secondary,
        tca=_EPOCH,
        miss_distance_km=0.5,
        relative_speed_km_s=7.5,
        relative_position_rtn_km=np.array([0.5, 0.0, 0.0]),
        relative_velocity_rtn_km_s=np.array([0.0, 0.0, 7.5]),
        primary_state=_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]),
        secondary_state=_state([7000.5, 0.0, 0.0], [0.0, 7.5, 7.5]),
        screening_window_start=_EPOCH,
        screening_window_end=shift(_EPOCH, 3600.0),
    )


def _fake_assessment(conjunction_id: str, probability: float) -> RiskAssessment:
    return RiskAssessment(
        conjunction_id=conjunction_id,
        probability=probability,
        method="ALFANO-2005-GAUSSCHEBYSHEV",
        hard_body_radius_m=5.0,
        miss_distance_km=0.5,
        max_probability=1.0,
        mahalanobis_distance=1.0,
        sigma_major_km=1.0,
        sigma_minor_km=0.3,
    )


def test_measure_induced_distinguishes_induced_from_worsened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aegis.experiments.metrics as metrics_mod
    from aegis.risk.batch import RankedConjunction

    objects = [
        _circular("A1", operator=_FLEET),
        _circular("A2", operator=_FLEET),
        _circular("B1", operator=_FLEET),
        _circular("B2", operator=_FLEET),
        _circular("C1", operator=_FLEET),
        _circular("C2", operator=_FLEET),
    ]
    by_id = {obj.object_id: obj for obj in objects}

    watch = RiskLevel.WATCH
    assert RiskAssessment(
        conjunction_id="probe", probability=2e-5, method="x", hard_body_radius_m=1.0,
        miss_distance_km=1.0, max_probability=1.0, mahalanobis_distance=1.0,
        sigma_major_km=1.0, sigma_minor_km=1.0,
    ).risk_level == watch, "test assumes 2e-5 lands at/above WATCH"

    # A1-A2: already WATCH before, gets worse after -> "worsened", not induced.
    worsened_before = _fake_assessment("worsened", 2e-5)
    worsened_after = _fake_assessment("worsened", 8e-5)
    # B1-B2: below WATCH before (absent from baseline entirely), WATCH after
    # -> genuinely "induced".
    induced_after = _fake_assessment("induced", 3e-5)
    # C1-C2: WATCH before and after but unchanged -> neither induced nor worsened.
    unchanged_before = _fake_assessment("unchanged", 2e-5)
    unchanged_after = _fake_assessment("unchanged", 2e-5)

    baseline = AssessedCatalog(
        source="SYNTHETIC",
        entries=[
            RankedConjunction(
                conjunction=_fake_conjunction("worsened", by_id["A1"], by_id["A2"]),
                assessment=worsened_before, rank=1,
            ),
            RankedConjunction(
                conjunction=_fake_conjunction("unchanged", by_id["C1"], by_id["C2"]),
                assessment=unchanged_before, rank=2,
            ),
        ],
    )
    after_catalog = AssessedCatalog(
        source="SYNTHETIC",
        entries=[
            RankedConjunction(
                conjunction=_fake_conjunction("worsened", by_id["A1"], by_id["A2"]),
                assessment=worsened_after, rank=1,
            ),
            RankedConjunction(
                conjunction=_fake_conjunction("induced", by_id["B1"], by_id["B2"]),
                assessment=induced_after, rank=2,
            ),
            RankedConjunction(
                conjunction=_fake_conjunction("unchanged", by_id["C1"], by_id["C2"]),
                assessment=unchanged_after, rank=3,
            ),
        ],
    )

    monkeypatch.setattr(metrics_mod, "screen", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        metrics_mod, "assess_catalog", lambda *args, **kwargs: after_catalog
    )
    monkeypatch.setattr(
        metrics_mod,
        "apply_fleet_burns",
        lambda objects, maneuver_plan: (objects, ApplyReport()),
    )

    maneuver = Maneuver(
        satellite_id="A1", epoch=_EPOCH, delta_v_rtn_km_s=np.array([0.0, 1e-4, 0.0])
    )
    maneuver_plan = ManeuverPlan(
        plan_id="fake-plan",
        generated_at=_EPOCH,
        satellite_sets={"A1": SatelliteManeuverSet(satellite_id="A1", maneuvers=[maneuver])},
    )
    fake_plan = SimpleNamespace(maneuver_plan=maneuver_plan)

    measurement = measure_induced(
        objects, fake_plan, baseline, _EPOCH, 3600.0,
        step_s=30.0, box_km=(2.0, 44.0, 51.0),
    )

    assert measurement.measured_pairs == ["B1:B2"]
    assert measurement.measured_count == 1
    assert measurement.worsened_pairs == ["A1:A2"]
    assert measurement.worsened_count == 1
