"""Solver and certification tests (Step 14 contract, sections 7, 8 and 9).

``aegis/specs/step14_fleet_optimization.md``:

* Section 7, ``aegis.fleetopt.problem`` -- ``assemble`` is a pure function of
  its inputs, every row and column carries a stable label matching the
  documented grammar, and slack columns are mandatory for every resolve and
  latent row so an unresolvable geometry returns a plan rather than raising.
* Section 8, ``aegis.fleetopt.solver`` -- the LP/MILP backends, the certified
  sequential refinement (Propositions 2 and 3), and the exactness guarantee
  of the lazy-constraint loop.
* Section 9, ``aegis.fleetopt.certify`` -- independent re-verification of a
  plan, the exact-penalty threshold (Proposition 6), and the minimal
  infeasible subsystem returned by ``irreconcilable_subset``.

Problems are built directly from ``BurnGrid`` + ``MissSensitivity`` +
``LatentConstraint`` (as ``test_fleetopt_bplane.py`` builds its
``MissSensitivity`` fixtures) rather than through a scenario or a real
conjunction assessment, so a failure points at the solver/certify code
itself rather than at propagation or screening. Tester writes tests from the
contract document. Builder must not edit ``tests/``.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aegis.fleetopt import (
    BurnGrid,
    FleetProblem,
    LatentConstraint,
    LinearProgramData,
    LpSolution,
    MissSensitivity,
    assemble,
    cone_cost_model,
    exact_penalty_threshold,
    irreconcilable_subset,
    lazy_solve,
    sequential_solve,
    solve_lp,
    solve_milp,
    solve_problem,
    verify_plan,
)

_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_grid(
    rng: np.random.Generator, *, n_sats: int = 2, n_slots: int = 2, axes: int = 3
) -> BurnGrid:
    satellite_ids = tuple(f"SAT{i}" for i in range(n_sats))
    epochs = {
        sid: tuple(_EPOCH + timedelta(minutes=20 * k) for k in range(n_slots))
        for sid in satellite_ids
    }
    mean_motion = {sid: float(rng.uniform(0.0009, 0.0013)) for sid in satellite_ids}
    return BurnGrid(satellite_ids=satellite_ids, epochs=epochs, mean_motion=mean_motion, axes=axes)


def _single_slot_grid(*, axes: int = 1, n_slots: int = 1, mean_motion: float = 0.0011) -> BurnGrid:
    return BurnGrid(
        satellite_ids=("SAT0",),
        epochs={"SAT0": tuple(_EPOCH + timedelta(minutes=20 * k) for k in range(n_slots))},
        mean_motion={"SAT0": mean_motion},
        axes=axes,
    )


def _make_resolve(
    rng: np.random.Generator, grid: BurnGrid, conjunction_id: str, *, required_slack: float = 0.6
) -> MissSensitivity:
    n_vars = grid.n_vars
    miss_vector = rng.normal(size=3) * 0.2
    # Realistic CW sensitivities are large: a few hundred to a few thousand km
    # of displacement per km/s of impulse over a multi-orbit lead time.
    sensitivity = rng.normal(size=(3, n_vars)) * rng.uniform(300.0, 1500.0)
    nominal = float(np.linalg.norm(miss_vector))
    return MissSensitivity(
        conjunction_id=conjunction_id,
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=miss_vector,
        sensitivity=sensitivity,
        required_miss_km=nominal + float(required_slack),
        nominal_miss_km=nominal,
        relative_speed_km_s=7.5,
    )


def _make_latent(
    rng: np.random.Generator, grid: BurnGrid, pair_id: str, *, floor_km: float = 1.0
) -> LatentConstraint:
    n_vars = grid.n_vars
    offset = rng.normal(size=3) * 0.5
    sensitivity = rng.normal(size=(3, n_vars)) * rng.uniform(300.0, 1500.0)
    norm = float(np.linalg.norm(offset))
    direction = offset / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    return LatentConstraint(
        pair_id=pair_id,
        object_a=f"{pair_id}A",
        object_b=f"{pair_id}B",
        epoch=_EPOCH,
        separation_km=norm,
        direction=direction,
        offset_km=offset,
        sensitivity=sensitivity,
        floor_km=float(floor_km),
        kind="bplane_minimum",
    )


def _random_problem(
    rng: np.random.Generator,
    *,
    n_resolve: int,
    n_latent: int,
    n_sats: int = 2,
    n_slots: int = 2,
    axes: int = 3,
    integer: bool = False,
) -> FleetProblem:
    grid = _make_grid(rng, n_sats=n_sats, n_slots=n_slots, axes=axes)
    resolve = [
        _make_resolve(rng, grid, f"C{i}", required_slack=float(rng.uniform(0.2, 1.0)))
        for i in range(n_resolve)
    ]
    latent = [
        _make_latent(rng, grid, f"L{j}", floor_km=float(rng.uniform(0.3, 1.5)))
        for j in range(n_latent)
    ]
    return FleetProblem(grid=grid, resolve=resolve, latent=latent, integer=integer)


def _zero_slack_labels(
    solution: LpSolution, data: LinearProgramData, *, tol: float = 1e-9
) -> tuple[set[str], set[str]]:
    resolve_slack = solution.slack(data, "resolve")
    latent_slack = solution.slack(data, "latent")
    zero_resolve = {label for label, value in resolve_slack.items() if value <= tol}
    zero_latent = {label for label, value in latent_slack.items() if value <= tol}
    return zero_resolve, zero_latent


def _zero_slack_feasible(
    problem: FleetProblem,
    resolve: list[MissSensitivity],
    latent: list[LatentConstraint],
) -> bool:
    """Solve a subsystem with slack pinned to zero, using only public API.

    Mirrors what a minimal-infeasible-subsystem search must mean by
    "feasible": a zero-slack (hard-constrained) solve succeeds. Built from
    ``FleetProblem``/``assemble``/``LinearProgramData``/``solve_lp`` alone so
    the test does not reach into ``certify``'s private probe helper.
    """
    candidate = replace(problem, resolve=list(resolve), latent=list(latent))
    directions = candidate.nominal_directions()
    data = assemble(candidate, directions)
    upper = data.upper.copy()
    for index, label in enumerate(data.col_labels):
        if label.startswith("slack:"):
            upper[index] = 0.0
    forced = LinearProgramData(
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
    return solve_lp(forced).status == "optimal"


def _feasible_single_row_problem() -> FleetProblem:
    """One satellite, one controllable resolve row, comfortably within budget."""
    grid = _single_slot_grid(axes=1, n_slots=2)
    resolve = MissSensitivity(
        conjunction_id="R1",
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.array([0.3, 0.0, 0.0]),
        sensitivity=np.array([[2000.0, 3000.0], [0.0, 0.0], [0.0, 0.0]]),
        required_miss_km=1.0,
        nominal_miss_km=0.3,
        relative_speed_km_s=7.5,
    )
    return FleetProblem(grid=grid, resolve=[resolve])


# ===========================================================================
# Section 7: aegis.fleetopt.problem.assemble
# ===========================================================================


def test_assemble_is_pure_across_repeated_calls() -> None:
    rng = _rng(1)
    problem = _random_problem(rng, n_resolve=2, n_latent=2)
    directions = problem.nominal_directions()

    first = assemble(problem, directions)
    second = assemble(problem, directions)

    assert np.array_equal(first.c, second.c)
    assert np.array_equal(first.a_ub, second.a_ub)
    assert np.array_equal(first.b_ub, second.b_ub)
    assert np.array_equal(first.lower, second.lower)
    assert np.array_equal(first.upper, second.upper)
    assert first.row_labels == second.row_labels
    assert first.col_labels == second.col_labels


def test_assemble_is_pure_with_default_directions() -> None:
    rng = _rng(2)
    problem = _random_problem(rng, n_resolve=1, n_latent=1)

    first = assemble(problem)
    second = assemble(problem)

    assert np.array_equal(first.c, second.c)
    assert np.array_equal(first.a_ub, second.a_ub)
    assert np.array_equal(first.b_ub, second.b_ub)


def test_row_labels_match_documented_grammar_and_every_column_is_labeled() -> None:
    rng = _rng(42)
    grid = _make_grid(rng, n_sats=2, n_slots=2, axes=3)
    cost_model = cone_cost_model(grid, order=1)
    resolve = [_make_resolve(rng, grid, "C1"), _make_resolve(rng, grid, "C2")]
    latent = [_make_latent(rng, grid, "PAIR1"), _make_latent(rng, grid, "PAIR2")]
    problem = FleetProblem(
        grid=grid,
        cost_model=cost_model,
        resolve=resolve,
        latent=latent,
        induced_budget=10.0,
    )
    data = assemble(problem)

    # every column has a label
    assert len(data.col_labels) == data.c.shape[0]
    assert all(isinstance(label, str) and label for label in data.col_labels)

    grammar = re.compile(
        r"^(resolve:.+"
        r"|latent:.+@.+:(grid|bplane_minimum)"
        r"|budget:.+"
        r"|sk:.+:[TN]:[+-]"
        r"|cone:.+:\d+:\d+"
        r"|induced-budget)$"
    )
    for label in data.row_labels:
        assert grammar.match(label), f"row label {label!r} does not match the documented grammar"

    # the specific documented forms are actually present for this assembly
    for sensitivity in resolve:
        assert f"resolve:{sensitivity.conjunction_id}" in data.row_labels
    for constraint in latent:
        assert constraint.label in data.row_labels
    assert "induced-budget" in data.row_labels
    assert any(label.startswith("budget:") for label in data.row_labels)
    assert any(label.startswith("sk:") for label in data.row_labels)
    assert any(label.startswith("cone:") for label in data.row_labels)

    # slack columns exist for every resolve and latent row
    for sensitivity in resolve:
        assert f"slack:resolve:{sensitivity.conjunction_id}" in data.col_labels
    for constraint in latent:
        assert f"slack:{constraint.label}" in data.col_labels
    assert len(data.resolve_slack_columns()) == len(resolve)
    assert len(data.latent_slack_columns()) == len(latent)


def test_uncontrollable_resolve_geometry_returns_positive_slack_instead_of_raising() -> None:
    """A geometry no burn can fix must return a plan with slack, never raise."""
    grid = _single_slot_grid(axes=1, n_slots=1)
    shortfall = 4.0
    resolve = MissSensitivity(
        conjunction_id="X",
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.array([1.0, 0.0, 0.0]),
        sensitivity=np.zeros((3, grid.n_vars)),
        required_miss_km=1.0 + shortfall,
        nominal_miss_km=1.0,
        relative_speed_km_s=7.5,
    )
    problem = FleetProblem(grid=grid, resolve=[resolve])

    data = assemble(problem)
    solution = solve_lp(data)

    assert solution.status == "optimal"
    slack = solution.slack(data, "resolve")
    assert slack["X"] == pytest.approx(shortfall, rel=1e-6, abs=1e-6)
    assert slack["X"] > 0.0


def test_uncontrollable_latent_geometry_returns_positive_slack_instead_of_raising() -> None:
    grid = _single_slot_grid(axes=1, n_slots=1)
    shortfall = 2.5
    constraint = LatentConstraint(
        pair_id="A:B",
        object_a="A",
        object_b="B",
        epoch=_EPOCH,
        separation_km=1.0,
        direction=np.array([1.0, 0.0, 0.0]),
        offset_km=np.array([1.0, 0.0, 0.0]),
        sensitivity=np.zeros((3, grid.n_vars)),
        floor_km=1.0 + shortfall,
        kind="bplane_minimum",
    )
    problem = FleetProblem(grid=grid, resolve=[], latent=[constraint])

    data = assemble(problem)
    solution = solve_lp(data)

    assert solution.status == "optimal"
    slack = solution.slack(data, "latent")
    key = constraint.label
    assert slack[key] == pytest.approx(shortfall, rel=1e-6, abs=1e-6)
    assert slack[key] > 0.0


# ===========================================================================
# Section 8: aegis.fleetopt.solver
# ===========================================================================


@pytest.mark.parametrize("seed", range(15))
def test_solve_lp_status_is_always_one_of_the_documented_four(seed: int) -> None:
    rng = _rng(seed)
    n_resolve = int(rng.integers(1, 4))
    n_latent = int(rng.integers(0, 3))
    problem = _random_problem(rng, n_resolve=n_resolve, n_latent=n_latent)
    data = assemble(problem)
    solution = solve_lp(data)
    assert solution.status in {"optimal", "infeasible", "unbounded", "failed"}


@pytest.mark.parametrize("seed", range(20))
def test_solve_lp_optimal_continuous_solve_reports_nonnegative_duals(seed: int) -> None:
    rng = _rng(seed)
    problem = _random_problem(rng, n_resolve=2, n_latent=1)
    data = assemble(problem)
    solution = solve_lp(data)

    assert solution.status == "optimal"
    assert solution.has_duals is True
    assert solution.duals  # non-empty: some backend actually reported marginals
    for label, value in solution.duals.items():
        assert value >= -1e-9, f"row {label!r} has a negative dual {value}"


def test_solve_lp_infeasible_status_reports_no_duals() -> None:
    """Forcing the mandatory slack column to zero makes the geometry unsatisfiable."""
    grid = _single_slot_grid(axes=1, n_slots=1)
    resolve = MissSensitivity(
        conjunction_id="X",
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.array([1.0, 0.0, 0.0]),
        sensitivity=np.zeros((3, grid.n_vars)),
        required_miss_km=6.0,
        nominal_miss_km=1.0,
        relative_speed_km_s=7.5,
    )
    problem = FleetProblem(grid=grid, resolve=[resolve], slack_caps={"resolve:X": 0.0})

    data = assemble(problem)
    solution = solve_lp(data)

    assert solution.status == "infeasible"
    assert solution.duals == {}
    assert solution.has_duals is False


def test_solve_milp_status_and_never_fakes_duals() -> None:
    rng = _rng(5)
    problem = _random_problem(
        rng, n_resolve=2, n_latent=1, n_sats=1, n_slots=1, axes=1, integer=True
    )
    data = assemble(problem)
    assert data.is_integer

    solution = solve_milp(data)
    assert solution.status in {"optimal", "infeasible", "unbounded", "failed"}
    assert solution.duals == {}
    assert solution.has_duals is False


def test_solve_problem_dispatches_to_milp_backend_when_problem_is_integer() -> None:
    rng = _rng(9)
    problem = _random_problem(
        rng, n_resolve=1, n_latent=0, n_sats=1, n_slots=1, axes=1, integer=True
    )
    solution, data = solve_problem(problem)
    assert data.is_integer
    assert solution.backend == "highs-milp"
    assert solution.duals == {}
    assert solution.has_duals is False


# ---------------------------------------------------------------------------
# sequential_solve: Propositions 2 and 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_sequential_solve_cost_history_is_nonincreasing_and_monotone_flag_true(seed: int) -> None:
    rng = _rng(seed)
    n_resolve = int(rng.integers(2, 5))
    n_latent = int(rng.integers(0, 3))
    problem = _random_problem(rng, n_resolve=n_resolve, n_latent=n_latent, n_sats=2, n_slots=2)

    result = sequential_solve(problem, max_iterations=6)

    assert result.solution.ok
    assert len(result.cost_history) >= 1
    for earlier, later in zip(result.cost_history, result.cost_history[1:]):
        assert later <= earlier + 1e-9 * max(1.0, abs(earlier))
    assert result.monotone is True


@pytest.mark.parametrize("seed", range(20))
def test_sequential_solve_every_iterate_is_safe_on_zero_slack_rows(seed: int) -> None:
    """Proposition 2/3: every SCP iterate, not just the converged one, must be
    safe under the true norm constraint on any row that used no slack.

    ``sequential_solve`` always restarts from the nominal directions, so
    calling it with an increasing ``max_iterations`` cap replays exactly the
    prefix of one continuous refinement -- which is the only way to inspect
    "every iterate" through the public surface.
    """
    rng = _rng(seed)
    n_resolve = int(rng.integers(2, 5))
    n_latent = int(rng.integers(1, 3))
    problem = _random_problem(rng, n_resolve=n_resolve, n_latent=n_latent, n_sats=2, n_slots=2)
    resolve_by_id = {s.conjunction_id: s for s in problem.resolve}
    latent_by_label = {c.label: c for c in problem.latent}

    for cap in range(1, 5):
        result = sequential_solve(problem, max_iterations=cap)
        assert result.solution.ok
        zero_resolve, zero_latent = _zero_slack_labels(result.solution, result.data)
        for conjunction_id in zero_resolve:
            residual = resolve_by_id[conjunction_id].residual_km(result.x)
            assert residual >= -1e-9, (cap, conjunction_id, residual)
        for label in zero_latent:
            residual = latent_by_label[label].residual_km(result.x)
            assert residual >= -1e-9, (cap, label, residual)


@pytest.mark.parametrize("seed", range(20))
def test_sequential_solve_max_iterations_1_is_still_safe(seed: int) -> None:
    """Proposition 2, called out explicitly: a single iteration is safe too."""
    rng = _rng(seed)
    problem = _random_problem(
        rng, n_resolve=int(rng.integers(2, 4)), n_latent=int(rng.integers(1, 3))
    )

    result = sequential_solve(problem, max_iterations=1)

    assert result.iterations == 1
    assert result.solution.ok
    resolve_by_id = {s.conjunction_id: s for s in problem.resolve}
    latent_by_label = {c.label: c for c in problem.latent}
    zero_resolve, zero_latent = _zero_slack_labels(result.solution, result.data)
    for conjunction_id in zero_resolve:
        assert resolve_by_id[conjunction_id].residual_km(result.x) >= -1e-9
    for label in zero_latent:
        assert latent_by_label[label].residual_km(result.x) >= -1e-9


# ---------------------------------------------------------------------------
# lazy_solve: the central exactness claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_linear_residual_km_never_exceeds_the_physical_residual_km(seed: int) -> None:
    """Pin the Cauchy-Schwarz gap ``linear_residual_km(x) <= residual_km(x)``.

    This is the exact distinction ``lazy_solve`` must respect (see its
    docstring): checking the physical norm instead of the enforced row
    disagreed with the full solve on 6/27 measured scenarios before the fix.
    """
    rng = _rng(seed)
    n_vars = 8
    offset = rng.normal(size=3) * rng.uniform(0.1, 3.0)
    sensitivity = rng.normal(size=(3, n_vars)) * rng.uniform(10.0, 1000.0)
    norm = float(np.linalg.norm(offset))
    direction = offset / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    constraint = LatentConstraint(
        pair_id="A:B",
        object_a="A",
        object_b="B",
        epoch=_EPOCH,
        separation_km=norm,
        direction=direction,
        offset_km=offset,
        sensitivity=sensitivity,
        floor_km=float(rng.uniform(0.01, 2.0)),
        kind="bplane_minimum",
    )
    x = rng.normal(size=n_vars) * rng.uniform(0.0, 5.0)

    assert constraint.linear_residual_km(x) <= constraint.residual_km(x) + 1e-9


@pytest.mark.parametrize("seed", range(30))
def test_lazy_solve_matches_full_objective_for_empty_true_and_adversarial_guesses(
    seed: int,
) -> None:
    rng = _rng(1000 + seed)
    n_resolve = int(rng.integers(1, 3))
    n_latent = int(rng.integers(3, 6))
    problem = _random_problem(
        rng, n_resolve=n_resolve, n_latent=n_latent, n_sats=2, n_slots=1, axes=3
    )

    full = sequential_solve(problem, max_iterations=10)
    assert full.solution.ok
    full_objective = full.objective

    true_active = {
        label
        for label, value in full.solution.duals.items()
        if label.startswith("latent:") and value > 1e-9
    }
    all_labels = [constraint.label for constraint in problem.latent]
    adversarial = set(all_labels) - true_active
    if not adversarial and all_labels:
        # every row happened to be active: fall back to a guess that omits
        # everything, which is still adversarial relative to "the truth".
        adversarial = set()

    for guess_name, guess in (
        ("empty", set()),
        ("true-active", true_active),
        ("adversarial", adversarial),
    ):
        lazy = lazy_solve(problem, active_guess=set(guess), max_rounds=10, max_iterations=10)
        assert lazy.rounds <= 10, guess_name
        assert lazy.closed is True, f"{guess_name} guess did not close: {lazy.notes}"
        assert lazy.objective == pytest.approx(
            full_objective, rel=0.0, abs=1e-7
        ), f"{guess_name} guess: {lazy.objective} != {full_objective}"


# ===========================================================================
# Section 9: aegis.fleetopt.certify
# ===========================================================================


def test_verify_plan_flags_resolve_violation_independently_of_other_checks() -> None:
    grid = _single_slot_grid(axes=1, n_slots=1)
    resolve = MissSensitivity(
        conjunction_id="R1",
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.array([0.5, 0.0, 0.0]),
        sensitivity=np.zeros((3, grid.n_vars)),
        required_miss_km=2.0,
        nominal_miss_km=0.5,
        relative_speed_km_s=7.5,
    )
    problem = FleetProblem(grid=grid, resolve=[resolve], latent=[])

    certificate = verify_plan(problem, np.zeros(grid.n_vars))

    assert [c[0] for c in certificate.resolve_violations] == ["R1"]
    assert certificate.resolve_violations[0][1] == pytest.approx(0.5 - 2.0, abs=1e-9)
    assert certificate.latent_violations == []
    assert certificate.budget_violations == []
    assert certificate.cap_violations == []
    assert certificate.station_keeping_violations == []
    assert certificate.linearized_safe is False


def test_verify_plan_flags_latent_violation_independently_of_other_checks() -> None:
    grid = _single_slot_grid(axes=1, n_slots=1)
    constraint = LatentConstraint(
        pair_id="A:B",
        object_a="A",
        object_b="B",
        epoch=_EPOCH,
        separation_km=0.3,
        direction=np.array([1.0, 0.0, 0.0]),
        offset_km=np.array([0.3, 0.0, 0.0]),
        sensitivity=np.zeros((3, grid.n_vars)),
        floor_km=2.0,
        kind="bplane_minimum",
    )
    problem = FleetProblem(grid=grid, resolve=[], latent=[constraint])

    certificate = verify_plan(problem, np.zeros(grid.n_vars))

    assert [c[0] for c in certificate.latent_violations] == [constraint.label]
    assert certificate.latent_violations[0][1] == pytest.approx(0.3 - 2.0, abs=1e-9)
    assert certificate.resolve_violations == []
    assert certificate.budget_violations == []
    assert certificate.cap_violations == []
    assert certificate.station_keeping_violations == []
    assert certificate.linearized_safe is False


def test_verify_plan_flags_budget_violation_independently_of_other_checks() -> None:
    grid = _single_slot_grid(axes=1, n_slots=2)
    problem = FleetProblem(
        grid=grid,
        resolve=[],
        latent=[],
        budgets={"SAT0": 0.001},
        per_burn_cap_km_s=1.0,
        enforce_station_keeping=False,
    )
    x = np.array([0.0006, 0.0006])  # sum 0.0012 > budget 0.001; each well under cap

    certificate = verify_plan(problem, x)

    assert [c[0] for c in certificate.budget_violations] == ["budget:SAT0"]
    assert certificate.budget_violations[0][1] == pytest.approx(0.0002, abs=1e-9)
    assert certificate.resolve_violations == []
    assert certificate.latent_violations == []
    assert certificate.cap_violations == []
    assert certificate.station_keeping_violations == []
    assert certificate.linearized_safe is False


def test_verify_plan_flags_cap_violation_independently_of_other_checks() -> None:
    grid = _single_slot_grid(axes=1, n_slots=1)
    problem = FleetProblem(
        grid=grid,
        resolve=[],
        latent=[],
        budgets={"SAT0": 1.0},
        per_burn_cap_km_s=0.0005,
        enforce_station_keeping=False,
    )
    x = np.array([0.0007])  # exceeds the per-burn cap; well under the budget

    certificate = verify_plan(problem, x)

    assert [c[0] for c in certificate.cap_violations] == ["cap:SAT0:0:T"]
    assert certificate.cap_violations[0][1] == pytest.approx(0.0002, abs=1e-9)
    assert certificate.resolve_violations == []
    assert certificate.latent_violations == []
    assert certificate.budget_violations == []
    assert certificate.station_keeping_violations == []
    assert certificate.linearized_safe is False


def test_verify_plan_flags_station_keeping_violation_independently_of_other_checks() -> None:
    grid = _single_slot_grid(axes=1, n_slots=1)
    problem = FleetProblem(
        grid=grid,
        resolve=[],
        latent=[],
        budgets={"SAT0": 10.0},
        per_burn_cap_km_s=10.0,
        enforce_station_keeping=True,
    )
    x = np.array([1.0])  # huge along-track burn: guaranteed outside the SK box

    certificate = verify_plan(problem, x)

    assert len(certificate.station_keeping_violations) >= 1
    assert certificate.resolve_violations == []
    assert certificate.latent_violations == []
    assert certificate.budget_violations == []
    assert certificate.cap_violations == []
    assert certificate.linearized_safe is False


def test_verify_plan_linearized_safe_true_only_when_every_list_is_empty() -> None:
    grid = _single_slot_grid(axes=1, n_slots=1)
    problem = FleetProblem(grid=grid, resolve=[], latent=[])

    certificate = verify_plan(problem, np.zeros(grid.n_vars))

    assert certificate.resolve_violations == []
    assert certificate.latent_violations == []
    assert certificate.budget_violations == []
    assert certificate.cap_violations == []
    assert certificate.station_keeping_violations == []
    assert certificate.linearized_safe is True


def test_verify_plan_linearized_safe_false_when_multiple_categories_violate_at_once() -> None:
    grid = _single_slot_grid(axes=1, n_slots=1)
    resolve = MissSensitivity(
        conjunction_id="R1",
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.array([0.1, 0.0, 0.0]),
        sensitivity=np.zeros((3, grid.n_vars)),
        required_miss_km=3.0,
        nominal_miss_km=0.1,
        relative_speed_km_s=7.5,
    )
    problem = FleetProblem(
        grid=grid,
        resolve=[resolve],
        latent=[],
        budgets={"SAT0": 1e-6},
        per_burn_cap_km_s=1e-6,
        enforce_station_keeping=False,
    )
    x = np.array([1e-5])  # exceeds both budget and cap; sensitivity is 0 so resolve is untouched

    certificate = verify_plan(problem, x)

    assert certificate.resolve_violations
    assert certificate.budget_violations
    assert certificate.cap_violations
    assert certificate.linearized_safe is False


# ---------------------------------------------------------------------------
# Proposition 6: exact penalty threshold
# ---------------------------------------------------------------------------


def test_proposition_6_penalty_below_and_above_threshold() -> None:
    problem = _feasible_single_row_problem()

    hard_problem = replace(problem, slack_caps={"resolve:R1": 0.0})
    hard_data = assemble(hard_problem)
    hard_solution = solve_lp(hard_data)
    assert hard_solution.status == "optimal"

    lambda_star = exact_penalty_threshold(hard_solution)
    assert lambda_star > 0.0

    low_problem = replace(problem, resolve_slack_penalty=0.5 * lambda_star)
    low_solution, low_data = solve_problem(low_problem)
    assert low_solution.status == "optimal"
    low_slack = sum(low_solution.slack(low_data, "resolve").values())
    assert low_slack > 1e-9

    high_problem = replace(problem, resolve_slack_penalty=2.0 * lambda_star)
    high_solution, high_data = solve_problem(high_problem)
    assert high_solution.status == "optimal"
    high_slack = sum(high_solution.slack(high_data, "resolve").values())
    assert high_slack == pytest.approx(0.0, abs=1e-9)
    assert high_solution.objective == pytest.approx(hard_solution.objective, rel=0.0, abs=1e-7)


def test_exact_penalty_threshold_is_nan_with_no_duals() -> None:
    solution = LpSolution(status="optimal", objective=0.0, z=np.zeros(0), duals={})
    assert solution.has_duals is False
    assert math.isnan(exact_penalty_threshold(solution))


def test_exact_penalty_threshold_is_nan_for_an_actual_milp_solve() -> None:
    rng = _rng(3)
    problem = _random_problem(
        rng, n_resolve=1, n_latent=0, n_sats=1, n_slots=1, axes=1, integer=True
    )
    data = assemble(problem)
    solution = solve_milp(data)
    assert math.isnan(exact_penalty_threshold(solution))


# ---------------------------------------------------------------------------
# irreconcilable_subset: minimal infeasible subsystem
# ---------------------------------------------------------------------------


def test_irreconcilable_subset_reports_feasible_on_a_feasible_problem() -> None:
    problem = _feasible_single_row_problem()

    farkas = irreconcilable_subset(problem, problem.nominal_directions())

    assert farkas.infeasible is False
    assert farkas.resolve_rows == []
    assert farkas.latent_rows == []


def test_irreconcilable_subset_is_minimal_on_a_genuinely_conflicting_pair() -> None:
    """Two resolve rows on the same scalar burn that pull in opposite signs:
    together infeasible with zero slack, each alone trivially satisfiable.
    """
    grid = _single_slot_grid(axes=1, n_slots=1)
    row_a = MissSensitivity(
        conjunction_id="A",
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.zeros(3),
        sensitivity=np.array([[3000.0], [0.0], [0.0]]),
        required_miss_km=0.9,
        nominal_miss_km=0.0,
        relative_speed_km_s=7.5,
    )
    row_b = MissSensitivity(
        conjunction_id="B",
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.zeros(3),
        sensitivity=np.array([[-3000.0], [0.0], [0.0]]),
        required_miss_km=0.9,
        nominal_miss_km=0.0,
        relative_speed_km_s=7.5,
    )
    # Station keeping is disabled: it is a structural row outside the
    # resolve/latent search this function performs, and a burn large enough
    # to satisfy either row alone would otherwise also trip the along-track
    # station-keeping box, muddying the isolation this test wants.
    problem = FleetProblem(grid=grid, resolve=[row_a, row_b], enforce_station_keeping=False)
    assert not _zero_slack_feasible(problem, [row_a, row_b], [])
    assert _zero_slack_feasible(problem, [row_a], [])
    assert _zero_slack_feasible(problem, [row_b], [])

    farkas = irreconcilable_subset(problem)

    assert farkas.infeasible is True
    assert set(farkas.resolve_rows) == {"A", "B"}
    assert farkas.latent_rows == []
    assert farkas.minimal is True

    # minimality, checked directly: removing any single returned row must
    # restore zero-slack feasibility of the returned subsystem.
    named_resolve = [row for row in problem.resolve if row.conjunction_id in farkas.resolve_rows]
    assert not _zero_slack_feasible(problem, named_resolve, [])
    for removed_id in farkas.resolve_rows:
        remaining = [row for row in named_resolve if row.conjunction_id != removed_id]
        assert _zero_slack_feasible(problem, remaining, []), (
            f"removing {removed_id!r} did not restore feasibility; subset is not minimal"
        )
