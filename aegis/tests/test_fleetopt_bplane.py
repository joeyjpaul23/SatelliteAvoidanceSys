"""B-plane sensitivity and cost-norm models (Step 14 contract, sections 3 and 6).

Writes against the public ``aegis.fleetopt`` surface only:
``bplane_projector``, ``MissSensitivity``, ``linearize``, ``refine_direction``,
``polyhedral_cone``, ``covering_radius_deg``, ``l1_cost_model``,
``cone_cost_model``. Does not import private (leading-underscore) helpers.

Section 3 propositions covered:

- 3.1 ``bplane_projector`` -- symmetry, idempotence, annihilation of the
  relative velocity, eigenvalues (0, 1, 1), identity for zero velocity.
- Proposition 2 (conservative affine restriction) as a standalone
  Cauchy-Schwarz check: ``u^T y >= r ==> ||y|| >= r`` for any unit ``u``.
- 3.3 ``linearize`` -- the returned affine row is an exact restatement of
  the projected linear constraint, and a zero direction raises.
- 3.4 ``refine_direction`` -- unit vector, equals the normalized perturbed
  miss vector, falls back to the nominal direction on collapse.
- ``MissSensitivity`` derived quantities: ``miss_after_km``,
  ``residual_km``, ``shortfall_km``, ``is_controllable``.

Section 6 (norms) propositions covered:

- 6.2 ``polyhedral_cone`` -- vertex counts, unit norms, covering radius.
- The cone approximation bracket ``cos(covering_radius) <= max_l(g_l.v)/||v|| <= 1``.
- 6.1 ``l1_cost_model`` -- lift recovers ``x`` from ``z``, the objective
  never under-charges the Euclidean norm, unit weights recover plain L1,
  non-positive weights raise.
- 6.3 ``cone_cost_model`` -- one magnitude column per burn slot, documented
  shape of the extra rows.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aegis.fleetopt import (
    AssemblyError,
    BurnGrid,
    FleetOptError,
    MissSensitivity,
    bplane_projector,
    cone_cost_model,
    covering_radius_deg,
    l1_cost_model,
    linearize,
    polyhedral_cone,
    refine_direction,
)

_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
_ZERO_TOL = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_unit(rng: np.random.Generator, dim: int = 3) -> np.ndarray:
    v = rng.normal(size=dim)
    while np.linalg.norm(v) < 1e-9:
        v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def _sensitivity(
    miss_vector_km: np.ndarray,
    sensitivity: np.ndarray,
    required_miss_km: float = 0.5,
    conjunction_id: str = "c-1",
) -> MissSensitivity:
    n_vars = sensitivity.shape[1]
    return MissSensitivity(
        conjunction_id=conjunction_id,
        primary_id="P",
        secondary_id="S",
        tca=_EPOCH,
        miss_vector_km=np.asarray(miss_vector_km, dtype=float),
        sensitivity=np.asarray(sensitivity, dtype=float).reshape(3, n_vars),
        required_miss_km=required_miss_km,
        nominal_miss_km=float(np.linalg.norm(miss_vector_km)),
        relative_speed_km_s=7.5,
    )


def _small_grid(n_slots: tuple[int, ...] = (2, 3), axes: int = 3) -> BurnGrid:
    satellite_ids = tuple(f"SAT{i}" for i in range(len(n_slots)))
    epochs = {
        sat_id: tuple(_EPOCH + timedelta(minutes=30 * k) for k in range(count))
        for sat_id, count in zip(satellite_ids, n_slots)
    }
    mean_motion = {sat_id: 0.0011 for sat_id in satellite_ids}
    return BurnGrid(
        satellite_ids=satellite_ids,
        epochs=epochs,
        mean_motion=mean_motion,
        axes=axes,
    )


# ---------------------------------------------------------------------------
# 3.1 bplane_projector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_bplane_projector_symmetric_idempotent_and_annihilates_velocity(seed: int) -> None:
    rng = _rng(seed)
    w = rng.normal(size=3) * rng.uniform(0.1, 10.0)
    P = bplane_projector(w)

    np.testing.assert_allclose(P, P.T, rtol=0.0, atol=_ZERO_TOL)
    np.testing.assert_allclose(P @ P, P, rtol=0.0, atol=_ZERO_TOL)
    np.testing.assert_allclose(P @ w, np.zeros(3), rtol=0.0, atol=_ZERO_TOL)


@pytest.mark.parametrize("seed", range(10))
def test_bplane_projector_eigenvalues_are_0_1_1(seed: int) -> None:
    rng = _rng(seed)
    w = rng.normal(size=3) * rng.uniform(0.1, 10.0)
    P = bplane_projector(w)

    eigenvalues = np.sort(np.linalg.eigvalsh(P))
    np.testing.assert_allclose(eigenvalues, [0.0, 1.0, 1.0], atol=1e-9)


def test_bplane_projector_zero_relative_velocity_returns_identity() -> None:
    P = bplane_projector(np.zeros(3))
    np.testing.assert_allclose(P, np.eye(3), rtol=0.0, atol=_ZERO_TOL)


# ---------------------------------------------------------------------------
# Proposition 2: conservative affine restriction (Cauchy-Schwarz)
# ---------------------------------------------------------------------------


def test_proposition_2_linear_feasibility_implies_norm_feasibility() -> None:
    """For any unit u: u.T @ y >= r ==> ||y|| >= r. Zero violations allowed.

    This is the safety argument underlying ``linearize``: it must hold for
    every draw, with no tolerance slack, because it is what makes every
    linearized-feasible point in the LP genuinely safe under the true
    (nonlinear-in-direction) norm constraint.
    """
    rng = _rng(20140)
    n_draws = 20_000
    violations = 0
    checked_true_branch = 0
    for _ in range(n_draws):
        u = _random_unit(rng)
        y = rng.normal(size=3) * rng.uniform(0.0, 5.0)
        r = float(rng.uniform(-2.0, 5.0))
        if float(u @ y) >= r:
            checked_true_branch += 1
            if not float(np.linalg.norm(y)) >= r - 1e-12:
                violations += 1
    assert violations == 0
    # sanity: the antecedent must actually have fired many times, otherwise
    # this test would pass vacuously
    assert checked_true_branch > n_draws // 10


# ---------------------------------------------------------------------------
# 3.3 linearize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
def test_linearize_row_rhs_exactly_restates_projected_linear_constraint(seed: int) -> None:
    rng = _rng(seed)
    n_vars = 5
    miss_vector = rng.normal(size=3)
    sensitivity_matrix = rng.normal(size=(3, n_vars))
    required = float(rng.uniform(0.1, 2.0))
    sensitivity = _sensitivity(miss_vector, sensitivity_matrix, required)

    direction = rng.normal(size=3)
    u_hat = direction / np.linalg.norm(direction)

    row, rhs = linearize(sensitivity, direction)
    assert row.shape == (n_vars,)

    for _ in range(50):
        x = rng.normal(size=n_vars) * rng.uniform(0.0, 3.0)
        lhs_form = float(u_hat @ (miss_vector + sensitivity_matrix @ x))
        row_form = float(row @ x)
        # row@x - rhs must equal u.(d+Bx) - required exactly (same algebra)
        assert row_form - rhs == pytest.approx(lhs_form - required, rel=1e-9, abs=1e-9)
        # and therefore the two feasibility checks agree exactly
        assert (row_form >= rhs) == (lhs_form >= required)


def test_linearize_zero_direction_raises_fleetopt_error() -> None:
    sensitivity = _sensitivity(np.array([1.0, 0.0, 0.0]), np.eye(3, 4)[:, :4])
    with pytest.raises(FleetOptError):
        linearize(sensitivity, np.zeros(3))


# ---------------------------------------------------------------------------
# 3.4 refine_direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_refine_direction_is_unit_and_equals_normalized_perturbed_miss(seed: int) -> None:
    rng = _rng(seed)
    n_vars = 4
    miss_vector = rng.normal(size=3) * 3.0
    sensitivity_matrix = rng.normal(size=(3, n_vars))
    sensitivity = _sensitivity(miss_vector, sensitivity_matrix)

    x = rng.normal(size=n_vars)
    direction = refine_direction(sensitivity, x)

    assert direction.shape == (3,)
    assert np.linalg.norm(direction) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    expected_vector = miss_vector + sensitivity_matrix @ x
    expected = expected_vector / np.linalg.norm(expected_vector)
    np.testing.assert_allclose(direction, expected, atol=1e-12)


def test_refine_direction_falls_back_to_nominal_when_perturbed_miss_collapses() -> None:
    miss_vector = np.array([2.0, 0.0, 0.0])
    # sensitivity chosen so that miss_vector + sensitivity @ x == 0 exactly
    sensitivity_matrix = np.array(
        [
            [-2.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    sensitivity = _sensitivity(miss_vector, sensitivity_matrix)
    x = np.array([1.0, 0.0])  # miss_vector + sensitivity @ x == [0, 0, 0]

    direction = refine_direction(sensitivity, x)
    np.testing.assert_allclose(direction, sensitivity.nominal_direction, atol=1e-12)
    # nominal_direction is itself the normalized (non-zero) miss_vector here
    np.testing.assert_allclose(direction, np.array([1.0, 0.0, 0.0]), atol=1e-12)


# ---------------------------------------------------------------------------
# MissSensitivity derived quantities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_miss_after_and_residual_km_agree_with_definitions(seed: int) -> None:
    rng = _rng(seed)
    n_vars = 6
    miss_vector = rng.normal(size=3) * 2.0
    sensitivity_matrix = rng.normal(size=(3, n_vars))
    required = float(rng.uniform(0.05, 1.5))
    sensitivity = _sensitivity(miss_vector, sensitivity_matrix, required)

    x = rng.normal(size=n_vars)
    expected_after = float(np.linalg.norm(miss_vector + sensitivity_matrix @ x))

    assert sensitivity.miss_after_km(x) == pytest.approx(expected_after, rel=1e-12, abs=1e-12)
    assert sensitivity.residual_km(x) == pytest.approx(
        expected_after - required, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize(
    ("nominal_miss", "required_miss"),
    [(2.0, 0.5), (0.5, 0.5), (0.1, 2.0), (0.0, 3.0)],
)
def test_shortfall_km_is_nonnegative_and_matches_definition(
    nominal_miss: float, required_miss: float
) -> None:
    miss_vector = np.array([nominal_miss, 0.0, 0.0])
    sensitivity = _sensitivity(miss_vector, np.zeros((3, 2)), required_miss)

    shortfall = sensitivity.shortfall_km
    assert shortfall >= 0.0
    assert shortfall == pytest.approx(max(0.0, required_miss - nominal_miss), abs=1e-12)


def test_is_controllable_false_for_all_zero_sensitivity() -> None:
    sensitivity = _sensitivity(np.array([1.0, 2.0, 3.0]), np.zeros((3, 7)))
    assert sensitivity.is_controllable is False


@pytest.mark.parametrize("seed", range(5))
def test_is_controllable_true_when_any_entry_nonzero(seed: int) -> None:
    rng = _rng(seed)
    sensitivity_matrix = np.zeros((3, 5))
    row, col = rng.integers(0, 3), rng.integers(0, 5)
    sensitivity_matrix[row, col] = 1e-6  # comfortably above the 1e-12 zero tolerance
    sensitivity = _sensitivity(np.array([1.0, 0.0, 0.0]), sensitivity_matrix)
    assert sensitivity.is_controllable is True


# ---------------------------------------------------------------------------
# 6.2 polyhedral_cone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("order", "expected_count"), [(0, 12), (1, 42), (2, 162)])
def test_polyhedral_cone_direction_counts_and_unit_norms(order: int, expected_count: int) -> None:
    directions = polyhedral_cone(order)
    assert directions.shape == (expected_count, 3)
    norms = np.linalg.norm(directions, axis=1)
    np.testing.assert_allclose(norms, np.ones(expected_count), rtol=0.0, atol=1e-12)


def test_polyhedral_cone_covering_radius_below_contract_bounds() -> None:
    radius_order1 = covering_radius_deg(polyhedral_cone(1), samples=5000, seed=1)
    radius_order2 = covering_radius_deg(polyhedral_cone(2), samples=5000, seed=2)
    assert radius_order1 < 25.0
    assert radius_order2 < 13.0


# ---------------------------------------------------------------------------
# Cone approximation bracket
# ---------------------------------------------------------------------------


def test_cone_directional_max_ratio_lies_in_cos_covering_radius_bracket() -> None:
    order = 2
    directions = polyhedral_cone(order)
    # covering_radius_deg is itself a Monte-Carlo estimate (max over sampled
    # probes), so it must be estimated with far more samples than the 2000
    # test vectors below to converge close to the true worst-case direction;
    # otherwise an independent draw could legitimately find a worse angle
    # than a loosely-sampled "covering radius" predicts. 200000 samples
    # converges to within a few hundredths of a degree for this direction set.
    covering_radius_deg_value = covering_radius_deg(directions, samples=200_000, seed=3)
    lower_bound = math.cos(math.radians(covering_radius_deg_value))

    rng = _rng(99)
    n_samples = 2000
    vectors = rng.normal(size=(n_samples, 3))
    magnitudes = np.linalg.norm(vectors, axis=1)

    projections = vectors @ directions.T
    max_projection = np.max(projections, axis=1)
    ratio = max_projection / magnitudes

    # allow a small numerical slack below the Monte-Carlo covering-radius bound
    assert np.all(ratio >= lower_bound - 1e-3)
    assert np.all(ratio <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# 6.1 l1_cost_model
# ---------------------------------------------------------------------------


def test_l1_cost_model_lift_recovers_x_from_tight_plus_minus_split() -> None:
    grid = _small_grid()
    model = l1_cost_model(grid, axis_weights=(1.0, 1.0, 1.0))

    rng = _rng(7)
    x = rng.normal(size=grid.n_vars) * rng.uniform(0.1, 5.0)
    z = np.empty(model.n_lifted)
    z[0::2] = np.maximum(x, 0.0)
    z[1::2] = np.maximum(-x, 0.0)

    recovered = model.recover(z)
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_l1_cost_model_unit_weights_recover_plain_l1() -> None:
    grid = _small_grid()
    model = l1_cost_model(grid, axis_weights=(1.0, 1.0, 1.0))

    rng = _rng(8)
    x = rng.normal(size=grid.n_vars) * rng.uniform(0.1, 5.0)
    z = np.empty(model.n_lifted)
    z[0::2] = np.maximum(x, 0.0)
    z[1::2] = np.maximum(-x, 0.0)

    objective = float(model.cost @ z)
    assert objective == pytest.approx(float(np.sum(np.abs(x))), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_l1_cost_model_objective_never_undercharges_euclidean_norm(seed: int) -> None:
    """||x||_2 <= weighted-L1(x) for any positive axis weights (weights >= 1
    trivially dominate; the default weights (3, 1, 5) all exceed 1)."""
    grid = _small_grid(n_slots=(3,))
    model = l1_cost_model(grid)  # default axis weights (3.0, 1.0, 5.0)

    rng = _rng(seed)
    x = rng.normal(size=grid.n_vars) * rng.uniform(0.1, 4.0)
    z = np.empty(model.n_lifted)
    z[0::2] = np.maximum(x, 0.0)
    z[1::2] = np.maximum(-x, 0.0)

    objective = float(model.cost @ z)
    euclidean_norm = float(np.linalg.norm(x))
    assert objective >= euclidean_norm - 1e-9


def test_l1_cost_model_nonpositive_axis_weights_raise() -> None:
    grid = _small_grid()
    with pytest.raises(AssemblyError):
        l1_cost_model(grid, axis_weights=(1.0, 0.0, 1.0))
    with pytest.raises(AssemblyError):
        l1_cost_model(grid, axis_weights=(-1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# 6.3 cone_cost_model
# ---------------------------------------------------------------------------


def test_cone_cost_model_exposes_one_magnitude_column_per_burn_slot() -> None:
    n_slots = (2, 3)
    grid = _small_grid(n_slots=n_slots)
    model = cone_cost_model(grid, order=1)

    expected_slots = {
        (sat_id, slot)
        for sat_id, count in zip(grid.satellite_ids, n_slots)
        for slot in range(count)
    }
    assert set(model.magnitude_columns.keys()) == expected_slots
    for columns in model.magnitude_columns.values():
        assert len(columns) == 1


def test_cone_cost_model_extra_rows_have_documented_shape() -> None:
    n_slots = (2, 3)
    order = 1
    grid = _small_grid(n_slots=n_slots)
    directions = polyhedral_cone(order)
    n_facets = directions.shape[0]
    n_total_slots = sum(n_slots)

    model = cone_cost_model(grid, order=order)

    n_lifted_expected = 2 * grid.n_vars + n_total_slots
    assert model.n_lifted == n_lifted_expected
    assert model.extra_rows.shape == (n_total_slots * n_facets, n_lifted_expected)
    assert model.extra_rhs.shape == (n_total_slots * n_facets,)
    assert len(model.row_labels) == n_total_slots * n_facets
    # every extra row is `g_l . dv - t <= 0`, i.e. rhs is exactly zero
    np.testing.assert_allclose(model.extra_rhs, np.zeros(n_total_slots * n_facets), atol=0.0)
