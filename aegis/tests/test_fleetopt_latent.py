"""Latent (induced-conjunction) constraints and structural graph metrics.

Contract: ``aegis/specs/step14_fleet_optimization.md``, sections 5
(``aegis.fleetopt.latent``), 10 (``aegis.fleetopt.graph``) and 12
(``aegis.fleetopt.pareto``).

Writes against the public ``aegis.fleetopt`` surface, including public names
that live only on a submodule's own ``__all__`` and are not re-exported at
the package root (``tidal_inflation_km``, ``thin_constraints``,
``RANK_DEFICIENT_SENTINEL``, ``premium_frontier``,
``fixed_linearization_premium``, ``RATIO_DENOMINATOR_FLOOR_KM_S``) --
none of these are leading-underscore names, so importing them directly from
``aegis.fleetopt.latent`` / ``aegis.fleetopt.graph`` / ``aegis.fleetopt.pareto``
is still the public surface, just not funnelled through the package
``__init__``. Does not import any private (leading-underscore) helper.

Section 5 propositions covered:

- ``tidal_inflation_km`` vs the crude ``grid_inflation_km``: the two-order-
  of-magnitude gap the contract requires pinning, monotonicity, the zero at
  ``step_s == 0``, and the ``FleetOptError`` on negative inputs.
- ``LatentConstraint.row()``: the affine restriction it returns implies the
  true norm constraint, and ``linear_residual_km`` never exceeds
  ``residual_km`` (Cauchy-Schwarz), over 2000 random draws.
- ``enumerate_latent_constraints``: unknown-mode error, ``"none"`` mode,
  pairs with no maneuverable member, pairs outside the reachability gate,
  deterministic ordering, and the resolve-guard behaviour whose absence was
  a real bug (an in-plane fleet-mate a dozen km away is inside the
  screening box, so excluding the whole pair silently let the optimizer
  slide straight into it).
- ``thin_constraints``: the row cap, the per-pair cap, the binding-margin
  preference, and that every kept row is one of the inputs.

Section 10 propositions covered:

- ``coupling_number``: 1.0 for a single controllable conjunction, 0.0 when
  nothing is controllable, the rank-deficient sentinel when the Gram matrix
  is singular.
- ``conflict_dimension``: zero when the latent rows lie in the resolve span.
- ``graph_metrics``: every metric finite or the documented sentinel, never
  NaN, on both an empty graph and a populated one.

Section 12 propositions covered:

- ``premium_frontier``: non-increasing, convex, ``feasible_points`` excludes
  infeasible epsilons, and the finite-difference slope agrees with the
  reported multiplier to the contract's tolerance.
- ``safety_premium``: an undefined premium is reported with a reason, never
  as zero or infinity, and ``absolute_premium_mm_s`` is always defined.
- ``premium_statistics``: never NaN; empty input gives zeros.
- ``fixed_linearization_premium``: never negative where it is defined --
  the headline research number of this contract.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from aegis.core.conjunction import RiskLevel
from aegis.fleetopt import (
    BurnGrid,
    ConjunctionGraph,
    FleetOptError,
    FleetProblem,
    LatentConstraint,
    MissSensitivity,
    ReachabilityModel,
    conflict_dimension,
    coupling_number,
    enumerate_latent_constraints,
    graph_metrics,
    grid_inflation_km,
    pair_id,
    premium_statistics,
    safety_premium,
)
from aegis.fleetopt.graph import RANK_DEFICIENT_SENTINEL
from aegis.fleetopt.latent import thin_constraints, tidal_inflation_km
from aegis.fleetopt.pareto import (
    RATIO_DENOMINATOR_FLOOR_KM_S,
    fixed_linearization_premium,
    premium_frontier,
)
from aegis.propagation import PropagationGrid

_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
_MEAN_MOTION = 0.0011


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# helpers: synthetic MissSensitivity / LatentConstraint, no propagation
# ---------------------------------------------------------------------------


def _sensitivity(
    miss_vector_km: np.ndarray,
    sensitivity: np.ndarray,
    required_miss_km: float = 0.5,
    conjunction_id: str = "c-1",
) -> MissSensitivity:
    n_vars = np.asarray(sensitivity).shape[1]
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


def _latent(
    offset_km: np.ndarray,
    sensitivity: np.ndarray,
    floor_km: float,
    *,
    direction: np.ndarray | None = None,
    pair: str = "A:B",
    object_a: str = "A",
    object_b: str = "B",
    epoch: datetime = _EPOCH,
    kind: str = "bplane_minimum",
) -> LatentConstraint:
    offset_km = np.asarray(offset_km, dtype=float)
    norm = float(np.linalg.norm(offset_km))
    if direction is None:
        direction = offset_km / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    return LatentConstraint(
        pair_id=pair,
        object_a=object_a,
        object_b=object_b,
        epoch=epoch,
        separation_km=norm,
        direction=np.asarray(direction, dtype=float),
        offset_km=offset_km,
        sensitivity=np.asarray(sensitivity, dtype=float),
        floor_km=float(floor_km),
        kind=kind,
    )


def _fake_object(object_id: str, *, maneuverable: bool) -> SimpleNamespace:
    """A stand-in ``SpaceObject`` carrying only what ``ConjunctionGraph`` reads."""
    return SimpleNamespace(object_id=object_id, is_maneuverable=maneuverable)


# ---------------------------------------------------------------------------
# 5.3 tidal_inflation_km vs the crude grid_inflation_km
# ---------------------------------------------------------------------------


def test_tidal_inflation_beats_crude_grid_inflation_by_two_orders_of_magnitude() -> None:
    """Pin the gap the contract exists to make visible.

    A vacuous latent constraint set is what you get if the crude
    ``v_max * h / 2`` bound is used as the floor inflation instead of the
    tidal bound: it swamps any realistic exclusion radius. This is exactly
    the numeric example in section 5.3 of the contract.
    """
    tidal = tidal_inflation_km(30.0, 100.0, displacement_rate_km_s=math.sqrt(51.0) * 2e-3)
    crude = grid_inflation_km(30.0)

    assert tidal < 1.0
    assert crude > 200.0
    assert crude > 200.0 * tidal, "the two bounds must differ by more than two orders of magnitude"


def test_tidal_inflation_km_is_nondecreasing_in_step_and_gate() -> None:
    steps = np.linspace(0.0, 120.0, 25)
    values_over_step = [
        tidal_inflation_km(float(step), 80.0, displacement_rate_km_s=1e-3) for step in steps
    ]
    assert np.all(np.diff(values_over_step) >= -1e-15)

    gates = np.linspace(0.0, 500.0, 25)
    values_over_gate = [
        tidal_inflation_km(30.0, float(gate), displacement_rate_km_s=1e-3) for gate in gates
    ]
    assert np.all(np.diff(values_over_gate) >= -1e-15)


def test_tidal_inflation_km_is_zero_at_zero_step_regardless_of_rate() -> None:
    assert tidal_inflation_km(0.0, 100.0, displacement_rate_km_s=0.0) == 0.0
    assert tidal_inflation_km(0.0, 100.0, displacement_rate_km_s=5.0) == 0.0
    assert tidal_inflation_km(0.0, 0.0, displacement_rate_km_s=5.0) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_s": -1.0, "gate_km": 10.0},
        {"step_s": 10.0, "gate_km": -1.0},
        {"step_s": 10.0, "gate_km": 10.0, "radius_km": 0.0},
        {"step_s": 10.0, "gate_km": 10.0, "radius_km": -1.0},
    ],
)
def test_tidal_inflation_km_negative_or_nonpositive_inputs_raise(kwargs: dict) -> None:
    with pytest.raises(FleetOptError):
        tidal_inflation_km(**kwargs)


# ---------------------------------------------------------------------------
# 5.1 LatentConstraint.row() / linear_residual_km vs residual_km
# ---------------------------------------------------------------------------


def test_latent_row_feasibility_implies_true_separation_feasibility() -> None:
    """row @ x >= rhs must imply ||offset + sensitivity @ x|| >= floor.

    This is Proposition 2's Cauchy-Schwarz argument restated for latent
    rows: the linear program's row is a restriction of the true keep-out
    norm constraint, so anything the row accepts is genuinely safe.
    """
    rng = _rng(4101)
    n_vars = 5
    offset = rng.normal(size=3) * rng.uniform(0.5, 5.0)
    sensitivity = rng.normal(size=(3, n_vars))
    floor_km = float(rng.uniform(0.1, 3.0))
    constraint = _latent(offset, sensitivity, floor_km)
    row, rhs = constraint.row()

    checked_true_branch = 0
    for _ in range(2000):
        x = rng.normal(size=n_vars) * rng.uniform(0.0, 4.0)
        if float(row @ x) >= rhs:
            checked_true_branch += 1
            assert constraint.separation_after_km(x) >= floor_km - 1e-9
    assert checked_true_branch > 50, "antecedent must actually fire, or this test is vacuous"


def test_latent_linear_residual_never_exceeds_true_residual() -> None:
    """Cauchy-Schwarz: u.(y) <= ||y|| for a unit u, so the row's margin is
    never larger than the physical margin.

    This is exactly the distinction whose confusion produced a real bug
    (documented in ``docs/pignn-acceleration-results.md``): the lazy
    cutting-plane loop must check ``linear_residual_km``, not
    ``residual_km``, or it can terminate believing a violated row is
    satisfied.
    """
    rng = _rng(4102)
    n_vars = 6
    offset = rng.normal(size=3) * rng.uniform(0.5, 5.0)
    sensitivity = rng.normal(size=(3, n_vars))
    floor_km = float(rng.uniform(0.1, 3.0))
    constraint = _latent(offset, sensitivity, floor_km)

    for _ in range(2000):
        x = rng.normal(size=n_vars) * rng.uniform(0.0, 5.0)
        assert constraint.linear_residual_km(x) <= constraint.residual_km(x) + 1e-9


# ---------------------------------------------------------------------------
# 5.2 enumerate_latent_constraints
#
# A hand-built propagation scene, calibrated (not randomised) so the
# reachability gate, the local-minimum structure and the resolve guard all
# land in known places. Three maneuverable satellites (M1, M2, M3) each
# approach M1 -- wait, M1 is the shared anchor -- along independent axes so
# their *mutual* separation never enters the gate; three non-maneuverable
# objects (D1, D2 close together; D3 always far from everything) probe the
# "no maneuverable member" and "outside the gate" exclusions independently.
# ---------------------------------------------------------------------------

_SCENE_REF = datetime(2024, 1, 1, tzinfo=timezone.utc)
_SCENE_BUDGET = 2e-5
_SCENE_EXCLUSION = 0.5
_SCENE_GUARD_S = 120.0
_SCENE_RESOLVE_TCA_S = 150.0  # seconds after _SCENE_REF


def _dip(t: np.ndarray, center: float, depth: float, width: float = 60.0) -> np.ndarray:
    return depth * np.exp(-(((t - center) / width) ** 2))


def _build_scene() -> tuple[BurnGrid, PropagationGrid, ReachabilityModel]:
    period_s = 2.0 * np.pi / _MEAN_MOTION
    t0 = _SCENE_REF - timedelta(seconds=period_s)
    sats = ("M1", "M2", "M3")
    grid = BurnGrid(
        satellite_ids=sats,
        epochs={s: (t0,) for s in sats},
        mean_motion={s: _MEAN_MOTION for s in sats},
        axes=3,
    )
    reachability = ReachabilityModel(
        mean_motion={s: _MEAN_MOTION for s in sats},
        budgets={s: _SCENE_BUDGET for s in sats},
        earliest_burn={s: t0 for s in sats},
        mode="analytic",
        exclusion_radius_km=_SCENE_EXCLUSION,
    )

    times_s = np.linspace(0.0, 600.0, 21)
    # M1-M2 dips to 0.30 km at t=150s (the "resolve" epoch, guarded) and to
    # 0.35 km at t=450s (far from the guard, must survive).
    sep_m1m2 = 3.0 - _dip(times_s, _SCENE_RESOLVE_TCA_S, 2.7) - _dip(times_s, 450.0, 2.65)
    # M1-M3 dips to 0.45 km at t=300s, on an independent axis.
    sep_m1m3 = 3.0 - _dip(times_s, 300.0, 2.55)

    base_pos = np.array([7000.0, 0.0, 0.0])
    base_vel = np.array([0.0, 7.5, 0.0])  # shared by every object => zero relative velocity
    object_ids = ["M1", "M2", "M3", "D1", "D2", "D3"]
    n_t = times_s.size
    positions = np.zeros((len(object_ids), n_t, 3))
    velocities = np.zeros((len(object_ids), n_t, 3))
    for i in range(n_t):
        positions[0, i] = base_pos
        positions[1, i] = base_pos + np.array([sep_m1m2[i], 0.0, 0.0])
        positions[2, i] = base_pos + np.array([0.0, sep_m1m3[i], 0.0])
        positions[3, i] = base_pos + np.array([0.0, 0.0, 200.0])  # D1
        positions[4, i] = base_pos + np.array([0.0, 0.0, 200.05])  # D2: 0.05 km from D1
        positions[5, i] = base_pos + np.array([50.0, 0.0, 0.0])  # D3: always 50 km from M1
        for k in range(len(object_ids)):
            velocities[k, i] = base_vel
    valid = np.ones((len(object_ids), n_t), dtype=bool)

    propagation = PropagationGrid(
        object_ids=object_ids,
        times_s=times_s,
        positions_km=positions,
        velocities_km_s=velocities,
        valid=valid,
        reference_epoch=_SCENE_REF,
    )
    return grid, propagation, reachability


def test_enumerate_latent_constraints_unknown_mode_raises() -> None:
    grid, propagation, reachability = _build_scene()
    with pytest.raises(FleetOptError):
        enumerate_latent_constraints(
            grid, propagation, reachability, exclusion_radius_km=_SCENE_EXCLUSION, mode="bogus"
        )


def test_enumerate_latent_constraints_none_mode_returns_nothing() -> None:
    grid, propagation, reachability = _build_scene()
    constraints, report = enumerate_latent_constraints(
        grid, propagation, reachability, exclusion_radius_km=_SCENE_EXCLUSION, mode="none"
    )
    assert constraints == []
    assert report.pairs_considered == 0


def test_enumerate_latent_constraints_pair_without_maneuverable_member_produces_no_row() -> None:
    """D1 and D2 sit 0.05 km apart the whole window -- well inside any gate
    -- but neither is maneuverable, so the pair must never appear."""
    grid, propagation, reachability = _build_scene()
    constraints, _ = enumerate_latent_constraints(
        grid, propagation, reachability, exclusion_radius_km=_SCENE_EXCLUSION, mode="certified_minima"
    )
    d_pair = pair_id("D1", "D2")
    assert all(c.pair_id != d_pair for c in constraints)
    assert not any({"D1", "D2"} <= {c.object_a, c.object_b} for c in constraints)


def test_enumerate_latent_constraints_pair_outside_gate_produces_no_row() -> None:
    """D3 is always 50 km from every maneuverable satellite, comfortably
    outside the certified gate (~1.4 km here), even though it does have a
    maneuverable partner."""
    grid, propagation, reachability = _build_scene()
    constraints, report = enumerate_latent_constraints(
        grid, propagation, reachability, exclusion_radius_km=_SCENE_EXCLUSION, mode="certified_minima"
    )
    assert not any("D3" in (c.object_a, c.object_b) for c in constraints)
    # D3 pairs were considered (they have a maneuverable member) but none
    # were ever inside the gate.
    gate = reachability.max_gate_km("M1", "D3", propagation.epoch_at(propagation.n_times - 1))
    assert gate < 50.0
    assert report.pairs_considered > report.pairs_inside_gate


def test_enumerate_latent_constraints_ordering_is_sorted_by_pair_epoch_kind() -> None:
    grid, propagation, reachability = _build_scene()
    constraints, _ = enumerate_latent_constraints(
        grid, propagation, reachability, exclusion_radius_km=_SCENE_EXCLUSION, mode="certified_minima"
    )
    assert len(constraints) >= 2, "test scene must actually produce multiple rows"
    keys = [(c.pair_id, c.epoch, c.kind) for c in constraints]
    assert keys == sorted(keys)


def test_enumerate_latent_constraints_resolve_guard_excludes_near_tca_keeps_far_one() -> None:
    """The pair already has a resolve row at t=150s. Without a guard it
    would also contribute a latent row there (the local minimum genuinely
    exists); with the guard, only its second, unrelated approach at t=450s
    survives -- excluding the whole pair, which the first implementation
    did, would have thrown away exactly this row.
    """
    grid, propagation, reachability = _build_scene()
    key = pair_id("M1", "M2")
    resolve_tca = _SCENE_REF + timedelta(seconds=_SCENE_RESOLVE_TCA_S)

    unguarded, _ = enumerate_latent_constraints(
        grid, propagation, reachability, exclusion_radius_km=_SCENE_EXCLUSION,
        mode="certified_minima",
    )
    unguarded_m1m2 = [c for c in unguarded if c.pair_id == key]
    assert len(unguarded_m1m2) == 2, "sanity: both minima exist absent any guard"

    guarded, report = enumerate_latent_constraints(
        grid, propagation, reachability, exclusion_radius_km=_SCENE_EXCLUSION,
        mode="certified_minima",
        resolve_epochs={key: [resolve_tca]},
        resolve_guard_s=_SCENE_GUARD_S,
    )
    guarded_m1m2 = [c for c in guarded if c.pair_id == key]

    assert len(guarded_m1m2) == 1
    kept = guarded_m1m2[0]
    assert abs((kept.epoch - resolve_tca).total_seconds()) > _SCENE_GUARD_S
    # and the row within the guard window is genuinely gone, not just moved
    assert not any(
        abs((c.epoch - resolve_tca).total_seconds()) <= _SCENE_GUARD_S for c in guarded_m1m2
    )
    assert report.excluded_resolve_pairs == 1


# ---------------------------------------------------------------------------
# 5.2 thin_constraints
# ---------------------------------------------------------------------------


def test_thin_constraints_respects_caps_and_prefers_smallest_margin() -> None:
    rng = _rng(4200)

    def make(pair: str, epoch_offset: int, slack: float) -> LatentConstraint:
        floor_km = 1.0
        separation_km = floor_km + slack
        sensitivity = rng.normal(size=(3, 2))
        return _latent(
            offset_km=np.array([separation_km, 0.0, 0.0]),
            sensitivity=sensitivity,
            floor_km=floor_km,
            pair=pair,
            object_a=pair.split(":")[0],
            object_b=pair.split(":")[1],
            epoch=_EPOCH + timedelta(seconds=epoch_offset),
        )

    # Pair P1: 5 rows, slacks 0.1..0.5 (smallest margin = tightest).
    p1 = [make("P1:X", k, 0.1 * (k + 1)) for k in range(5)]
    # Pair P2: 4 rows, slacks 0.05..0.35 -- generally tighter than P1's.
    p2 = [make("P2:X", 100 + k, 0.05 + 0.1 * k) for k in range(4)]
    # Pair P3: 3 rows, slacks 1.0..1.2 -- the loosest of all.
    p3 = [make("P3:X", 200 + k, 1.0 + 0.1 * k) for k in range(3)]
    everything = p1 + p2 + p3

    kept = thin_constraints(everything, max_rows=6, per_pair=2)

    assert len(kept) <= 6
    assert len(kept) == 6, "enough rows are available across enough pairs to fill the cap"
    counts: dict[str, int] = {}
    for c in kept:
        counts[c.pair_id] = counts.get(c.pair_id, 0) + 1
    assert all(count <= 2 for count in counts.values())

    # Every kept row must be one of the inputs (identity, not a rebuild).
    kept_ids = {id(c) for c in kept}
    input_ids = {id(c) for c in everything}
    assert kept_ids <= input_ids

    # Within any single pair, the kept members must be exactly that pair's
    # smallest-margin ones -- the per-pair cap can legitimately let a
    # loose-margin row from an under-represented pair outrank a tighter row
    # from a pair that has already hit its cap, so this property only holds
    # *within* a pair, not globally across pairs.
    for group in (p1, p2, p3):
        pair_kept = sorted(c.slack_km for c in group if id(c) in kept_ids)
        pair_dropped = sorted(c.slack_km for c in group if id(c) not in kept_ids)
        if pair_kept and pair_dropped:
            assert max(pair_kept) <= min(pair_dropped) + 1e-9
        assert len(pair_kept) <= 2


def test_thin_constraints_max_rows_zero_keeps_nothing() -> None:
    rng = _rng(4201)
    constraints = [
        _latent(
            offset_km=np.array([1.2, 0.0, 0.0]),
            sensitivity=rng.normal(size=(3, 1)),
            floor_km=1.0,
            pair=f"Q{i}:X",
            object_a=f"Q{i}",
            object_b="X",
            epoch=_EPOCH + timedelta(seconds=i),
        )
        for i in range(3)
    ]
    assert thin_constraints(constraints, max_rows=0) == []


# ---------------------------------------------------------------------------
# 10.2 coupling_number / conflict_dimension
# ---------------------------------------------------------------------------


def test_coupling_number_is_one_for_a_single_controllable_conjunction() -> None:
    sensitivity = _sensitivity(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.5], [0.0, 0.0], [0.0, 0.0]]))
    assert coupling_number([sensitivity]) == 1.0


def test_coupling_number_is_zero_when_nothing_is_controllable() -> None:
    assert coupling_number([]) == 0.0
    all_zero = [
        _sensitivity(np.array([1.0, 0.0, 0.0]), np.zeros((3, 3))),
        _sensitivity(np.array([0.0, 1.0, 0.0]), np.zeros((3, 3))),
    ]
    assert coupling_number(all_zero) == 0.0


def test_coupling_number_is_sentinel_when_gram_matrix_is_singular() -> None:
    """Two conjunctions whose normalised control-space rows are collinear
    make B B^T rank-deficient -- the hardest structural case, reported as
    the documented finite sentinel rather than raising or returning inf."""
    # direction = [1,0,0] for both, so nominal_direction @ sensitivity is
    # exactly the sensitivity's first row; [1, 0] and [2, 0] are parallel.
    collinear = [
        _sensitivity(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), conjunction_id="c1"),
        _sensitivity(np.array([1.0, 0.0, 0.0]), np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), conjunction_id="c2"),
    ]
    value = coupling_number(collinear)
    assert value == RANK_DEFICIENT_SENTINEL
    assert math.isfinite(value)


def test_conflict_dimension_zero_when_latent_rows_lie_in_resolve_span() -> None:
    resolve = [
        _sensitivity(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    ]
    # A scalar multiple of the resolve row: same 1-D control-space span.
    latent_same_span = [_latent(np.array([1.0, 0.0, 0.0]), np.array([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), 0.5)]
    assert conflict_dimension(resolve, latent_same_span) == 0

    # An orthogonal direction is a genuinely new control-space dimension.
    latent_new_span = [_latent(np.array([1.0, 0.0, 0.0]), np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]), 0.5)]
    assert conflict_dimension(resolve, latent_new_span) == 1


def test_graph_metrics_are_always_finite_or_the_documented_sentinel() -> None:
    empty_graph = ConjunctionGraph()
    metrics = graph_metrics(empty_graph)
    values = metrics.as_dict()  # raises ValueError internally on any non-finite value
    assert values["nodes"] == 0
    assert values["coupling_number"] == 0.0

    fleet_a = _fake_object("F1", maneuverable=True)
    fleet_b = _fake_object("F2", maneuverable=True)
    debris = _fake_object("D9", maneuverable=False)
    graph = ConjunctionGraph()
    for obj in (fleet_a, fleet_b, debris):
        graph.add_node(obj)
    graph.add_edge(
        "F1", "F2", "conj-1",
        tca=_EPOCH, miss_distance_km=0.3, relative_speed_km_s=7.5,
        probability=1e-4, risk_level=RiskLevel.WATCH, intra_fleet=True,
        maneuverable_ids=("F1", "F2"), short_encounter_valid=True,
    )
    graph.add_edge(
        "F1", "D9", "conj-2",
        tca=_EPOCH + timedelta(seconds=30.0), miss_distance_km=1.2, relative_speed_km_s=1.0,
        probability=1e-6, risk_level=RiskLevel.CLEAR, intra_fleet=False,
        maneuverable_ids=("F1",), short_encounter_valid=False,
    )

    # Feed it a rank-deficient resolve pair too, so the sentinel path is
    # exercised inside graph_metrics as well, not just the bare function.
    collinear = [
        _sensitivity(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), conjunction_id="c1"),
        _sensitivity(np.array([1.0, 0.0, 0.0]), np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), conjunction_id="c2"),
    ]
    populated = graph_metrics(graph, sensitivities=collinear, latent=[])
    populated_values = populated.as_dict()
    assert populated_values["coupling_number"] == RANK_DEFICIENT_SENTINEL
    for key, value in populated_values.items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{key} is not finite: {value}"


# ---------------------------------------------------------------------------
# 12.1 / 12.2 helpers: a minimal, hand-built fuel-only / fleet-safe pair.
#
# One maneuverable satellite, one burn slot, three axes. The resolve
# conjunction is only sensitive to the transverse (T) impulse; the latent
# (induced) constraint is only sensitive to the normal (N) impulse. This
# keeps the two requirements on independent axes, so the safe problem's
# minimum-cost solution is exactly "resolve's T burn, plus latent's N burn"
# -- a closed-form prediction this file can check the solver against,
# without needing station-keeping or budget rows to matter.
# ---------------------------------------------------------------------------

_PARETO_GRID = BurnGrid(
    satellite_ids=("SAT",),
    epochs={"SAT": (_EPOCH,)},
    mean_motion={"SAT": _MEAN_MOTION},
    axes=3,
)


def _pareto_problems(
    a_dv_km_s: float,
    b_dv_km_s: float,
    *,
    s_r: float = 100.0,
    s_n: float = 1000.0,
    m0: float = 0.005,
    l0: float = 0.5,
    budget_km_s: float = 10.0,
) -> tuple[FleetProblem, FleetProblem]:
    """A (fuel-only, fleet-safe) pair whose optimal delta-v is exactly
    ``a_dv_km_s`` and ``a_dv_km_s + b_dv_km_s`` respectively."""
    rho = m0 + a_dv_km_s * s_r
    resolve_sensitivity = np.zeros((3, 3))
    resolve_sensitivity[0, 1] = s_r  # R-row, T-column
    resolve = MissSensitivity(
        conjunction_id="resolve-1", primary_id="P", secondary_id="S", tca=_EPOCH,
        miss_vector_km=np.array([m0, 0.0, 0.0]), sensitivity=resolve_sensitivity,
        required_miss_km=rho, nominal_miss_km=m0, relative_speed_km_s=1.0,
    )

    floor = l0 + b_dv_km_s * s_n
    latent_sensitivity = np.zeros((3, 3))
    latent_sensitivity[2, 2] = s_n  # N-row, N-column
    latent = LatentConstraint(
        pair_id="latent-1", object_a="X", object_b="Y", epoch=_EPOCH,
        separation_km=l0, direction=np.array([0.0, 0.0, 1.0]),
        offset_km=np.array([0.0, 0.0, l0]), sensitivity=latent_sensitivity,
        floor_km=floor, kind="bplane_minimum",
    )

    common = dict(
        grid=_PARETO_GRID,
        resolve=[resolve],
        budgets={"SAT": budget_km_s},
        per_burn_cap_km_s=budget_km_s,
        enforce_station_keeping=False,
    )
    fuel_only = FleetProblem(latent=[], **common)
    fleet_safe = FleetProblem(latent=[latent], **common)
    return fuel_only, fleet_safe


# ---------------------------------------------------------------------------
# 12.1 premium_frontier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_premium_frontier_is_nonincreasing_convex_with_slopes_matching_duals(seed: int) -> None:
    rng = _rng(50_000 + seed)
    a = float(rng.uniform(1.5e-4, 5e-4))
    b = float(rng.uniform(0.5e-4, 3e-4))
    s_r = float(rng.uniform(50.0, 500.0))
    s_n = float(rng.uniform(200.0, 2000.0))
    m0 = float(rng.uniform(0.001, 0.02))
    l0 = float(rng.uniform(0.1, 1.0))

    fuel_only, fleet_safe = _pareto_problems(a, b, s_r=s_r, s_n=s_n, m0=m0, l0=l0)
    frontier = premium_frontier(fuel_only, fleet_safe, {}, points=9)

    assert len(frontier.feasible_points) == len(frontier.points), "this construction is always feasible"
    assert frontier.non_increasing
    assert frontier.is_convex
    assert frontier.slope_agreement() < 1e-6


def test_premium_frontier_feasible_points_excludes_infeasible_epsilons() -> None:
    """A tight shared budget makes full latent enforcement (low epsilon)
    infeasible while a generous induced-risk allowance (high epsilon) is
    not -- ``feasible_points`` must drop exactly the infeasible end rather
    than letting a placeholder cost corrupt the convexity check."""
    a, b = 3e-4, 4e-4
    tight_budget = a + 0.5 * b  # not enough for both burns at once
    fuel_only, fleet_safe = _pareto_problems(a, b, budget_km_s=tight_budget)

    frontier = premium_frontier(fuel_only, fleet_safe, {}, points=9)

    assert len(frontier.points) > len(frontier.feasible_points) > 0
    assert frontier.infeasible_below is not None
    for point in frontier.feasible_points:
        assert point.epsilon > frontier.infeasible_below - 1e-12
    # the true residuals reported for infeasible points must never leak into
    # the arrays the convexity/monotonicity checks use
    assert frontier.costs.size == len(frontier.feasible_points)
    assert np.all(np.isfinite(frontier.costs))


# ---------------------------------------------------------------------------
# 12.2 safety_premium (a pure function of precomputed numbers -- no LP needed)
# ---------------------------------------------------------------------------


def _record(delta_v_fuel_km_s: float, delta_v_safe_km_s: float = 0.001):
    return safety_premium(
        "scenario-1",
        delta_v_fuel_km_s=delta_v_fuel_km_s,
        delta_v_safe_km_s=delta_v_safe_km_s,
        induced_fuel_only=1,
        induced_fleet_safe=0,
        resolved_fuel_only=1,
        resolved_fleet_safe=1,
        total_conjunctions=1,
        fuel_only_feasible=True,
        fleet_safe_feasible=True,
    )


def test_safety_premium_is_none_with_a_reason_when_fuel_only_needs_no_burn() -> None:
    record = _record(delta_v_fuel_km_s=0.0, delta_v_safe_km_s=0.0002)
    assert record.premium is None
    assert record.reason == "fuel_only_requires_no_maneuver"
    assert record.infeasible is False
    assert record.absolute_premium_mm_s == pytest.approx(0.0002 * 1e6)


def test_safety_premium_is_none_below_ratio_floor_not_zero_not_infinite() -> None:
    """Below the ratio floor but strictly positive: undefined, not zero and
    not infinity -- a different reason than the exactly-zero case."""
    fuel_dv = RATIO_DENOMINATOR_FLOOR_KM_S * 0.5
    assert fuel_dv > 0.0
    record = _record(delta_v_fuel_km_s=fuel_dv, delta_v_safe_km_s=0.0005)

    assert record.premium is None
    assert record.reason == "fuel_only_delta_v_below_ratio_floor"
    assert record.premium != 0.0  # it is undefined, not a numeric zero
    assert record.absolute_premium_mm_s == pytest.approx((0.0005 - fuel_dv) * 1e6)


def test_safety_premium_absolute_premium_mm_s_always_defined() -> None:
    for fuel_dv in (0.0, RATIO_DENOMINATOR_FLOOR_KM_S * 0.5, RATIO_DENOMINATOR_FLOOR_KM_S * 5.0):
        record = _record(delta_v_fuel_km_s=fuel_dv, delta_v_safe_km_s=0.002)
        value = record.absolute_premium_mm_s
        assert math.isfinite(value)
        assert value == pytest.approx((0.002 - fuel_dv) * 1e6)


# ---------------------------------------------------------------------------
# 12.3 premium_statistics
# ---------------------------------------------------------------------------


def test_premium_statistics_empty_input_gives_count_zero_and_zeros() -> None:
    stats = premium_statistics([])
    assert stats["count"] == 0
    assert stats["defined"] == 0
    for key, value in stats.items():
        if isinstance(value, float):
            assert value == 0.0
            assert math.isfinite(value)


def test_premium_statistics_never_returns_nan() -> None:
    records = [
        _record(delta_v_fuel_km_s=0.001, delta_v_safe_km_s=0.0015),  # defined, positive
        _record(delta_v_fuel_km_s=0.002, delta_v_safe_km_s=0.002),  # defined, exactly zero
        _record(delta_v_fuel_km_s=0.0, delta_v_safe_km_s=0.0003),  # undefined (no burn)
        safety_premium(
            "infeasible-1", delta_v_fuel_km_s=0.001, delta_v_safe_km_s=0.0,
            induced_fuel_only=1, induced_fleet_safe=1, resolved_fuel_only=1,
            resolved_fleet_safe=0, total_conjunctions=1, fuel_only_feasible=True,
            fleet_safe_feasible=False,
        ),
    ]
    stats = premium_statistics(records)
    assert stats["count"] == 4
    for key, value in stats.items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{key} is NaN or infinite: {value}"


# ---------------------------------------------------------------------------
# 12.1 fixed_linearization_premium -- the headline research number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_fixed_linearization_premium_is_never_negative(seed: int) -> None:
    rng = _rng(60_000 + seed)
    a = float(rng.uniform(1.5e-4, 5e-4))
    b = float(rng.uniform(0.5e-4, 3e-4))
    s_r = float(rng.uniform(50.0, 500.0))
    s_n = float(rng.uniform(200.0, 2000.0))
    m0 = float(rng.uniform(0.001, 0.02))
    l0 = float(rng.uniform(0.1, 1.0))

    fuel_only, fleet_safe = _pareto_problems(a, b, s_r=s_r, s_n=s_n, m0=m0, l0=l0)
    premium, dv_fuel, dv_safe, note = fixed_linearization_premium(fuel_only, fleet_safe, {})

    assert dv_fuel == pytest.approx(a, rel=1e-6)
    if premium is not None:
        assert premium >= -1e-9, f"seed {seed}: negative premium {premium} ({note})"
        assert note == ""
        assert dv_safe >= dv_fuel - 1e-12


def test_fixed_linearization_premium_reports_undefined_not_negative_when_infeasible() -> None:
    """When no zero-induced plan is at least as safe as the fuel-only plan
    at this linearization, the premium is reported as undefined -- never as
    a negative number standing in for infeasibility."""
    a, b = 3e-4, 4e-4
    tight_budget = a + 0.5 * b
    fuel_only, fleet_safe = _pareto_problems(a, b, budget_km_s=tight_budget)

    premium, dv_fuel, dv_safe, note = fixed_linearization_premium(fuel_only, fleet_safe, {})

    assert dv_fuel == pytest.approx(a, rel=1e-6)
    assert premium is None
    assert note != ""
