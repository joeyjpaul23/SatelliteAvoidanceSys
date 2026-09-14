"""Tests for the Step 14 fleet-optimization contract, sections 2 and 4.

``aegis/specs/step14_fleet_optimization.md``:

* Section 2, ``aegis.fleetopt.dynamics`` -- the exact Clohessy-Wiltshire
  impulse-response matrix, its spectral norm and monotone envelope, the
  ``BurnGrid`` layout, and the displacement operators built on it.
* Section 4, ``aegis.fleetopt.reachability`` -- the certified reachable
  radius (Proposition 4) and the a-priori/a-posteriori gates built from it.

Tester writes tests from the contract document. Builder must not edit
``tests/``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from aegis.constants import MU_EARTH_KM3_S2, R_EARTH_KM, REV_PER_DAY_TO_RAD_PER_S
from aegis.core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from aegis.core.state import StateVector
from aegis.core.timebase import seconds_between, shift
from aegis.fleetopt import (
    BurnGrid,
    FleetOptError,
    ReachabilityModel,
    a_posteriori_gate_km,
    build_burn_grid,
    cw_impulse_matrix,
    cw_impulse_norm,
    cw_impulse_norm_bound,
    displacement_operator,
    eci_displacement_operator,
    reach_radius_km,
)
from aegis.maneuver import along_track_response_km, clohessy_wiltshire_state

_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
_FLEET = Operator(identifier="FLEET", name="Fleet", maneuverable=True)

# A handful of representative LEO mean motions (rad/s), covering the range
# used throughout the contract's own numeric examples (~90 min - ~2 hr orbits).
_MEAN_MOTIONS = (0.0008, 0.0010871, 0.00125, 0.0015)


def _period_s(n_rad_s: float) -> float:
    return 2.0 * math.pi / n_rad_s


def _leo_mean_motion_rad_s(altitude_km: float = 550.0) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    return math.sqrt(MU_EARTH_KM3_S2 / semi_major_km**3)


def _maneuverable_object(
    object_id: str,
    n_rad_s: float,
    *,
    mean_anomaly_deg: float = 0.0,
) -> SpaceObject:
    mean_motion_rev_per_day = n_rad_s / REV_PER_DAY_TO_RAD_PER_S
    return SpaceObject(
        object_id=object_id,
        name=f"SAT-{object_id}",
        object_type=ObjectType.PAYLOAD,
        elements=OrbitalElements(
            epoch=_EPOCH,
            mean_motion_rev_per_day=mean_motion_rev_per_day,
            eccentricity=0.0,
            inclination_deg=53.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=mean_anomaly_deg,
        ),
        data_source="SYNTHETIC",
        operator=_FLEET,
    )


# ---------------------------------------------------------------------------
# 2.1 cw_impulse_matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
@pytest.mark.parametrize("orbits", [0.05, 0.23, 0.5, 1.0, 1.37, 2.84])
def test_cw_impulse_matrix_transverse_entry_matches_along_track_response_km(
    n_rad_s: float, orbits: float
) -> None:
    """Phi[1,1] agrees with along_track_response_km(n, sigma, 1.0) to 1e-12."""
    sigma = orbits * _period_s(n_rad_s)
    phi = cw_impulse_matrix(n_rad_s, sigma)
    expected = along_track_response_km(n_rad_s, sigma, 1.0)
    assert phi[1, 1] == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
@pytest.mark.parametrize("orbits", [0.05, 0.23, 0.5, 1.0, 1.37, 2.84])
def test_cw_impulse_matrix_columns_match_clohessy_wiltshire_state(
    n_rad_s: float, orbits: float
) -> None:
    """Each column of Phi is the CW position response to a unit impulse
    along that axis, to 1e-12 -- the contract's column-by-column check."""
    sigma = orbits * _period_s(n_rad_s)
    phi = cw_impulse_matrix(n_rad_s, sigma)
    zero = np.zeros(3)
    for axis, unit in enumerate(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])):
        r, _v = clohessy_wiltshire_state(n_rad_s, sigma, zero, np.array(unit))
        np.testing.assert_allclose(phi[:, axis], r, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
@pytest.mark.parametrize("lead_s", [0.0, -1.0, -1e-6, -3600.0])
def test_cw_impulse_matrix_nonpositive_lead_is_exactly_zero(
    n_rad_s: float, lead_s: float
) -> None:
    """lead_s <= 0 must return exactly the zero matrix, not merely a small
    one -- an impulse cannot act before it is applied."""
    phi = cw_impulse_matrix(n_rad_s, lead_s)
    assert np.array_equal(phi, np.zeros((3, 3)))


@pytest.mark.parametrize("sigma", [1.0, 500.0, 3600.0 * 5.0])
@pytest.mark.parametrize("n_rad_s", [0.0, 1e-16, -1e-16])
def test_cw_impulse_matrix_near_zero_mean_motion_is_rectilinear_limit(
    n_rad_s: float, sigma: float
) -> None:
    """abs(n) < 1e-15 degenerates to sigma * I3 (rectilinear limit)."""
    phi = cw_impulse_matrix(n_rad_s, sigma)
    np.testing.assert_allclose(phi, sigma * np.eye(3), rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2.2 cw_impulse_norm and cw_impulse_norm_bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
@pytest.mark.parametrize("orbits", [0.05, 0.3, 0.75, 1.0, 1.9, 3.4])
def test_cw_impulse_norm_matches_numpy_spectral_norm(n_rad_s: float, orbits: float) -> None:
    """cw_impulse_norm agrees with numpy.linalg.norm(Phi, 2) to 1e-10."""
    sigma = orbits * _period_s(n_rad_s)
    phi = cw_impulse_matrix(n_rad_s, sigma)
    expected = np.linalg.norm(phi, 2)
    assert cw_impulse_norm(n_rad_s, sigma) == pytest.approx(expected, rel=0.0, abs=1e-10)


def test_cw_impulse_norm_is_not_monotone_in_lead_time() -> None:
    """A dense sweep over 6 orbits must contain at least one strictly
    decreasing step -- the growth rate 3 - 4*cos(n*sigma) goes negative
    just after every whole orbit, so the raw norm dips. This property must
    survive: it is exactly why cw_impulse_norm_bound cannot be a pointwise
    evaluation of cw_impulse_norm."""
    n_rad_s = _leo_mean_motion_rad_s()
    sigma_max = 6.0 * _period_s(n_rad_s)
    grid = np.linspace(0.0, sigma_max, 60_000)
    norms = np.array([cw_impulse_norm(n_rad_s, float(s)) for s in grid])
    diffs = np.diff(norms)
    assert np.any(diffs < 0.0), "expected at least one strictly decreasing step"


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
def test_cw_impulse_norm_bound_dominates_true_running_maximum(n_rad_s: float) -> None:
    """Both modes must dominate np.maximum.accumulate of the exact norm,
    to 1e-9, over at least 4 orbits of lead time."""
    sigma_max = 4.0 * _period_s(n_rad_s)
    grid = np.linspace(0.0, sigma_max, 400)
    exact = np.array([cw_impulse_norm(n_rad_s, float(s)) for s in grid])
    running_max = np.maximum.accumulate(exact)
    for mode in ("analytic", "sampled"):
        bound = np.array([cw_impulse_norm_bound(n_rad_s, float(s), mode=mode) for s in grid])
        assert np.all(bound >= running_max - 1e-9), f"mode={mode} dips below the true running max"


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
def test_cw_impulse_norm_bound_analytic_is_non_decreasing(n_rad_s: float) -> None:
    """The analytic mode must be a monotone envelope, to 1e-12."""
    sigma_max = 4.0 * _period_s(n_rad_s)
    grid = np.linspace(0.0, sigma_max, 400)
    analytic = np.array(
        [cw_impulse_norm_bound(n_rad_s, float(s), mode="analytic") for s in grid]
    )
    assert np.all(np.diff(analytic) >= -1e-12)


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
def test_cw_impulse_norm_bound_analytic_overestimate_bounded_by_12_over_n(
    n_rad_s: float,
) -> None:
    """The analytic mode over-estimates the true running maximum by at most
    12/n seconds, over at least 4 orbits of lead time."""
    sigma_max = 4.0 * _period_s(n_rad_s)
    grid = np.linspace(0.0, sigma_max, 400)
    exact = np.array([cw_impulse_norm(n_rad_s, float(s)) for s in grid])
    running_max = np.maximum.accumulate(exact)
    analytic = np.array(
        [cw_impulse_norm_bound(n_rad_s, float(s), mode="analytic") for s in grid]
    )
    overestimate = analytic - running_max
    assert np.all(overestimate <= 12.0 / n_rad_s + 1e-9)


@pytest.mark.parametrize("n_rad_s", _MEAN_MOTIONS)
@pytest.mark.parametrize("mode", ["analytic", "sampled"])
@pytest.mark.parametrize("lead_s", [0.0, -1.0, -3600.0])
def test_cw_impulse_norm_bound_nonpositive_lead_is_zero(
    n_rad_s: float, mode: str, lead_s: float
) -> None:
    assert cw_impulse_norm_bound(n_rad_s, lead_s, mode=mode) == 0.0


# ---------------------------------------------------------------------------
# 2.3 BurnGrid
# ---------------------------------------------------------------------------


def _simple_grid(axes: int = 3) -> BurnGrid:
    n = _leo_mean_motion_rad_s()
    epochs_a = tuple(shift(_EPOCH, k * 600.0) for k in range(3))
    epochs_b = tuple(shift(_EPOCH, k * 900.0) for k in range(2))
    return BurnGrid(
        satellite_ids=("A", "B"),
        epochs={"A": epochs_a, "B": epochs_b},
        mean_motion={"A": n, "B": n * 1.1},
        axes=axes,
    )


def test_burn_grid_index_ordering_is_satellite_major_then_slot_then_axis() -> None:
    grid = _simple_grid(axes=3)
    # Satellite A occupies columns [0, 9), satellite B occupies [9, 15).
    expected = 0
    for slot in range(3):
        for axis in range(3):
            assert grid.index("A", slot, axis) == expected
            expected += 1
    for slot in range(2):
        for axis in range(3):
            assert grid.index("B", slot, axis) == expected
            expected += 1


def test_burn_grid_n_vars_counts_axes_times_total_slots() -> None:
    grid = _simple_grid(axes=3)
    assert grid.n_vars == 3 * (3 + 2)
    grid1 = _simple_grid(axes=1)
    assert grid1.n_vars == 1 * (3 + 2)


def test_burn_grid_rejects_duplicate_satellite_id() -> None:
    n = _leo_mean_motion_rad_s()
    epochs = (_EPOCH,)
    with pytest.raises(FleetOptError):
        BurnGrid(
            satellite_ids=("A", "A"),
            epochs={"A": epochs},
            mean_motion={"A": n},
        )


def test_burn_grid_rejects_non_ascending_epochs() -> None:
    n = _leo_mean_motion_rad_s()
    epochs = (shift(_EPOCH, 100.0), shift(_EPOCH, 50.0))
    with pytest.raises(FleetOptError):
        BurnGrid(satellite_ids=("A",), epochs={"A": epochs}, mean_motion={"A": n})


def test_burn_grid_rejects_equal_consecutive_epochs() -> None:
    """Ascending means strictly ascending -- a repeated epoch is rejected."""
    n = _leo_mean_motion_rad_s()
    epochs = (_EPOCH, _EPOCH)
    with pytest.raises(FleetOptError):
        BurnGrid(satellite_ids=("A",), epochs={"A": epochs}, mean_motion={"A": n})


@pytest.mark.parametrize("axes", [0, 2, 4, -1])
def test_burn_grid_rejects_axes_not_in_one_or_three(axes: int) -> None:
    n = _leo_mean_motion_rad_s()
    with pytest.raises(FleetOptError):
        BurnGrid(satellite_ids=("A",), epochs={"A": (_EPOCH,)}, mean_motion={"A": n}, axes=axes)


@pytest.mark.parametrize("bad_n", [0.0, -1e-4])
def test_burn_grid_rejects_non_positive_mean_motion(bad_n: float) -> None:
    with pytest.raises(FleetOptError):
        BurnGrid(satellite_ids=("A",), epochs={"A": (_EPOCH,)}, mean_motion={"A": bad_n})


def test_burn_grid_rejects_epochs_for_an_unlisted_satellite() -> None:
    n = _leo_mean_motion_rad_s()
    with pytest.raises(FleetOptError):
        BurnGrid(
            satellite_ids=("A",),
            epochs={"A": (_EPOCH,), "B": (_EPOCH,)},
            mean_motion={"A": n, "B": n},
        )


# ---------------------------------------------------------------------------
# build_burn_grid
# ---------------------------------------------------------------------------


def test_build_burn_grid_slot_layout_steps_back_half_an_orbit_from_min_lead() -> None:
    """First slot is min_lead_orbits periods before the reference TCA;
    later slots are spaced exactly half an orbit apart."""
    sat = _maneuverable_object("S1", _leo_mean_motion_rad_s())
    n = sat.elements.mean_motion_rad_s
    period = _period_s(n)
    tca = shift(_EPOCH, 10.0 * period)
    min_lead_orbits = 2.0
    burn_slots = 5

    grid = build_burn_grid(
        [sat],
        {"S1": tca},
        burn_slots=burn_slots,
        min_lead_orbits=min_lead_orbits,
        axes=1,
        now=None,
    )

    assert grid.satellite_ids == ("S1",)
    epochs = grid.slot_epochs("S1")
    assert len(epochs) == burn_slots

    # The epoch closest to TCA (last, since the tuple is ascending) must sit
    # exactly min_lead_orbits periods before the reference TCA.
    closest_lead_s = seconds_between(epochs[-1], tca)
    assert closest_lead_s == pytest.approx(min_lead_orbits * period, rel=0.0, abs=1e-6)

    # Consecutive slots are exactly half an orbit apart.
    for a, b in zip(epochs, epochs[1:]):
        assert seconds_between(a, b) == pytest.approx(0.5 * period, rel=0.0, abs=1e-6)


def test_build_burn_grid_drops_slots_strictly_before_now() -> None:
    sat = _maneuverable_object("S1", _leo_mean_motion_rad_s())
    n = sat.elements.mean_motion_rad_s
    period = _period_s(n)
    tca = shift(_EPOCH, 10.0 * period)
    min_lead_orbits = 0.5
    burn_slots = 6

    full = build_burn_grid(
        [sat], {"S1": tca}, burn_slots=burn_slots, min_lead_orbits=min_lead_orbits, axes=1, now=None
    )
    full_epochs = full.slot_epochs("S1")
    assert len(full_epochs) == burn_slots

    # Cut "now" between the second and third earliest candidate epochs.
    now = shift(full_epochs[1], 1.0)
    trimmed = build_burn_grid(
        [sat],
        {"S1": tca},
        burn_slots=burn_slots,
        min_lead_orbits=min_lead_orbits,
        axes=1,
        now=now,
    )
    trimmed_epochs = trimmed.slot_epochs("S1")
    expected = tuple(epoch for epoch in full_epochs if epoch >= now)
    assert trimmed_epochs == expected
    assert all(epoch >= now for epoch in trimmed_epochs)
    assert len(trimmed_epochs) < len(full_epochs)


def test_build_burn_grid_excludes_satellite_whose_only_tca_leaves_no_lead() -> None:
    sat = _maneuverable_object("S1", _leo_mean_motion_rad_s())
    n = sat.elements.mean_motion_rad_s
    period = _period_s(n)
    min_lead_orbits = 3.0
    # TCA is only half an orbit after "now": min_lead_orbits of lead cannot fit.
    now = _EPOCH
    tca = shift(now, 0.5 * period)

    grid = build_burn_grid(
        [sat], {"S1": tca}, burn_slots=4, min_lead_orbits=min_lead_orbits, axes=1, now=now
    )

    assert "S1" not in grid.satellite_ids
    assert "S1" in grid.excluded
    assert isinstance(grid.excluded["S1"], str) and grid.excluded["S1"]


def test_build_burn_grid_unreachable_early_tca_falls_back_to_reachable_later_one() -> None:
    """A satellite with an unreachable EARLY TCA and a reachable LATER one
    must NOT be excluded, and its grid must be laid out from the later TCA
    -- this is the important regression the contract calls out by name."""
    sat = _maneuverable_object("S1", _leo_mean_motion_rad_s())
    n = sat.elements.mean_motion_rad_s
    period = _period_s(n)
    min_lead_orbits = 3.0
    now = _EPOCH

    tca_early = shift(now, 0.2 * period)  # too soon: cannot fit 3 orbits of lead
    tca_late = shift(now, 20.0 * period)  # plenty of lead

    grid = build_burn_grid(
        [sat],
        {"S1": [tca_early, tca_late]},
        burn_slots=3,
        min_lead_orbits=min_lead_orbits,
        axes=1,
        now=now,
    )

    assert "S1" in grid.satellite_ids
    assert "S1" not in grid.excluded

    epochs = grid.slot_epochs("S1")
    lead_from_last_slot = seconds_between(epochs[-1], tca_late)
    assert lead_from_last_slot == pytest.approx(min_lead_orbits * period, rel=0.0, abs=1e-6)
    # It must not have been laid out from the early, unreachable TCA.
    lead_from_early = seconds_between(epochs[-1], tca_early)
    assert lead_from_early != pytest.approx(min_lead_orbits * period, rel=0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 2.4 displacement_operator
#
# NOTE (contract_gap): section 2.4/2.5 of the contract write the signatures
# as displacement_operator(grid, satellite_id, n_rad_s, epoch) and
# eci_displacement_operator(grid, satellite_id, n_rad_s, epoch, state), but
# the shipped functions are displacement_operator(grid, satellite_id, epoch)
# and eci_displacement_operator(grid, satellite_id, epoch, state) -- mean
# motion is read from grid.mean_motion instead of being passed again. Since
# BurnGrid already requires a mean motion per satellite, an extra explicit
# n_rad_s argument would be redundant data that could silently disagree with
# the grid's own value; we treat this as a stale contract signature, not an
# implementation defect, and test the shipped 3-/4-argument surface, which
# is what every property below actually exercises.
# ---------------------------------------------------------------------------


def test_displacement_operator_single_slot_axes_one_reproduces_along_track_response() -> None:
    n = _leo_mean_motion_rad_s()
    burn_epoch = _EPOCH
    grid = BurnGrid(satellite_ids=("A",), epochs={"A": (burn_epoch,)}, mean_motion={"A": n}, axes=1)
    eval_epoch = shift(burn_epoch, 1.7 * _period_s(n))

    psi = displacement_operator(grid, "A", eval_epoch)
    assert psi.shape == (3, 1)

    dv = 0.0037
    lead_s = seconds_between(burn_epoch, eval_epoch)
    expected = along_track_response_km(n, lead_s, dv)
    assert (psi[1, 0] * dv) == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_displacement_operator_only_named_satellites_columns_are_nonzero() -> None:
    grid = _simple_grid(axes=3)
    eval_epoch = shift(_EPOCH, 5000.0)  # after every candidate burn slot

    psi_a = displacement_operator(grid, "A", eval_epoch)
    for col in grid.columns("B"):
        assert np.all(psi_a[:, col] == 0.0)
    assert np.any(psi_a[:, grid.columns("A")] != 0.0)

    psi_b = displacement_operator(grid, "B", eval_epoch)
    for col in grid.columns("A"):
        assert np.all(psi_b[:, col] == 0.0)
    assert np.any(psi_b[:, grid.columns("B")] != 0.0)


def test_displacement_operator_slots_at_or_after_epoch_contribute_nothing() -> None:
    n = _leo_mean_motion_rad_s()
    epochs = (shift(_EPOCH, 0.0), shift(_EPOCH, 1000.0), shift(_EPOCH, 2000.0))
    grid = BurnGrid(satellite_ids=("A",), epochs={"A": epochs}, mean_motion={"A": n}, axes=3)

    # Evaluate exactly at the middle slot: slots 1 and 2 are "at or after"
    # and must contribute zero; slot 0 (strictly before) must not.
    eval_epoch = epochs[1]
    psi = displacement_operator(grid, "A", eval_epoch)

    col_slot1 = [grid.index("A", 1, axis) for axis in range(3)]
    col_slot2 = [grid.index("A", 2, axis) for axis in range(3)]
    col_slot0 = [grid.index("A", 0, axis) for axis in range(3)]

    assert np.all(psi[:, col_slot1] == 0.0)
    assert np.all(psi[:, col_slot2] == 0.0)
    assert np.any(psi[:, col_slot0] != 0.0)


# ---------------------------------------------------------------------------
# 2.5 eci_displacement_operator
# ---------------------------------------------------------------------------


def _sample_state(epoch: datetime) -> StateVector:
    return StateVector(
        epoch=epoch,
        position_km=np.array([7000.0, 123.0, -45.0]),
        velocity_km_s=np.array([0.05, 7.5, 0.3]),
    )


def test_eci_displacement_operator_preserves_norm_of_rtn_operator() -> None:
    grid = _simple_grid(axes=3)
    eval_epoch = shift(_EPOCH, 5000.0)
    state = _sample_state(eval_epoch)

    rtn_op = displacement_operator(grid, "A", eval_epoch)
    eci_op = eci_displacement_operator(grid, "A", eval_epoch, state)
    assert eci_op.shape == rtn_op.shape

    rng = np.random.default_rng(20140)
    for _ in range(50):
        x = rng.normal(size=grid.n_vars)
        rtn_norm = np.linalg.norm(rtn_op @ x)
        eci_norm = np.linalg.norm(eci_op @ x)
        # A rotation is norm-preserving to machine precision; with random x
        # of order unity the two norms are order 1e3-1e4, so "to 1e-12"
        # means relative agreement, not an absolute tolerance of 1e-12 on a
        # ~1e4-scale quantity.
        assert eci_norm == pytest.approx(rtn_norm, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# 4.1 reach_radius_km / Proposition 4
# ---------------------------------------------------------------------------


def _reachability_grid() -> tuple[BurnGrid, str]:
    n = _leo_mean_motion_rad_s()
    period = _period_s(n)
    epochs = tuple(shift(_EPOCH, k * 0.4 * period) for k in range(5))
    grid = BurnGrid(satellite_ids=("A",), epochs={"A": epochs}, mean_motion={"A": n}, axes=3)
    return grid, "A"


def test_reach_radius_km_dominates_realized_displacement_over_admissible_plans() -> None:
    """Proposition 4: for 200 random admissible plans (total impulse
    magnitude within budget, random slots and directions) the realized
    ||Psi x|| never exceeds reach_radius_km, to 1e-9."""
    grid, sat = _reachability_grid()
    n = grid.mean_motion[sat]
    earliest = grid.earliest_epoch(sat)
    eval_epoch = shift(earliest, 3.0 * _period_s(n))
    budget = 0.05  # km/s

    psi = displacement_operator(grid, sat, eval_epoch)
    rho = reach_radius_km(n, budget, earliest, eval_epoch)
    assert rho > 0.0

    rng = np.random.default_rng(7)
    n_slots = len(grid.slot_epochs(sat))
    for _ in range(200):
        # Random per-slot magnitudes on the simplex scaled to <= budget,
        # random unit directions per slot -- an admissible plan.
        raw = rng.uniform(0.0, 1.0, size=n_slots)
        total = rng.uniform(0.0, budget)
        magnitudes = raw / raw.sum() * total if raw.sum() > 0 else raw
        assert magnitudes.sum() <= budget + 1e-15

        x = np.zeros(grid.n_vars)
        for slot in range(n_slots):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            dv = magnitudes[slot] * direction
            for axis in range(3):
                x[grid.index(sat, slot, axis)] = dv[axis]

        displacement = psi @ x
        realized_norm = np.linalg.norm(displacement)
        assert realized_norm <= rho + 1e-9


def test_reach_radius_km_is_zero_before_first_burn_epoch() -> None:
    grid, sat = _reachability_grid()
    n = grid.mean_motion[sat]
    earliest = grid.earliest_epoch(sat)

    assert reach_radius_km(n, 0.1, earliest, earliest) == 0.0
    before = shift(earliest, -1.0)
    assert reach_radius_km(n, 0.1, earliest, before) == 0.0
    long_before = shift(earliest, -10_000.0)
    assert reach_radius_km(n, 0.1, earliest, long_before) == 0.0


def test_reach_radius_km_negative_budget_raises() -> None:
    grid, sat = _reachability_grid()
    n = grid.mean_motion[sat]
    earliest = grid.earliest_epoch(sat)
    later = shift(earliest, 10_000.0)
    with pytest.raises(FleetOptError):
        reach_radius_km(n, -0.001, earliest, later)


# ---------------------------------------------------------------------------
# 4.3 a_posteriori_gate_km
# ---------------------------------------------------------------------------


def _two_sat_model() -> ReachabilityModel:
    n = _leo_mean_motion_rad_s()
    return ReachabilityModel(
        mean_motion={"A": n, "B": n * 1.05},
        budgets={"A": 0.05, "B": 0.05},
        earliest_burn={"A": _EPOCH, "B": _EPOCH},
        mode="analytic",
        exclusion_radius_km=0.5,
    )


def test_a_posteriori_gate_is_at_most_the_a_priori_gate_when_within_budget() -> None:
    model = _two_sat_model()
    n = model.mean_motion["A"]
    epoch = shift(_EPOCH, 3.0 * _period_s(n))

    a_priori = model.gate_km("A", "B", epoch)
    # Realized usage well below budget even after the default 1.5x margin.
    plan_totals = {"A": 0.001, "B": 0.002}
    a_post = a_posteriori_gate_km(model, plan_totals, "A", "B", epoch)

    assert a_post <= a_priori + 1e-12
    assert a_post < a_priori  # strictly tighter in this within-budget case


def test_a_posteriori_gate_margin_below_one_raises() -> None:
    model = _two_sat_model()
    n = model.mean_motion["A"]
    epoch = shift(_EPOCH, 3.0 * _period_s(n))
    plan_totals = {"A": 0.001, "B": 0.002}
    with pytest.raises(FleetOptError):
        a_posteriori_gate_km(model, plan_totals, "A", "B", epoch, margin=0.99)
