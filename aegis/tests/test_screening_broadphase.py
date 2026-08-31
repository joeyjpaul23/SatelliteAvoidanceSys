"""Broadphase candidate-generation contract (Step 2).

Builds a ``PropagationGrid`` from known inertial states. Does not import
screening internals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from numbers import Integral

import numpy as np

from aegis.constants import MAX_RELATIVE_SPEED_KM_S, SCREENING_BOX_STARLINK_KM, SCREENING_STEP_S
from aegis.core.frames import rtn_to_eci_matrix
from aegis.core.timebase import ensure_utc
from aegis.propagation.propagator import PropagationGrid
from aegis.screening import broadphase

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)

# Circular LEO-like primary. RTN of this state is aligned with ECI axes:
# R = +X, T = +Y, N = +Z.
_R_PRIMARY_KM = np.array([6928.0, 0.0, 0.0])
_V_PRIMARY_KM_S = np.array([0.0, 7.6, 0.0])


def _grid(
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    valid: np.ndarray,
    times_s: np.ndarray,
    object_ids: list[str] | None = None,
) -> PropagationGrid:
    n_objects = positions_km.shape[0]
    if object_ids is None:
        object_ids = [str(10_000 + i) for i in range(n_objects)]
    return PropagationGrid(
        object_ids=object_ids,
        times_s=np.asarray(times_s, dtype=float),
        positions_km=np.asarray(positions_km, dtype=float),
        velocities_km_s=np.asarray(velocities_km_s, dtype=float),
        valid=np.asarray(valid, dtype=bool),
        reference_epoch=ensure_utc(_EPOCH),
    )


def _pair_at_rtn(
    relative_rtn_km: np.ndarray,
    *,
    n_times: int = 1,
    step_s: float = SCREENING_STEP_S,
    valid: np.ndarray | None = None,
) -> PropagationGrid:
    rotation = rtn_to_eci_matrix(_R_PRIMARY_KM, _V_PRIMARY_KM_S)
    secondary_km = _R_PRIMARY_KM + rotation @ np.asarray(relative_rtn_km, dtype=float)
    positions = np.zeros((2, n_times, 3))
    velocities = np.zeros((2, n_times, 3))
    for k in range(n_times):
        positions[0, k] = _R_PRIMARY_KM
        positions[1, k] = secondary_km
        velocities[0, k] = _V_PRIMARY_KM_S
        velocities[1, k] = _V_PRIMARY_KM_S
    if valid is None:
        valid = np.ones((2, n_times), dtype=bool)
    times_s = np.arange(n_times, dtype=float) * step_s
    return _grid(positions, velocities, valid, times_s, object_ids=["100", "200"])


def _assert_candidate_triples(candidates: list) -> None:
    assert isinstance(candidates, list)
    for item in candidates:
        assert isinstance(item, tuple)
        assert len(item) == 3
        i, j, time_index = item
        assert isinstance(i, Integral)
        assert isinstance(j, Integral)
        assert isinstance(time_index, Integral)
        assert i < j
        assert time_index >= 0


def test_broadphase_returns_index_time_tuples() -> None:
    radial, transverse, normal = SCREENING_BOX_STARLINK_KM
    grid = _pair_at_rtn(np.array([0.5 * radial, 0.5 * transverse, 0.5 * normal]))
    candidates = broadphase(grid)
    _assert_candidate_triples(candidates)
    assert (0, 1, 0) in candidates


def test_broadphase_detects_pair_inside_rtn_box() -> None:
    grid = _pair_at_rtn(np.array([0.5, 10.0, 5.0]))
    candidates = broadphase(grid, box_km=SCREENING_BOX_STARLINK_KM)
    assert (0, 1, 0) in candidates


def test_broadphase_skips_invalid_grid_samples() -> None:
    inside = np.array([0.4, 2.0, 1.0])
    valid = np.array([[True, False], [True, True]])
    grid = _pair_at_rtn(inside, n_times=2, valid=valid)
    candidates = broadphase(grid)
    _assert_candidate_triples(candidates)
    assert (0, 1, 0) in candidates
    assert (0, 1, 1) not in candidates


def test_broadphase_skips_when_either_object_is_invalid() -> None:
    valid = np.array([[True], [False]])
    grid = _pair_at_rtn(np.array([0.4, 2.0, 1.0]), valid=valid)
    candidates = broadphase(grid)
    assert (0, 1, 0) not in candidates


def test_broadphase_keeps_pair_outside_box_within_nomiss_gate() -> None:
    """Radial 3 km is outside the 2 km half-width, but closable in one step."""
    grid = _pair_at_rtn(np.array([3.0, 0.0, 0.0]), n_times=2)
    candidates = broadphase(
        grid,
        box_km=SCREENING_BOX_STARLINK_KM,
        max_relative_speed_km_s=MAX_RELATIVE_SPEED_KM_S,
    )
    assert (0, 1, 0) in candidates


def test_broadphase_excludes_pair_too_far_to_close() -> None:
    """Thousands of km apart: no-miss gate can rule the step out."""
    grid = _pair_at_rtn(np.array([3000.0, 0.0, 0.0]), n_times=2)
    candidates = broadphase(
        grid,
        box_km=SCREENING_BOX_STARLINK_KM,
        max_relative_speed_km_s=MAX_RELATIVE_SPEED_KM_S,
    )
    assert (0, 1, 0) not in candidates
    assert (0, 1, 1) not in candidates
