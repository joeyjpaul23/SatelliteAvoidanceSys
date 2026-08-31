"""Coarse pair / epoch sieve over a shared propagation grid.

A pair is a candidate at an epoch when the relative position in the
primary's RTN frame falls inside the screening box, or when the no-miss
gate cannot rule the step out: if the remaining distance to the box could
be closed within one grid step at ``max_relative_speed_km_s``, the sample
is kept. Invalid SGP4 samples are skipped.

When ``cell_km`` is set, or via :func:`broadphase_partitioned`, pairs are
generated from a conservative spatial hash (same cell or neighbour cells)
instead of an all-pairs loop. The RTN-box / no-miss gate is unchanged, so
the partitioned path cannot drop a pair the naive path would keep.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..constants import MAX_RELATIVE_SPEED_KM_S, SCREENING_BOX_STARLINK_KM, SCREENING_STEP_S
from ..propagation.propagator import PropagationGrid
from .geometry import rtn_to_eci_batch

__all__ = ["broadphase", "broadphase_partitioned"]


def _step_s(grid: PropagationGrid) -> float:
    """Grid spacing for the no-miss gate.

    Uses ``grid.times_s`` spacing when at least two samples exist; a
    single-sample grid falls back to :data:`SCREENING_STEP_S`.
    """
    times = np.asarray(grid.times_s, dtype=float)
    if times.size >= 2:
        spacing = float(np.min(np.diff(times)))
        if spacing > 0.0:
            return spacing
    return float(SCREENING_STEP_S)


def _primary_is_first(ids: np.ndarray, index_i: np.ndarray, index_j: np.ndarray) -> np.ndarray:
    """True where object ``i`` is primary (lower ``object_id``)."""
    return ids[index_i] <= ids[index_j]


def _partition_geometry(
    half_widths: np.ndarray,
    reach_km: float,
    cell_km: float | None,
) -> tuple[float, int]:
    """Cell size and Chebyshev neighbour radius with no false negatives.

    The farthest inertial separation the RTN-box / no-miss gate can keep
    is ``||box + reach||``. Objects that far apart have cell-index
    difference at most ``ceil(d_max / cell)`` in each axis, so that
    neighbourhood plus the same gate reproduces the naive candidate set.
    """
    d_max = float(np.linalg.norm(np.asarray(half_widths, dtype=float) + float(reach_km)))
    if d_max <= 0.0:
        d_max = float(reach_km) if reach_km > 0.0 else 1.0
    if cell_km is None or cell_km <= 0.0:
        cell = d_max
    else:
        cell = float(cell_km)
    radius = int(np.ceil(d_max / cell))
    if radius < 1:
        radius = 1
    return cell, radius


def _pairs_in_neighbourhood(
    positions_km: np.ndarray,
    usable: np.ndarray,
    cell_km: float,
    neighbor_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(i, j)`` with ``i < j`` that share a cell or a neighbour cell."""
    usable_idx = np.flatnonzero(np.asarray(usable, dtype=bool))
    if usable_idx.size < 2:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    cells = np.floor(np.asarray(positions_km, dtype=float) / cell_km).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in usable_idx.tolist():
        key = (int(cells[idx, 0]), int(cells[idx, 1]), int(cells[idx, 2]))
        buckets[key].append(int(idx))

    pair_i: list[int] = []
    pair_j: list[int] = []
    offsets = [
        (dx, dy, dz)
        for dx in range(-neighbor_radius, neighbor_radius + 1)
        for dy in range(-neighbor_radius, neighbor_radius + 1)
        for dz in range(-neighbor_radius, neighbor_radius + 1)
    ]
    for cell, occupants in buckets.items():
        for dx, dy, dz in offsets:
            neighbour = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if neighbour < cell:
                continue
            others = buckets.get(neighbour)
            if others is None:
                continue
            if neighbour == cell:
                for a_pos in range(len(occupants)):
                    for b_pos in range(a_pos + 1, len(occupants)):
                        left, right = occupants[a_pos], occupants[b_pos]
                        if left > right:
                            left, right = right, left
                        pair_i.append(left)
                        pair_j.append(right)
            else:
                for left in occupants:
                    for right in others:
                        if left > right:
                            left, right = right, left
                        pair_i.append(left)
                        pair_j.append(right)

    if not pair_i:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    return np.asarray(pair_i, dtype=np.intp), np.asarray(pair_j, dtype=np.intp)


def _gate_pairs(
    positions: np.ndarray,
    rotations: np.ndarray,
    usable: np.ndarray,
    object_ids: np.ndarray,
    index_i: np.ndarray,
    index_j: np.ndarray,
    half_widths: np.ndarray,
    reach_km: float,
) -> np.ndarray:
    """Boolean keep mask for candidate pairs under the RTN-box / no-miss gate."""
    if index_i.size == 0:
        return np.zeros(0, dtype=bool)

    pair_ok = usable[index_i] & usable[index_j]
    if not np.any(pair_ok):
        return pair_ok

    primary_first = _primary_is_first(object_ids, index_i, index_j)
    primary_index = np.where(primary_first, index_i, index_j)
    other_index = np.where(primary_first, index_j, index_i)
    relative_eci = positions[other_index] - positions[primary_index]
    relative_rtn = np.einsum("pji,pj->pi", rotations[primary_index], relative_eci)
    excess = np.maximum(np.abs(relative_rtn) - half_widths, 0.0)
    remaining = np.linalg.norm(excess, axis=1)
    return pair_ok & ((remaining == 0.0) | (remaining <= reach_km))


def _collect_candidates(
    grid: PropagationGrid,
    *,
    box_km: tuple[float, float, float],
    max_relative_speed_km_s: float,
    partitioned: bool,
    cell_km: float | None,
) -> list[tuple[int, int, int]]:
    n_objects = grid.n_objects
    n_times = grid.n_times
    if n_objects < 2 or n_times == 0:
        return []

    step_s = _step_s(grid)
    reach_km = float(max_relative_speed_km_s) * step_s
    half_widths = np.asarray(box_km, dtype=float).reshape(3)
    object_ids = np.asarray(grid.object_ids)

    all_i: np.ndarray | None = None
    all_j: np.ndarray | None = None
    if not partitioned:
        all_i, all_j = np.triu_indices(n_objects, k=1)

    part_cell = 0.0
    neighbor_radius = 1
    if partitioned:
        part_cell, neighbor_radius = _partition_geometry(half_widths, reach_km, cell_km)

    candidates: list[tuple[int, int, int]] = []
    for time_index in range(n_times):
        positions, velocities, valid_sgp4 = grid.slice_epoch(time_index)
        rotations, valid_rtn = rtn_to_eci_batch(positions, velocities)
        usable = np.asarray(valid_sgp4, dtype=bool) & valid_rtn

        if partitioned:
            index_i, index_j = _pairs_in_neighbourhood(
                positions, usable, part_cell, neighbor_radius
            )
        else:
            index_i, index_j = all_i, all_j

        keep = _gate_pairs(
            positions,
            rotations,
            usable,
            object_ids,
            index_i,
            index_j,
            half_widths,
            reach_km,
        )
        if not np.any(keep):
            continue

        kept_i = index_i[keep]
        kept_j = index_j[keep]
        for object_i, object_j in zip(kept_i.tolist(), kept_j.tolist(), strict=True):
            candidates.append((int(object_i), int(object_j), time_index))

    return candidates


def broadphase(
    grid: PropagationGrid,
    *,
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM,
    max_relative_speed_km_s: float = MAX_RELATIVE_SPEED_KM_S,
    cell_km: float | None = None,
) -> list[tuple[int, int, int]]:
    """Return ``(i, j, time_index)`` candidates with ``i < j``.

    Primary for the RTN box is the lower ``object_id`` -- the grid has no
    maneuverability flags. ``screen`` re-evaluates geometry in the
    assigned primary's RTN after TCA refinement.

    ``cell_km is None`` keeps the vectorised all-pairs path. A positive
    ``cell_km`` (or :func:`broadphase_partitioned`) uses the conservative
    spatial hash.
    """
    return _collect_candidates(
        grid,
        box_km=box_km,
        max_relative_speed_km_s=max_relative_speed_km_s,
        partitioned=cell_km is not None,
        cell_km=cell_km,
    )


def broadphase_partitioned(
    grid: PropagationGrid,
    *,
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM,
    max_relative_speed_km_s: float = MAX_RELATIVE_SPEED_KM_S,
    cell_km: float | None = None,
) -> list[tuple[int, int, int]]:
    """Like :func:`broadphase` but always spatially hashed.

    Cell size defaults to the conservative inertial bound
    ``||box + reach||`` so neighbour-cell tests cannot drop a pair the
    naive gate would keep.
    """
    return _collect_candidates(
        grid,
        box_km=box_km,
        max_relative_speed_km_s=max_relative_speed_km_s,
        partitioned=True,
        cell_km=cell_km,
    )
