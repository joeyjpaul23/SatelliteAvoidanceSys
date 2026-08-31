"""Catalog-level screening: prefilter, propagate, broad phase, refine TCA."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..constants import (
    SCREENING_BLOCK_DURATION_S,
    SCREENING_BOX_STARLINK_KM,
    SCREENING_PARTITION_MIN_OBJECTS,
    SCREENING_STEP_S,
)
from ..core.conjunction import Conjunction
from ..core.frames import rtn_to_eci_matrix
from ..core.objects import SpaceObject
from ..core.timebase import ensure_utc, shift
from ..ingest import MixedDataSourceError
from ..propagation.propagator import PropagationError, Sgp4Propagator
from .broadphase import broadphase, broadphase_partitioned
from .errors import ScreeningError
from .geometry import assign_primary, inside_box
from .prefilter import prefilter_pairs
from .tca import refine_tca

__all__ = ["screen"]


def _reject_mixed_sources(objects: list[SpaceObject]) -> None:
    if not objects:
        return
    sources = {obj.data_source for obj in objects}
    if len(sources) > 1:
        raise MixedDataSourceError(
            "cannot screen objects from mixed data sources: " + ", ".join(sorted(repr(s) for s in sources))
        )


def _cluster_time_indices(time_indices: list[int]) -> list[list[int]]:
    """Split a sorted list of epochs into contiguous runs."""
    if not time_indices:
        return []
    ordered = sorted(time_indices)
    clusters: list[list[int]] = [[ordered[0]]]
    for time_index in ordered[1:]:
        if time_index <= clusters[-1][-1] + 1:
            clusters[-1].append(time_index)
        else:
            clusters.append([time_index])
    return clusters


def _best_guess_epoch(grid, index_a: int, index_b: int, cluster: list[int]) -> datetime:
    """Grid epoch of minimum inertial range inside a candidate cluster."""
    best_index = cluster[0]
    best_range = np.inf
    for time_index in cluster:
        if not (grid.valid[index_a, time_index] and grid.valid[index_b, time_index]):
            continue
        delta = grid.positions_km[index_a, time_index] - grid.positions_km[index_b, time_index]
        separation = float(np.linalg.norm(delta))
        if separation < best_range:
            best_range = separation
            best_index = time_index
    return grid.epoch_at(best_index)


def _cluster_entered_box(
    grid,
    index_a: int,
    index_b: int,
    cluster: list[int],
    box_km: tuple[float, float, float],
    object_a: SpaceObject,
    object_b: SpaceObject,
) -> bool:
    """True if any cluster sample lies inside the assigned primary's RTN box."""
    primary, _secondary = assign_primary(object_a, object_b)
    primary_index = index_a if primary is object_a else index_b
    other_index = index_b if primary_index == index_a else index_a
    for time_index in cluster:
        if not (grid.valid[primary_index, time_index] and grid.valid[other_index, time_index]):
            continue
        rotation = rtn_to_eci_matrix(
            grid.positions_km[primary_index, time_index],
            grid.velocities_km_s[primary_index, time_index],
        )
        relative_rtn = rotation.T @ (
            grid.positions_km[other_index, time_index] - grid.positions_km[primary_index, time_index]
        )
        if inside_box(relative_rtn, box_km):
            return True
    return False


def screen(
    objects: list[SpaceObject],
    start: datetime,
    duration_s: float,
    *,
    step_s: float = SCREENING_STEP_S,
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM,
    propagator: Sgp4Propagator | None = None,
    partitioned: bool | None = None,
) -> list[Conjunction]:
    """Find close approaches that enter the RTN screening box.

    Intra-fleet pairs are not excluded. Mixed ``data_source`` values raise
    :class:`~aegis.ingest.MixedDataSourceError`.

    ``partitioned is None`` turns the conservative spatial-hash broad
    phase on when ``len(objects) >= SCREENING_PARTITION_MIN_OBJECTS``.
    Catalogs at that size also propagate in
    :data:`~aegis.constants.SCREENING_BLOCK_DURATION_S` blocks.
    """
    _reject_mixed_sources(objects)
    if len(objects) < 2:
        return []
    if duration_s < 0.0:
        raise ScreeningError("screening duration must be non-negative")
    if step_s <= 0.0:
        raise ScreeningError("screening step must be positive")

    start = ensure_utc(start)
    window_end = shift(start, duration_s)
    n_objects = len(objects)
    use_partition = (
        n_objects >= SCREENING_PARTITION_MIN_OBJECTS if partitioned is None else bool(partitioned)
    )
    block_duration_s = (
        SCREENING_BLOCK_DURATION_S if n_objects >= SCREENING_PARTITION_MIN_OBJECTS else None
    )

    if propagator is None:
        try:
            propagator = Sgp4Propagator(objects)
        except PropagationError as error:
            raise ScreeningError(str(error)) from error

    grid_index = {object_id: i for i, object_id in enumerate(propagator.object_ids)}
    surviving: set[tuple[int, int]] = set()
    for index_a, index_b in prefilter_pairs(objects):
        grid_a = grid_index.get(objects[index_a].object_id)
        grid_b = grid_index.get(objects[index_b].object_id)
        if grid_a is None or grid_b is None:
            continue
        if grid_a > grid_b:
            grid_a, grid_b = grid_b, grid_a
        surviving.add((grid_a, grid_b))
    if not surviving:
        return []

    try:
        grid = propagator.propagate_grid(
            start, duration_s, step_s, block_duration_s=block_duration_s
        )
    except PropagationError as error:
        raise ScreeningError(str(error)) from error

    sieve = broadphase_partitioned if use_partition else broadphase
    by_pair: dict[tuple[int, int], list[int]] = {}
    for index_a, index_b, time_index in sieve(grid, box_km=box_km):
        pair = (index_a, index_b)
        if pair not in surviving:
            continue
        by_pair.setdefault(pair, []).append(time_index)

    conjunctions: list[Conjunction] = []
    seen_ids: set[str] = set()
    for (index_a, index_b), time_indices in by_pair.items():
        for cluster in _cluster_time_indices(time_indices):
            t_guess = _best_guess_epoch(grid, index_a, index_b, cluster)
            conjunction = refine_tca(propagator, index_a, index_b, t_guess)
            entered = inside_box(conjunction.relative_position_rtn_km, box_km) or _cluster_entered_box(
                grid,
                index_a,
                index_b,
                cluster,
                box_km,
                propagator.objects[index_a],
                propagator.objects[index_b],
            )
            if not entered:
                continue
            conjunction.screening_window_start = start
            conjunction.screening_window_end = window_end
            if conjunction.conjunction_id in seen_ids:
                continue
            seen_ids.add(conjunction.conjunction_id)
            conjunctions.append(conjunction)

    conjunctions.sort(key=lambda item: (item.tca, item.conjunction_id))
    return conjunctions
