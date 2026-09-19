"""Catalog-level screening: sweep, refine, accept, as columns or as objects."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime

import numpy as np

from ..constants import (
    SCREENING_BOX_STARLINK_KM,
    SCREENING_PARTITION_MIN_OBJECTS,
    SCREENING_STEP_S,
)
from ..core.conjunction import Conjunction
from ..core.objects import SpaceObject
from ..core.timebase import ensure_utc, shift
from ..ingest import MixedDataSourceError
from ..propagation.propagator import PropagationError, Sgp4Propagator
from .errors import ScreeningError
from .results import ConjunctionTable, Rows, accept, dedupe
from .sweep import make_executor, make_spec, parallel_workers, run_blocks, use_parallel

__all__ = ["screen", "screen_table"]


def _reject_mixed_sources(objects: list[SpaceObject]) -> None:
    if not objects:
        return
    sources = {obj.data_source for obj in objects}
    if len(sources) > 1:
        raise MixedDataSourceError(
            "cannot screen objects from mixed data sources: " + ", ".join(sorted(repr(s) for s in sources))
        )


def screen_table(
    objects: list[SpaceObject],
    start: datetime,
    duration_s: float,
    *,
    step_s: float = SCREENING_STEP_S,
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM,
    propagator: Sgp4Propagator | None = None,
    partitioned: bool | None = None,
    workers: int | None = None,
) -> ConjunctionTable:
    """Every close approach that enters the RTN screening box, as columns.

    Use this at catalog scale; :func:`screen` builds ``Conjunction`` objects
    from it. Intra-fleet pairs are not excluded. Mixed ``data_source``
    values raise :class:`~aegis.ingest.MixedDataSourceError`.

    Detection streams time blocks through a chord gate
    (:mod:`~aegis.screening.sweep`), so no pair that can reach the box is
    dropped and the full grid is never held in memory. Each approach is
    refined in batches (:mod:`~aegis.screening.batch_tca`) and kept when its
    TCA (the local minimum of relative distance) lies inside either object's
    box, each box aligned with that object's own orbit (RTN). This is SpaceX
    Space Safety's close-approach definition, and it does not depend on the
    grid step. A pair that leaves the boxes and comes back gets one row per
    approach.

    ``partitioned`` picks candidate-pair generation: a k-d tree (True) or all
    pairs (False); ``None`` uses the tree from
    :data:`~aegis.constants.SCREENING_PARTITION_MIN_OBJECTS` objects. Both
    give the same result.

    ``workers`` caps parallel worker processes (default ``AEGIS_WORKERS`` or
    the CPU count). Catalogs too small to benefit run in-process. Workers
    are spawned, so a script screening a large catalog must call this under
    ``if __name__ == "__main__":`` (every AEGIS entry point does).
    """
    _reject_mixed_sources(objects)
    if duration_s < 0.0:
        raise ScreeningError("screening duration must be non-negative")
    if step_s <= 0.0:
        raise ScreeningError("screening step must be positive")
    start = ensure_utc(start)
    window_end = shift(start, duration_s)
    if len(objects) < 2:
        return ConjunctionTable(list(objects), start, window_end, Rows.concat([]))
    use_tree = (
        len(objects) >= SCREENING_PARTITION_MIN_OBJECTS if partitioned is None else bool(partitioned)
    )
    if propagator is None:
        try:
            propagator = Sgp4Propagator(objects)
        except PropagationError as error:
            raise ScreeningError(str(error)) from error
    catalog = propagator.objects

    spec = make_spec(
        propagator,
        start,
        duration_s,
        step_s,
        box_km,
        requested_ids={obj.object_id for obj in objects},
        use_tree=use_tree,
    )
    n_workers = parallel_workers(workers)
    parallel = use_parallel(len(catalog), spec.n_samples, workers)
    pool = make_executor(catalog, spec, n_workers) if parallel else nullcontext()
    with pool as executor:
        rows, stitched = run_blocks(propagator, spec, executor=executor, workers=n_workers)
    # Approaches that crossed a block edge were stitched here; refine them in-process.
    live = stitched.possible | stitched.entered
    rows = Rows.concat(
        [
            rows,
            accept(
                propagator,
                spec,
                stitched.index_a[live],
                stitched.index_b[live],
                stitched.sample[live],
            ),
        ]
    )
    return ConjunctionTable(catalog, start, window_end, dedupe(rows, spec))


def screen(
    objects: list[SpaceObject],
    start: datetime,
    duration_s: float,
    *,
    step_s: float = SCREENING_STEP_S,
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM,
    propagator: Sgp4Propagator | None = None,
    partitioned: bool | None = None,
    keep_pair: Callable[[SpaceObject, SpaceObject], bool] | None = None,
    workers: int | None = None,
) -> list[Conjunction]:
    """Close approaches that enter the RTN screening box, as ``Conjunction`` objects.

    Same search as :func:`screen_table`. ``keep_pair(obj_a, obj_b)`` if set
    must return True to retain a pair; the console uses it to skip
    debris–debris.
    """
    if len(objects) < 2:
        _reject_mixed_sources(objects)
        return []
    table = screen_table(
        objects,
        start,
        duration_s,
        step_s=step_s,
        box_km=box_km,
        propagator=propagator,
        partitioned=partitioned,
        workers=workers,
    )
    if keep_pair is not None and len(table):
        verdicts: dict[tuple[int, int], bool] = {}
        keep = np.ones(len(table), dtype=bool)
        for position, pair in enumerate(zip(table.rows.primary.tolist(), table.rows.secondary.tolist())):
            if pair not in verdicts:
                verdicts[pair] = bool(keep_pair(table.catalog[pair[0]], table.catalog[pair[1]]))
            keep[position] = verdicts[pair]
        table = ConjunctionTable(table.catalog, table.start, table.window_end, table.rows.take(np.flatnonzero(keep)))
    return table.to_conjunctions()
