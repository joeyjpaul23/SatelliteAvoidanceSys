"""Streaming close-approach sweep: time blocks, k-d tree, chord gate, arrays.

The catalog-scale detection pass behind :func:`~aegis.screening.screen`.
Measured on the full Space-Track Starlink + LEO-debris catalog (14,067
objects, 2026-09-18), the grid-wide broad phase it replaces kept 4.1 M
pairs per 60 s sample; this keeps about 6,600, with no false negatives.

**Chord gate.** For each interval ``[t_k, t_k+1]`` a pair is kept when
the straight segment between its two sampled relative positions passes
within the screening box's circumscribed sphere, after subtracting the
linear-interpolation bound ``(1/8) A dt^2``. ``A`` bounds the relative
acceleration: gravity gradient over the separation plus a floor covering
J2 and drag differences. The gate deliberately uses positions at both
ends rather than SGP4's reported velocity: for high-eccentricity objects
(e ~ 0.3 rocket bodies) that velocity disagrees with SGP4's own positions
by tens of m/s. Validated against 0.5 s propagation on 96,265 real pairs,
including 35,043 dropped pairs that truly came within 20 km of the sphere:
zero false negatives, and actual curvature never used more than 54% of
the bound.

**Candidates** come from a k-d tree over positions, bounded by
:data:`MAX_RELATIVE_SPEED_KM_S` so no pair that can close the gap within
one step is skipped. Nothing is enumerated pair by pair in Python.

**Events.** A pair can stay close for hours (satellites in one launch train)
and approach many times. Each kept run of a pair is split into *events* at
range maxima that lie outside the RTN box: leaving the box and coming back is
a new approach, while a pair that stays inside the box (two satellites 50 m
apart in one slot) is one event however much its range wiggles. Each event
seeds one TCA refinement at its closest sample and records whether any of
its samples entered the box. The previous engine made one seed per
contiguous run and so reported only the closest of repeated approaches.

**Radial test.** Separately from the gate, each kept interval records
whether the pair's orbital radii can come within the box's radial half-width
(plus ``sphere^2 / R_E`` for the RTN projection at that separation), with
each orbit's radius allowed to curve by ``mu e / r_p^2`` plus a J2/drag
floor. A pass whose intervals all fail it cannot be inside the box, so it is
not refined. For Starlink this drops the cross-shell passes: the shells are
5 to 10 km apart and the box is 2 km deep radially.

**Time blocks** are independent: they stream (the full grid is never held
in memory) and run in parallel worker processes on large catalogs. Blocks
share their edge sample; an event cut by a block edge is stitched back
together afterwards using the ranges on both sides of the edge, so the
result does not depend on where blocks fall.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context

import numpy as np
from scipy.spatial import cKDTree

from ..constants import (
    J2,
    MAX_RELATIVE_SPEED_KM_S,
    MU_EARTH_KM3_S2,
    PERIGEE_APOGEE_PAD_KM,
    R_EARTH_KM,
    SCREENING_STEP_S,
)
from ..core.objects import SpaceObject
from ..propagation.propagator import Sgp4Propagator
from .geometry import rtn_to_eci_batch

__all__ = [
    "Seeds",
    "make_executor",
    "make_spec",
    "parallel_workers",
    "process_block",
    "run_blocks",
    "sweep",
    "use_parallel",
]

#: Gravity-gradient bound on relative acceleration per km of separation,
#: ``2 mu / R_E^3`` (the largest tidal eigenvalue anywhere above the
#: surface) with a 50% margin.
_TIDAL_PER_KM = 1.5 * 2.0 * MU_EARTH_KM3_S2 / R_EARTH_KM**3
#: Separation-independent relative-acceleration floor, km/s^2: more than the
#: entire J2 acceleration at LEO (~1.5e-5) plus any drag difference.
_ACCEL_FLOOR_KM_S2 = 2.0e-5
#: Memory budget for one block's positions and velocities, bytes.
_BLOCK_BYTES = 150_000_000
#: Object-samples below which the sweep runs in-process: spawning workers
#: costs about a second, which only pays off on large catalogs.
_PARALLEL_MIN_WORK = 20_000_000

# J2 is referenced so the floor above stays tied to the constant it covers.
assert _ACCEL_FLOOR_KM_S2 > 1.5 * J2 * MU_EARTH_KM3_S2 / R_EARTH_KM**2


@dataclass(frozen=True)
class Seeds:
    """One entry per close-approach event, as parallel arrays.

    ``index_a < index_b`` are propagator indices; ``sample`` is the event's
    closest grid sample (the TCA seed). ``entered`` is true when any of the
    event's samples lies inside the assigned primary's RTN box.
    """

    index_a: np.ndarray
    index_b: np.ndarray
    sample: np.ndarray
    range_km: np.ndarray
    entered: np.ndarray
    possible: np.ndarray

    def __len__(self) -> int:
        return int(self.index_a.size)


def _no_seeds() -> Seeds:
    empty = np.empty(0, dtype=np.int64)
    return Seeds(empty, empty, empty, np.empty(0), np.empty(0, dtype=bool), np.empty(0, dtype=bool))


def parallel_workers(requested: int | None = None) -> int:
    """Worker count: ``requested``, else ``AEGIS_WORKERS``, else the usable CPUs."""
    if requested is not None:
        return max(1, int(requested))
    env = os.environ.get("AEGIS_WORKERS")
    if env:
        return max(1, int(env))
    counter = getattr(os, "process_cpu_count", None) or os.cpu_count
    return max(1, int(counter() or 1))


# ---------------------------------------------------------------------------
# Static per-object arrays
# ---------------------------------------------------------------------------


def _radial_bands(objects: list[SpaceObject]) -> tuple[np.ndarray, np.ndarray]:
    """Perigee / apogee radii; unknown elements get an unbounded band."""
    perigee = np.full(len(objects), -np.inf)
    apogee = np.full(len(objects), np.inf)
    for index, obj in enumerate(objects):
        if obj.elements is None:
            continue
        low = float(obj.elements.perigee_radius_km)
        high = float(obj.elements.apogee_radius_km)
        perigee[index], apogee[index] = min(low, high), max(low, high)
    return perigee, apogee


def _radius_accel(objects: list[SpaceObject]) -> np.ndarray:
    """Bound on |d^2 r / dt^2| of each orbit's radius, km/s^2.

    Keplerian ``r'' = (mu / r^2) e cos(nu)``, so ``mu e / r_p^2`` bounds it;
    the floor covers J2 and drag. Unknown elements get no bound (inf).
    """
    bounds = np.full(len(objects), np.inf)
    for index, obj in enumerate(objects):
        if obj.elements is None:
            continue
        perigee = max(float(obj.elements.perigee_radius_km), R_EARTH_KM)
        bounds[index] = MU_EARTH_KM3_S2 * float(obj.elements.eccentricity) / perigee**2 + _ACCEL_FLOOR_KM_S2
    return bounds


def _primary_keys(objects: list[SpaceObject]) -> tuple[np.ndarray, np.ndarray]:
    """Maneuverability flags and object-id ranks, as used by ``assign_primary``."""
    maneuverable = np.array([bool(obj.is_maneuverable) for obj in objects], dtype=bool)
    _unique, id_rank = np.unique(np.array([obj.object_id for obj in objects]), return_inverse=True)
    return maneuverable, id_rank.astype(np.int64)


@dataclass(frozen=True)
class _SweepSpec:
    start: datetime
    step_s: float
    n_samples: int
    box_km: tuple[float, float, float]
    pad_km: float
    use_tree: bool
    perigee: np.ndarray
    apogee: np.ndarray
    maneuverable: np.ndarray
    id_rank: np.ndarray
    in_request: np.ndarray
    radius_accel: np.ndarray


# ---------------------------------------------------------------------------
# Per-interval gate
# ---------------------------------------------------------------------------


def _candidates(positions: np.ndarray, usable: np.ndarray, radius_km: float, use_tree: bool):
    """Pairs ``i < j`` of usable objects closer than ``radius_km``."""
    idx = np.flatnonzero(usable)
    if idx.size < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    if use_tree:
        local = cKDTree(positions[idx]).query_pairs(radius_km, output_type="ndarray")
        return idx[local[:, 0]].astype(np.int64), idx[local[:, 1]].astype(np.int64)
    local_i, local_j = np.triu_indices(idx.size, k=1)
    return idx[local_i].astype(np.int64), idx[local_j].astype(np.int64)


def _chord_keep(p0: np.ndarray, p1: np.ndarray, dt: float, sphere_km: float) -> np.ndarray:
    """True where the relative path can enter the sphere during the interval."""
    chord = p1 - p0
    chord2 = np.einsum("ij,ij->i", chord, chord)
    safe = np.where(chord2 > 0.0, chord2, 1.0)
    along = np.clip(np.where(chord2 > 0.0, -np.einsum("ij,ij->i", p0, chord) / safe, 0.0), 0.0, 1.0)
    closest = np.linalg.norm(p0 + chord * along[:, None], axis=1)
    reach = np.maximum(np.linalg.norm(p0, axis=1), np.linalg.norm(p1, axis=1)) * 1.05 + 1.0
    slack = 0.125 * (_TIDAL_PER_KM * reach + _ACCEL_FLOOR_KM_S2) * dt * dt
    return closest - slack <= sphere_km


def _in_box(spec: _SweepSpec, positions, velocities, valid, i, j) -> np.ndarray:
    """Whether either object lies inside the other's RTN box at this sample.

    Used only to split a pair's close run into separate approaches (leaving
    both boxes and coming back is a new approach); acceptance is decided at
    the refined TCA.
    """
    if i.size == 0:
        return np.zeros(0, dtype=bool)
    ok = valid[i] & valid[j]
    half = np.asarray(spec.box_km, dtype=float)
    rot_i, ok_i = rtn_to_eci_batch(positions[i], velocities[i])
    rot_j, ok_j = rtn_to_eci_batch(positions[j], velocities[j])
    from_i = np.einsum("pji,pj->pi", rot_i, positions[j] - positions[i])
    from_j = np.einsum("pji,pj->pi", rot_j, positions[i] - positions[j])
    in_i = ok_i & np.all(np.abs(from_i) <= half, axis=1)
    in_j = ok_j & np.all(np.abs(from_j) <= half, axis=1)
    return ok & (in_i | in_j)


def _radial_possible(spec: _SweepSpec, dt: float, i, j, pos0, pos1, ok1) -> np.ndarray:
    """Whether the pair's radii can come within the box's radial reach during the interval."""
    reach = spec.box_km[0] + float(np.linalg.norm(spec.box_km)) ** 2 / R_EARTH_KM
    dr0 = np.linalg.norm(pos0[j], axis=1) - np.linalg.norm(pos0[i], axis=1)
    if pos1 is None:
        return np.ones(i.size, dtype=bool)
    dr1 = np.linalg.norm(pos1[j], axis=1) - np.linalg.norm(pos1[i], axis=1)
    closest = np.where(dr0 * dr1 <= 0.0, 0.0, np.minimum(np.abs(dr0), np.abs(dr1)))
    curve = (spec.radius_accel[i] + spec.radius_accel[j]) * dt * dt / 8.0
    return ~(ok1[i] & ok1[j]) | (closest - curve <= reach)


def _gate_interval(spec: _SweepSpec, dt: float, pos0, vel0, ok0, pos1, vel1, ok1):
    """Pairs kept for the interval starting at sample ``0``:
    ``(i, j, r0, r1, box0, box1, radial_ok)``.

    ``pos1 is None`` is a single-sample window: no chord exists, so keep what
    the worst-case closing speed could bring into the sphere within ``dt``.
    """
    sphere = float(np.linalg.norm(spec.box_km))
    slack_max = 0.125 * (_TIDAL_PER_KM * 2 * (sphere + MAX_RELATIVE_SPEED_KM_S * dt) + _ACCEL_FLOOR_KM_S2) * dt * dt
    radius = (sphere + MAX_RELATIVE_SPEED_KM_S * dt + slack_max) * 1.05 + 1.0
    i, j = _candidates(pos0, ok0 & spec.in_request, radius, spec.use_tree)
    if i.size == 0:
        return None
    band_ok = (spec.perigee[i] - spec.pad_km <= spec.apogee[j] + spec.pad_km) & (
        spec.perigee[j] - spec.pad_km <= spec.apogee[i] + spec.pad_km
    )
    i, j = i[band_ok], j[band_ok]
    if i.size == 0:
        return None
    p0 = pos0[j] - pos0[i]
    if pos1 is None:
        keep = np.linalg.norm(p0, axis=1) <= radius
    else:
        both_next = ok1[i] & ok1[j]
        keep = ~both_next  # no chord (SGP4 failed at the next sample): keep conservatively
        if np.any(both_next):
            keep[both_next] = _chord_keep(p0[both_next], pos1[j[both_next]] - pos1[i[both_next]], dt, sphere)
    i, j = i[keep], j[keep]
    if i.size == 0:
        return None
    r0 = np.linalg.norm(pos0[j] - pos0[i], axis=1)
    box0 = _in_box(spec, pos0, vel0, ok0, i, j)
    radial_ok = _radial_possible(spec, dt, i, j, pos0, pos1, ok1)
    if pos1 is None:
        return i, j, r0, np.full(i.size, np.inf), box0, np.zeros(i.size, dtype=bool), radial_ok
    next_ok = ok1[i] & ok1[j]
    r1 = np.where(next_ok, np.linalg.norm(pos1[j] - pos1[i], axis=1), np.inf)
    box1 = _in_box(spec, pos1, vel1, ok1, i, j)
    return i, j, r0, r1, box0, box1, radial_ok


# ---------------------------------------------------------------------------
# Rows -> events -> seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Events:
    """Per-block events plus what is needed to stitch those cut by block edges."""

    index_a: np.ndarray
    index_b: np.ndarray
    first: np.ndarray
    last: np.ndarray
    best: np.ndarray
    best_range: np.ndarray
    entered: np.ndarray
    possible: np.ndarray  # some interval passes the radial test
    open_start: np.ndarray  # first sample is the block's leading edge (not the window's)
    open_end: np.ndarray  # last sample is the block's trailing edge (not the window's)
    range_first: np.ndarray
    range_after_first: np.ndarray
    range_before_last: np.ndarray
    range_last: np.ndarray
    box_last: np.ndarray

    def __len__(self) -> int:
        return int(self.index_a.size)


def _events_from_rows(rows: list[tuple], n_objects: int, view: tuple[int, int], n_samples: int) -> _Events | None:
    """Split each pair's kept runs into events at out-of-box range maxima.

    ``rows`` are kept intervals ``(i, j, r0, r1, box0, box1, radial_ok, k)``
    over the inclusive sample range ``view``.
    """
    if not rows:
        return None
    i = np.concatenate([row[0] for row in rows])
    j = np.concatenate([row[1] for row in rows])
    r0 = np.concatenate([row[2] for row in rows])
    r1 = np.concatenate([row[3] for row in rows])
    box0 = np.concatenate([row[4] for row in rows])
    box1 = np.concatenate([row[5] for row in rows])
    radial = np.concatenate([row[6] for row in rows])
    k = np.concatenate([np.full(row[0].size, row[7], dtype=np.int64) for row in rows])

    key = i * np.int64(n_objects) + j
    order = np.lexsort((k, key))
    key, i, j, k, r0, r1, box0, box1, radial = (a[order] for a in (key, i, j, k, r0, r1, box0, box1, radial))
    run_start = np.ones(key.size, dtype=bool)
    run_start[1:] = (key[1:] != key[:-1]) | (k[1:] != k[:-1] + 1)
    run_end = np.r_[run_start[1:], True]
    run = np.cumsum(run_start) - 1

    # Samples of each run: every row's first sample, plus the run's last sample.
    s_run = np.r_[run, run[run_end]]
    s_k = np.r_[k, k[run_end] + 1]
    s_r = np.r_[r0, r1[run_end]]
    s_box = np.r_[box0, box1[run_end]]
    s_radial = np.r_[radial, radial[run_end]]  # each sample: its interval (the last reuses the run's final one)
    s_i = np.r_[i, i[run_end]]
    s_j = np.r_[j, j[run_end]]
    by_sample = np.lexsort((s_k, s_run))
    s_run, s_k, s_r, s_box, s_radial, s_i, s_j = (
        a[by_sample] for a in (s_run, s_k, s_r, s_box, s_radial, s_i, s_j)
    )

    first = np.r_[True, s_run[1:] != s_run[:-1]]
    last = np.r_[s_run[1:] != s_run[:-1], True]
    prev_r = np.r_[np.inf, s_r[:-1]]
    next_r = np.r_[s_r[1:], np.inf]
    prev_r[first] = np.inf
    next_r[last] = np.inf
    # An out-of-box range maximum inside a run separates two approaches.
    split = (s_r >= prev_r) & (s_r > next_r) & ~s_box & ~first & ~last

    event = np.cumsum(first | split) - 1
    starts = np.flatnonzero(first | split)
    ends = np.r_[starts[1:] - 1, s_k.size - 1]
    n_events = starts.size
    entered = np.zeros(n_events, dtype=bool)
    np.logical_or.at(entered, event, s_box)
    possible = np.zeros(n_events, dtype=bool)
    np.logical_or.at(possible, event, s_radial | s_box)
    closest = np.lexsort((s_r, event))
    best_pos = closest[np.r_[True, event[closest][1:] != event[closest][:-1]]]

    view_lo, view_hi = view
    return _Events(
        index_a=s_i[starts],
        index_b=s_j[starts],
        first=s_k[starts],
        last=s_k[ends],
        best=s_k[best_pos],
        best_range=s_r[best_pos],
        entered=entered,
        possible=possible,
        open_start=(s_k[starts] == view_lo) & (view_lo > 0),
        open_end=(s_k[ends] == view_hi) & (view_hi < n_samples - 1),
        range_first=s_r[starts],
        range_after_first=next_r[starts],
        range_before_last=prev_r[ends],
        range_last=s_r[ends],
        box_last=s_box[ends],
    )


def _stitch(parts: list[_Events], n_objects: int) -> Seeds:
    """Join block events; an event cut at a shared edge sample continues
    unless that sample is itself an out-of-box range maximum."""
    parts = [part for part in parts if part is not None and len(part)]
    if not parts:
        return _no_seeds()
    cat = {name: np.concatenate([getattr(part, name) for part in parts]) for name in _Events.__dataclass_fields__}
    key = cat["index_a"] * np.int64(n_objects) + cat["index_b"]
    order = np.lexsort((cat["first"], key))
    cat = {name: values[order] for name, values in cat.items()}
    key = key[order]

    joins = np.zeros(key.size, dtype=bool)  # joins[n]: event n continues event n-1
    if key.size > 1:
        a, b = slice(None, -1), slice(1, None)
        edge = (
            (key[b] == key[a])
            & cat["open_end"][a]
            & cat["open_start"][b]
            & (cat["last"][a] == cat["first"][b])
        )
        # At the shared sample: previous from the earlier block, next from the later one.
        at_edge = cat["range_last"][a]
        edge_is_split = (
            (at_edge >= cat["range_before_last"][a])
            & (at_edge > cat["range_after_first"][b])
            & ~cat["box_last"][a]
        )
        joins[1:] = edge & ~edge_is_split

    group = np.cumsum(~joins) - 1
    n_groups = int(group[-1]) + 1
    entered = np.zeros(n_groups, dtype=bool)
    np.logical_or.at(entered, group, cat["entered"])
    possible = np.zeros(n_groups, dtype=bool)
    np.logical_or.at(possible, group, cat["possible"])
    closest = np.lexsort((cat["best_range"], group))
    pick = closest[np.r_[True, group[closest][1:] != group[closest][:-1]]]
    return Seeds(
        index_a=cat["index_a"][pick],
        index_b=cat["index_b"][pick],
        sample=cat["best"][pick],
        range_km=cat["best_range"][pick],
        entered=entered,
        possible=possible,
    )


def _block_ranges(n_samples: int, n_objects: int, target_blocks: int = 1) -> list[tuple[int, int]]:
    """Inclusive sample ranges ``[k0, k1]``; consecutive blocks share ``k1``.

    Blocks are capped by memory, and split further to reach ``target_blocks``
    so every worker has several to take (uneven blocks otherwise leave most
    workers idle at the end).
    """
    if n_samples <= 1:
        return [(0, 0)]
    per_sample = max(n_objects, 1) * 6 * 8
    span = int(np.clip(_BLOCK_BYTES // per_sample, 16, 1440))
    span = max(16, min(span, int(np.ceil((n_samples - 1) / max(target_blocks, 1)))))
    blocks = []
    k0 = 0
    while k0 < n_samples - 1:
        k1 = min(k0 + span, n_samples - 1)
        blocks.append((k0, k1))
        k0 = k1
    return blocks


def _sweep_block(propagator: Sgp4Propagator, spec: _SweepSpec, block: tuple[int, int]) -> _Events | None:
    view_lo, view_hi = block
    times = np.arange(view_lo, view_hi + 1, dtype=float) * spec.step_s
    grid = propagator._propagate_times(spec.start, times)
    positions, velocities, valid = grid.positions_km, grid.velocities_km_s, grid.valid
    rows: list[tuple] = []
    if view_hi == view_lo:
        kept = _gate_interval(spec, SCREENING_STEP_S, positions[:, 0], velocities[:, 0], valid[:, 0], None, None, None)
        if kept is not None:
            i, j, r0, _r1, box0, _box1, _radial = kept
            # One sample: each kept pair is its own event at that sample.
            return _Events(
                index_a=i, index_b=j, first=np.zeros_like(i), last=np.zeros_like(i), best=np.zeros_like(i),
                best_range=r0, entered=box0, possible=np.ones(i.size, bool),
                open_start=np.zeros(i.size, bool), open_end=np.zeros(i.size, bool),
                range_first=r0, range_after_first=np.full(i.size, np.inf), range_before_last=np.full(i.size, np.inf),
                range_last=r0, box_last=box0,
            )
        return None
    for local in range(view_hi - view_lo):
        kept = _gate_interval(
            spec,
            spec.step_s,
            positions[:, local],
            velocities[:, local],
            valid[:, local],
            positions[:, local + 1],
            velocities[:, local + 1],
            valid[:, local + 1],
        )
        if kept is not None:
            rows.append((*kept, view_lo + local))
    return _events_from_rows(rows, positions.shape[0], (view_lo, view_hi), spec.n_samples)


def _split(events: _Events | None) -> tuple[Seeds, _Events | None]:
    """Events wholly inside the block (ready to refine) and those cut by its edges."""
    if events is None or len(events) == 0:
        return _no_seeds(), None
    open_mask = events.open_start | events.open_end
    closed = ~open_mask
    seeds = Seeds(
        index_a=events.index_a[closed],
        index_b=events.index_b[closed],
        sample=events.best[closed],
        range_km=events.best_range[closed],
        entered=events.entered[closed],
        possible=events.possible[closed],
    )
    if not open_mask.any():
        return seeds, None
    return seeds, _Events(*(getattr(events, name)[open_mask] for name in _Events.__dataclass_fields__))


def process_block(propagator: Sgp4Propagator, spec: _SweepSpec, block: tuple[int, int]):
    """Sweep one block, refine and accept its complete events; return
    ``(accepted rows, events cut by the block's edges)``."""
    from .results import accept

    seeds, open_events = _split(_sweep_block(propagator, spec, block))
    live = seeds.possible | seeds.entered
    rows = accept(propagator, spec, seeds.index_a[live], seeds.index_b[live], seeds.sample[live])
    return rows, open_events


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(objects: list[SpaceObject], spec: _SweepSpec) -> None:
    _WORKER["propagator"] = Sgp4Propagator(objects)
    _WORKER["spec"] = spec


def _worker_process(block: tuple[int, int]):
    return process_block(_WORKER["propagator"], _WORKER["spec"], block)


def use_parallel(n_objects: int, n_samples: int, workers: int | None = None) -> bool:
    """Whether a catalog is large enough for worker processes to pay off."""
    return parallel_workers(workers) > 1 and n_objects * n_samples >= _PARALLEL_MIN_WORK


def make_spec(
    propagator: Sgp4Propagator,
    start: datetime,
    duration_s: float,
    step_s: float,
    box_km: tuple[float, float, float],
    *,
    requested_ids: set[str] | None = None,
    use_tree: bool = True,
    pad_km: float = PERIGEE_APOGEE_PAD_KM,
) -> _SweepSpec:
    """Everything a block needs, in picklable form. ``requested_ids`` limits
    screening to those objects when the propagator holds more."""
    objects = propagator.objects
    perigee, apogee = _radial_bands(objects)
    maneuverable, id_rank = _primary_keys(objects)
    if requested_ids is None:
        in_request = np.ones(len(objects), dtype=bool)
    else:
        in_request = np.array([obj.object_id in requested_ids for obj in objects], dtype=bool)
    return _SweepSpec(
        start=start,
        step_s=float(step_s),
        n_samples=int(np.floor(duration_s / step_s)) + 1,
        box_km=tuple(float(v) for v in box_km),
        pad_km=float(pad_km),
        use_tree=use_tree,
        perigee=perigee,
        apogee=apogee,
        maneuverable=maneuverable,
        id_rank=id_rank,
        in_request=in_request,
        radius_accel=_radius_accel(objects),
    )


def make_executor(objects: list[SpaceObject], spec: _SweepSpec, n_workers: int) -> ProcessPoolExecutor:
    """Spawned workers, each holding its own propagator for these objects."""
    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(objects, spec),
    )


def run_blocks(
    propagator: Sgp4Propagator,
    spec: _SweepSpec,
    *,
    executor: ProcessPoolExecutor | None = None,
    workers: int = 1,
):
    """Every block's accepted rows, plus seeds for the events stitched across block edges."""
    from .results import Rows

    n_objects = len(propagator.objects)
    blocks = _block_ranges(spec.n_samples, n_objects, target_blocks=4 * workers if executor else 1)
    if executor is None:
        results = [process_block(propagator, spec, block) for block in blocks]
    else:
        results = list(executor.map(_worker_process, blocks))
    rows = Rows.concat([part for part, _open in results])
    stitched = _stitch([open_events for _rows, open_events in results], n_objects)
    return rows, stitched


def sweep(propagator: Sgp4Propagator, spec: _SweepSpec) -> Seeds:
    """Detection only: one seed per close-approach event of every pair that can enter the box."""
    n_objects = len(propagator.objects)
    parts = [_sweep_block(propagator, spec, block) for block in _block_ranges(spec.n_samples, n_objects)]
    return _stitch(parts, n_objects)
