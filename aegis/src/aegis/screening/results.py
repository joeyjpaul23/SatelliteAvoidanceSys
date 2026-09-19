"""Columnar screening results, and the per-block refine-and-accept step.

A full-fleet screen can report millions of conjunctions: 1.8 M inside the
Starlink box for 14,067 objects over three days. As ``Conjunction`` objects
that is several GB. :class:`ConjunctionTable` keeps them as numpy columns
(about 180 bytes a row) and builds objects only on request.

:func:`accept` turns sweep seeds into table rows. It runs where the seeds
were found, inside each worker for its own time block, so only accepted
rows cross process boundaries:

* batched TCA refinement (:mod:`~aegis.screening.batch_tca`), with the
  scalar path for the rare rows it flags;
* the box test at TCA, vectorised, as SpaceX's Space Safety screening
  defines it: a close approach is a local minimum of relative distance that
  falls inside either object's box, each box in that object's own RTN frame
  (radial along position, normal along orbital angular momentum, transverse
  completing the triad). The box turns with the orbit, not the satellite's
  body attitude, which no public data carries. Membership is judged at the
  refined TCA only, so the result does not depend on the grid step.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

import numpy as np

from ..core.conjunction import Conjunction
from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import seconds_between, shift
from ..propagation.propagator import PropagationError, Sgp4Propagator
from .batch_tca import refine_batch
from .errors import ScreeningError
from .geometry import rtn_to_eci_batch

__all__ = ["ConjunctionTable", "Rows", "accept", "box_membership", "empty_rows"]

#: Rows per batched-refinement chunk: bounds the Hermite scan's memory
#: (rows x 241 samples x 3 floats) to about 60 MB.
_REFINE_CHUNK = 10_000
#: Two refinements of one pair closer together than this are one approach.
SAME_APPROACH_S = 1.0


@dataclass(frozen=True)
class Rows:
    """Accepted conjunctions as parallel arrays (propagator indices, primary first)."""

    primary: np.ndarray
    secondary: np.ndarray
    tca_s: np.ndarray
    primary_position_km: np.ndarray
    primary_velocity_km_s: np.ndarray
    secondary_position_km: np.ndarray
    secondary_velocity_km_s: np.ndarray
    relative_position_rtn_km: np.ndarray
    relative_velocity_rtn_km_s: np.ndarray
    miss_distance_km: np.ndarray
    relative_speed_km_s: np.ndarray
    in_primary_box: np.ndarray
    in_secondary_box: np.ndarray

    def __len__(self) -> int:
        return int(self.primary.size)

    def take(self, index: np.ndarray) -> Rows:
        return Rows(*(getattr(self, f.name)[index] for f in fields(self)))

    @staticmethod
    def concat(parts: list[Rows]) -> Rows:
        parts = [part for part in parts if part is not None and len(part)]
        if not parts:
            return empty_rows()
        return Rows(*(np.concatenate([getattr(part, f.name) for part in parts]) for f in fields(Rows)))


def empty_rows() -> Rows:
    ints = np.empty(0, dtype=np.int64)
    scalars = np.empty(0)
    vectors = np.empty((0, 3))
    flags = np.empty(0, dtype=bool)
    return Rows(ints, ints, scalars, vectors, vectors, vectors, vectors, vectors, vectors, scalars, scalars, flags, flags)


def box_membership(primary_pos, primary_vel, secondary_pos, secondary_vel, box_km):
    """Each object's view of the other through its own RTN-aligned box.

    Returns ``(in_primary_box, in_secondary_box, relative_position_rtn,
    relative_velocity_rtn)``; the relative state is the secondary's, in the
    primary's RTN frame (the convention every downstream consumer uses).
    """
    half = np.asarray(box_km, dtype=float)
    p_rot, p_ok = rtn_to_eci_batch(primary_pos, primary_vel)
    s_rot, s_ok = rtn_to_eci_batch(secondary_pos, secondary_vel)
    rel_rtn = np.einsum("pji,pj->pi", p_rot, secondary_pos - primary_pos)
    rel_vel_rtn = np.einsum("pji,pj->pi", p_rot, secondary_vel - primary_vel)
    seen_by_secondary = np.einsum("pji,pj->pi", s_rot, primary_pos - secondary_pos)
    in_primary = p_ok & np.all(np.abs(rel_rtn) <= half, axis=1)
    in_secondary = s_ok & np.all(np.abs(seen_by_secondary) <= half, axis=1)
    return in_primary, in_secondary, rel_rtn, rel_vel_rtn


def _primary_first(spec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True where ``a`` is the assigned primary (maneuverable, else lower id)."""
    man_a, man_b = spec.maneuverable[a], spec.maneuverable[b]
    return (man_a & ~man_b) | ((man_a == man_b) & (spec.id_rank[a] <= spec.id_rank[b]))


def accept(propagator: Sgp4Propagator, spec, index_a, index_b, sample) -> Rows:
    """Refine every seed; keep the approaches whose TCA lies inside either object's box."""
    parts = [
        _accept_chunk(
            propagator,
            spec,
            index_a[lo : lo + _REFINE_CHUNK],
            index_b[lo : lo + _REFINE_CHUNK],
            sample[lo : lo + _REFINE_CHUNK],
        )
        for lo in range(0, np.asarray(index_a).size, _REFINE_CHUNK)
    ]
    return Rows.concat(parts)


def _accept_chunk(propagator: Sgp4Propagator, spec, a, b, sample) -> Rows:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    guess_s = np.asarray(sample, dtype=float) * spec.step_s
    batch = refine_batch(propagator, a, b, guess_s, spec.start)
    tca_s = batch.tca_s.copy()
    pos_a, vel_a = batch.position_a.copy(), batch.velocity_a.copy()
    pos_b, vel_b = batch.position_b.copy(), batch.velocity_b.copy()
    if batch.fallback.any():
        from .tca import refine_tca_states

        for row in np.flatnonzero(batch.fallback):
            try:
                tca, state_a, state_b = refine_tca_states(
                    propagator, int(a[row]), int(b[row]), shift(spec.start, float(guess_s[row]))
                )
            except PropagationError as error:
                raise ScreeningError(str(error)) from error
            tca_s[row] = seconds_between(spec.start, tca)
            pos_a[row], vel_a[row] = state_a.position_km, state_a.velocity_km_s
            pos_b[row], vel_b[row] = state_b.position_km, state_b.velocity_km_s

    a_first = _primary_first(spec, a, b)
    primary = np.where(a_first, a, b)
    secondary = np.where(a_first, b, a)
    p_pos = np.where(a_first[:, None], pos_a, pos_b)
    p_vel = np.where(a_first[:, None], vel_a, vel_b)
    s_pos = np.where(a_first[:, None], pos_b, pos_a)
    s_vel = np.where(a_first[:, None], vel_b, vel_a)
    in_p, in_s, rel_rtn, rel_vel_rtn = box_membership(p_pos, p_vel, s_pos, s_vel, spec.box_km)
    keep = in_p | in_s
    return Rows(
        primary=primary[keep],
        secondary=secondary[keep],
        tca_s=tca_s[keep],
        primary_position_km=p_pos[keep],
        primary_velocity_km_s=p_vel[keep],
        secondary_position_km=s_pos[keep],
        secondary_velocity_km_s=s_vel[keep],
        relative_position_rtn_km=rel_rtn[keep],
        relative_velocity_rtn_km_s=rel_vel_rtn[keep],
        miss_distance_km=np.linalg.norm(p_pos - s_pos, axis=1)[keep],
        relative_speed_km_s=np.linalg.norm(p_vel - s_vel, axis=1)[keep],
        in_primary_box=in_p[keep],
        in_secondary_box=in_s[keep],
    )


def dedupe(rows: Rows, spec) -> Rows:
    """One row per approach: refinements of a pair within ``SAME_APPROACH_S`` merge,
    keeping the smallest miss. Output is ordered by TCA."""
    if len(rows) == 0:
        return rows
    key_p, key_s = spec.id_rank[rows.primary], spec.id_rank[rows.secondary]
    order = np.lexsort((rows.miss_distance_km, rows.tca_s, key_s, key_p))
    rows = rows.take(order)
    key_p, key_s = key_p[order], key_s[order]
    same_pair = np.r_[False, (key_p[1:] == key_p[:-1]) & (key_s[1:] == key_s[:-1])]
    close = np.r_[False, np.diff(rows.tca_s) < SAME_APPROACH_S]
    group = np.cumsum(~(same_pair & close)) - 1
    by_miss = np.lexsort((rows.miss_distance_km, group))
    first = by_miss[np.r_[True, group[by_miss][1:] != group[by_miss][:-1]]]
    rows = rows.take(first)
    return rows.take(np.argsort(rows.tca_s, kind="stable"))


@dataclass(frozen=True)
class ConjunctionTable:
    """Screening results as columns: one row per approach whose TCA is inside either object's box.

    ``rows`` indexes into ``catalog`` (the propagator's objects). Times are
    seconds from ``start``. :meth:`to_conjunctions` builds the objects the
    rest of AEGIS consumes; use the columns directly at catalog scale.
    """

    catalog: list[SpaceObject]
    start: datetime
    window_end: datetime
    rows: Rows

    def __len__(self) -> int:
        return len(self.rows)

    def to_conjunctions(self) -> list[Conjunction]:
        from .tca import _build_conjunction

        rows = self.rows
        built: list[Conjunction] = []
        for row in range(len(rows)):
            tca = shift(self.start, float(rows.tca_s[row]))
            primary_state = StateVector(
                epoch=tca,
                position_km=rows.primary_position_km[row],
                velocity_km_s=rows.primary_velocity_km_s[row],
                frame="TEME",
            )
            secondary_state = StateVector(
                epoch=tca,
                position_km=rows.secondary_position_km[row],
                velocity_km_s=rows.secondary_velocity_km_s[row],
                frame="TEME",
            )
            conjunction = _build_conjunction(
                self.catalog[int(rows.primary[row])],
                self.catalog[int(rows.secondary[row])],
                primary_state,
                secondary_state,
                tca,
            )
            conjunction.screening_window_start = self.start
            conjunction.screening_window_end = self.window_end
            conjunction.metadata["screening_box"] = (
                "both" if rows.in_primary_box[row] and rows.in_secondary_box[row]
                else "primary" if rows.in_primary_box[row] else "secondary"
            )
            built.append(conjunction)
        built.sort(key=lambda item: (item.tca, item.conjunction_id))
        return built
