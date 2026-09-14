"""Exact linear relative dynamics for impulsive fleet maneuvers.

Everything the optimizer knows about physics enters through one matrix: the
Clohessy-Wiltshire position-from-impulse block

.. math::

    \\Phi(n, \\sigma) =
    \\begin{bmatrix}
      s/n            & 2(1-c)/n          & 0   \\\\
      2(c-1)/n       & (4s - 3 n\\sigma)/n & 0   \\\\
      0              & 0                 & s/n
    \\end{bmatrix}

with :math:`s = \\sin(n\\sigma)`, :math:`c = \\cos(n\\sigma)`. Rows are
(radial, transverse, normal) *displacement*; columns are (radial,
transverse, normal) *impulse*. Column 2 row 2 is exactly the expression
:func:`aegis.maneuver.cw.along_track_response_km` uses, so the new
three-axis machinery reduces to the existing along-track code rather than
competing with it.

Two structural facts drive every design decision downstream:

* The (2,2) entry contains the secular term :math:`-3\\sigma`. Only a
  tangential impulse buys displacement that grows without bound in lead
  time; radial and cross-track impulses produce bounded periodic motion of
  amplitude at most :math:`2\\,\\Delta v/n`. That asymmetry is why fuel cost
  scales as :math:`1/(3 n \\sigma)` and why lead time, not propellant, is
  the scarce resource.
* :math:`\\lVert\\Phi(n,\\sigma)\\rVert_2` grows asymptotically as
  :math:`3\\sigma` but is **not** monotone. Its growth rate is
  :math:`3 - 4\\cos(n\\sigma)`, which turns negative whenever
  :math:`\\cos(n\\sigma) > 3/4`, so the norm dips slightly just after every
  whole orbit of lead time. The first draft of this module asserted
  monotonicity; a 400-point sweep disproved it. The reachability bound
  therefore needs a genuine envelope, and :func:`cw_impulse_norm_bound`
  supplies one in closed form:

  .. math::

      \\lVert\\Phi(n,\\sigma)\\rVert_2 \\le
      \\lVert\\Phi(n,\\sigma)\\rVert_F \\le
      \\frac{\\sqrt{(3n\\sigma + 4)^2 + 34}}{n} =: M(n,\\sigma)

  because :math:`2\\sin^2 \\le 2`, :math:`8(1-\\cos)^2 \\le 32`, and
  :math:`|4\\sin(n\\sigma) - 3n\\sigma| \\le 4 + 3n\\sigma`. :math:`M` is
  manifestly increasing in :math:`\\sigma`, is asymptotic to
  :math:`3\\sigma`, and over-estimates by at most about :math:`7/n`
  seconds -- roughly 7 km of extra screening gate at a 1 m/s budget in LEO.
  A tighter sampled envelope is available via ``mode="sampled"``, made
  rigorous by the Lipschitz constant :data:`CW_NORM_LIPSCHITZ`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import ensure_utc, seconds_between, shift
from .errors import GridError

__all__ = [
    "cw_impulse_matrix",
    "cw_impulse_rate_matrix",
    "cw_impulse_norm",
    "cw_impulse_norm_bound",
    "CW_NORM_LIPSCHITZ",
    "BurnGrid",
    "build_burn_grid",
    "displacement_operator",
    "eci_displacement_operator",
    "AXIS_NAMES",
]

#: Index order of the three impulse axes everywhere in this package.
AXIS_NAMES = ("R", "T", "N")

#: Below this mean motion the rectilinear limit is used.
_N_FLOOR = 1e-15

#: Lipschitz constant of ``sigma -> ||Phi(n, sigma)||_2`` with respect to
#: ``n * sigma``. The derivative matrix has entries
#: ``[[c, 2s, 0], [-2s, 4c-3, 0], [0, 0, c]]`` whose squared Frobenius norm is
#: ``10c^2 - 24c + 17``, maximised at ``c = -1`` giving 51. Used to convert a
#: sampled running maximum into a rigorous envelope.
CW_NORM_LIPSCHITZ = 51.0**0.5


def cw_impulse_matrix(n_rad_s: float, lead_s: float) -> np.ndarray:
    """Position response to a unit impulse applied ``lead_s`` seconds earlier.

    Returns a ``(3, 3)`` matrix mapping an RTN impulse to the RTN
    displacement it produces after ``lead_s`` seconds of coasting.

    A non-positive ``lead_s`` returns zeros: an impulse cannot displace a
    spacecraft before it is applied, and the optimizer relies on that to
    zero out burn slots that fall after the epoch being constrained.
    """
    sigma = float(lead_s)
    if not sigma > 0.0:
        return np.zeros((3, 3), dtype=float)

    n = float(n_rad_s)
    if abs(n) < _N_FLOOR:
        return sigma * np.eye(3, dtype=float)

    nt = n * sigma
    s = np.sin(nt)
    c = np.cos(nt)
    return np.array(
        [
            [s / n, 2.0 * (1.0 - c) / n, 0.0],
            [2.0 * (c - 1.0) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, s / n],
        ],
        dtype=float,
    )


def cw_impulse_rate_matrix(n_rad_s: float, lead_s: float) -> np.ndarray:
    """Velocity response to a unit impulse applied ``lead_s`` seconds earlier.

    The exact time derivative of :func:`cw_impulse_matrix`, differentiated
    term by term in ``sigma``. It exists because locating a *minimum* of the
    perturbed separation needs the relative velocity, not just the relative
    position: a pair crossing at several kilometres per second sweeps hundreds
    of kilometres between two grid samples, so a minimum can sit between them
    with both neighbours far away and no sign change in the sampled
    separation at all. The range rate changes sign there regardless, which is
    what makes the minimum findable.

    A non-positive ``lead_s`` returns zeros, matching
    :func:`cw_impulse_matrix`.
    """
    sigma = float(lead_s)
    if not sigma > 0.0:
        return np.zeros((3, 3), dtype=float)

    n = float(n_rad_s)
    if abs(n) < _N_FLOOR:
        return np.eye(3, dtype=float)

    nt = n * sigma
    s = np.sin(nt)
    c = np.cos(nt)
    return np.array(
        [
            [c, 2.0 * s, 0.0],
            [-2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )


def cw_impulse_norm(n_rad_s: float, lead_s: float) -> float:
    """Exact spectral norm of :func:`cw_impulse_matrix`.

    Asymptotic to ``3 * lead_s`` but **not** monotone -- see the module
    docstring. Use :func:`cw_impulse_norm_bound` wherever a bound over an
    interval of lead times is needed.
    """
    matrix = cw_impulse_matrix(n_rad_s, lead_s)
    if not matrix.any():
        return 0.0
    return float(np.linalg.norm(matrix, 2))


def cw_impulse_norm_bound(
    n_rad_s: float,
    lead_s: float,
    *,
    mode: str = "analytic",
    samples_per_orbit: int = 256,
) -> float:
    """Monotone upper envelope of ``max_{0 <= u <= lead_s} ||Phi(n, u)||_2``.

    ``mode="analytic"`` returns the closed form ``M(n, sigma)`` from the
    module docstring: cheap, rigorous, monotone, and loose by at most about
    ``7 / n`` seconds.

    ``mode="sampled"`` returns a sampled running maximum inflated by the
    Lipschitz slack ``CW_NORM_LIPSCHITZ * delta / 2``, which is also
    rigorous and much tighter at short lead times, at the cost of
    ``samples_per_orbit`` norm evaluations per orbit of lead.

    Both modes return 0 for a non-positive ``lead_s``.
    """
    sigma = float(lead_s)
    if not sigma > 0.0:
        return 0.0
    n = abs(float(n_rad_s))
    if n < _N_FLOOR:
        return sigma

    analytic = float(np.sqrt((3.0 * n * sigma + 4.0) ** 2 + 34.0) / n)
    if mode == "analytic":
        return analytic
    if mode != "sampled":
        raise ValueError(f"unknown bound mode {mode!r}; use 'analytic' or 'sampled'")

    if samples_per_orbit < 8:
        raise ValueError("samples_per_orbit must be at least 8")
    period_s = 2.0 * np.pi / n
    count = max(16, int(np.ceil(samples_per_orbit * sigma / period_s)) + 1)
    grid = np.linspace(0.0, sigma, count)
    delta = float(grid[1] - grid[0])
    running = max(cw_impulse_norm(n, float(u)) for u in grid)
    sampled = running + CW_NORM_LIPSCHITZ * delta / 2.0
    return float(min(analytic, sampled))


@dataclass(frozen=True)
class BurnGrid:
    """Fixed set of candidate burn epochs, and the column layout they imply.

    The decision vector ``x`` is laid out satellite-major, then slot, then
    axis. With ``axes == 1`` only the transverse axis is represented, which
    reproduces the along-track-only decision space of
    :func:`aegis.maneuver.plan_maneuvers`.
    """

    satellite_ids: tuple[str, ...]
    epochs: dict[str, tuple[datetime, ...]]
    mean_motion: dict[str, float]
    axes: int = 3
    excluded: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axes not in (1, 3):
            raise GridError(f"axes must be 1 or 3, got {self.axes}")
        if len(set(self.satellite_ids)) != len(self.satellite_ids):
            raise GridError("duplicate satellite id in burn grid")
        for sat_id in self.satellite_ids:
            if sat_id not in self.epochs:
                raise GridError(f"no burn epochs supplied for {sat_id!r}")
            if sat_id not in self.mean_motion:
                raise GridError(f"no mean motion supplied for {sat_id!r}")
            if not self.mean_motion[sat_id] > 0.0:
                raise GridError(f"mean motion for {sat_id!r} must be positive")
            epochs = self.epochs[sat_id]
            if not epochs:
                raise GridError(f"empty burn epoch tuple for {sat_id!r}")
            if any(b <= a for a, b in zip(epochs, epochs[1:])):
                raise GridError(f"burn epochs for {sat_id!r} are not strictly ascending")
        for sat_id in self.epochs:
            if sat_id not in self.satellite_ids:
                raise GridError(f"epochs supplied for unlisted satellite {sat_id!r}")

    @property
    def axis_labels(self) -> tuple[str, ...]:
        return AXIS_NAMES if self.axes == 3 else ("T",)

    @property
    def slot_counts(self) -> dict[str, int]:
        return {sat_id: len(self.epochs[sat_id]) for sat_id in self.satellite_ids}

    @property
    def n_vars(self) -> int:
        return self.axes * sum(len(self.epochs[s]) for s in self.satellite_ids)

    @property
    def max_slots(self) -> int:
        if not self.satellite_ids:
            return 0
        return max(len(self.epochs[s]) for s in self.satellite_ids)

    def _offset(self, satellite_id: str) -> int:
        offset = 0
        for sat_id in self.satellite_ids:
            if sat_id == satellite_id:
                return offset
            offset += self.axes * len(self.epochs[sat_id])
        raise GridError(f"{satellite_id!r} is not in this burn grid")

    def index(self, satellite_id: str, slot: int, axis: int = 1) -> int:
        """Column of ``x`` holding one impulse component.

        ``axis`` is 0/1/2 for radial/transverse/normal. With ``axes == 1``
        only ``axis == 1`` is addressable.
        """
        epochs = self.epochs.get(satellite_id)
        if epochs is None:
            raise GridError(f"{satellite_id!r} is not in this burn grid")
        if not 0 <= slot < len(epochs):
            raise GridError(f"slot {slot} out of range for {satellite_id!r}")
        if self.axes == 1:
            if axis != 1:
                raise GridError("an along-track-only grid can only address axis 1")
            return self._offset(satellite_id) + slot
        if not 0 <= axis < 3:
            raise GridError(f"axis {axis} out of range")
        return self._offset(satellite_id) + slot * 3 + axis

    def columns(self, satellite_id: str) -> list[int]:
        """Every column owned by one satellite, in layout order."""
        offset = self._offset(satellite_id)
        width = self.axes * len(self.epochs[satellite_id])
        return list(range(offset, offset + width))

    def slot_epochs(self, satellite_id: str) -> tuple[datetime, ...]:
        epochs = self.epochs.get(satellite_id)
        if epochs is None:
            raise GridError(f"{satellite_id!r} is not in this burn grid")
        return epochs

    def earliest_epoch(self, satellite_id: str) -> datetime:
        return self.slot_epochs(satellite_id)[0]

    def column_labels(self) -> list[str]:
        labels: list[str] = []
        for sat_id in self.satellite_ids:
            for slot in range(len(self.epochs[sat_id])):
                for axis in self.axis_labels:
                    labels.append(f"dv:{sat_id}:{slot}:{axis}")
        return labels

    def has(self, satellite_id: str) -> bool:
        return satellite_id in self.epochs


def build_burn_grid(
    objects: list[SpaceObject],
    earliest_tca: dict[str, datetime] | dict[str, list[datetime]],
    *,
    burn_slots: int,
    min_lead_orbits: float,
    axes: int = 3,
    now: datetime | None = None,
) -> BurnGrid:
    """Lay out candidate burn epochs backwards from an actionable TCA.

    For each maneuverable satellite with a known mean motion, the reference
    TCA is the **earliest one that still leaves room to act** -- the earliest
    involved TCA satisfying ``tca - min_lead_orbits * period >= now``. The
    first slot sits ``min_lead_orbits`` periods before it and later slots step
    back half an orbit at a time, stopping at ``now``.

    Taking the earliest TCA unconditionally, which is what the first version
    did, throws away satellites that have an unreachable early encounter and a
    perfectly actionable later one. That is not a rare corner: a repeating
    crossing pair generates an encounter every orbit, so screening a longer
    window makes the *earliest* TCA arbitrarily close to the window start and
    the satellite would drop out of the grid entirely -- measured on the first
    benchmark sweep as every solving planner returning an empty plan.

    ``earliest_tca`` accepts either one TCA per satellite or a list of the
    TCAs it is involved in; a bare datetime is treated as a one-element list.
    A satellite with no actionable TCA is excluded, with the reason recorded
    in :attr:`BurnGrid.excluded`.
    """
    if burn_slots < 1:
        raise GridError("burn_slots must be at least 1")
    if min_lead_orbits < 0.0:
        raise GridError("min_lead_orbits must be non-negative")

    now = ensure_utc(now) if now is not None else None
    by_id = {obj.object_id: obj for obj in objects}

    satellite_ids: list[str] = []
    epochs: dict[str, tuple[datetime, ...]] = {}
    mean_motion: dict[str, float] = {}
    excluded: dict[str, str] = {}

    for sat_id in sorted(earliest_tca):
        obj = by_id.get(sat_id)
        if obj is None:
            excluded[sat_id] = "object not present in the supplied catalog"
            continue
        if not obj.is_maneuverable:
            excluded[sat_id] = "object is not maneuverable"
            continue
        if obj.elements is None or obj.elements.mean_motion_rev_per_day <= 0.0:
            excluded[sat_id] = "no mean motion available"
            continue

        n_rad_s = obj.elements.mean_motion_rad_s
        period_s = 2.0 * np.pi / n_rad_s
        lead_s = min_lead_orbits * period_s

        entry = earliest_tca[sat_id]
        tcas = [ensure_utc(entry)] if isinstance(entry, datetime) else sorted(
            ensure_utc(value) for value in entry
        )
        if not tcas:
            excluded[sat_id] = "no involved conjunction"
            continue

        if now is None:
            reference = tcas[0]
        else:
            actionable = [tca for tca in tcas if shift(tca, -lead_s) >= now]
            if not actionable:
                excluded[sat_id] = (
                    "no involved conjunction leaves "
                    f"{min_lead_orbits:g} orbit(s) of lead after the planning epoch"
                )
                continue
            reference = actionable[0]

        candidates = [
            shift(reference, -(lead_s + slot * 0.5 * period_s)) for slot in range(burn_slots)
        ]
        if now is not None:
            candidates = [epoch for epoch in candidates if epoch >= now]
        if not candidates:
            excluded[sat_id] = "all candidate burn slots precede the planning epoch"
            continue

        candidates.sort()
        satellite_ids.append(sat_id)
        epochs[sat_id] = tuple(candidates)
        mean_motion[sat_id] = float(n_rad_s)

    return BurnGrid(
        satellite_ids=tuple(satellite_ids),
        epochs=epochs,
        mean_motion=mean_motion,
        axes=axes,
        excluded=excluded,
    )


def displacement_operator(
    grid: BurnGrid,
    satellite_id: str,
    epoch: datetime,
) -> np.ndarray:
    """``Psi_i(t)``: displacement of one satellite as a linear map of ``x``.

    Returns a ``(3, grid.n_vars)`` matrix. Only the columns owned by
    ``satellite_id`` are non-zero, and a slot whose epoch is at or after
    ``epoch`` contributes nothing.
    """
    operator = np.zeros((3, grid.n_vars), dtype=float)
    if not grid.has(satellite_id):
        return operator

    n_rad_s = grid.mean_motion[satellite_id]
    epoch = ensure_utc(epoch)
    for slot, burn_epoch in enumerate(grid.slot_epochs(satellite_id)):
        lead_s = seconds_between(burn_epoch, epoch)
        block = cw_impulse_matrix(n_rad_s, lead_s)
        if not block.any():
            continue
        if grid.axes == 1:
            column = grid.index(satellite_id, slot, 1)
            operator[:, column] = block[:, 1]
        else:
            for axis in range(3):
                operator[:, grid.index(satellite_id, slot, axis)] = block[:, axis]
    return operator


def eci_displacement_operator(
    grid: BurnGrid,
    satellite_id: str,
    epoch: datetime,
    state: StateVector,
) -> np.ndarray:
    """:func:`displacement_operator` rotated into ECI at ``epoch``.

    The CW displacement is expressed in the satellite's own co-rotating RTN
    frame; conjunction geometry is expressed in ECI. Rotating here rather
    than approximating the frame difference with a scalar alignment term is
    what makes the B-plane sensitivity in :mod:`aegis.fleetopt.bplane`
    correct for crossing geometries.
    """
    rotation = np.asarray(state.rtn_to_eci(), dtype=float)
    return rotation @ displacement_operator(grid, satellite_id, epoch)
