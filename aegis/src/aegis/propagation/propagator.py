"""Orbit propagation.

The screening engine needs positions for thousands of objects at tens of
thousands of epochs. Doing that one call at a time in Python is hopeless, so
propagation is vectorised: all objects are advanced across the entire time
grid in a handful of array operations.

Frame note
----------
SGP4 emits TEME-of-date, and this system keeps everything in TEME rather than
converting to a standard inertial frame. That is deliberate. Conjunction
assessment depends only on the *relative* geometry of two objects, which is
frame-consistent, so a conversion would cost time and introduce error while
changing no answer. Frames only need reconciling if this system is ever fed
externally-produced ephemeris in a different frame -- which is exactly why
:class:`~aegis.core.state.StateVector` carries a frame label.

Theory note
-----------
Two-line elements are Brouwer/Kozai *mean* elements and are meaningful only
when paired with SGP4. Converting them to osculating elements and handing them
to a numerical integrator produces worse results, not better: the element set
and the theory are a matched pair, and the theory's approximations are what
the elements were fitted to absorb.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sgp4.api import WGS72, SatrecArray, Satrec

from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import JULIAN_DATE_UNIX_EPOCH, ensure_utc, shift

__all__ = ["PropagationGrid", "Sgp4Propagator", "PropagationError"]


class PropagationError(RuntimeError):
    """Raised when propagation fails for reasons the caller must handle."""


@dataclass
class PropagationGrid:
    """Positions and velocities for many objects across a shared time grid.

    A single dense array rather than per-object ephemeris objects, because
    consumers want *all objects at one epoch* -- the opposite of the access
    pattern per-object storage optimises for.

    Attributes
    ----------
    object_ids
        Row order of the arrays.
    times_s
        Seconds since ``reference_epoch``, shape ``(n_times,)``.
    positions_km, velocities_km_s
        Shape ``(n_objects, n_times, 3)``.
    valid
        Boolean mask, shape ``(n_objects, n_times)``. False where SGP4
        reported an error -- typically a decayed object or an element set
        propagated far past its useful span. Invalid samples must be excluded
        from screening rather than silently treated as position zero.
    """

    object_ids: list[str]
    times_s: np.ndarray
    positions_km: np.ndarray
    velocities_km_s: np.ndarray
    valid: np.ndarray
    reference_epoch: datetime

    @property
    def n_objects(self) -> int:
        return len(self.object_ids)

    @property
    def n_times(self) -> int:
        return len(self.times_s)

    def index_of(self, object_id: str) -> int:
        return self.object_ids.index(object_id)

    def epoch_at(self, time_index: int) -> datetime:
        return shift(self.reference_epoch, float(self.times_s[time_index]))

    def state(self, object_index: int, time_index: int) -> StateVector:
        """One object at one epoch, as a :class:`StateVector`."""
        return StateVector(
            epoch=self.epoch_at(time_index),
            position_km=self.positions_km[object_index, time_index],
            velocity_km_s=self.velocities_km_s[object_index, time_index],
            frame="TEME",
        )


class Sgp4Propagator:
    """Vectorised SGP4 propagation for a catalog of objects.

    Objects are initialised once and reused across every time block, which is
    what makes blocked screening cheap.
    """

    def __init__(self, objects: list[SpaceObject]) -> None:
        """
        Parameters
        ----------
        objects
            Catalog entries carrying either TLE lines or mean elements.

        Raises
        ------
        PropagationError
            If any object cannot be initialised. Failing loudly is
            deliberate: a silently dropped object is a silently missed
            conjunction.
        """
        self.objects = objects
        self.object_ids = [obj.object_id for obj in objects]
        self._satrecs = [self._build_satrec(obj) for obj in objects]
        self._array = SatrecArray(self._satrecs)

    @staticmethod
    def _build_satrec(obj: SpaceObject) -> Satrec:
        """Initialise an SGP4 record from TLE lines or mean elements."""
        if obj.tle_line1 and obj.tle_line2:
            try:
                return Satrec.twoline2rv(obj.tle_line1, obj.tle_line2)
            except Exception as error:  # noqa: BLE001 - sgp4 raises bare exceptions
                raise PropagationError(
                    f"could not initialise SGP4 for {obj.object_id} from TLE lines: {error}"
                ) from error

        if obj.elements is None:
            raise PropagationError(
                f"object {obj.object_id} has neither TLE lines nor mean elements"
            )

        elements = obj.elements
        satrec = Satrec()

        # sgp4init expects the epoch as days since 1949-12-31 00:00 UT.
        epoch_days = (
            elements.epoch.timestamp() / 86400.0
            + JULIAN_DATE_UNIX_EPOCH
            - 2433281.5
        )

        # Mean motion derivatives carry the TLE convention (n-dot/2 and
        # n-double-dot/6) and radians-per-minute units that sgp4init expects.
        satrec.sgp4init(
            WGS72,                                 # WGS-72, the gravity model SGP4 was fitted with
            "i",                                   # improved mode
            int(obj.object_id) if obj.object_id.isdigit() else 0,
            epoch_days,
            elements.bstar,
            elements.mean_motion_dot * (2 * np.pi) / (1440.0**2),
            elements.mean_motion_ddot * (2 * np.pi) / (1440.0**3),
            elements.eccentricity,
            np.radians(elements.arg_perigee_deg),
            np.radians(elements.inclination_deg),
            np.radians(elements.mean_anomaly_deg),
            elements.mean_motion_rev_per_day * (2 * np.pi) / 1440.0,
            np.radians(elements.raan_deg),
        )
        return satrec

    def _propagate_times(self, start: datetime, times_s: np.ndarray) -> PropagationGrid:
        """Propagate every object at ``times_s`` offsets from ``start``."""
        times_s = np.asarray(times_s, dtype=float)
        n_steps = int(times_s.size)

        # SGP4 takes Julian date split into whole days plus fraction, which
        # preserves precision far better than a single float across a week.
        start_julian = JULIAN_DATE_UNIX_EPOCH + start.timestamp() / 86400.0
        whole_days = np.full(n_steps, np.floor(start_julian))
        fractions = (start_julian - np.floor(start_julian)) + times_s / 86400.0

        error_codes, positions, velocities = self._array.sgp4(whole_days, fractions)

        return PropagationGrid(
            object_ids=list(self.object_ids),
            times_s=times_s,
            positions_km=positions,
            velocities_km_s=velocities,
            valid=error_codes == 0,
            reference_epoch=start,
        )

    def propagate_grid(
        self,
        start: datetime,
        duration_s: float,
        step_s: float,
    ) -> PropagationGrid:
        """Propagate every object across a uniform time grid.

        Parameters
        ----------
        start
            First epoch of the grid.
        duration_s
            Total span.
        step_s
            Grid spacing. See :data:`aegis.constants.SCREENING_STEP_S` for
            why 30 s is a reasonable default -- the cost model trades
            propagation count against neighbour count, and neighbour counts
            grow as the cube of the step.

        Notes
        -----
        The whole grid is materialised at once: a 10,000-object catalog at
        30 s spacing over seven days is roughly 4.8 GB. Catalog-scale
        screening streams time blocks through :mod:`aegis.screening.sweep`
        instead.
        """
        start = ensure_utc(start)
        n_steps = int(np.floor(duration_s / step_s)) + 1
        times_s = np.arange(n_steps, dtype=float) * step_s
        return self._propagate_times(start, times_s)

    def propagate_one(self, object_index: int, epoch: datetime) -> StateVector:
        """Propagate a single object to a single epoch.

        Used by closest-approach refinement, which needs a handful of
        precisely-placed evaluations rather than a grid.
        """
        epoch = ensure_utc(epoch)
        julian = JULIAN_DATE_UNIX_EPOCH + epoch.timestamp() / 86400.0
        whole = np.floor(julian)
        fraction = julian - whole

        error_code, position, velocity = self._satrecs[object_index].sgp4(whole, fraction)
        if error_code != 0:
            raise PropagationError(
                f"SGP4 error {error_code} for {self.object_ids[object_index]} at {epoch}"
            )

        return StateVector(
            epoch=epoch,
            position_km=np.array(position),
            velocity_km_s=np.array(velocity),
            frame="TEME",
        )

    def propagate_pair(
        self, index_a: int, index_b: int, epoch: datetime
    ) -> tuple[StateVector, StateVector]:
        """Propagate two objects to the same epoch."""
        return self.propagate_one(index_a, epoch), self.propagate_one(index_b, epoch)
