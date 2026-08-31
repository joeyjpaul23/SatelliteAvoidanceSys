"""Catalogued space objects and the operators who fly them.

The entity model mirrors the CCSDS conjunction-assessment vocabulary that
operational systems converge on -- Operators own Objects, Objects have
Trajectories, pairs of Trajectories produce Conjunctions. Staying close to
that vocabulary means CDMs and OEMs map onto these types without an awkward
translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..constants import HBR_DEFAULT_M
from .timebase import ensure_utc

__all__ = ["ObjectType", "Operator", "SpaceObject", "OrbitalElements"]


class ObjectType:
    """CCSDS ``OBJECT_TYPE`` values."""

    PAYLOAD = "PAYLOAD"
    ROCKET_BODY = "ROCKET BODY"
    DEBRIS = "DEBRIS"
    UNKNOWN = "UNKNOWN"
    OTHER = "OTHER"

    ALL = (PAYLOAD, ROCKET_BODY, DEBRIS, UNKNOWN, OTHER)


@dataclass(frozen=True)
class Operator:
    """An entity that operates satellites in orbit."""

    identifier: str
    name: str
    organization: str = ""
    contact_email: str = ""

    #: Whether this operator's objects can be commanded to maneuver by this
    #: system. Only our own fleet is maneuverable; everything else in the
    #: catalog is an obstacle to be avoided, never moved.
    maneuverable: bool = False


@dataclass
class OrbitalElements:
    """Mean orbital elements, as carried by a TLE or OMM record.

    These are Brouwer/Kozai mean elements and are only meaningful when paired
    with SGP4. Converting them to osculating elements and feeding a numerical
    propagator produces worse results, not better -- the element set and the
    theory are a matched pair.
    """

    epoch: datetime
    mean_motion_rev_per_day: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    bstar: float = 0.0
    mean_motion_dot: float = 0.0
    mean_motion_ddot: float = 0.0
    element_set_number: int = 0
    revolution_number: int = 0

    def __post_init__(self) -> None:
        self.epoch = ensure_utc(self.epoch)

    @property
    def semi_major_axis_km(self) -> float:
        """Semi-major axis from mean motion via Kepler's third law."""
        from ..constants import MU_EARTH_KM3_S2, REV_PER_DAY_TO_RAD_PER_S

        n = self.mean_motion_rev_per_day * REV_PER_DAY_TO_RAD_PER_S
        return (MU_EARTH_KM3_S2 / (n * n)) ** (1.0 / 3.0)

    @property
    def period_s(self) -> float:
        """Orbital period in seconds."""
        from ..constants import SECONDS_PER_DAY

        return SECONDS_PER_DAY / self.mean_motion_rev_per_day

    @property
    def perigee_radius_km(self) -> float:
        return self.semi_major_axis_km * (1.0 - self.eccentricity)

    @property
    def apogee_radius_km(self) -> float:
        return self.semi_major_axis_km * (1.0 + self.eccentricity)

    @property
    def perigee_altitude_km(self) -> float:
        from ..constants import R_EARTH_KM

        return self.perigee_radius_km - R_EARTH_KM

    @property
    def apogee_altitude_km(self) -> float:
        from ..constants import R_EARTH_KM

        return self.apogee_radius_km - R_EARTH_KM

    @property
    def mean_motion_rad_s(self) -> float:
        from ..constants import REV_PER_DAY_TO_RAD_PER_S

        return self.mean_motion_rev_per_day * REV_PER_DAY_TO_RAD_PER_S


@dataclass
class SpaceObject:
    """One catalogued object: an owned satellite or an external hazard.

    Attributes
    ----------
    object_id
        Primary key within this system. For catalogued objects this is the
        NORAD catalog number as a string.
    name
        Human-readable name, e.g. ``STARLINK-1008``.
    international_designator
        COSPAR designator, ``YYYY-NNNP``.
    object_type
        One of :class:`ObjectType`.
    elements
        Mean elements for propagation. ``None`` for objects supplied purely
        as ephemeris.
    operator
        Owning operator, if known.
    hard_body_radius_m
        Circumscribing-sphere radius from centre of mass to the furthest
        extremity -- deployed solar arrays included, not the bus dimension.
        Because Pc scales as the square of the combined radius, this value is
        a first-order driver of every risk number and is always reported
        alongside results rather than left implicit.
    tle_line1, tle_line2
        Raw TLE lines when the object was ingested from TLE data, kept for
        provenance and for handing directly to the SGP4 initialiser.
    """

    object_id: str
    name: str = ""
    international_designator: str = ""
    object_type: str = ObjectType.UNKNOWN
    elements: OrbitalElements | None = None
    operator: Operator | None = None
    hard_body_radius_m: float | None = None
    tle_line1: str = ""
    tle_line2: str = ""
    data_source: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hard_body_radius_m is None:
            self.hard_body_radius_m = HBR_DEFAULT_M.get(
                self.object_type, HBR_DEFAULT_M[ObjectType.UNKNOWN]
            )

    @property
    def is_maneuverable(self) -> bool:
        """Whether this system may plan maneuvers for this object."""
        return self.operator is not None and self.operator.maneuverable

    @property
    def epoch_age_days(self) -> float | None:
        """Age of this object's element set, in days.

        The single best available proxy for how far to trust a TLE-derived
        result, and reported alongside every conjunction for that reason.
        """
        if self.elements is None:
            return None
        from .timebase import seconds_between, utc_now

        return seconds_between(self.elements.epoch, utc_now()) / 86400.0

    def __repr__(self) -> str:
        return f"SpaceObject({self.object_id}, {self.name!r}, {self.object_type})"
