"""Conjunctions, risk assessments, and conjunction events.

Three distinct concepts that are easy to conflate:

Conjunction
    A single predicted close approach between two objects -- a geometry
    result. Produced by the screening engine.

RiskAssessment
    The probability analysis of one conjunction. Produced by the risk engine.
    Kept separate from the geometry because the same close approach can be
    assessed with different covariance assumptions or hard-body radii and
    yield different numbers.

ConjunctionEvent
    The history of one physical close approach as understanding of it evolves
    -- the sequence of assessments produced as new tracking data arrives.
    This mirrors the CCSDS/operational distinction between a CDM (one
    snapshot) and a conjunction event (the whole thread).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..constants import (
    PC_THRESHOLD_ACT,
    PC_THRESHOLD_ASSESS,
    PC_THRESHOLD_WATCH,
)
from .objects import SpaceObject
from .state import StateVector
from .timebase import ensure_utc

__all__ = [
    "Conjunction",
    "RiskAssessment",
    "ConjunctionEvent",
    "RiskLevel",
]


class RiskLevel:
    """Severity bands for a conjunction, ordered least to most severe."""

    CLEAR = "CLEAR"        # below the assessment threshold
    MONITOR = "MONITOR"    # above assessment, below watch
    WATCH = "WATCH"        # above watch, below action
    ACT = "ACT"            # above the action threshold; maneuver candidate

    ORDER = (CLEAR, MONITOR, WATCH, ACT)

    @staticmethod
    def from_probability(probability: float) -> str:
        """Band a collision probability.

        Thresholds come from :mod:`aegis.constants` and are configurable --
        1e-4 is the widely used industry action level, but a dense
        constellation screening against itself may justify something far more
        conservative (Starlink reportedly operates at 3e-7).
        """
        if probability >= PC_THRESHOLD_ACT:
            return RiskLevel.ACT
        if probability >= PC_THRESHOLD_WATCH:
            return RiskLevel.WATCH
        if probability >= PC_THRESHOLD_ASSESS:
            return RiskLevel.MONITOR
        return RiskLevel.CLEAR

    @staticmethod
    def rank(level: str) -> int:
        """Numeric severity, for sorting."""
        return RiskLevel.ORDER.index(level)


@dataclass
class Conjunction:
    """A single predicted close approach between two objects.

    Pure geometry -- no probability. Attributes mirror the CCSDS CDM relative
    metadata section so that emitting a conforming CDM is a direct mapping.

    Attributes
    ----------
    conjunction_id
        Stable identifier derived from the object pair and TCA.
    primary, secondary
        The two objects. By convention the primary is the one this system may
        maneuver, when exactly one of the pair is ours.
    tca
        Time of closest approach, UTC.
    miss_distance_km
        Separation at closest approach.
    relative_speed_km_s
        Relative speed at closest approach.
    relative_position_rtn_km, relative_velocity_rtn_km_s
        Relative state expressed in the primary's RTN frame, which is how
        CDMs report it and how screening volumes are defined.
    primary_state, secondary_state
        Full inertial states at closest approach, retained because the risk
        engine needs them to rebuild the encounter frame.
    """

    conjunction_id: str
    primary: SpaceObject
    secondary: SpaceObject
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float
    relative_position_rtn_km: np.ndarray
    relative_velocity_rtn_km_s: np.ndarray
    primary_state: StateVector
    secondary_state: StateVector
    screening_window_start: datetime | None = None
    screening_window_end: datetime | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tca = ensure_utc(self.tca)
        self.relative_position_rtn_km = np.asarray(
            self.relative_position_rtn_km, dtype=float
        ).reshape(3)
        self.relative_velocity_rtn_km_s = np.asarray(
            self.relative_velocity_rtn_km_s, dtype=float
        ).reshape(3)

    @property
    def combined_hard_body_radius_m(self) -> float:
        """Sum of both objects' circumscribing radii."""
        return float(self.primary.hard_body_radius_m + self.secondary.hard_body_radius_m)

    @property
    def is_intra_fleet(self) -> bool:
        """Whether both objects belong to the same maneuverable operator.

        These are the conjunctions no public service will compute for you --
        SOCRATES explicitly excludes intra-fleet close approaches for large
        constellations on the assumption the operator tracks them internally.
        They are also the ones where both objects can maneuver, which makes
        them the interesting case for fleet-wide coordination.
        """
        if self.primary.operator is None or self.secondary.operator is None:
            return False
        return (
            self.primary.operator.identifier == self.secondary.operator.identifier
            and self.primary.operator.maneuverable
        )

    @property
    def maneuverable_object_ids(self) -> tuple[str, ...]:
        """Which of the two objects this system may plan maneuvers for."""
        ids = []
        if self.primary.is_maneuverable:
            ids.append(self.primary.object_id)
        if self.secondary.is_maneuverable:
            ids.append(self.secondary.object_id)
        return tuple(ids)

    def time_to_tca_s(self, now: datetime | None = None) -> float:
        """Seconds from ``now`` until closest approach."""
        from .timebase import seconds_between, utc_now

        return seconds_between(now or utc_now(), self.tca)


@dataclass
class RiskAssessment:
    """Probability analysis of a single conjunction.

    Deliberately reports far more than a bare Pc. A single probability is a
    poor decision statistic on its own: it is non-monotonic in covariance
    size, so a large uncertainty can produce a small, falsely reassuring
    number. Reporting the miss distance, the Mahalanobis distance, the
    covariance scale and the geometric maximum alongside it lets an operator
    see *why* the probability is what it is.

    Attributes
    ----------
    probability
        Collision probability from the primary method (Alfano 2D).
    method
        Identifier of the method used, recorded in the CDM.
    cross_check_probability
        Independent estimate from a second method, when applicable.
    max_probability
        The covariance-independent ceiling ``HBR^2 / (e * d^2)``. Driven
        purely by geometry, so it bounds risk even when the covariance is
        untrusted.
    mahalanobis_distance
        Miss distance in units of the projected covariance -- the scale-free
        measure of how unusual this approach is.
    sigma_major_km, sigma_minor_km
        Principal one-sigma axes of the projected 2D covariance.
    dilution_flag
        Set when the covariance is large relative to the miss distance, i.e.
        the event sits in the regime where a lower Pc reflects worse
        knowledge rather than greater safety.
    remediated_flag
        Set when the projected covariance was not positive definite and had
        to be repaired before integration.
    short_encounter_valid
        Cleared when the relative velocity is too low or the encounter too
        long for the 2D model to apply.
    """

    conjunction_id: str
    probability: float
    method: str
    hard_body_radius_m: float
    miss_distance_km: float
    max_probability: float
    mahalanobis_distance: float
    sigma_major_km: float
    sigma_minor_km: float
    cross_check_probability: float | None = None
    cross_check_method: str | None = None
    dilution_flag: bool = False
    remediated_flag: bool = False
    short_encounter_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    assessed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.assessed_at is None:
            from .timebase import utc_now

            self.assessed_at = utc_now()
        else:
            self.assessed_at = ensure_utc(self.assessed_at)

    @property
    def risk_level(self) -> str:
        return RiskLevel.from_probability(self.probability)

    @property
    def log10_probability(self) -> float:
        """Base-10 log of Pc, the conventional reporting scale.

        Floors at -30 rather than returning ``-inf`` so that plots, sorts and
        serialisation all behave. Genuine probabilities below 1e-30 are far
        beyond the precision of any input this system receives.
        """
        if self.probability <= 1e-30:
            return -30.0
        return float(np.log10(self.probability))

    @property
    def cross_check_agrees(self) -> bool | None:
        """Whether the cross-check method corroborates the primary result.

        ``None`` when no cross-check ran. Disagreement does not necessarily
        mean the primary result is wrong -- the analytic cross-check has a
        known validity envelope -- but it always warrants a look.
        """
        if self.cross_check_probability is None:
            return None
        if self.probability == 0.0 and self.cross_check_probability == 0.0:
            return True
        scale = max(self.probability, self.cross_check_probability, 1e-30)
        return abs(self.probability - self.cross_check_probability) / scale < 1e-2


@dataclass
class ConjunctionEvent:
    """The evolving history of one physical close approach.

    A conjunction is re-assessed as new tracking data arrives; this collects
    that sequence so an operator can see whether risk is trending up or down.
    Mirrors the CDM-versus-event distinction in operational systems.
    """

    event_id: str
    primary_id: str
    secondary_id: str
    tca: datetime
    assessments: list[RiskAssessment] = field(default_factory=list)
    conjunctions: list[Conjunction] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tca = ensure_utc(self.tca)

    @property
    def latest_assessment(self) -> RiskAssessment | None:
        return self.assessments[-1] if self.assessments else None

    @property
    def peak_probability(self) -> float:
        """Highest probability seen across the event's history."""
        if not self.assessments:
            return 0.0
        return max(assessment.probability for assessment in self.assessments)

    @property
    def trend(self) -> str:
        """Direction of the two most recent assessments."""
        if len(self.assessments) < 2:
            return "UNKNOWN"
        latest, previous = self.assessments[-1].probability, self.assessments[-2].probability
        if latest > previous * 1.1:
            return "RISING"
        if latest < previous * 0.9:
            return "FALLING"
        return "STABLE"

    def add(self, conjunction: Conjunction, assessment: RiskAssessment) -> None:
        """Append a new snapshot of this event."""
        self.conjunctions.append(conjunction)
        self.assessments.append(assessment)


def cumulative_probability(probabilities: list[float]) -> float:
    """Combine independent conjunction probabilities into a total.

    ``P_total = 1 - prod(1 - P_i)``

    Screening a constellation means a satellite can accumulate many
    individually sub-threshold events whose combined risk exceeds the action
    threshold. A per-event threshold alone understates exposure, so aggregate
    per-object risk over the screening window is reported alongside it.
    """
    survival = 1.0
    for probability in probabilities:
        survival *= 1.0 - probability
    return 1.0 - survival
