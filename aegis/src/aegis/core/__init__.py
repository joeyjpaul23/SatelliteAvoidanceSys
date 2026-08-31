"""Core domain types.

This package holds the vocabulary of the system: state vectors, covariance,
frames, catalogued objects, conjunctions and maneuvers. It has no I/O, no
network access, and no algorithms beyond linear algebra and geometry.

Every other package depends on ``core``; ``core`` depends on nothing but
``aegis.constants``. Keeping that arrow one-directional is what makes the
rest of the codebase easy to reason about.
"""

from .conjunction import (
    Conjunction,
    ConjunctionEvent,
    RiskAssessment,
    RiskLevel,
    cumulative_probability,
)
from .frames import (
    EncounterFrame,
    build_encounter_frame,
    enforce_tca,
    rotate_covariance,
    rtn_basis,
    rtn_to_eci_matrix,
)
from .maneuver import (
    ManeuverPlan,
    Maneuver,
    ResolvedConjunction,
    SatelliteManeuverSet,
    propellant_mass_kg,
)
from .objects import ObjectType, Operator, OrbitalElements, SpaceObject
from .state import Covariance, CovarianceSource, Ephemeris, StateVector
from .timebase import (
    ensure_utc,
    format_epoch,
    parse_epoch,
    seconds_between,
    shift,
    utc_now,
)

__all__ = [
    "Conjunction",
    "ConjunctionEvent",
    "Covariance",
    "CovarianceSource",
    "EncounterFrame",
    "Ephemeris",
    "Maneuver",
    "ManeuverPlan",
    "ObjectType",
    "Operator",
    "OrbitalElements",
    "ResolvedConjunction",
    "RiskAssessment",
    "RiskLevel",
    "SatelliteManeuverSet",
    "SpaceObject",
    "StateVector",
    "build_encounter_frame",
    "cumulative_probability",
    "enforce_tca",
    "ensure_utc",
    "format_epoch",
    "parse_epoch",
    "propellant_mass_kg",
    "rotate_covariance",
    "rtn_basis",
    "rtn_to_eci_matrix",
    "seconds_between",
    "shift",
    "utc_now",
]
