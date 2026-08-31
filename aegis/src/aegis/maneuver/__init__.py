"""Fleet-wide maneuver optimisation.

Public surface
--------------
``clohessy_wiltshire_state``, ``along_track_response_km``,
``required_miss_distance_km``, ``plan_maneuvers``, ``apply_along_track_burns``,
``rescreen_until_stable``, ``ManeuverSolverError``.

Value types live in :mod:`aegis.core.maneuver`. This package is the
optimizer that produces them.

Clohessy-Wiltshire
------------------
Along-track displacement from a tangential impulse uses the exact linear
response ``(4*sin(n*dt) - 3*n*dt) / n * dv``. The ``3*dv*dt`` shortcut is
never used.

Applying burns to SGP4 copies
-----------------------------
SGP4 has no impulsive-burn table. :func:`apply_along_track_burns` maps each
planned along-track ``dv`` to the Clohessy-Wiltshire along-track
displacement at the relevant TCA and writes that as a mean-anomaly shift
``ΔM = Δy_cw / a`` on a copy of the object (TLE line 2 when present). See
the :mod:`aegis.maneuver.rescreen` module docstring for the full mapping.
"""

from .cw import along_track_response_km, clohessy_wiltshire_state
from .errors import ManeuverSolverError
from .miss import required_miss_distance_km
from .planner import plan_maneuvers
from .rescreen import apply_along_track_burns, rescreen_until_stable

__all__ = [
    "clohessy_wiltshire_state",
    "along_track_response_km",
    "required_miss_distance_km",
    "plan_maneuvers",
    "apply_along_track_burns",
    "rescreen_until_stable",
    "ManeuverSolverError",
]
