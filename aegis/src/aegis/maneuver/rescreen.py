"""Plan / apply / re-screen loop.

Burn application mapping (SGP4-visible)
---------------------------------------
SGP4 has no impulsive-burn table. ``apply_along_track_burns`` therefore
does not change velocity in the propagator. It maps each planned
tangential ``dv`` onto a mean-element change that ``Sgp4Propagator``
reads.

Chosen mapping — constant phase offset, mean motion unchanged::

    Δy_cw = Σ along_track_response_km(n, t_ref − t_burn, dv_T)
    ΔM    = Δy_cw / a          (radians; same sign as Δy_cw)

``t_ref`` is the earliest TCA that involves the satellite, or one orbit
after its last burn when no catalog is supplied. ``a`` is the Keplerian
semi-major axis from mean motion. Semi-major axis and mean motion are
left alone, so the secular CW drift rate is not reproduced — only the
total along-track Clohessy-Wiltshire displacement at ``t_ref`` is
applied as a timing shift.

That shift is written onto the copy SGP4 will initialise from:

- mean elements: ``elements.mean_anomaly_deg += deg(ΔM)``
- TLE line 2 columns 44–51, when TLE lines are present, so
  ``Satrec.twoline2rv`` sees the same ``ΔM``

A later ``screen`` of the copies therefore sees a larger along-track
separation (or the pair leaves the screening box). At the original TCA
the maneuvering body itself is displaced by ``Δy_cw`` along-track; the
primary-RTN relative ``y`` (secondary minus primary) changes by
``Δy_cw_secondary − Δy_cw_primary``. For a two-body along-track pair
that relative change matches the intended separation in sign and in
magnitude to the contract tolerance (50 % of ``|Δy_cw|`` or 2 km).
"""

from __future__ import annotations

import copy
import math
from datetime import datetime

from ..constants import MU_EARTH_KM3_S2
from ..core.conjunction import RiskLevel
from ..core.maneuver import ManeuverPlan
from ..core.objects import OrbitalElements, SpaceObject
from ..core.timebase import seconds_between, shift
from ..risk.batch import AssessedCatalog, assess_catalog
from ..screening import screen
from .cw import along_track_response_km
from .errors import ManeuverSolverError
from .planner import plan_maneuvers

__all__ = ["rescreen_until_stable", "apply_along_track_burns"]

_SCREEN_KEYS = frozenset({"step_s", "box_km", "propagator"})
_ASSESS_KEYS = frozenset({"covariance_model", "cross_check"})
_PLAN_KEYS = frozenset(
    {
        "now",
        "target_pc",
        "dv_budget_km_s",
        "slack_penalty",
        "burn_slots",
        "min_lead_orbits",
    }
)


def _pair_id(primary_id: str, secondary_id: str) -> str:
    return f"{min(primary_id, secondary_id)}:{max(primary_id, secondary_id)}"


def _watch_pairs(assessed: AssessedCatalog) -> set[str]:
    return {
        _pair_id(entry.conjunction.primary.object_id, entry.conjunction.secondary.object_id)
        for entry in assessed.above(RiskLevel.WATCH)
    }


def _split_kwargs(kwargs: dict) -> tuple[dict, dict, dict]:
    screen_kwargs = {key: kwargs[key] for key in _SCREEN_KEYS if key in kwargs}
    assess_kwargs = {key: kwargs[key] for key in _ASSESS_KEYS if key in kwargs}
    plan_kwargs = {key: kwargs[key] for key in _PLAN_KEYS if key in kwargs}
    return screen_kwargs, assess_kwargs, plan_kwargs


def _tle_checksum(line68: str) -> str:
    total = 0
    for char in line68[:68]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return str(total % 10)


def _patch_tle_mean_anomaly(line2: str, mean_anomaly_deg: float) -> str | None:
    text = line2.rstrip("\r\n")
    if len(text) < 68:
        return None
    body = (text[:68] + " " * 68)[:68]
    field = f"{(mean_anomaly_deg % 360.0):8.4f}"
    if len(field) != 8:
        return None
    patched = body[:43] + field + body[51:]
    return patched[:68] + _tle_checksum(patched)


def _semi_major_axis_km(elements: OrbitalElements | None, n_rad_s: float) -> float:
    if elements is not None:
        return elements.semi_major_axis_km
    if n_rad_s <= 0.0:
        return 0.0
    return float((MU_EARTH_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0))


def _mean_motion_rad_s(obj: SpaceObject) -> float | None:
    if obj.elements is None or obj.elements.mean_motion_rev_per_day <= 0.0:
        return None
    return obj.elements.mean_motion_rad_s


def _apply_mean_anomaly_shift(obj: SpaceObject, delta_m_deg: float) -> None:
    """Write ``ΔM`` onto every source ``Sgp4Propagator`` can initialise from."""
    if obj.elements is not None:
        obj.elements.mean_anomaly_deg = (
            obj.elements.mean_anomaly_deg + delta_m_deg
        ) % 360.0
        target_deg = obj.elements.mean_anomaly_deg
    elif obj.tle_line1 and obj.tle_line2:
        try:
            target_deg = (float(obj.tle_line2[43:51]) + delta_m_deg) % 360.0
        except (ValueError, IndexError):
            return
    else:
        return

    if obj.tle_line1 and obj.tle_line2:
        patched = _patch_tle_mean_anomaly(obj.tle_line2, target_deg)
        if patched is None:
            # Fall back to mean elements so SGP4 still sees the shift.
            obj.tle_line1 = ""
            obj.tle_line2 = ""
        else:
            obj.tle_line2 = patched


def apply_along_track_burns(
    objects: list[SpaceObject],
    plan: ManeuverPlan,
    assessed: AssessedCatalog | None = None,
) -> list[SpaceObject]:
    """Return copies of ``objects`` with planned along-track burns applied.

    Mapping: ``ΔM = Δy_cw / a`` at the satellite's reference TCA. See the
    module docstring. Zero-``dv`` plans return unchanged copies.
    """
    copies = [copy.deepcopy(obj) for obj in objects]
    by_id = {obj.object_id: obj for obj in copies}

    for sat_id, sat_set in plan.satellite_sets.items():
        obj = by_id.get(sat_id)
        if obj is None or not sat_set.maneuvers:
            continue
        n_rad_s = _mean_motion_rad_s(obj)
        if n_rad_s is None or n_rad_s <= 0.0:
            continue

        t_ref = None
        if assessed is not None:
            tcas = [
                entry.conjunction.tca
                for entry in assessed.entries
                if sat_id
                in (
                    entry.conjunction.primary.object_id,
                    entry.conjunction.secondary.object_id,
                )
            ]
            if tcas:
                t_ref = min(tcas)
        if t_ref is None:
            last = max(maneuver.epoch for maneuver in sat_set.maneuvers)
            t_ref = shift(last, 2.0 * math.pi / n_rad_s)

        delta_y = 0.0
        for maneuver in sat_set.maneuvers:
            dv = float(maneuver.delta_v_rtn_km_s[1])
            dt_s = seconds_between(maneuver.epoch, t_ref)
            delta_y += along_track_response_km(n_rad_s, dt_s, dv)

        axis_km = _semi_major_axis_km(obj.elements, n_rad_s)
        if axis_km <= 0.0:
            continue
        _apply_mean_anomaly_shift(obj, math.degrees(delta_y / axis_km))

    return copies


def rescreen_until_stable(
    objects: list[SpaceObject],
    start: datetime,
    duration_s: float,
    *,
    max_iterations: int = 5,
    **screen_and_plan_kwargs,
) -> ManeuverPlan:
    """Alternate screening, planning, and burn application until pair-ids settle.

    Stops when a re-screen introduces no new ``WATCH`` / ``ACT`` pair-ids, or
    ``max_iterations`` is reached. ``converged`` is True only when that last
    re-screen added no new pair above ``WATCH``.
    """
    if max_iterations < 1:
        raise ManeuverSolverError("max_iterations must be at least 1")

    screen_kwargs, assess_kwargs, plan_kwargs = _split_kwargs(screen_and_plan_kwargs)
    first_screen_kwargs = dict(screen_kwargs)
    later_screen_kwargs = {k: v for k, v in screen_kwargs.items() if k != "propagator"}

    working = list(objects)
    seen_watch: set[str] = set()
    plan: ManeuverPlan | None = None

    for iteration in range(1, max_iterations + 1):
        kwargs = first_screen_kwargs if iteration == 1 else later_screen_kwargs
        conjunctions = screen(working, start, duration_s, **kwargs)
        assessed = assess_catalog(conjunctions, objects=working, **assess_kwargs)
        watch_pairs = _watch_pairs(assessed)

        if plan is not None and watch_pairs <= seen_watch:
            plan.iterations = iteration - 1
            plan.converged = True
            return plan

        seen_watch |= watch_pairs
        plan = plan_maneuvers(assessed, working, **plan_kwargs)
        plan.iterations = iteration
        plan.converged = False

        if iteration == max_iterations:
            working = apply_along_track_burns(working, plan, assessed)
            conjunctions = screen(working, start, duration_s, **later_screen_kwargs)
            assessed = assess_catalog(conjunctions, objects=working, **assess_kwargs)
            plan.converged = _watch_pairs(assessed) <= seen_watch
            return plan

        working = apply_along_track_burns(working, plan, assessed)

    assert plan is not None
    plan.iterations = max_iterations
    plan.converged = False
    return plan
