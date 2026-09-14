"""Deterministic conjunction-graph scenario generation.

Section 13 of ``aegis/specs/step14_fleet_optimization.md`` needs catalogs of
*known* graph shape -- a star, a chain, a clique -- so the fleetopt planners
and certifier can be exercised against a conjunction network whose structure
is a design choice, not whatever a live catalog happens to contain. Every
family here is placed with closed-form orbital geometry rather than trial
and error:

Two satellites in circular orbits at the same altitude trace two great
circles on a common sphere. Those circles are either identical, parallel
(never crossing), or intersect at exactly two antipodal points -- the line
of nodes between the two orbital planes, ``normal_a x normal_b``. A
satellite following either orbit passes through each of those two points
once per revolution. Given target orbital elements for satellite A and any
distinct plane for satellite B, :func:`_solve_crossing_candidates` computes the mean
anomaly that puts B through one of those two points at the same instant A
is there -- which is precisely a conjunction. Chains and stars are built by
repeating that one solve against different reference satellites and target
times; clusters (clique, debris-shower) are built by placing several
satellites near one solved crossing point with a small mean-anomaly spread
that keeps them inside the screening box across the whole encounter.

Every synthetic family requires the same
:class:`~aegis.ingest.synthetic.SyntheticAuthorization` object
``aegis.ingest.synthetic`` uses, checked against the same
``AEGIS_ALLOW_SYNTHETIC=1`` gate, and sets ``data_source=SYNTHETIC`` on every
object it creates -- so the existing mixed-source guards in
``aegis.screening`` and ``aegis.risk`` keep applying to scenario output
exactly as they would to any other synthetic catalog. Orbital geometry is
built directly with :class:`~aegis.core.objects.OrbitalElements` rather than
by calling :func:`aegis.ingest.synthetic.generate_synthetic`, because that
function's Walker-constellation placement logic has no notion of "solve for
a specific conjunction" -- it was never meant to.

``replay-tle`` is the one family that does not synthesize anything: it
parses the committed TLE fixtures under ``aegis/tests/fixtures/`` with
:func:`aegis.ingest.celestrak.catalog_from_tle_file`, sets
``data_source=CELESTRAK``, and requires no authorization at all.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from ..constants import (
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    REV_PER_DAY_TO_RAD_PER_S,
    SCREENING_BOX_STARLINK_KM,
    SCREENING_BOX_TLE_GRADE_KM,
    SCREENING_HORIZON_S,
    SECONDS_PER_DAY,
)
from ..core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from ..core.timebase import ensure_utc, format_epoch, shift
from ..propagation.propagator import Sgp4Propagator
from ..ingest.celestrak import CelesTrakError, catalog_from_tle_file
from ..ingest.sources import Catalog, DataSource, SyntheticNotAuthorizedError
from ..ingest.synthetic import DEFAULT_SYNTHETIC_EPOCH, SyntheticAuthorization
from .errors import ScenarioError
from .spec import Scenario, ScenarioSpec, scenario_digest

__all__ = ["FAMILIES", "generate", "describe"]

FAMILIES: tuple[str, ...] = (
    "isolated-pair",
    "chain",
    "star",
    "clique",
    "intra-plane",
    "crossing-planes",
    "debris-shower",
    "dense-shell",
    "induced-cascade",
    "replay-tle",
)

_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "isolated-pair": "one conjunction, one maneuverable satellite",
    "chain": "conjunctions sharing satellites in a path graph",
    "star": "one satellite in k simultaneous conjunctions",
    "clique": "a tightly coupled cluster, all TCAs within one orbit",
    "intra-plane": "same-plane, low relative velocity (tests the 2D-Pc guard)",
    "crossing-planes": "high relative velocity, different RAAN",
    "debris-shower": "many non-maneuverable secondaries on one plane",
    "dense-shell": "large catalog, realistic Starlink-like shell",
    "induced-cascade": (
        "one satellite forced to maneuver by an external hazard, with in-plane "
        "fleet-mates close enough that the avoidance displacement can create a "
        "new conjunction -- the canonical induced-conjunction geometry"
    ),
    "replay-tle": "built from the committed TLE fixtures, no synthesis",
}

#: Families that synthesize objects and therefore need dual-gated authorization.
_SYNTHETIC_FAMILIES = frozenset(FAMILIES) - {"replay-tle"}

_SYNTHETIC_REFUSED = (
    "synthetic scenario generation was refused: AEGIS_ALLOW_SYNTHETIC=1 "
    "plus an explicit SyntheticAuthorization are required"
)

#: SGP4 is far happier with a tiny eccentricity than with exactly zero, and
#: the ~1 km apogee/perigee spread it introduces is negligible next to the
#: screening boxes below. Matches the value ``aegis.ingest.synthetic`` uses.
_ECCENTRICITY = 1.0e-4

#: Default LEO shell: Starlink's operational altitude band and inclination.
_DEFAULT_ALTITUDE_KM = 550.0
_DEFAULT_INCLINATION_DEG = 53.0


def describe(family: str) -> str:
    """One-line description of the structure ``family`` is built to exhibit."""
    if family not in _FAMILY_DESCRIPTIONS:
        raise ScenarioError(f"unknown scenario family: {family!r}; choices are {FAMILIES}")
    return _FAMILY_DESCRIPTIONS[family]


def _require_synthetic_authorization(authorization: object) -> None:
    if not isinstance(authorization, SyntheticAuthorization):
        raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)
    if authorization.acknowledge_synthetic is not True:
        raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)
    if os.environ.get("AEGIS_ALLOW_SYNTHETIC") != "1":
        raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)


# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------


def _mean_motion_rev_per_day(altitude_km: float) -> float:
    """Kepler's third law: ``n = sqrt(mu / a^3)``, rad/s converted to rev/day.

    Deliberately duplicated rather than imported from
    ``aegis.ingest.synthetic``'s private helper of the same name: that
    function is an implementation detail of the Walker-constellation path,
    and this contract (step14 section 13) asks scenarios to build
    ``OrbitalElements`` directly instead of routing through
    ``generate_synthetic``.
    """
    semi_major_axis_km = R_EARTH_KM + altitude_km
    mean_motion_rad_s = (MU_EARTH_KM3_S2 / (semi_major_axis_km**3)) ** 0.5
    return mean_motion_rad_s / REV_PER_DAY_TO_RAD_PER_S


def _mean_motion_rad_s(altitude_km: float) -> float:
    return _mean_motion_rev_per_day(altitude_km) * REV_PER_DAY_TO_RAD_PER_S


def _period_s(altitude_km: float) -> float:
    return SECONDS_PER_DAY / _mean_motion_rev_per_day(altitude_km)


def _circular_speed_km_s(altitude_km: float) -> float:
    return math.sqrt(MU_EARTH_KM3_S2 / (R_EARTH_KM + altitude_km))


def _wrap_deg(angle_deg: float) -> float:
    return angle_deg % 360.0


def _rotation_matrix(inclination_deg: float, raan_deg: float) -> np.ndarray:
    """``Rz(RAAN) @ Rx(inclination)``: ECI frame for ``arg_perigee = 0``.

    Every family here places circular-ish orbits (``e ~= 1e-4``) with
    ``arg_perigee_deg = 0``, so "true anomaly" and "argument of latitude"
    coincide and a position at angle ``u`` is
    ``rotation @ (R cos u, R sin u, 0)`` for orbit radius ``R``. This matches
    the classical Rz(RAAN)-Rx(i)-Rz(arg_perigee) element rotation with the
    last (identity, since arg_perigee = 0) term dropped.
    """
    inclination = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    cos_r, sin_r = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inclination), math.sin(inclination)
    rz = np.array([[cos_r, -sin_r, 0.0], [sin_r, cos_r, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cos_i, -sin_i], [0.0, sin_i, cos_i]])
    return rz @ rx


def _plane_normal(inclination_deg: float, raan_deg: float) -> np.ndarray:
    return _rotation_matrix(inclination_deg, raan_deg) @ np.array([0.0, 0.0, 1.0])


def _argument_of_latitude_rad(rotation: np.ndarray, direction: np.ndarray) -> float:
    """Angle ``u`` such that ``rotation @ (cos u, sin u, 0)`` is parallel to ``direction``.

    ``direction`` must already lie in the orbital plane ``rotation``
    describes (its out-of-plane component, ``local[2]`` below, is then
    exactly zero up to floating point) -- true whenever ``direction`` comes
    from :func:`_solve_crossing_candidates`, which constructs it as the cross product
    of two plane normals.
    """
    local = rotation.T @ direction
    return math.atan2(local[1], local[0])


def _tangent_direction(rotation: np.ndarray, u_rad: float) -> np.ndarray:
    """Unit velocity direction at argument of latitude ``u`` (circular orbit)."""
    local = np.array([-math.sin(u_rad), math.cos(u_rad), 0.0])
    return rotation @ local


def _solve_crossing_candidates(
    inc_a_deg: float,
    raan_a_deg: float,
    ma_a0_rad: float,
    mean_motion_a_rad_s: float,
    inc_b_deg: float,
    raan_b_deg: float,
    mean_motion_b_rad_s: float,
    target_time_s: float,
    period_s: float,
    *,
    max_revolutions: int = 8,
    top_n: int = 4,
) -> list[tuple[float, float]]:
    """Mean anomalies for satellite B that put it where A is, when A is there.

    Two distinct circular orbital planes intersect the unit sphere along one
    line through the origin (two antipodal points on the sphere). Satellite
    A, moving at constant angular rate, passes through each of those points
    once per revolution -- this enumerates those passages across the next
    few revolutions in both directions and returns the ``top_n`` closest to
    ``target_time_s``, each paired with the B mean anomaly that puts B at
    that same point at that same instant.

    Returned in ascending order of ``|crossing_time - target_time_s|``, so
    ``candidates[0]`` is the pure two-body answer -- but SGP4/J2 departs
    from that two-body idealization differently at each candidate (observed:
    anywhere from sub-km to several km even after mean-anomaly refinement,
    see :func:`_min_separation_km`), so a caller placing a real conjunction
    should score a few candidates against the real propagator
    (:func:`_place_conjunction` does this) rather than always trust
    ``candidates[0]``.

    Raises
    ------
    ScenarioError
        If the two planes are (numerically) coincident and have no distinct
        crossing line to place a conjunction on.
    """
    rot_a = _rotation_matrix(inc_a_deg, raan_a_deg)
    rot_b = _rotation_matrix(inc_b_deg, raan_b_deg)
    normal_a = rot_a @ np.array([0.0, 0.0, 1.0])
    normal_b = rot_b @ np.array([0.0, 0.0, 1.0])
    line = np.cross(normal_a, normal_b)
    norm = float(np.linalg.norm(line))
    if norm < 1e-9:
        raise ScenarioError(
            "requested orbital planes are coincident (inclination/RAAN too "
            "close together); choose a larger separation so the two orbits "
            "have a distinct crossing line to place a conjunction on"
        )
    line = line / norm

    scored: list[tuple[float, float, float]] = []
    for direction in (line, -line):
        u_a = _argument_of_latitude_rad(rot_a, direction)
        u_b = _argument_of_latitude_rad(rot_b, direction)
        base_delta = (u_a - ma_a0_rad) % (2.0 * math.pi)
        for revolution in range(max_revolutions):
            crossing_time = base_delta / mean_motion_a_rad_s + revolution * period_s
            gap = abs(crossing_time - target_time_s)
            ma_b0 = (u_b - mean_motion_b_rad_s * crossing_time) % (2.0 * math.pi)
            scored.append((gap, ma_b0, crossing_time))
    scored.sort(key=lambda item: item[0])
    return [(ma_b0, crossing_time) for _gap, ma_b0, crossing_time in scored[: max(top_n, 1)]]


def _place_conjunction(
    altitude_km: float,
    inc_a_deg: float,
    raan_a_deg: float,
    ma_a0_rad: float,
    inc_b_deg: float,
    raan_b_deg: float,
    epoch: datetime,
    target_time_s: float,
    period_s: float,
    *,
    candidate_count: int = 4,
) -> tuple[float, float]:
    """Pick and tighten the best of several candidate B mean anomalies.

    :func:`_solve_crossing_candidates` ranks candidates purely by two-body
    proximity to ``target_time_s``; that ranking is not always the same as
    ranking by *achievable real conjunction quality*, because SGP4/J2's
    departure from the two-body idealization varies candidate to candidate
    (observed directly: one candidate refines to a sub-km true minimum while
    another, equally close to the target time, floors around 3 km no matter
    how its mean anomaly is tuned). This scores the top few candidates with
    a coarse real-propagator check, keeps the best, and only then spends the
    expensive fine 1-D search (:func:`_refine_secondary_mean_anomaly`) on it.
    """
    n_motion = _mean_motion_rad_s(altitude_km)
    candidates = _solve_crossing_candidates(
        inc_a_deg, raan_a_deg, ma_a0_rad, n_motion,
        inc_b_deg, raan_b_deg, n_motion,
        target_time_s, period_s, top_n=candidate_count,
    )
    elements_a = _make_elements(epoch, altitude_km, inc_a_deg, raan_a_deg, ma_a0_rad)

    best_separation_km = math.inf
    best_ma_b0_guess = candidates[0][0]
    best_crossing_time = candidates[0][1]
    for ma_b0_guess, crossing_time in candidates:
        elements_b = _make_elements(epoch, altitude_km, inc_b_deg, raan_b_deg, ma_b0_guess)
        separation_km = _min_separation_km(elements_a, elements_b, epoch, crossing_time, 150.0, step_s=1.0)
        if separation_km < best_separation_km:
            best_separation_km = separation_km
            best_ma_b0_guess = ma_b0_guess
            best_crossing_time = crossing_time

    ma_b0 = _refine_secondary_mean_anomaly(
        altitude_km, inc_a_deg, raan_a_deg, ma_a0_rad, inc_b_deg, raan_b_deg,
        best_ma_b0_guess, epoch, best_crossing_time,
    )
    return ma_b0, best_crossing_time


def _relative_speed_km_s(
    altitude_km: float,
    inc_a_deg: float,
    raan_a_deg: float,
    ma_a_rad: float,
    inc_b_deg: float,
    raan_b_deg: float,
    ma_b_rad: float,
) -> float:
    """Relative speed of two same-altitude circular orbits at a shared point.

    Both satellites move at the same circular speed ``v`` (same altitude);
    they differ only in velocity *direction*. Used to log an honest number
    into scenario provenance rather than asserting a target and hoping.
    """
    speed = _circular_speed_km_s(altitude_km)
    tangent_a = _tangent_direction(_rotation_matrix(inc_a_deg, raan_a_deg), ma_a_rad)
    tangent_b = _tangent_direction(_rotation_matrix(inc_b_deg, raan_b_deg), ma_b_rad)
    return speed * float(np.linalg.norm(tangent_a - tangent_b))


def _min_separation_km(
    elements_a: OrbitalElements, elements_b: OrbitalElements, epoch: datetime, center_time_s: float, half_window_s: float,
    *, step_s: float = 0.2,
) -> float:
    """True SGP4-propagated minimum separation near ``center_time_s``.

    ``_solve_crossing_candidates`` places satellites using pure two-body circular
    geometry; SGP4 additionally applies J2 secular *and* short-period
    corrections, which measurably shift the true closest approach even
    right at epoch (observed: several km to a few tens of km at 550 km
    altitude, comparable to the 44/51 km Starlink screening box and larger
    than its 2 km radial gate). This samples the real propagator in a
    window around the analytic guess so a caller can correct for it.
    """
    dummy_operator = Operator(identifier="SCENARIO-REFINE", name="scenario geometry refinement", maneuverable=False)
    object_a = SpaceObject(object_id="1", elements=elements_a, operator=dummy_operator, data_source=DataSource.SYNTHETIC)
    object_b = SpaceObject(object_id="2", elements=elements_b, operator=dummy_operator, data_source=DataSource.SYNTHETIC)
    propagator = Sgp4Propagator([object_a, object_b])
    window_start = shift(epoch, max(center_time_s - half_window_s, 0.0))
    grid = propagator.propagate_grid(window_start, 2.0 * half_window_s, step_s)
    separations_km = np.linalg.norm(grid.positions_km[0] - grid.positions_km[1], axis=1)
    return float(np.min(separations_km))


def _refine_secondary_mean_anomaly(
    altitude_km: float,
    inc_a_deg: float,
    raan_a_deg: float,
    ma_a0_rad: float,
    inc_b_deg: float,
    raan_b_deg: float,
    ma_b0_rad_guess: float,
    epoch: datetime,
    crossing_time_guess_s: float,
    *,
    search_half_window_s: float = 60.0,
    ma_search_deg: float = 1.0,
) -> float:
    """Nudge B's analytically-solved mean anomaly against real SGP4 output.

    A bounded 1-D Brent search (``scipy.optimize.minimize_scalar``) over a
    small offset to B's mean anomaly, minimizing the true SGP4 minimum
    separation found by :func:`_min_separation_km` in a window around the
    analytic crossing time. This is the numerical-solve fallback the
    contract calls for where the closed form (exact for two-body circular
    orbits) is measurably wrong once SGP4/J2 is the actual propagator.
    """
    elements_a = _make_elements(epoch, altitude_km, inc_a_deg, raan_a_deg, ma_a0_rad)

    def objective(delta_deg: float) -> float:
        ma_b_rad = ma_b0_rad_guess + math.radians(delta_deg)
        elements_b = _make_elements(epoch, altitude_km, inc_b_deg, raan_b_deg, ma_b_rad)
        return _min_separation_km(elements_a, elements_b, epoch, crossing_time_guess_s, search_half_window_s)

    result = minimize_scalar(
        objective, bounds=(-ma_search_deg, ma_search_deg), method="bounded", options={"xatol": 1e-6}
    )
    return ma_b0_rad_guess + math.radians(float(result.x))


def _make_elements(
    epoch: datetime,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    mean_anomaly_rad: float,
    *,
    eccentricity: float = _ECCENTRICITY,
    bstar: float = 0.0,
) -> OrbitalElements:
    return OrbitalElements(
        epoch=epoch,
        mean_motion_rev_per_day=_mean_motion_rev_per_day(altitude_km),
        eccentricity=eccentricity,
        inclination_deg=_wrap_deg(inclination_deg),
        raan_deg=_wrap_deg(raan_deg),
        arg_perigee_deg=0.0,
        mean_anomaly_deg=_wrap_deg(math.degrees(mean_anomaly_rad)),
        bstar=bstar,
    )


def _object_id(seed: int, index: int) -> str:
    """A numeric-only, per-scenario-unique object id.

    Numeric so ``aegis.propagation.propagator.Sgp4Propagator``'s
    ``object_id.isdigit()`` fast path (real NORAD numbering) applies to
    synthetic objects too, exactly as ``aegis.ingest.synthetic`` relies on.
    Only needs to be unique *within* one scenario's object list, not
    globally. Kept under SGP4's satellite-number ceiling (339999, the Alpha-5
    limit) with room for up to 1000 objects per scenario and 300 distinct
    seeds before wrapping.
    """
    return str(10_000 + (seed % 300) * 1_000 + index)


def _make_object(
    object_id: str,
    name: str,
    elements: OrbitalElements,
    operator: Operator | None,
    *,
    object_type: str = ObjectType.PAYLOAD,
) -> SpaceObject:
    return SpaceObject(
        object_id=object_id,
        name=name,
        object_type=object_type,
        elements=elements,
        operator=operator,
        data_source=DataSource.SYNTHETIC,
    )


# ---------------------------------------------------------------------------
# Family builders
#
# Each returns (objects, window_duration_s, window_start, extra_provenance,
# screening_box_km). ``generate`` applies any caller ``window_start`` /
# ``window_duration_s`` override uniformly afterwards.
# ---------------------------------------------------------------------------

_BuiltFamily = tuple[list[SpaceObject], float, datetime, dict, tuple[float, float, float]]


def _build_isolated_pair(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    inclination_a = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    # Same inclination as the primary, moderate RAAN offset: this keeps the
    # true (SGP4, not two-body) minimum separation comfortably under a
    # kilometre after refinement. A large inclination *difference* instead
    # (tried first) adds a differential J2 short-period radial bias between
    # the two orbits that floors the achievable miss distance around ~2 km
    # regardless of mean-anomaly tuning -- right at the screening box's
    # radial gate. See the self-check script and _build_crossing_planes.
    inclination_b = float(overrides.get("secondary_inclination_deg", inclination_a))
    raan_b = float(overrides.get("secondary_raan_deg", 45.0)) + float(rng.uniform(-10.0, 10.0))
    period = _period_s(altitude_km)
    # The two antipodal plane-crossing points are visited half an orbit
    # apart; a window as long as a full period would catch both and turn
    # "one conjunction" into two. A window well under half a period, with
    # the target crossing near its middle, guarantees only the intended one
    # is inside it.
    window_duration_s = float(overrides.get("window_duration_s", period * 0.35))
    target_time_s = float(overrides.get("target_time_s", window_duration_s / 2.0))

    ma_a0 = 0.0
    ma_b0, crossing_time = _place_conjunction(
        altitude_km, inclination_a, 0.0, ma_a0, inclination_b, raan_b, epoch, target_time_s, period,
    )

    fleet_operator = Operator(identifier=f"FLEET-{seed}", name="Fleet primary", maneuverable=True)
    hazard_operator = Operator(identifier=f"HAZARD-{seed}", name="Uncontrolled hazard", maneuverable=False)

    primary = _make_object(
        _object_id(seed, 0), f"ISOLATED-PAIR-{seed}-PRIMARY",
        _make_elements(epoch, altitude_km, inclination_a, 0.0, ma_a0), fleet_operator,
    )
    secondary = _make_object(
        _object_id(seed, 1), f"ISOLATED-PAIR-{seed}-SECONDARY",
        _make_elements(epoch, altitude_km, inclination_b, raan_b, ma_b0), hazard_operator,
        object_type=ObjectType.DEBRIS,
    )

    provenance = {
        "altitude_km": altitude_km,
        "crossing_time_s": crossing_time,
        "relative_speed_km_s": _relative_speed_km_s(
            altitude_km, inclination_a, 0.0, ma_a0, inclination_b, raan_b, ma_b0
        ),
    }
    return [primary, secondary], window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_chain(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    base_inclination = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    chain_length = int(overrides.get("chain_length", 4))
    if chain_length < 3:
        raise ScenarioError("chain scenarios need chain_length >= 3 to exhibit a real path graph")
    period = _period_s(altitude_km)
    window_duration_s = float(overrides.get("window_duration_s", period))

    operator = Operator(identifier=f"CHAIN-FLEET-{seed}", name="Chain fleet", maneuverable=True)

    inclinations = [base_inclination]
    raans = [0.0]
    mean_anomalies_rad = [0.0]
    crossing_times = []

    for k in range(1, chain_length):
        # Alternate the inclination offset and space RAANs well apart so a
        # non-adjacent pair's own (unsolved, incidental) plane crossing does
        # not happen to fall inside this same window -- verified empirically
        # in the self-check script, not just asserted here.
        sign = 1.0 if k % 2 == 1 else -1.0
        candidate_inclination = base_inclination + sign * (3.0 + 1.5 * k)
        candidate_raan = _wrap_deg(70.0 * k + float(rng.uniform(-10.0, 10.0)))
        target_time_s = window_duration_s * (k / (chain_length + 1.0))

        ma_b0, crossing_time = _place_conjunction(
            altitude_km, inclinations[k - 1], raans[k - 1], mean_anomalies_rad[k - 1],
            candidate_inclination, candidate_raan, epoch, target_time_s, period,
        )
        inclinations.append(candidate_inclination)
        raans.append(candidate_raan)
        mean_anomalies_rad.append(ma_b0)
        crossing_times.append(crossing_time)

    objects = []
    for idx in range(chain_length):
        elements = _make_elements(epoch, altitude_km, inclinations[idx], raans[idx], mean_anomalies_rad[idx])
        objects.append(_make_object(_object_id(seed, idx), f"CHAIN-{seed}-{idx}", elements, operator))

    provenance = {
        "altitude_km": altitude_km,
        "chain_length": chain_length,
        "crossing_times_s": crossing_times,
        "inclinations_deg": inclinations,
        "raans_deg": raans,
    }
    return objects, window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_star(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    hub_inclination = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    fanout = int(overrides.get("fanout", 4))
    if fanout < 3:
        raise ScenarioError("star scenarios need fanout >= 3 to exhibit a real hub")
    period = _period_s(altitude_km)
    window_duration_s = float(overrides.get("window_duration_s", period))

    hub_operator = Operator(identifier=f"STAR-HUB-{seed}", name="Star hub fleet", maneuverable=True)
    leaf_operator = Operator(identifier=f"STAR-LEAF-{seed}", name="Star leaf hazards", maneuverable=False)

    hub_ma0 = 0.0
    hub_elements = _make_elements(epoch, altitude_km, hub_inclination, 0.0, hub_ma0)
    objects = [_make_object(_object_id(seed, 0), f"STAR-{seed}-HUB", hub_elements, hub_operator)]

    crossing_times = []
    for j in range(fanout):
        sign = 1.0 if j % 2 == 0 else -1.0
        leaf_inclination = hub_inclination + sign * (4.0 + 2.0 * j)
        leaf_raan = _wrap_deg(360.0 * (j + 1.0) / (fanout + 1.0) + float(rng.uniform(-8.0, 8.0)))
        target_time_s = window_duration_s * ((j + 1.0) / (fanout + 1.0))

        ma_leaf0, crossing_time = _place_conjunction(
            altitude_km, hub_inclination, 0.0, hub_ma0, leaf_inclination, leaf_raan, epoch, target_time_s, period,
        )
        crossing_times.append(crossing_time)
        elements = _make_elements(epoch, altitude_km, leaf_inclination, leaf_raan, ma_leaf0)
        objects.append(
            _make_object(
                _object_id(seed, j + 1), f"STAR-{seed}-LEAF-{j}", elements, leaf_operator,
                object_type=ObjectType.DEBRIS,
            )
        )

    provenance = {"altitude_km": altitude_km, "fanout": fanout, "crossing_times_s": crossing_times}
    return objects, window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_clique(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    inclination = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    clique_size = int(overrides.get("clique_size", 4))
    if clique_size < 3:
        raise ScenarioError("clique scenarios need clique_size >= 3 to exhibit a real cluster")
    spacing_deg = float(overrides.get("spacing_deg", 0.05))
    period = _period_s(altitude_km)
    window_duration_s = float(overrides.get("window_duration_s", period))

    operator = Operator(identifier=f"CLIQUE-FLEET-{seed}", name="Clique fleet", maneuverable=True)
    base_ma_deg = float(rng.uniform(0.0, 360.0))

    objects = []
    for idx in range(clique_size):
        ma_rad = math.radians(base_ma_deg + idx * spacing_deg)
        elements = _make_elements(epoch, altitude_km, inclination, 0.0, ma_rad)
        objects.append(_make_object(_object_id(seed, idx), f"CLIQUE-{seed}-{idx}", elements, operator))

    along_track_span_km = math.radians(spacing_deg * (clique_size - 1)) * (R_EARTH_KM + altitude_km)
    provenance = {
        "altitude_km": altitude_km,
        "clique_size": clique_size,
        "spacing_deg": spacing_deg,
        "along_track_span_km": along_track_span_km,
    }
    return objects, window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_intra_plane(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    inclination = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    n_pairs = int(overrides.get("n_pairs", 1))
    # Matches aegis.ingest.synthetic's _CONJUNCTION_MA_OFFSET_DEG rationale:
    # at 550 km this is ~10 km along-track (well inside the Starlink 44 km
    # box) and, because both satellites share one mean motion, a relative
    # speed of order 1 cm/s -- squarely below LOW_RELATIVE_VELOCITY_KM_S.
    offset_deg = float(overrides.get("ma_offset_deg", 0.08))
    period = _period_s(altitude_km)
    window_duration_s = float(overrides.get("window_duration_s", period))

    operator = Operator(identifier=f"INTRA-PLANE-FLEET-{seed}", name="Intra-plane fleet", maneuverable=True)

    objects = []
    relative_speeds = []
    for pair_idx in range(max(n_pairs, 1)):
        base_ma_deg = float(rng.uniform(0.0, 360.0))
        ma_a_rad = math.radians(base_ma_deg)
        ma_b_rad = math.radians(base_ma_deg + offset_deg)
        elements_a = _make_elements(epoch, altitude_km, inclination, 0.0, ma_a_rad)
        elements_b = _make_elements(epoch, altitude_km, inclination, 0.0, ma_b_rad)
        objects.append(_make_object(_object_id(seed, 2 * pair_idx), f"INTRA-PLANE-{seed}-{pair_idx}A", elements_a, operator))
        objects.append(_make_object(_object_id(seed, 2 * pair_idx + 1), f"INTRA-PLANE-{seed}-{pair_idx}B", elements_b, operator))
        relative_speeds.append(_relative_speed_km_s(altitude_km, inclination, 0.0, ma_a_rad, inclination, 0.0, ma_b_rad))

    provenance = {"altitude_km": altitude_km, "ma_offset_deg": offset_deg, "relative_speed_km_s": relative_speeds}
    return objects, window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_crossing_planes(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    inclination_a = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    # Same inclination as the primary, ~105 deg RAAN apart: relative speed
    # is still ~5.5-6 km/s from the RAAN difference alone (two same-altitude
    # circular orbits meeting at this angle), and -- unlike combining this
    # with a large inclination *difference* -- it does not add a several-km
    # differential J2 short-period radial bias between the two orbits at
    # the crossing point (that bias scales with sin^2(i) and is identical
    # here since both share one inclination). Verified numerically: with a
    # large inclination split (e.g. 53/97.4) the true SGP4 minimum
    # separation floors around ~2 km regardless of mean-anomaly tuning --
    # right at the screening box's radial gate -- while this combination
    # floors under 1 km. See the self-check script's table.
    inclination_b = float(overrides.get("secondary_inclination_deg", inclination_a))
    raan_b = float(overrides.get("secondary_raan_deg", 105.0)) + float(rng.uniform(-5.0, 5.0))
    period = _period_s(altitude_km)
    target_time_s = float(overrides.get("target_time_s", period * 0.2))

    fleet_operator = Operator(identifier=f"CROSSING-FLEET-{seed}", name="Crossing-plane fleet", maneuverable=True)
    hazard_operator = Operator(identifier=f"CROSSING-HAZARD-{seed}", name="Crossing-plane hazard", maneuverable=False)

    ma_a0 = 0.0
    ma_b0, crossing_time = _place_conjunction(
        altitude_km, inclination_a, 0.0, ma_a0, inclination_b, raan_b, epoch, target_time_s, period,
    )
    # Picking the candidate with the best real (SGP4) separation, rather
    # than the one purely closest to target_time_s, can land the crossing
    # later than a window sized off target_time_s alone would cover -- so
    # size the window from the crossing actually achieved, not the target.
    # (No risk of also catching the *other* antipodal crossing here the way
    # isolated-pair must avoid it: this family makes no single-edge claim.)
    window_duration_s = float(overrides.get("window_duration_s", crossing_time + 300.0))
    relative_speed = _relative_speed_km_s(altitude_km, inclination_a, 0.0, ma_a0, inclination_b, raan_b, ma_b0)

    primary = _make_object(
        _object_id(seed, 0), f"CROSSING-PLANES-{seed}-PRIMARY",
        _make_elements(epoch, altitude_km, inclination_a, 0.0, ma_a0), fleet_operator,
    )
    secondary = _make_object(
        _object_id(seed, 1), f"CROSSING-PLANES-{seed}-SECONDARY",
        _make_elements(epoch, altitude_km, inclination_b, raan_b, ma_b0), hazard_operator,
        object_type=ObjectType.DEBRIS,
    )

    provenance = {
        "altitude_km": altitude_km,
        "inclination_a_deg": inclination_a,
        "inclination_b_deg": inclination_b,
        "raan_b_deg": raan_b,
        "crossing_time_s": crossing_time,
        "relative_speed_km_s": relative_speed,
    }
    return [primary, secondary], window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_debris_shower(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    primary_inclination = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    debris_inclination = float(overrides.get("debris_inclination_deg", primary_inclination + 15.0))
    debris_raan = float(overrides.get("debris_raan_deg", 45.0))
    debris_count = int(overrides.get("debris_count", 8))
    if debris_count < 5:
        raise ScenarioError("debris-shower scenarios need debris_count >= 5 non-maneuverable secondaries")
    spacing_deg = float(overrides.get("spacing_deg", 0.12))
    period = _period_s(altitude_km)
    window_duration_s = float(overrides.get("window_duration_s", period))
    target_time_s = float(overrides.get("target_time_s", window_duration_s / 2.0))

    fleet_operator = Operator(identifier=f"SHOWER-FLEET-{seed}", name="Debris-shower fleet", maneuverable=True)
    debris_operator = Operator(identifier=f"SHOWER-DEBRIS-{seed}", name="Fragmentation debris", maneuverable=False)

    primary_ma0 = 0.0
    base_debris_ma0, crossing_time = _place_conjunction(
        altitude_km, primary_inclination, 0.0, primary_ma0, debris_inclination, debris_raan, epoch, target_time_s, period,
    )

    objects = [
        _make_object(
            _object_id(seed, 0), f"DEBRIS-SHOWER-{seed}-PRIMARY",
            _make_elements(epoch, altitude_km, primary_inclination, 0.0, primary_ma0), fleet_operator,
        )
    ]
    # All debris share one plane (one fragmentation event) with a small mean
    # -anomaly spread centered on the solved crossing point, so the primary
    # flies through several of them in the same pass -- the along-track span
    # for debris_count=8 at spacing_deg=0.12 is ~1.4 * (n-1) km, comfortably
    # inside the 44 km Starlink screening box.
    half_span = spacing_deg * (debris_count - 1) / 2.0
    for j in range(debris_count):
        offset_deg = -half_span + j * spacing_deg
        ma_rad = base_debris_ma0 + math.radians(offset_deg)
        elements = _make_elements(epoch, altitude_km, debris_inclination, debris_raan, ma_rad)
        objects.append(
            _make_object(
                _object_id(seed, j + 1), f"DEBRIS-SHOWER-{seed}-FRAG-{j}", elements, debris_operator,
                object_type=ObjectType.DEBRIS,
            )
        )

    provenance = {
        "altitude_km": altitude_km,
        "debris_count": debris_count,
        "crossing_time_s": crossing_time,
        "spacing_deg": spacing_deg,
    }
    return objects, window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_dense_shell(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    inclination = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    n_planes = int(overrides.get("n_planes", 6))
    sats_per_plane = int(overrides.get("sats_per_plane", 22))
    window_duration_s = float(overrides.get("window_duration_s", SCREENING_HORIZON_S))

    operator = Operator(identifier=f"DENSE-SHELL-FLEET-{seed}", name="Dense shell fleet", maneuverable=True)
    raan_offset = float(rng.uniform(0.0, 360.0))
    ma_offset = float(rng.uniform(0.0, 360.0))

    objects = []
    idx = 0
    n_planes = max(n_planes, 0)
    sats_per_plane = max(sats_per_plane, 0)
    for plane in range(n_planes):
        raan = _wrap_deg(raan_offset + (plane * 360.0 / n_planes if n_planes else 0.0))
        # Walker-delta phasing: successive planes are offset by a fraction
        # of the in-plane spacing so satellites interleave rather than line
        # up radially -- the same pattern real Walker shells use.
        walker_phase = plane * (360.0 / (n_planes * sats_per_plane)) if n_planes and sats_per_plane else 0.0
        for sat in range(sats_per_plane):
            ma_deg = _wrap_deg(ma_offset + sat * (360.0 / sats_per_plane if sats_per_plane else 0.0) + walker_phase)
            elements = _make_elements(epoch, altitude_km, inclination, raan, math.radians(ma_deg))
            objects.append(_make_object(_object_id(seed, idx), f"DENSE-SHELL-{seed}-{idx}", elements, operator))
            idx += 1

    provenance = {
        "altitude_km": altitude_km,
        "n_planes": n_planes,
        "sats_per_plane": sats_per_plane,
        "n_objects": len(objects),
    }
    return objects, window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _default_fixture_dir() -> Path:
    # generator.py -> scenarios -> aegis (package) -> src -> aegis (project root)
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures"


def _build_induced_cascade(
    rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict
) -> _BuiltFamily:
    """One forced maneuver, and neighbours in the way of it.

    This is the family the induced-conjunction question is actually about,
    and none of the others produce it. The geometry is deliberately the one
    Chen et al. measured at Starlink scale (arXiv:2406.06068, where 81.4 % of
    Starlink-on-Starlink maneuvers are cascade-induced and a single external
    event triggered as many as 41 induced maneuvers):

    * one maneuverable **primary**, in a Walker-like plane;
    * an external, non-maneuverable **hazard** on a crossing orbit, placed to
      pass close enough that the primary must move;
    * a ring of maneuverable **neighbours** in the primary's own plane, spaced
      a few kilometres along-track ahead and behind.

    The avoidance burn is along-track -- that is the only axis with unbounded
    response -- so it displaces the primary *along the line its neighbours sit
    on*. A fuel-optimal plan will happily slide it into one of them; a plan
    that must not create a new conjunction has to pick the direction and
    magnitude that threads between them, or spend the extra delta-v to clear
    the far side. The difference between those two plans is the safety
    premium this project exists to measure.

    ``neighbour_spacing_km`` controls how hard the problem is. The default
    12 km is a little over the ~9.7 km along-track separation
    :mod:`aegis.ingest.synthetic` uses for its own known-conjunction triple,
    and comfortably inside the 44 km along-track screening box, so the
    neighbours are genuinely latent: close enough to be reachable, far enough
    not to be conjunctions already.
    """
    altitude_km = float(overrides.get("altitude_km", _DEFAULT_ALTITUDE_KM))
    inclination = float(overrides.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
    neighbours = int(overrides.get("neighbour_count", 4))
    # Spacing is the difficulty knob and it is sampled per seed, so a suite of
    # seeds sweeps the coupling regime rather than repeating one geometry.
    # Measured on this family: about 0-3 % premium at 12 km, 3-7 % at 6 km,
    # and no zero-induced plan at all at 3 km. The range brackets that
    # transition deliberately.
    spacing_km = float(overrides.get("neighbour_spacing_km", rng.uniform(3.0, 15.0)))
    hazard_raan_offset = float(overrides.get("hazard_raan_deg", 105.0)) + float(
        rng.uniform(-4.0, 4.0)
    )
    period = _period_s(altitude_km)
    target_time_s = float(overrides.get("target_time_s", period * 0.45))

    fleet = Operator(identifier=f"CASCADE-FLEET-{seed}", name="Cascade fleet", maneuverable=True)
    hazard_operator = Operator(
        identifier=f"CASCADE-HAZARD-{seed}", name="Cascade hazard", maneuverable=False
    )

    radius_km = R_EARTH_KM + altitude_km
    spacing_deg = math.degrees(spacing_km / radius_km)

    primary_ma_rad = math.radians(float(rng.uniform(0.0, 360.0)))
    primary_elements = _make_elements(epoch, altitude_km, inclination, 0.0, primary_ma_rad)
    objects = [
        _make_object(_object_id(seed, 0), f"CASCADE-{seed}-PRIMARY", primary_elements, fleet)
    ]

    # Neighbours alternate ahead and behind so the primary is boxed in on both
    # sides; a one-sided ring would let the optimizer escape for free in the
    # other direction and the premium would be identically zero by
    # construction rather than by measurement.
    offsets_deg: list[float] = []
    for index in range(max(neighbours, 0)):
        step = index // 2 + 1
        sign = 1.0 if index % 2 == 0 else -1.0
        jitter = float(rng.uniform(-0.15, 0.15)) * spacing_deg
        offsets_deg.append(sign * step * spacing_deg + jitter)

    for index, offset_deg in enumerate(offsets_deg):
        neighbour_ma = primary_ma_rad + math.radians(offset_deg)
        elements = _make_elements(epoch, altitude_km, inclination, 0.0, neighbour_ma)
        objects.append(
            _make_object(
                _object_id(seed, 1 + index),
                f"CASCADE-{seed}-NEIGHBOUR-{index}",
                elements,
                fleet,
            )
        )

    hazard_ma, crossing_time = _place_conjunction(
        altitude_km,
        inclination,
        0.0,
        primary_ma_rad,
        inclination,
        hazard_raan_offset,
        epoch,
        target_time_s,
        period,
    )
    hazard_elements = _make_elements(
        epoch, altitude_km, inclination, hazard_raan_offset, hazard_ma
    )
    objects.append(
        _make_object(
            _object_id(seed, 1 + len(offsets_deg)),
            f"CASCADE-{seed}-HAZARD",
            hazard_elements,
            hazard_operator,
            object_type=ObjectType.DEBRIS,
        )
    )

    window_duration_s = float(
        overrides.get("window_duration_s", max(period, crossing_time + 0.35 * period))
    )
    provenance = {
        "altitude_km": altitude_km,
        "inclination_deg": inclination,
        "neighbour_count": len(offsets_deg),
        "neighbour_spacing_km": spacing_km,
        "neighbour_offsets_deg": [round(value, 6) for value in offsets_deg],
        "hazard_raan_deg": hazard_raan_offset,
        "crossing_time_s": crossing_time,
        "relative_speed_km_s": _relative_speed_km_s(
            altitude_km, inclination, 0.0, primary_ma_rad, inclination, hazard_raan_offset, hazard_ma
        ),
    }
    return objects, window_duration_s, epoch, provenance, SCREENING_BOX_STARLINK_KM


def _build_replay_tle(rng: np.random.Generator, seed: int, epoch: datetime, overrides: dict) -> _BuiltFamily:
    fixture_dir = Path(overrides.get("fixture_dir", _default_fixture_dir()))
    fixture_names = tuple(overrides.get("fixtures", ("starlink_slice.tle", "debris_slice.tle")))
    window_duration_s = float(overrides.get("window_duration_s", SCREENING_HORIZON_S))

    catalog: Catalog | None = None
    for name in fixture_names:
        path = fixture_dir / name
        try:
            piece = catalog_from_tle_file(path)
        except CelesTrakError as error:
            raise ScenarioError(f"replay-tle fixture {path} failed to parse: {error}") from error
        catalog = piece if catalog is None else catalog.merge(piece)
    if catalog is None or len(catalog) == 0:
        raise ScenarioError("replay-tle produced no objects from the configured fixtures")

    # Real name-based operator/type enrichment, not synthesis: the elements
    # are exactly what the fixture TLEs say. Starlink genuinely is
    # maneuverable and operator-owned; catalogued debris genuinely is not.
    starlink_operator = Operator(identifier="STARLINK", name="Starlink", maneuverable=True)
    objects = list(catalog.objects)
    for obj in objects:
        upper_name = obj.name.upper()
        if upper_name.startswith("STARLINK"):
            obj.operator = starlink_operator
            obj.object_type = ObjectType.PAYLOAD
        elif "DEB" in upper_name:
            obj.object_type = ObjectType.DEBRIS

    # replay-tle synthesizes nothing, but still draws from rng once so that
    # "same seed -> same output" remains literally true of every family's
    # call signature rather than a documented exception for this one.
    _ = rng.integers(0, 2**31 - 1)

    window_start = min(obj.elements.epoch for obj in objects if obj.elements is not None)
    provenance = {"fixtures": list(fixture_names), "fixture_dir": str(fixture_dir), "n_objects": len(objects)}
    return objects, window_duration_s, window_start, provenance, SCREENING_BOX_TLE_GRADE_KM


_FAMILY_BUILDERS = {
    "isolated-pair": _build_isolated_pair,
    "chain": _build_chain,
    "star": _build_star,
    "clique": _build_clique,
    "intra-plane": _build_intra_plane,
    "crossing-planes": _build_crossing_planes,
    "debris-shower": _build_debris_shower,
    "dense-shell": _build_dense_shell,
    "induced-cascade": _build_induced_cascade,
    "replay-tle": _build_replay_tle,
}


def _provenance_params(overrides: dict) -> dict:
    """JSON-friendly copy of ``overrides`` for storage in ``Scenario.provenance``."""
    params = {}
    for key, value in overrides.items():
        if isinstance(value, datetime):
            params[key] = format_epoch(ensure_utc(value))
        else:
            params[key] = value
    return params


def generate(
    family: str,
    seed: int,
    *,
    authorization: SyntheticAuthorization | None = None,
    **overrides: Any,
) -> Scenario:
    """Build one scenario. Pure: the same ``(family, seed, overrides)`` always
    produces byte-identical objects (see ``scenario_digest``).

    Parameters
    ----------
    family
        One of :data:`FAMILIES`.
    seed
        Drives the only randomness this function uses --
        ``numpy.random.default_rng(seed)``. Never Python's ``random``
        module, never global NumPy state, never wall-clock time.
    authorization
        Required (and checked against ``AEGIS_ALLOW_SYNTHETIC=1``) for every
        family except ``replay-tle``, which reads real committed TLE data
        and needs no such gate.
    **overrides
        Family-specific parameters (e.g. ``altitude_km``, ``chain_length``,
        ``fanout``, ``debris_count``, ``window_duration_s``, ``epoch``,
        ``window_start``). Unknown keys are accepted and simply unused by a
        given family's builder, so a caller can pass one override dict
        across a mixed-family sweep.
    """
    if family not in _FAMILY_BUILDERS:
        raise ScenarioError(f"unknown scenario family: {family!r}; choices are {FAMILIES}")
    if family in _SYNTHETIC_FAMILIES:
        _require_synthetic_authorization(authorization)

    rng = np.random.default_rng(seed)
    epoch = ensure_utc(overrides.get("epoch", DEFAULT_SYNTHETIC_EPOCH))

    builder = _FAMILY_BUILDERS[family]
    objects, window_duration_s, window_start, extra_provenance, screening_box_km = builder(
        rng, seed, epoch, overrides
    )

    if "window_start" in overrides:
        window_start = ensure_utc(overrides["window_start"])
    if "window_duration_s" in overrides:
        window_duration_s = float(overrides["window_duration_s"])
    if "screening_box_km" in overrides:
        screening_box_km = tuple(overrides["screening_box_km"])

    spec = ScenarioSpec(family=family, seed=seed, params=_provenance_params(overrides))
    provenance = {
        "description": describe(family),
        "generator": "aegis.scenarios.generator.generate",
        "params": spec.params,
        **extra_provenance,
    }

    scenario = Scenario(
        scenario_id=f"{family}-{seed}",
        objects=objects,
        window_start=window_start,
        window_duration_s=window_duration_s,
        family=family,
        seed=seed,
        provenance=provenance,
        screening_box_km=screening_box_km,
    )
    digest = scenario_digest(scenario)
    scenario.scenario_id = f"{family}-{seed}-{digest[:10]}"
    scenario.provenance["scenario_digest"] = digest
    return scenario
