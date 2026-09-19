"""The screening box behaves as SpaceX Space Safety defines it.

Half-widths 2 km radial, 44 km along-track, 51 km out of plane, in each
object's own RTN frame (so the box turns with the orbit), checked from both
objects, judged at the closest approach (so the grid step cannot change the
answer).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_screening_sweep import EPOCH, _crowd, _satellite

from aegis.constants import SCREENING_BOX_STARLINK_KM
from aegis.screening import screen
from aegis.screening.results import box_membership

RADIUS_KM = 6928.0
SPEED_KM_S = 7.59


def _state(angle_rad: float, *, inclination_rad: float = 0.0):
    """Circular-orbit position and velocity at ``angle_rad`` along the orbit."""
    position = RADIUS_KM * np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])
    velocity = SPEED_KM_S * np.array([-math.sin(angle_rad), math.cos(angle_rad), 0.0])
    tilt = np.array(
        [[1, 0, 0], [0, math.cos(inclination_rad), -math.sin(inclination_rad)], [0, math.sin(inclination_rad), math.cos(inclination_rad)]]
    )
    return tilt @ position, tilt @ velocity


def _membership(primary, offset_rtn, secondary_velocity=None):
    p_pos, p_vel = primary
    r_hat = p_pos / np.linalg.norm(p_pos)
    n_hat = np.cross(p_pos, p_vel)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    s_pos = p_pos + offset_rtn[0] * r_hat + offset_rtn[1] * t_hat + offset_rtn[2] * n_hat
    s_vel = p_vel if secondary_velocity is None else secondary_velocity
    in_p, in_s, rel, _ = box_membership(p_pos[None], p_vel[None], s_pos[None], s_vel[None], SCREENING_BOX_STARLINK_KM)
    return bool(in_p[0]), bool(in_s[0]), rel[0]


@pytest.mark.parametrize("angle_deg", [0, 37, 90, 180, 271])
@pytest.mark.parametrize("inclination_deg", [0, 53, 97.6])
def test_box_turns_with_the_orbit(angle_deg: float, inclination_deg: float) -> None:
    primary = _state(math.radians(angle_deg), inclination_rad=math.radians(inclination_deg))
    assert _membership(primary, (0.0, 40.0, 0.0))[0]  # 40 km ahead: inside (along-track 44)
    assert _membership(primary, (0.0, 0.0, 50.0))[0]  # 50 km out of plane: inside (51)
    assert _membership(primary, (1.9, 0.0, 0.0))[0]  # 1.9 km above: inside (radial 2)
    assert not _membership(primary, (2.1, 0.0, 0.0))[0]  # 2.1 km above: outside
    assert not _membership(primary, (0.0, 45.0, 0.0))[0]  # 45 km ahead: outside
    assert not _membership(primary, (0.0, 0.0, 52.0))[0]  # 52 km out of plane: outside


def test_relative_position_is_reported_in_the_primarys_frame() -> None:
    primary = _state(math.radians(123), inclination_rad=math.radians(53))
    _in_p, _in_s, rel = _membership(primary, (1.0, -30.0, 20.0))
    assert rel == pytest.approx([1.0, -30.0, 20.0], abs=1e-9)


def test_either_objects_box_counts() -> None:
    # Crossing orbits: 48 km ahead of the primary is outside its box (44 km),
    # but for a secondary crossing at right angles that offset is out of *its*
    # plane (51 km allowed) -- inside its box.
    primary = _state(0.0)
    crossing_velocity = np.array([0.0, 0.0, SPEED_KM_S])
    in_primary, in_secondary, _ = _membership(primary, (0.0, 48.0, 0.0), secondary_velocity=crossing_velocity)
    assert not in_primary
    assert in_secondary


def _passes(conjunctions) -> set[tuple[str, str, int]]:
    """Encounters with a real closing speed, keyed by pair and TCA (to the second)."""
    return {
        (c.primary.object_id, c.secondary.object_id, round(c.tca.timestamp()))
        for c in conjunctions
        if not c.metadata.get("low_relative_velocity")
    }


def _slow_pairs(conjunctions) -> set[tuple[str, str]]:
    return {(c.primary.object_id, c.secondary.object_id) for c in conjunctions if c.metadata.get("low_relative_velocity")}


def _crossing_crowd():
    """The slow crowd plus satellites in steeper planes that cross the co-orbiting
    pair (7001/7002, mean anomaly 100 deg) at every node, at 1-2 km/s."""
    objects = _crowd()
    for k, (inclination, anomaly) in enumerate([(63.0, 100.0), (63.0, 100.02), (43.0, 100.05), (70.0, 99.97)]):
        objects.append(_satellite(f"{9100 + k}", mean_anomaly_deg=anomaly, inclination_deg=inclination))
    return objects


def test_acceptance_does_not_depend_on_the_grid_step() -> None:
    # Passes are judged at their closest approach, so the step cannot change them.
    # A co-orbiting pair (7001/7002, 50 m apart all window) has no single closest
    # approach -- its range only wobbles -- so it must be *found* at every step,
    # but which wobble is reported may move; that is the standard low-relative-
    # velocity caveat, flagged on the conjunction.
    objects = _crossing_crowd()
    results = {step: screen(objects, EPOCH, 3 * 3600.0, step_s=step) for step in (60.0, 30.0, 20.0)}
    reference = _passes(results[60.0])
    assert reference, "the crowd must produce passes"
    assert ("7001", "7002") in _slow_pairs(results[60.0])
    for step in (30.0, 20.0):
        assert _passes(results[step]) == reference
        assert _slow_pairs(results[step]) == _slow_pairs(results[60.0])


def test_conjunctions_say_which_box_caught_them() -> None:
    conjunctions = screen(_crowd(), EPOCH, 3 * 3600.0)
    assert conjunctions
    assert {c.metadata["screening_box"] for c in conjunctions} <= {"primary", "secondary", "both"}
