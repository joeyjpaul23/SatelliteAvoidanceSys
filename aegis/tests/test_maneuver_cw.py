"""Clohessy-Wiltshire and miss-distance helpers (Step 4).

Writes against the public ``aegis.maneuver`` surface and allowed fixtures.
Does not import ``aegis.maneuver`` submodules.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aegis.constants import MU_EARTH_KM3_S2, PC_TARGET_POST_MANEUVER, R_EARTH_KM
from aegis.maneuver import (
    ManeuverSolverError,
    along_track_response_km,
    clohessy_wiltshire_state,
    plan_maneuvers,
    required_miss_distance_km,
    rescreen_until_stable,
)
from aegis.risk.alfano import collision_probability


def _leo_mean_motion_rad_s(altitude_km: float = 550.0) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    return math.sqrt(MU_EARTH_KM3_S2 / semi_major_km**3)


def test_maneuver_public_exports() -> None:
    assert callable(clohessy_wiltshire_state)
    assert callable(along_track_response_km)
    assert callable(required_miss_distance_km)
    assert callable(plan_maneuvers)
    assert callable(rescreen_until_stable)
    assert isinstance(ManeuverSolverError, type)


def test_clohessy_wiltshire_state_dt_zero_is_identity() -> None:
    n = _leo_mean_motion_rad_s()
    r0 = np.array([1.25, -0.8, 0.4], dtype=float)
    v0 = np.array([0.01, -0.02, 0.03], dtype=float)

    r, v = clohessy_wiltshire_state(n, 0.0, r0, v0)

    np.testing.assert_allclose(r, r0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(v, v0, rtol=0.0, atol=1e-12)


def test_clohessy_wiltshire_pure_along_track_from_origin() -> None:
    n = _leo_mean_motion_rad_s()
    dt = 0.15 * (2.0 * math.pi / n)
    dv = 1.0e-3
    r0 = np.zeros(3)
    v0 = np.array([0.0, dv, 0.0])

    r, _v = clohessy_wiltshire_state(n, dt, r0, v0)

    ndt = n * dt
    expected_radial = (2.0 / n) * (1.0 - math.cos(ndt)) * dv
    expected_along = (4.0 * math.sin(ndt) - 3.0 * ndt) / n * dv

    assert r[0] == pytest.approx(expected_radial, rel=1e-9, abs=1e-12)
    assert r[1] == pytest.approx(expected_along, rel=1e-9, abs=1e-12)
    assert r[2] == pytest.approx(0.0, abs=1e-12)


def test_clohessy_wiltshire_pure_cross_track_from_origin() -> None:
    n = _leo_mean_motion_rad_s()
    dt = 0.15 * (2.0 * math.pi / n)
    dv = 1.0e-3
    r0 = np.zeros(3)
    v0 = np.array([0.0, 0.0, dv])

    r, _v = clohessy_wiltshire_state(n, dt, r0, v0)

    expected_z = math.sin(n * dt) / n * dv
    assert r[0] == pytest.approx(0.0, abs=1e-12)
    assert r[1] == pytest.approx(0.0, abs=1e-12)
    assert r[2] == pytest.approx(expected_z, rel=1e-9, abs=1e-12)


def test_quarter_orbit_along_track_not_within_10_percent_of_3dvdt() -> None:
    n = _leo_mean_motion_rad_s()
    dt = 0.25 * (2.0 * math.pi / n)
    dv = 1.0
    r0 = np.zeros(3)
    v0 = np.array([0.0, dv, 0.0])

    r, _v = clohessy_wiltshire_state(n, dt, r0, v0)

    shortcut = 3.0 * dv * dt
    assert abs(float(r[1]) - shortcut) > 0.10 * abs(shortcut)


def test_along_track_response_km_matches_along_track_formula() -> None:
    n = _leo_mean_motion_rad_s()
    dt = 0.15 * (2.0 * math.pi / n)
    dv = 1.0e-3
    ndt = n * dt
    expected = (4.0 * math.sin(ndt) - 3.0 * ndt) / n * dv

    got = along_track_response_km(n, dt, dv)
    assert float(got) == pytest.approx(expected, rel=1e-9, abs=1e-12)

    r, _v = clohessy_wiltshire_state(
        n, dt, np.zeros(3), np.array([0.0, dv, 0.0])
    )
    assert float(got) == pytest.approx(float(r[1]), rel=1e-9, abs=1e-12)


def test_required_miss_distance_monotonic_in_target_pc() -> None:
    hard_body_km = 0.010
    sigma_major_km = 1.0
    sigma_minor_km = 0.30

    miss_loose = required_miss_distance_km(
        hard_body_km, sigma_major_km, sigma_minor_km, target_pc=1e-4
    )
    miss_mid = required_miss_distance_km(
        hard_body_km, sigma_major_km, sigma_minor_km, target_pc=1e-6
    )
    miss_tight = required_miss_distance_km(
        hard_body_km, sigma_major_km, sigma_minor_km, target_pc=1e-8
    )

    assert miss_mid >= miss_loose
    assert miss_tight >= miss_mid

    default_miss = required_miss_distance_km(
        hard_body_km, sigma_major_km, sigma_minor_km
    )
    named_default = required_miss_distance_km(
        hard_body_km,
        sigma_major_km,
        sigma_minor_km,
        target_pc=PC_TARGET_POST_MANEUVER,
    )
    assert default_miss == pytest.approx(named_default, rel=0.0, abs=0.0)


def test_required_miss_distance_alfano_at_or_below_target_or_sentinel() -> None:
    hard_body_km = 0.010
    sigma_major_km = 1.0
    sigma_minor_km = 0.30
    targets = (1e-4, 1e-6, PC_TARGET_POST_MANEUVER, 1e-8)

    for target_pc in targets:
        miss_km = required_miss_distance_km(
            hard_body_km, sigma_major_km, sigma_minor_km, target_pc=target_pc
        )
        assert miss_km >= 0.0
        probability = collision_probability(
            sigma_major_km,
            sigma_minor_km,
            miss_km,
            0.0,
            hard_body_km,
        )
        sentinel = miss_km >= 90.0
        assert probability <= target_pc or sentinel

    impossible = required_miss_distance_km(
        10.0, 1.0, 0.30, target_pc=1e-20
    )
    assert impossible == pytest.approx(100.0, rel=0.1, abs=5.0)
    assert not math.isnan(impossible)
