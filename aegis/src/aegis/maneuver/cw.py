"""Exact Clohessy-Wiltshire relative motion.

The along-track response to a tangential impulse is

    (4 * sin(n*dt) - 3*n*dt) / n * dv

never the ``3 * dv * dt`` shortcut. That shortcut is wrong by about 85% at
a quarter-orbit lead; this module always uses the exact linear map.
"""

from __future__ import annotations

import numpy as np

__all__ = ["clohessy_wiltshire_state", "along_track_response_km"]

_N_FLOOR = 1e-15


def clohessy_wiltshire_state(
    n_rad_s: float,
    dt_s: float,
    r0_rtn_km: np.ndarray,
    v0_rtn_km_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact linear Clohessy-Wiltshire state at ``dt_s``.

    Parameters
    ----------
    n_rad_s
        Chief mean motion, rad/s.
    dt_s
        Propagation interval, seconds.
    r0_rtn_km
        Initial relative position in RTN, shape ``(3,)``.
    v0_rtn_km_s
        Initial relative velocity in RTN, shape ``(3,)``.

    Returns
    -------
    (r_rtn, v_rtn)
        Relative position and velocity at ``dt_s``, each shape ``(3,)``.
    """
    r0 = np.asarray(r0_rtn_km, dtype=float).reshape(3)
    v0 = np.asarray(v0_rtn_km_s, dtype=float).reshape(3)
    n = float(n_rad_s)
    dt = float(dt_s)

    if abs(n) < _N_FLOOR:
        return r0 + v0 * dt, v0.copy()

    nt = n * dt
    c = np.cos(nt)
    s = np.sin(nt)
    x0, y0, z0 = r0
    vx0, vy0, vz0 = v0

    x = (4.0 - 3.0 * c) * x0 + (s / n) * vx0 + (2.0 / n) * (1.0 - c) * vy0
    y = (
        6.0 * (s - nt) * x0
        + y0
        + (2.0 / n) * (c - 1.0) * vx0
        + ((4.0 * s - 3.0 * nt) / n) * vy0
    )
    z = c * z0 + (s / n) * vz0

    vx = 3.0 * n * s * x0 + c * vx0 + 2.0 * s * vy0
    vy = -6.0 * n * (1.0 - c) * x0 - 2.0 * s * vx0 + (4.0 * c - 3.0) * vy0
    vz = -n * s * z0 + c * vz0

    return np.array([x, y, z], dtype=float), np.array([vx, vy, vz], dtype=float)


def along_track_response_km(
    n_rad_s: float,
    dt_s: float,
    dv_along_track_km_s: float,
) -> float:
    """Along-track Clohessy-Wiltshire displacement from a pure tangential impulse.

    Equals the along-track component of :func:`clohessy_wiltshire_state`
    from the origin with ``v0 = [0, dv, 0]``. Uses
    ``(4*sin(n*dt) - 3*n*dt) / n * dv``.
    """
    r_rtn, _ = clohessy_wiltshire_state(
        n_rad_s,
        dt_s,
        np.zeros(3),
        np.array([0.0, float(dv_along_track_km_s), 0.0]),
    )
    return float(r_rtn[1])
