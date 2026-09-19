"""Shared screening geometry: primary assignment, conjunction ids, RTN relatives."""

from __future__ import annotations

import numpy as np

from ..core.frames import rtn_to_eci_matrix
from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import format_epoch

__all__ = [
    "assign_primary",
    "conjunction_id",
    "relative_rtn",
    "rtn_to_eci_batch",
]


def assign_primary(object_a: SpaceObject, object_b: SpaceObject) -> tuple[SpaceObject, SpaceObject]:
    """Choose primary / secondary for a pair.

    Primary is the maneuverable object if exactly one of the pair is
    maneuverable; otherwise the lower ``object_id`` is primary.
    """
    a_maneuverable = object_a.is_maneuverable
    b_maneuverable = object_b.is_maneuverable
    if a_maneuverable and not b_maneuverable:
        return object_a, object_b
    if b_maneuverable and not a_maneuverable:
        return object_b, object_a
    if object_a.object_id <= object_b.object_id:
        return object_a, object_b
    return object_b, object_a


def conjunction_id(primary_id: str, secondary_id: str, tca) -> str:
    """Stable identifier for a pair and TCA.

    Same pair and same TCA always produce the same id. The pair is stored
    in assigned primary/secondary order so the convention is part of the key.
    """
    return f"{primary_id}:{secondary_id}:{format_epoch(tca)}"


def relative_rtn(
    primary: StateVector, secondary: StateVector
) -> tuple[np.ndarray, np.ndarray]:
    """Relative state of ``secondary`` in the primary's RTN frame.

    ``R`` has RTN basis vectors as columns, so ``R.T`` maps inertial
    (world) vectors into RTN.
    """
    rotation = rtn_to_eci_matrix(primary.position_km, primary.velocity_km_s)
    position_rtn = rotation.T @ (secondary.position_km - primary.position_km)
    velocity_rtn = rotation.T @ (secondary.velocity_km_s - primary.velocity_km_s)
    return position_rtn, velocity_rtn


def rtn_to_eci_batch(
    positions_km: np.ndarray, velocities_km_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``rtn_to_eci_matrix`` for many objects.

    Returns
    -------
    rotations, valid
        ``rotations`` has shape ``(n, 3, 3)`` with RTN basis vectors as
        columns. ``valid`` is false where the RTN frame is undefined
        (zero radius or zero angular momentum).
    """
    positions_km = np.asarray(positions_km, dtype=float)
    velocities_km_s = np.asarray(velocities_km_s, dtype=float)

    radius = np.linalg.norm(positions_km, axis=1)
    angular_momentum = np.cross(positions_km, velocities_km_s)
    h_norm = np.linalg.norm(angular_momentum, axis=1)
    valid = (radius > 0.0) & (h_norm > 0.0)

    radius_safe = np.where(radius > 0.0, radius, 1.0)
    h_safe = np.where(h_norm > 0.0, h_norm, 1.0)

    r_hat = positions_km / radius_safe[:, None]
    n_hat = angular_momentum / h_safe[:, None]
    t_hat = np.cross(n_hat, r_hat)
    rotations = np.stack((r_hat, t_hat, n_hat), axis=2)
    return rotations, valid
