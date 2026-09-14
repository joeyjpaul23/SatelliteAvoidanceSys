"""Reference frame construction and rotation.

Three frames matter in this system:

ECI (Earth-Centred Inertial)
    Where propagation happens and where all state vectors are stored
    internally. SGP4 natively outputs TEME-of-date; we keep everything in
    TEME because relative geometry between two objects is frame-consistent,
    so no conversion is needed for conjunction work.

RTN (Radial / Transverse / Normal, also called RSW or RIC)
    A per-object local orbital frame. CDM covariances are published in RTN,
    screening volumes are expressed in RTN, and maneuvers are naturally
    described in RTN. Every object has its OWN RTN frame, built from its own
    state vector -- a common source of bugs when combining two objects'
    covariances.

Encounter frame (also called the conjunction plane or B-plane)
    A frame built from the RELATIVE state at time of closest approach, in
    which the 2D probability of collision integral is evaluated. One axis is
    aligned with relative velocity and is marginalised away, leaving a 2D
    plane containing the miss vector.

Everything here is pure geometry: no I/O, no orbital dynamics, no algorithms
beyond linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "rtn_basis",
    "rtn_to_eci_matrix",
    "rotate_covariance",
    "EncounterFrame",
    "build_encounter_frame",
    "enforce_tca",
]


# ---------------------------------------------------------------------------
# RTN (local orbital) frame
# ---------------------------------------------------------------------------


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two length-3 vectors, without numpy.cross's dispatch."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def rtn_basis(position_km: np.ndarray, velocity_km_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the RTN basis vectors for one object from its inertial state.

    Parameters
    ----------
    position_km, velocity_km_s
        Inertial position and velocity, shape ``(3,)``.

    Returns
    -------
    (r_hat, t_hat, n_hat)
        Radial (outward along position), transverse (in-plane, completing the
        triad), and normal (along orbital angular momentum) unit vectors.

    Notes
    -----
    The transverse axis is deliberately built as ``n_hat x r_hat`` and NOT as
    ``velocity / |velocity|``. Those two coincide only for a perfectly
    circular orbit. Using the velocity direction on an eccentric orbit
    introduces a real rotation error that no symmetry or positive-definiteness
    check will catch downstream.
    """
    position_km = np.asarray(position_km, dtype=float)
    velocity_km_s = np.asarray(velocity_km_s, dtype=float)

    r_norm = np.linalg.norm(position_km)
    if r_norm == 0.0:
        raise ValueError("cannot build an RTN frame at the origin")

    r_hat = position_km / r_norm

    # The cross products are written out componentwise rather than via
    # numpy.cross. For length-3 vectors the arithmetic is identical, but
    # numpy.cross dispatches through moveaxis and normalize_axis_tuple, and
    # this function is the hottest in the whole screening path -- profiling a
    # 26-object screen showed 174 057 calls with 6.8 s of the 18 s total spent
    # inside that dispatch machinery rather than on arithmetic.
    angular_momentum = _cross3(position_km, velocity_km_s)
    h_norm = np.linalg.norm(angular_momentum)
    if h_norm == 0.0:
        raise ValueError("cannot build an RTN frame for a radial (zero angular momentum) trajectory")

    n_hat = angular_momentum / h_norm
    t_hat = _cross3(n_hat, r_hat)

    return r_hat, t_hat, n_hat


def rtn_to_eci_matrix(position_km: np.ndarray, velocity_km_s: np.ndarray) -> np.ndarray:
    """Rotation matrix taking RTN-frame vectors into the inertial frame.

    The basis vectors are the COLUMNS of the returned matrix, so that
    ``v_eci = A @ v_rtn``. Since ``A`` is orthonormal, ``A.T`` is its inverse
    and takes inertial vectors back into RTN.
    """
    r_hat, t_hat, n_hat = rtn_basis(position_km, velocity_km_s)
    return np.column_stack((r_hat, t_hat, n_hat))


def rotate_covariance(covariance: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply the similarity transform ``R C R^T`` to a covariance matrix.

    Parameters
    ----------
    covariance
        Square covariance matrix, ``(n, n)``.
    rotation
        Rotation matrix mapping the covariance's current frame into the
        target frame, i.e. ``v_target = rotation @ v_current``.

    Notes
    -----
    Transposing ``rotation`` by mistake rotates the wrong way and produces a
    result that is still symmetric and still positive definite -- it will not
    trip any sanity check. Verify direction with a deliberately anisotropic
    test case rather than trusting the shape of the output.
    """
    covariance = np.asarray(covariance, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    return rotation @ covariance @ rotation.T


# ---------------------------------------------------------------------------
# Encounter frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncounterFrame:
    """The conjunction plane in which 2D collision probability is evaluated.

    Attributes
    ----------
    x_hat, y_hat, z_hat
        Orthonormal basis. ``y_hat`` is along relative velocity and is the
        axis marginalised away; ``x_hat`` and ``z_hat`` span the encounter
        plane.
    rotation
        3x3 matrix with the basis vectors as ROWS, mapping inertial vectors
        into the encounter frame.
    miss_distance_km
        Magnitude of the relative position vector at closest approach.
    relative_speed_km_s
        Magnitude of the relative velocity vector.
    relative_position_km, relative_velocity_km_s
        The relative state used to build the frame, primary minus secondary.
    """

    x_hat: np.ndarray
    y_hat: np.ndarray
    z_hat: np.ndarray
    rotation: np.ndarray
    miss_distance_km: float
    relative_speed_km_s: float
    relative_position_km: np.ndarray
    relative_velocity_km_s: np.ndarray

    @property
    def miss_vector_2d_km(self) -> np.ndarray:
        """Miss vector projected into the encounter plane, ``(x, z)``.

        At true closest approach this is exactly ``(miss_distance, 0)``. See
        the identity proved in :func:`build_encounter_frame`; we return the
        analytic value rather than a numerical projection because it is both
        exact and cheaper.
        """
        return np.array([self.miss_distance_km, 0.0])

    def project_covariance(self, covariance_eci: np.ndarray) -> np.ndarray:
        """Project a 3x3 inertial position covariance into the encounter plane.

        Rotating into the encounter frame and then deleting the row and column
        corresponding to relative velocity is a MARGINALISATION, not a slice.
        That is precisely what makes it legitimate: integrating the Gaussian
        along the relative-velocity axis over the whole real line yields unity
        under the rectilinear relative motion assumption.
        """
        covariance_encounter = rotate_covariance(covariance_eci, self.rotation)
        keep = np.array([0, 2])
        return covariance_encounter[np.ix_(keep, keep)]


def enforce_tca(
    position_primary_km: np.ndarray,
    velocity_primary_km_s: np.ndarray,
    position_secondary_km: np.ndarray,
    velocity_secondary_km_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Linearly advance both states to the instant of closest approach.

    The encounter frame construction is only valid where relative position is
    perpendicular to relative velocity. If the caller's states are even
    slightly off closest approach the frame is subtly rotated -- a third of a
    millisecond of timing error is enough to rotate the x-axis by more than
    1e-2 radians.

    Returns
    -------
    (r1, v1, r2, v2, dt)
        Both states advanced by ``dt`` seconds under constant velocity, plus
        the applied offset. Velocities are unchanged by the linear step.
    """
    r1 = np.asarray(position_primary_km, dtype=float)
    v1 = np.asarray(velocity_primary_km_s, dtype=float)
    r2 = np.asarray(position_secondary_km, dtype=float)
    v2 = np.asarray(velocity_secondary_km_s, dtype=float)

    relative_position = r1 - r2
    relative_velocity = v1 - v2

    speed_squared = float(relative_velocity @ relative_velocity)
    if speed_squared == 0.0:
        return r1, v1, r2, v2, 0.0

    dt = -float(relative_position @ relative_velocity) / speed_squared

    return r1 + v1 * dt, v1, r2 + v2 * dt, v2, dt


def build_encounter_frame(
    position_primary_km: np.ndarray,
    velocity_primary_km_s: np.ndarray,
    position_secondary_km: np.ndarray,
    velocity_secondary_km_s: np.ndarray,
    *,
    already_at_tca: bool = False,
) -> EncounterFrame:
    """Construct the encounter plane from two states at closest approach.

    Parameters
    ----------
    position_primary_km, velocity_primary_km_s
        Primary object's inertial state.
    position_secondary_km, velocity_secondary_km_s
        Secondary object's inertial state.
    already_at_tca
        Skip the closest-approach correction. Only pass ``True`` when the
        caller has already converged the states; the correction is cheap and
        the default is to apply it.

    Notes
    -----
    Basis construction::

        r = r1 - r2                 relative position
        v = v1 - v2                 relative velocity
        h = r x v                   normal to the relative motion plane

        y_hat = v / |v|             the collapsed axis
        z_hat = h / |h|             out of the relative motion plane
        x_hat = y_hat x z_hat       completes the right-handed triad

    A useful identity holds at closest approach: ``x_hat == r / |r|``. Proof:
    ``x_hat = y_hat x z_hat = (v x (r x v)) / (|v| |h|)``, and by the vector
    triple product this is ``(r (v.v) - v (v.r)) / (|v| |h|)``. At closest
    approach ``v.r = 0``, so the second term vanishes and ``|h| = |r| |v|``,
    giving ``x_hat = r |v|^2 / (|v| |r| |v|) = r / |r|``.

    The practical consequence is that the miss vector in the encounter plane
    is exactly ``(|r|, 0)`` -- no numerical projection required. This identity
    is asserted in the test suite.

    This is the *conjunction plane*, not the planetary-flyby B-plane. The two
    share the "perpendicular to relative velocity" idea but differ in how the
    in-plane axes are fixed, so do not import B-plane axis conventions here.
    """
    if not already_at_tca:
        (
            position_primary_km,
            velocity_primary_km_s,
            position_secondary_km,
            velocity_secondary_km_s,
            _,
        ) = enforce_tca(
            position_primary_km,
            velocity_primary_km_s,
            position_secondary_km,
            velocity_secondary_km_s,
        )

    relative_position = np.asarray(position_primary_km, dtype=float) - np.asarray(
        position_secondary_km, dtype=float
    )
    relative_velocity = np.asarray(velocity_primary_km_s, dtype=float) - np.asarray(
        velocity_secondary_km_s, dtype=float
    )

    miss_distance = float(np.linalg.norm(relative_position))
    relative_speed = float(np.linalg.norm(relative_velocity))

    if relative_speed == 0.0:
        raise ValueError(
            "relative velocity is zero; the 2D encounter model does not apply. "
            "Escalate this conjunction to a 3D or Monte Carlo treatment."
        )

    y_hat = relative_velocity / relative_speed

    angular_momentum = np.cross(relative_position, relative_velocity)
    h_norm = float(np.linalg.norm(angular_momentum))

    if h_norm < 1e-12:
        # Degenerate: the objects are effectively co-located, or relative
        # position is parallel to relative velocity. Physically Pc is at its
        # maximum here, so precision matters far less than not crashing.
        # Perturb perpendicular to the relative velocity to define a plane.
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(float(y_hat @ arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(y_hat, arbitrary)
        z_hat = perpendicular / np.linalg.norm(perpendicular)
    else:
        z_hat = angular_momentum / h_norm

    x_hat = np.cross(y_hat, z_hat)

    rotation = np.vstack((x_hat, y_hat, z_hat))

    return EncounterFrame(
        x_hat=x_hat,
        y_hat=y_hat,
        z_hat=z_hat,
        rotation=rotation,
        miss_distance_km=miss_distance,
        relative_speed_km_s=relative_speed,
        relative_position_km=relative_position,
        relative_velocity_km_s=relative_velocity,
    )
