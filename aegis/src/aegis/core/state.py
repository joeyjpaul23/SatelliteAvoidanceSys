"""State vectors and covariance.

These are the value types that flow between every layer of the system. They
carry data and light validation only -- no propagation, no risk maths, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import frames
from .timebase import ensure_utc

__all__ = ["StateVector", "Covariance", "CovarianceSource"]


@dataclass(frozen=True)
class StateVector:
    """An object's inertial position and velocity at a single epoch.

    Attributes
    ----------
    epoch
        Timezone-aware UTC datetime.
    position_km, velocity_km_s
        Inertial (TEME) position and velocity, shape ``(3,)``.
    frame
        Frame label carried for provenance. SGP4 output is TEME-of-date; the
        label exists so that a future numerically-propagated or operator-
        supplied state can be distinguished rather than silently mixed.
    """

    epoch: datetime
    position_km: np.ndarray
    velocity_km_s: np.ndarray
    frame: str = "TEME"

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", ensure_utc(self.epoch))
        position = np.asarray(self.position_km, dtype=float).reshape(3)
        velocity = np.asarray(self.velocity_km_s, dtype=float).reshape(3)
        object.__setattr__(self, "position_km", position)
        object.__setattr__(self, "velocity_km_s", velocity)

    @property
    def radius_km(self) -> float:
        """Geocentric distance."""
        return float(np.linalg.norm(self.position_km))

    @property
    def speed_km_s(self) -> float:
        """Inertial speed."""
        return float(np.linalg.norm(self.velocity_km_s))

    @property
    def altitude_km(self) -> float:
        """Altitude above a spherical Earth.

        Spherical rather than ellipsoidal on purpose: this value is used for
        binning and display, never for geodetic work, and the ~21 km
        equator-to-pole difference is irrelevant at that resolution.
        """
        from ..constants import R_EARTH_KM

        return self.radius_km - R_EARTH_KM

    def rtn_to_eci(self) -> np.ndarray:
        """Rotation matrix from this object's RTN frame into the inertial frame."""
        return frames.rtn_to_eci_matrix(self.position_km, self.velocity_km_s)


class CovarianceSource:
    """How a covariance came to exist. Provenance drives how far to trust it."""

    #: Derived from an actual orbit determination process.
    CALCULATED = "CALCULATED"
    #: A default assigned by object class because none was supplied.
    DEFAULT = "DEFAULT"
    #: Synthesised from an empirical TLE error model. Not a real covariance.
    SYNTHETIC_TLE = "SYNTHETIC_TLE"


@dataclass(frozen=True)
class Covariance:
    """Position (and optionally velocity) uncertainty for one object.

    Stored as a 3x3 or 6x6 matrix in the RTN frame, in km^2 (and km^2/s,
    km^2/s^2 for the velocity blocks). RTN is the storage frame because that
    is what CCSDS mandates for CDMs and what error models are naturally
    expressed in; rotation to inertial happens at the point of use.

    Attributes
    ----------
    matrix
        Symmetric covariance, ``(3, 3)`` or ``(6, 6)``, RTN frame, km-based.
    source
        One of :class:`CovarianceSource`.
    scale_factor
        Multiplier already applied to the raw covariance. Operationally
        derived covariances are known to be optimistic and inflation is
        standard practice; recording the factor keeps that visible rather
        than hidden inside the numbers.
    """

    matrix: np.ndarray
    source: str = CovarianceSource.DEFAULT
    scale_factor: float = 1.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape not in {(3, 3), (6, 6)}:
            raise ValueError(f"covariance must be 3x3 or 6x6, got {matrix.shape}")
        if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-12):
            raise ValueError("covariance matrix must be symmetric")
        object.__setattr__(self, "matrix", matrix)

    @property
    def position_rtn(self) -> np.ndarray:
        """The 3x3 position block in RTN, km^2."""
        return self.matrix[:3, :3]

    @property
    def sigma_rtn_km(self) -> np.ndarray:
        """One-sigma position uncertainty per RTN axis, km."""
        return np.sqrt(np.diag(self.position_rtn))

    def position_eci(self, state: StateVector) -> np.ndarray:
        """Rotate the position covariance into the inertial frame.

        The rotation must be built from THIS object's own state -- each
        object's covariance lives in its own RTN frame, so combining two
        objects' covariances requires two different rotations.
        """
        rotation = state.rtn_to_eci()
        return frames.rotate_covariance(self.position_rtn, rotation)

    def scaled(self, factor: float) -> "Covariance":
        """Return an inflated (or deflated) copy.

        Note the direction of travel: if an event sits in the probability
        dilution regime, inflating covariance *lowers* Pc. Inflation is
        therefore not automatically conservative, and the dilution flag
        reported alongside every risk assessment exists to make that visible.
        """
        return Covariance(
            matrix=self.matrix * factor,
            source=self.source,
            scale_factor=self.scale_factor * factor,
        )

    @classmethod
    def from_sigmas_rtn(
        cls,
        sigma_radial_km: float,
        sigma_transverse_km: float,
        sigma_normal_km: float,
        *,
        source: str = CovarianceSource.DEFAULT,
    ) -> "Covariance":
        """Build a diagonal RTN covariance from per-axis one-sigma values."""
        matrix = np.diag(
            [sigma_radial_km**2, sigma_transverse_km**2, sigma_normal_km**2]
        )
        return cls(matrix=matrix, source=source)


@dataclass
class Ephemeris:
    """A time-ordered sequence of states for one object.

    Kept deliberately thin -- a container with interpolation, not a
    propagator. Populated by :mod:`aegis.propagation` and consumed by
    :mod:`aegis.screening`.
    """

    object_id: str
    epochs: np.ndarray  # seconds relative to `reference_epoch`
    positions_km: np.ndarray  # (n, 3)
    velocities_km_s: np.ndarray  # (n, 3)
    reference_epoch: datetime
    frame: str = "TEME"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.epochs = np.asarray(self.epochs, dtype=float)
        self.positions_km = np.asarray(self.positions_km, dtype=float).reshape(-1, 3)
        self.velocities_km_s = np.asarray(self.velocities_km_s, dtype=float).reshape(-1, 3)
        self.reference_epoch = ensure_utc(self.reference_epoch)

        if not (len(self.epochs) == len(self.positions_km) == len(self.velocities_km_s)):
            raise ValueError("ephemeris epochs, positions and velocities must be the same length")

    def __len__(self) -> int:
        return len(self.epochs)

    def interpolate(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Cubic Hermite interpolation of position and velocity.

        Hermite rather than a spline because this container already holds
        velocities, which are exactly the derivative information Hermite
        needs -- so interpolation is free of extra propagation and has
        O(h^4) error. A spline fitted to positions alone would discard that
        information and behave poorly near a fast conjunction, where the
        distance function has a near-kink.
        """
        if seconds < self.epochs[0] or seconds > self.epochs[-1]:
            raise ValueError(
                f"requested time {seconds} s lies outside the ephemeris span "
                f"[{self.epochs[0]}, {self.epochs[-1]}] s"
            )

        index = int(np.searchsorted(self.epochs, seconds) - 1)
        index = max(0, min(index, len(self.epochs) - 2))

        t0, t1 = self.epochs[index], self.epochs[index + 1]
        step = t1 - t0
        if step <= 0:
            return self.positions_km[index], self.velocities_km_s[index]

        s = (seconds - t0) / step
        s2, s3 = s * s, s * s * s

        # Hermite basis functions.
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2

        p0, p1 = self.positions_km[index], self.positions_km[index + 1]
        v0, v1 = self.velocities_km_s[index], self.velocities_km_s[index + 1]

        position = h00 * p0 + step * h10 * v0 + h01 * p1 + step * h11 * v1

        # Derivative of the same basis, divided by the step to return km/s.
        dh00 = 6 * s2 - 6 * s
        dh10 = 3 * s2 - 4 * s + 1
        dh01 = -6 * s2 + 6 * s
        dh11 = 3 * s2 - 2 * s

        velocity = (dh00 * p0 + step * dh10 * v0 + dh01 * p1 + step * dh11 * v1) / step

        return position, velocity
