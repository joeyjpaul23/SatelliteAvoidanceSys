"""Projecting a conjunction into the 2D encounter plane.

Every 2D collision probability method -- Alfano, Foster, Chan, Patera --
integrates the *same* bivariate Gaussian over the *same* disc. They differ
only in quadrature scheme. That means the projection performed here is shared
by all of them, and getting it right matters more than the choice of method.

The reduction, in order:

1. Advance both states to true closest approach (the frame is only valid
   where relative position is perpendicular to relative velocity).
2. Rotate each object's RTN covariance into the inertial frame, using that
   object's *own* state.
3. Sum them. Position errors are treated as independent between objects.
4. Rotate the combined covariance into the encounter frame and marginalise
   away the relative-velocity axis.
5. Diagonalise the resulting 2x2 so the Gaussian is axis-aligned.
6. Repair the covariance if it is not positive definite.

What comes out is a canonical problem: a zero-mean, axis-aligned 2D Gaussian
with standard deviations ``sigma_major >= sigma_minor``, and a disc of radius
equal to the combined hard-body radius centred at the projected miss vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import COV_CLIP_FRACTIONS
from ..core.frames import EncounterFrame, build_encounter_frame
from ..core.state import Covariance, StateVector

__all__ = ["ProjectedEncounter", "project_encounter", "remediate_covariance_2x2"]


@dataclass(frozen=True)
class ProjectedEncounter:
    """The canonical 2D problem, ready for quadrature.

    Attributes
    ----------
    sigma_major, sigma_minor
        Principal one-sigma axes of the projected covariance, major first.
        Same length unit as ``miss_x`` / ``miss_z`` / ``hard_body_radius``.
    miss_x, miss_z
        Miss vector components along the principal axes.
    hard_body_radius
        Combined circumscribing radius of the two objects.
    frame
        The encounter frame this projection was built from.
    remediated
        Whether eigenvalue clipping was applied to repair a non-positive-
        definite covariance.
    """

    sigma_major: float
    sigma_minor: float
    miss_x: float
    miss_z: float
    hard_body_radius: float
    frame: EncounterFrame
    remediated: bool = False

    @property
    def miss_distance(self) -> float:
        """Total miss distance, recoverable from the principal components."""
        return float(np.hypot(self.miss_x, self.miss_z))

    @property
    def mahalanobis_distance(self) -> float:
        """Miss distance scaled by the covariance.

        The scale-free measure of how unusual this approach is. Unlike Pc it
        is monotonic in covariance size, which makes it a more stable
        decision aid when covariance realism is in doubt.
        """
        return float(
            np.sqrt(
                (self.miss_x / self.sigma_major) ** 2
                + (self.miss_z / self.sigma_minor) ** 2
            )
        )

    @property
    def aspect_ratio(self) -> float:
        """Ratio of principal axes. Operational conjunctions run 10-100."""
        return self.sigma_major / self.sigma_minor

    @property
    def scaled_radius(self) -> float:
        """``R / sqrt(sigma_major * sigma_minor)``.

        Governs whether the analytic Chan cross-check is trustworthy: its
        error grows roughly as the square of this quantity, so the cross-check
        is only meaningful below about 0.1.
        """
        return self.hard_body_radius / np.sqrt(self.sigma_major * self.sigma_minor)


def remediate_covariance_2x2(
    covariance: np.ndarray, hard_body_radius: float
) -> tuple[np.ndarray, bool]:
    """Repair a non-positive-definite 2x2 covariance by clipping eigenvalues.

    Real CDM covariances are frequently not positive definite, usually from
    accumulated numerical error in the orbit determination that produced them.
    Integrating against one produces a meaningless number, so it must be
    repaired or refused -- never silently used.

    The clipping floor is tied to the hard-body radius rather than an absolute
    epsilon, which keeps the repair scale-free: what counts as negligible
    uncertainty depends on how big the objects are. Progressively larger
    floors are tried until the matrix becomes definite.

    Returns
    -------
    (repaired, was_remediated)

    Raises
    ------
    ValueError
        If no clipping fraction restores positive definiteness. This is a
        genuine data quality failure and the caller must not fabricate a
        probability from it.
    """
    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    if np.all(eigenvalues > 0):
        return covariance, False

    for fraction in COV_CLIP_FRACTIONS:
        floor = (fraction * hard_body_radius) ** 2
        clipped = np.maximum(eigenvalues, floor)
        repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
        if np.all(np.linalg.eigvalsh(repaired) > 0):
            return repaired, True

    raise ValueError(
        "projected covariance is not positive definite and could not be "
        "remediated at any clipping fraction; the input covariance is "
        "unusable and no probability should be reported for this event"
    )


def project_encounter(
    primary_state: StateVector,
    secondary_state: StateVector,
    primary_covariance: Covariance,
    secondary_covariance: Covariance,
    hard_body_radius_km: float,
    *,
    already_at_tca: bool = False,
) -> ProjectedEncounter:
    """Reduce a conjunction to the canonical 2D problem.

    Parameters
    ----------
    primary_state, secondary_state
        Inertial states at (or very near) closest approach.
    primary_covariance, secondary_covariance
        Each object's position covariance, in its own RTN frame.
    hard_body_radius_km
        Combined circumscribing radius of the pair, in kilometres.
    already_at_tca
        Skip the closest-approach correction.

    Notes
    -----
    Summing the two covariances assumes their position errors are
    statistically independent. That assumption is violated when both objects
    share an atmospheric density forecast error -- two LEO objects during the
    same geomagnetic storm, for instance. There is no fix for this inside the
    2D method; it inflates true risk relative to computed Pc, and is noted
    here so the limitation stays visible.
    """
    frame = build_encounter_frame(
        primary_state.position_km,
        primary_state.velocity_km_s,
        secondary_state.position_km,
        secondary_state.velocity_km_s,
        already_at_tca=already_at_tca,
    )

    # Each object's covariance rotates with its OWN state -- two different
    # rotations, because each RTN frame is built from a different orbit.
    combined_eci = primary_covariance.position_eci(
        primary_state
    ) + secondary_covariance.position_eci(secondary_state)

    projected = frame.project_covariance(combined_eci)

    projected, remediated = remediate_covariance_2x2(projected, hard_body_radius_km)

    sigma_major, sigma_minor, principal_axis = _diagonalise_2x2(projected)

    # The miss vector in the encounter plane is exactly (miss_distance, 0) by
    # the identity proved in core.frames. Its components along the principal
    # axes are therefore just its projections onto them. Absolute values are
    # correct here because both the disc and the Gaussian ellipse are
    # symmetric under reflection about either principal axis.
    miss = frame.miss_distance_km
    miss_x = miss * abs(principal_axis[0])
    miss_z = miss * abs(principal_axis[1])

    return ProjectedEncounter(
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        miss_x=miss_x,
        miss_z=miss_z,
        hard_body_radius=hard_body_radius_km,
        frame=frame,
        remediated=remediated,
    )


def _diagonalise_2x2(covariance: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Closed-form eigendecomposition of a symmetric 2x2 covariance.

    Written out rather than calling a general eigensolver: for a 2x2 the
    closed form is faster, exactly reproducible, and lets us clamp the
    discriminant against round-off producing a spurious negative.

    Returns
    -------
    (sigma_major, sigma_minor, principal_axis)
        Standard deviations along the major and minor axes, and the unit
        eigenvector of the major axis.
    """
    sigma_xx = float(covariance[0, 0])
    sigma_zz = float(covariance[1, 1])
    sigma_xz = float(covariance[0, 1])

    trace = sigma_xx + sigma_zz
    determinant = sigma_xx * sigma_zz - sigma_xz * sigma_xz

    discriminant = np.sqrt(max(trace * trace / 4.0 - determinant, 0.0))

    lambda_major = trace / 2.0 + discriminant
    lambda_minor = trace / 2.0 - discriminant

    # Guard against a zero or negative minor eigenvalue surviving remediation.
    lambda_minor = max(lambda_minor, np.finfo(float).tiny)

    if sigma_xz != 0.0:
        axis = np.array([lambda_major - sigma_zz, sigma_xz])
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 0 else np.array([1.0, 0.0])
    else:
        axis = np.array([1.0, 0.0]) if sigma_xx >= sigma_zz else np.array([0.0, 1.0])

    return float(np.sqrt(lambda_major)), float(np.sqrt(lambda_minor)), axis
