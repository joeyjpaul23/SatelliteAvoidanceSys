"""Risk assessment: turning a geometric conjunction into a decision aid.

This module is the public entry point of the risk layer. It takes a
:class:`~aegis.core.conjunction.Conjunction` plus covariance for both objects
and produces a :class:`~aegis.core.conjunction.RiskAssessment`.

Design stance: a bare collision probability is a poor decision statistic on
its own. It is non-monotonic in covariance size, so a large uncertainty can
produce a small and falsely reassuring number -- the well-documented dilution
effect. This assessor therefore always reports the supporting quantities
(miss distance, Mahalanobis distance, projected sigmas, geometric maximum)
and raises explicit flags when the model's assumptions are strained, so an
operator can see why a probability is what it is rather than taking it on
faith.
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    LOW_RELATIVE_VELOCITY_KM_S,
    MAX_SHORT_ENCOUNTER_DURATION_S,
)
from ..core.conjunction import Conjunction, RiskAssessment
from ..core.state import Covariance
from . import alfano, chan
from .projection import ProjectedEncounter, project_encounter

__all__ = ["assess", "assess_projection", "RiskAssessmentError"]


class RiskAssessmentError(RuntimeError):
    """Raised when a conjunction cannot be assessed at all.

    Distinct from a low-confidence assessment. This means no defensible
    number can be produced -- an unusable covariance, or zero relative
    velocity -- and the caller must not substitute a fabricated value.
    """


def assess(
    conjunction: Conjunction,
    primary_covariance: Covariance,
    secondary_covariance: Covariance,
    *,
    cross_check: bool = True,
) -> RiskAssessment:
    """Assess the collision risk of one conjunction.

    Parameters
    ----------
    conjunction
        The geometric close approach, carrying both states at closest
        approach.
    primary_covariance, secondary_covariance
        Each object's position covariance in its own RTN frame.
    cross_check
        Run the independent analytic method as corroboration where its
        validity envelope permits.

    Raises
    ------
    RiskAssessmentError
        If the encounter cannot be projected or the covariance is unusable.
    """
    # Hard-body radius is stored in metres on the objects but the whole
    # geometry pipeline works in kilometres. Convert once, here.
    hard_body_radius_km = conjunction.combined_hard_body_radius_m / 1000.0

    try:
        projection = project_encounter(
            conjunction.primary_state,
            conjunction.secondary_state,
            primary_covariance,
            secondary_covariance,
            hard_body_radius_km,
        )
    except ValueError as error:
        raise RiskAssessmentError(
            f"cannot project conjunction {conjunction.conjunction_id}: {error}"
        ) from error

    assessment = assess_projection(
        projection,
        conjunction_id=conjunction.conjunction_id,
        cross_check=cross_check,
    )

    _apply_short_encounter_checks(assessment, projection, conjunction)

    return assessment


def assess_projection(
    projection: ProjectedEncounter,
    *,
    conjunction_id: str = "",
    cross_check: bool = True,
) -> RiskAssessment:
    """Assess an already-projected encounter.

    Separated from :func:`assess` so the probability mathematics can be
    tested directly against published reference vectors without constructing
    a full conjunction.
    """
    probability = alfano.collision_probability(
        projection.sigma_major,
        projection.sigma_minor,
        projection.miss_x,
        projection.miss_z,
        projection.hard_body_radius,
    )

    cross_probability: float | None = None
    cross_method: str | None = None
    warnings: list[str] = []

    if cross_check and chan.is_applicable(
        projection.sigma_major, projection.sigma_minor, projection.hard_body_radius
    ):
        cross_probability = chan.collision_probability(
            projection.sigma_major,
            projection.sigma_minor,
            projection.miss_x,
            projection.miss_z,
            projection.hard_body_radius,
        )
        cross_method = chan.METHOD_NAME

        scale = max(probability, cross_probability, 1e-30)
        if abs(probability - cross_probability) / scale > 1e-2:
            warnings.append(
                f"primary and cross-check methods disagree by more than 1% "
                f"({probability:.3e} vs {cross_probability:.3e}); inspect this event"
            )

    max_probability = alfano.max_collision_probability(
        projection.hard_body_radius, projection.miss_distance
    )

    # Dilution: the encounter sits on the falling side of the Pc-versus-sigma
    # curve, where a lower probability reflects worse knowledge rather than
    # greater safety.
    dilution_threshold = alfano.dilution_sigma(projection.miss_distance)
    dilution = projection.sigma_major >= dilution_threshold
    if dilution:
        warnings.append(
            "covariance is large relative to miss distance (dilution regime): "
            "a low probability here may reflect poor tracking rather than safety; "
            "weigh the maximum probability and miss distance alongside it"
        )

    if projection.remediated:
        warnings.append(
            "projected covariance was not positive definite and required "
            "eigenvalue clipping before integration"
        )

    return RiskAssessment(
        conjunction_id=conjunction_id,
        probability=probability,
        method=alfano.METHOD_NAME,
        hard_body_radius_m=projection.hard_body_radius * 1000.0,
        miss_distance_km=projection.miss_distance,
        max_probability=max_probability,
        mahalanobis_distance=projection.mahalanobis_distance,
        sigma_major_km=projection.sigma_major,
        sigma_minor_km=projection.sigma_minor,
        cross_check_probability=cross_probability,
        cross_check_method=cross_method,
        dilution_flag=dilution,
        remediated_flag=projection.remediated,
        warnings=warnings,
    )


def _apply_short_encounter_checks(
    assessment: RiskAssessment,
    projection: ProjectedEncounter,
    conjunction: Conjunction,
) -> None:
    """Flag conjunctions where the 2D short-encounter model does not apply.

    The 2D formulation assumes relative motion is rectilinear at constant
    velocity through the encounter and that the covariance is effectively
    constant over it. Two situations break that:

    Low relative velocity
        Nearly co-orbiting objects. This is the dominant failure mode and it
        is *directly* relevant to intra-constellation screening, where
        satellites in the same shell have very low relative speeds. The
        error is generally an underestimate of risk.

    Long encounter duration
        When the objects remain within the covariance ellipsoid for an
        extended period, the single-snapshot integral misses accumulated
        exposure.

    Neither condition invalidates the number automatically, but both mean it
    should be escalated to a 3D or Monte Carlo treatment before being acted
    on, and the flag makes that visible rather than silent.
    """
    relative_speed = conjunction.relative_speed_km_s

    if relative_speed < LOW_RELATIVE_VELOCITY_KM_S:
        assessment.short_encounter_valid = False
        assessment.warnings.append(
            f"relative speed {relative_speed * 1000:.1f} m/s is below the "
            f"short-encounter validity floor; the 2D model likely "
            f"underestimates risk. Escalate to a 3D or Monte Carlo method."
        )

    # Encounter duration: roughly the time spent traversing the along-track
    # extent of the combined covariance plus the hard-body radius.
    if relative_speed > 0:
        duration = 2.0 * (projection.sigma_major + projection.hard_body_radius) / relative_speed
        if duration > MAX_SHORT_ENCOUNTER_DURATION_S:
            assessment.short_encounter_valid = False
            assessment.warnings.append(
                f"estimated encounter duration {duration:.0f} s exceeds the "
                f"short-encounter limit; consider a 3D treatment"
            )


def screen_by_upper_bound(
    projection: ProjectedEncounter, threshold: float
) -> bool:
    """Cheap conservative test: could this event possibly exceed ``threshold``?

    Uses the circumscribing-square upper bound, which is a strict overestimate
    and costs two error function evaluations rather than a quadrature. Returns
    ``False`` only when the event provably cannot reach the threshold, so it
    is safe to skip the full computation for those.
    """
    bound = chan.collision_probability_equal_area_square(
        projection.sigma_major,
        projection.sigma_minor,
        projection.miss_x,
        projection.miss_z,
        projection.hard_body_radius,
        circumscribing=True,
    )
    return bound >= threshold
