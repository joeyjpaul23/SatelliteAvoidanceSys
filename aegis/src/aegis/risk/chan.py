"""Chan's analytic collision probability -- an independent cross-check.

Chan replaces the circular hard-body disc with an equal-area disc in
coordinates scaled by the two principal sigmas, which makes the integral
separable and yields a closed-form series::

    u = R^2 / (sx * sz)                     scaled area
    v = xm^2/sx^2 + zm^2/sz^2               squared Mahalanobis miss distance

           inf                             m
    Pc =   sum  [ e^(-v/2) (v/2)^m / m! ] [ 1 - e^(-u/2) sum (u/2)^k / k! ]
          m=0                                            k=0

That series is exactly the cumulative distribution function of a noncentral
chi-square with two degrees of freedom, ``F(u; k=2, lambda=v)``, so in
practice we evaluate it that way. It is better conditioned than summing by
hand and removes any truncation decision.

Why cross-check at all
----------------------
The primary method (Alfano by Gauss-Chebyshev) and this one share the same
projection but nothing else -- different mathematics, different failure
modes. Agreement between them is meaningful evidence that a result is sound;
disagreement is worth a human look.

Validity envelope -- important
------------------------------
Chan is *exact* for isotropic covariance (``sx == sz``) at any radius, which
makes it a precise regression test for that case. Its error otherwise grows
roughly as the square of ``R / sqrt(sx*sz)`` times a function of the aspect
ratio. Real conjunctions are strongly anisotropic (along-track dominant,
ratios of 10-100), so this method is only trustworthy when the scaled radius
is small -- comfortably true for typical LEO events, where the hard-body
radius is metres and the sigmas hundreds of metres. Above that threshold the
cross-check is skipped rather than reported as a disagreement.
"""

from __future__ import annotations

import math

from scipy.stats import ncx2

__all__ = ["collision_probability", "is_applicable", "METHOD_NAME", "MAX_SCALED_RADIUS"]

METHOD_NAME = "CHAN-2008-NCX2"

#: Above this scaled radius the equal-area approximation degrades enough that
#: a disagreement with the primary method says more about Chan than about the
#: primary result.
MAX_SCALED_RADIUS = 0.1


def collision_probability(
    sigma_major: float,
    sigma_minor: float,
    miss_x: float,
    miss_z: float,
    hard_body_radius: float,
) -> float:
    """Chan's analytic collision probability.

    All length arguments must share a unit. See :mod:`aegis.risk.alfano` for
    the argument meanings -- they are identical.
    """
    if hard_body_radius <= 0.0:
        return 0.0
    if sigma_major <= 0.0 or sigma_minor <= 0.0:
        raise ValueError("covariance standard deviations must be positive")

    scaled_area = (hard_body_radius**2) / (sigma_major * sigma_minor)
    mahalanobis_squared = (miss_x / sigma_major) ** 2 + (miss_z / sigma_minor) ** 2

    probability = float(ncx2.cdf(scaled_area, 2, mahalanobis_squared))
    return float(min(max(probability, 0.0), 1.0))


def is_applicable(
    sigma_major: float, sigma_minor: float, hard_body_radius: float
) -> bool:
    """Whether Chan's approximation is accurate enough to serve as a check."""
    scaled_radius = hard_body_radius / math.sqrt(sigma_major * sigma_minor)
    return scaled_radius <= MAX_SCALED_RADIUS


def collision_probability_equal_area_square(
    sigma_major: float,
    sigma_minor: float,
    miss_x: float,
    miss_z: float,
    hard_body_radius: float,
    *,
    circumscribing: bool = False,
) -> float:
    """Closed-form equal-area-square approximation. Fast pre-filter only.

    Replaces the disc with a square of the same area, half-width
    ``H = sqrt(pi)/2 * R``, making the integral fully separable into a product
    of two one-dimensional error function differences. With ``circumscribing``
    set, ``H = R`` instead, which circumscribes the disc and therefore yields
    a strict upper bound -- useful for cheaply discarding events that cannot
    breach a threshold.

    Never use this as a final answer. It fails catastrophically in the far
    tail, returning exactly zero where the true probability is small but
    nonzero, because both error function terms saturate.
    """
    from scipy.special import erf

    if hard_body_radius <= 0.0:
        return 0.0

    half_width = hard_body_radius if circumscribing else math.sqrt(math.pi) / 2.0 * hard_body_radius
    root_two = math.sqrt(2.0)

    x_term = erf((miss_x + half_width) / (sigma_major * root_two)) - erf(
        (miss_x - half_width) / (sigma_major * root_two)
    )
    z_term = erf((miss_z + half_width) / (sigma_minor * root_two)) - erf(
        (miss_z - half_width) / (sigma_minor * root_two)
    )

    return float(min(max(0.25 * x_term * z_term, 0.0), 1.0))
