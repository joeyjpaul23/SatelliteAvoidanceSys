"""Alfano 2D collision probability.

This is the method Starlink's Space Safety Platform documents as its own:
"the two-dimensional Alfano collision probability method", after Alfano
(2005), *A Numerical Implementation of Spherical Object Collision
Probability*, Journal of the Astronautical Sciences 53(1).

The quantity being computed is the integral of a bivariate Gaussian over a
disc::

                1                    /  /
    Pc = ---------------             |  |  exp[ -1/2 ( x^2/sx^2 + z^2/sz^2 ) ] dz dx
          2 pi sx sz                 /  /
                            (x-xm)^2 + (z-zm)^2 <= R^2

Alfano's contribution is reducing that 2D integral to a 1D integral over the
error function, by integrating the ``z`` direction analytically. For a fixed
``x`` the chord half-length through the disc is ``h = sqrt(R^2 - (x-xm)^2)``,
and the inner integral collapses to a difference of two ``erf`` terms::

              1        / R
    Pc = ------------  |    [ erf( (zm + sqrt(R^2-u^2)) / (sz sqrt2) )
          sx sqrt(8pi) / -R      - erf( (zm - sqrt(R^2-u^2)) / (sz sqrt2) ) ]
                              * exp( -(u + xm)^2 / (2 sx^2) ) du

Quadrature choice
-----------------
The integrand contains ``sqrt(R^2 - u^2)``, whose derivative is infinite at
``u = +/-R``. Polynomial quadrature converges badly against that endpoint
singularity -- Simpson's rule converges as roughly O(N^-1.5), reaching only
two or three significant figures at the step counts Alfano's original adaptive
scheme selects.

Substituting ``u = R t`` makes the ``sqrt(1 - t^2)`` factor exactly the
Gauss-Chebyshev second-kind weight function, which absorbs the singularity
analytically. The result is machine precision at 8 nodes where Simpson needs
thousands. This is also what NASA CARA ships operationally, so the choice is
both more accurate and better precedented.

Simpson's rule is retained in this module as a cross-check path only.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import erfc

from ..constants import ALFANO_QUAD_ORDER

__all__ = [
    "collision_probability",
    "collision_probability_simpson",
    "max_collision_probability",
    "METHOD_NAME",
]

#: Recorded in the CDM ``COLLISION_PROBABILITY_METHOD`` field.
METHOD_NAME = "ALFANO-2005-GAUSSCHEBYSHEV"

_SQRT_8PI = math.sqrt(8.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


@lru_cache(maxsize=8)
def _gauss_chebyshev_nodes(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Chebyshev second-kind nodes and weights on ``[-1, 1]``.

    The ``sqrt(8 pi)`` normalisation from the Alfano prefactor is folded into
    the weights, leaving a clean ``R / sigma_major`` multiplier at the call
    site.

    Returns
    -------
    (t, y, w)
        Abscissae ``t = cos(theta)``, the companion ``y = sin(theta)`` used
        for the chord half-length, and the weights.
    """
    step = math.pi / (order + 1)
    indices = np.arange(order, 0, -1, dtype=float)
    theta = step * indices

    t = np.cos(theta)
    y = np.sin(theta)
    w = step * y / _SQRT_8PI

    return t, y, w


def _erf_difference(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Compute ``erf(upper) - erf(lower)`` without tail cancellation.

    Using the identity ``erf(b) - erf(a) == erfc(a) - erfc(b)``. This matters
    in the far tail: once both ``erf`` terms saturate to 1.0 their difference
    is exactly zero and the answer is destroyed, whereas ``erfc`` retains
    precision down to about 1e-300. Reference vectors in the test suite
    include a case with a true probability near 1e-29 specifically to catch a
    regression here.
    """
    return erfc(lower) - erfc(upper)


def collision_probability(
    sigma_major: float,
    sigma_minor: float,
    miss_x: float,
    miss_z: float,
    hard_body_radius: float,
    *,
    order: int = ALFANO_QUAD_ORDER,
) -> float:
    """Collision probability by Gauss-Chebyshev quadrature of Alfano's integral.

    Parameters
    ----------
    sigma_major, sigma_minor
        Principal one-sigma axes of the projected 2D covariance.
    miss_x, miss_z
        Miss vector components along those principal axes.
    hard_body_radius
        Combined circumscribing radius of the two objects.
    order
        Number of quadrature nodes. The default of 64 is NASA CARA's
        operational value; 16 suffices for essentially all short-duration
        encounters.

    All five length arguments must share a unit. The caller is responsible for
    that consistency -- this function is deliberately unit-agnostic so it can
    be tested directly against published reference vectors.

    Returns
    -------
    float
        Probability in ``[0, 1]``.
    """
    if hard_body_radius <= 0.0:
        return 0.0
    if sigma_major <= 0.0 or sigma_minor <= 0.0:
        raise ValueError("covariance standard deviations must be positive")

    # An infinite hard-body radius means certain collision. Guarding this
    # explicitly avoids the quadrature saturating in a confusing way.
    if math.isinf(hard_body_radius):
        return 1.0

    t, y, w = _gauss_chebyshev_nodes(order)

    # Exponential factor: the Gaussian along the major axis, evaluated at the
    # abscissa offset from the miss vector.
    exponent = -(((miss_x + hard_body_radius * t) / (sigma_major * _SQRT_2)) ** 2)
    gaussian = np.exp(exponent)

    # Chord contribution: the analytically-integrated minor-axis direction.
    chord = hard_body_radius * y
    upper = (miss_z + chord) / (sigma_minor * _SQRT_2)
    lower = (miss_z - chord) / (sigma_minor * _SQRT_2)
    chord_term = _erf_difference(upper, lower)

    total = float(np.sum(w * gaussian * chord_term))
    probability = (hard_body_radius / sigma_major) * total

    # Clamp: quadrature round-off can produce a value a few ulp outside range.
    return float(min(max(probability, 0.0), 1.0))


def collision_probability_simpson(
    sigma_major: float,
    sigma_minor: float,
    miss_x: float,
    miss_z: float,
    hard_body_radius: float,
    *,
    intervals: int = 512,
) -> float:
    """Alfano's integral by composite Simpson's rule.

    Provided as a cross-check against the Gauss-Chebyshev path, and because
    it is the literal form of Alfano's published implementation. It is not
    the production method: convergence against the endpoint singularity is
    roughly O(N^-1.5), so reaching six significant figures needs on the order
    of a million intervals where Gauss-Chebyshev needs eight nodes.

    ``intervals`` must be even.
    """
    if intervals % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals")
    if hard_body_radius <= 0.0:
        return 0.0

    step = 2.0 * hard_body_radius / intervals
    u = -hard_body_radius + step * np.arange(intervals + 1)

    # Clamp against round-off driving the radicand slightly negative at the
    # endpoints, which would produce NaN.
    chord = np.sqrt(np.maximum(hard_body_radius**2 - u**2, 0.0))

    upper = (miss_z + chord) / (sigma_minor * _SQRT_2)
    lower = (miss_z - chord) / (sigma_minor * _SQRT_2)
    integrand = _erf_difference(upper, lower) * np.exp(
        -((u + miss_x) ** 2) / (2.0 * sigma_major**2)
    )

    weights = np.ones(intervals + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0

    total = float(np.sum(weights * integrand)) * step / 3.0
    probability = total / (sigma_major * _SQRT_8PI)

    return float(min(max(probability, 0.0), 1.0))


def max_collision_probability(hard_body_radius: float, miss_distance: float) -> float:
    """The geometric ceiling on collision probability, independent of covariance.

    ``Pc_max = R^2 / (e * d^2)``

    Collision probability is *not* monotonic in covariance size. As
    uncertainty grows, Pc rises to a peak and then falls again -- the
    "dilution" regime, where worse tracking yields a lower and falsely
    reassuring probability. Maximising the small-radius approximation
    ``Pc ~ (R^2 / 2 sigma^2) exp(-d^2 / 2 sigma^2)`` over ``sigma^2`` gives a
    peak at ``sigma = d / sqrt(2)`` and this value at the peak.

    Because it depends only on geometry, it bounds the risk of an encounter
    even when the covariance is untrusted or absent. Reported alongside every
    assessment for that reason.

    Reference: Alfano, *Relating Position Uncertainty to Maximum Conjunction
    Probability*, JAS 53(2), 2005.
    """
    if miss_distance <= 0.0:
        return 1.0
    value = (hard_body_radius**2) / (math.e * miss_distance**2)
    return float(min(value, 1.0))


def dilution_sigma(miss_distance: float) -> float:
    """Covariance scale at which collision probability peaks.

    ``sigma* = d / sqrt(2)``. When the actual projected sigma exceeds this,
    the encounter sits on the falling side of the curve and a small Pc may
    reflect poor knowledge rather than genuine safety.
    """
    return miss_distance / _SQRT_2
