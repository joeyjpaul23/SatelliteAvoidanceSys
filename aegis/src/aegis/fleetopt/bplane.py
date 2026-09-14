"""B-plane miss-distance sensitivity: how a fleet of burns moves a conjunction.

Why this module exists
----------------------
:func:`aegis.maneuver.planner.plan_maneuvers` models the post-maneuver miss
distance as

.. code-block:: text

    miss_after = miss_before + (y_rtn / miss) * delta_y_along_track

That is, it projects the along-track displacement onto the *y* component of
the relative position unit vector. Two things are wrong with it, and both
matter:

1. **It keeps the component of the displacement that lies along the
   relative velocity.** At closest approach the relative trajectory is
   locally a straight line, so sliding the primary along the relative
   velocity direction does not change the miss distance at all -- it only
   moves the time of closest approach. For a crossing LEO encounter the
   along-track displacement of one satellite can be largely parallel to the
   relative velocity, so the scalar model credits the burn with separation
   it does not actually buy.
2. **It approximates the rotation between the two satellites' RTN frames
   with a single dot product** (``_along_track_alignment``), which is only
   valid when the two orbit planes nearly coincide.

The fix is to work in the encounter B-plane, which the codebase already
understands: :class:`aegis.core.frames.EncounterFrame` builds exactly this
basis for the probability calculation. Let

.. math::

    P_j = I_3 - \\hat{w}_j \\hat{w}_j^{\\top}

be the projector onto the plane normal to the relative velocity
:math:`w_j`. Then:

**Proposition 1 (first-order miss distance).** Let the fleet displacement
change the relative position at the nominal TCA by :math:`B_j x`. The
post-maneuver minimum separation near that epoch is

.. math::

    m_j(x) = \\lVert P_j ( d_j + B_j x ) \\rVert + O(\\lVert \\cdot \\rVert^2).

*Proof.* Near TCA the relative motion is
:math:`d(t) = d_j + B_j x + w_j (t - t_j) + O((t-t_j)^2)`. Minimising the
norm over :math:`t` gives the residual
:math:`\\lVert P_j (d_j + B_j x) \\rVert`; the neglected relative
acceleration acts over a time shift that is itself first order in the
displacement, so its contribution is second order. :math:`\\square`

**Proposition 2 (conservative affine restriction).** For any unit vector
:math:`u`,

.. math::

    u^{\\top} P_j ( d_j + B_j x ) \\ge \\rho_j
    \\Longrightarrow
    \\lVert P_j ( d_j + B_j x ) \\rVert \\ge \\rho_j

by Cauchy-Schwarz. The linear constraint is therefore an **inner**
approximation -- a restriction -- of the true reverse-convex keep-out
constraint. Every point the linear program accepts is safe under the
linearised dynamics, for *any* choice of :math:`u`. Nothing downstream,
including a neural network that proposes :math:`u`, can make the plan
unsafe; it can only make it more expensive.

**Proposition 3 (monotone sequential refinement).** If :math:`x^\\star`
solves the program linearised at directions :math:`u`, and we update
:math:`u_j' = P_j(d_j + B_j x^\\star) / \\lVert P_j(d_j + B_j x^\\star)\\rVert`,
then :math:`x^\\star` remains feasible for the program linearised at
:math:`u'`, because
:math:`u_j'^{\\top} P_j (d_j + B_j x^\\star)
= \\lVert P_j(d_j+B_j x^\\star)\\rVert \\ge \\rho_j`.
Hence the optimal cost is non-increasing across refinements and, being
bounded below by zero, converges. :math:`\\square`

Those three propositions are the whole optimisation story: an exact linear
displacement model, a provably conservative convexification, and a
monotone refinement that tightens it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..core.conjunction import Conjunction, RiskAssessment
from .dynamics import BurnGrid, eci_displacement_operator
from .errors import FleetOptError

__all__ = [
    "bplane_projector",
    "MissSensitivity",
    "relative_sensitivity",
    "build_miss_sensitivity",
    "linearize",
    "refine_direction",
    "realized_miss_km",
]

_ZERO = 1e-12


def bplane_projector(relative_velocity_km_s: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the plane normal to the relative velocity.

    A zero relative velocity has no well-defined encounter plane, so the
    identity is returned: every displacement direction then counts, which is
    the conservative choice. Low relative velocity is exactly the
    intra-constellation regime the codebase already flags through
    :data:`aegis.constants.LOW_RELATIVE_VELOCITY_KM_S`, and those events are
    marked ``short_encounter_valid=False`` upstream.
    """
    w = np.asarray(relative_velocity_km_s, dtype=float).reshape(3)
    speed = float(np.linalg.norm(w))
    if speed < _ZERO:
        return np.eye(3, dtype=float)
    w_hat = w / speed
    return np.eye(3, dtype=float) - np.outer(w_hat, w_hat)


def relative_sensitivity(
    grid: BurnGrid,
    primary_id: str,
    secondary_id: str,
    epoch: datetime,
    primary_state,
    secondary_state,
) -> np.ndarray:
    """``B = Psi_secondary(t) - Psi_primary(t)`` in ECI, shape ``(3, n_vars)``.

    The sign convention matches
    :attr:`aegis.core.conjunction.Conjunction.relative_position_rtn_km`,
    which is secondary minus primary. A non-maneuverable or ungridded object
    contributes a zero block.
    """
    secondary = eci_displacement_operator(grid, secondary_id, epoch, secondary_state)
    primary = eci_displacement_operator(grid, primary_id, epoch, primary_state)
    return secondary - primary


@dataclass(frozen=True)
class MissSensitivity:
    """Linear model of one conjunction's miss distance as a function of ``x``.

    ``miss_vector_km`` is the projected nominal miss vector ``P d``, whose
    norm is the geometric miss distance. ``sensitivity`` is ``P B``, so the
    post-maneuver miss distance is ``||miss_vector_km + sensitivity @ x||``
    to first order.
    """

    conjunction_id: str
    primary_id: str
    secondary_id: str
    tca: datetime
    miss_vector_km: np.ndarray
    sensitivity: np.ndarray
    required_miss_km: float
    nominal_miss_km: float
    relative_speed_km_s: float
    probability_before: float = 0.0
    hard_body_radius_km: float = 0.0
    sigma_major_km: float = 0.0
    sigma_minor_km: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "miss_vector_km", np.asarray(self.miss_vector_km, dtype=float).reshape(3)
        )
        sensitivity = np.asarray(self.sensitivity, dtype=float)
        if sensitivity.ndim != 2 or sensitivity.shape[0] != 3:
            raise FleetOptError("sensitivity must have shape (3, n_vars)")
        object.__setattr__(self, "sensitivity", sensitivity)

    @property
    def n_vars(self) -> int:
        return int(self.sensitivity.shape[1])

    @property
    def shortfall_km(self) -> float:
        """How far the unmaneuvered geometry falls short of the requirement."""
        return max(0.0, self.required_miss_km - self.nominal_miss_km)

    @property
    def is_controllable(self) -> bool:
        """Whether any burn in the grid can move this conjunction at all."""
        return bool(np.any(np.abs(self.sensitivity) > _ZERO))

    @property
    def nominal_direction(self) -> np.ndarray:
        """Unit vector along the nominal projected miss -- the default ``u``."""
        norm = float(np.linalg.norm(self.miss_vector_km))
        if norm < _ZERO:
            return np.array([1.0, 0.0, 0.0])
        return self.miss_vector_km / norm

    def miss_vector_after(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.n_vars)
        return self.miss_vector_km + self.sensitivity @ x

    def miss_after_km(self, x: np.ndarray) -> float:
        """First-order post-maneuver miss distance (Proposition 1)."""
        return float(np.linalg.norm(self.miss_vector_after(x)))

    def residual_km(self, x: np.ndarray) -> float:
        """``miss_after - required``; non-negative means safe."""
        return self.miss_after_km(x) - self.required_miss_km

    def max_response_km(self, bound: float) -> float:
        """Largest miss change any ``x`` with ``||x||_inf <= bound`` can produce.

        Useful for deciding, before solving, whether a conjunction is
        reachable at all. Equals ``bound * sum(|P B|)`` row-wise worst case,
        which is the dual norm of the sensitivity.
        """
        return float(bound * np.sum(np.abs(self.sensitivity)))


def build_miss_sensitivity(
    conjunction: Conjunction,
    assessment: RiskAssessment,
    grid: BurnGrid,
    required_miss_km: float,
) -> MissSensitivity:
    """Assemble the B-plane sensitivity for one assessed conjunction."""
    projector = bplane_projector(
        np.asarray(conjunction.secondary_state.velocity_km_s, dtype=float)
        - np.asarray(conjunction.primary_state.velocity_km_s, dtype=float)
    )
    relative_position = np.asarray(
        conjunction.secondary_state.position_km, dtype=float
    ) - np.asarray(conjunction.primary_state.position_km, dtype=float)

    sensitivity = relative_sensitivity(
        grid,
        conjunction.primary.object_id,
        conjunction.secondary.object_id,
        conjunction.tca,
        conjunction.primary_state,
        conjunction.secondary_state,
    )

    miss_vector = projector @ relative_position
    return MissSensitivity(
        conjunction_id=conjunction.conjunction_id,
        primary_id=conjunction.primary.object_id,
        secondary_id=conjunction.secondary.object_id,
        tca=conjunction.tca,
        miss_vector_km=miss_vector,
        sensitivity=projector @ sensitivity,
        required_miss_km=float(required_miss_km),
        nominal_miss_km=float(np.linalg.norm(miss_vector)),
        relative_speed_km_s=float(conjunction.relative_speed_km_s),
        probability_before=float(assessment.probability),
        hard_body_radius_km=float(assessment.hard_body_radius_m) / 1000.0,
        sigma_major_km=float(assessment.sigma_major_km),
        sigma_minor_km=float(assessment.sigma_minor_km),
    )


def linearize(
    sensitivity: MissSensitivity,
    direction: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Affine restriction of the keep-out constraint (Proposition 2).

    Returns ``(row, rhs)`` such that ``row @ x >= rhs`` implies
    ``||P (d + B x)|| >= required_miss_km``.
    """
    u = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(u))
    if norm < _ZERO:
        raise FleetOptError("linearization direction must be non-zero")
    u = u / norm
    row = u @ sensitivity.sensitivity
    rhs = sensitivity.required_miss_km - float(u @ sensitivity.miss_vector_km)
    return row, rhs


def refine_direction(sensitivity: MissSensitivity, x: np.ndarray) -> np.ndarray:
    """Tightest linearization direction at ``x`` (Proposition 3).

    Returns the nominal direction when the perturbed miss vector collapses
    to zero, which is the only case where no tightest direction exists.
    """
    vector = sensitivity.miss_vector_after(x)
    norm = float(np.linalg.norm(vector))
    if norm < _ZERO:
        return sensitivity.nominal_direction
    return vector / norm


def realized_miss_km(sensitivities: list[MissSensitivity], x: np.ndarray) -> np.ndarray:
    """First-order post-maneuver miss distance for every conjunction."""
    if not sensitivities:
        return np.zeros(0, dtype=float)
    return np.array([s.miss_after_km(x) for s in sensitivities], dtype=float)
