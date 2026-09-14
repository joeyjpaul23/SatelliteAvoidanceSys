"""Apply impulsive maneuvers to SGP4 objects by mean-element re-fitting.

The problem
-----------
SGP4 has no impulsive-burn table, and its element set is not osculating --
the Brouwer/Kozai mean elements and the SGP4 theory are a matched pair, as
:class:`aegis.core.objects.OrbitalElements` says in as many words. So a burn
has to be expressed as a *change of mean elements*, and the question is how.

:func:`aegis.maneuver.rescreen.apply_along_track_burns` answers it with a
constant phase offset: it converts the total Clohessy-Wiltshire along-track
displacement at one reference epoch into ``dM = dy / a`` and leaves mean
motion alone. That is adequate for the step-10 contract it was written for,
but it is *not* dynamically consistent with the optimizer:

* Leaving mean motion unchanged removes the secular drift. The CW response
  contains the term ``-3 dv (t - tau)``, which is precisely a mean-motion
  change; a pure phase offset reproduces the displacement at one epoch and
  nowhere else. Measured against SGP4, a plan evaluated at a later
  conjunction is then wrong by tens of percent.
* The mean-anomaly field of a TLE has 1e-4 degree resolution, which is 12 m
  of along-track position at 550 km. A 2 mm/s burn one and a half orbits
  ahead displaces the satellite by 51 m, so writing the shift back into the
  TLE text quantises away a quarter of it.

Both problems disappear if the burn is applied the way an operator's
ephemeris tool applies it: propagate to the burn epoch, add the impulse to
the inertial velocity, then **re-fit the mean elements** so that SGP4
reproduces the post-burn state. That is an exact, six-dimensional,
any-axis, any-eccentricity construction, and it leaves the optimizer's CW
model as the only approximation in the chain -- which is where we want it,
because then measuring CW linearisation error against SGP4 measures
something real.

The fit is a damped Gauss-Newton iteration on the six mean elements
``(n, e, i, RAAN, argp, M)`` with a finite-difference Jacobian. SGP4 is
smooth in its elements at these scales, the seed is the pre-burn element set
(which is already within a few metres per second of the answer), and
convergence to below a millimetre and a micrometre per second typically
takes three or four iterations.

``aegis.maneuver.apply_along_track_burns`` is left exactly as it is and
remains the mapping the ``legacy-lp`` baseline uses, so the step-10 contract
is untouched and the comparison between the two mappings is itself a
reportable result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares

from ..constants import MU_EARTH_KM3_S2, SECONDS_PER_DAY
from ..core.maneuver import Maneuver, ManeuverPlan
from ..core.objects import OrbitalElements, SpaceObject
from ..core.state import StateVector
from ..core.timebase import ensure_utc
from ..propagation.propagator import PropagationError, Sgp4Propagator
from .errors import FleetOptError

__all__ = [
    "ElementFitResult",
    "elements_from_state",
    "gauss_element_update",
    "fit_mean_elements",
    "apply_impulse",
    "apply_fleet_burns",
    "ApplyReport",
]

#: Convergence targets for the element fit.
_POS_TOL_KM = 1e-7
_VEL_TOL_KM_S = 1e-10
_MAX_ITERATIONS = 40

#: Seed-to-target distance beyond which an analytic re-seed is attempted.
_RESEED_THRESHOLD_KM = 50.0

#: Velocity residual weight, seconds. About an eighth of a LEO orbit, chosen
#: so a km/s of velocity error and a km of position error matter comparably.
_VELOCITY_WEIGHT_S = 1000.0

#: Absolute finite-difference steps per element, in the element's own units.
#: Each moves the satellite by roughly a millimetre at LEO: above the ~0.3 m
#: floor SGP4's Julian-date arithmetic imposes only when accumulated over a
#: propagation, and far above the 1 nm resolution of a double at 7000 km.
_FD_STEPS = np.array(
    [
        1e-9,    # mean motion, rev/day
        1e-10,   # eccentricity
        1e-8,    # inclination, deg
        1e-8,    # RAAN, deg
        1e-8,    # argument of perigee, deg
        1e-8,    # mean anomaly, deg
    ]
)

#: Weighted-residual norm below which the fit is finished.
_TARGET_COST = 1e-9

#: Floor on the Jacobian step scaling. Below this the finite differences
#: approach the ~1 nm resolution of a double at a 7000 km radius.
_MIN_FD_SCALE = 1e-3

#: Impulse magnitude the default Jacobian steps are tuned for, km/s.
_REFERENCE_IMPULSE_KM_S = 1e-4

#: Deterministic multi-start used only when the deterministic solve stalls.
_MULTISTART_SEED = 20260912
_MULTISTART_ATTEMPTS = 24
_MULTISTART_SCALES = np.array([1e-4, 1e-5, 1e-2, 1e-2, 1e-2, 1e-2])

#: How many times the trust-region solve is restarted from its own result.
_REFINEMENT_RESTARTS = 4

#: A sweep that does not improve the cost by at least this fraction is taken
#: as converged, which stops the loop spending evaluations on noise.
_MIN_RELATIVE_GAIN = 1e-3

#: Residual returned when SGP4 refuses a trial element set, large enough that
#: the solver always steps away from it and finite enough not to poison the
#: trust-region scaling with an inf.
_DIVERGED_RESIDUAL = 1e6

# The solve is deliberately UNBOUNDED. Every orbit this project plans for is
# near-circular, so the eccentricity sits on a lower bound of zero, and a
# bounded trust-region step against an active bound collapses: measured
# against SGP4 the bounded solve stalled at kilometres where the unbounded one
# reaches nanometres. Validity is instead enforced where it belongs -- by
# clamping in `_elements_from_vector` and by returning `_DIVERGED_RESIDUAL`
# when SGP4 refuses a trial set.

#: Function-evaluation budget per restart.
_MAX_FEV = 400

_ELEMENT_NAMES = (
    "mean_motion_rev_per_day",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_perigee_deg",
    "mean_anomaly_deg",
)


@dataclass
class ElementFitResult:
    """Outcome of one mean-element re-fit."""

    elements: OrbitalElements
    iterations: int
    position_error_km: float
    velocity_error_km_s: float
    converged: bool

    @property
    def residual_summary(self) -> str:
        return (
            f"pos {self.position_error_km * 1000.0:.4f} m, "
            f"vel {self.velocity_error_km_s * 1e6:.4f} mm/s, "
            f"{self.iterations} iterations"
        )


def elements_from_state(
    epoch: datetime,
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    *,
    bstar: float = 0.0,
) -> OrbitalElements:
    """Osculating Keplerian elements from an inertial state.

    Standard rv-to-classical-elements conversion, with the near-circular
    degeneracy handled: below an eccentricity of 1e-7 the argument of perigee
    is meaningless, so it is pinned to zero and the whole phase is carried in
    the mean anomaly as the argument of latitude. Every constellation orbit
    this project touches is in that regime.

    These are *osculating* elements, not the Brouwer mean elements SGP4 wants.
    For a near-circular LEO orbit the two differ by the J2 short-period terms,
    of order ten kilometres in radius -- close enough to seed
    :func:`fit_mean_elements`, which then closes the remaining gap, and far
    too far apart to use directly. That distinction is the whole reason
    :class:`aegis.core.objects.OrbitalElements` warns that the element set and
    the theory are a matched pair.
    """
    r = np.asarray(position_km, dtype=float).reshape(3)
    v = np.asarray(velocity_km_s, dtype=float).reshape(3)
    radius = float(np.linalg.norm(r))
    speed = float(np.linalg.norm(v))
    if radius <= 0.0 or speed <= 0.0:
        raise FleetOptError("cannot derive elements from a degenerate state")

    angular_momentum = np.cross(r, v)
    h_norm = float(np.linalg.norm(angular_momentum))
    if h_norm <= 0.0:
        raise FleetOptError("cannot derive elements from a rectilinear state")

    node = np.cross(np.array([0.0, 0.0, 1.0]), angular_momentum)
    node_norm = float(np.linalg.norm(node))

    eccentricity_vector = (
        (speed * speed - MU_EARTH_KM3_S2 / radius) * r - float(np.dot(r, v)) * v
    ) / MU_EARTH_KM3_S2
    eccentricity = float(np.linalg.norm(eccentricity_vector))

    energy = 0.5 * speed * speed - MU_EARTH_KM3_S2 / radius
    if energy >= 0.0:
        raise FleetOptError("state is not bound; SGP4 cannot represent it")
    semi_major = -MU_EARTH_KM3_S2 / (2.0 * energy)
    mean_motion_rad_s = float(np.sqrt(MU_EARTH_KM3_S2 / semi_major**3))

    inclination = float(np.degrees(np.arccos(np.clip(angular_momentum[2] / h_norm, -1.0, 1.0))))
    raan = float(np.degrees(np.arctan2(node[1], node[0])) % 360.0) if node_norm > 1e-12 else 0.0

    if eccentricity < 1e-7:
        # Circular: carry the phase as the argument of latitude.
        if node_norm > 1e-12:
            node_hat = node / node_norm
            in_plane = np.cross(angular_momentum / h_norm, node_hat)
            argument_of_latitude = float(
                np.degrees(np.arctan2(float(np.dot(r, in_plane)), float(np.dot(r, node_hat))))
            )
        else:
            argument_of_latitude = float(np.degrees(np.arctan2(r[1], r[0])))
        arg_perigee = 0.0
        mean_anomaly = argument_of_latitude % 360.0
    else:
        e_hat = eccentricity_vector / eccentricity
        if node_norm > 1e-12:
            node_hat = node / node_norm
            arg_perigee = float(
                np.degrees(
                    np.arctan2(
                        float(np.dot(np.cross(node_hat, e_hat), angular_momentum / h_norm)),
                        float(np.dot(node_hat, e_hat)),
                    )
                )
                % 360.0
            )
        else:
            arg_perigee = float(np.degrees(np.arctan2(e_hat[1], e_hat[0])) % 360.0)
        true_anomaly = float(
            np.arctan2(
                float(np.dot(np.cross(e_hat, r / radius), angular_momentum / h_norm)),
                float(np.dot(e_hat, r / radius)),
            )
        )
        eccentric_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly),
            np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly),
        )
        mean_anomaly = float(
            np.degrees(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) % 360.0
        )

    return OrbitalElements(
        epoch=ensure_utc(epoch),
        mean_motion_rev_per_day=mean_motion_rad_s * SECONDS_PER_DAY / (2.0 * np.pi),
        eccentricity=eccentricity,
        inclination_deg=inclination,
        raan_deg=raan,
        arg_perigee_deg=arg_perigee,
        mean_anomaly_deg=mean_anomaly,
        bstar=float(bstar),
    )


def _elements_vector(elements: OrbitalElements) -> np.ndarray:
    return np.array([getattr(elements, name) for name in _ELEMENT_NAMES], dtype=float)


def _elements_from_vector(seed: OrbitalElements, vector: np.ndarray) -> OrbitalElements:
    """Elements from the solver's parameter vector.

    Angles are wrapped here rather than bounded in the solver, because a fit
    whose true answer sits near 0 or 360 degrees would otherwise be pushed
    against a bound it should be free to cross.
    """
    updated = copy.deepcopy(seed)
    updated.mean_motion_rev_per_day = float(vector[0])
    updated.eccentricity = float(min(max(vector[1], 0.0), 0.95))
    updated.inclination_deg = float(vector[2])
    updated.raan_deg = float(vector[3] % 360.0)
    updated.arg_perigee_deg = float(vector[4] % 360.0)
    updated.mean_anomaly_deg = float(vector[5] % 360.0)
    return updated


def _state_from_elements(
    object_id: str, elements: OrbitalElements, epoch: datetime
) -> np.ndarray:
    """SGP4 position and velocity at ``epoch`` as a length-6 vector."""
    probe = SpaceObject(
        object_id=object_id,
        name=object_id,
        elements=elements,
        data_source="FIT",
    )
    propagator = Sgp4Propagator([probe])
    state = propagator.propagate_one(0, epoch)
    return np.concatenate(
        [
            np.asarray(state.position_km, dtype=float),
            np.asarray(state.velocity_km_s, dtype=float),
        ]
    )


def gauss_element_update(
    elements: OrbitalElements,
    epoch: datetime,
    delta_v_rtn_km_s: np.ndarray,
) -> OrbitalElements:
    """First-order Gauss variational update of the mean elements.

    Impulsive form for a near-circular orbit, with the argument of latitude
    ``u`` taken at the burn epoch:

    .. code-block:: text

        da        = 2 dv_T / n
        d(e cos w) = ( dv_R sin u + 2 dv_T cos u) / (n a)
        d(e sin w) = (-dv_R cos u + 2 dv_T sin u) / (n a)
        di        =  dv_N cos u / (n a)
        dRAAN     =  dv_N sin u / (n a sin i)
        d(w + M)  = -2 dv_R / (n a) - dRAAN cos i

    The eccentricity is carried as the vector ``(e cos w, e sin w)`` rather
    than as ``(e, w)`` because at ``e = 0`` the argument of perigee is not
    defined and the scalar form is singular -- which is every orbit this
    project plans for.

    Consistency with the optimizer's model is not a coincidence: ``da``
    changes the mean motion by ``dn = -3 dv_T / a``, whose along-track effect
    grows as ``-3 dv_T (t - tau)``, and that is exactly the secular term of
    the Clohessy-Wiltshire response in
    :func:`aegis.fleetopt.dynamics.cw_impulse_matrix`. The eccentricity change
    supplies the periodic ``4 sin(n sigma) / n`` term. The two descriptions are
    the same physics written in different coordinates.

    Accurate to first order in ``dv / v``, which for avoidance magnitudes is
    about 1e-5. The remaining error against SGP4 comes from J2 short-period
    terms responding to the element change, not from the impulse model, and is
    removed by :func:`fit_mean_elements`.
    """
    impulse = np.asarray(delta_v_rtn_km_s, dtype=float).reshape(3)
    epoch = ensure_utc(epoch)

    n_rad_s = elements.mean_motion_rad_s
    if n_rad_s <= 0.0:
        raise FleetOptError("cannot apply an impulse to an object with no mean motion")
    semi_major = elements.semi_major_axis_km
    elapsed_s = (epoch - elements.epoch).total_seconds()

    mean_anomaly = np.radians(elements.mean_anomaly_deg) + n_rad_s * elapsed_s
    arg_perigee = np.radians(elements.arg_perigee_deg)
    inclination = np.radians(elements.inclination_deg)
    latitude = arg_perigee + mean_anomaly
    speed = n_rad_s * semi_major

    radial, transverse, normal = (float(v) for v in impulse)
    sin_u, cos_u = np.sin(latitude), np.cos(latitude)

    delta_a = 2.0 * transverse / n_rad_s
    delta_q1 = (radial * sin_u + 2.0 * transverse * cos_u) / speed
    delta_q2 = (-radial * cos_u + 2.0 * transverse * sin_u) / speed
    delta_i = normal * cos_u / speed
    sin_i = float(np.sin(inclination))
    delta_raan = normal * sin_u / (speed * sin_i) if abs(sin_i) > 1e-9 else 0.0
    delta_lambda = -2.0 * radial / speed - delta_raan * float(np.cos(inclination))

    q1 = elements.eccentricity * np.cos(arg_perigee) + delta_q1
    q2 = elements.eccentricity * np.sin(arg_perigee) + delta_q2
    eccentricity = float(np.hypot(q1, q2))
    new_arg_perigee = float(np.arctan2(q2, q1)) if eccentricity > 1e-12 else arg_perigee

    new_semi_major = semi_major + delta_a
    if new_semi_major <= 0.0:
        raise FleetOptError("impulse drives the semi-major axis non-positive")
    new_n = float(np.sqrt(MU_EARTH_KM3_S2 / new_semi_major**3))

    new_latitude = latitude + delta_lambda
    mean_anomaly_at_epoch = (new_latitude - new_arg_perigee) - new_n * elapsed_s

    return OrbitalElements(
        epoch=elements.epoch,
        mean_motion_rev_per_day=new_n * SECONDS_PER_DAY / (2.0 * np.pi),
        eccentricity=eccentricity,
        inclination_deg=float(np.degrees(inclination + delta_i)) % 360.0,
        raan_deg=(elements.raan_deg + float(np.degrees(delta_raan))) % 360.0,
        arg_perigee_deg=float(np.degrees(new_arg_perigee)) % 360.0,
        mean_anomaly_deg=float(np.degrees(mean_anomaly_at_epoch)) % 360.0,
        bstar=elements.bstar,
        mean_motion_dot=elements.mean_motion_dot,
        mean_motion_ddot=elements.mean_motion_ddot,
        element_set_number=elements.element_set_number,
        revolution_number=elements.revolution_number,
    )


def fit_mean_elements(
    object_id: str,
    seed: OrbitalElements,
    epoch: datetime,
    target_position_km: np.ndarray,
    target_velocity_km_s: np.ndarray,
    *,
    reseed: bool = True,
    fd_scale: float = 1.0,
) -> ElementFitResult:
    """Find mean elements whose SGP4 state at ``epoch`` matches the target.

    Trust-region least squares over the six mean elements with an explicit
    finite-difference Jacobian. The Jacobian uses **absolute** element steps
    (:data:`_FD_STEPS`), sized so each moves the satellite by roughly a
    millimetre: large enough to clear the ~0.3 m of quantisation in SGP4's
    Julian-date arithmetic, small enough to resolve the last nanometre. A
    relative step, which is what ``least_squares`` would use by default,
    cannot do both -- 1e-6 of a 50 degree angle is six metres of position, and
    the fit stalls there.

    Two callers with opposite needs share this function, and the seed is what
    separates them. :func:`apply_impulse` seeds it with the analytic Gauss
    update, which is already right to a fraction of a millimetre, so the fit
    only polishes. Scenario construction seeds it from far away, and
    ``reseed=True`` first substitutes the osculating elements of the target
    (:func:`elements_from_state`) when the seed is more than
    :data:`_RESEED_THRESHOLD_KM` off, which brings the start inside the
    convergence basin.

    ``fd_scale`` shrinks the Jacobian steps proportionally. The correction a
    fit is chasing scales with the size of the perturbation that caused it, so
    a step tuned for a 100 mm/s impulse is far too coarse for a 2 mm/s one --
    the eccentricity change a 2 mm/s tangential burn produces is only about
    two and a half default steps wide, and the resulting Jacobian is too
    blunt to converge. :func:`apply_impulse` sets this from the impulse
    magnitude.

    Near-circular orbits make the Jacobian rank deficient -- at ``e = 0`` only
    ``w + M`` is observable, not ``w`` and ``M`` separately. The trust-region
    method absorbs that; a normal-equation Gauss-Newton does not, and stalls.
    """
    epoch = ensure_utc(epoch)
    target = np.concatenate(
        [
            np.asarray(target_position_km, dtype=float).reshape(3),
            np.asarray(target_velocity_km_s, dtype=float).reshape(3),
        ]
    )
    scale = np.array(
        [1.0, 1.0, 1.0, _VELOCITY_WEIGHT_S, _VELOCITY_WEIGHT_S, _VELOCITY_WEIGHT_S]
    )

    try:
        current = _state_from_elements(object_id, seed, epoch)
    except PropagationError as error:
        raise FleetOptError(
            f"seed elements for {object_id} will not propagate: {error}"
        ) from error

    if reseed and float(np.linalg.norm(current[:3] - target[:3])) > _RESEED_THRESHOLD_KM:
        try:
            analytic = elements_from_state(epoch, target[:3], target[3:], bstar=seed.bstar)
            analytic_state = _state_from_elements(object_id, analytic, epoch)
        except (FleetOptError, PropagationError):
            pass
        else:
            if float(np.linalg.norm(analytic_state[:3] - target[:3])) < float(
                np.linalg.norm(current[:3] - target[:3])
            ):
                seed = analytic
                current = analytic_state

    def residual(vector: np.ndarray) -> np.ndarray:
        try:
            state = _state_from_elements(object_id, _elements_from_vector(seed, vector), epoch)
        except PropagationError:
            return np.full(6, _DIVERGED_RESIDUAL)
        return (state - target) * scale

    steps = _FD_STEPS * float(np.clip(fd_scale, _MIN_FD_SCALE, 1.0))

    def jacobian(vector: np.ndarray) -> np.ndarray:
        base = residual(vector)
        columns = np.zeros((6, 6), dtype=float)
        for index in range(6):
            probe = vector.copy()
            probe[index] += steps[index]
            columns[:, index] = (residual(probe) - base) / steps[index]
        return columns

    def solve_from(start: np.ndarray) -> tuple[np.ndarray, float, int]:
        result = least_squares(
            residual,
            start,
            jac=jacobian,
            method="trf",
            x_scale="jac",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=_MAX_FEV,
        )
        candidate = np.asarray(result.x, dtype=float)
        return candidate, float(np.linalg.norm(residual(candidate))), int(result.nfev)

    best_vector = _elements_vector(seed)
    best_cost = float(np.linalg.norm(residual(best_vector)))
    evaluations = 1

    for _ in range(_REFINEMENT_RESTARTS):
        if best_cost <= _TARGET_COST:
            break
        candidate, cost, used = solve_from(best_vector)
        evaluations += used
        if cost >= best_cost * (1.0 - _MIN_RELATIVE_GAIN):
            if cost < best_cost:
                best_cost, best_vector = cost, candidate
            break
        best_cost, best_vector = cost, candidate

    # Multi-start escape. A far seed can land in a local minimum a few
    # kilometres out -- suspiciously close to the J2 short-period radial
    # amplitude, which is presumably what it is. Perturbing and re-solving
    # escapes it. The perturbation RNG is seeded from a constant so the whole
    # function stays deterministic: the same inputs must always give the same
    # elements, or scenario digests stop being reproducible.
    if best_cost > _TARGET_COST:
        rng = np.random.default_rng(_MULTISTART_SEED)
        for _ in range(_MULTISTART_ATTEMPTS):
            if best_cost <= _TARGET_COST:
                break
            perturbed = best_vector + rng.normal(size=6) * _MULTISTART_SCALES
            candidate, cost, used = solve_from(perturbed)
            evaluations += used
            if cost < best_cost:
                best_cost, best_vector = cost, candidate

    final_elements = _elements_from_vector(seed, best_vector)
    try:
        final_state = _state_from_elements(object_id, final_elements, epoch)
    except PropagationError as error:
        raise FleetOptError(
            f"fitted elements for {object_id} will not propagate: {error}"
        ) from error

    position_error = float(np.linalg.norm(final_state[:3] - target[:3]))
    velocity_error = float(np.linalg.norm(final_state[3:] - target[3:]))
    return ElementFitResult(
        elements=final_elements,
        iterations=evaluations,
        position_error_km=position_error,
        velocity_error_km_s=velocity_error,
        converged=position_error < _POS_TOL_KM and velocity_error < _VEL_TOL_KM_S,
    )


def apply_impulse(
    obj: SpaceObject,
    epoch: datetime,
    delta_v_rtn_km_s: np.ndarray,
    *,
    refine: bool = True,
) -> tuple[SpaceObject, ElementFitResult]:
    """Return a copy of ``obj`` with one RTN impulse applied at ``epoch``.

    Two steps, and both are needed:

    1. :func:`gauss_element_update` produces the first-order element change.
       It is analytic, deterministic, and already consistent with the
       optimizer's Clohessy-Wiltshire model by construction.
    2. :func:`fit_mean_elements`, seeded with that result, removes the
       remaining J2 short-period discrepancy by matching the exact post-burn
       state. Seeding matters: fitting from the *unperturbed* elements
       instead leaves the solver to discover the whole perturbation, and
       measured against SGP4 that failed by up to 990 % on small impulses.

    ``refine=False`` skips step 2 and returns the analytic result, which is
    accurate to first order in ``dv/v`` (about 1e-5 for avoidance burns) and
    needs no solver.

    TLE text lines are cleared on the copy: their field widths cannot carry
    the resulting precision -- the mean-anomaly field quantises at 1e-4 deg,
    which is 12 m of along-track position, a quarter of what a 2 mm/s burn
    buys -- so SGP4 initialises from the mean elements instead, losslessly.
    """
    epoch = ensure_utc(epoch)
    impulse = np.asarray(delta_v_rtn_km_s, dtype=float).reshape(3)

    updated = copy.deepcopy(obj)
    if updated.elements is None:
        raise FleetOptError(
            f"{obj.object_id} has no mean elements; cannot apply an impulse by element update"
        )
    if not np.any(np.abs(impulse) > 0.0):
        return updated, ElementFitResult(
            elements=updated.elements,
            iterations=0,
            position_error_km=0.0,
            velocity_error_km_s=0.0,
            converged=True,
        )

    element_object = _as_element_object(updated)
    propagator = Sgp4Propagator([element_object])
    pre_burn: StateVector = propagator.propagate_one(0, epoch)
    rotation = np.asarray(pre_burn.rtn_to_eci(), dtype=float)
    position_before = np.asarray(pre_burn.position_km, dtype=float)
    velocity_after = np.asarray(pre_burn.velocity_km_s, dtype=float) + rotation @ impulse

    analytic = gauss_element_update(element_object.elements, epoch, impulse)
    updated.elements = analytic
    updated.tle_line1 = ""
    updated.tle_line2 = ""

    if not refine:
        analytic_state = _state_from_elements(updated.object_id, analytic, epoch)
        return updated, ElementFitResult(
            elements=analytic,
            iterations=0,
            position_error_km=float(np.linalg.norm(analytic_state[:3] - position_before)),
            velocity_error_km_s=float(np.linalg.norm(analytic_state[3:] - velocity_after)),
            converged=False,
        )

    fit = fit_mean_elements(
        updated.object_id,
        analytic,
        epoch,
        position_before,
        velocity_after,
        reseed=False,
        fd_scale=float(np.linalg.norm(impulse)) / _REFERENCE_IMPULSE_KM_S,
    )
    if fit.converged:
        updated.elements = fit.elements
        return updated, fit

    # The refinement is accepted only when it genuinely converged. A partial
    # fit is worse than no fit: for a 2 mm/s impulse the solver settled a
    # metre from the target at the burn epoch, and because that metre lands
    # mostly in the semi-major axis it grows secularly -- the measured
    # displacement was then wrong by a factor of ten, against 1.6 % for the
    # analytic update it replaced. Below roughly 20 mm/s the correction the
    # fit is chasing is finer than the finite-difference Jacobian can resolve,
    # so the analytic result stands.
    analytic_state = _state_from_elements(updated.object_id, analytic, epoch)
    return updated, ElementFitResult(
        elements=analytic,
        iterations=fit.iterations,
        position_error_km=float(np.linalg.norm(analytic_state[:3] - position_before)),
        velocity_error_km_s=float(np.linalg.norm(analytic_state[3:] - velocity_after)),
        converged=False,
    )


def _as_element_object(obj: SpaceObject) -> SpaceObject:
    """A copy that SGP4 will initialise from mean elements, not TLE text.

    Used so the pre-burn state and the fitted post-burn state come from the
    same initialisation path. Mixing ``twoline2rv`` for the baseline with
    ``sgp4init`` for the maneuvered copy injects a metre-scale offset that
    would otherwise be mistaken for maneuver effect.
    """
    if not (obj.tle_line1 and obj.tle_line2):
        return obj
    stripped = copy.deepcopy(obj)
    stripped.tle_line1 = ""
    stripped.tle_line2 = ""
    return stripped


@dataclass
class ApplyReport:
    """Diagnostics from applying a whole plan."""

    fits: dict[str, list[ElementFitResult]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return all(fit.converged for fits in self.fits.values() for fit in fits)

    @property
    def worst_position_error_km(self) -> float:
        errors = [fit.position_error_km for fits in self.fits.values() for fit in fits]
        return max(errors) if errors else 0.0

    @property
    def worst_velocity_error_km_s(self) -> float:
        errors = [fit.velocity_error_km_s for fits in self.fits.values() for fit in fits]
        return max(errors) if errors else 0.0


def apply_fleet_burns(
    objects: list[SpaceObject],
    plan: ManeuverPlan,
    *,
    canonicalize: bool = True,
) -> tuple[list[SpaceObject], ApplyReport]:
    """Apply every burn in ``plan`` to copies of ``objects``, in time order.

    ``canonicalize=True`` (the default) strips TLE text from **every** copy,
    maneuvered or not, so that the baseline and the maneuvered catalog are
    propagated through the identical SGP4 initialisation path. Without it a
    comparison of the two measures TLE round-tripping as well as the burn.
    """
    report = ApplyReport()
    copies = [copy.deepcopy(obj) for obj in objects]
    if canonicalize:
        copies = [_as_element_object(obj) for obj in copies]
    by_id = {obj.object_id: index for index, obj in enumerate(copies)}

    burns: list[Maneuver] = sorted(plan.all_maneuvers, key=lambda m: m.epoch)
    for maneuver in burns:
        index = by_id.get(maneuver.satellite_id)
        if index is None:
            report.notes.append(
                f"burn for {maneuver.satellite_id} skipped: object not in catalog"
            )
            continue
        if copies[index].elements is None:
            report.notes.append(
                f"burn for {maneuver.satellite_id} skipped: no mean elements available"
            )
            continue
        updated, fit = apply_impulse(
            copies[index], maneuver.epoch, maneuver.delta_v_rtn_km_s
        )
        copies[index] = updated
        report.fits.setdefault(maneuver.satellite_id, []).append(fit)
        if not fit.converged:
            report.notes.append(
                f"element re-fit for {maneuver.satellite_id} at "
                f"{maneuver.epoch.isoformat()} did not fully converge: {fit.residual_summary}"
            )

    return copies, report
