"""Induced-conjunction constraints: the pairs a plan must not bring together.

A resolve constraint says "push these two objects apart." A *latent*
constraint says "and do not push anything else together." Latent constraints
are what make this optimizer different from the multi-encounter CAM
literature, which either handles one spacecraft against passive secondaries
or, as Pavanello et al. (arXiv:2406.03654) put it in as many words, assumes
"there is no reason to suppose that a close approach with a particular
secondary is likely to promote a close approach with some other secondary."

At mega-constellation scale that supposition is measurably false. Chen et al.
(arXiv:2406.06068) report that 81.4 % of Starlink-on-Starlink maneuvers are
cascade-induced and that a single external avoidance event triggered as many
as 41 induced maneuvers. Handling the cascade in an outer re-screening loop --
the universal practice, documented in NASA/SP-20230002470 Rev.1 -- cannot
prevent it, only discover it after the fact.

Two constraint families
-----------------------
Let :math:`d_{ab}(t)` be the nominal separation vector of a latent pair and
:math:`B_{ab}(t)` its sensitivity to the decision vector.

**Interval-minimum constraints** (``kind="bplane_minimum"``, the default).
At every local minimum of the nominal separation that falls inside the
reachability gate, impose the B-plane form used for resolve rows:

.. math::

    \\hat u_{ab}(t^{\\star})^{\\top} P_{ab}\\big(d_{ab}(t^{\\star})
      + B_{ab}(t^{\\star}) x\\big) \\ge s_{\\min} + \\gamma .

The projector is what makes one constraint per approach sufficient: it
removes the component along the relative velocity, which is exactly the
direction a displacement slides the closest-approach *time* rather than its
*distance* (Proposition 1 in :mod:`aegis.fleetopt.bplane`). The inflation
:math:`\\gamma` covers everything the first-order model at a grid sample does
not see, and it is small:

.. math::

    \\gamma = \\tfrac{1}{2} A h^2 + \\Lambda h, \\qquad
    A \\le \\frac{2\\mu G}{r_{\\min}^3}, \\qquad
    \\Lambda = \\sqrt{51}\\,(D_a + D_b) .

:math:`A` bounds the **tidal** relative acceleration of a pair already inside
the gate :math:`G` -- not the full two-body acceleration difference, because
two objects within a hundred kilometres of each other experience nearly the
same gravity, and the residual is the tidal term
:math:`(\\mu/r^3)\\lvert\\delta\\rvert`. :math:`\\Lambda` bounds how fast the
maneuver-induced displacement itself changes, via the Lipschitz constant of
:math:`\\Phi` (:data:`aegis.fleetopt.dynamics.CW_NORM_LIPSCHITZ`).

The numbers matter. At a 30 s step, a 100 km gate and a 1 m/s budget,
:math:`\\gamma \\approx 0.54` km against an exclusion radius of 2 km; at the
5 cm/s budget avoidance actually uses it falls to about 0.13 km. Contrast the
naive Lipschitz bound :math:`\\tfrac{1}{2}v_{\\max}h` with
:data:`aegis.constants.MAX_RELATIVE_SPEED_KM_S`: at the same step that is
**285 km**, which swamps any realistic exclusion radius and makes the
constraint set vacuous. The first draft of this module used exactly that
bound; measuring it is what produced the version above.

**Dense B-plane constraints** (``mode="bplane_dense"``, the default). The
B-plane form at **every** gate-admissible epoch, not only at nominal local
minima. This is the mode to use, and the reason is a measured failure of the
local-minimum version.

For a co-orbital pair -- two fleet-mates on the same shell, a few kilometres
apart along-track -- the nominal separation is *flat*: it read 9.345 km at every
sampled epoch. There are no meaningful local minima to place rows at, so the
enumeration picked four arbitrary epochs. The plan then satisfied the
constraint at all four (4.46 km against a 4.33 km floor, with the first-order
model accurate to **6 metres**) while the true minimum over the window was
**2.57 km, at an epoch where no row existed**. The maneuver itself creates the
time-variation, by introducing secular along-track drift, so the perturbed
minimum is nowhere near the nominal one.

Dense enumeration is affordable because row count is nearly free downstream:
:func:`aegis.fleetopt.solver.lazy_solve` starts from an empty guess and uses
one to eleven rows regardless of how many exist, so the cost of enumerating
more rows is enumeration time alone, not solve time.

**Crude grid constraints** (``kind="grid"``). One unprojected row per sample
inside the gate, with the :math:`\\tfrac{1}{2}v_{\\max}h` inflation. Retained as
a diagnostic only; far too conservative to plan against.

Cost control
------------
The latent set is potentially enormous: every pair with a maneuverable member,
at every grid epoch, inside a gate that can be hundreds of kilometres wide.
Three things keep it finite and small in practice.

1. The reachability gate (:mod:`aegis.fleetopt.reachability`) is linear in the
   delta-v budget. Budgeting a few cm/s -- which is all avoidance needs --
   gives a gate of a few km.
2. Only the *binding* epoch of each pair matters. A pair that dips inside the
   gate over a 40-minute approach contributes dozens of near-identical rows;
   :func:`thin_constraints` keeps the tightest one per pair per separation
   bucket, which is exact when the rows are nested and conservative otherwise
   because every retained row is a row of the full problem.
3. Rows can be held back and added lazily
   (:func:`aegis.fleetopt.solver.lazy_solve`), which is how the learned
   active-set predictor earns its speedup without ever compromising
   feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..constants import (
    MAX_RELATIVE_SPEED_KM_S,
    MU_EARTH_KM3_S2,
    PC_THRESHOLD_WATCH,
    R_EARTH_KM,
)
from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import ensure_utc, seconds_between
from .bplane import bplane_projector
from .dynamics import (
    CW_NORM_LIPSCHITZ,
    BurnGrid,
    cw_impulse_matrix,
    cw_impulse_rate_matrix,
    eci_displacement_operator,
)
from .errors import FleetOptError
from .reachability import ReachabilityModel

__all__ = [
    "pair_required_miss_km",
    "LatentConstraint",
    "LATENT_MODES",
    "pair_id",
    "grid_inflation_km",
    "tidal_inflation_km",
    "LATENT_DEFAULT_MODE",
    "build_latent_constraint",
    "enumerate_latent_constraints",
    "thin_constraints",
    "satellite_displacements",
    "satellite_displacement_rates",
    "perturbed_minimum_epochs",
    "LatentEnumerationReport",
    "VALIDATED_LATENT_STEP_S",
]

LATENT_MODES = ("none", "certified_minima", "bplane_dense", "dense_grid", "both")

#: The mode worth planning against.
#:
#: Nominal local minima, *plus* the lazy epoch loop in
#: :func:`aegis.fleetopt.planners.solve_lazy_two_stage` that adds a row at each
#: pair's true perturbed minimum after solving. Local minima alone are not
#: enough -- see the module docstring -- but enumerating densely to cover the
#: gap costs more in enumeration than it saves, and measured worse: dense
#: enumeration left context build at 133 ms where minima plus the epoch loop
#: runs at 92 ms and reaches zero measured induced conjunctions either way.
LATENT_DEFAULT_MODE = "certified_minima"

_ZERO = 1e-12

#: Enumeration step at which the induced-conjunction result has actually been
#: measured, in seconds.
#:
#: This is a **correctness** parameter, not a performance knob, and it is the
#: only one in this module that is. Coarsening it does not merely place fewer
#: rows; it changes which close approaches are *visible at all*, because
#: admission and row placement both start from sampled separations and a fast
#: pair can pass through its whole encounter between two samples.
#:
#: Measured over 3066 planner checks (five non-vacuous families, sixteen seeds,
#: four delta-v budgets, three planners):
#:
#: =========  ==========================  ===========================
#: step       measured induced conj.      of which certified "safe"
#: =========  ==========================  ===========================
#: 30 s       **0 / 900**                 --
#: 60 s       4 / 900                     2
#: 120 s      24 / 900                    17
#: =========  ==========================  ===========================
#:
#: A further 366 checks varying burn slots, refinement iterations, control
#: axes, the cost model, the probability threshold and the risk level -- all at
#: 30 s -- produced no violation either. So the guarantee is measured at 30 s
#: and demonstrably degrades above it; anything coarser is reported as
#: unvalidated rather than silently accepted.
VALIDATED_LATENT_STEP_S = 30.0


def pair_id(object_a: str, object_b: str) -> str:
    """Order-independent pair key, matching ``aegis.maneuver.rescreen``."""
    return f"{min(object_a, object_b)}:{max(object_a, object_b)}"


def grid_inflation_km(step_s: float, max_relative_speed_km_s: float = MAX_RELATIVE_SPEED_KM_S) -> float:
    """``v_max * h / 2`` -- the separation a pair can lose between samples.

    Enforcing ``separation >= s_min + this`` at every sample implies
    ``separation >= s_min`` continuously. Halving the step halves the tax,
    which is the knob to turn when the inflation starts dominating
    ``s_min``.
    """
    if step_s < 0.0:
        raise FleetOptError("grid step must be non-negative")
    return 0.5 * float(max_relative_speed_km_s) * float(step_s)


def tidal_inflation_km(
    step_s: float,
    gate_km: float,
    *,
    radius_km: float = R_EARTH_KM + 300.0,
    displacement_rate_km_s: float = 0.0,
) -> float:
    """Second-order slack for a B-plane constraint placed at a grid sample.

    Two terms, both derived in the module docstring:

    * ``0.5 * A * h**2`` with ``A <= 2 * mu * gate / radius**3`` -- the tidal
      relative acceleration of a pair already inside the gate. Using the tidal
      term rather than the full two-body acceleration difference is the whole
      point: objects a hundred kilometres apart feel almost the same gravity,
      and it is only the residual that bends their relative trajectory.
    * ``Lambda * h`` with ``Lambda = sqrt(51) * (D_a + D_b)`` -- how fast the
      maneuver-induced displacement itself can change within one step.

    ``radius_km`` defaults to a 300 km altitude, the lowest orbit worth
    screening, which maximises the tidal term and so keeps the bound valid for
    anything higher.
    """
    if step_s < 0.0 or gate_km < 0.0:
        raise FleetOptError("step and gate must be non-negative")
    if radius_km <= 0.0:
        raise FleetOptError("radius must be positive")
    acceleration = 2.0 * MU_EARTH_KM3_S2 * float(gate_km) / float(radius_km) ** 3
    return 0.5 * acceleration * float(step_s) ** 2 + float(displacement_rate_km_s) * float(step_s)


def pair_required_miss_km(
    object_a: SpaceObject,
    object_b: SpaceObject,
    state_a: StateVector,
    state_b: StateVector,
    *,
    covariance_model,
    epoch: datetime,
    target_pc: float = PC_THRESHOLD_WATCH,
) -> float:
    """Separation at which *this* pair would reach ``target_pc``.

    A geometric exclusion radius and a probability threshold are different
    objects, and under TLE-grade covariance they are far apart. Using one
    global radius for every latent pair -- even one derived from the median
    assessed covariance -- therefore mis-prices most of them: a pair of fleet
    satellites at similar altitudes has a very different projected covariance
    from a fleet-versus-debris pair, and it is the *latent* pairs, not the
    assessed ones, whose floors matter here.

    This projects both objects' own covariances into the pair's own encounter
    B-plane and inverts Alfano for the separation giving ``target_pc``, which
    is the floor that makes "do not create a conjunction" mean what it says.

    Falls back to 0.0 when the geometry cannot be projected -- the caller then
    uses the configured floor, and records that it did.
    """
    from ..maneuver.miss import required_miss_distance_km
    from ..risk.projection import project_encounter

    hbr_km = (object_a.hard_body_radius_m + object_b.hard_body_radius_m) / 1000.0
    days_a = seconds_between(object_a.elements.epoch, epoch) / 86400.0 if object_a.elements else 0.0
    days_b = seconds_between(object_b.elements.epoch, epoch) / 86400.0 if object_b.elements else 0.0
    try:
        projection = project_encounter(
            state_a,
            state_b,
            covariance_model.covariance(object_a, days_a),
            covariance_model.covariance(object_b, days_b),
            hbr_km,
        )
    except Exception:  # noqa: BLE001 - an unprojectable pair falls back to the configured floor
        return 0.0
    return required_miss_distance_km(
        hbr_km, projection.sigma_major, projection.sigma_minor, target_pc
    )


@dataclass(frozen=True)
class LatentConstraint:
    """One "do not create this conjunction" row.

    The constraint is ``direction @ (offset_km + sensitivity @ x) >= floor_km``,
    which by Cauchy-Schwarz implies ``||offset + sensitivity @ x|| >= floor``.
    """

    pair_id: str
    object_a: str
    object_b: str
    epoch: datetime
    separation_km: float
    direction: np.ndarray
    offset_km: np.ndarray
    sensitivity: np.ndarray
    floor_km: float
    kind: str
    relative_speed_km_s: float = 0.0
    gate_km: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", np.asarray(self.direction, dtype=float).reshape(3))
        object.__setattr__(self, "offset_km", np.asarray(self.offset_km, dtype=float).reshape(3))
        sensitivity = np.asarray(self.sensitivity, dtype=float)
        if sensitivity.ndim != 2 or sensitivity.shape[0] != 3:
            raise FleetOptError("latent sensitivity must have shape (3, n_vars)")
        object.__setattr__(self, "sensitivity", sensitivity)
        if self.kind not in ("grid", "bplane_minimum"):  # noqa: PLR6201 - explicit is clearer
            raise FleetOptError(f"unknown latent constraint kind {self.kind!r}")

    @property
    def label(self) -> str:
        return f"latent:{self.pair_id}@{self.epoch.isoformat()}:{self.kind}"

    @property
    def n_vars(self) -> int:
        return int(self.sensitivity.shape[1])

    @property
    def is_controllable(self) -> bool:
        return bool(np.any(np.abs(self.sensitivity) > _ZERO))

    @property
    def slack_km(self) -> float:
        """How much room the nominal geometry already has."""
        return self.separation_km - self.floor_km

    def row(self) -> tuple[np.ndarray, float]:
        """``(row, rhs)`` with ``row @ x >= rhs``."""
        row = self.direction @ self.sensitivity
        rhs = self.floor_km - float(self.direction @ self.offset_km)
        return row, rhs

    def separation_after_km(self, x: np.ndarray) -> float:
        """True first-order separation, used for verification not assembly."""
        x = np.asarray(x, dtype=float).reshape(self.n_vars)
        return float(np.linalg.norm(self.offset_km + self.sensitivity @ x))

    def residual_km(self, x: np.ndarray) -> float:
        """True separation margin: ``||offset + B x|| - floor``.

        This is the physical question -- is the pair actually far enough
        apart -- and it is what the certificate and the induced-conjunction
        report use.
        """
        return self.separation_after_km(x) - self.floor_km

    def linear_residual_km(self, x: np.ndarray) -> float:
        """Margin on the row the linear program actually enforces.

        ``u . (offset + B x) - floor``. By Cauchy-Schwarz this is never larger
        than :meth:`residual_km`, so a point can satisfy the true separation
        while violating the row -- and anything deciding whether the *program*
        would have been changed by including this row must ask this question,
        not the physical one. The lazy-constraint loop in
        :mod:`aegis.fleetopt.solver` gets this wrong if it uses the norm: it
        terminates believing every omitted row is satisfied, while the full
        program would have forced a different solution.
        """
        row, rhs = self.row()
        return float(row @ np.asarray(x, dtype=float).reshape(self.n_vars)) - rhs


@dataclass
class LatentEnumerationReport:
    """What the enumeration looked at and what it kept."""

    mode: str
    pairs_considered: int = 0
    pairs_inside_gate: int = 0
    epochs_scanned: int = 0
    rows_grid: int = 0
    rows_bplane: int = 0
    rows_after_thinning: int = 0
    rows_uncontrollable: int = 0
    rows_probability_floored: int = 0
    max_gate_km: float = 0.0
    inflation_km: float = 0.0
    certified_inflation_km: float = 0.0
    linearization_margin_km: float = 0.0
    excluded_resolve_pairs: int = 0
    admitted_pairs: list[tuple[str, str]] = None  # type: ignore[assignment]
    """Every pair the widened gate admits, whether or not it got a row.

    Candidate membership and row placement are different questions. A pair can
    be admissible -- reachable, and possibly able to come inside the gate --
    while every *sampled* epoch of it sits outside, so no row is placed. The
    perturbed-minimum search can only refine pairs it is given, so deriving the
    candidate list from the rows that happened to be placed is what let a
    coarse grid drop a pair entirely. See ``worst_sweep_km``."""

    grid_coarser_than_validated: bool = False
    """Set when the enumeration step exceeds :data:`VALIDATED_LATENT_STEP_S`.

    Not an error -- a coarse grid is a legitimate thing to ask for -- but the
    zero-induced-conjunction result does not hold above the validated step and
    the flag says so rather than letting the run look like the validated one."""

    worst_sweep_km: float = 0.0
    """Largest ``0.5 * v_rel * h`` over admitted pairs.

    How far a pair can move between two samples, and therefore how much the
    gate must be widened for sampled separations to decide admission soundly.
    It is large -- hundreds of kilometres for a crossing at a 30 s step -- and
    that is exactly why this term belongs on the *gate* and not on the
    constraint floor, where it would be vacuous."""

    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []
        if self.admitted_pairs is None:
            self.admitted_pairs = []

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "pairs_considered": self.pairs_considered,
            "pairs_inside_gate": self.pairs_inside_gate,
            "worst_sweep_km": self.worst_sweep_km,
            "grid_coarser_than_validated": self.grid_coarser_than_validated,
            "epochs_scanned": self.epochs_scanned,
            "rows_grid": self.rows_grid,
            "rows_bplane": self.rows_bplane,
            "rows_after_thinning": self.rows_after_thinning,
            "rows_uncontrollable": self.rows_uncontrollable,
            "rows_probability_floored": self.rows_probability_floored,
            "max_gate_km": round(self.max_gate_km, 4),
            "inflation_km": round(self.inflation_km, 6),
            "certified_inflation_km": round(self.certified_inflation_km, 6),
            "linearization_margin_km": round(self.linearization_margin_km, 6),
            "excluded_resolve_pairs": self.excluded_resolve_pairs,
            "notes": list(self.notes),
        }


def build_latent_constraint(
    grid: BurnGrid,
    object_a: str,
    object_b: str,
    epoch: datetime,
    state_a: StateVector,
    state_b: StateVector,
    *,
    floor_km: float,
    kind: str,
    gate_km: float = 0.0,
) -> LatentConstraint:
    """Assemble one latent row from a pair of propagated states."""
    epoch = ensure_utc(epoch)
    position_a = np.asarray(state_a.position_km, dtype=float)
    position_b = np.asarray(state_b.position_km, dtype=float)
    velocity_a = np.asarray(state_a.velocity_km_s, dtype=float)
    velocity_b = np.asarray(state_b.velocity_km_s, dtype=float)

    separation_vector = position_b - position_a
    relative_velocity = velocity_b - velocity_a
    sensitivity = eci_displacement_operator(grid, object_b, epoch, state_b) - (
        eci_displacement_operator(grid, object_a, epoch, state_a)
    )

    if kind == "bplane_minimum":
        projector = bplane_projector(relative_velocity)
        offset = projector @ separation_vector
        sensitivity = projector @ sensitivity
    else:
        offset = separation_vector

    norm = float(np.linalg.norm(offset))
    direction = offset / norm if norm > _ZERO else np.array([1.0, 0.0, 0.0])

    return LatentConstraint(
        pair_id=pair_id(object_a, object_b),
        object_a=object_a,
        object_b=object_b,
        epoch=epoch,
        separation_km=norm,
        direction=direction,
        offset_km=offset,
        sensitivity=sensitivity,
        floor_km=float(floor_km),
        kind=kind,
        relative_speed_km_s=float(np.linalg.norm(relative_velocity)),
        gate_km=float(gate_km),
    )


def _local_minimum_indices(separations: np.ndarray) -> list[int]:
    """Interior strict local minima, plus the endpoints when they are minima."""
    count = separations.size
    if count == 0:
        return []
    if count == 1:
        return [0]
    minima: list[int] = []
    if separations[0] <= separations[1]:
        minima.append(0)
    for index in range(1, count - 1):
        if separations[index] <= separations[index - 1] and separations[index] <= separations[index + 1]:
            minima.append(index)
    if separations[-1] <= separations[-2]:
        minima.append(count - 1)
    return minima


def enumerate_latent_constraints(
    grid: BurnGrid,
    propagation_grid,
    reachability: ReachabilityModel,
    *,
    exclusion_radius_km: float,
    mode: str = LATENT_DEFAULT_MODE,
    resolve_epochs: dict[str, list[datetime]] | None = None,
    resolve_guard_s: float | None = None,
    max_relative_speed_km_s: float = MAX_RELATIVE_SPEED_KM_S,
    linearization_margin_km: float = 0.0,
    maneuverable_ids: frozenset[str] | None = None,
    max_rows: int | None = None,
    covariance_model=None,
    objects_by_id: dict[str, SpaceObject] | None = None,
    target_pc: float = PC_THRESHOLD_WATCH,
) -> tuple[list[LatentConstraint], LatentEnumerationReport]:
    """Every latent row the reachability gate admits.

    ``propagation_grid`` is a
    :class:`aegis.propagation.propagator.PropagationGrid` of the *nominal*
    (unmaneuvered) catalog over the planning window. Only pairs with at least
    one maneuverable member are considered: a pair of third-party objects
    cannot be moved by any plan, so no row of theirs can ever bind.

    A pair that already carries a resolve row is **not** excluded outright.
    Both constraints are lower bounds on the same separation, so they cannot
    contradict each other; only the epochs are different. What is skipped is a
    latent row sitting within ``resolve_guard_s`` of one of that pair's own
    resolve TCAs, where the resolve row already governs and usually with a
    tighter floor.

    Excluding the whole pair -- which the first version did -- silently removed
    exactly the rows that matter. An in-plane fleet-mate a dozen kilometres
    along-track is inside the 44 km screening box, so it is screened as a
    (negligible-probability) conjunction, and dropping the pair meant the
    optimizer was free to slide the maneuvering satellite straight into it.
    """
    if mode not in LATENT_MODES:
        raise FleetOptError(f"unknown latent mode {mode!r}; expected one of {LATENT_MODES}")

    report = LatentEnumerationReport(mode=mode)
    if mode == "none":
        return [], report

    object_ids = list(propagation_grid.object_ids)
    times_s = np.asarray(propagation_grid.times_s, dtype=float)
    if times_s.size < 1:
        report.notes.append("propagation grid is empty; no latent rows enumerated")
        return [], report

    step_s = float(np.min(np.diff(times_s))) if times_s.size > 1 else 0.0
    report.epochs_scanned = int(times_s.size)

    resolve_epochs = resolve_epochs or {}
    if resolve_guard_s is None:
        # A quarter orbit at the fastest mean motion in the grid. Wide enough
        # that the resolve row genuinely governs the approach, narrow enough
        # that a second approach by the same pair still gets its own row.
        fastest = max(grid.mean_motion.values(), default=0.0)
        resolve_guard_s = (0.25 * 2.0 * np.pi / fastest) if fastest > 0.0 else 0.0
    guard_s = float(resolve_guard_s)

    if maneuverable_ids is None:
        maneuverable_ids = frozenset(
            sid for sid in grid.satellite_ids if reachability.is_maneuverable(sid)
        )

    positions = np.asarray(propagation_grid.positions_km, dtype=float)
    velocities = np.asarray(propagation_grid.velocities_km_s, dtype=float)
    valid = np.asarray(propagation_grid.valid, dtype=bool)
    index_of = {object_id: index for index, object_id in enumerate(object_ids)}
    window_end = propagation_grid.epoch_at(int(times_s.size) - 1)

    crude_inflation = grid_inflation_km(step_s, max_relative_speed_km_s)
    report.inflation_km = crude_inflation
    if step_s > VALIDATED_LATENT_STEP_S + 1e-9:
        report.grid_coarser_than_validated = True
        report.notes.append(
            f"latent step {step_s:.0f}s is coarser than the {VALIDATED_LATENT_STEP_S:.0f}s "
            "at which the induced-conjunction result was measured; at 60s and 120s "
            "this produced measured induced conjunctions the certificate did not "
            "catch (4/900 and 24/900). Treat any 'no induced conjunction' result at "
            "this step as unvalidated"
        )
    report.linearization_margin_km = float(linearization_margin_km)

    constraints: list[LatentConstraint] = []
    worst_certified_inflation = 0.0

    for a_position, object_a in enumerate(object_ids):
        for b_position in range(a_position + 1, len(object_ids)):
            object_b = object_ids[b_position]
            if object_a not in maneuverable_ids and object_b not in maneuverable_ids:
                continue
            report.pairs_considered += 1
            key = pair_id(object_a, object_b)
            guarded = [ensure_utc(value) for value in resolve_epochs.get(key, ())]

            gate = reachability.max_gate_km(object_a, object_b, window_end)
            report.max_gate_km = max(report.max_gate_km, gate)

            index_a = index_of[object_a]
            index_b = index_of[object_b]
            usable = valid[index_a] & valid[index_b]
            if not np.any(usable):
                continue
            separations = np.linalg.norm(positions[index_b] - positions[index_a], axis=1)
            separations = np.where(usable, separations, np.inf)

            # Admission is decided on sampled separations, so the gate has to
            # absorb how far the pair can move between two samples: a true
            # approach can sit entirely between samples with both neighbours
            # far outside a bare gate. The bound is `0.5 * v_rel * h` -- any
            # instant is within half a step of a sample -- computed from *this
            # pair's* relative speed rather than a catalog-wide maximum, which
            # keeps it tight. It is still large, and that is the point: this
            # term is affordable on the gate, where it only admits more
            # candidates, and vacuous on the floor, where it would swamp any
            # realistic exclusion radius.
            relative_speed = np.linalg.norm(
                velocities[index_b] - velocities[index_a], axis=1
            )
            usable_speed = relative_speed[usable]
            sweep = (
                0.5 * float(np.max(usable_speed)) * step_s if usable_speed.size else 0.0
            )

            inside = np.flatnonzero(separations <= gate + sweep)
            if inside.size == 0:
                continue
            report.pairs_inside_gate += 1
            report.admitted_pairs.append((object_a, object_b))
            report.worst_sweep_km = max(report.worst_sweep_km, sweep)

            displacement_rate = CW_NORM_LIPSCHITZ * (
                reachability.budgets.get(object_a, 0.0)
                + reachability.budgets.get(object_b, 0.0)
            )
            certified_inflation = tidal_inflation_km(
                step_s, gate, displacement_rate_km_s=displacement_rate
            )
            worst_certified_inflation = max(worst_certified_inflation, certified_inflation)

            selected: list[tuple[int, str, float]] = []
            if mode in ("certified_minima", "both"):
                for offset in _local_minimum_indices(separations[inside]):
                    selected.append(
                        (
                            int(inside[offset]),
                            "bplane_minimum",
                            exclusion_radius_km + certified_inflation + linearization_margin_km,
                        )
                    )
            if mode == "bplane_dense":
                for time_index in inside:
                    selected.append(
                        (
                            int(time_index),
                            "bplane_minimum",
                            exclusion_radius_km + certified_inflation + linearization_margin_km,
                        )
                    )
            if mode in ("dense_grid", "both"):
                for time_index in inside:
                    selected.append(
                        (int(time_index), "grid", exclusion_radius_km + crude_inflation)
                    )

            for time_index, kind, floor in selected:
                epoch = propagation_grid.epoch_at(time_index)
                if covariance_model is not None and objects_by_id is not None:
                    obj_a = objects_by_id.get(object_a)
                    obj_b = objects_by_id.get(object_b)
                    if obj_a is not None and obj_b is not None:
                        derived = pair_required_miss_km(
                            obj_a,
                            obj_b,
                            propagation_grid.state(index_a, time_index),
                            propagation_grid.state(index_b, time_index),
                            covariance_model=covariance_model,
                            epoch=epoch,
                            target_pc=target_pc,
                        )
                        if derived > 0.0:
                            # The configured floor stays a floor; the derived
                            # value only ever raises it.
                            floor = max(floor, derived + (floor - exclusion_radius_km))
                            report.rows_probability_floored += 1
                if guarded and any(
                    abs(seconds_between(governed, epoch)) <= guard_s for governed in guarded
                ):
                    report.excluded_resolve_pairs += 1
                    continue
                constraint = build_latent_constraint(
                    grid,
                    object_a,
                    object_b,
                    epoch,
                    propagation_grid.state(index_a, time_index),
                    propagation_grid.state(index_b, time_index),
                    floor_km=floor,
                    kind=kind,
                    gate_km=gate,
                )
                if not constraint.is_controllable:
                    report.rows_uncontrollable += 1
                    continue
                if kind == "grid":
                    report.rows_grid += 1
                else:
                    report.rows_bplane += 1
                constraints.append(constraint)

    report.certified_inflation_km = worst_certified_inflation
    if mode in ("dense_grid", "both") and crude_inflation > 10.0 * max(exclusion_radius_km, 1e-9):
        report.notes.append(
            f"dense-grid inflation is {crude_inflation:.1f} km against an exclusion radius "
            f"of {exclusion_radius_km:.1f} km, so those rows are far too conservative to "
            "plan against; they are diagnostic only"
        )

    constraints.sort(key=lambda item: (item.pair_id, item.epoch, item.kind))
    if max_rows is not None and len(constraints) > max_rows:
        kept = thin_constraints(constraints, max_rows=max_rows)
        report.notes.append(
            f"latent rows thinned from {len(constraints)} to {len(kept)} by binding margin; "
            "every retained row is a row of the full problem, so feasibility remains sound, "
            "but the enumeration is no longer certified complete -- close the gap with "
            "solver.lazy_solve, which re-checks every dropped row"
        )
        constraints = kept
    report.rows_after_thinning = len(constraints)
    return constraints, report


def satellite_displacements(
    grid: BurnGrid,
    propagation_grid,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """ECI displacement of every gridded satellite at every grid epoch.

    Shape ``(n_epochs, 3)`` per satellite. Computed by summing
    ``Phi(n, t - tau_k) dv_k`` and rotating into ECI, which costs a handful of
    3x3 products per epoch rather than forming the full ``(3, n_vars)``
    operator -- the point being that once ``x`` is known, evaluating where the
    *perturbed* trajectory goes is cheap, even though assembling the
    constraint that controls it is not.
    """
    from ..core.frames import rtn_to_eci_matrix

    x = np.asarray(x, dtype=float).reshape(grid.n_vars)
    times = np.asarray(propagation_grid.times_s, dtype=float)
    index_of = {oid: i for i, oid in enumerate(propagation_grid.object_ids)}
    out: dict[str, np.ndarray] = {}

    for sat_id in grid.satellite_ids:
        row = index_of.get(sat_id)
        if row is None:
            continue
        n_rad_s = grid.mean_motion[sat_id]
        impulses = []
        for slot, epoch in enumerate(grid.slot_epochs(sat_id)):
            vector = np.zeros(3)
            for axis in (range(3) if grid.axes == 3 else (1,)):
                vector[axis] = x[grid.index(sat_id, slot, axis)]
            if np.any(vector):
                impulses.append((epoch, vector))
        displacement = np.zeros((times.size, 3), dtype=float)
        if not impulses:
            out[sat_id] = displacement
            continue
        for time_index in range(times.size):
            epoch = propagation_grid.epoch_at(time_index)
            local = np.zeros(3)
            for burn_epoch, vector in impulses:
                lead = seconds_between(burn_epoch, epoch)
                if lead <= 0.0:
                    continue
                local += cw_impulse_matrix(n_rad_s, lead) @ vector
            if np.any(local):
                rotation = rtn_to_eci_matrix(
                    propagation_grid.positions_km[row, time_index],
                    propagation_grid.velocities_km_s[row, time_index],
                )
                displacement[time_index] = rotation @ local
        out[sat_id] = displacement
    return out


def satellite_displacement_rates(
    grid: BurnGrid,
    propagation_grid,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """ECI velocity change of every gridded satellite at every grid epoch.

    The exact time derivative of :func:`satellite_displacements`, and needed
    for the same reason :func:`aegis.fleetopt.dynamics.cw_impulse_rate_matrix`
    is: the range rate is what locates a minimum of the perturbed separation
    when the minimum falls between two samples.

    The rotation into ECI is treated as constant over the step, which is the
    same approximation :func:`satellite_displacements` already makes about the
    RTN frame and is accurate to the frame's own rotation rate -- of order
    ``n`` times the displacement, several orders below the relative velocities
    this is used to bracket.
    """
    from ..core.frames import rtn_to_eci_matrix

    x = np.asarray(x, dtype=float).reshape(grid.n_vars)
    times = np.asarray(propagation_grid.times_s, dtype=float)
    index_of = {oid: i for i, oid in enumerate(propagation_grid.object_ids)}
    out: dict[str, np.ndarray] = {}

    for sat_id in grid.satellite_ids:
        row = index_of.get(sat_id)
        if row is None:
            continue
        n_rad_s = grid.mean_motion[sat_id]
        impulses = []
        for slot, epoch in enumerate(grid.slot_epochs(sat_id)):
            vector = np.zeros(3)
            for axis in (range(3) if grid.axes == 3 else (1,)):
                vector[axis] = x[grid.index(sat_id, slot, axis)]
            if np.any(vector):
                impulses.append((epoch, vector))
        rate = np.zeros((times.size, 3), dtype=float)
        if not impulses:
            out[sat_id] = rate
            continue
        for time_index in range(times.size):
            epoch = propagation_grid.epoch_at(time_index)
            local = np.zeros(3)
            for burn_epoch, vector in impulses:
                lead = seconds_between(burn_epoch, epoch)
                if lead <= 0.0:
                    continue
                local += cw_impulse_rate_matrix(n_rad_s, lead) @ vector
            if np.any(local):
                rotation = rtn_to_eci_matrix(
                    propagation_grid.positions_km[row, time_index],
                    propagation_grid.velocities_km_s[row, time_index],
                )
                rate[time_index] = rotation @ local
        out[sat_id] = rate
    return out


def _hermite_minimum(
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    step_s: float,
    samples: int = 64,
) -> tuple[float, float]:
    """Minimum of ``|p(t)|`` over one step, by cubic Hermite interpolation.

    Returns ``(fraction_of_step, separation_km)``. The relative position is
    interpolated from its endpoints *and* their derivatives, so the cubic
    matches the true trajectory to fourth order over the step -- far more than
    is needed to resolve a minimum that the endpoints themselves bracket.

    A dense scan followed by one parabolic polish is used rather than a root
    find on the range rate: the scan cannot converge to the wrong critical
    point, and 64 evaluations of a cubic cost nothing next to the SGP4 call
    that produced the endpoints.
    """
    u = np.linspace(0.0, 1.0, samples)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    curve = (
        h00[:, None] * p0[None, :]
        + h10[:, None] * (step_s * v0)[None, :]
        + h01[:, None] * p1[None, :]
        + h11[:, None] * (step_s * v1)[None, :]
    )
    norms = np.linalg.norm(curve, axis=1)
    best = int(np.argmin(norms))
    if 0 < best < samples - 1:
        y0, y1, y2 = norms[best - 1], norms[best], norms[best + 1]
        denominator = y0 - 2.0 * y1 + y2
        if abs(denominator) > 1e-18:
            shift = 0.5 * (y0 - y2) / denominator
            if -1.0 < shift < 1.0:
                return float(u[best] + shift * (u[1] - u[0])), float(
                    y1 - 0.25 * (y0 - y2) * shift
                )
    return float(u[best]), float(norms[best])


def perturbed_minimum_epochs(
    grid: BurnGrid,
    propagation_grid,
    x: np.ndarray,
    pairs: list[tuple[str, str]],
    floors: dict[str, float],
) -> list[tuple[str, str, int, float, float]]:
    """Where each pair's *perturbed* separation actually bottoms out.

    Returns ``(object_a, object_b, time_index, separation, fraction)`` for
    every pair whose perturbed minimum falls below its floor, sorted worst
    first. The minimum sits at ``fraction`` of the way from sample
    ``time_index`` to the next one, so it is generally **between** samples.

    Two distinct things make the sampled grid insufficient, and only the
    second is obvious.

    *Nominal minima are the wrong epochs.* For a co-orbital pair the nominal
    separation is flat -- 9.345 km at every sampled epoch -- so there are no
    meaningful nominal minima and the maneuver's own drift decides where the
    perturbed one falls.

    *Sampled minima are the wrong epochs too, for fast pairs.* Taking
    ``argmin`` over the samples silently assumes the minimum is visible at one
    of them. It need not be. A pair crossing at ``7.24`` km/s sweeps
    ``217`` km per 30 s step, and one measured case has a true minimum of
    **0.143 km** bracketed by samples of **76.3** and **140.8** km: the
    sampled separation is monotone increasing straight through a minimum three
    orders of magnitude below either neighbour. No test on sampled separations
    alone can find it, and the tidal inflation :math:`\tfrac12 A h^2` cannot
    cover it either -- that bounds the *curvature* of the relative trajectory,
    which is the right object only when the relative velocity at the minimum
    is near zero. For a crossing geometry the honest inter-sample bound is
    :math:`\tfrac12 v_{\text{rel}} h`, and at 7.24 km/s that is 108 km
    against a 2.9 km floor.

    The range rate resolves it. ``r_dot = (delta . delta_dot) / |delta|``
    changes sign at every minimum however deep, so brackets are found by a
    sign change in ``r_dot`` rather than by comparing separations, and each
    bracket is then refined by cubic Hermite interpolation of the relative
    position from its endpoints and their derivatives. This makes the search
    independent of how fast the pair is moving relative to the grid step.
    """
    displacements = satellite_displacements(grid, propagation_grid, x)
    if not displacements:
        return []
    rates = satellite_displacement_rates(grid, propagation_grid, x)
    index_of = {oid: i for i, oid in enumerate(propagation_grid.object_ids)}
    positions = np.asarray(propagation_grid.positions_km, dtype=float)
    velocities = np.asarray(propagation_grid.velocities_km_s, dtype=float)
    valid = np.asarray(propagation_grid.valid, dtype=bool)
    times = np.asarray(propagation_grid.times_s, dtype=float)
    if times.size < 2:
        return []
    zero = np.zeros((positions.shape[1], 3), dtype=float)

    found: list[tuple[str, str, int, float, float]] = []
    for object_a, object_b in pairs:
        row_a, row_b = index_of.get(object_a), index_of.get(object_b)
        if row_a is None or row_b is None:
            continue
        usable = valid[row_a] & valid[row_b]
        if not np.any(usable):
            continue
        delta = (positions[row_b] + displacements.get(object_b, zero)) - (
            positions[row_a] + displacements.get(object_a, zero)
        )
        delta_dot = (velocities[row_b] + rates.get(object_b, zero)) - (
            velocities[row_a] + rates.get(object_a, zero)
        )
        separations = np.linalg.norm(delta, axis=1)
        safe = np.where(separations > _ZERO, separations, _ZERO)
        range_rate = np.einsum("ij,ij->i", delta, delta_dot) / safe

        floor = floors.get(pair_id(object_a, object_b), 0.0)
        best: tuple[float, int, float] | None = None

        # Samples themselves, so a minimum sitting exactly on one is not lost.
        sampled = np.where(usable, separations, np.inf)
        index = int(np.argmin(sampled))
        if np.isfinite(sampled[index]):
            best = (float(sampled[index]), index, 0.0)

        # Brackets: approaching at the left sample, receding at the right one.
        approach = range_rate[:-1] <= 0.0
        recede = range_rate[1:] >= 0.0
        both_valid = usable[:-1] & usable[1:]
        for index in np.flatnonzero(approach & recede & both_valid):
            index = int(index)
            step = float(times[index + 1] - times[index])
            if step <= 0.0:
                continue
            fraction, separation = _hermite_minimum(
                delta[index], delta_dot[index], delta[index + 1], delta_dot[index + 1], step
            )
            if best is None or separation < best[0]:
                best = (separation, index, fraction)

        if best is not None and best[0] < floor - 1e-9:
            found.append((object_a, object_b, best[1], best[0], best[2]))

    found.sort(key=lambda item: item[3])
    return found


def thin_constraints(
    constraints: list[LatentConstraint],
    *,
    max_rows: int,
    per_pair: int = 3,
) -> list[LatentConstraint]:
    """Keep the rows most likely to bind, up to ``max_rows``.

    Ranking is by nominal margin ``separation - floor`` ascending, because a
    pair already close to the floor is the one a burn is most likely to push
    through it. At most ``per_pair`` rows are kept for any single pair so a
    long slow approach cannot crowd out every other pair.

    Dropping rows is sound but not complete: the returned program is a
    relaxation, so its solution may violate a dropped row. Always close the
    gap with :func:`aegis.fleetopt.solver.lazy_solve`, which re-checks every
    dropped row and adds back any that is violated.
    """
    if max_rows < 0:
        raise FleetOptError("max_rows must be non-negative")
    ordered = sorted(constraints, key=lambda item: (item.slack_km, item.pair_id, item.epoch))
    counts: dict[str, int] = {}
    kept: list[LatentConstraint] = []
    for constraint in ordered:
        if len(kept) >= max_rows:
            break
        seen = counts.get(constraint.pair_id, 0)
        if seen >= per_pair:
            continue
        counts[constraint.pair_id] = seen + 1
        kept.append(constraint)
    kept.sort(key=lambda item: (item.pair_id, item.epoch, item.kind))
    return kept
