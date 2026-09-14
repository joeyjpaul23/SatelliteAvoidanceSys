"""How far a bounded maneuver budget can move a satellite, and what that
means for screening completeness.

The problem this solves
-----------------------
Everyone handles maneuver-induced conjunctions with an outer loop: apply the
plan, regenerate the ephemeris, re-screen, repeat. NASA's Conjunction
Assessment Best Practices Handbook (NASA/SP-20230002470 Rev.1) describes
exactly that and calls it the best available approach. The loop has no
termination guarantee and, more importantly, no *completeness* guarantee: it
re-screens the one trajectory the optimizer happened to choose, so it can
only discover conjunctions that plan created. It cannot tell you which pairs
the optimizer had the power to endanger.

For an optimizer that takes "create no new conjunction" as a constraint, that
is the wrong question anyway. The constraint has to be imposed on every pair
the decision variables can reach, and to write that constraint set down we
need a bound on reachable displacement.

The bound
---------
**Proposition 4 (displacement bound).** Let satellite :math:`i` carry a total
impulse budget :math:`D_i = \\sum_k \\lVert \\Delta v_{i,k}\\rVert` with
earliest burn epoch :math:`\\tau_i^{\\min}`. Then for every admissible plan and
every :math:`t`,

.. math::

    \\lVert \\delta r_i(t) \\rVert
      = \\Big\\lVert \\sum_k \\Phi(n_i, t-\\tau_{i,k}) \\Delta v_{i,k} \\Big\\rVert
      \\le \\sum_k \\lVert \\Phi(n_i, t-\\tau_{i,k})\\rVert_2 \\lVert \\Delta v_{i,k}\\rVert
      \\le D_i \\max_{\\tau \\le t} \\lVert \\Phi(n_i, t-\\tau)\\rVert_2
      =: \\rho_i(t),

by the triangle inequality and submultiplicativity of the spectral norm. The
maximum over :math:`\\tau` is supplied in closed form by
:func:`aegis.fleetopt.dynamics.cw_impulse_norm_bound`, and asymptotically
:math:`\\rho_i(t) \\approx 3 D_i (t - \\tau_i^{\\min})` -- the secular
along-track term, and nothing else, sets the reachable radius.

**Proposition 5 (candidate-set completeness).** If a pair :math:`(a,b)`
satisfies

.. math::

    \\lVert d_{ab}(t) \\rVert > s_{\\min} + \\rho_a(t) + \\rho_b(t)
    \\quad \\text{for every } t \\text{ in the window,}

then no admissible plan can bring that pair within :math:`s_{\\min}`.
Equivalently: screening the *nominal* catalog with the inflated, time-varying
gate :math:`s_{\\min} + \\rho_a(t) + \\rho_b(t)` enumerates **every** pair any
admissible plan could turn into a conjunction. :math:`\\square`

*Proof.* The perturbed separation is
:math:`\\lVert d_{ab}(t) + \\delta r_b(t) - \\delta r_a(t)\\rVert \\ge
\\lVert d_{ab}(t)\\rVert - \\rho_a(t) - \\rho_b(t) > s_{\\min}`.

That converts "re-screen and hope" into a finite, certified candidate set.
The price is the gate size: at a 1 m/s budget and one day of lead,
:math:`3 D (t-\\tau) = 259` km per satellite, so the gate is hundreds of
kilometres wide. This is not pessimism in the bound, it is the truth about
what a 1 m/s budget can do over a day.

Two practical consequences follow, and both are implemented here:

* **Budget the gate, not just the fuel.** The gate is linear in :math:`D_i`,
  so capping the per-satellite budget at what avoidance actually needs (a few
  cm/s) shrinks the certified gate proportionally. A 5 cm/s budget over one
  day gives a 13 km gate, which is comparable to the operational screening
  volumes in :mod:`aegis.constants`.
* **Certify after the fact.** Once a plan is solved, re-evaluate the bound
  with :math:`D_i` replaced by the *realised* usage plus a margin
  (:func:`a_posteriori_gate_km`). The resulting gate is typically one to two
  orders of magnitude smaller than the a-priori gate, and the resulting
  certificate applies to the plan actually being flown.

Closest prior art: Rivero, Vazquez & Merz, "Advancing Geometric Conjunction
Filters to Handle Satellite Manoeuvres," SDC9 2025, which screens under
maneuvers but requires the *actual* planned ephemeris and establishes
zero-false-negative behaviour empirically rather than by proof. Hejduk &
Pachura (NASA NTRS 20170007928) explicitly decline to claim completeness for
screening volumes even in the non-maneuvering case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.timebase import ensure_utc, seconds_between
from .dynamics import BurnGrid, cw_impulse_norm_bound
from .errors import FleetOptError

__all__ = [
    "ReachabilityModel",
    "reach_radius_km",
    "certified_gate_km",
    "asymptotic_gate_km",
    "a_posteriori_gate_km",
    "realized_budgets",
]


def reach_radius_km(
    n_rad_s: float,
    budget_km_s: float,
    earliest_burn_epoch: datetime,
    epoch: datetime,
    *,
    mode: str = "analytic",
) -> float:
    """``rho_i(t)``: the largest displacement the budget can produce by ``epoch``.

    Zero before the first burn opportunity. See Proposition 4 in the module
    docstring for the derivation.
    """
    if budget_km_s < 0.0:
        raise FleetOptError("budget_km_s must be non-negative")
    lead_s = seconds_between(ensure_utc(earliest_burn_epoch), ensure_utc(epoch))
    if lead_s <= 0.0:
        return 0.0
    return float(budget_km_s) * cw_impulse_norm_bound(n_rad_s, lead_s, mode=mode)


def asymptotic_gate_km(budget_km_s: float, lead_s: float) -> float:
    """The quotable closed form ``3 * D * lead``.

    This is the asymptote of :func:`reach_radius_km`, not an upper bound on
    it -- the analytic envelope exceeds it by about ``7/n`` seconds of
    lead-equivalent. Use it for order-of-magnitude discussion and
    :func:`reach_radius_km` for anything that has to be correct.
    """
    return 3.0 * float(budget_km_s) * max(0.0, float(lead_s))


@dataclass
class ReachabilityModel:
    """Per-satellite reachable displacement over a planning window.

    ``budgets`` maps satellite id to its total impulse budget in km/s.
    ``earliest_burn`` maps satellite id to its first burn opportunity. Any
    object absent from both is treated as non-maneuverable, with a reach
    radius of exactly zero -- which is what makes third-party debris cheap to
    screen against.
    """

    mean_motion: dict[str, float]
    budgets: dict[str, float]
    earliest_burn: dict[str, datetime]
    mode: str = "analytic"
    exclusion_radius_km: float = 0.0
    _cache: dict[tuple[str, float], float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_grid(
        cls,
        grid: BurnGrid,
        budgets: dict[str, float],
        *,
        mode: str = "analytic",
        exclusion_radius_km: float = 0.0,
    ) -> "ReachabilityModel":
        return cls(
            mean_motion=dict(grid.mean_motion),
            budgets={sid: float(budgets.get(sid, 0.0)) for sid in grid.satellite_ids},
            earliest_burn={sid: grid.earliest_epoch(sid) for sid in grid.satellite_ids},
            mode=mode,
            exclusion_radius_km=float(exclusion_radius_km),
        )

    def is_maneuverable(self, object_id: str) -> bool:
        return object_id in self.budgets and self.budgets[object_id] > 0.0

    def radius_km(self, object_id: str, epoch: datetime) -> float:
        """``rho_i(t)`` for one object, memoised on the epoch."""
        if not self.is_maneuverable(object_id):
            return 0.0
        key = (object_id, ensure_utc(epoch).timestamp())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = reach_radius_km(
            self.mean_motion[object_id],
            self.budgets[object_id],
            self.earliest_burn[object_id],
            epoch,
            mode=self.mode,
        )
        self._cache[key] = value
        return value

    def gate_km(self, object_a: str, object_b: str, epoch: datetime) -> float:
        """The certified gate of Proposition 5 for one pair at one epoch."""
        return (
            self.exclusion_radius_km
            + self.radius_km(object_a, epoch)
            + self.radius_km(object_b, epoch)
        )

    def max_gate_km(self, object_a: str, object_b: str, window_end: datetime) -> float:
        """The largest gate over the window.

        The reach radius is non-decreasing in time (the envelope is), so the
        window maximum is attained at ``window_end``. This is the single
        number to hand to a fixed-box screener.
        """
        return self.gate_km(object_a, object_b, window_end)

    def summary(self, window_end: datetime) -> dict:
        radii = {
            sid: self.radius_km(sid, window_end) for sid in sorted(self.budgets)
        }
        return {
            "exclusion_radius_km": self.exclusion_radius_km,
            "mode": self.mode,
            "maneuverable_count": sum(1 for sid in self.budgets if self.is_maneuverable(sid)),
            "max_reach_km": max(radii.values()) if radii else 0.0,
            "median_reach_km": float(np.median(list(radii.values()))) if radii else 0.0,
            "worst_pair_gate_km": (
                self.exclusion_radius_km + 2.0 * max(radii.values()) if radii else 0.0
            ),
        }


def certified_gate_km(
    model: ReachabilityModel,
    object_a: str,
    object_b: str,
    epoch: datetime,
) -> float:
    """Free-function form of :meth:`ReachabilityModel.gate_km`."""
    return model.gate_km(object_a, object_b, epoch)


def realized_budgets(plan_totals: dict[str, float], *, margin: float = 1.5) -> dict[str, float]:
    """Realised per-satellite usage inflated by ``margin``.

    The margin exists because the a-posteriori certificate should survive
    small re-plans and execution errors. A margin below 1 would certify a
    plan against a budget smaller than the one it used, which is unsound, so
    values below 1 are rejected rather than clamped.
    """
    if margin < 1.0:
        raise FleetOptError("a-posteriori margin must be at least 1.0")
    return {sid: float(total) * float(margin) for sid, total in plan_totals.items()}


def a_posteriori_gate_km(
    model: ReachabilityModel,
    plan_totals: dict[str, float],
    object_a: str,
    object_b: str,
    epoch: datetime,
    *,
    margin: float = 1.5,
) -> float:
    """Re-certify a solved plan with a gate sized by its realised usage.

    Never larger than the a-priori gate when realised usage times ``margin``
    is within budget, which is the common case: avoidance maneuvers are
    centimetres per second against budgets of metres per second.
    """
    tightened = ReachabilityModel(
        mean_motion=model.mean_motion,
        budgets={
            sid: min(model.budgets.get(sid, 0.0), value)
            for sid, value in realized_budgets(plan_totals, margin=margin).items()
        },
        earliest_burn=model.earliest_burn,
        mode=model.mode,
        exclusion_radius_km=model.exclusion_radius_km,
    )
    return tightened.gate_km(object_a, object_b, epoch)
