"""Maneuvers and maneuver plans.

The output of the whole system is a :class:`ManeuverPlan`: for each satellite,
which burns to execute, when, and what risk reduction results. Everything here
is a value type -- the optimisation that produces plans lives in
:mod:`aegis.maneuver`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..constants import G0_M_S2
from .timebase import ensure_utc

__all__ = ["Maneuver", "SatelliteManeuverSet", "ManeuverPlan", "propellant_mass_kg"]


def propellant_mass_kg(delta_v_km_s: float, wet_mass_kg: float, isp_s: float) -> float:
    """Propellant consumed by an impulsive burn, via Tsiolkovsky.

    ``dm = m0 * (1 - exp(-dv / (Isp * g0)))``

    For collision avoidance magnitudes this is very nearly linear in delta-v
    (the error is below 0.1% for anything under 5 m/s), which is what allows
    fuel to enter the optimiser as a linear cost.

    The scale is worth internalising: a 5 cm/s avoidance maneuver on an 800 kg
    satellite with an argon Hall thruster at Isp 2500 s consumes about 1.6
    grams. Fuel is not the binding constraint in fleet operations -- service
    disruption and the re-screening burden are.
    """
    exhaust_velocity_km_s = isp_s * G0_M_S2 / 1000.0
    return wet_mass_kg * (1.0 - float(np.exp(-abs(delta_v_km_s) / exhaust_velocity_km_s)))


@dataclass
class Maneuver:
    """A single impulsive burn.

    Attributes
    ----------
    satellite_id
        Object this burn applies to.
    epoch
        When the impulse is applied, UTC.
    delta_v_rtn_km_s
        Impulse in the satellite's RTN frame at the burn epoch, shape ``(3,)``.
    rationale
        Which conjunctions motivated this burn, for operator review.
    """

    satellite_id: str
    epoch: datetime
    delta_v_rtn_km_s: np.ndarray
    rationale: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.epoch = ensure_utc(self.epoch)
        self.delta_v_rtn_km_s = np.asarray(self.delta_v_rtn_km_s, dtype=float).reshape(3)

    @property
    def magnitude_km_s(self) -> float:
        return float(np.linalg.norm(self.delta_v_rtn_km_s))

    @property
    def magnitude_mm_s(self) -> float:
        """Magnitude in mm/s -- the natural unit for avoidance maneuvers."""
        return self.magnitude_km_s * 1e6

    @property
    def is_along_track_only(self) -> bool:
        """Whether the burn is purely tangential.

        Along-track burns are overwhelmingly preferred. The reason is
        structural: in the Clohessy-Wiltshire response, radial and cross-track
        impulses produce only bounded periodic displacement (at most 2*dv/n),
        whereas the along-track response contains a secular term growing
        linearly in time. A tangential burn therefore buys unlimited
        displacement that costs nothing extra to grow -- you simply burn
        earlier. Cross-track burns are further penalised because they change
        inclination and node, which breaks constellation slot geometry.
        """
        radial, transverse, normal = np.abs(self.delta_v_rtn_km_s)
        return transverse > 0 and radial < transverse * 1e-6 and normal < transverse * 1e-6

    def propellant_kg(self, wet_mass_kg: float, isp_s: float) -> float:
        return propellant_mass_kg(self.magnitude_km_s, wet_mass_kg, isp_s)


@dataclass
class SatelliteManeuverSet:
    """All burns planned for one satellite in a single planning cycle."""

    satellite_id: str
    maneuvers: list[Maneuver] = field(default_factory=list)
    wet_mass_kg: float = 800.0
    isp_s: float = 2500.0

    @property
    def total_delta_v_km_s(self) -> float:
        return sum(maneuver.magnitude_km_s for maneuver in self.maneuvers)

    @property
    def total_delta_v_mm_s(self) -> float:
        return self.total_delta_v_km_s * 1e6

    @property
    def total_propellant_g(self) -> float:
        """Total propellant in grams."""
        return propellant_mass_kg(self.total_delta_v_km_s, self.wet_mass_kg, self.isp_s) * 1000.0

    @property
    def burn_count(self) -> int:
        return len(self.maneuvers)

    @property
    def first_burn_epoch(self) -> datetime | None:
        if not self.maneuvers:
            return None
        return min(maneuver.epoch for maneuver in self.maneuvers)


@dataclass
class ResolvedConjunction:
    """How one conjunction fared under a plan."""

    conjunction_id: str
    probability_before: float
    probability_after: float
    miss_distance_before_km: float
    miss_distance_after_km: float
    required_miss_distance_km: float
    resolved: bool
    shortfall_km: float = 0.0

    @property
    def risk_reduction_factor(self) -> float:
        """How many times smaller the probability became."""
        if self.probability_after <= 0:
            return float("inf")
        return self.probability_before / self.probability_after


@dataclass
class ManeuverPlan:
    """The system's deliverable: a coordinated fleet-wide maneuver plan.

    Attributes
    ----------
    plan_id
        Unique identifier for this planning cycle.
    generated_at
        When the plan was produced.
    satellite_sets
        Per-satellite burn sets, keyed by satellite id.
    resolved
        Per-conjunction outcomes, including any that could not be resolved.
    iterations
        How many plan / re-screen / re-plan cycles were needed to converge.
    converged
        Whether the re-screening loop reached a fixed point, i.e. the final
        plan created no new conjunctions above threshold.
    """

    plan_id: str
    generated_at: datetime
    satellite_sets: dict[str, SatelliteManeuverSet] = field(default_factory=dict)
    resolved: list[ResolvedConjunction] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.generated_at = ensure_utc(self.generated_at)

    @property
    def total_delta_v_mm_s(self) -> float:
        return sum(s.total_delta_v_mm_s for s in self.satellite_sets.values())

    @property
    def total_propellant_g(self) -> float:
        return sum(s.total_propellant_g for s in self.satellite_sets.values())

    @property
    def maneuvering_satellite_count(self) -> int:
        return sum(1 for s in self.satellite_sets.values() if s.burn_count > 0)

    @property
    def total_burns(self) -> int:
        return sum(s.burn_count for s in self.satellite_sets.values())

    @property
    def unresolved(self) -> list[ResolvedConjunction]:
        """Conjunctions the optimiser could not drive below threshold.

        These are the operationally important output. An unresolved
        conjunction usually means the required displacement was radial or
        cross-track dominant, where the achievable response is bounded -- no
        finite delta-v satisfies it. Surfacing them, ranked by shortfall, is
        far more useful than reporting an infeasible problem.
        """
        return [outcome for outcome in self.resolved if not outcome.resolved]

    @property
    def all_maneuvers(self) -> list[Maneuver]:
        """Every burn in the plan, ordered by execution time."""
        maneuvers = [m for s in self.satellite_sets.values() for m in s.maneuvers]
        return sorted(maneuvers, key=lambda m: m.epoch)

    def summary(self) -> dict:
        """Compact summary suitable for logging or an API response."""
        return {
            "plan_id": self.plan_id,
            "generated_at": self.generated_at.isoformat(),
            "maneuvering_satellites": self.maneuvering_satellite_count,
            "total_burns": self.total_burns,
            "total_delta_v_mm_s": round(self.total_delta_v_mm_s, 4),
            "total_propellant_g": round(self.total_propellant_g, 4),
            "conjunctions_addressed": len(self.resolved),
            "conjunctions_resolved": len(self.resolved) - len(self.unresolved),
            "conjunctions_unresolved": len(self.unresolved),
            "iterations": self.iterations,
            "converged": self.converged,
        }
