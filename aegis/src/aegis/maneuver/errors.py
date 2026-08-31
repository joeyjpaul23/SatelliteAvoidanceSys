"""Maneuver-layer errors."""

from __future__ import annotations

__all__ = ["ManeuverSolverError"]


class ManeuverSolverError(RuntimeError):
    """Raised when the planner cannot even form a problem.

    Distinct from an infeasible geometry. Slack is mandatory: a conjunction
    that cannot be separated along-track still yields a
    :class:`~aegis.core.maneuver.ManeuverPlan` with unresolved entries.
    This error is reserved for invalid arguments or a solver setup that
    cannot run at all.
    """
