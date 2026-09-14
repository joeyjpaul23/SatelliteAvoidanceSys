"""Exceptions for the fleet optimization layer."""

from __future__ import annotations

__all__ = [
    "FleetOptError",
    "GridError",
    "AssemblyError",
    "SolverBackendError",
    "CertificationError",
]


class FleetOptError(RuntimeError):
    """Base class for every fleet optimization failure."""


class GridError(FleetOptError):
    """A burn grid was requested that cannot be built."""


class AssemblyError(FleetOptError):
    """The linear program could not be assembled from the problem data."""


class SolverBackendError(FleetOptError):
    """No solver backend produced a usable result."""


class CertificationError(FleetOptError):
    """A certificate could not be computed."""
