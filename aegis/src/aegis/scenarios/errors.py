"""Errors raised while building or digesting scenarios."""

from __future__ import annotations

__all__ = ["ScenarioError"]


class ScenarioError(Exception):
    """Raised when a scenario cannot be built or is internally inconsistent.

    Covers an unknown family name, a geometrically degenerate parameter
    choice (e.g. two orbital planes requested so close together that they
    have no distinct crossing line), and a fixture file that fails to
    parse. A scenario builder never returns a partial or silently-degraded
    ``Scenario`` in place of raising this -- see the "no silent fallback"
    policy that governs the rest of AEGIS.
    """
