"""Deterministic conjunction-graph scenario generation (step14 section 13).

Builds small, structurally-verified catalogs -- an isolated pair, a chain, a
star, a clique, matched-plane and crossing-plane pairs, a debris shower, a
dense Starlink-like shell, and TLE replays -- so ``aegis.fleetopt``'s
planners and certifier have known-shape conjunction graphs to run against
instead of only whatever a live catalog happens to contain that day.
"""

from __future__ import annotations

from .errors import ScenarioError
from .generator import FAMILIES, describe, generate
from .registry import SCENARIO_SUITES, expand_suite
from .spec import Scenario, ScenarioSpec, scenario_digest

__all__ = [
    "ScenarioError",
    "ScenarioSpec",
    "Scenario",
    "scenario_digest",
    "FAMILIES",
    "generate",
    "describe",
    "SCENARIO_SUITES",
    "expand_suite",
]
