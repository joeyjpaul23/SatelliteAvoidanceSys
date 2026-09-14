"""Benchmark sweeps, honest metrics, and the safety-premium distribution.

The research deliverable. :mod:`aegis.fleetopt` can produce a plan; this
package is what turns plans into a claim you can check, by running every
planner over every scenario under identical inputs and measuring the result
against SGP4 rather than against the optimizer's own model.

Entry point: ``python -m aegis.experiments --help``.
"""

from __future__ import annotations

from .metrics import (
    InducedMeasurement,
    LinearizationError,
    PlannerMetrics,
    collect_metrics,
    measure_induced,
    measure_linearization,
)
from .report import premium_by_family, premium_by_structure, render_json, render_markdown
from .runner import (
    DEFAULT_PLANNERS,
    LEARNED_PLANNERS,
    BenchmarkConfig,
    BenchmarkReport,
    ScenarioOutcome,
    run_benchmark,
    run_scenario,
)

__all__ = [
    "InducedMeasurement",
    "LinearizationError",
    "PlannerMetrics",
    "collect_metrics",
    "measure_induced",
    "measure_linearization",
    "premium_by_family",
    "premium_by_structure",
    "render_json",
    "render_markdown",
    "DEFAULT_PLANNERS",
    "LEARNED_PLANNERS",
    "BenchmarkConfig",
    "BenchmarkReport",
    "ScenarioOutcome",
    "run_benchmark",
    "run_scenario",
]
