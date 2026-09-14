"""Command line for the benchmark sweeps.

``--help`` on every subcommand is meant to be enough to reproduce a result
without reading the source, because a research number nobody can regenerate is
an anecdote.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..fleetopt.pareto import premium_frontier
from ..ingest.synthetic import SyntheticAuthorization
from ..scenarios import FAMILIES, SCENARIO_SUITES, describe, generate, scenario_digest
from .report import render_json, render_markdown
from .runner import DEFAULT_PLANNERS, LEARNED_PLANNERS, BenchmarkConfig, run_benchmark

__all__ = ["main", "build_parser"]

_ALL_PLANNERS = DEFAULT_PLANNERS + LEARNED_PLANNERS


def _authorization(args) -> SyntheticAuthorization | None:
    """Synthetic scenarios stay opt-in, exactly as `aegis.ingest` requires."""
    if not args.acknowledge_synthetic:
        return None
    return SyntheticAuthorization(acknowledge_synthetic=True)


def _stamp(args) -> str:
    if getattr(args, "stamp", None):
        return str(args.stamp)
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aegis.experiments",
        description=(
            "Fleet maneuver optimization benchmarks. Synthetic scenarios require "
            "AEGIS_ALLOW_SYNTHETIC=1 and --acknowledge-synthetic; there is no silent "
            "fallback to fake data."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--acknowledge-synthetic", action="store_true",
                         help="acknowledge that synthetic catalogs are not real data")
        sub.add_argument("--output", type=Path, default=None,
                         help="directory for report.json and report.md")
        sub.add_argument("--store", type=str, default=None,
                         help="sqlite path for the experiment store")
        sub.add_argument("--stamp", type=str, default=None,
                         help="explicit UTC stamp, so a rerun lands in the same run id")
        sub.add_argument("--quiet", action="store_true")

    bench = subparsers.add_parser("benchmark", help="run a planner sweep")
    add_common(bench)
    bench.add_argument("--suite", choices=sorted(SCENARIO_SUITES), default="smoke")
    bench.add_argument("--family", action="append", default=None,
                       help="restrict to one family (repeatable)")
    bench.add_argument("--seeds", type=int, default=None,
                       help="number of seeds per family when --family is used")
    bench.add_argument("--planner", action="append", default=None,
                       choices=list(_ALL_PLANNERS), help="planner to include (repeatable)")
    bench.add_argument("--target-pc", type=float, default=None)
    bench.add_argument("--exclusion-km", type=float, default=None)
    bench.add_argument("--budget-mm-s", type=float, default=None)
    bench.add_argument("--burn-slots", type=int, default=None)
    bench.add_argument("--latent-mode", type=str, default=None,
                       choices=["none", "certified_minima", "dense_grid", "both"])
    bench.add_argument("--latent-step-s", type=float, default=None)
    bench.add_argument("--cost-model", type=str, default=None, choices=["l1", "cone"])
    bench.add_argument("--no-measure-induced", action="store_true",
                       help="skip the SGP4 re-screen (much faster, much less evidence)")
    bench.add_argument("--no-measure-linearization", action="store_true")
    bench.add_argument("--checkpoint", type=str, default=None,
                       help="trained model checkpoint for the pignn-* planners")
    bench.add_argument("--notes", type=str, default="")

    frontier = subparsers.add_parser("frontier", help="trace V*(epsilon) for one scenario")
    add_common(frontier)
    frontier.add_argument("--family", type=str, required=True, choices=list(FAMILIES))
    frontier.add_argument("--seed", type=int, default=0)
    frontier.add_argument("--points", type=int, default=9)
    frontier.add_argument("--exclusion-km", type=float, default=None)

    listing = subparsers.add_parser("scenario-list", help="describe the scenario families")
    listing.add_argument("--suite", choices=sorted(SCENARIO_SUITES), default=None)

    inspect = subparsers.add_parser("scenario-show", help="generate and summarise one scenario")
    add_common(inspect)
    inspect.add_argument("--family", type=str, required=True, choices=list(FAMILIES))
    inspect.add_argument("--seed", type=int, default=0)

    validate = subparsers.add_parser(
        "validate", help="measure model fidelity against SGP4 for one scenario"
    )
    add_common(validate)
    validate.add_argument("--family", type=str, default="crossing-planes", choices=list(FAMILIES))
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--planner", type=str, default="fleet-safe", choices=list(_ALL_PLANNERS))

    report = subparsers.add_parser("report", help="re-render a stored report.json")
    report.add_argument("path", type=Path)
    report.add_argument("--output", type=Path, default=None)

    return parser


def _config_from(args) -> BenchmarkConfig:
    config = BenchmarkConfig(suite=args.suite, notes=args.notes)
    if args.planner:
        config.planners = tuple(args.planner)
    if args.family:
        seeds = args.seeds if args.seeds is not None else 3
        config.scenarios = tuple(
            (family, seed) for family in args.family for seed in range(seeds)
        )
    if args.target_pc is not None:
        config.target_pc = args.target_pc
    if args.exclusion_km is not None:
        config.induced_exclusion_km = args.exclusion_km
    if args.budget_mm_s is not None:
        config.dv_budget_km_s = args.budget_mm_s * 1e-6
    if args.burn_slots is not None:
        config.burn_slots = args.burn_slots
    if args.latent_mode is not None:
        config.latent_mode = args.latent_mode
    if args.latent_step_s is not None:
        config.latent_step_s = args.latent_step_s
    if args.cost_model is not None:
        config.cost_model_name = args.cost_model
    if args.no_measure_induced:
        config.measure_induced = False
    if args.no_measure_linearization:
        config.measure_linearization = False
    if args.checkpoint:
        config.model_checkpoint = args.checkpoint
    return config


def _write(output: Path | None, name: str, text: str, quiet: bool) -> None:
    if output is None:
        if not quiet:
            print(text)
        return
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    path.write_text(text, encoding="utf-8")
    if not quiet:
        print(f"wrote {path}")


def _cmd_benchmark(args) -> int:
    config = _config_from(args)
    authorization = _authorization(args)

    def progress(index: int, total: int, name: str) -> None:
        if not args.quiet:
            print(f"[{index}/{total}] {name}", file=sys.stderr, flush=True)

    report = run_benchmark(
        config,
        stamp=_stamp(args),
        authorization=authorization,
        store_path=args.store,
        progress=progress,
    )
    _write(args.output, "report.json", render_json(report), args.quiet)
    _write(args.output, "report.md", render_markdown(report), args.quiet)
    if args.output is not None and not args.quiet:
        stats = report.premium_summary()
        print(
            f"{report.scenario_count} scenarios, {report.row_count} rows, "
            f"{report.error_count} errors; median premium "
            f"{100 * stats['median']:.2f}%, infeasible {100 * stats['infeasible_fraction']:.1f}%"
        )
    return 0


def _cmd_frontier(args) -> int:
    from ..fleetopt.planners import PlanRequest, build_context
    from ..risk.batch import assess_catalog
    from ..screening import screen

    scenario = generate(args.family, args.seed, authorization=_authorization(args))
    conjunctions = screen(
        scenario.objects,
        scenario.window_start,
        scenario.window_duration_s,
        box_km=scenario.screening_box_km,
    )
    assessed = assess_catalog(conjunctions, objects=scenario.objects)
    request = PlanRequest(
        assessed=assessed,
        objects=scenario.objects,
        window_start=scenario.window_start,
        window_duration_s=scenario.window_duration_s,
        now=scenario.window_start,
        screening_box_km=scenario.screening_box_km,
    )
    if args.exclusion_km is not None:
        request.induced_exclusion_km = args.exclusion_km
    from ..fleetopt.solver import sequential_solve

    context = build_context(request)
    fuel = context.problem(latent=[])
    reference = sequential_solve(fuel, max_iterations=request.scp_iterations)
    frontier = premium_frontier(
        fuel,
        context.problem(latent=list(context.latent)),
        dict(reference.directions),
        points=args.points,
    )
    payload = {
        "scenario": scenario.scenario_id,
        "digest": scenario_digest(scenario),
        "conjunctions": len(assessed.entries),
        "latent_rows": len(context.latent),
        "frontier": frontier.summary(),
        "points": [
            {
                "epsilon_km": point.epsilon,
                "delta_v_mm_s": point.delta_v_km_s * 1e6,
                "multiplier": point.multiplier,
                "induced_shortfall_km": point.induced_shortfall_km,
                "status": point.status,
            }
            for point in frontier.points
        ],
    }
    _write(args.output, "frontier.json", json.dumps(payload, indent=2, default=str), args.quiet)
    return 0


def _cmd_scenario_list(args) -> int:
    if args.suite:
        pairs = SCENARIO_SUITES[args.suite]
        print(f"suite {args.suite}: {len(pairs)} scenarios")
        families = sorted({family for family, _ in pairs})
        for family in families:
            seeds = [seed for f, seed in pairs if f == family]
            print(f"  {family:18s} seeds {min(seeds)}..{max(seeds)} ({len(seeds)})")
        return 0
    for family in FAMILIES:
        print(f"{family:18s} {describe(family)}")
    return 0


def _cmd_scenario_show(args) -> int:
    from ..fleetopt.graph import build_conjunction_graph, graph_metrics
    from ..risk.batch import assess_catalog
    from ..screening import screen

    scenario = generate(args.family, args.seed, authorization=_authorization(args))
    conjunctions = screen(
        scenario.objects,
        scenario.window_start,
        scenario.window_duration_s,
        box_km=scenario.screening_box_km,
    )
    assessed = assess_catalog(conjunctions, objects=scenario.objects)
    metrics = graph_metrics(build_conjunction_graph(assessed, scenario.objects))
    payload = {
        "scenario_id": scenario.scenario_id,
        "digest": scenario_digest(scenario),
        "family": scenario.family,
        "seed": scenario.seed,
        "objects": len(scenario.objects),
        "window_duration_s": scenario.window_duration_s,
        "conjunctions": len(assessed.entries),
        "graph": metrics.as_dict(),
        "provenance": scenario.provenance,
    }
    _write(args.output, "scenario.json", json.dumps(payload, indent=2, default=str), args.quiet)
    return 0


def _cmd_validate(args) -> int:
    from ..fleetopt.planners import PlanRequest, build_context, get_planner
    from ..risk.batch import assess_catalog
    from ..screening import screen
    from .metrics import measure_induced, measure_linearization

    scenario = generate(args.family, args.seed, authorization=_authorization(args))
    conjunctions = screen(
        scenario.objects,
        scenario.window_start,
        scenario.window_duration_s,
        box_km=scenario.screening_box_km,
    )
    assessed = assess_catalog(conjunctions, objects=scenario.objects)
    request = PlanRequest(
        assessed=assessed,
        objects=scenario.objects,
        window_start=scenario.window_start,
        window_duration_s=scenario.window_duration_s,
        now=scenario.window_start,
        screening_box_km=scenario.screening_box_km,
    )
    context = build_context(request)
    plan = get_planner(args.planner).plan(context)
    linearization = measure_linearization(scenario.objects, plan, context.sensitivities)
    induced = measure_induced(
        scenario.objects,
        plan,
        assessed,
        scenario.window_start,
        scenario.window_duration_s,
        step_s=request.latent_step_s,
        box_km=scenario.screening_box_km,
    )
    payload = {
        "scenario": scenario.scenario_id,
        "planner": plan.planner,
        "total_dv_mm_s": plan.total_delta_v_mm_s,
        "resolved": plan.resolved_count,
        "unresolved": plan.unresolved_count,
        "certified_safe": plan.certified_safe,
        "linearization": linearization.as_dict(),
        "induced": induced.as_dict(),
        "induced_predicted": plan.induced.as_dict() if plan.induced else None,
    }
    _write(args.output, "validation.json", json.dumps(payload, indent=2, default=str), args.quiet)
    return 0


def _cmd_report(args) -> int:
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(text, encoding="utf-8")
        print(f"wrote {args.output / 'report.json'}")
    else:
        print(text[:20000])
    return 0


_COMMANDS = {
    "benchmark": _cmd_benchmark,
    "frontier": _cmd_frontier,
    "scenario-list": _cmd_scenario_list,
    "scenario-show": _cmd_scenario_show,
    "validate": _cmd_validate,
    "report": _cmd_report,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "acknowledge_synthetic", False) and not os.environ.get(
        "AEGIS_ALLOW_SYNTHETIC"
    ):
        print(
            "AEGIS_ALLOW_SYNTHETIC=1 is also required. Synthetic catalogs are opt-in "
            "twice on purpose: once in the environment, once on the command line.",
            file=sys.stderr,
        )
        return 2
    return _COMMANDS[args.command](args)
