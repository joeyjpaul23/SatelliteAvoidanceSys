"""Rendering a sweep into something a person can argue with.

Every table here reports the thing that could falsify the claim next to the
claim itself: predicted induced conjunctions beside measured ones, delta-v
beside the linearisation error that qualifies it, the premium as a
distribution rather than a mean, and the infeasible fraction rather than a
quietly smaller denominator.
"""

from __future__ import annotations

import json

import numpy as np

from ..fleetopt.pareto import Frontier, PremiumRecord, premium_statistics
from .runner import BenchmarkReport

__all__ = ["render_markdown", "render_json", "premium_by_family", "premium_by_structure"]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def premium_by_family(records: list[PremiumRecord]) -> dict[str, dict]:
    grouped: dict[str, list[PremiumRecord]] = {}
    for record in records:
        grouped.setdefault(record.family or "unknown", []).append(record)
    return {family: premium_statistics(rows) for family, rows in sorted(grouped.items())}


def premium_by_structure(
    records: list[PremiumRecord],
    *,
    coupling_edges: tuple[float, ...] = (1.0, 2.0, 10.0, 100.0, 1e6),
) -> dict[str, dict]:
    """Stratify the premium by the fleet coupling number.

    The working hypothesis the handoff document set out is that sparse,
    weakly-coupled conjunction networks pay a small premium while tightly
    coupled clusters pay a large one or become infeasible. This is the table
    that tests it.
    """
    buckets: dict[str, list[PremiumRecord]] = {}
    for record in records:
        kappa = record.coupling_number
        label = f">{coupling_edges[-1]:g}"
        for low, high in zip((0.0,) + coupling_edges[:-1], coupling_edges):
            if low <= kappa < high:
                label = f"[{low:g}, {high:g})"
                break
        buckets.setdefault(label, []).append(record)
    return {label: premium_statistics(rows) for label, rows in sorted(buckets.items())}


def render_json(report: BenchmarkReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str)


def render_markdown(report: BenchmarkReport, *, frontier: Frontier | None = None) -> str:
    lines: list[str] = []
    config = report.config
    lines.append(f"# Fleet optimization benchmark — `{report.run_id}`")
    lines.append("")
    lines.append(
        f"Suite `{config.suite}` · {report.scenario_count} scenarios · "
        f"{report.row_count} planner rows · {report.error_count} errors"
    )
    lines.append("")
    # The exclusion radius is None when it is derived per scenario from the
    # assessed covariances, which is the default. Formatting None with ":g"
    # raises, and it raised on the exact command line the README prints.
    exclusion = (
        "derived per scenario from the assessed covariances"
        if config.induced_exclusion_km is None
        else f"{config.induced_exclusion_km:g} km"
    )
    lines.append(
        f"Target Pc {config.target_pc:.0e} · exclusion radius {exclusion} · "
        f"budget {config.dv_budget_km_s * 1e6:.0f} mm/s · {config.burn_slots} burn slots · "
        f"latent mode `{config.latent_mode}` at {config.latent_step_s:g} s"
    )
    lines.append("")
    lines.append(
        "> Every collision probability on this page inherits TLE-grade covariance "
        "(`CovarianceSource.SYNTHETIC_TLE`). See `docs/LIMITATIONS.md`. "
        "Induced-conjunction counts are reported twice: `predicted` is the optimizer's "
        "own constraint residual, `measured` is a full SGP4 re-screen of the maneuvered "
        "catalog. Only the second is evidence."
    )
    lines.append("")

    lines.append("## Planner comparison")
    lines.append("")
    header = (
        "| planner | scenarios | median Δv (mm/s) | total Δv (mm/s) | induced pred. | "
        "induced meas. | scenarios w/ induced | feasible | certified | unresolved | median solve (s) |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 11)
    for planner, row in report.planner_table().items():
        lines.append(
            f"| `{planner}` | {row['scenarios']} | {_fmt(row['median_dv_mm_s'])} | "
            f"{_fmt(row['total_dv_mm_s'])} | {row['induced_predicted_total']} | "
            f"**{row['induced_measured_total']}** | {row['scenarios_with_induced_measured']} | "
            f"{_fmt(100 * row['feasible_fraction'], 1)}% | "
            f"{_fmt(100 * row['certified_fraction'], 1)}% | {row['unresolved_total']} | "
            f"{_fmt(row['median_solver_time_s'], 4)} |"
        )
    lines.append("")

    stats = report.premium_summary()
    lines.append("## Safety premium")
    lines.append("")
    lines.append(
        "`pi = (dv_fleet-safe - dv_fuel-only) / dv_fuel-only`, the extra delta-v "
        "a coordinated plan pays to create no new conjunction."
    )
    lines.append("")
    lines.append("| statistic | value |")
    lines.append("|---|---|")
    lines.append(f"| scenarios compared | {stats['count']} |")
    lines.append(f"| premium defined for | {stats['defined']} |")
    lines.append(f"| median | {_fmt(100 * stats['median'], 2)}% |")
    lines.append(f"| p90 | {_fmt(100 * stats['p90'], 2)}% |")
    lines.append(f"| p99 | {_fmt(100 * stats['p99'], 2)}% |")
    lines.append(f"| worst | {_fmt(100 * stats['worst'], 2)}% |")
    lines.append(f"| mean | {_fmt(100 * stats['mean'], 2)}% |")
    lines.append(f"| zero-premium fraction | {_fmt(100 * stats['zero_premium_fraction'], 1)}% |")
    lines.append(f"| **infeasible fraction** | {_fmt(100 * stats['infeasible_fraction'], 1)}% |")
    lines.append("")
    lines.append("**At a fixed linearization** — both planners solved once at the same")
    lines.append("directions, where `fleet-safe`'s feasible set is provably a subset of")
    lines.append("`fuel-only`'s and the premium is therefore a genuine lower bound:")
    lines.append("")
    lines.append("| statistic | value |")
    lines.append("|---|---|")
    lines.append(f"| defined for | {stats['fixed_defined']} |")
    lines.append(f"| median | {_fmt(100 * stats['fixed_median'], 2)}% |")
    lines.append(f"| p90 | {_fmt(100 * stats['fixed_p90'], 2)}% |")
    lines.append(f"| p99 | {_fmt(100 * stats['fixed_p99'], 2)}% |")
    lines.append(f"| worst | {_fmt(100 * stats['fixed_worst'], 2)}% |")
    lines.append(f"| **negative (should be 0)** | {stats['fixed_negative_count']} |")
    lines.append("")
    lines.append("**Absolute premium**, which is defined for every scenario and is not")
    lines.append("inflated by a small denominator:")
    lines.append("")
    lines.append("| statistic | as planned | at fixed linearization |")
    lines.append("|---|---|---|")
    lines.append(
        f"| median | {_fmt(stats['absolute_median_mm_s'], 3)} mm/s | "
        f"{_fmt(stats['fixed_absolute_median_mm_s'], 3)} mm/s |"
    )
    lines.append(
        f"| p90 | {_fmt(stats['absolute_p90_mm_s'], 3)} mm/s | "
        f"{_fmt(stats['fixed_absolute_p90_mm_s'], 3)} mm/s |"
    )
    lines.append("")
    lines.append(
        "The ratio is withheld when the fuel-only plan costs less than "
        "100 mm/s: a denominator that small turns an operationally trivial "
        "difference into a four-figure percentage. Three scenarios in an "
        "earlier sweep reported 1565 %, 2764 % and 3717 % that way."
    )
    lines.append(f"| undefined (no maneuver needed) | {_fmt(100 * stats['undefined_fraction'], 1)}% |")
    lines.append("")
    lines.append(
        "The mean is reported last and deliberately: the premium is bounded below by "
        "zero and unbounded above, so the tail and the infeasible fraction are the "
        "result. A comparison that drops infeasible scenarios reports a flattering "
        "number for a planner that simply refused the hard cases."
    )
    lines.append("")

    by_family = premium_by_family(report.premiums)
    if by_family:
        lines.append("### By scenario family")
        lines.append("")
        lines.append("| family | n | median | p90 | worst | infeasible |")
        lines.append("|---|---|---|---|---|---|")
        for family, row in by_family.items():
            lines.append(
                f"| `{family}` | {row['count']} | {_fmt(100 * row['median'], 2)}% | "
                f"{_fmt(100 * row['p90'], 2)}% | {_fmt(100 * row['worst'], 2)}% | "
                f"{_fmt(100 * row['infeasible_fraction'], 1)}% |"
            )
        lines.append("")

    by_structure = premium_by_structure(report.premiums)
    if by_structure:
        lines.append("### By fleet coupling number")
        lines.append("")
        lines.append(
            "`coupling_number = cond(B B^T)` of the row-normalised projected resolve "
            "sensitivities. Large means several conjunctions demand nearly collinear "
            "displacements, so satisfying one nearly determines the others."
        )
        lines.append("")
        lines.append("| coupling κ | n | median | p90 | worst | infeasible |")
        lines.append("|---|---|---|---|---|---|")
        for label, row in by_structure.items():
            lines.append(
                f"| {label} | {row['count']} | {_fmt(100 * row['median'], 2)}% | "
                f"{_fmt(100 * row['p90'], 2)}% | {_fmt(100 * row['worst'], 2)}% | "
                f"{_fmt(100 * row['infeasible_fraction'], 1)}% |"
            )
        lines.append("")

    samples = [
        row.values
        for outcome in report.outcomes
        for row in outcome.metrics
        if row.values.get("linearization_samples", 0)
    ]
    if samples:
        median = float(np.median([s["linearization_error_km_median"] for s in samples]))
        p95 = float(np.percentile([s["linearization_error_km_p95"] for s in samples], 95))
        worst = float(max(s["linearization_error_km_max"] for s in samples))
        relative = float(np.median([s["linearization_relative_median"] for s in samples]))
        lines.append("## Linearisation error against SGP4")
        lines.append("")
        lines.append(
            "Predicted post-maneuver miss distance versus the minimum separation an "
            "SGP4 re-propagation of the maneuvered catalog actually shows."
        )
        lines.append("")
        lines.append(
            f"| samples | median | p95 | max | median relative |\n|---|---|---|---|---|\n"
            f"| {len(samples)} | {median * 1000:.1f} m | {p95 * 1000:.1f} m | "
            f"{worst * 1000:.1f} m | {100 * relative:.2f}% |"
        )
        lines.append("")

    if frontier is not None and frontier.points:
        lines.append("## Induced-risk frontier")
        lines.append("")
        lines.append(
            "`V*(epsilon)` over the induced-shortfall budget. Proposition 7 says this "
            "is convex, piecewise linear and non-increasing, with slope equal to minus "
            "the `induced-budget` dual — so convexity is a checkable invariant, not an "
            "impression."
        )
        lines.append("")
        summary = frontier.summary()
        lines.append(
            f"non-increasing: **{summary['non_increasing']}** · convex: "
            f"**{summary['is_convex']}** (residual {summary['convexity_residual']:.2e}) · "
            f"slope/dual agreement {summary['slope_agreement']:.2e} · "
            f"breakpoints at {summary['breakpoints']}"
        )
        lines.append("")
        lines.append("| ε (km) | Δv (mm/s) | multiplier |")
        lines.append("|---|---|---|")
        for point in frontier.points:
            lines.append(
                f"| {point.epsilon:.4g} | {point.delta_v_km_s * 1e6:.4f} | "
                f"{point.multiplier:.6g} |"
            )
        lines.append("")

    if report.error_count:
        lines.append("## Failures")
        lines.append("")
        lines.append(
            "Recorded, not dropped. A sweep that silently discards the cases its "
            "planners could not handle reports a distribution about the easy half."
        )
        lines.append("")
        for outcome in report.outcomes:
            for planner, message in outcome.errors.items():
                first = message.splitlines()[0] if message else ""
                lines.append(f"- `{outcome.scenario.scenario_id}` / `{planner}`: {first}")
        lines.append("")

    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)
