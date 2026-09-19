"""Honest maneuver-plan artifact: JSON, human-readable text, and OEM files."""

from __future__ import annotations

import json
from pathlib import Path

from ..ccsds import CcsdsError, write_oem
from ..core.maneuver import Maneuver, ResolvedConjunction
from ..core.state import CovarianceSource
from ..maneuver import apply_along_track_burns
from ..propagation import PropagationError, Sgp4Propagator
from .config import PipelineResult

__all__ = ["plan_artifact", "write_plan_json", "write_plan_text", "write_plan_oems"]

_OEM_SAMPLES = 11
_DEFAULT_OEM_DURATION_S = 5400.0

_SYNTHETIC_TLE_WARNING = (
    "Covariance is synthetic / TLE-grade (SYNTHETIC_TLE) and is not for "
    "operational maneuver decisions."
)


def _iso_utc(moment) -> str:
    return moment.isoformat()


def _burn_record(maneuver: Maneuver) -> dict:
    radial, transverse, normal = (float(v) for v in maneuver.delta_v_rtn_km_s)
    return {
        "satellite_id": maneuver.satellite_id,
        "epoch": _iso_utc(maneuver.epoch),
        "delta_v_rtn_km_s": [radial, transverse, normal],
        "magnitude_km_s": float(maneuver.magnitude_km_s),
    }


def _outcome_record(outcome: ResolvedConjunction) -> dict:
    return {
        "conjunction_id": outcome.conjunction_id,
        "probability_before": float(outcome.probability_before),
        "probability_after": float(outcome.probability_after),
        "miss_distance_before_km": float(outcome.miss_distance_before_km),
        "miss_distance_after_km": float(outcome.miss_distance_after_km),
        "resolved": bool(outcome.resolved),
        "shortfall_km": float(outcome.shortfall_km),
    }


def _warnings(result: PipelineResult) -> list[str]:
    warnings: list[str] = []
    if result.covariance_source == CovarianceSource.SYNTHETIC_TLE:
        warnings.append(_SYNTHETIC_TLE_WARNING)
    warnings.extend(result.plan.notes)
    return warnings


def plan_artifact(result: PipelineResult) -> dict:
    """JSON-serializable plan artifact from a pipeline result."""
    plan = result.plan
    resolved = [_outcome_record(outcome) for outcome in plan.resolved]
    unresolved = [row for row in resolved if row["resolved"] is False]
    unresolved.sort(key=lambda row: row["shortfall_km"], reverse=True)
    return {
        "source": result.source,
        "covariance_source": result.covariance_source,
        "object_count": len(result.catalog),
        "generated_at": _iso_utc(plan.generated_at),
        "plan_id": plan.plan_id,
        "summary": plan.summary(),
        "burns": [_burn_record(maneuver) for maneuver in plan.all_maneuvers],
        "resolved": resolved,
        "unresolved": unresolved,
        "warnings": _warnings(result),
        "converged": bool(plan.converged),
        "iterations": int(plan.iterations),
    }


def write_plan_json(result: PipelineResult, path: str | Path) -> Path:
    """Write :func:`plan_artifact` as indented JSON."""
    dest = Path(path)
    dest.write_text(json.dumps(plan_artifact(result), indent=2) + "\n", encoding="utf-8")
    return dest


def _format_plan_text(result: PipelineResult) -> str:
    artifact = plan_artifact(result)
    summary = artifact["summary"]
    lines = [
        "AEGIS maneuver plan",
        "===================",
        f"source: {artifact['source']}",
        f"covariance_source: {artifact['covariance_source']}",
        f"object_count: {artifact['object_count']}",
        f"plan_id: {artifact['plan_id']}",
        f"generated_at: {artifact['generated_at']}",
        f"converged: {artifact['converged']}",
        f"iterations: {artifact['iterations']}",
        "",
        "Honesty",
        "-------",
    ]
    if artifact["warnings"]:
        lines.extend(f"- {warning}" for warning in artifact["warnings"])
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "Summary",
            "-------",
            f"maneuvering_satellites: {summary.get('maneuvering_satellites')}",
            f"total_burns: {summary.get('total_burns')}",
            f"total_delta_v_mm_s: {summary.get('total_delta_v_mm_s')}",
            f"total_propellant_g: {summary.get('total_propellant_g')}",
            f"conjunctions_addressed: {summary.get('conjunctions_addressed')}",
            f"conjunctions_resolved: {summary.get('conjunctions_resolved')}",
            f"conjunctions_unresolved: {summary.get('conjunctions_unresolved')}",
            "",
            "Burns",
            "-----",
        ]
    )
    burns = artifact["burns"]
    if not burns:
        lines.append("(none)")
    else:
        for burn in burns:
            dv = burn["delta_v_rtn_km_s"]
            lines.append(
                f"- satellite_id={burn['satellite_id']} epoch={burn['epoch']} "
                f"delta_v_rtn_km_s=[{dv[0]}, {dv[1]}, {dv[2]}] "
                f"magnitude_km_s={burn['magnitude_km_s']}"
            )
    lines.extend(
        [
            "",
            "Unresolved (ranked by shortfall_km descending)",
            "----------------------------------------------",
        ]
    )
    unresolved = artifact["unresolved"]
    if not unresolved:
        lines.append("(none)")
    else:
        for row in unresolved:
            lines.append(
                f"- conjunction_id={row['conjunction_id']} "
                f"shortfall_km={row['shortfall_km']} "
                f"miss_before_km={row['miss_distance_before_km']} "
                f"miss_after_km={row['miss_distance_after_km']} "
                f"probability_before={row['probability_before']} "
                f"probability_after={row['probability_after']}"
            )
    lines.append("")
    return "\n".join(lines)


def write_plan_text(result: PipelineResult, path: str | Path) -> Path:
    """Write a human-readable plan report."""
    dest = Path(path)
    dest.write_text(_format_plan_text(result), encoding="utf-8")
    return dest


def write_plan_oems(result: PipelineResult, directory: str | Path) -> list[Path]:
    """Write a CCSDS OEM for each satellite in the plan that has burns.

    Propagates with :class:`~aegis.propagation.Sgp4Propagator` after
    :func:`~aegis.maneuver.apply_along_track_burns` when burns exist,
    sampling ~11 states across ``config.duration_s`` (or 5400 s).
    Satellites with an empty burn list are skipped.
    """
    dest_dir = Path(directory)
    dest_dir.mkdir(parents=True, exist_ok=True)

    burning = {
        sat_id: sat_set
        for sat_id, sat_set in result.plan.satellite_sets.items()
        if sat_set.burn_count > 0
    }
    if not burning:
        return []

    objects = apply_along_track_burns(
        list(result.catalog.objects), result.plan, result.assessed
    )
    by_id = {obj.object_id: obj for obj in objects}

    duration_s = result.config.duration_s
    if duration_s <= 0.0:
        duration_s = _DEFAULT_OEM_DURATION_S
    step_s = duration_s / (_OEM_SAMPLES - 1)
    start = result.catalog.screening_start()

    written: list[Path] = []
    for sat_id in sorted(burning):
        obj = by_id.get(sat_id)
        if obj is None:
            raise CcsdsError(f"no catalog object for satellite {sat_id}")
        try:
            propagator = Sgp4Propagator([obj])
            grid = propagator.propagate_grid(start, duration_s, step_s)
        except PropagationError as error:
            raise CcsdsError(f"could not propagate {sat_id} for OEM: {error}") from error
        states = [
            grid.state(0, index)
            for index in range(grid.n_times)
            if bool(grid.valid[0, index])
        ]
        if not states:
            raise CcsdsError(f"propagation produced no valid states for {sat_id}")
        path = dest_dir / f"{sat_id}.oem"
        path.write_text(write_oem(obj, states), encoding="utf-8")
        written.append(path)
    return written
