"""Build the console scene payload from the existing pipeline.

CelesTrak Starlink plus overlapping debris when live fetch works;
committed slices otherwise. The globe and event list only include
MONITOR / WATCH / ACT — CLEAR tracks are omitted.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from ..constants import (
    CONSOLE_TRACK_STEP_S,
    LOW_RELATIVE_VELOCITY_KM_S,
    SCREENING_BOX_STARLINK_KM,
    SCREENING_HORIZON_S,
)
from ..core.conjunction import RiskLevel
from ..core.state import CovarianceSource
from ..core.timebase import ensure_utc, seconds_between, shift
from ..ingest.ops import load_ops_catalog
from ..ingest.sources import Catalog
from ..maneuver import plan_maneuvers
from ..propagation import Sgp4Propagator, default_covariance_model
from ..risk import assess_catalog
from ..risk.batch import AssessedCatalog
from ..screening import screen
from .color import display_band

__all__ = ["build_scene"]

_HONESTY_SYNTHETIC_TLE = (
    "SYNTHETIC_TLE covariance is TLE-grade triage, not an operational go/no-go."
)
_HONESTY_INFLATE = "DISPLAY BAND inflates one step — TLE covariance"
_HONESTY_LIVE_FAIL = "live fetch failed — FALLBACK SLICE"
_HONESTY_DEBRIS = "Starlink vs catalog debris over a 3-day TLE screen."
_HONESTY_RISK_ONLY = (
    "Only MONITOR / WATCH / ACT are shown. CLEAR objects and events are omitted."
)
_HONESTY_NO_RISK = "No MONITOR+ events in this window."
_TRACK_DURATION_S = 5400.0
_AT_RISK = frozenset({RiskLevel.MONITOR, RiskLevel.WATCH, RiskLevel.ACT})


def _screening_start(catalog: Catalog) -> datetime:
    """Earliest ``elements.epoch`` if any object has elements, else ``fetched_at``."""
    epochs = [
        obj.elements.epoch
        for obj in catalog.objects
        if obj.elements is not None
    ]
    if epochs:
        return min(epochs)
    return catalog.fetched_at


def _iso(moment: datetime) -> str:
    return ensure_utc(moment).isoformat()


def _xyz(vec: Any) -> list[float]:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _ingest(live: bool, max_objects: int):
    return load_ops_catalog(live=live, max_objects=max_objects)


def _involves_fleet(obj_a, obj_b) -> bool:
    """Keep fleet–fleet and fleet–debris; drop debris–debris.

    Untagged catalogs (no ``catalog_role``) still screen every pair so
    pipeline tests keep their existing all-pairs contract.
    """
    roles = {
        (obj_a.metadata or {}).get("catalog_role"),
        (obj_b.metadata or {}).get("catalog_role"),
    }
    if "fleet" in roles:
        return True
    if obj_a.is_maneuverable or obj_b.is_maneuverable:
        return True
    if "debris" in roles:
        return False
    return True


def _pair_key(primary_id: str, secondary_id: str) -> tuple[str, str]:
    return tuple(sorted((primary_id, secondary_id)))


def _keep_closest_per_pair(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _pair_key(row["primary_id"], row["secondary_id"])
        previous = best.get(key)
        if previous is None or float(row["miss_km"]) < float(previous["miss_km"]):
            best[key] = row
    return sorted(best.values(), key=lambda item: (-float(item["pc"]), item["tca"]))


def _lead_days(obj, epoch: datetime) -> float:
    if obj.elements is None:
        return 0.0
    return max(seconds_between(obj.elements.epoch, epoch) / 86400.0, 0.0)


def _flags(assessment, conjunction) -> list[str]:
    flags: list[str] = []
    if assessment.dilution_flag:
        flags.append("DILUTION")
    if assessment.remediated_flag:
        flags.append("REMEDIATED")
    if assessment.short_encounter_valid:
        flags.append("SHORT_ENCOUNTER_VALID")
    else:
        flags.append("SHORT_ENCOUNTER_INVALID")
    low_rel = bool(conjunction.metadata.get("low_relative_velocity")) or (
        conjunction.relative_speed_km_s < LOW_RELATIVE_VELOCITY_KM_S
    )
    if low_rel:
        flags.append("LOW_RELATIVE_VELOCITY")
    return flags


def _worst_band(bands: list[str]) -> str:
    worst = "CLEAR"
    worst_rank = RiskLevel.rank(worst)
    for band in bands:
        if band not in RiskLevel.ORDER:
            continue
        rank = RiskLevel.rank(band)
        if rank > worst_rank:
            worst = band
            worst_rank = rank
    return worst


def _track_rows(positions: np.ndarray) -> list[list[float]]:
    return [
        [float(sample[0]), float(sample[1]), float(sample[2])]
        for sample in np.asarray(positions, dtype=float)
    ]


def _burn_record(maneuver) -> dict[str, Any]:
    return {
        "satellite_id": maneuver.satellite_id,
        "epoch": _iso(maneuver.epoch),
        "delta_v_rtn_km_s": _xyz(maneuver.delta_v_rtn_km_s),
        "magnitude_km_s": float(maneuver.magnitude_km_s),
    }


def _unresolved_record(outcome) -> dict[str, Any]:
    return {
        "conjunction_id": outcome.conjunction_id,
        "shortfall_km": float(outcome.shortfall_km),
        "probability_before": float(outcome.probability_before),
        "probability_after": float(outcome.probability_after),
        "miss_distance_before_km": float(outcome.miss_distance_before_km),
        "miss_distance_after_km": float(outcome.miss_distance_after_km),
        "required_miss_distance_km": float(outcome.required_miss_distance_km),
        "resolved": False,
    }


def build_scene(
    *,
    max_objects: int = 40,
    duration_s: float = SCREENING_HORIZON_S,
    step_s: float = 60.0,
    live: bool = True,
) -> dict[str, Any]:
    """Ingest CelesTrak or the slice, run existing physics, return scene JSON."""
    catalog, fallback = _ingest(live, max_objects)
    objects = list(catalog.objects)
    start = _screening_start(catalog)

    conjunctions = screen(
        objects,
        start,
        duration_s,
        step_s=step_s,
        keep_pair=_involves_fleet,
    )
    assessed = assess_catalog(conjunctions, objects=objects)
    covariance_source = assessed.covariance_source or CovarianceSource.SYNTHETIC_TLE
    cov_model = default_covariance_model()

    inflation_applied = False
    risk_rows: list[dict[str, Any]] = []
    for entry in assessed.entries:
        conjunction = entry.conjunction
        assessment = entry.assessment
        band = display_band(
            assessment.probability,
            covariance_source=covariance_source,
            dilution=assessment.dilution_flag,
            miss_km=assessment.miss_distance_km,
            sigma_major_km=assessment.sigma_major_km,
        )
        if band != assessment.risk_level:
            inflation_applied = True
        if band not in _AT_RISK:
            continue
        cid = conjunction.conjunction_id
        primary_id = conjunction.primary.object_id
        secondary_id = conjunction.secondary.object_id
        low_rel = bool(conjunction.metadata.get("low_relative_velocity")) or (
            conjunction.relative_speed_km_s < LOW_RELATIVE_VELOCITY_KM_S
        )
        row: dict[str, Any] = {
            "id": cid,
            "primary_id": primary_id,
            "secondary_id": secondary_id,
            "tca": _iso(conjunction.tca),
            "miss_km": float(assessment.miss_distance_km),
            "pc": float(assessment.probability),
            "risk_level": assessment.risk_level,
            "display_band": band,
            "relative_speed_km_s": float(conjunction.relative_speed_km_s),
            "mahalanobis": float(assessment.mahalanobis_distance),
            "dilution": bool(assessment.dilution_flag),
            "remediated": bool(assessment.remediated_flag),
            "short_encounter_valid": bool(assessment.short_encounter_valid),
            "low_relative_velocity": low_rel,
            "flags": _flags(assessment, conjunction),
            "sigma_major_km": float(assessment.sigma_major_km),
            "sigma_minor_km": float(assessment.sigma_minor_km),
            "tca_primary_km": _xyz(conjunction.primary_state.position_km),
            "tca_secondary_km": _xyz(conjunction.secondary_state.position_km),
        }
        if assessment.cross_check_probability is not None:
            row["pc_chan"] = float(assessment.cross_check_probability)
        risk_rows.append(row)

    conjunction_rows = _keep_closest_per_pair(risk_rows)
    kept_ids = {row["id"] for row in conjunction_rows}
    risk_ids = {
        object_id
        for row in conjunction_rows
        for object_id in (row["primary_id"], row["secondary_id"])
    }
    display_objects = [obj for obj in objects if obj.object_id in risk_ids]
    risk_entries = [
        entry
        for entry in assessed.entries
        if entry.conjunction.conjunction_id in kept_ids
    ]
    plan = plan_maneuvers(
        AssessedCatalog(
            source=assessed.source,
            entries=risk_entries,
            covariance_source=assessed.covariance_source,
        ),
        display_objects,
        now=start,
    )

    track_start = start
    if conjunction_rows:
        first_tca = min(
            ensure_utc(datetime.fromisoformat(row["tca"])) for row in conjunction_rows
        )
        track_start = shift(first_tca, -_TRACK_DURATION_S / 2.0)
        if track_start < start:
            track_start = start
    track_step = (
        min(float(step_s), float(CONSOLE_TRACK_STEP_S))
        if duration_s > 7200
        else float(step_s)
    )

    if display_objects:
        grid = Sgp4Propagator(display_objects).propagate_grid(
            track_start, _TRACK_DURATION_S, track_step
        )
        times_s = [float(t) for t in np.asarray(grid.times_s, dtype=float)]
        tracks = {
            object_id: _track_rows(grid.positions_km[index])
            for index, object_id in enumerate(grid.object_ids)
        }
    else:
        times_s = []
        tracks = {}

    by_object_bands: dict[str, list[str]] = {obj.object_id: [] for obj in display_objects}
    by_object_events: dict[str, list[str]] = {obj.object_id: [] for obj in display_objects}
    for row in conjunction_rows:
        by_object_bands.setdefault(row["primary_id"], []).append(row["display_band"])
        by_object_bands.setdefault(row["secondary_id"], []).append(row["display_band"])
        by_object_events.setdefault(row["primary_id"], []).append(row["id"])
        by_object_events.setdefault(row["secondary_id"], []).append(row["id"])

    object_rows: list[dict[str, Any]] = []
    for obj in display_objects:
        sigmas = cov_model.sigmas_rtn_km(obj, _lead_days(obj, start))
        object_rows.append(
            {
                "id": obj.object_id,
                "name": obj.name or obj.object_id,
                "object_type": obj.object_type,
                "role": obj.metadata.get("catalog_role", "fleet"),
                "maneuverable": bool(obj.is_maneuverable),
                "track": tracks.get(obj.object_id, []),
                "color_band": _worst_band(by_object_bands.get(obj.object_id, [])),
                "sigma_rtn_km": [float(sigmas[0]), float(sigmas[1]), float(sigmas[2])],
                "conjunction_ids": list(by_object_events.get(obj.object_id, [])),
            }
        )

    unresolved = sorted(plan.unresolved, key=lambda item: item.shortfall_km, reverse=True)
    honesty = [_HONESTY_SYNTHETIC_TLE, _HONESTY_DEBRIS, _HONESTY_RISK_ONLY]
    if fallback == "slice":
        honesty.append(_HONESTY_LIVE_FAIL)
    if inflation_applied:
        honesty.append(_HONESTY_INFLATE)
    if not conjunction_rows:
        honesty.append(_HONESTY_NO_RISK)

    payload = {
        "source": catalog.source,
        "fallback": fallback,
        "covariance_source": covariance_source,
        "fetched_at": _iso(catalog.fetched_at),
        "query": catalog.query,
        "epoch": _iso(track_start),
        "duration_s": float(duration_s),
        "step_s": float(step_s),
        "track_step_s": float(track_step),
        "max_objects": int(max_objects),
        "screened_objects": len(objects),
        "live": bool(live),
        "box_km": [float(v) for v in SCREENING_BOX_STARLINK_KM],
        "times_s": times_s,
        "objects": object_rows,
        "conjunctions": conjunction_rows,
        "plan": {
            "summary": plan.summary(),
            "burns": [_burn_record(burn) for burn in plan.all_maneuvers],
            "unresolved": [_unresolved_record(item) for item in unresolved],
        },
        "honesty": honesty,
    }
    return _json_safe(payload)
