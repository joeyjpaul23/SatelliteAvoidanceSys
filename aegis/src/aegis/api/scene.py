"""Build the console scene payload from the existing pipeline.

CelesTrak when live fetch works; committed Starlink slice otherwise.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from ..constants import LOW_RELATIVE_VELOCITY_KM_S, SCREENING_BOX_STARLINK_KM
from ..core.conjunction import RiskLevel
from ..core.state import CovarianceSource
from ..core.timebase import ensure_utc, seconds_between
from ..ingest.celestrak import CelesTrakError
from ..ingest.sources import Catalog, DataSource
from ..maneuver import plan_maneuvers
from ..propagation import Sgp4Propagator, default_covariance_model
from ..risk import assess_catalog
from ..screening import screen
from .color import display_band

__all__ = ["build_scene"]

_HONESTY_SYNTHETIC_TLE = (
    "SYNTHETIC_TLE covariance is TLE-grade triage, not an operational go/no-go."
)
_HONESTY_INFLATE = "DISPLAY BAND inflates one step — TLE covariance"
_HONESTY_LIVE_FAIL = "live fetch failed — FALLBACK SLICE"

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    CelesTrakError,
    OSError,
    TimeoutError,
    ConnectionError,
)


def _cap_catalog(catalog: Catalog, max_objects: int) -> Catalog:
    if max_objects >= len(catalog.objects):
        return catalog
    return Catalog(
        source=catalog.source,
        objects=list(catalog.objects[:max_objects]),
        fetched_at=catalog.fetched_at,
        query=catalog.query,
    )


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


def _ingest(live: bool) -> tuple[Catalog, str | None]:
    from aegis.ingest import celestrak
    from aegis.pipeline.run import load_starlink_slice

    if not live:
        return load_starlink_slice(), "slice"
    try:
        return celestrak.fetch_celestrak("starlink"), None
    except _NETWORK_ERRORS:
        return load_starlink_slice(), "slice"


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
    duration_s: float = 5400.0,
    step_s: float = 60.0,
    live: bool = True,
) -> dict[str, Any]:
    """Ingest CelesTrak or the slice, run existing physics, return scene JSON."""
    catalog, fallback = _ingest(live)
    catalog = _cap_catalog(catalog, max_objects)
    objects = list(catalog.objects)
    start = _screening_start(catalog)

    conjunctions = screen(objects, start, duration_s, step_s=step_s)
    assessed = assess_catalog(conjunctions, objects=objects)
    plan = plan_maneuvers(assessed, objects, now=start)

    covariance_source = assessed.covariance_source or CovarianceSource.SYNTHETIC_TLE
    cov_model = default_covariance_model()

    if objects:
        grid = Sgp4Propagator(objects).propagate_grid(start, duration_s, step_s)
        times_s = [float(t) for t in np.asarray(grid.times_s, dtype=float)]
        tracks = {
            object_id: _track_rows(grid.positions_km[index])
            for index, object_id in enumerate(grid.object_ids)
        }
    else:
        times_s = []
        tracks = {}

    by_object_bands: dict[str, list[str]] = {obj.object_id: [] for obj in objects}
    by_object_events: dict[str, list[str]] = {obj.object_id: [] for obj in objects}
    inflation_possible = covariance_source == CovarianceSource.SYNTHETIC_TLE
    inflation_applied = False

    conjunction_rows: list[dict[str, Any]] = []
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
        cid = conjunction.conjunction_id
        primary_id = conjunction.primary.object_id
        secondary_id = conjunction.secondary.object_id
        by_object_bands.setdefault(primary_id, []).append(band)
        by_object_bands.setdefault(secondary_id, []).append(band)
        by_object_events.setdefault(primary_id, []).append(cid)
        by_object_events.setdefault(secondary_id, []).append(cid)
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
        conjunction_rows.append(row)

    object_rows: list[dict[str, Any]] = []
    for obj in objects:
        sigmas = cov_model.sigmas_rtn_km(obj, _lead_days(obj, start))
        object_rows.append(
            {
                "id": obj.object_id,
                "name": obj.name or obj.object_id,
                "track": tracks.get(obj.object_id, []),
                "color_band": _worst_band(by_object_bands.get(obj.object_id, [])),
                "sigma_rtn_km": [float(sigmas[0]), float(sigmas[1]), float(sigmas[2])],
                "conjunction_ids": list(by_object_events.get(obj.object_id, [])),
            }
        )

    unresolved = sorted(plan.unresolved, key=lambda item: item.shortfall_km, reverse=True)
    honesty = [_HONESTY_SYNTHETIC_TLE]
    if fallback == "slice":
        honesty.append(_HONESTY_LIVE_FAIL)
    if inflation_possible or inflation_applied:
        honesty.append(_HONESTY_INFLATE)

    payload = {
        "source": DataSource.CELESTRAK,
        "fallback": fallback,
        "covariance_source": covariance_source,
        "fetched_at": _iso(catalog.fetched_at),
        "query": catalog.query,
        "epoch": _iso(start),
        "duration_s": float(duration_s),
        "step_s": float(step_s),
        "max_objects": int(max_objects),
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
