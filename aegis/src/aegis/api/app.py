"""FastAPI application: health, scene JSON, and the static console."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aegis import __version__
from aegis.constants import CONSOLE_STEP_S, SCREENING_HORIZON_S
from aegis.ingest.socrates import SocratesError, fetch_starlink_alerts
from aegis.ingest.spacetrack import (
    SpaceTrackClient,
    SpaceTrackError,
    credentials_from_env,
)
from aegis.pipeline.crosscheck import CrossCheckReport, run_crosscheck

from .scene import build_scene

__all__ = ["app", "UI_DIR"]

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

_LOCAL_ORIGIN = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app = FastAPI(title="AEGIS", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=_LOCAL_ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_live(value: bool | int | str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise HTTPException(status_code=400, detail="live must be 1/0/true/false")
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(status_code=400, detail="live must be 1/0/true/false")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "version": __version__}


@app.get("/api/starlink-alerts")
def starlink_alerts(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    min_max_probability: float = Query(0.0, ge=0.0, le=1.0),
    max_miss_km: float = Query(5.0, ge=0.0),
) -> dict[str, object]:
    """All free SOCRATES candidates involving Starlink, ranked and paginated."""
    try:
        catalog = fetch_starlink_alerts()
    except SocratesError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    matching = [
        event
        for event in catalog.events
        if event.max_probability >= min_max_probability
        and event.miss_distance_km <= max_miss_km
    ]
    matching.sort(key=lambda event: (-event.max_probability, event.tca, event.event_id))
    page = matching[offset : offset + limit]
    horizon_start = catalog.horizon_start
    horizon_end = catalog.horizon_end
    return {
        "source": "CELESTRAK_SOCRATES",
        "source_url": catalog.source_url,
        "fetched_at": catalog.fetched_at.isoformat(),
        "stale": catalog.stale,
        "probability_kind": "SOCRATES_MAXIMUM_PROBABILITY",
        "total_feed_events": catalog.total_feed_events,
        "total_starlink_events": len(catalog.events),
        "filtered_events": len(matching),
        "unique_starlink_objects": len(catalog.unique_starlink_ids),
        "horizon_start": horizon_start.isoformat() if horizon_start else None,
        "horizon_end": horizon_end.isoformat() if horizon_end else None,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < len(matching),
        "events": [event.as_dict() for event in page],
        "honesty": [
            "Complete SOCRATES rows involving Starlink, not the console object slider sample.",
            "Maximum probability is a conservative SOCRATES metric, not AEGIS Alfano Pc.",
            "Candidates are public-data screening alerts, not flight-ready maneuver decisions.",
        ],
    }


_PUBLIC_CDM_HONESTY = [
    "18 SDS values are the latest public CDM per event: special-perturbations orbits with real covariance.",
    "AEGIS values are recomputed from Space-Track GP (TLE) elements with AEGIS's synthetic TLE covariance.",
    "Public CDMs cover only emergency-reportable events, not a random sample of conjunctions.",
]

# One report per downloaded cdm_public snapshot; the cross-check is ~1 s of
# screening, so recompute only when the hourly refresh brings a new feed.
_crosscheck_memo: dict[str, CrossCheckReport] = {}


def _crosscheck_report() -> CrossCheckReport:
    client = SpaceTrackClient()
    _events, fetched_at = client.fetch_public_conjunctions_with_time()
    key = fetched_at.isoformat()
    report = _crosscheck_memo.get(key)
    if report is None:
        report = run_crosscheck(client)
        _crosscheck_memo.clear()
        _crosscheck_memo[key] = report
    return report


@app.get("/api/public-cdms")
def public_cdms() -> dict[str, object]:
    """Space-Track public CDMs with AEGIS's independent result for each pair."""
    if credentials_from_env() is None:
        return {
            "configured": False,
            "summary": None,
            "events": [],
            "honesty": ["Space-Track is not configured: set SPACETRACK_USER and SPACETRACK_PASS."],
        }
    try:
        report = _crosscheck_report()
    except SpaceTrackError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "configured": True,
        "source": "SPACETRACK_CDM_PUBLIC",
        "summary": report.summary(),
        "events": [check.as_dict() for check in report.checks],
        "honesty": _PUBLIC_CDM_HONESTY,
    }


# The longest standard screening period, and the coarsest step validated
# against dense propagation; both bound the work one request can ask for.
_MAX_SCENE_DURATION_S = 7 * 86400.0
_MAX_SCENE_STEP_S = 120.0


@app.get("/api/scene")
def scene(
    max_objects: int = Query(40),
    duration_s: float = Query(SCREENING_HORIZON_S, gt=0.0, le=_MAX_SCENE_DURATION_S),
    step_s: float = Query(CONSOLE_STEP_S, ge=1.0, le=_MAX_SCENE_STEP_S),
    live: str = Query("true"),
) -> dict:
    if max_objects < 1 or max_objects > 200:
        raise HTTPException(
            status_code=400,
            detail="max_objects must be between 1 and 200 inclusive",
        )
    return build_scene(
        max_objects=max_objects,
        duration_s=float(duration_s),
        step_s=float(step_s),
        live=_parse_live(live),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html", media_type="text/html")


app.mount("/css", StaticFiles(directory=UI_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=UI_DIR / "js"), name="js")
app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")
