"""FastAPI application: health, scene JSON, and the static console."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aegis import __version__

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


@app.get("/api/scene")
def scene(
    max_objects: int = Query(40),
    duration_s: float = Query(5400),
    step_s: float = Query(60),
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
