"""End-to-end ingest → screen → assess → plan orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.state import CovarianceSource
from ..ingest.ops import load_debris_slice, load_ops_catalog, load_starlink_slice
from ..ingest.sources import (
    Catalog,
    DataSource,
    MixedDataSourceError,
    SyntheticNotAuthorizedError,
)
from ..maneuver import plan_maneuvers, rescreen_until_stable
from ..risk import assess_catalog
from ..screening import screen
from .config import PipelineConfig, PipelineResult
from .errors import PipelineError

if TYPE_CHECKING:
    from ..ingest.synthetic import SyntheticAuthorization, SyntheticSpec

__all__ = [
    "run_pipeline",
    "load_starlink_slice",
    "load_debris_slice",
    "load_ops_catalog",
]

_SYNTHETIC_REFUSED = (
    "synthetic data was refused: AEGIS_ALLOW_SYNTHETIC=1 "
    "plus an explicit authorization are required"
)


def _normalize_source(source: str) -> str:
    if source not in DataSource.ALL:
        raise PipelineError(
            f"unsupported pipeline source: {source!r} "
            f"(allowed: {', '.join(DataSource.ALL)})"
        )
    return source


def _cap_catalog(catalog: Catalog, max_objects: int | None) -> Catalog:
    if max_objects is None:
        return catalog
    if max_objects < 0:
        raise PipelineError("max_objects must be non-negative")
    if max_objects >= len(catalog.objects):
        return catalog
    return Catalog(
        source=catalog.source,
        objects=list(catalog.objects[:max_objects]),
        fetched_at=catalog.fetched_at,
        query=catalog.query,
    )


def _screening_start(catalog: Catalog) -> datetime:
    """Earliest ``elements.epoch`` if any object has elements, else ``fetched_at``.

    Planning ``now`` uses this same epoch. Wall-clock is never the screening
    start when a catalog epoch is available.
    """
    epochs = [
        obj.elements.epoch
        for obj in catalog.objects
        if obj.elements is not None
    ]
    if epochs:
        return min(epochs)
    return catalog.fetched_at


def _ingest_celestrak(group: str, session, cache_dir) -> Catalog:
    from ..ingest.celestrak import fetch_celestrak

    try:
        return fetch_celestrak(group, session=session, cache_dir=cache_dir)
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(f"CelesTrak ingest failed: {error}") from error


def _ingest_spacetrack(group: str, session, cache_dir) -> Catalog:
    from ..ingest.spacetrack import SpaceTrackClient

    try:
        client = SpaceTrackClient(session=session, cache_dir=cache_dir)
        return client.fetch_fleet(group)
    except Exception as error:
        raise PipelineError(f"Space-Track ingest failed: {error}") from error


def _ingest_tle_file(path: str | Path) -> Catalog:
    from ..ingest.celestrak import catalog_from_tle_file

    try:
        return catalog_from_tle_file(path)
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(f"CelesTrak ingest failed: {error}") from error


def _ingest_synthetic(
    authorization: SyntheticAuthorization | None,
    synthetic_spec: SyntheticSpec | None,
) -> Catalog:
    from ..ingest.synthetic import SyntheticSpec, generate_synthetic

    if authorization is None:
        raise SyntheticNotAuthorizedError(_SYNTHETIC_REFUSED)
    spec = synthetic_spec if synthetic_spec is not None else SyntheticSpec()
    return generate_synthetic(authorization, spec)


def run_pipeline(
    source: str = DataSource.CELESTRAK,
    *,
    group: str = "starlink",
    catalog: Catalog | None = None,
    authorization: SyntheticAuthorization | None = None,
    synthetic_spec: SyntheticSpec | None = None,
    session=None,
    cache_dir=None,
    tle_path: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Ingest a catalog, screen it, assess risk, and plan maneuvers.

    Default ``source`` is CelesTrak. That path never imports or calls the
    synthetic generator, never constructs a synthetic authorization or
    spec, and never consults ``AEGIS_ALLOW_SYNTHETIC``. A CelesTrak
    failure raises :class:`PipelineError` and does not fall back to a
    generated catalog. ``source=SPACETRACK`` is held to the same rules;
    there ``group`` is an ``OBJECT_NAME`` prefix (``starlink`` matches
    ``STARLINK-*``).

    When ``tle_path`` is set, the catalog is read from that local TLE
    file (no HTTP) and ``group`` is ignored. ``tle_path`` is illegal
    with ``source=SYNTHETIC``.
    """
    source = _normalize_source(source)
    if source != DataSource.SYNTHETIC and (
        authorization is not None or synthetic_spec is not None
    ):
        label = "CelesTrak" if source == DataSource.CELESTRAK else "Space-Track"
        raise PipelineError(
            f"authorization and synthetic_spec are illegal on the {label} path"
        )
    if tle_path is not None and source != DataSource.CELESTRAK:
        raise PipelineError("tle_path requires source CELESTRAK")

    config = config if config is not None else PipelineConfig()

    if catalog is not None:
        if catalog.source != source:
            raise MixedDataSourceError(
                f"catalog source {catalog.source!r} does not match "
                f"pipeline source {source!r}"
            )
    elif tle_path is not None:
        catalog = _ingest_tle_file(tle_path)
    elif source == DataSource.CELESTRAK:
        catalog = _ingest_celestrak(group, session, cache_dir)
    elif source == DataSource.SPACETRACK:
        catalog = _ingest_spacetrack(group, session, cache_dir)
    else:
        catalog = _ingest_synthetic(authorization, synthetic_spec)

    catalog = _cap_catalog(catalog, config.max_objects)
    objects = list(catalog.objects)
    start = _screening_start(catalog)

    conjunctions = screen(
        objects,
        start,
        config.duration_s,
        step_s=config.step_s,
        box_km=config.box_km,
    )
    assessed = assess_catalog(conjunctions, objects=objects)

    plan_kwargs = {
        "now": start,
        "target_pc": config.target_pc,
        "dv_budget_km_s": config.dv_budget_km_s,
    }
    if len(objects) == 0:
        plan = plan_maneuvers(assessed, objects, **plan_kwargs)
    else:
        plan = rescreen_until_stable(
            objects,
            start,
            config.duration_s,
            max_iterations=config.max_iterations,
            step_s=config.step_s,
            box_km=config.box_km,
            **plan_kwargs,
        )

    covariance_source = assessed.covariance_source or CovarianceSource.SYNTHETIC_TLE
    return PipelineResult(
        source=source,
        catalog=catalog,
        assessed=assessed,
        plan=plan,
        covariance_source=covariance_source,
        config=config,
    )
