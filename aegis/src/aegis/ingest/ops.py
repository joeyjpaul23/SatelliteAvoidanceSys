"""Starlink fleet plus overlapping debris for operational screening.

Live source order: Space-Track when ``SPACETRACK_USER`` / ``SPACETRACK_PASS``
are set, else CelesTrak, else the committed CelesTrak slices. Both halves of
one catalog always come from the same source. This module never calls the
synthetic generator.
"""

from __future__ import annotations

from pathlib import Path

from ..constants import (
    CELESTRAK_DEBRIS_NAME,
    CELESTRAK_FLEET_GROUP,
    PERIGEE_APOGEE_PAD_KM,
)
from ..core.objects import ObjectType, Operator, SpaceObject
from .celestrak import CelesTrakError, catalog_from_tle_file
from .sources import Catalog

__all__ = [
    "load_starlink_slice",
    "load_debris_slice",
    "load_ops_catalog",
]

_AEGIS_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STARLINK_SLICE = _AEGIS_ROOT / "tests" / "fixtures" / "starlink_slice.tle"
_DEFAULT_DEBRIS_SLICE = _AEGIS_ROOT / "tests" / "fixtures" / "debris_slice.tle"

_SPACEX = Operator(
    identifier="SPACEX",
    name="SpaceX",
    organization="Space Exploration Technologies",
    maneuverable=True,
)

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    CelesTrakError,
    OSError,
    TimeoutError,
    ConnectionError,
)


def _cap(catalog: Catalog, max_objects: int | None) -> Catalog:
    if max_objects is None:
        return catalog
    if max_objects < 0:
        raise ValueError("max_objects must be non-negative")
    if max_objects >= len(catalog.objects):
        return catalog
    return Catalog(
        source=catalog.source,
        objects=list(catalog.objects[:max_objects]),
        fetched_at=catalog.fetched_at,
        query=catalog.query,
    )


def load_starlink_slice(
    path: str | Path | None = None,
    *,
    max_objects: int | None = None,
) -> Catalog:
    """Load the committed offline Starlink TLE slice as a CelesTrak catalog."""
    file_path = Path(path) if path is not None else _DEFAULT_STARLINK_SLICE
    return _cap(catalog_from_tle_file(file_path), max_objects)


def load_debris_slice(
    path: str | Path | None = None,
    *,
    max_objects: int | None = None,
) -> Catalog:
    """Load the committed offline LEO debris TLE slice as a CelesTrak catalog."""
    file_path = Path(path) if path is not None else _DEFAULT_DEBRIS_SLICE
    catalog = catalog_from_tle_file(file_path)
    for obj in catalog.objects:
        _tag_debris(obj)
    return _cap(catalog, max_objects)


def _altitude_band_km(obj: SpaceObject) -> tuple[float, float] | None:
    elements = obj.elements
    if elements is None:
        return None
    lo = float(elements.perigee_altitude_km)
    hi = float(elements.apogee_altitude_km)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _fleet_altitude_window_km(fleet: Catalog) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for obj in fleet.objects:
        band = _altitude_band_km(obj)
        if band is None:
            continue
        lows.append(band[0])
        highs.append(band[1])
    if not lows:
        return 400.0, 700.0
    return min(lows), max(highs)


def overlapping_debris(
    debris: Catalog,
    fleet: Catalog,
    *,
    pad_km: float = PERIGEE_APOGEE_PAD_KM,
) -> Catalog:
    """Keep debris whose radial band can meet the fleet after prefilter pad."""
    lo, hi = _fleet_altitude_window_km(fleet)
    kept: list[SpaceObject] = []
    for obj in debris.objects:
        band = _altitude_band_km(obj)
        if band is None:
            kept.append(obj)
            continue
        if (band[0] - pad_km) <= hi and lo <= (band[1] + pad_km):
            kept.append(obj)
    return Catalog(
        source=debris.source,
        objects=kept,
        fetched_at=debris.fetched_at,
        query=debris.query,
    )


def _tag_fleet(obj: SpaceObject) -> SpaceObject:
    obj.object_type = ObjectType.PAYLOAD
    obj.operator = _SPACEX
    obj.metadata["catalog_role"] = "fleet"
    return obj


def _tag_debris(obj: SpaceObject) -> SpaceObject:
    name = (obj.name or "").upper()
    if obj.object_type in (ObjectType.DEBRIS, ObjectType.ROCKET_BODY):
        pass  # typed by the source (Space-Track OMM); keep it
    elif "R/B" in name:
        obj.object_type = ObjectType.ROCKET_BODY
    else:
        obj.object_type = ObjectType.DEBRIS
    obj.operator = None
    obj.metadata["catalog_role"] = "debris"
    return obj


def _split_budget(max_objects: int) -> tuple[int, int]:
    """Return ``(n_fleet, n_debris)`` summing to ``max_objects``."""
    if max_objects < 2:
        return max(0, max_objects), 0
    n_debris = max(1, max_objects // 2)
    n_fleet = max_objects - n_debris
    if n_fleet < 1:
        n_fleet = 1
        n_debris = max_objects - 1
    return n_fleet, n_debris


def mix_fleet_and_debris(
    fleet: Catalog,
    debris: Catalog,
    *,
    max_objects: int,
) -> Catalog:
    """Cap a Starlink catalog and an overlapping debris catalog into one."""
    if max_objects < 1:
        raise ValueError("max_objects must be at least 1")
    n_fleet, n_debris = _split_budget(max_objects)
    fleet_ids = {obj.object_id for obj in fleet.objects[:n_fleet]}
    debris_unique = [obj for obj in debris.objects if obj.object_id not in fleet_ids]
    if len(debris_unique) < n_debris:
        n_fleet = min(len(fleet.objects), max_objects - len(debris_unique))
        n_debris = min(len(debris_unique), max_objects - n_fleet)
    fleet_part = [_tag_fleet(obj) for obj in fleet.objects[:n_fleet]]
    debris_part = [_tag_debris(obj) for obj in debris_unique[:n_debris]]
    objects = fleet_part + debris_part
    query = f"{fleet.source.lower()} starlink+debris"
    if fleet.query or debris.query:
        query = f"{fleet.query} | {debris.query}".strip(" |")
    return Catalog(
        source=fleet.source,
        objects=objects,
        fetched_at=max(fleet.fetched_at, debris.fetched_at),
        query=query,
    )


def _fetch_fleet(*, live: bool) -> tuple[Catalog, bool]:
    from . import celestrak

    if not live:
        return load_starlink_slice(), True
    try:
        return celestrak.fetch_celestrak(CELESTRAK_FLEET_GROUP, fmt="json"), False
    except _NETWORK_ERRORS:
        return load_starlink_slice(), True


def _fetch_debris(*, live: bool) -> tuple[Catalog, bool]:
    from . import celestrak

    if not live:
        return load_debris_slice(), True
    try:
        return celestrak.fetch_celestrak(
            CELESTRAK_DEBRIS_NAME, fmt="json", field="NAME"
        ), False
    except _NETWORK_ERRORS:
        return load_debris_slice(), True


def _fetch_spacetrack() -> tuple[Catalog, Catalog] | None:
    """Both halves from Space-Track, or ``None`` if unconfigured or failing."""
    from . import spacetrack

    if spacetrack.credentials_from_env() is None:
        return None
    client = spacetrack.SpaceTrackClient()
    try:
        return client.fetch_fleet(), client.fetch_debris()
    except (spacetrack.SpaceTrackError, *_NETWORK_ERRORS):
        return None


def load_ops_catalog(*, live: bool = True, max_objects: int = 40) -> tuple[Catalog, str | None]:
    """Starlink + overlapping debris: Space-Track, then CelesTrak, then slices.

    Never generates synthetic objects. ``fallback`` is ``\"slice\"`` when
    either half came from a committed fixture. A Space-Track failure falls
    back to CelesTrak for both halves, which ``catalog.source`` reports.
    """
    if live:
        pair = _fetch_spacetrack()
        if pair is not None:
            fleet, debris = pair
            debris = overlapping_debris(debris, fleet)
            return mix_fleet_and_debris(fleet, debris, max_objects=max_objects), None
    fleet, fleet_slice = _fetch_fleet(live=live)
    debris, debris_slice = _fetch_debris(live=live)
    debris = overlapping_debris(debris, fleet)
    catalog = mix_fleet_and_debris(fleet, debris, max_objects=max_objects)
    fallback = "slice" if fleet_slice or debris_slice else None
    return catalog, fallback
