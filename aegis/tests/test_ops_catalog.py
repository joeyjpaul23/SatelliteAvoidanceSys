"""Starlink + overlapping debris ops catalog (no network, no pipeline)."""

from __future__ import annotations

from pathlib import Path

from aegis.constants import SCREENING_HORIZON_S
from aegis.core.objects import ObjectType
from aegis.ingest.ops import (
    load_debris_slice,
    load_ops_catalog,
    load_starlink_slice,
    mix_fleet_and_debris,
    overlapping_debris,
)
from aegis.ingest.sources import DataSource

_DEBRIS = Path(__file__).resolve().parent / "fixtures" / "debris_slice.tle"


def test_screening_horizon_is_three_days() -> None:
    assert SCREENING_HORIZON_S == 3 * 86400


def test_debris_slice_exists_and_is_celestrak() -> None:
    assert _DEBRIS.is_file()
    catalog = load_debris_slice()
    assert catalog.source == DataSource.CELESTRAK
    assert len(catalog) >= 10
    assert all("DEB" in (obj.name or "").upper() for obj in catalog)
    assert all(obj.object_type in (ObjectType.DEBRIS, ObjectType.ROCKET_BODY) for obj in catalog)
    assert all(not obj.is_maneuverable for obj in catalog)


def test_ops_catalog_offline_mixes_starlink_and_debris() -> None:
    catalog, fallback = load_ops_catalog(live=False, max_objects=8)
    assert fallback == "slice"
    assert catalog.source == DataSource.CELESTRAK
    assert len(catalog) == 8
    roles = {obj.metadata.get("catalog_role") for obj in catalog}
    assert roles == {"fleet", "debris"}
    fleet = [obj for obj in catalog if obj.metadata.get("catalog_role") == "fleet"]
    debris = [obj for obj in catalog if obj.metadata.get("catalog_role") == "debris"]
    assert fleet and debris
    assert all(obj.is_maneuverable for obj in fleet)
    assert all(not obj.is_maneuverable for obj in debris)


def test_overlapping_debris_keeps_leo_band() -> None:
    fleet = load_starlink_slice(max_objects=8)
    debris = load_debris_slice()
    kept = overlapping_debris(debris, fleet)
    assert len(kept) >= 1
    assert kept.source == DataSource.CELESTRAK


def test_mix_respects_max_objects() -> None:
    fleet = load_starlink_slice()
    debris = load_debris_slice()
    mixed = mix_fleet_and_debris(fleet, debris, max_objects=5)
    assert len(mixed) == 5
    assert mixed.source == DataSource.CELESTRAK


def test_live_ops_uses_six_digit_safe_omm_json(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    def fake_fetch(group: str, **kwargs):
        calls.append({"group": group, **kwargs})
        if group == "starlink":
            return load_starlink_slice()
        return load_debris_slice()

    monkeypatch.setattr("aegis.ingest.celestrak.fetch_celestrak", fake_fetch)
    catalog, fallback = load_ops_catalog(live=True, max_objects=8)

    assert fallback is None
    assert len(catalog) == 8
    assert len(calls) == 2
    assert all(call.get("fmt") == "json" for call in calls)
