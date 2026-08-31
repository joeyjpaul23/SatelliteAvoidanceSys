"""Synthetic catalog contract tests. In-memory only; no CelesTrak cache or HTTP."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from aegis.core.objects import ObjectType
from aegis.ingest import (
    DataSource,
    SyntheticAuthorization,
    SyntheticSpec,
    generate_synthetic,
)
from aegis.propagation.propagator import Sgp4Propagator

_DEFAULT_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)


def _auth() -> SyntheticAuthorization:
    return SyntheticAuthorization(acknowledge_synthetic=True)


def _generate(monkeypatch: pytest.MonkeyPatch, spec: SyntheticSpec | None = None):
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    return generate_synthetic(_auth(), spec if spec is not None else SyntheticSpec())


def test_synthetic_spec_defaults() -> None:
    spec = SyntheticSpec()
    assert spec.n_planes == 1
    assert spec.sats_per_plane == 3
    assert spec.altitude_km == 550.0
    assert spec.inclination_deg == 53.0
    assert spec.epoch == _DEFAULT_EPOCH
    assert spec.seed == 0
    assert spec.operator_id == "SYNTHETIC-OP"
    assert spec.operator_name == "Synthetic Operator"
    assert spec.include_known_conjunction_triple is True


def test_object_count_is_planes_times_sats(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SyntheticSpec(n_planes=2, sats_per_plane=4, include_known_conjunction_triple=True)
    catalog = _generate(monkeypatch, spec)
    assert len(catalog) == 2 * 4
    assert len(catalog.objects) == 8


def test_object_ids_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SyntheticSpec(n_planes=3, sats_per_plane=5)
    catalog = _generate(monkeypatch, spec)
    ids = [obj.object_id for obj in catalog]
    assert len(ids) == len(set(ids))
    assert len(ids) == 15


def test_object_ids_numeric_looking(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _generate(monkeypatch)
    for obj in catalog:
        assert obj.object_id, "object_id must be non-empty"
        assert obj.object_id.isdigit()


def test_operator_maneuverable(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SyntheticSpec(operator_id="OP-A", operator_name="Operator A")
    catalog = _generate(monkeypatch, spec)
    for obj in catalog:
        assert obj.operator is not None
        assert obj.operator.maneuverable is True
        assert obj.operator.identifier == spec.operator_id
        assert obj.operator.name == spec.operator_name


def test_deterministic_with_same_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SyntheticSpec(n_planes=2, sats_per_plane=3, seed=42)
    first = _generate(monkeypatch, spec)
    second = _generate(monkeypatch, spec)
    assert [obj.object_id for obj in first] == [obj.object_id for obj in second]
    for a, b in zip(first, second, strict=True):
        assert a.elements is not None and b.elements is not None
        assert a.elements.mean_motion_rev_per_day == b.elements.mean_motion_rev_per_day
        assert a.elements.eccentricity == b.elements.eccentricity
        assert a.elements.inclination_deg == b.elements.inclination_deg
        assert a.elements.raan_deg == b.elements.raan_deg
        assert a.elements.arg_perigee_deg == b.elements.arg_perigee_deg
        assert a.elements.mean_anomaly_deg == b.elements.mean_anomaly_deg
        assert a.elements.bstar == b.elements.bstar
        assert a.elements.epoch == b.elements.epoch


def test_different_seeds_produce_different_phasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = dict(n_planes=1, sats_per_plane=3)
    a = _generate(monkeypatch, SyntheticSpec(seed=1, **base))
    b = _generate(monkeypatch, SyntheticSpec(seed=2, **base))
    ids_a = [obj.object_id for obj in a]
    ids_b = [obj.object_id for obj in b]
    phase_a = [
        (obj.elements.raan_deg, obj.elements.mean_anomaly_deg) for obj in a
    ]
    phase_b = [
        (obj.elements.raan_deg, obj.elements.mean_anomaly_deg) for obj in b
    ]
    assert ids_a != ids_b or phase_a != phase_b


def test_sgp4_can_propagate_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SyntheticSpec(n_planes=1, sats_per_plane=3, seed=0)
    catalog = _generate(monkeypatch, spec)
    propagator = Sgp4Propagator(list(catalog))
    start = catalog.objects[0].elements.epoch
    grid = propagator.propagate_grid(start, duration_s=3600.0, step_s=120.0)
    assert grid.n_objects == len(catalog)
    assert bool(grid.valid.any())


def test_include_known_conjunction_triple_first_objects_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = SyntheticSpec(
        n_planes=1,
        sats_per_plane=3,
        include_known_conjunction_triple=True,
    )
    catalog = _generate(monkeypatch, spec)
    assert len(catalog) >= 3
    assert catalog.objects[0] is not None
    assert catalog.objects[1] is not None
    assert catalog.objects[0].object_id != catalog.objects[1].object_id
    assert catalog.objects[0].elements is not None
    assert catalog.objects[1].elements is not None


def test_conjunction_triple_does_not_change_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    on = _generate(
        monkeypatch,
        SyntheticSpec(n_planes=2, sats_per_plane=4, include_known_conjunction_triple=True),
    )
    off = _generate(
        monkeypatch,
        SyntheticSpec(n_planes=2, sats_per_plane=4, include_known_conjunction_triple=False),
    )
    assert len(on) == 8
    assert len(off) == 8


def test_every_object_data_source_is_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _generate(monkeypatch, SyntheticSpec(n_planes=2, sats_per_plane=3))
    assert catalog.source == DataSource.SYNTHETIC
    for obj in catalog:
        assert obj.data_source == DataSource.SYNTHETIC


def test_object_type_is_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _generate(monkeypatch)
    for obj in catalog:
        assert obj.object_type == ObjectType.PAYLOAD


def test_elements_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _generate(monkeypatch)
    for obj in catalog:
        assert obj.elements is not None
        assert obj.elements.epoch.tzinfo is not None
        assert obj.elements.mean_motion_rev_per_day > 0
        assert 0.0 <= obj.elements.eccentricity < 1.0


def test_catalog_source_and_fetched_at(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _generate(monkeypatch)
    assert catalog.source == DataSource.SYNTHETIC
    assert catalog.fetched_at.tzinfo is not None
    assert catalog.fetched_at.utcoffset() == timezone.utc.utcoffset(catalog.fetched_at)
    assert catalog.query


def test_generate_synthetic_creates_no_celestrak_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    generate_synthetic(_auth(), SyntheticSpec())

    named = [p for p in tmp_path.rglob("*") if p.name == "celestrak"]
    assert named == []
    assert not (tmp_path / "celestrak").exists()


def test_generate_synthetic_creates_no_cache_files_in_tmp_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    generate_synthetic(_auth(), SyntheticSpec())
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def test_generate_synthetic_does_not_open_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("synthetic path must not open HTTP connections")

    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr("requests.get", _boom, raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _boom, raising=False)
    catalog = generate_synthetic(_auth(), SyntheticSpec())
    assert catalog.source == DataSource.SYNTHETIC
