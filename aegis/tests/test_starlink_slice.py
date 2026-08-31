"""Cached Starlink slice through the pipeline (Step 9).

Writes against the public ``aegis.pipeline`` / ``aegis.ingest`` surfaces.
Never hits celestrak.org. Does not import pipeline submodules.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from aegis.core.maneuver import ManeuverPlan
from aegis.core.state import CovarianceSource
from aegis.ingest import DataSource, catalog_from_tle_file
from aegis.pipeline import PipelineConfig, run_pipeline
from aegis.propagation.propagator import Sgp4Propagator

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "starlink_slice.tle"


def _load_starlink_slice(*args, **kwargs):
    """Resolve the Step 9 helper from pipeline or ingest."""
    import aegis.pipeline as pipeline_pkg

    loader = getattr(pipeline_pkg, "load_starlink_slice", None)
    if callable(loader):
        return loader(*args, **kwargs)

    from aegis import ingest as ingest_pkg

    loader = getattr(ingest_pkg, "load_starlink_slice", None)
    if callable(loader):
        return loader(*args, **kwargs)

    raise AssertionError(
        "load_starlink_slice must be exported from aegis.pipeline or aegis.ingest"
    )


def _name_lines(text: str) -> list[str]:
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("1 ") or line.startswith("2 "):
            continue
        names.append(line)
    return names


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("Starlink slice tests must not open the network")

    monkeypatch.setattr("requests.get", _boom, raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _boom, raising=False)
    monkeypatch.setattr(socket.socket, "connect", _boom)
    monkeypatch.setattr(socket.socket, "connect_ex", _boom)


def test_starlink_slice_fixture_exists() -> None:
    assert _FIXTURE.is_file(), f"missing fixture: {_FIXTURE}"


def test_fixture_has_at_least_20_starlink_name_lines() -> None:
    assert _FIXTURE.is_file(), f"missing fixture: {_FIXTURE}"
    names = _name_lines(_FIXTURE.read_text(encoding="utf-8"))
    starlink_names = [name for name in names if "STARLINK" in name.upper()]
    assert len(starlink_names) >= 20
    assert all("STARLINK" in name.upper() for name in names)
    assert 20 <= len(names) <= 80


def test_load_starlink_slice_and_catalog_from_tle_file_are_celestrak() -> None:
    via_file = catalog_from_tle_file(_FIXTURE)
    via_helper = _load_starlink_slice()

    assert via_file.source == DataSource.CELESTRAK
    assert via_helper.source == DataSource.CELESTRAK
    assert len(via_file) >= 20
    assert len(via_helper) >= 20
    assert 20 <= len(via_file) <= 80
    assert all(obj.data_source == DataSource.CELESTRAK for obj in via_file)
    assert all(obj.data_source == DataSource.CELESTRAK for obj in via_helper)


def test_sgp4_propagator_initialises_starlink_slice() -> None:
    catalog = catalog_from_tle_file(_FIXTURE)
    propagator = Sgp4Propagator(list(catalog))
    start = min(obj.elements.epoch for obj in catalog)
    grid = propagator.propagate_grid(start, duration_s=60.0, step_s=60.0)
    assert grid.n_objects == len(catalog)
    assert bool(grid.valid.any())


def test_run_pipeline_starlink_slice_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_network(monkeypatch)

    def _no_synth(*_a, **_k):
        raise AssertionError("Starlink slice must not call generate_synthetic")

    monkeypatch.setattr("aegis.ingest.generate_synthetic", _no_synth)

    result = run_pipeline(
        tle_path=_FIXTURE,
        source=DataSource.CELESTRAK,
        config=PipelineConfig(duration_s=2700, max_objects=20, step_s=60),
    )

    assert result.source == DataSource.CELESTRAK
    assert result.catalog.source == DataSource.CELESTRAK
    assert result.covariance_source == CovarianceSource.SYNTHETIC_TLE
    assert isinstance(result.plan, ManeuverPlan)
    assert len(result.catalog) <= 20
    assert all(obj.data_source == DataSource.CELESTRAK for obj in result.catalog)
    assert all(obj.data_source != DataSource.SYNTHETIC for obj in result.catalog)


def test_load_starlink_slice_max_objects_five() -> None:
    catalog = _load_starlink_slice(max_objects=5)
    assert len(catalog) == 5
    assert catalog.source == DataSource.CELESTRAK
    assert all(obj.data_source == DataSource.CELESTRAK for obj in catalog)


def test_pipeline_result_objects_are_not_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_network(monkeypatch)
    result = run_pipeline(
        tle_path=_FIXTURE,
        source=DataSource.CELESTRAK,
        config=PipelineConfig(duration_s=2700, max_objects=20, step_s=60),
    )
    assert result.source != DataSource.SYNTHETIC
    assert result.catalog.source != DataSource.SYNTHETIC
    assert all(obj.data_source != DataSource.SYNTHETIC for obj in result.catalog)
    loaded = _load_starlink_slice()
    assert loaded.source != DataSource.SYNTHETIC
    assert all(obj.data_source != DataSource.SYNTHETIC for obj in loaded)
