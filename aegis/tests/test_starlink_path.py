"""Offline CelesTrak TLE-file path (Step 7).

Writes against the public ``aegis.ingest`` and ``aegis.pipeline`` surfaces.
Never hits celestrak.org. Does not import pipeline submodules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aegis.core.state import CovarianceSource
from aegis.ingest import (
    CelesTrakError,
    DataSource,
    SyntheticAuthorization,
    catalog_from_tle_file,
)
from aegis.pipeline import PipelineConfig, PipelineError, run_pipeline
from aegis.propagation.propagator import Sgp4Propagator

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "two_leo.tle"


def _fast_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(duration_s=600.0, step_s=60.0, max_iterations=1, **kwargs)


def _block_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("TLE path must not open HTTP")

    monkeypatch.setattr("requests.get", _boom, raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _boom, raising=False)


def test_catalog_from_tle_file_source_and_sgp4() -> None:
    catalog = catalog_from_tle_file(_FIXTURE)

    assert catalog.source == DataSource.CELESTRAK
    assert len(catalog) == 2
    assert all(obj.data_source == DataSource.CELESTRAK for obj in catalog)
    assert _FIXTURE.name in catalog.query

    propagator = Sgp4Propagator(list(catalog))
    start = min(obj.elements.epoch for obj in catalog)
    grid = propagator.propagate_grid(start, duration_s=600.0, step_s=60.0)
    assert grid.n_objects == 2
    assert bool(grid.valid.any())


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "not a two-line element set\n",
    ],
    ids=["empty", "invalid"],
)
def test_empty_or_invalid_tle_file_raises(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "bad.tle"
    path.write_text(contents)
    with pytest.raises(CelesTrakError):
        catalog_from_tle_file(path)


def test_run_pipeline_tle_path_celestrak_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_http(monkeypatch)
    result = run_pipeline(
        tle_path=_FIXTURE,
        source=DataSource.CELESTRAK,
        config=_fast_config(),
    )
    assert result.source == DataSource.CELESTRAK
    assert result.catalog.source == DataSource.CELESTRAK
    assert result.covariance_source == CovarianceSource.SYNTHETIC_TLE
    assert result.covariance_source != CovarianceSource.CALCULATED


def test_tle_path_with_synthetic_source_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    with pytest.raises(PipelineError):
        run_pipeline(
            tle_path=_FIXTURE,
            source=DataSource.SYNTHETIC,
            authorization=SyntheticAuthorization(acknowledge_synthetic=True),
            config=_fast_config(),
        )


def test_max_objects_caps_two_object_tle() -> None:
    result = run_pipeline(
        tle_path=_FIXTURE,
        source=DataSource.CELESTRAK,
        config=_fast_config(max_objects=1),
    )
    assert len(result.catalog) == 1
    assert result.source == DataSource.CELESTRAK
    assert result.catalog.source == DataSource.CELESTRAK
    assert result.catalog.objects[0].data_source == DataSource.CELESTRAK


def test_generate_synthetic_not_used_on_tle_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_synth(*_a, **_k):
        raise AssertionError("TLE path must not call generate_synthetic")

    monkeypatch.setattr("aegis.ingest.generate_synthetic", _no_synth)
    result = run_pipeline(
        tle_path=_FIXTURE,
        source=DataSource.CELESTRAK,
        config=_fast_config(),
    )
    assert result.source != DataSource.SYNTHETIC
    assert result.catalog.source != DataSource.SYNTHETIC
    assert all(obj.data_source != DataSource.SYNTHETIC for obj in result.catalog)
    assert all(obj.data_source == DataSource.CELESTRAK for obj in result.catalog)
