"""End-to-end pipeline contract (Step 5).

Writes against the public ``aegis.pipeline`` surface plus allowed imports
(core types, constants, ingest, screening, risk.batch). Offline only:
CelesTrak runs inject ``catalog=`` or a mocked session. Does not import
``aegis.pipeline`` submodules.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aegis.constants import (
    DEFAULT_DV_BUDGET_KM_S,
    MU_EARTH_KM3_S2,
    PC_TARGET_POST_MANEUVER,
    R_EARTH_KM,
    REV_PER_DAY_TO_RAD_PER_S,
    SCREENING_BOX_STARLINK_KM,
    SCREENING_STEP_S,
)
from aegis.core.maneuver import ManeuverPlan
from aegis.core.objects import ObjectType, OrbitalElements, SpaceObject
from aegis.core.state import CovarianceSource
from aegis.ingest import (
    Catalog,
    DataSource,
    MixedDataSourceError,
    SyntheticAuthorization,
    SyntheticNotAuthorizedError,
    SyntheticSpec,
)
from aegis.pipeline import (
    PipelineConfig,
    PipelineError,
    PipelineResult,
    run_pipeline,
)
import aegis.pipeline as pipeline_pkg
from aegis.risk import AssessedCatalog

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"

_REQUIRED_EXPORTS = {
    "PipelineConfig",
    "PipelineResult",
    "PipelineError",
    "run_pipeline",
}

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)

_GATE_ERRORS = (SyntheticNotAuthorizedError, PipelineError)
_MISMATCH_ERRORS = (MixedDataSourceError, PipelineError)


class _FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.ok = status_code < 400
        self.content = text.encode("utf-8")
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def get(self, url, timeout=None, headers=None, **kwargs):  # noqa: ANN001
        self.calls.append(
            {"url": url, "timeout": timeout, "headers": headers or {}, **kwargs}
        )
        return self._response


class _RaisingSession:
    def get(self, url, timeout=None, headers=None, **kwargs):  # noqa: ANN001
        raise RuntimeError("CelesTrak unreachable (offline test)")


def _aware_now() -> datetime:
    return datetime.now(timezone.utc)


def _auth() -> SyntheticAuthorization:
    return SyntheticAuthorization(acknowledge_synthetic=True)


def _mean_motion_rev_per_day(altitude_km: float) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    mean_motion_rad_s = math.sqrt(MU_EARTH_KM3_S2 / semi_major_km**3)
    return mean_motion_rad_s / REV_PER_DAY_TO_RAD_PER_S


def _celestrak_object(
    object_id: str,
    altitude_km: float = 550.0,
    *,
    mean_anomaly_deg: float = 0.0,
) -> SpaceObject:
    return SpaceObject(
        object_id=object_id,
        name=f"SAT-{object_id}",
        object_type=ObjectType.PAYLOAD,
        elements=OrbitalElements(
            epoch=_EPOCH,
            mean_motion_rev_per_day=_mean_motion_rev_per_day(altitude_km),
            eccentricity=0.0,
            inclination_deg=53.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=mean_anomaly_deg,
        ),
        data_source=DataSource.CELESTRAK,
    )


def _catalog(
    source: str,
    objects: list[SpaceObject],
    query: str = "test",
) -> Catalog:
    return Catalog(
        source=source,
        objects=objects,
        fetched_at=_aware_now(),
        query=query,
    )


def _fast_config() -> PipelineConfig:
    return PipelineConfig(duration_s=600.0, step_s=60.0, max_iterations=1)


def _three_sat_spec() -> SyntheticSpec:
    return SyntheticSpec(
        n_planes=1,
        sats_per_plane=3,
        include_known_conjunction_triple=True,
    )


def _run_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    existing = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"
    merged.pop("AEGIS_ALLOW_SYNTHETIC", None)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-m", "aegis.pipeline", *args],
        cwd=str(PACKAGE_ROOT),
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_exports_appear_in_all() -> None:
    assert _REQUIRED_EXPORTS <= set(pipeline_pkg.__all__)


def test_public_exports_are_importable() -> None:
    for name in _REQUIRED_EXPORTS:
        assert hasattr(pipeline_pkg, name)
    assert callable(run_pipeline)
    assert isinstance(PipelineError, type)
    PipelineConfig()
    assert issubclass(PipelineError, Exception)


def test_pipeline_config_defaults() -> None:
    config = PipelineConfig()
    assert config.duration_s is None or config.duration_s == pytest.approx(5400.0)
    assert config.step_s == SCREENING_STEP_S
    assert config.box_km == SCREENING_BOX_STARLINK_KM
    assert config.target_pc == PC_TARGET_POST_MANEUVER
    assert config.dv_budget_km_s == DEFAULT_DV_BUDGET_KM_S
    assert config.max_iterations == 5
    assert config.max_objects is None


# ---------------------------------------------------------------------------
# Source wall
# ---------------------------------------------------------------------------


def test_default_source_is_celestrak_with_injected_catalog() -> None:
    catalog = _catalog(
        DataSource.CELESTRAK,
        [_celestrak_object("25544"), _celestrak_object("25545", mean_anomaly_deg=40.0)],
        query="injected-celestrak",
    )
    result = run_pipeline(catalog=catalog, config=_fast_config())
    assert isinstance(result, PipelineResult)
    assert result.source == DataSource.CELESTRAK
    assert result.catalog.source == DataSource.CELESTRAK
    assert all(obj.data_source == DataSource.CELESTRAK for obj in result.catalog)


def test_synthetic_without_auth_raises() -> None:
    with pytest.raises(_GATE_ERRORS):
        run_pipeline(source=DataSource.SYNTHETIC, config=_fast_config())


def test_synthetic_without_env_raises() -> None:
    with pytest.raises(_GATE_ERRORS):
        run_pipeline(
            source=DataSource.SYNTHETIC,
            authorization=_auth(),
            synthetic_spec=_three_sat_spec(),
            config=_fast_config(),
        )


def test_synthetic_env_without_auth_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    with pytest.raises(_GATE_ERRORS):
        run_pipeline(
            source="SYNTHETIC",
            synthetic_spec=_three_sat_spec(),
            config=_fast_config(),
        )


def test_synthetic_authorized_3sat_spec_returns_synthetic_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    result = run_pipeline(
        source=DataSource.SYNTHETIC,
        authorization=_auth(),
        synthetic_spec=_three_sat_spec(),
        config=_fast_config(),
    )
    assert isinstance(result, PipelineResult)
    assert result.source == DataSource.SYNTHETIC
    assert result.catalog.source == DataSource.SYNTHETIC
    assert len(result.catalog) == 3
    assert all(obj.data_source == DataSource.SYNTHETIC for obj in result.catalog)
    assert isinstance(result.assessed, AssessedCatalog)
    assert isinstance(result.plan, ManeuverPlan)
    assert isinstance(result.config, PipelineConfig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"authorization": SyntheticAuthorization(acknowledge_synthetic=True)},
        {"synthetic_spec": SyntheticSpec()},
    ],
)
def test_celestrak_rejects_synthetic_arguments(kwargs: dict) -> None:
    catalog = _catalog(DataSource.CELESTRAK, [_celestrak_object("1")], query="wall")
    with pytest.raises(PipelineError):
        run_pipeline(
            source=DataSource.CELESTRAK,
            catalog=catalog,
            config=_fast_config(),
            **kwargs,
        )


@pytest.mark.parametrize("source", ["NORAD", "celestrak", "synthetic", "BOTH", "", "SPACETRACK"])
def test_invalid_source_string_raises(source: str) -> None:
    catalog = _catalog(DataSource.CELESTRAK, [_celestrak_object("1")], query="invalid-source")
    with pytest.raises(PipelineError):
        run_pipeline(source, catalog=catalog, config=_fast_config())


def test_injected_catalog_source_mismatch_raises() -> None:
    synthetic = _catalog(
        DataSource.SYNTHETIC,
        [SpaceObject(object_id="9", data_source=DataSource.SYNTHETIC)],
        query="synth",
    )
    with pytest.raises(_MISMATCH_ERRORS):
        run_pipeline(
            source=DataSource.CELESTRAK,
            catalog=synthetic,
            config=_fast_config(),
        )


def test_injected_celestrak_catalog_with_synthetic_source_arg_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    celestrak = _catalog(
        DataSource.CELESTRAK,
        [_celestrak_object("1")],
        query="celestrak",
    )
    with pytest.raises(_MISMATCH_ERRORS):
        run_pipeline(
            source=DataSource.SYNTHETIC,
            catalog=celestrak,
            authorization=_auth(),
            config=_fast_config(),
        )


# ---------------------------------------------------------------------------
# Empty catalog and CelesTrak failure
# ---------------------------------------------------------------------------


def test_empty_celestrak_catalog_returns_empty_plan() -> None:
    catalog = _catalog(DataSource.CELESTRAK, [], query="empty")
    result = run_pipeline(source=DataSource.CELESTRAK, catalog=catalog, config=_fast_config())
    assert result.source == DataSource.CELESTRAK
    assert len(result.catalog) == 0
    assert result.assessed.entries == []
    assert result.plan.total_burns == 0
    assert len(result.plan.resolved) == 0


def test_celestrak_http_500_does_not_produce_synthetic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    session = _FakeSession(_FakeResponse(text="internal error", status_code=500))
    result = None
    with pytest.raises(PipelineError):
        result = run_pipeline(
            source=DataSource.CELESTRAK,
            session=session,
            cache_dir=tmp_path,
            config=_fast_config(),
        )
    assert result is None


def test_celestrak_session_raise_does_not_produce_synthetic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    result = None
    with pytest.raises(PipelineError):
        result = run_pipeline(
            source="CELESTRAK",
            session=_RaisingSession(),
            cache_dir=tmp_path,
            config=_fast_config(),
        )
    assert result is None


# ---------------------------------------------------------------------------
# Result surface
# ---------------------------------------------------------------------------


def test_plan_summary_and_covariance_source_on_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    result = run_pipeline(
        source="SYNTHETIC",
        authorization=_auth(),
        synthetic_spec=_three_sat_spec(),
        config=_fast_config(),
    )
    assert result.covariance_source == CovarianceSource.SYNTHETIC_TLE
    summary = result.plan.summary()
    assert isinstance(summary, dict)
    assert "plan_id" in summary
    assert "total_burns" in summary
    assert "conjunctions_resolved" in summary
    assert summary["total_burns"] == result.plan.total_burns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_synthetic_without_acknowledge_exits_nonzero() -> None:
    completed = _run_cli(["--source", "SYNTHETIC"])
    assert completed.returncode != 0
    stderr = completed.stderr.lower()
    assert "synthetic" in stderr
    assert "refused" in stderr
    for marker in ("plan_id", "total_burns", "maneuvering_satellites"):
        assert marker not in completed.stdout
