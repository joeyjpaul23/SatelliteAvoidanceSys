"""Data-source wall, authorization gate, catalog homogeneity, import isolation.

These tests encode the Step 1 ingest contract. They import only the public
``aegis.ingest`` surface plus allowed core types.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aegis.core.objects import SpaceObject
from aegis.ingest import (
    Catalog,
    CatalogError,
    CelesTrakError,
    DataSource,
    MixedDataSourceError,
    SyntheticAuthorization,
    SyntheticNotAuthorizedError,
    SyntheticSpec,
    fetch_celestrak,
    generate_synthetic,
)
import aegis.ingest as ingest_pkg

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"

# Well-known ISS TLE pair (valid checksums). Used only to exercise the
# CelesTrak path through a mocked session — never against the network.
_ISS_NAME = "ISS (ZARYA)"
_ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
_ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
_ISS_TLE = f"{_ISS_NAME}\n{_ISS_L1}\n{_ISS_L2}\n"

_REQUIRED_EXPORTS = {
    "DataSource",
    "Catalog",
    "CatalogError",
    "MixedDataSourceError",
    "SyntheticNotAuthorizedError",
    "SyntheticAuthorization",
    "SyntheticSpec",
    "CelesTrakClient",
    "CelesTrakError",
    "fetch_celestrak",
    "generate_synthetic",
}


class _FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.ok = status_code < 400
        self.content = text.encode("utf-8")
        self.headers: dict[str, str] = {}

    def json(self):  # noqa: ANN201 - requests-like
        raise ValueError("not a JSON fixture")

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


def _aware_now() -> datetime:
    return datetime.now(timezone.utc)


def _auth() -> SyntheticAuthorization:
    return SyntheticAuthorization(acknowledge_synthetic=True)


def _catalog(
    source: str,
    objects: list[SpaceObject],
    query: str = "test",
) -> Catalog:
    return Catalog(source=source, objects=objects, fetched_at=_aware_now(), query=query)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_exports_appear_in_all() -> None:
    assert _REQUIRED_EXPORTS <= set(ingest_pkg.__all__)


def test_public_exports_are_importable() -> None:
    for name in _REQUIRED_EXPORTS:
        assert hasattr(ingest_pkg, name)


def test_datasource_constants() -> None:
    assert DataSource.CELESTRAK == "CELESTRAK"
    assert DataSource.SYNTHETIC == "SYNTHETIC"


def test_error_hierarchy() -> None:
    assert issubclass(MixedDataSourceError, CatalogError)
    assert issubclass(SyntheticNotAuthorizedError, CatalogError)
    assert issubclass(CelesTrakError, Exception)


# ---------------------------------------------------------------------------
# Synthetic authorization gate
# ---------------------------------------------------------------------------


def test_generate_synthetic_without_auth_raises() -> None:
    spec = SyntheticSpec()
    with pytest.raises(SyntheticNotAuthorizedError):
        generate_synthetic(None, spec)


def test_generate_synthetic_without_auth_raises_even_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    with pytest.raises(SyntheticNotAuthorizedError):
        generate_synthetic(None, SyntheticSpec())


@pytest.mark.parametrize("value", [None, "true", "yes", "0", "", "TRUE", "True", "1 "])
def test_generate_synthetic_with_auth_but_env_not_exactly_1_raises(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
    else:
        monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", value)
    with pytest.raises(SyntheticNotAuthorizedError):
        generate_synthetic(_auth(), SyntheticSpec())


def test_generate_synthetic_auth_and_env_1_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    catalog = generate_synthetic(_auth(), SyntheticSpec())
    assert catalog.source == DataSource.SYNTHETIC
    assert all(obj.data_source == DataSource.SYNTHETIC for obj in catalog)


def test_synthetic_authorization_positional_true_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        SyntheticAuthorization(True)  # type: ignore[misc]


def test_acknowledge_synthetic_false_raises() -> None:
    with pytest.raises(SyntheticNotAuthorizedError):
        SyntheticAuthorization(acknowledge_synthetic=False)


@pytest.mark.parametrize("value", [False, None, 1, "true", "True", 0, ""])
def test_acknowledge_synthetic_non_true_raises(value: object) -> None:
    with pytest.raises(SyntheticNotAuthorizedError):
        SyntheticAuthorization(acknowledge_synthetic=value)  # type: ignore[arg-type]


def test_unauthorized_error_message_mentions_refused_and_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
    with pytest.raises(SyntheticNotAuthorizedError) as exc_info:
        generate_synthetic(_auth(), SyntheticSpec())
    text = str(exc_info.value)
    lowered = text.lower()
    assert "refused" in lowered
    assert "AEGIS_ALLOW_SYNTHETIC" in text
    assert "1" in text
    assert "authorization" in lowered


# ---------------------------------------------------------------------------
# CelesTrak path must not consult the synthetic env var
# ---------------------------------------------------------------------------


def test_fetch_celestrak_works_without_synthetic_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
    session = _FakeSession(_FakeResponse(text=_ISS_TLE))
    catalog = fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    assert catalog.source == DataSource.CELESTRAK
    assert len(catalog) >= 1


@pytest.mark.parametrize("env_value", [None, "0", "true", "1", ""])
def test_fetch_celestrak_works_regardless_of_synthetic_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv("AEGIS_ALLOW_SYNTHETIC", raising=False)
    else:
        monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", env_value)
    session = _FakeSession(_FakeResponse(text=_ISS_TLE))
    catalog = fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    assert catalog.source == DataSource.CELESTRAK


# ---------------------------------------------------------------------------
# Catalog homogeneity
# ---------------------------------------------------------------------------


def test_catalog_rejects_mixed_object_sources() -> None:
    celestrak = SpaceObject(object_id="1", data_source=DataSource.CELESTRAK)
    synthetic = SpaceObject(object_id="2", data_source=DataSource.SYNTHETIC)
    with pytest.raises(MixedDataSourceError):
        _catalog(DataSource.CELESTRAK, [celestrak, synthetic])


def test_catalog_rejects_object_source_mismatching_catalog_source() -> None:
    synthetic = SpaceObject(object_id="9", data_source=DataSource.SYNTHETIC)
    with pytest.raises(MixedDataSourceError):
        _catalog(DataSource.CELESTRAK, [synthetic])


def test_catalog_empty_allowed_and_carries_source() -> None:
    catalog = _catalog(DataSource.CELESTRAK, [])
    assert catalog.source == DataSource.CELESTRAK
    assert len(catalog) == 0
    assert list(catalog) == []


def test_catalog_len_and_iteration() -> None:
    objects = [
        SpaceObject(object_id="10", data_source=DataSource.CELESTRAK),
        SpaceObject(object_id="20", data_source=DataSource.CELESTRAK),
    ]
    catalog = _catalog(DataSource.CELESTRAK, objects)
    assert len(catalog) == 2
    assert list(catalog) == objects


def test_catalog_same_source_construction_succeeds() -> None:
    objects = [
        SpaceObject(object_id="1", data_source=DataSource.SYNTHETIC),
        SpaceObject(object_id="2", data_source=DataSource.SYNTHETIC),
    ]
    catalog = _catalog(DataSource.SYNTHETIC, objects, query="synth")
    assert catalog.source == DataSource.SYNTHETIC
    assert catalog.query == "synth"
    assert catalog.fetched_at.tzinfo is not None
    assert catalog.fetched_at.utcoffset() == timezone.utc.utcoffset(catalog.fetched_at)


def test_synthetic_catalog_objects_are_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    catalog = generate_synthetic(_auth(), SyntheticSpec())
    assert catalog.source == DataSource.SYNTHETIC
    assert catalog.objects
    for obj in catalog:
        assert obj.data_source == DataSource.SYNTHETIC


def test_celestrak_catalog_objects_are_celestrak(tmp_path: Path) -> None:
    session = _FakeSession(_FakeResponse(text=_ISS_TLE))
    catalog = fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    assert catalog.source == DataSource.CELESTRAK
    assert catalog.objects
    for obj in catalog:
        assert obj.data_source == DataSource.CELESTRAK


def test_catalog_mixed_merge_extend_or_add_raises() -> None:
    left = _catalog(
        DataSource.CELESTRAK,
        [SpaceObject(object_id="1", data_source=DataSource.CELESTRAK)],
        query="left",
    )
    right = _catalog(
        DataSource.SYNTHETIC,
        [SpaceObject(object_id="2", data_source=DataSource.SYNTHETIC)],
        query="right",
    )
    if hasattr(left, "merge"):
        with pytest.raises(MixedDataSourceError):
            left.merge(right)
    if hasattr(left, "extend"):
        with pytest.raises(MixedDataSourceError):
            left.extend(right)
    if hasattr(left, "__add__"):
        with pytest.raises(MixedDataSourceError):
            _ = left + right


# ---------------------------------------------------------------------------
# Import isolation (fresh subprocess so this tester process stays clean)
# ---------------------------------------------------------------------------


def _run_isolated(snippet: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(PACKAGE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_celestrak_does_not_load_synthetic() -> None:
    snippet = (
        "import importlib, sys; "
        "importlib.import_module('aegis.ingest.celestrak'); "
        "assert 'aegis.ingest.synthetic' not in sys.modules"
    )
    result = _run_isolated(snippet)
    assert result.returncode == 0, result.stderr + result.stdout


def test_importing_synthetic_does_not_load_celestrak() -> None:
    snippet = (
        "import importlib, sys; "
        "importlib.import_module('aegis.ingest.synthetic'); "
        "assert 'aegis.ingest.celestrak' not in sys.modules"
    )
    result = _run_isolated(snippet)
    assert result.returncode == 0, result.stderr + result.stdout


def test_celestrak_client_is_importable_without_loading_synthetic() -> None:
    snippet = (
        "import aegis.ingest.celestrak as c; import sys; "
        "assert 'aegis.ingest.synthetic' not in sys.modules"
    )
    result = _run_isolated(snippet)
    assert result.returncode == 0, result.stderr + result.stdout


def test_synthetic_module_importable_without_loading_celestrak() -> None:
    snippet = (
        "import aegis.ingest.synthetic as s; import sys; "
        "assert 'aegis.ingest.celestrak' not in sys.modules"
    )
    result = _run_isolated(snippet)
    assert result.returncode == 0, result.stderr + result.stdout
