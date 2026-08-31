"""GET /api/scene contract tests (ui_console).

Live CelesTrak is always mocked. These tests never hit the network and
never require ``generate_synthetic``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aegis.api import app
from aegis.ingest.celestrak import CelesTrakError

_SLICE = Path(__file__).resolve().parent / "fixtures" / "starlink_slice.tle"
_COLOR_BANDS = frozenset({"CLEAR", "MONITOR", "WATCH", "ACT"})
_FAKE_NAME_MARKERS = (
    "SYNTHETIC-",
    "SYNTHETIC OPERATOR",
    "SYNTHETIC-OP",
)


def _fail_celestrak(*_args, **_kwargs):
    raise CelesTrakError("forced live CelesTrak failure")


def _fail_synthetic(*_args, **_kwargs):
    raise AssertionError("generate_synthetic must not be required for GET /api/scene")


def _patch_live_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aegis.ingest.celestrak.fetch_celestrak", _fail_celestrak
    )
    monkeypatch.setattr(
        "aegis.ingest.fetch_celestrak", _fail_celestrak, raising=False
    )
    monkeypatch.setattr(
        "aegis.ingest.generate_synthetic", _fail_synthetic, raising=False
    )
    monkeypatch.setattr(
        "aegis.ingest.synthetic.generate_synthetic",
        _fail_synthetic,
        raising=False,
    )

    import sys

    for name, mod in list(sys.modules.items()):
        if not name.startswith("aegis."):
            continue
        if hasattr(mod, "fetch_celestrak"):
            monkeypatch.setattr(mod, "fetch_celestrak", _fail_celestrak)
        if hasattr(mod, "generate_synthetic"):
            monkeypatch.setattr(mod, "generate_synthetic", _fail_synthetic)


def _client() -> TestClient:
    return TestClient(app)


def _get_scene(client: TestClient, **params):
    return client.get("/api/scene", params=params)


def _assert_not_synthetic_catalog(payload: dict) -> None:
    source = payload.get("source")
    assert source == "CELESTRAK"
    assert source != "SYNTHETIC"

    objects = payload.get("objects")
    assert isinstance(objects, list)

    for obj in objects:
        assert isinstance(obj, dict)
        if "data_source" in obj:
            assert obj["data_source"] != "SYNTHETIC"
        identity = " ".join(
            str(obj.get(key) or "") for key in ("id", "name", "object_id")
        )
        upper = identity.upper()
        for marker in _FAKE_NAME_MARKERS:
            assert marker not in upper, (
                f"object identity looks like a synthetic constellation: {identity!r}"
            )

    honesty = payload.get("honesty")
    if honesty is not None:
        assert isinstance(honesty, list)
        for line in honesty:
            text = str(line)
            if "SYNTHETIC_TLE" in text.upper():
                continue
            assert "catalog" not in text.lower() or "synthetic" not in text.lower()


def test_live_fail_falls_back_to_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    response = _get_scene(_client(), max_objects=5, live=1)

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "CELESTRAK"
    assert payload["fallback"] == "slice"
    _assert_not_synthetic_catalog(payload)

    objects = payload["objects"]
    if _SLICE.is_file():
        assert objects, (
            "committed slice exists; scene objects must not be empty on fallback"
        )


def test_max_objects_five_caps_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    response = _get_scene(_client(), max_objects=5, live=0)

    assert response.status_code == 200
    payload = response.json()
    objects = payload["objects"]
    assert isinstance(objects, list)
    assert len(objects) <= 5


def test_max_objects_over_cap_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    response = _get_scene(_client(), max_objects=201)

    assert response.status_code == 400


def test_scene_color_band_and_honesty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    response = _get_scene(_client(), max_objects=5, live=0)

    assert response.status_code == 200
    payload = response.json()
    objects = payload["objects"]
    assert isinstance(objects, list)
    assert objects
    for obj in objects:
        assert obj["color_band"] in _COLOR_BANDS

    honesty = payload["honesty"]
    assert isinstance(honesty, list)
    assert honesty
    assert all(isinstance(line, str) for line in honesty)
    assert any("SYNTHETIC_TLE" in line for line in honesty)
