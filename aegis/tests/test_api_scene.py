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
_AT_RISK_BANDS = frozenset({"MONITOR", "WATCH", "ACT"})
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
    response = _get_scene(_client(), max_objects=5, duration_s=600, live=1)

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "CELESTRAK"
    assert payload["fallback"] == "slice"
    _assert_not_synthetic_catalog(payload)

    objects = payload["objects"]
    assert isinstance(objects, list)
    if _SLICE.is_file():
        assert payload["screened_objects"] > 0
        assert payload["fallback"] == "slice"


def test_scene_fallback_includes_debris(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    response = _get_scene(_client(), max_objects=8, duration_s=600, live=1)

    assert response.status_code == 200
    payload = response.json()
    query = str(payload.get("query") or "").upper()
    assert payload["screened_objects"] == 8
    assert "STARLINK" in query or "STARLINK" in " ".join(
        str(obj.get("name") or "") for obj in payload["objects"]
    ).upper()
    assert "DEB" in query or "DEBRIS" in query
    assert payload["duration_s"] == pytest.approx(600)
    objects = payload["objects"]
    if objects:
        names = " ".join(str(obj.get("name") or "") for obj in objects)
        roles = {obj.get("role") for obj in objects}
        assert "STARLINK" in names.upper() or "debris" in roles


def test_max_objects_five_caps_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    response = _get_scene(_client(), max_objects=5, duration_s=600, live=0)

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
    response = _get_scene(_client(), max_objects=5, duration_s=600, live=0)

    assert response.status_code == 200
    payload = response.json()
    objects = payload["objects"]
    assert isinstance(objects, list)
    for obj in objects:
        assert obj["color_band"] in _AT_RISK_BANDS
        assert obj["color_band"] != "CLEAR"

    honesty = payload["honesty"]
    assert isinstance(honesty, list)
    assert honesty
    assert all(isinstance(line, str) for line in honesty)
    assert any("SYNTHETIC_TLE" in line for line in honesty)
    assert any("MONITOR" in line and "WATCH" in line for line in honesty)


def test_scene_omits_clear_events_and_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    response = _get_scene(_client(), max_objects=8, duration_s=600, live=0)

    assert response.status_code == 200
    payload = response.json()
    assert "screened_objects" in payload
    assert payload["screened_objects"] == 8
    for obj in payload["objects"]:
        assert obj["color_band"] in _AT_RISK_BANDS
        assert obj["color_band"] != "CLEAR"
    for row in payload["conjunctions"]:
        assert row["display_band"] in _AT_RISK_BANDS
        assert row["display_band"] != "CLEAR"
    displayed_ids = {obj["id"] for obj in payload["objects"]}
    for row in payload["conjunctions"]:
        assert row["primary_id"] in displayed_ids
        assert row["secondary_id"] in displayed_ids
    if not payload["conjunctions"]:
        assert payload["objects"] == []
        assert any("No MONITOR+" in line for line in payload["honesty"])


@pytest.mark.parametrize(
    "params",
    [
        {"duration_s": 0},
        {"duration_s": -60},
        {"duration_s": 8 * 86400},
        {"duration_s": "nan"},
        {"duration_s": "inf"},
        {"step_s": 0},
        {"step_s": 0.01},
        {"step_s": 600},
    ],
)
def test_scene_rejects_windows_and_steps_out_of_range(monkeypatch: pytest.MonkeyPatch, params) -> None:
    _patch_live_fail(monkeypatch)
    assert _get_scene(_client(), max_objects=5, live=0, **params).status_code == 422


def test_scene_honesty_names_the_requested_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_live_fail(monkeypatch)
    honesty = _get_scene(_client(), max_objects=5, duration_s=600, live=0).json()["honesty"]
    assert any("10-minute" in line for line in honesty)


def test_each_pair_keeps_its_highest_pc_approach() -> None:
    from aegis.api.scene import _keep_riskiest_per_pair

    near = {"id": "a", "primary_id": "1", "secondary_id": "2", "miss_km": 0.4, "pc": 1e-6, "tca": "t1"}
    riskier = {"id": "b", "primary_id": "2", "secondary_id": "1", "miss_km": 0.9, "pc": 3e-5, "tca": "t2"}
    other = {"id": "c", "primary_id": "1", "secondary_id": "3", "miss_km": 2.0, "pc": 1e-7, "tca": "t3"}
    assert [row["id"] for row in _keep_riskiest_per_pair([near, riskier, other])] == ["b", "c"]
