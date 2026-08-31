"""CelesTrak client contract tests. HTTP is always mocked; never hit the network."""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pytest

from aegis.constants import CELESTRAK_MAX_RETRIES, CELESTRAK_USER_AGENT
from aegis.ingest import (
    CelesTrakClient,
    CelesTrakError,
    DataSource,
    fetch_celestrak,
)
from aegis.propagation.propagator import Sgp4Propagator

# Valid ISS TLE (checksums verified). Used as a tiny well-known pair.
ISS_NAME = "ISS (ZARYA)"
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
ISS_TLE_3LINE = f"{ISS_NAME}\n{ISS_LINE1}\n{ISS_LINE2}\n"
ISS_TLE_2LINE = f"{ISS_LINE1}\n{ISS_LINE2}\n"

ISS_OMM = {
    "OBJECT_NAME": ISS_NAME,
    "OBJECT_ID": "1998-067A",
    "NORAD_CAT_ID": 25544,
    "EPOCH": "2008-09-20T12:25:40.104192",
    "MEAN_MOTION": 15.72125391,
    "ECCENTRICITY": 0.0006703,
    "INCLINATION": 51.6416,
    "RA_OF_ASC_NODE": 247.4627,
    "ARG_OF_PERICENTER": 130.5360,
    "MEAN_ANOMALY": 325.0288,
    "BSTAR": -1.1606e-5,
    "TLE_LINE1": ISS_LINE1,
    "TLE_LINE2": ISS_LINE2,
}


class FakeResponse:
    def __init__(
        self,
        *,
        text: str = "",
        json_data=None,
        status_code: int = 200,
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.ok = status_code < 400
        self.headers: dict[str, str] = {}
        self._json = json_data
        self.content = text.encode("utf-8")

    def json(self):  # noqa: ANN201 - requests-like
        if self._json is not None:
            return self._json
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """requests-like session that records ``get(url, timeout=..., headers=...)``."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[dict] = []

    def get(self, url, timeout=None, headers=None, **kwargs):  # noqa: ANN001
        self.calls.append(
            {"url": url, "timeout": timeout, "headers": headers or {}, **kwargs}
        )
        if callable(self._responder):
            return self._responder(url)
        return self._responder


def _tle_response() -> FakeResponse:
    return FakeResponse(text=ISS_TLE_3LINE)


def _json_response() -> FakeResponse:
    payload = [ISS_OMM]
    return FakeResponse(text=json.dumps(payload), json_data=payload)


def _user_agent(headers: dict) -> str | None:
    for key, value in headers.items():
        if key.lower() == "user-agent":
            return value
    return None


def test_fetch_gp_tle_parses_catalog(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    client = CelesTrakClient(cache_dir=tmp_path, session=session)
    catalog = client.fetch_gp("stations", fmt="tle")

    assert catalog.source == DataSource.CELESTRAK
    assert "stations" in catalog.query
    assert catalog.fetched_at.tzinfo is not None
    assert catalog.fetched_at.utcoffset() == timezone.utc.utcoffset(catalog.fetched_at)
    assert len(catalog) == 1

    obj = catalog.objects[0]
    assert obj.object_id == "25544"
    assert obj.tle_line1 == ISS_LINE1
    assert obj.tle_line2 == ISS_LINE2
    assert obj.name == ISS_NAME
    assert obj.elements is not None
    assert obj.data_source == DataSource.CELESTRAK


def test_fetch_celestrak_tle_lines_verbatim(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    catalog = fetch_celestrak("stations", fmt="tle", session=session, cache_dir=tmp_path)
    obj = catalog.objects[0]
    assert obj.tle_line1 == ISS_LINE1
    assert obj.tle_line2 == ISS_LINE2


def test_fetch_gp_json_parses_catalog(tmp_path: Path) -> None:
    session = FakeSession(_json_response())
    client = CelesTrakClient(cache_dir=tmp_path, session=session)
    catalog = client.fetch_gp("stations", fmt="json")

    assert catalog.source == DataSource.CELESTRAK
    assert "stations" in catalog.query
    assert len(catalog) == 1

    obj = catalog.objects[0]
    assert obj.name == ISS_NAME
    assert obj.data_source == DataSource.CELESTRAK
    assert obj.elements is not None
    assert obj.elements.inclination_deg == pytest.approx(51.6416)
    assert obj.elements.eccentricity == pytest.approx(0.0006703)
    assert obj.elements.mean_motion_rev_per_day == pytest.approx(15.72125391)
    assert obj.object_id in {"25544", "1998-067A"}


def test_fetch_celestrak_json_path(tmp_path: Path) -> None:
    session = FakeSession(_json_response())
    catalog = fetch_celestrak("stations", fmt="json", session=session, cache_dir=tmp_path)
    assert catalog.source == DataSource.CELESTRAK
    assert catalog.objects[0].elements is not None


def test_json_tle_lines_kept_verbatim(tmp_path: Path) -> None:
    session = FakeSession(_json_response())
    catalog = fetch_celestrak("stations", fmt="json", session=session, cache_dir=tmp_path)
    obj = catalog.objects[0]
    assert obj.tle_line1 == ISS_LINE1
    assert obj.tle_line2 == ISS_LINE2


def test_user_agent_header_equals_constant(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    assert session.calls, "session.get must be called"
    ua = _user_agent(session.calls[0]["headers"])
    assert ua == CELESTRAK_USER_AGENT


def test_cache_reused_within_ttl_does_not_call_get(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    client = CelesTrakClient(cache_dir=tmp_path, session=session)
    first = client.fetch_gp("stations", fmt="tle")
    assert len(session.calls) == 1
    second = client.fetch_gp("stations", fmt="tle")
    assert len(session.calls) == 1
    assert first.objects[0].object_id == second.objects[0].object_id
    assert first.objects[0].tle_line1 == second.objects[0].tle_line1


def test_cache_reused_across_client_instances(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    first = CelesTrakClient(cache_dir=tmp_path, session=session)
    first.fetch_gp("stations", fmt="tle")
    assert len(session.calls) == 1
    second = CelesTrakClient(cache_dir=tmp_path, session=session)
    second.fetch_gp("stations", fmt="tle")
    assert len(session.calls) == 1


def test_cache_key_differs_by_group(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    client = CelesTrakClient(cache_dir=tmp_path, session=session)
    client.fetch_gp("stations", fmt="tle")
    client.fetch_gp("visual", fmt="tle")
    assert len(session.calls) == 2
    assert "stations" in session.calls[0]["url"]
    assert "visual" in session.calls[1]["url"]


def test_http_500_after_retries_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    session = FakeSession(FakeResponse(text="internal error", status_code=500))
    with pytest.raises(CelesTrakError):
        fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    assert session.calls, "session.get must be attempted"
    assert len(session.calls) <= CELESTRAK_MAX_RETRIES + 1


def test_supplement_true_uses_supplemental_url(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    fetch_celestrak(
        "starlink",
        supplement=True,
        session=session,
        cache_dir=tmp_path,
    )
    assert session.calls, "session.get must be called"
    url = session.calls[0]["url"]
    assert "supplemental" in url
    assert "starlink" in url


def test_default_url_is_gp_not_supplemental(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    fetch_celestrak("stations", supplement=False, session=session, cache_dir=tmp_path)
    url = session.calls[0]["url"]
    assert "supplemental" not in url
    assert "gp.php" in url
    assert "stations" in url


def test_query_includes_group_name(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    catalog = fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    assert "stations" in catalog.query


def test_two_line_tle_without_name(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(text=ISS_TLE_2LINE))
    catalog = fetch_celestrak("stations", fmt="tle", session=session, cache_dir=tmp_path)
    assert len(catalog) == 1
    obj = catalog.objects[0]
    assert obj.object_id == "25544"
    assert obj.tle_line1 == ISS_LINE1
    assert obj.tle_line2 == ISS_LINE2
    assert obj.elements is not None


def test_celestrak_objects_propagate_with_sgp4(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    catalog = fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    propagator = Sgp4Propagator(list(catalog))
    epoch = catalog.objects[0].elements.epoch
    grid = propagator.propagate_grid(epoch, duration_s=600.0, step_s=60.0)
    assert grid.n_objects == 1
    assert bool(grid.valid.any())


def test_get_receives_timeout(tmp_path: Path) -> None:
    session = FakeSession(_tle_response())
    fetch_celestrak("stations", session=session, cache_dir=tmp_path)
    assert session.calls[0]["timeout"] is not None


def test_client_fetch_gp_and_module_function_agree_on_source(tmp_path: Path) -> None:
    session_a = FakeSession(_tle_response())
    session_b = FakeSession(_tle_response())
    via_client = CelesTrakClient(cache_dir=tmp_path / "a", session=session_a).fetch_gp(
        "stations"
    )
    via_fn = fetch_celestrak("stations", session=session_b, cache_dir=tmp_path / "b")
    assert via_client.source == via_fn.source == DataSource.CELESTRAK
    assert via_client.objects[0].object_id == via_fn.objects[0].object_id
