"""Complete Starlink SOCRATES feed ingestion and API coverage."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from aegis.api import app
from aegis.ingest.socrates import (
    SOCRATES_CSV_URL,
    SocratesError,
    fetch_starlink_alerts,
    parse_socrates_csv,
)

_HEADER = (
    "NORAD_CAT_ID_1,OBJECT_NAME_1,DSE_1,NORAD_CAT_ID_2,OBJECT_NAME_2,DSE_2,"
    "TCA,TCA_RANGE,TCA_RELATIVE_SPEED,MAX_PROB,DILUTION\n"
)
_CSV = _HEADER + """43711,SHIYAN-6 01 [+],1.5,47772,STARLINK-2195 [+],2.3,2026-09-11 04:37:54.148,0.558,14.681,1.812E-04,0.311
66372,STARLINK-35821 [+],3.9,100038,STARLINK-37672 [+],3.3,2026-09-12 19:41:13.228,0.981,4.744,4.384E-04,0.242
57462,STARLINK-30088 [+],2.3,60749,OBJECT E [+],2.1,2026-09-11 18:00:11.463,1.751,7.670,3.834E-05,0.469
11111,NOT STAR LINK,1.0,22222,COSMOS DEB,1.0,2026-09-13 00:00:00,0.010,8.000,9.000E-01,0.100
"""


class _Response:
    status_code = 200
    text = _CSV


class _Session:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url, **kwargs):  # noqa: ANN001
        self.calls.append({"url": url, **kwargs})
        return _Response()


class _FailedResponse:
    status_code = 503
    text = "unavailable"


def test_parser_keeps_starlink_on_either_side_and_six_digit_ids() -> None:
    catalog = parse_socrates_csv(
        _CSV, fetched_at=datetime(2026, 9, 10, tzinfo=timezone.utc)
    )

    assert catalog.total_feed_events == 4
    assert len(catalog.events) == 3
    assert {event.norad_id_2 for event in catalog.events} >= {"100038", "60749"}
    assert catalog.events[0].starlink_ids == ("47772",)
    assert catalog.events[1].starlink_ids == ("66372", "100038")
    assert catalog.unique_starlink_ids == ("100038", "47772", "57462", "66372")


def test_parser_rejects_schema_drift() -> None:
    with pytest.raises(SocratesError, match="missing required fields"):
        parse_socrates_csv("OBJECT_NAME_1,TCA\nSTARLINK-1,2026-01-01 00:00:00\n")


def test_fetch_uses_cache_and_identifies_itself(tmp_path) -> None:
    session = _Session()
    first = fetch_starlink_alerts(session=session, cache_dir=tmp_path)
    second = fetch_starlink_alerts(session=session, cache_dir=tmp_path)

    assert len(first.events) == len(second.events) == 3
    assert len(session.calls) == 1
    assert session.calls[0]["url"] == SOCRATES_CSV_URL
    assert "AEGIS" in session.calls[0]["headers"]["User-Agent"]


def test_fetch_falls_back_to_marked_stale_cache(tmp_path, monkeypatch) -> None:
    first = fetch_starlink_alerts(session=_Session(), cache_dir=tmp_path)
    cache = tmp_path / "sort-minRange.csv"
    old = cache.stat().st_mtime - 86400
    cache.touch()
    monkeypatch.setattr("aegis.ingest.socrates.time.time", lambda: old + 172800)

    failed = _Session()
    failed.get = lambda *_args, **_kwargs: _FailedResponse()
    fallback = fetch_starlink_alerts(session=failed, cache_dir=tmp_path)

    assert first.stale is False
    assert fallback.stale is True
    assert len(fallback.events) == 3


def test_api_reports_complete_counts_filters_and_pagination(monkeypatch) -> None:
    catalog = parse_socrates_csv(
        _CSV, fetched_at=datetime(2026, 9, 10, tzinfo=timezone.utc)
    )
    app_module = importlib.import_module("aegis.api.app")
    monkeypatch.setattr(app_module, "fetch_starlink_alerts", lambda: catalog)

    response = TestClient(app).get(
        "/api/starlink-alerts",
        params={"limit": 1, "offset": 0, "min_max_probability": 1e-4},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_feed_events"] == 4
    assert payload["total_starlink_events"] == 3
    assert payload["filtered_events"] == 2
    assert payload["unique_starlink_objects"] == 4
    assert payload["has_more"] is True
    assert len(payload["events"]) == 1
    assert payload["events"][0]["norad_id_2"] == "100038"
    assert payload["events"][0]["probability_kind"] == "SOCRATES_MAXIMUM_PROBABILITY"
