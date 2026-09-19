"""Space-Track client contract tests. HTTP is always mocked; never hit the network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aegis.ingest import Catalog, DataSource, MixedDataSourceError
from aegis.ingest.ops import load_debris_slice, load_ops_catalog, load_starlink_slice
from aegis.ingest.spacetrack import (
    RequestThrottle,
    SpaceTrackAuthError,
    SpaceTrackClient,
    SpaceTrackCredentials,
    SpaceTrackError,
    credentials_from_env,
    main,
    parse_cdm_public,
    query_url,
)
from aegis.pipeline.errors import PipelineError
from aegis.pipeline.run import run_pipeline

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

CREDS = SpaceTrackCredentials(user="analyst@example.com", password="hunter2-secret")

STARLINK_LINE1 = "1 44714U 19074B   26260.50000000  .00001234  00000-0  89012-4 0  9991"
STARLINK_LINE2 = "2 44714  53.0543 123.4567 0001234  90.1234 269.9876 15.06398765123456"

STARLINK_GP = {
    "NORAD_CAT_ID": "44714",
    "OBJECT_NAME": "STARLINK-1008",
    "OBJECT_ID": "2019-074B",
    "OBJECT_TYPE": "PAYLOAD",
    "EPOCH": "2026-09-17T12:00:00.000000",
    "MEAN_MOTION": "15.06398765",
    "ECCENTRICITY": "0.00012340",
    "INCLINATION": "53.0543",
    "RA_OF_ASC_NODE": "123.4567",
    "ARG_OF_PERICENTER": "90.1234",
    "MEAN_ANOMALY": "269.9876",
    "BSTAR": "0.000089012",
    "MEAN_MOTION_DOT": "0.00001234",
    "MEAN_MOTION_DDOT": "0",
    "ELEMENT_SET_NO": "999",
    "REV_AT_EPOCH": "12345",
    "TLE_LINE1": STARLINK_LINE1,
    "TLE_LINE2": STARLINK_LINE2,
}

CDM_PUBLIC = [
    {
        "CDM_ID": "1234567890",
        "CREATED": "2026-09-18 03:14:15.000000",
        "EMERGENCY_REPORTABLE": "Y",
        "TCA": "2026-09-19T10:20:30.500000",
        "MIN_RNG": "187",
        "PC": "0.000213",
        "SAT_1_ID": "44714",
        "SAT_1_NAME": "STARLINK-1008",
        "SAT1_OBJECT_TYPE": "PAYLOAD",
        "SAT_2_ID": "33759",
        "SAT_2_NAME": "COSMOS 2251 DEB",
        "SAT2_OBJECT_TYPE": "DEBRIS",
    },
    {
        "CDM_ID": "1234567891",
        "CREATED": "2026-09-18 04:00:00.000000",
        "EMERGENCY_REPORTABLE": "N",
        "TCA": "2026-09-20T01:02:03.000000",
        "MIN_RNG": "950",
        "PC": "",
        "SAT_1_ID": "25544",
        "SAT_1_NAME": "ISS (ZARYA)",
        "SAT1_OBJECT_TYPE": "PAYLOAD",
        "SAT_2_ID": "49863",
        "SAT_2_NAME": "FENGYUN 1C DEB",
        "SAT2_OBJECT_TYPE": "DEBRIS",
    },
]


class FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class FakeSession:
    """Scripted ``requests.Session`` stand-in recording every call."""

    def __init__(self, gets=(), login: FakeResponse | None = None) -> None:
        self.gets = list(gets)
        self.login_response = login if login is not None else FakeResponse("")
        self.posts: list[dict] = []
        self.urls: list[str] = []

    def post(self, url, data=None, **_kwargs):
        self.posts.append({"url": url, "data": dict(data or {})})
        return self.login_response

    def get(self, url, **_kwargs):
        self.urls.append(url)
        if not self.gets:
            raise AssertionError(f"unexpected GET {url}")
        return self.gets.pop(0)


class NoNetwork:
    def post(self, *args, **kwargs):
        raise AssertionError("network used")

    def get(self, *args, **kwargs):
        raise AssertionError("network used")


def _client(tmp_path: Path, session, credentials=CREDS) -> SpaceTrackClient:
    return SpaceTrackClient(credentials, cache_dir=tmp_path, session=session)


# ---------------------------------------------------------------------------
# Credentials and URLs
# ---------------------------------------------------------------------------


def test_credentials_need_both_variables() -> None:
    assert credentials_from_env({}) is None
    assert credentials_from_env({"SPACETRACK_USER": "a"}) is None
    assert credentials_from_env({"SPACETRACK_USER": " ", "SPACETRACK_PASS": "p"}) is None
    creds = credentials_from_env({"SPACETRACK_USER": "a@b.c", "SPACETRACK_PASS": "p"})
    assert creds == SpaceTrackCredentials("a@b.c", "p")


def test_password_is_not_in_repr() -> None:
    assert CREDS.password not in repr(CREDS)


def test_query_url_keeps_operators_encoded() -> None:
    url = query_url(
        "gp",
        [("OBJECT_NAME", "^STARLINK"), ("EPOCH", ">now-10"), ("OBJECT_TYPE", "DEBRIS,ROCKET BODY")],
        orderby="NORAD_CAT_ID",
        predicates=("NORAD_CAT_ID", "EPOCH"),
    )
    assert url == (
        "https://www.space-track.org/basicspacedata/query/class/gp"
        "/OBJECT_NAME/%5ESTARLINK/EPOCH/%3Enow-10/OBJECT_TYPE/DEBRIS,ROCKET%20BODY"
        "/orderby/NORAD_CAT_ID/predicates/NORAD_CAT_ID,EPOCH/format/json"
    )


# ---------------------------------------------------------------------------
# GP fetch, login, cache
# ---------------------------------------------------------------------------


def test_fetch_fleet_logs_in_once_and_labels_spacetrack(tmp_path: Path) -> None:
    session = FakeSession(gets=[FakeResponse(json.dumps([STARLINK_GP]))])
    catalog = _client(tmp_path, session).fetch_fleet()

    assert catalog.source == DataSource.SPACETRACK
    assert [obj.object_id for obj in catalog] == ["44714"]
    assert all(obj.data_source == DataSource.SPACETRACK for obj in catalog)
    assert catalog.objects[0].object_type == "PAYLOAD"
    assert session.posts == [
        {
            "url": "https://www.space-track.org/ajaxauth/login",
            "data": {"identity": CREDS.user, "password": CREDS.password},
        }
    ]
    url = session.urls[0]
    assert "/class/gp/OBJECT_NAME/%5ESTARLINK/" in url
    assert "/DECAY_DATE/null-val/" in url
    assert "/predicates/" in url and url.endswith("/format/json")


def test_fresh_cache_needs_no_login_or_network(tmp_path: Path) -> None:
    _client(tmp_path, FakeSession(gets=[FakeResponse(json.dumps([STARLINK_GP]))])).fetch_fleet()
    again = SpaceTrackClient(None, cache_dir=tmp_path, session=NoNetwork()).fetch_fleet()
    assert [obj.object_id for obj in again] == ["44714"]


def test_stale_cache_is_refetched(tmp_path: Path) -> None:
    _client(tmp_path, FakeSession(gets=[FakeResponse(json.dumps([STARLINK_GP]))])).fetch_fleet()
    session = FakeSession(gets=[FakeResponse("[]")])
    client = SpaceTrackClient(CREDS, cache_dir=tmp_path, session=session, cache_ttl_s=-1)
    assert len(client.fetch_fleet()) == 0
    assert len(session.urls) == 1


def test_debris_query_is_non_payload_leo_band(tmp_path: Path) -> None:
    session = FakeSession(gets=[FakeResponse("[]")])
    _client(tmp_path, session).fetch_debris()
    url = session.urls[0]
    assert "/OBJECT_TYPE/DEBRIS,ROCKET%20BODY,UNKNOWN/" in url
    assert "/PERIAPSIS/%3C1000/APOAPSIS/%3E200/" in url


def test_rejected_login_raises_without_leaking_credentials(tmp_path: Path) -> None:
    session = FakeSession(login=FakeResponse('{"Login":"Failed"}'))
    with pytest.raises(SpaceTrackAuthError) as caught:
        _client(tmp_path, session).fetch_fleet()
    assert CREDS.password not in str(caught.value)
    assert CREDS.user not in str(caught.value)
    assert session.urls == []


def test_rejected_login_is_not_retried_by_the_next_client(tmp_path: Path) -> None:
    # The console builds a client per request; a wrong password must not turn
    # every request into another failed login.
    session = FakeSession(login=FakeResponse('{"Login":"Failed"}'))
    with pytest.raises(SpaceTrackAuthError):
        _client(tmp_path, session).fetch_fleet()
    with pytest.raises(SpaceTrackAuthError, match="recently"):
        _client(tmp_path, session).fetch_fleet()
    assert len(session.posts) == 1


def test_login_server_error_is_transient_not_a_rejection(tmp_path: Path) -> None:
    session = FakeSession(login=FakeResponse("", 503), gets=[FakeResponse(json.dumps([STARLINK_GP]))])
    with pytest.raises(SpaceTrackError) as caught:
        _client(tmp_path, session).fetch_fleet()
    assert not isinstance(caught.value, SpaceTrackAuthError)
    session.login_response = FakeResponse("")
    assert len(_client(tmp_path, session).fetch_fleet()) == 1


def test_clients_share_one_throttle() -> None:
    assert SpaceTrackClient(CREDS, session=NoNetwork()).throttle is SpaceTrackClient(
        CREDS, session=NoNetwork()
    ).throttle


def test_concurrent_stale_requests_download_once(tmp_path: Path) -> None:
    import threading

    release = threading.Event()

    class SlowSession(FakeSession):
        def get(self, url, **kwargs):
            release.wait(5)
            return super().get(url, **kwargs)

    session = SlowSession(gets=[FakeResponse(json.dumps([STARLINK_GP]))])
    results: list[int] = []
    threads = [
        threading.Thread(target=lambda: results.append(len(_client(tmp_path, session).fetch_fleet())))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(10)
    assert results == [1, 1, 1, 1]
    assert len(session.urls) == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_missing_credentials_raise_before_any_request(tmp_path: Path) -> None:
    with pytest.raises(SpaceTrackAuthError):
        SpaceTrackClient(None, cache_dir=tmp_path, session=NoNetwork()).fetch_fleet()


def test_error_payload_raises_and_is_not_cached(tmp_path: Path) -> None:
    session = FakeSession(gets=[FakeResponse('{"error":"bad predicate"}')])
    with pytest.raises(SpaceTrackError, match="bad predicate"):
        _client(tmp_path, session).fetch_fleet()
    assert not any(tmp_path.iterdir())


def test_expired_session_logs_in_again_once(tmp_path: Path) -> None:
    session = FakeSession(
        gets=[
            FakeResponse('{"error":"You must be logged in to complete this action"}', 401),
            FakeResponse(json.dumps([STARLINK_GP])),
        ]
    )
    assert len(_client(tmp_path, session).fetch_fleet()) == 1
    assert len(session.posts) == 2


def test_http_429_is_not_retried(tmp_path: Path) -> None:
    session = FakeSession(gets=[FakeResponse("slow down", 429), FakeResponse("[]")])
    with pytest.raises(SpaceTrackError, match="429"):
        _client(tmp_path, session).fetch_fleet()
    assert len(session.urls) == 1


def test_server_errors_retry_then_fail(tmp_path: Path) -> None:
    session = FakeSession(gets=[FakeResponse("", 503), FakeResponse("", 503)])
    with pytest.raises(SpaceTrackError, match="503"):
        _client(tmp_path, session).fetch_fleet()


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


def test_throttle_waits_out_the_minute_and_refuses_past_the_hour() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    throttle = RequestThrottle(2, 3, clock=lambda: now[0], sleep=sleep)
    throttle.acquire()
    now[0] = 10.0
    throttle.acquire()
    throttle.acquire()
    assert slept == [50.0]
    with pytest.raises(SpaceTrackError, match="hourly"):
        throttle.acquire()
    now[0] = 3600.0
    throttle.acquire()


# ---------------------------------------------------------------------------
# cdm_public
# ---------------------------------------------------------------------------


def test_parse_cdm_public() -> None:
    first, second = parse_cdm_public(json.dumps(CDM_PUBLIC))
    assert first.miss_distance_km == pytest.approx(0.187)
    assert first.collision_probability == pytest.approx(2.13e-4)
    assert first.emergency_reportable is True
    assert first.tca.tzinfo is not None and first.tca.hour == 10
    assert first.involves("STARLINK") and not second.involves("STARLINK")
    assert second.collision_probability is None


def test_cdm_public_rejects_bad_probability() -> None:
    record = dict(CDM_PUBLIC[0], PC="1.5")
    with pytest.raises(SpaceTrackError):
        parse_cdm_public(json.dumps([record]))


def test_fetch_public_conjunctions_queries_future_tcas(tmp_path: Path) -> None:
    session = FakeSession(gets=[FakeResponse(json.dumps(CDM_PUBLIC))])
    events = _client(tmp_path, session).fetch_public_conjunctions()
    assert len(events) == 2
    assert "/class/cdm_public/TCA/%3Enow/orderby/TCA/format/json" in session.urls[0]


# ---------------------------------------------------------------------------
# Data-source wall
# ---------------------------------------------------------------------------


def _as_spacetrack(catalog: Catalog) -> Catalog:
    for obj in catalog.objects:
        obj.data_source = DataSource.SPACETRACK
    return Catalog(source=DataSource.SPACETRACK, objects=catalog.objects, query=catalog.query)


def test_spacetrack_and_celestrak_catalogs_never_merge() -> None:
    spacetrack = _as_spacetrack(load_starlink_slice(max_objects=2))
    with pytest.raises(MixedDataSourceError):
        spacetrack.merge(load_debris_slice(max_objects=2))


def test_pipeline_spacetrack_failure_never_falls_through_to_synthetic(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="Space-Track"):
        run_pipeline(source=DataSource.SPACETRACK, session=NoNetwork(), cache_dir=tmp_path)


def test_pipeline_rejects_synthetic_arguments_on_spacetrack() -> None:
    from aegis.ingest.synthetic import SyntheticSpec

    with pytest.raises(PipelineError):
        run_pipeline(source=DataSource.SPACETRACK, synthetic_spec=SyntheticSpec())


def test_importing_spacetrack_does_not_load_synthetic() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; importlib.import_module('aegis.ingest.spacetrack'); "
                "assert 'aegis.ingest.synthetic' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": str(PACKAGE_ROOT / "src")},
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


# ---------------------------------------------------------------------------
# Ops catalog source preference
# ---------------------------------------------------------------------------


def _fake_celestrak(calls: list[str]):
    def fake_fetch(group: str, **_kwargs):
        calls.append(group)
        return load_starlink_slice() if group == "starlink" else load_debris_slice()

    return fake_fetch


def test_ops_prefers_spacetrack_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("SPACETRACK_USER", CREDS.user)
    monkeypatch.setenv("SPACETRACK_PASS", CREDS.password)
    monkeypatch.setattr(
        SpaceTrackClient, "fetch_fleet", lambda self: _as_spacetrack(load_starlink_slice())
    )
    monkeypatch.setattr(
        SpaceTrackClient, "fetch_debris", lambda self: _as_spacetrack(load_debris_slice())
    )
    calls: list[str] = []
    monkeypatch.setattr("aegis.ingest.celestrak.fetch_celestrak", _fake_celestrak(calls))

    catalog, fallback = load_ops_catalog(live=True, max_objects=8)

    assert catalog.source == DataSource.SPACETRACK
    assert fallback is None
    assert {obj.metadata.get("catalog_role") for obj in catalog} == {"fleet", "debris"}
    assert calls == []


def test_ops_falls_back_to_celestrak_when_spacetrack_fails(monkeypatch) -> None:
    monkeypatch.setenv("SPACETRACK_USER", CREDS.user)
    monkeypatch.setenv("SPACETRACK_PASS", CREDS.password)

    def refuse(self):
        raise SpaceTrackError("down")

    monkeypatch.setattr(SpaceTrackClient, "fetch_fleet", refuse)
    calls: list[str] = []
    monkeypatch.setattr("aegis.ingest.celestrak.fetch_celestrak", _fake_celestrak(calls))

    catalog, fallback = load_ops_catalog(live=True, max_objects=8)

    assert catalog.source == DataSource.CELESTRAK
    assert fallback is None
    assert len(calls) == 2


def test_ops_without_credentials_never_builds_a_spacetrack_client(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Space-Track used without credentials")

    monkeypatch.setattr(SpaceTrackClient, "__init__", forbidden)
    calls: list[str] = []
    monkeypatch.setattr("aegis.ingest.celestrak.fetch_celestrak", _fake_celestrak(calls))

    catalog, _fallback = load_ops_catalog(live=True, max_objects=8)
    assert catalog.source == DataSource.CELESTRAK


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_cli_without_credentials_exits_2(capsys) -> None:
    assert main(["check"]) == 2
    assert "SPACETRACK_USER" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["check", "refresh"])
def test_cli_exit_codes_and_no_password_in_output(command, monkeypatch, capsys, tmp_path: Path) -> None:
    # aegis-refresh.service relies on 0 = ok, 1 = Space-Track failed.
    from aegis.ingest import spacetrack

    monkeypatch.setenv("SPACETRACK_USER", CREDS.user)
    monkeypatch.setenv("SPACETRACK_PASS", CREDS.password)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    gp = FakeResponse(json.dumps([STARLINK_GP]))
    gp_queries = [gp, gp] if command == "refresh" else [gp]  # refresh adds the debris query
    session = FakeSession(gets=[*gp_queries, FakeResponse(json.dumps(CDM_PUBLIC))])
    real = spacetrack.SpaceTrackClient
    monkeypatch.setattr(
        spacetrack, "SpaceTrackClient", lambda credentials, **kw: real(credentials, session=session, **kw)
    )
    assert main([command]) == 0
    out = capsys.readouterr()
    assert "cdm_public: 2 upcoming events" in out.out
    assert CREDS.password not in out.out + out.err

    failing = FakeSession(gets=[FakeResponse("", 503), FakeResponse("", 503)])
    monkeypatch.setattr(
        spacetrack, "SpaceTrackClient", lambda credentials, **kw: real(credentials, session=failing, cache_ttl_s=-1)
    )
    assert main([command]) == 1
    out = capsys.readouterr()
    assert "503" in out.err and CREDS.password not in out.out + out.err


# ---------------------------------------------------------------------------
# Event grouping and GP by catalog number
# ---------------------------------------------------------------------------


def _cdm(cdm_id: str, created: str, tca: str, sat1: str, sat2: str, miss_m: str = "200"):
    return {
        "CDM_ID": cdm_id,
        "CREATED": created,
        "EMERGENCY_REPORTABLE": "Y",
        "TCA": tca,
        "MIN_RNG": miss_m,
        "PC": "0.0002",
        "SAT_1_ID": sat1,
        "SAT_1_NAME": f"OBJ {sat1}",
        "SAT1_OBJECT_TYPE": "DEBRIS",
        "SAT_2_ID": sat2,
        "SAT_2_NAME": f"OBJ {sat2}",
        "SAT2_OBJECT_TYPE": "DEBRIS",
    }


def test_latest_per_event_merges_updates_and_swapped_sides() -> None:
    from aegis.ingest.spacetrack import latest_per_event

    rows = parse_cdm_public(
        json.dumps(
            [
                _cdm("1", "2026-09-18 01:00:00", "2026-09-19T10:00:00.000", "100", "200", "300"),
                _cdm("2", "2026-09-18 01:00:00", "2026-09-19T10:00:00.000", "200", "100", "300"),
                _cdm("3", "2026-09-18 09:00:00", "2026-09-19T10:00:00.400", "100", "200", "150"),
                # Same pair, next orbit: a separate event.
                _cdm("4", "2026-09-18 09:00:00", "2026-09-19T11:35:00.000", "100", "200"),
                _cdm("5", "2026-09-18 09:00:00", "2026-09-19T10:00:00.000", "100", "300"),
            ]
        )
    )
    events = latest_per_event(rows)
    assert [(event.cdm_id, count) for event, count in events] == [("3", 3), ("5", 1), ("4", 1)]
    assert events[0][0].miss_distance_km == pytest.approx(0.150)


def test_fetch_objects_queries_ids_in_chunks_and_prunes_old_sets(tmp_path: Path) -> None:
    (tmp_path / "gp_ids_oldset.json").write_text("[]")
    ids = [str(n) for n in range(1, 302)]
    session = FakeSession(gets=[FakeResponse(json.dumps([STARLINK_GP])), FakeResponse("[]")])
    catalog = _client(tmp_path, session).fetch_objects(ids)

    assert len(session.urls) == 2
    assert "/NORAD_CAT_ID/1,2,3," in session.urls[0]
    assert "/NORAD_CAT_ID/251,252," in session.urls[1]
    assert "/EPOCH/%3E" not in session.urls[0]  # no age filter: any epoch is usable
    assert [obj.object_id for obj in catalog] == ["44714"]
    assert not (tmp_path / "gp_ids_oldset.json").exists()
    assert len(list(tmp_path.glob("gp_ids_*.json"))) == 1


def test_fetch_objects_with_no_ids_makes_no_request(tmp_path: Path) -> None:
    assert len(_client(tmp_path, NoNetwork()).fetch_objects([])) == 0


# ---------------------------------------------------------------------------
# .env loading and pipeline default source
# ---------------------------------------------------------------------------


def test_env_file_sets_only_unset_variables(tmp_path: Path, monkeypatch) -> None:
    from aegis.envfile import load_env_file

    monkeypatch.setenv("AEGIS_DOTENV", "1")
    monkeypatch.setenv("SPACETRACK_USER", "from-shell")
    # Register SPACETRACK_PASS with monkeypatch so the value the file sets is
    # removed again at teardown.
    monkeypatch.setenv("SPACETRACK_PASS", "placeholder")
    monkeypatch.delenv("SPACETRACK_PASS")
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nSPACETRACK_USER='from-file'\nexport SPACETRACK_PASS=\"p a'ss\"\nnot a line\n"
    )
    assert load_env_file(env) == ["SPACETRACK_PASS"]
    import os

    assert os.environ["SPACETRACK_USER"] == "from-shell"
    assert os.environ["SPACETRACK_PASS"] == "p a'ss"


def test_env_file_can_be_disabled_and_may_be_missing(tmp_path: Path, monkeypatch) -> None:
    from aegis.envfile import load_env_file

    env = tmp_path / ".env"
    env.write_text("SPACETRACK_USER=x\n")
    assert load_env_file(env) == []  # conftest sets AEGIS_DOTENV=0
    monkeypatch.setenv("AEGIS_DOTENV", "1")
    assert load_env_file(tmp_path / "absent.env") == []


def test_pipeline_cli_defaults_to_spacetrack_only_with_credentials(monkeypatch) -> None:
    from aegis.pipeline import cli

    seen: list[str] = []

    def fake_run_pipeline(*, source, **_kwargs):
        seen.append(source)
        raise PipelineError("stop here")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    assert cli.main([]) == 1
    monkeypatch.setenv("SPACETRACK_USER", CREDS.user)
    monkeypatch.setenv("SPACETRACK_PASS", CREDS.password)
    assert cli.main([]) == 1
    assert cli.main(["--source", "celestrak"]) == 1
    assert cli.main(["--tle-path", "fleet.tle"]) == 1  # a local file implies CelesTrak
    assert seen == [DataSource.CELESTRAK, DataSource.SPACETRACK, DataSource.CELESTRAK, DataSource.CELESTRAK]


def test_pipeline_spacetrack_empty_name_match_is_an_error(tmp_path: Path, monkeypatch) -> None:
    from aegis.pipeline.run import _ingest_spacetrack

    monkeypatch.setenv("SPACETRACK_USER", CREDS.user)
    monkeypatch.setenv("SPACETRACK_PASS", CREDS.password)
    session = FakeSession(gets=[FakeResponse("[]")])
    with pytest.raises(PipelineError, match="--source CELESTRAK"):
        _ingest_spacetrack("gps-ops", session, tmp_path)
