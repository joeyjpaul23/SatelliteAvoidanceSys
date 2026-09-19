"""Space-Track.org GP and public-CDM client.

The login-gated real acquisition path, alongside ``celestrak``. It never
imports the synthetic generator. Credentials come from ``SPACETRACK_USER`` and
``SPACETRACK_PASS`` and are never written to the cache, printed, or embedded
in error messages.

Space-Track suspends accounts that exceed its API throttle, so every request
passes through :class:`RequestThrottle`, and GP / CDM responses are cached on
disk under ``$XDG_CACHE_HOME/aegis/spacetrack``. The console builds a client
per request, so every client in a process shares one throttle, one download
lock (concurrent requests on a stale cache wait for a single download) and a
back-off after a rejected login. The cache is what stops the console and the
scheduled refresh, separate processes, from both downloading in the same
hour. Tests inject a fake ``session`` so the client never needs the network.

``python -m aegis.ingest.spacetrack check`` verifies credentials and
``python -m aegis.ingest.spacetrack refresh`` warms the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from ..constants import (
    CELESTRAK_USER_AGENT,
    SPACETRACK_CACHE_TTL_S,
    SPACETRACK_DEBRIS_APOGEE_MIN_KM,
    SPACETRACK_DEBRIS_PERIGEE_MAX_KM,
    SPACETRACK_FLEET_PREFIX,
    SPACETRACK_MAX_EPOCH_AGE_DAYS,
    SPACETRACK_MAX_PER_HOUR,
    SPACETRACK_MAX_PER_MINUTE,
    SPACETRACK_MAX_RETRIES,
    SPACETRACK_REFRESH_AGE_S,
)
from ..core.objects import ObjectType
from ..core.timebase import ensure_utc, parse_epoch, utc_now
from ..envfile import load_env_file
from .celestrak import CelesTrakError, _space_object_from_omm
from .sources import Catalog, DataSource

__all__ = [
    "PublicConjunction",
    "RequestThrottle",
    "SpaceTrackAuthError",
    "SpaceTrackClient",
    "SpaceTrackCredentials",
    "SpaceTrackError",
    "credentials_from_env",
    "latest_per_event",
    "parse_cdm_public",
    "query_url",
]

_BASE_URL = "https://www.space-track.org"
_LOGIN_URL = f"{_BASE_URL}/ajaxauth/login"
_QUERY_URL = f"{_BASE_URL}/basicspacedata/query"
# A full debris-band GP pull is tens of megabytes.
_REQUEST_TIMEOUT_S = 120.0
_ENV_USER = "SPACETRACK_USER"
_ENV_PASS = "SPACETRACK_PASS"

# Only the OMM fields the parser reads; roughly halves the GP payload.
_GP_PREDICATES = (
    "NORAD_CAT_ID",
    "OBJECT_NAME",
    "OBJECT_ID",
    "OBJECT_TYPE",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "TLE_LINE1",
    "TLE_LINE2",
)
_DEBRIS_TYPES = (ObjectType.DEBRIS, ObjectType.ROCKET_BODY, ObjectType.UNKNOWN)

# cdm_public publishes MIN_RNG in metres (verified live: 11 m to ~5 km).
_MIN_RNG_TO_KM = 1.0e-3

# Catalog numbers per GP-by-ID request; keeps each URL well under 2 KB.
_IDS_PER_REQUEST = 250


class SpaceTrackError(Exception):
    """Network, throttle, or parse failure on the Space-Track path."""


class SpaceTrackAuthError(SpaceTrackError):
    """Credentials are missing or Space-Track rejected them."""


@dataclass(frozen=True)
class SpaceTrackCredentials:
    user: str
    password: str = field(repr=False)


def credentials_from_env(environ=None) -> SpaceTrackCredentials | None:
    """Read ``SPACETRACK_USER`` / ``SPACETRACK_PASS``; ``None`` if either is unset."""
    env = os.environ if environ is None else environ
    user = (env.get(_ENV_USER) or "").strip()
    password = env.get(_ENV_PASS) or ""
    if not user or not password:
        return None
    return SpaceTrackCredentials(user=user, password=password)


class RequestThrottle:
    """Sliding-window limiter below Space-Track's per-minute and per-hour caps.

    A full minute window blocks until the oldest request ages out. A full
    hour window raises instead: sleeping up to an hour inside a web request
    is worse than failing over to CelesTrak.
    """

    def __init__(
        self,
        per_minute: int = SPACETRACK_MAX_PER_MINUTE,
        per_hour: int = SPACETRACK_MAX_PER_HOUR,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if per_minute < 1 or per_hour < per_minute:
            raise ValueError("need 1 <= per_minute <= per_hour")
        self.per_minute = per_minute
        self.per_hour = per_hour
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            self._acquire()

    def _acquire(self) -> None:
        now = self._clock()
        while self._stamps and now - self._stamps[0] >= 3600.0:
            self._stamps.popleft()
        if len(self._stamps) >= self.per_hour:
            raise SpaceTrackError(
                f"Space-Track hourly request budget ({self.per_hour}) used up; "
                "refusing to risk an account suspension"
            )
        recent = [stamp for stamp in self._stamps if now - stamp < 60.0]
        if len(recent) >= self.per_minute:
            self._sleep(60.0 - (now - recent[0]))
            now = self._clock()
        self._stamps.append(now)


# Shared by every client in the process (see the module docstring).
_PROCESS_THROTTLE = RequestThrottle()
_DOWNLOAD_LOCK = threading.Lock()
# A rejected login is not retried with the same credentials for this long, so
# console polling with a wrong password can't pile up failed logins.
_AUTH_BACKOFF_S = 900.0
_auth_rejected_at: dict[SpaceTrackCredentials, float] = {}


def query_url(
    request_class: str,
    filters: Sequence[tuple[str, str]] = (),
    *,
    orderby: str | None = None,
    predicates: Sequence[str] = (),
    limit: int | None = None,
) -> str:
    """Build a ``basicspacedata/query`` URL returning JSON.

    Filter values keep Space-Track's operators (``>now-10``, ``null-val``,
    ``^STARLINK``, ``A,B``); each path segment is percent-encoded, which the
    server decodes before applying them.
    """
    parts = ["class", request_class]
    for key, value in filters:
        parts += [key, value]
    if orderby:
        parts += ["orderby", orderby]
    if predicates:
        parts += ["predicates", ",".join(predicates)]
    if limit is not None:
        parts += ["limit", str(int(limit))]
    parts += ["format", "json"]
    return _QUERY_URL + "/" + "/".join(quote(part, safe="-_.,~") for part in parts)


@dataclass(frozen=True)
class PublicConjunction:
    """One row of Space-Track's ``cdm_public`` class.

    The public class carries screening results only: no state vectors and
    no covariance, so it can be matched against but not re-assessed.
    """

    cdm_id: str
    created: datetime
    tca: datetime
    miss_distance_km: float
    collision_probability: float | None
    emergency_reportable: bool
    norad_id_1: str
    object_name_1: str
    object_type_1: str
    norad_id_2: str
    object_name_2: str
    object_type_2: str

    def involves(self, name_prefix: str) -> bool:
        prefix = name_prefix.upper()
        return self.object_name_1.upper().startswith(prefix) or self.object_name_2.upper().startswith(
            prefix
        )


def _cdm_float(record: dict, key: str) -> float | None:
    value = record.get(key)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _cdm_record(record: dict) -> PublicConjunction:
    try:
        miss_m = _cdm_float(record, "MIN_RNG")
        if miss_m is None:
            raise ValueError("MIN_RNG is empty")
        pc = _cdm_float(record, "PC")
        if pc is not None and not 0.0 <= pc <= 1.0:
            raise ValueError(f"PC {pc} outside [0, 1]")
        return PublicConjunction(
            cdm_id=str(record["CDM_ID"]),
            created=ensure_utc(parse_epoch(str(record["CREATED"]))),
            tca=ensure_utc(parse_epoch(str(record["TCA"]))),
            miss_distance_km=miss_m * _MIN_RNG_TO_KM,
            collision_probability=pc,
            emergency_reportable=str(record.get("EMERGENCY_REPORTABLE") or "").upper() == "Y",
            norad_id_1=str(int(record["SAT_1_ID"])),
            object_name_1=str(record.get("SAT_1_NAME") or ""),
            object_type_1=str(record.get("SAT1_OBJECT_TYPE") or ObjectType.UNKNOWN),
            norad_id_2=str(int(record["SAT_2_ID"])),
            object_name_2=str(record.get("SAT_2_NAME") or ""),
            object_type_2=str(record.get("SAT2_OBJECT_TYPE") or ObjectType.UNKNOWN),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SpaceTrackError(f"invalid cdm_public record: {error}") from error


def _json_records(body: str, what: str) -> list[dict]:
    try:
        payload = json.loads(body) if body.strip() else []
    except json.JSONDecodeError as error:
        raise SpaceTrackError(f"Space-Track {what} response is not JSON: {error}") from error
    if isinstance(payload, dict) and "error" in payload:
        raise SpaceTrackError(f"Space-Track {what} query refused: {payload['error']}")
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise SpaceTrackError(f"Space-Track {what} response is not a list of records")
    return payload


def latest_per_event(
    conjunctions: Iterable[PublicConjunction], *, tca_tolerance_s: float = 60.0
) -> list[tuple[PublicConjunction, int]]:
    """Collapse ``cdm_public`` rows to one per conjunction event.

    The feed carries every CDM update for an event, each published once per
    side (SAT_1 and SAT_2 swapped). Rows for the same object pair whose TCAs
    fall within ``tca_tolerance_s`` of each other are one event. Returns each
    event's latest-created row (18 SDS's current estimate) with the number of
    rows that described it, ordered by TCA.
    """
    groups: list[tuple[frozenset[str], datetime, list[PublicConjunction]]] = []
    for row in sorted(conjunctions, key=lambda item: item.tca):
        pair = frozenset((row.norad_id_1, row.norad_id_2))
        for group_pair, group_tca, rows in groups:
            if group_pair == pair and abs((row.tca - group_tca).total_seconds()) <= tca_tolerance_s:
                rows.append(row)
                break
        else:
            groups.append((pair, row.tca, [row]))
    return [
        (max(rows, key=lambda item: (item.created, item.cdm_id)), len(rows))
        for _pair, _tca, rows in groups
    ]


def parse_cdm_public(body: str) -> tuple[PublicConjunction, ...]:
    """Parse a ``cdm_public`` JSON response."""
    return tuple(_cdm_record(record) for record in _json_records(body, "cdm_public"))


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "aegis" / "spacetrack"


def _response_status(response: object) -> int:
    status = getattr(response, "status_code", None)
    if status is None:
        raise SpaceTrackError("Space-Track response has no status code")
    return int(status)


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    raise SpaceTrackError("Space-Track response has no body")


def _is_logged_out(status: int, body: str) -> bool:
    return status == 401 or "must be logged in" in body[:500].lower()


class SpaceTrackClient:
    """Throttled, cached Space-Track GP and ``cdm_public`` client.

    Parameters
    ----------
    credentials
        Defaults to :func:`credentials_from_env`. Missing credentials raise
        :class:`SpaceTrackAuthError` on the first request that needs a login;
        fresh cache hits never need one.
    cache_ttl_s
        Maximum cache age served without downloading. The console uses the
        default; the scheduled refresh passes ``SPACETRACK_REFRESH_AGE_S``.
    """

    def __init__(
        self,
        credentials: SpaceTrackCredentials | None = None,
        *,
        cache_dir=None,
        session=None,
        throttle: RequestThrottle | None = None,
        cache_ttl_s: float = SPACETRACK_CACHE_TTL_S,
    ) -> None:
        self.credentials = credentials if credentials is not None else credentials_from_env()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.throttle = throttle if throttle is not None else _PROCESS_THROTTLE
        self.cache_ttl_s = float(cache_ttl_s)
        self._logged_in = False

    # -- HTTP ---------------------------------------------------------------

    def login(self) -> None:
        """Open an authenticated session (cookie held by ``session``)."""
        if self.credentials is None:
            raise SpaceTrackAuthError(f"set {_ENV_USER} and {_ENV_PASS} to use Space-Track")
        rejected_at = _auth_rejected_at.get(self.credentials)
        if rejected_at is not None and time.monotonic() - rejected_at < _AUTH_BACKOFF_S:
            raise SpaceTrackAuthError(
                "Space-Track rejected these credentials recently; "
                f"not retrying for {_AUTH_BACKOFF_S / 60:.0f} min"
            )
        self.throttle.acquire()
        try:
            response = self.session.post(
                _LOGIN_URL,
                data={"identity": self.credentials.user, "password": self.credentials.password},
                timeout=_REQUEST_TIMEOUT_S,
                headers={"User-Agent": CELESTRAK_USER_AGENT},
            )
        except Exception as error:
            raise SpaceTrackError(f"Space-Track login request failed: {type(error).__name__}") from error
        status = _response_status(response)
        body = _response_text(response)
        if status >= 500:
            raise SpaceTrackError(f"Space-Track login failed: HTTP {status}")
        if status >= 400 or "failed" in body[:200].lower():
            _auth_rejected_at[self.credentials] = time.monotonic()
            raise SpaceTrackAuthError(f"Space-Track rejected the login (HTTP {status})")
        _auth_rejected_at.pop(self.credentials, None)
        self._logged_in = True

    def _get(self, url: str) -> str:
        if not self._logged_in:
            self.login()
        relogged = False
        last_error: SpaceTrackError | None = None
        for _attempt in range(SPACETRACK_MAX_RETRIES):
            self.throttle.acquire()
            try:
                response = self.session.get(
                    url,
                    timeout=_REQUEST_TIMEOUT_S,
                    headers={"User-Agent": CELESTRAK_USER_AGENT},
                )
            except Exception as error:  # noqa: BLE001 - session may raise anything
                last_error = SpaceTrackError(f"Space-Track request failed: {error}")
                continue
            status = _response_status(response)
            body = _response_text(response)
            if _is_logged_out(status, body) and not relogged:
                # Sessions expire after about two hours of use.
                relogged = True
                self._logged_in = False
                self.login()
                last_error = SpaceTrackError("Space-Track session expired")
                continue
            if status == 429:
                raise SpaceTrackError("Space-Track throttled this account (HTTP 429); not retrying")
            if status >= 500:
                last_error = SpaceTrackError(f"Space-Track HTTP {status}")
                continue
            if status >= 400:
                raise SpaceTrackError(f"Space-Track HTTP {status} for {url}")
            return body
        raise last_error if last_error is not None else SpaceTrackError("Space-Track request failed")

    # -- cache --------------------------------------------------------------

    def _fresh_cache(self, path: Path, what: str) -> tuple[list[dict], datetime] | None:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        if time.time() - mtime > self.cache_ttl_s:
            return None
        fetched_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return _json_records(path.read_text(encoding="utf-8"), what), fetched_at

    def _cached_or_fetch(
        self, name: str, urls: Sequence[str], what: str
    ) -> tuple[list[dict], datetime]:
        path = self.cache_dir / f"{name}.json"
        cached = self._fresh_cache(path, what)
        if cached is not None:
            return cached
        with _DOWNLOAD_LOCK:
            # Another request may have downloaded it while this one waited.
            cached = self._fresh_cache(path, what)
            if cached is not None:
                return cached
            records: list[dict] = []
            for url in urls:
                records.extend(_json_records(self._get(url), what))
            # Validate before caching so an error payload is never served
            # later, and replace atomically because the console and refresh
            # share it.
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                encoding="utf-8", delete=False,
            ) as handle:
                handle.write(json.dumps(records))
            os.replace(handle.name, path)
        return records, utc_now()

    # -- queries ------------------------------------------------------------

    def _gp_catalog(self, name: str, filters: list[tuple[str, str]], label: str) -> Catalog:
        filters = filters + [
            ("DECAY_DATE", "null-val"),
            ("EPOCH", f">now-{SPACETRACK_MAX_EPOCH_AGE_DAYS}"),
        ]
        url = query_url("gp", filters, orderby="NORAD_CAT_ID", predicates=_GP_PREDICATES)
        records, fetched_at = self._cached_or_fetch(name, [url], "gp")
        return self._gp_records_catalog(records, fetched_at, label)

    @staticmethod
    def _gp_records_catalog(records: list[dict], fetched_at: datetime, label: str) -> Catalog:
        try:
            objects = [
                _space_object_from_omm(record, data_source=DataSource.SPACETRACK)
                for record in records
            ]
        except CelesTrakError as error:
            raise SpaceTrackError(str(error)) from error
        return Catalog(
            source=DataSource.SPACETRACK,
            objects=objects,
            fetched_at=fetched_at,
            query=f"spacetrack gp {label}",
        )

    def fetch_fleet(self, name_prefix: str = SPACETRACK_FLEET_PREFIX) -> Catalog:
        """On-orbit objects whose name starts with ``name_prefix``."""
        prefix = name_prefix.strip().upper()
        if not prefix:
            raise SpaceTrackError("fleet name prefix is empty")
        return self._gp_catalog(
            f"gp_name_{quote(prefix, safe='')}",
            [("OBJECT_NAME", f"^{prefix}")],
            f"name^={prefix}",
        )

    def fetch_debris(self) -> Catalog:
        """Non-payload objects whose orbit crosses the LEO band."""
        perigee_max = f"{SPACETRACK_DEBRIS_PERIGEE_MAX_KM:g}"
        apogee_min = f"{SPACETRACK_DEBRIS_APOGEE_MIN_KM:g}"
        return self._gp_catalog(
            "gp_leo_debris",
            [
                ("OBJECT_TYPE", ",".join(_DEBRIS_TYPES)),
                ("PERIAPSIS", f"<{perigee_max}"),
                ("APOAPSIS", f">{apogee_min}"),
            ],
            f"non-payload perigee<{perigee_max}km apogee>{apogee_min}km",
        )

    def fetch_objects(self, norad_ids: Iterable[str]) -> Catalog:
        """Latest GP for specific catalog numbers, whatever their epoch age.

        Cached per ID set; older ID-set caches are pruned so the directory
        holds only the latest.
        """
        ids = sorted({str(int(norad_id)) for norad_id in norad_ids}, key=int)
        label = f"ids ({len(ids)} objects)"
        if not ids:
            return Catalog(source=DataSource.SPACETRACK, objects=[], query=f"spacetrack gp {label}")
        digest = hashlib.sha1(",".join(ids).encode("ascii")).hexdigest()[:12]
        name = f"gp_ids_{digest}"
        urls = [
            query_url(
                "gp",
                [("NORAD_CAT_ID", ",".join(ids[start : start + _IDS_PER_REQUEST]))],
                orderby="NORAD_CAT_ID",
                predicates=_GP_PREDICATES,
            )
            for start in range(0, len(ids), _IDS_PER_REQUEST)
        ]
        records, fetched_at = self._cached_or_fetch(name, urls, "gp")
        for stale in self.cache_dir.glob("gp_ids_*.json"):
            if stale.stem != name:
                stale.unlink(missing_ok=True)
        return self._gp_records_catalog(records, fetched_at, label)

    def fetch_public_conjunctions(self) -> tuple[PublicConjunction, ...]:
        """Every ``cdm_public`` row with a TCA still in the future."""
        return self.fetch_public_conjunctions_with_time()[0]

    def fetch_public_conjunctions_with_time(self) -> tuple[tuple[PublicConjunction, ...], datetime]:
        """As :meth:`fetch_public_conjunctions`, plus when the feed was downloaded."""
        url = query_url("cdm_public", [("TCA", ">now")], orderby="TCA")
        records, fetched_at = self._cached_or_fetch("cdm_public_upcoming", [url], "cdm_public")
        return tuple(_cdm_record(record) for record in records), fetched_at


# -- command line ------------------------------------------------------------


def _describe(catalog: Catalog) -> str:
    epochs = [obj.elements.epoch for obj in catalog.objects if obj.elements is not None]
    newest = max(epochs).isoformat() if epochs else "n/a"
    return f"{len(catalog)} objects (newest epoch {newest}, fetched {catalog.fetched_at.isoformat()})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aegis.ingest.spacetrack",
        description="Verify Space-Track credentials or warm the Space-Track cache.",
    )
    parser.add_argument(
        "command",
        choices=("check", "refresh"),
        help="check: log in and summarise; refresh: re-download anything older than "
        f"{SPACETRACK_REFRESH_AGE_S // 60:.0f} minutes",
    )
    args = parser.parse_args(argv)

    load_env_file()
    credentials = credentials_from_env()
    if credentials is None:
        print(f"{_ENV_USER} and {_ENV_PASS} must both be set", file=sys.stderr)
        return 2

    ttl = SPACETRACK_REFRESH_AGE_S if args.command == "refresh" else SPACETRACK_CACHE_TTL_S
    client = SpaceTrackClient(credentials, cache_ttl_s=ttl)
    try:
        if args.command == "check":
            client.login()
            print(f"login ok as {credentials.user}")
        fleet = client.fetch_fleet()
        print(f"fleet  ({SPACETRACK_FLEET_PREFIX}): {_describe(fleet)}")
        if args.command == "refresh":
            print(f"debris (LEO band): {_describe(client.fetch_debris())}")
        conjunctions = client.fetch_public_conjunctions()
        events = [event for event, _rows in latest_per_event(conjunctions)]
        fleet_events = [event for event in events if event.involves(SPACETRACK_FLEET_PREFIX)]
        print(
            f"cdm_public: {len(events)} upcoming events ({len(conjunctions)} CDM rows), "
            f"{len(fleet_events)} involving {SPACETRACK_FLEET_PREFIX}"
        )
    except SpaceTrackError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
