"""CelesTrak GP / SupGP client.

This is the real acquisition path. It does not import the synthetic generator,
does not consult ``AEGIS_ALLOW_SYNTHETIC``, and writes cache files only under
a ``celestrak`` directory. Tests inject a fake ``session`` so the client never
requires the network when one is supplied.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from ..constants import (
    CELESTRAK_CACHE_TTL_S,
    CELESTRAK_MAX_RETRIES,
    CELESTRAK_USER_AGENT,
)
from ..core.objects import ObjectType, OrbitalElements, SpaceObject
from ..core.timebase import parse_epoch, utc_now
from .sources import Catalog, DataSource

__all__ = ["CelesTrakClient", "CelesTrakError", "catalog_from_tle_file", "fetch_celestrak"]

_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
_SUP_GP_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php"
_REQUEST_TIMEOUT_S = 30.0
_CACHE_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class CelesTrakError(Exception):
    """Network or parse failure on the CelesTrak path."""


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "aegis" / "celestrak"


def _tle_float(raw: str) -> float:
    text = raw.strip()
    if not text:
        return 0.0
    if text[0] == ".":
        text = "0" + text
    elif text.startswith("+.") or text.startswith("-."):
        text = text[0] + "0" + text[1:]
    return float(text)


def _tle_scientific(raw: str) -> float:
    """Parse a TLE implied-decimal scientific field (BSTAR, n-ddot/6)."""
    text = raw.strip()
    if not text:
        return 0.0
    match = re.fullmatch(r"([+-]?)(\d+)([+-]\d+)", text)
    if match:
        sign, digits, exponent = match.group(1), match.group(2), int(match.group(3))
        value = float("0." + digits) * (10 ** exponent)
        return -value if sign == "-" else value
    try:
        return _tle_float(text)
    except ValueError as error:
        raise CelesTrakError(f"unrecognised TLE scientific field: {raw!r}") from error


def _tle_int(raw: str, default: int = 0) -> int:
    text = raw.strip()
    if not text:
        return default
    return int(text)


def _parse_tle_epoch(field: str) -> datetime:
    text = field.strip()
    if len(text) < 5:
        raise CelesTrakError(f"TLE epoch too short: {field!r}")
    year = int(text[0:2])
    year += 2000 if year < 57 else 1900
    day_fraction = float(text[2:])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_fraction - 1)


def _international_designator(field: str) -> str:
    text = field.strip()
    if len(text) < 5:
        return text
    try:
        year_yy = int(text[0:2])
    except ValueError:
        return text
    year = 2000 + year_yy if year_yy < 57 else 1900 + year_yy
    launch = text[2:5]
    piece = text[5:].strip()
    return f"{year:04d}-{launch}{piece}"


def _is_tle_line(line: str, number: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(f"{number} ") and len(stripped) >= 68


def _verbatim_tle(line: str) -> str:
    return line.rstrip("\r\n")


def _space_object_from_tle(name: str, line1: str, line2: str) -> SpaceObject:
    parse1 = line1.strip()
    parse2 = line2.strip()
    if len(parse1) < 69 or len(parse2) < 69:
        # Permit 68-char lines missing only the checksum.
        if len(parse1) < 68 or len(parse2) < 68:
            raise CelesTrakError("TLE line shorter than the standard 68-character format")
    try:
        norad = str(int(parse1[2:7]))
        intl = _international_designator(parse1[9:17])
        epoch = _parse_tle_epoch(parse1[18:32])
        mean_motion_dot = _tle_float(parse1[33:43])
        mean_motion_ddot = _tle_scientific(parse1[44:52])
        bstar = _tle_scientific(parse1[53:61])
        element_set_number = _tle_int(parse1[64:68])
        inclination = _tle_float(parse2[8:16])
        raan = _tle_float(parse2[17:25])
        eccentricity = _tle_float("0." + parse2[26:33].strip())
        arg_perigee = _tle_float(parse2[34:42])
        mean_anomaly = _tle_float(parse2[43:51])
        mean_motion = _tle_float(parse2[52:63])
        revolution_number = _tle_int(parse2[63:68])
    except (ValueError, IndexError) as error:
        raise CelesTrakError(f"failed to parse TLE for {name or 'unnamed'}: {error}") from error

    elements = OrbitalElements(
        epoch=epoch,
        mean_motion_rev_per_day=mean_motion,
        eccentricity=eccentricity,
        inclination_deg=inclination,
        raan_deg=raan,
        arg_perigee_deg=arg_perigee,
        mean_anomaly_deg=mean_anomaly,
        bstar=bstar,
        mean_motion_dot=mean_motion_dot,
        mean_motion_ddot=mean_motion_ddot,
        element_set_number=element_set_number,
        revolution_number=revolution_number,
    )
    return SpaceObject(
        object_id=norad,
        name=name or norad,
        international_designator=intl,
        object_type=ObjectType.UNKNOWN,
        elements=elements,
        operator=None,
        tle_line1=line1,
        tle_line2=line2,
        data_source=DataSource.CELESTRAK,
    )


def _parse_tle_text(text: str) -> list[SpaceObject]:
    lines = text.splitlines()
    objects: list[SpaceObject] = []
    index = 0
    count = len(lines)
    while index < count:
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if _is_tle_line(line, "1"):
            if index + 1 >= count or not _is_tle_line(lines[index + 1], "2"):
                raise CelesTrakError("TLE line 1 is not followed by line 2")
            objects.append(
                _space_object_from_tle("", _verbatim_tle(line), _verbatim_tle(lines[index + 1]))
            )
            index += 2
            continue
        if _is_tle_line(line, "2"):
            raise CelesTrakError("TLE line 2 without a preceding line 1")
        name = line.strip()
        if name.startswith("0 "):
            name = name[2:].strip()
        index += 1
        while index < count and not lines[index].strip():
            index += 1
        if index >= count or not _is_tle_line(lines[index], "1"):
            raise CelesTrakError(f"name line {name!r} is not followed by TLE line 1")
        line1 = _verbatim_tle(lines[index])
        index += 1
        while index < count and not lines[index].strip():
            index += 1
        if index >= count or not _is_tle_line(lines[index], "2"):
            raise CelesTrakError(f"TLE line 1 for {name!r} is not followed by line 2")
        line2 = _verbatim_tle(lines[index])
        index += 1
        objects.append(_space_object_from_tle(name, line1, line2))
    return objects


def _optional_float(record: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in record and record[key] is not None and record[key] != "":
            return float(record[key])
    return default


def _optional_int(record: dict, *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in record and record[key] is not None and record[key] != "":
            return int(record[key])
    return default


def _space_object_from_omm(record: dict) -> SpaceObject:
    try:
        norad = str(int(record["NORAD_CAT_ID"]))
        name = str(record.get("OBJECT_NAME") or norad)
        intl = str(record.get("OBJECT_ID") or "")
        epoch = parse_epoch(str(record["EPOCH"]))
        elements = OrbitalElements(
            epoch=epoch,
            mean_motion_rev_per_day=float(record["MEAN_MOTION"]),
            eccentricity=float(record["ECCENTRICITY"]),
            inclination_deg=float(record["INCLINATION"]),
            raan_deg=float(record["RA_OF_ASC_NODE"]),
            arg_perigee_deg=float(record["ARG_OF_PERICENTER"]),
            mean_anomaly_deg=float(record["MEAN_ANOMALY"]),
            bstar=_optional_float(record, "BSTAR"),
            mean_motion_dot=_optional_float(record, "MEAN_MOTION_DOT"),
            mean_motion_ddot=_optional_float(record, "MEAN_MOTION_DDOT"),
            element_set_number=_optional_int(record, "ELEMENT_SET_NO", "ELEMENT_SET_NUMBER"),
            revolution_number=_optional_int(record, "REV_AT_EPOCH", "REVOLUTION_NUMBER"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CelesTrakError(f"failed to parse OMM JSON record: {error}") from error

    object_type = str(record.get("OBJECT_TYPE") or ObjectType.UNKNOWN)
    if object_type not in ObjectType.ALL:
        object_type = ObjectType.UNKNOWN

    line1 = str(record.get("TLE_LINE1") or "").rstrip("\r\n")
    line2 = str(record.get("TLE_LINE2") or "").rstrip("\r\n")
    return SpaceObject(
        object_id=norad,
        name=name,
        international_designator=intl,
        object_type=object_type,
        elements=elements,
        operator=None,
        tle_line1=line1,
        tle_line2=line2,
        data_source=DataSource.CELESTRAK,
    )


def _parse_omm_json(text: str) -> list[SpaceObject]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise CelesTrakError(f"CelesTrak JSON is not valid: {error}") from error
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise CelesTrakError("CelesTrak JSON must be a list of OMM records")
    return [_space_object_from_omm(record) for record in payload]


def _response_body(response: object) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return text
    content = getattr(response, "content", None)
    if content is None:
        raise CelesTrakError("session response has neither text nor content")
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)


def _response_status(response: object) -> int:
    status = getattr(response, "status_code", None)
    if status is None:
        raise CelesTrakError("session response has no status_code")
    return int(status)


class CelesTrakClient:
    """Fetch GP or supplemental GP catalogs from CelesTrak, with a file cache."""

    def __init__(self, cache_dir=None, session=None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def fetch_gp(self, group: str, *, fmt: str = "tle", supplement: bool = False) -> Catalog:
        """Download (or reuse a fresh cache of) one CelesTrak group.

        Parameters
        ----------
        group
            CelesTrak ``GROUP`` (GP) or ``FILE`` (SupGP) name.
        fmt
            ``tle`` for 2-line / 3-line element sets, ``json`` for OMM JSON.
        supplement
            When true, use the supplemental GP endpoint.
        """
        body, fetched_at = self._load_or_fetch(group, fmt=fmt, supplement=supplement)
        objects = self._parse_body(body, fmt=fmt)
        kind = "supplemental" if supplement else "gp"
        return Catalog(
            source=DataSource.CELESTRAK,
            objects=objects,
            fetched_at=fetched_at,
            query=f"celestrak {kind} group={group} fmt={fmt}",
        )

    def _cache_path(self, group: str, fmt: str, supplement: bool) -> Path:
        kind = "supgp" if supplement else "gp"
        safe_group = _CACHE_NAME_SAFE.sub("_", group)
        safe_fmt = _CACHE_NAME_SAFE.sub("_", fmt)
        return self.cache_dir / f"{safe_group}_{safe_fmt}_{kind}.cache"

    def _read_fresh_cache(self, path: Path) -> tuple[str, datetime] | None:
        if not path.is_file():
            return None
        age_s = time.time() - path.stat().st_mtime
        if age_s > CELESTRAK_CACHE_TTL_S:
            return None
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return path.read_text(encoding="utf-8"), fetched_at

    def _write_cache(self, path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def _url(self, group: str, fmt: str, supplement: bool) -> str:
        if supplement:
            return f"{_SUP_GP_URL}?FILE={quote(group, safe='')}&FORMAT={quote(fmt, safe='')}"
        return f"{_GP_URL}?GROUP={quote(group, safe='')}&FORMAT={quote(fmt, safe='')}"

    def _fetch_body(self, url: str) -> str:
        headers = {"User-Agent": CELESTRAK_USER_AGENT}
        last_error: Exception | None = None
        for _attempt in range(CELESTRAK_MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=_REQUEST_TIMEOUT_S, headers=headers)
            except CelesTrakError:
                raise
            except Exception as error:  # noqa: BLE001 - session may raise anything
                last_error = CelesTrakError(f"CelesTrak request failed for {url}: {error}")
                continue
            try:
                status = _response_status(response)
            except CelesTrakError as error:
                last_error = error
                continue
            if status >= 400:
                last_error = CelesTrakError(f"CelesTrak HTTP {status} for {url}")
                continue
            try:
                return _response_body(response)
            except CelesTrakError as error:
                last_error = error
                continue
        if last_error is not None:
            raise last_error
        raise CelesTrakError(f"CelesTrak request failed for {url}")

    def _load_or_fetch(self, group: str, *, fmt: str, supplement: bool) -> tuple[str, datetime]:
        path = self._cache_path(group, fmt, supplement)
        cached = self._read_fresh_cache(path)
        if cached is not None:
            return cached
        body = self._fetch_body(self._url(group, fmt, supplement))
        self._write_cache(path, body)
        return body, utc_now()

    @staticmethod
    def _parse_body(body: str, *, fmt: str) -> list[SpaceObject]:
        stripped = body.strip()
        if not stripped:
            return []
        if fmt.lower() == "json":
            return _parse_omm_json(stripped)
        return _parse_tle_text(body)


def catalog_from_tle_file(path: str | Path) -> Catalog:
    """Build a CelesTrak catalog from a local TLE file.

    Reads 2-line or 3-line element sets. Does not open HTTP and does not
    call the synthetic generator. Empty or invalid input raises
    :class:`CelesTrakError`.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise CelesTrakError(f"cannot read TLE file {file_path}: {error}") from error
    objects = _parse_tle_text(text)
    if not objects:
        raise CelesTrakError(f"TLE file {file_path} produced no objects")
    return Catalog(
        source=DataSource.CELESTRAK,
        objects=objects,
        fetched_at=utc_now(),
        query=f"tle_file {file_path.name}",
    )


def fetch_celestrak(
    group: str,
    *,
    fmt: str = "tle",
    supplement: bool = False,
    session=None,
    cache_dir=None,
) -> Catalog:
    """Fetch a CelesTrak GP or supplemental GP catalog.

    Does not consult ``AEGIS_ALLOW_SYNTHETIC``. When ``session`` is supplied
    the client does not open its own HTTP connection.
    """
    client = CelesTrakClient(cache_dir=cache_dir, session=session)
    return client.fetch_gp(group, fmt=fmt, supplement=supplement)
