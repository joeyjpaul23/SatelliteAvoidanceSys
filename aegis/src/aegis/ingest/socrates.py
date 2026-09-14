"""Free fleet-wide conjunction candidates from CelesTrak SOCRATES.

SOCRATES screens active payloads against the public catalog.  AEGIS uses its
raw CSV as a broad, free detection layer, then keeps every row where either
object is a Starlink.  These are candidate conjunctions, not operator-grade
collision probabilities or flight decisions.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from ..constants import CELESTRAK_CACHE_TTL_S, CELESTRAK_MAX_RETRIES, CELESTRAK_USER_AGENT
from ..core.conjunction import RiskLevel
from ..core.timebase import ensure_utc

__all__ = [
    "SOCRATES_CSV_URL",
    "SocratesAlert",
    "SocratesAlertCatalog",
    "SocratesError",
    "fetch_starlink_alerts",
    "parse_socrates_csv",
]

SOCRATES_CSV_URL = "https://celestrak.org/SOCRATES/sort-minRange.csv"
_REQUEST_TIMEOUT_S = 60.0
_REQUIRED_FIELDS = (
    "NORAD_CAT_ID_1",
    "OBJECT_NAME_1",
    "DSE_1",
    "NORAD_CAT_ID_2",
    "OBJECT_NAME_2",
    "DSE_2",
    "TCA",
    "TCA_RANGE",
    "TCA_RELATIVE_SPEED",
    "MAX_PROB",
    "DILUTION",
)


class SocratesError(RuntimeError):
    """Raised when the public SOCRATES candidate feed cannot be used safely."""


def _parse_tca(value: str) -> datetime:
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SocratesError(f"invalid SOCRATES TCA {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return ensure_utc(parsed)


def _is_starlink(name: str) -> bool:
    return "STARLINK" in name.upper()


@dataclass(frozen=True)
class SocratesAlert:
    norad_id_1: str
    object_name_1: str
    days_since_epoch_1: float
    norad_id_2: str
    object_name_2: str
    days_since_epoch_2: float
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float
    max_probability: float
    dilution_threshold_km: float

    @property
    def event_id(self) -> str:
        first, second = sorted((self.norad_id_1, self.norad_id_2))
        stamp = self.tca.strftime("%Y%m%dT%H%M%S%f")
        return f"SOCRATES-{first}-{second}-{stamp}"

    @property
    def risk_level(self) -> str:
        return RiskLevel.from_probability(self.max_probability)

    @property
    def starlink_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        if _is_starlink(self.object_name_1):
            ids.append(self.norad_id_1)
        if _is_starlink(self.object_name_2):
            ids.append(self.norad_id_2)
        return tuple(ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.event_id,
            "norad_id_1": self.norad_id_1,
            "object_name_1": self.object_name_1,
            "days_since_epoch_1": self.days_since_epoch_1,
            "norad_id_2": self.norad_id_2,
            "object_name_2": self.object_name_2,
            "days_since_epoch_2": self.days_since_epoch_2,
            "starlink_ids": list(self.starlink_ids),
            "tca": self.tca.isoformat(),
            "miss_distance_km": self.miss_distance_km,
            "relative_speed_km_s": self.relative_speed_km_s,
            "max_probability": self.max_probability,
            "dilution_threshold_km": self.dilution_threshold_km,
            "risk_level": self.risk_level,
            "probability_kind": "SOCRATES_MAXIMUM_PROBABILITY",
        }


@dataclass(frozen=True)
class SocratesAlertCatalog:
    events: tuple[SocratesAlert, ...]
    fetched_at: datetime
    total_feed_events: int
    source_url: str = SOCRATES_CSV_URL
    stale: bool = False

    @property
    def horizon_start(self) -> datetime | None:
        return min((event.tca for event in self.events), default=None)

    @property
    def horizon_end(self) -> datetime | None:
        return max((event.tca for event in self.events), default=None)

    @property
    def unique_starlink_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item for event in self.events for item in event.starlink_ids}))


def parse_socrates_csv(
    text: str,
    *,
    fetched_at: datetime | None = None,
    source_url: str = SOCRATES_CSV_URL,
    stale: bool = False,
) -> SocratesAlertCatalog:
    """Parse the complete feed and retain every row with Starlink on either side."""
    reader = csv.DictReader(StringIO(text))
    fields = tuple(reader.fieldnames or ())
    missing = [field for field in _REQUIRED_FIELDS if field not in fields]
    if missing:
        raise SocratesError("SOCRATES CSV is missing required fields: " + ", ".join(missing))

    events: list[SocratesAlert] = []
    total = 0
    for line_number, row in enumerate(reader, start=2):
        total += 1
        name_1 = (row.get("OBJECT_NAME_1") or "").strip()
        name_2 = (row.get("OBJECT_NAME_2") or "").strip()
        if not (_is_starlink(name_1) or _is_starlink(name_2)):
            continue
        try:
            event = SocratesAlert(
                norad_id_1=str(int((row.get("NORAD_CAT_ID_1") or "").strip())),
                object_name_1=name_1,
                days_since_epoch_1=float(row["DSE_1"]),
                norad_id_2=str(int((row.get("NORAD_CAT_ID_2") or "").strip())),
                object_name_2=name_2,
                days_since_epoch_2=float(row["DSE_2"]),
                tca=_parse_tca(row["TCA"]),
                miss_distance_km=float(row["TCA_RANGE"]),
                relative_speed_km_s=float(row["TCA_RELATIVE_SPEED"]),
                max_probability=float(row["MAX_PROB"]),
                dilution_threshold_km=float(row["DILUTION"]),
            )
        except (KeyError, TypeError, ValueError, SocratesError) as error:
            raise SocratesError(f"invalid SOCRATES row {line_number}: {error}") from error
        if event.miss_distance_km < 0.0:
            raise SocratesError(f"invalid SOCRATES row {line_number}: negative miss distance")
        if event.relative_speed_km_s < 0.0:
            raise SocratesError(f"invalid SOCRATES row {line_number}: negative relative speed")
        if not 0.0 <= event.max_probability <= 1.0:
            raise SocratesError(f"invalid SOCRATES row {line_number}: probability outside [0, 1]")
        if event.dilution_threshold_km < 0.0:
            raise SocratesError(f"invalid SOCRATES row {line_number}: negative dilution threshold")
        events.append(event)

    moment = ensure_utc(fetched_at or datetime.now(timezone.utc))
    return SocratesAlertCatalog(
        events=tuple(events),
        fetched_at=moment,
        total_feed_events=total,
        source_url=source_url,
        stale=stale,
    )


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "aegis" / "socrates"


def _response_text(response: object) -> str:
    status = getattr(response, "status_code", None)
    if status is None:
        raise SocratesError("SOCRATES response has no status code")
    if int(status) >= 400:
        raise SocratesError(f"SOCRATES HTTP {status}")
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    raise SocratesError("SOCRATES response has no body")


def fetch_starlink_alerts(
    *,
    session=None,
    cache_dir: str | Path | None = None,
) -> SocratesAlertCatalog:
    """Fetch or reuse the complete free feed and return all Starlink candidates."""
    directory = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path = directory / "sort-minRange.csv"
    cached_text: str | None = None
    cached_at: datetime | None = None
    if path.is_file():
        cached_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        cached_text = path.read_text(encoding="utf-8")
    if cached_text is not None and time.time() - path.stat().st_mtime <= CELESTRAK_CACHE_TTL_S:
        return parse_socrates_csv(cached_text, fetched_at=cached_at)

    if session is None:
        import requests

        session = requests.Session()

    last_error: Exception | None = None
    body: str | None = None
    for _attempt in range(CELESTRAK_MAX_RETRIES):
        try:
            response = session.get(
                SOCRATES_CSV_URL,
                timeout=_REQUEST_TIMEOUT_S,
                headers={"User-Agent": CELESTRAK_USER_AGENT},
            )
            body = _response_text(response)
            break
        except Exception as error:  # noqa: BLE001 - injected sessions vary
            last_error = error if isinstance(error, SocratesError) else SocratesError(str(error))
    if body is None:
        if cached_text is not None and cached_at is not None:
            return parse_socrates_csv(cached_text, fetched_at=cached_at, stale=True)
        raise SocratesError(f"SOCRATES fetch failed: {last_error}") from last_error

    catalog = parse_socrates_csv(body)
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return catalog
