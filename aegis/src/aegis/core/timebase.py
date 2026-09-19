"""Epoch handling.

The system uses timezone-aware UTC ``datetime`` at its boundaries (parsing,
reporting, serialisation) and floating-point seconds-since-epoch internally
(propagation grids, root finding). Mixing the two is the most common source of
off-by-one-leap-second style bugs, so conversions are funnelled through this
module.

A deliberate simplification worth stating: AEGIS treats UTC as a uniform
timescale and does not model leap seconds. For conjunction assessment over a
seven-day horizon this is immaterial next to the several-second timing
uncertainty that TLE-quality ephemeris already carries. If this system is ever
fed operator-grade ephemeris where sub-second timing matters, replace this
module with a proper TAI/UTC-aware timescale.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

__all__ = [
    "utc_now",
    "ensure_utc",
    "parse_epoch",
    "format_epoch",
    "seconds_between",
    "shift",
    "JULIAN_DATE_UNIX_EPOCH",
]

#: Julian date of the Unix epoch (1970-01-01T00:00:00Z).
JULIAN_DATE_UNIX_EPOCH = 2440587.5


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware one to UTC.

    Naive datetimes are assumed to already be UTC rather than local time.
    Every external format this system reads (CCSDS, CelesTrak OMM) specifies
    UTC, so that assumption is safe here and would not be elsewhere.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def parse_epoch(text: str) -> datetime:
    """Parse the epoch formats CCSDS and CelesTrak actually emit.

    Accepts both calendar and day-of-year forms, with or without fractional
    seconds and with or without a trailing ``Z``::

        2026-08-27T14:03:22.123456Z
        2026-08-27T14:03:22
        2026-240T14:03:22.5

    CCSDS 508.0-B-1 permits the trailing ``Z`` but recommends omitting it, so
    we accept it on read and never emit it.
    """
    cleaned = text.strip()
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace(" ", "T", 1) if "T" not in cleaned and " " in cleaned else cleaned

    date_part, _, time_part = cleaned.partition("T")
    pieces = date_part.split("-")

    if len(pieces) == 2:
        # Day-of-year form: YYYY-DDD
        year, day_of_year = int(pieces[0]), int(pieces[1])
        base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
    elif len(pieces) == 3:
        year, month, day = (int(piece) for piece in pieces)
        base = datetime(year, month, day, tzinfo=timezone.utc)
    else:
        raise ValueError(f"unrecognised epoch date component: {text!r}")

    if not time_part:
        return base

    time_pieces = time_part.split(":")
    if len(time_pieces) != 3:
        raise ValueError(f"unrecognised epoch time component: {text!r}")

    hours = int(time_pieces[0])
    minutes = int(time_pieces[1])
    seconds = float(time_pieces[2])

    return base + timedelta(hours=hours, minutes=minutes, seconds=seconds)


def format_epoch(moment: datetime, *, decimals: int = 6) -> str:
    """Render an epoch in the CCSDS calendar form, without a trailing ``Z``."""
    moment = ensure_utc(moment)
    whole = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if decimals <= 0:
        return whole
    fraction = moment.microsecond / 1_000_000
    return f"{whole}{f'{fraction:.{decimals}f}'[1:]}"


def seconds_between(start: datetime, end: datetime) -> float:
    """Signed seconds from ``start`` to ``end``."""
    return (ensure_utc(end) - ensure_utc(start)).total_seconds()


def shift(moment: datetime, seconds: float) -> datetime:
    """Offset an epoch by a floating-point number of seconds."""
    return ensure_utc(moment) + timedelta(seconds=seconds)
