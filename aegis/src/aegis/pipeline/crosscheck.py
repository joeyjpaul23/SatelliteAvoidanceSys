"""AEGIS versus 18 SDS on Space-Track's public conjunction feed.

``cdm_public`` lists upcoming high-risk conjunctions (Pc >= 1e-4) that 18 SDS
screened from special-perturbations ephemerides with real covariance. For each
event, AEGIS screens the same pair from Space-Track GP elements in a window
around the published TCA and assesses it with its own TLE covariance model.
The gap between the two is a direct, repeatable measure of how far public-data
risk sits from operator-grade screening (docs/LIMITATIONS.md section 3).

``python -m aegis.pipeline.crosscheck`` prints the comparison; ``--archive``
also appends it to a JSONL history so the validation set grows over time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..constants import SCREENING_BOX_TLE_GRADE_KM, SCREENING_STEP_S
from ..core.objects import SpaceObject
from ..core.timebase import ensure_utc, seconds_between, shift, utc_now
from ..envfile import load_env_file
from ..ingest.spacetrack import (
    PublicConjunction,
    SpaceTrackClient,
    SpaceTrackError,
    credentials_from_env,
    latest_per_event,
)
from ..risk import assess_catalog
from ..screening import ScreeningError, screen

__all__ = [
    "CROSSCHECK_WINDOW_S",
    "CrossCheck",
    "CrossCheckReport",
    "append_archive",
    "crosscheck_events",
    "default_archive_path",
    "run_crosscheck",
]

#: Half-width of the screening window around the 18 SDS TCA. Far wider than
#: any GP timing error, and well under half a LEO period, so it holds exactly
#: one approach of the pair.
CROSSCHECK_WINDOW_S = 900.0


@dataclass(frozen=True)
class CrossCheck:
    """One ``cdm_public`` event (its latest CDM) with AEGIS's result beside it.

    ``aegis_max_pc`` is the worst case over covariance scale for AEGIS's miss
    geometry; ``aegis_dilution`` means AEGIS's own TLE covariance is so wide
    that its Pc understates risk, which is the expected failure against
    18 SDS's tight covariance.
    """

    event: PublicConjunction
    cdm_rows: int = 1
    aegis_tca: datetime | None = None
    aegis_miss_km: float | None = None
    aegis_pc: float | None = None
    aegis_max_pc: float | None = None
    aegis_dilution: bool | None = None
    aegis_sigma_major_km: float | None = None
    propagation_days: float | None = None
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.aegis_miss_km is not None

    @property
    def tca_offset_s(self) -> float | None:
        if self.aegis_tca is None:
            return None
        return seconds_between(self.event.tca, self.aegis_tca)

    def as_dict(self) -> dict[str, object]:
        event = self.event
        return {
            "cdm_id": event.cdm_id,
            "created": event.created.isoformat(),
            "tca": event.tca.isoformat(),
            "norad_id_1": event.norad_id_1,
            "object_name_1": event.object_name_1,
            "object_type_1": event.object_type_1,
            "norad_id_2": event.norad_id_2,
            "object_name_2": event.object_name_2,
            "object_type_2": event.object_type_2,
            "emergency_reportable": event.emergency_reportable,
            "cdm_rows": self.cdm_rows,
            "sds_miss_km": event.miss_distance_km,
            "sds_pc": event.collision_probability,
            "aegis_found": self.found,
            "aegis_tca": self.aegis_tca.isoformat() if self.aegis_tca else None,
            "aegis_tca_offset_s": self.tca_offset_s,
            "aegis_miss_km": self.aegis_miss_km,
            "aegis_pc": self.aegis_pc,
            "aegis_max_pc": self.aegis_max_pc,
            "aegis_dilution": self.aegis_dilution,
            "aegis_sigma_major_km": self.aegis_sigma_major_km,
            "propagation_days": self.propagation_days,
            "reason": self.reason,
        }


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


@dataclass(frozen=True)
class CrossCheckReport:
    checks: tuple[CrossCheck, ...]
    cdm_fetched_at: datetime
    gp_fetched_at: datetime | None

    def summary(self) -> dict[str, object]:
        found = [check for check in self.checks if check.found]
        miss_error = [abs(check.aegis_miss_km - check.event.miss_distance_km) for check in found]
        tca_error = [abs(check.tca_offset_s) for check in found]
        # log10(AEGIS / 18 SDS): 0 agrees, +1 AEGIS ten times higher.
        pc_ratio = [
            math.log10(check.aegis_pc / check.event.collision_probability)
            for check in found
            if check.aegis_pc and check.event.collision_probability
        ]
        with_sds_pc = [check for check in found if check.event.collision_probability]
        bounded = [
            check
            for check in with_sds_pc
            if (check.aegis_max_pc or 0.0) >= check.event.collision_probability
        ]
        return {
            "events": len(self.checks),
            "cdm_rows": sum(check.cdm_rows for check in self.checks),
            "found": len(found),
            "median_abs_miss_error_km": _median(miss_error),
            "median_abs_tca_offset_s": _median(tca_error),
            "pc_compared": len(pc_ratio),
            "median_log10_pc_ratio": _median(pc_ratio),
            "dilution_flagged": sum(1 for check in found if check.aegis_dilution),
            "max_pc_bounds_sds": len(bounded),
            "max_pc_compared": len(with_sds_pc),
            "cdm_fetched_at": self.cdm_fetched_at.isoformat(),
            "gp_fetched_at": self.gp_fetched_at.isoformat() if self.gp_fetched_at else None,
        }


def _check_event(
    event: PublicConjunction,
    cdm_rows: int,
    objects_by_id: dict[str, SpaceObject],
    *,
    window_s: float,
    step_s: float,
) -> CrossCheck:
    missing = [
        norad_id for norad_id in (event.norad_id_1, event.norad_id_2) if norad_id not in objects_by_id
    ]
    if missing:
        return CrossCheck(event, cdm_rows, reason=f"no Space-Track GP for {', '.join(missing)}")
    pair = [objects_by_id[event.norad_id_1], objects_by_id[event.norad_id_2]]
    tca = ensure_utc(event.tca)
    try:
        conjunctions = screen(
            pair,
            shift(tca, -window_s),
            2.0 * window_s,
            step_s=step_s,
            box_km=SCREENING_BOX_TLE_GRADE_KM,
        )
    except ScreeningError as error:
        # One undecayable or decayed element set must not sink the whole report.
        return CrossCheck(event, cdm_rows, reason=f"screening failed: {error}")
    ages = [
        abs(seconds_between(obj.elements.epoch, tca)) / 86400.0
        for obj in pair
        if obj.elements is not None
    ]
    propagation_days = max(ages) if ages else None
    if not conjunctions:
        return CrossCheck(
            event,
            cdm_rows,
            propagation_days=propagation_days,
            reason="no approach inside the TLE-grade screening box",
        )
    assessed = assess_catalog(conjunctions, objects=pair)
    best = min(assessed.entries, key=lambda entry: entry.assessment.miss_distance_km)
    assessment = best.assessment
    return CrossCheck(
        event,
        cdm_rows,
        aegis_tca=best.conjunction.tca,
        aegis_miss_km=float(assessment.miss_distance_km),
        aegis_pc=float(assessment.probability),
        aegis_max_pc=float(assessment.max_probability),
        aegis_dilution=bool(assessment.dilution_flag),
        aegis_sigma_major_km=float(assessment.sigma_major_km),
        propagation_days=propagation_days,
    )


def crosscheck_events(
    conjunctions: tuple[PublicConjunction, ...] | list[PublicConjunction],
    objects: list[SpaceObject],
    *,
    window_s: float = CROSSCHECK_WINDOW_S,
    step_s: float = SCREENING_STEP_S,
) -> tuple[CrossCheck, ...]:
    """Collapse the feed to events, then screen and assess each pair. No network."""
    objects_by_id = {obj.object_id: obj for obj in objects}
    return tuple(
        _check_event(event, rows, objects_by_id, window_s=window_s, step_s=step_s)
        for event, rows in latest_per_event(conjunctions)
    )


def run_crosscheck(client: SpaceTrackClient | None = None) -> CrossCheckReport:
    """Fetch (or reuse cached) ``cdm_public`` and GP, then cross-check every event."""
    client = client if client is not None else SpaceTrackClient()
    events, cdm_fetched_at = client.fetch_public_conjunctions_with_time()
    ids = {norad_id for event in events for norad_id in (event.norad_id_1, event.norad_id_2)}
    catalog = client.fetch_objects(ids)
    return CrossCheckReport(
        checks=crosscheck_events(events, list(catalog.objects)),
        cdm_fetched_at=cdm_fetched_at,
        gp_fetched_at=catalog.fetched_at if ids else None,
    )


def default_archive_path() -> Path:
    xdg = os.environ.get("XDG_STATE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return root / "aegis" / "crosscheck" / "cdm_public.jsonl"


def append_archive(report: CrossCheckReport, path: str | Path | None = None) -> int:
    """Append checks whose CDM is not archived yet. Returns how many were added.

    Each line is the prediction made at check time; a later run never
    rewrites it with fresher elements.
    """
    archive = Path(path) if path is not None else default_archive_path()
    seen: set[str] = set()
    if archive.is_file():
        for line in archive.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(str(json.loads(line)["cdm_id"]))
            except (ValueError, KeyError, TypeError):
                # Blank, or cut short by a run killed mid-write: skip it
                # rather than fail every later hourly run.
                continue
    checked_at = utc_now().isoformat()
    context = {
        "checked_at": checked_at,
        "cdm_fetched_at": report.cdm_fetched_at.isoformat(),
        "gp_fetched_at": report.gp_fetched_at.isoformat() if report.gp_fetched_at else None,
    }
    new = [check for check in report.checks if check.event.cdm_id not in seen]
    if new:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("a", encoding="utf-8") as handle:
            for check in new:
                handle.write(json.dumps({**check.as_dict(), **context}) + "\n")
    return len(new)


def _format_pc(value: float | None) -> str:
    return f"{value:.1e}" if value else "—"


def _format_km(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aegis.pipeline.crosscheck",
        description="Compare AEGIS against 18 SDS on Space-Track's public conjunction feed.",
    )
    parser.add_argument("--archive", action="store_true", help="append new events to the JSONL history")
    parser.add_argument("--rows", type=int, default=20, help="events to print (default 20)")
    args = parser.parse_args(argv)

    load_env_file()
    if credentials_from_env() is None:
        print("SPACETRACK_USER and SPACETRACK_PASS must both be set", file=sys.stderr)
        return 2
    try:
        report = run_crosscheck()
    except SpaceTrackError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = report.summary()
    print(
        f"{summary['events']} events ({summary['cdm_rows']} CDM rows); AEGIS found {summary['found']}. "
        f"Median |miss error| {_format_km(summary['median_abs_miss_error_km'])} km, "
        f"median |TCA offset| {summary['median_abs_tca_offset_s'] or 0:.1f} s."
    )
    print(
        f"Pc: median log10(AEGIS/18 SDS) {summary['median_log10_pc_ratio'] or 0:+.2f} over "
        f"{summary['pc_compared']} events; AEGIS flagged dilution on {summary['dilution_flagged']}; "
        f"AEGIS max Pc >= 18 SDS Pc on {summary['max_pc_bounds_sds']}/{summary['max_pc_compared']}."
    )
    print(
        f"{'TCA (UTC)':<17} {'PAIR':<38} {'MISS 18SDS':>10} {'AEGIS':>7} "
        f"{'PC 18SDS':>9} {'AEGIS':>8} {'MAX':>8} {'DIL':>4}"
    )
    for check in report.checks[: max(args.rows, 0)]:
        event = check.event
        pair = f"{event.object_name_1} / {event.object_name_2}"[:38]
        aegis_miss = _format_km(check.aegis_miss_km) if check.found else "miss"
        dilution = "yes" if check.aegis_dilution else ""
        print(
            f"{event.tca:%Y-%m-%d %H:%M} {pair:<38} {event.miss_distance_km:>10.3f} "
            f"{aegis_miss:>7} {_format_pc(event.collision_probability):>9} "
            f"{_format_pc(check.aegis_pc):>8} {_format_pc(check.aegis_max_pc):>8} {dilution:>4}"
        )
    if args.archive:
        added = append_archive(report)
        print(f"archived {added} new events to {default_archive_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
