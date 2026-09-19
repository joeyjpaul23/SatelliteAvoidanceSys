"""AEGIS vs 18 SDS cross-check: event logic, summary, archive, API. No network."""

from __future__ import annotations

import importlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aegis.ingest.ops import load_starlink_slice
from aegis.ingest.spacetrack import parse_cdm_public
from aegis.pipeline import crosscheck
from aegis.pipeline.crosscheck import (
    CrossCheckReport,
    append_archive,
    crosscheck_events,
)

# ``aegis.api`` re-exports the FastAPI object as ``app``, shadowing the module.
app_module = importlib.import_module("aegis.api.app")


def _row(cdm_id: str, sat1: str, sat2: str, *, created="2026-09-18 01:00:00", pc="0.0002", miss_m="200"):
    return {
        "CDM_ID": cdm_id,
        "CREATED": created,
        "EMERGENCY_REPORTABLE": "Y",
        "TCA": "2026-09-19T10:00:00.000000",
        "MIN_RNG": miss_m,
        "PC": pc,
        "SAT_1_ID": sat1,
        "SAT_1_NAME": f"OBJ {sat1}",
        "SAT1_OBJECT_TYPE": "DEBRIS",
        "SAT_2_ID": sat2,
        "SAT_2_NAME": f"OBJ {sat2}",
        "SAT2_OBJECT_TYPE": "DEBRIS",
    }


def _objects(*ids: str):
    objects = load_starlink_slice(max_objects=len(ids)).objects
    for obj, norad_id in zip(objects, ids):
        obj.object_id = norad_id
    return objects


def _entry(tca, miss_km, pc, max_pc):
    return SimpleNamespace(
        conjunction=SimpleNamespace(tca=tca),
        assessment=SimpleNamespace(
            miss_distance_km=miss_km,
            probability=pc,
            max_probability=max_pc,
            dilution_flag=True,
            sigma_major_km=3.0,
        ),
    )


@pytest.fixture
def fake_physics(monkeypatch):
    """Screen returns one conjunction per call; assess returns two candidates."""
    calls: list[dict] = []

    def fake_screen(pair, start, duration_s, *, step_s, box_km):
        calls.append({"ids": [obj.object_id for obj in pair], "start": start, "duration_s": duration_s})
        return [] if pair[0].object_id == "900" else ["conjunction"]

    def fake_assess(conjunctions, *, objects):
        tca = calls[-1]["start"] + timedelta(seconds=calls[-1]["duration_s"] / 2 + 2.0)
        return SimpleNamespace(entries=[_entry(tca, 0.9, 1e-7, 1e-6), _entry(tca, 0.3, 2e-6, 5e-4)])

    monkeypatch.setattr(crosscheck, "screen", fake_screen)
    monkeypatch.setattr(crosscheck, "assess_catalog", fake_assess)
    return calls


def test_crosscheck_collapses_rows_and_keeps_closest_candidate(fake_physics) -> None:
    rows = parse_cdm_public(json.dumps([_row("1", "100", "200"), _row("2", "200", "100")]))
    (check,) = crosscheck_events(rows, _objects("100", "200"))

    assert check.cdm_rows == 2
    assert check.found
    assert check.aegis_miss_km == pytest.approx(0.3)
    assert check.aegis_pc == pytest.approx(2e-6)
    assert check.aegis_max_pc == pytest.approx(5e-4)
    assert check.aegis_dilution is True
    assert check.tca_offset_s == pytest.approx(2.0)
    assert fake_physics[0]["duration_s"] == 2 * crosscheck.CROSSCHECK_WINDOW_S


def test_crosscheck_reports_missing_gp_and_no_approach(fake_physics) -> None:
    rows = parse_cdm_public(json.dumps([_row("1", "100", "404"), _row("2", "900", "200")]))
    missing, no_approach = crosscheck_events(rows, _objects("100", "200", "900"))

    assert not missing.found and missing.reason == "no Space-Track GP for 404"
    assert not no_approach.found and "screening box" in no_approach.reason
    assert no_approach.propagation_days is not None


def _report(fake_physics) -> CrossCheckReport:
    rows = parse_cdm_public(
        json.dumps([_row("1", "100", "200"), _row("2", "300", "400", pc="", miss_m="4000")])
    )
    checks = crosscheck_events(rows, _objects("100", "200", "300", "400"))
    moment = checks[0].event.created
    return CrossCheckReport(checks=checks, cdm_fetched_at=moment, gp_fetched_at=moment)


def test_summary_statistics(fake_physics) -> None:
    summary = _report(fake_physics).summary()
    assert summary["events"] == 2 and summary["found"] == 2 and summary["cdm_rows"] == 2
    assert summary["pc_compared"] == 1  # the second event has no 18 SDS Pc
    assert summary["median_log10_pc_ratio"] == pytest.approx(-2.0)  # 2e-6 / 2e-4
    assert summary["max_pc_bounds_sds"] == 1 and summary["max_pc_compared"] == 1
    assert summary["dilution_flagged"] == 2
    assert summary["median_abs_tca_offset_s"] == pytest.approx(2.0)


def test_archive_appends_each_cdm_once(fake_physics, tmp_path: Path) -> None:
    archive = tmp_path / "history.jsonl"
    report = _report(fake_physics)
    assert append_archive(report, archive) == 2
    assert append_archive(report, archive) == 0
    lines = [json.loads(line) for line in archive.read_text().splitlines()]
    assert [line["cdm_id"] for line in lines] == ["1", "2"]
    assert all("checked_at" in line and "aegis_pc" in line for line in lines)


def test_api_public_cdms_unconfigured() -> None:
    body = TestClient(app_module.app).get("/api/public-cdms").json()
    assert body["configured"] is False and body["events"] == []


def test_api_public_cdms_serves_report(fake_physics, monkeypatch) -> None:
    monkeypatch.setenv("SPACETRACK_USER", "user@example.com")
    monkeypatch.setenv("SPACETRACK_PASS", "secret")
    report = _report(fake_physics)
    monkeypatch.setattr(app_module, "_crosscheck_report", lambda: report)

    body = TestClient(app_module.app).get("/api/public-cdms").json()

    assert body["configured"] is True
    assert body["summary"]["events"] == 2
    assert [event["cdm_id"] for event in body["events"]] == ["1", "2"]
    assert "secret" not in json.dumps(body)
