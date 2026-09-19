"""Honest plan artifact + LIMITATIONS.md contract (Step 8).

Writes against the public ``aegis.pipeline`` surface plus allowed imports
(core types, ingest, risk). Does not import ``aegis.pipeline`` submodules.
Offline only: synthetic path is dual-gated; no CelesTrak HTTP.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aegis.core.maneuver import (
    Maneuver,
    ManeuverPlan,
    ResolvedConjunction,
    SatelliteManeuverSet,
)
from aegis.core.state import CovarianceSource
from aegis.ingest import Catalog, DataSource, SyntheticAuthorization, SyntheticSpec
from aegis.pipeline import (
    PipelineConfig,
    PipelineResult,
    plan_artifact,
    run_pipeline,
    write_plan_json,
    write_plan_text,
)
from aegis.risk import AssessedCatalog
import aegis.pipeline as pipeline_pkg

REPO_ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS_MD = REPO_ROOT / "docs" / "LIMITATIONS.md"

_REQUIRED_EXPORTS = {"plan_artifact", "write_plan_json", "write_plan_text"}

_ARTIFACT_KEYS = {
    "source",
    "covariance_source",
    "object_count",
    "generated_at",
    "plan_id",
    "summary",
    "burns",
    "resolved",
    "unresolved",
    "warnings",
    "converged",
    "iterations",
}

_RESOLVED_KEYS = {
    "conjunction_id",
    "probability_before",
    "probability_after",
    "miss_distance_before_km",
    "miss_distance_after_km",
    "resolved",
    "shortfall_km",
}

_BURN_KEYS = {"satellite_id", "epoch", "delta_v_rtn_km_s", "magnitude_km_s"}

_THREE_SAT = SyntheticSpec(
    n_planes=1,
    sats_per_plane=3,
    include_known_conjunction_triple=True,
)


def _auth() -> SyntheticAuthorization:
    return SyntheticAuthorization(acknowledge_synthetic=True)


def _fast_config() -> PipelineConfig:
    return PipelineConfig(duration_s=600.0, step_s=60.0, max_iterations=1)


def _run_three_sat(monkeypatch: pytest.MonkeyPatch) -> PipelineResult:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    result = run_pipeline(
        source=DataSource.SYNTHETIC,
        authorization=_auth(),
        synthetic_spec=_THREE_SAT,
        config=_fast_config(),
    )
    assert isinstance(result, PipelineResult)
    return result


def _parse_iso_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    return parsed


def _assert_artifact_keys(payload: dict) -> None:
    missing = _ARTIFACT_KEYS - set(payload)
    assert not missing, f"plan artifact missing keys: {sorted(missing)}"


def _assert_unresolved_sorted(unresolved: list) -> None:
    assert isinstance(unresolved, list)
    shortfalls = [item["shortfall_km"] for item in unresolved]
    assert shortfalls == sorted(shortfalls, reverse=True)
    assert all(item["resolved"] is False for item in unresolved)


def _honesty_blob(parts: list[str] | str) -> str:
    if isinstance(parts, str):
        return parts.lower()
    return " ".join(str(item) for item in parts).lower()


def _mentions_synthetic_tle_honesty(text: str) -> bool:
    lowered = text.lower()
    mentions_grade = "synthetic" in lowered or "tle" in lowered
    mentions_ops = "operational" in lowered and "maneuver" in lowered
    return mentions_grade and mentions_ops


def _outcome(
    conjunction_id: str,
    *,
    resolved: bool,
    shortfall_km: float,
) -> ResolvedConjunction:
    return ResolvedConjunction(
        conjunction_id=conjunction_id,
        probability_before=0.2,
        probability_after=0.05 if resolved else 0.2,
        miss_distance_before_km=0.01,
        miss_distance_after_km=1.0 if resolved else 0.01,
        required_miss_distance_km=2.0,
        resolved=resolved,
        shortfall_km=shortfall_km,
    )


def _constructed_result_unsorted_unresolved() -> PipelineResult:
    """Public-type PipelineResult whose unresolved shortfalls are not sorted."""
    epoch = datetime(2010, 1, 1, tzinfo=timezone.utc)
    burn = Maneuver(
        satellite_id="SAT-A",
        epoch=epoch,
        delta_v_rtn_km_s=[0.0, 1.0e-6, 0.0],
    )
    plan = ManeuverPlan(
        plan_id="constructed-unsorted",
        generated_at=datetime(2020, 6, 1, tzinfo=timezone.utc),
        satellite_sets={
            "SAT-A": SatelliteManeuverSet(satellite_id="SAT-A", maneuvers=[burn]),
        },
        resolved=[
            _outcome("c-small", resolved=False, shortfall_km=0.5),
            _outcome("c-ok", resolved=True, shortfall_km=0.0),
            _outcome("c-large", resolved=False, shortfall_km=5.0),
        ],
        iterations=2,
        converged=False,
    )
    catalog = Catalog(
        source=DataSource.SYNTHETIC,
        objects=[],
        fetched_at=datetime.now(timezone.utc),
        query="constructed",
    )
    return PipelineResult(
        source=DataSource.SYNTHETIC,
        catalog=catalog,
        assessed=AssessedCatalog(source=DataSource.SYNTHETIC, entries=[]),
        plan=plan,
        covariance_source=CovarianceSource.SYNTHETIC_TLE,
        config=PipelineConfig(),
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_plan_artifact_exports_are_public() -> None:
    assert _REQUIRED_EXPORTS <= set(pipeline_pkg.__all__)
    for name in _REQUIRED_EXPORTS:
        assert hasattr(pipeline_pkg, name)
        assert callable(getattr(pipeline_pkg, name))


# ---------------------------------------------------------------------------
# plan_artifact from a synthetic 3-sat pipeline
# ---------------------------------------------------------------------------


def test_plan_artifact_required_keys_from_synthetic_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_three_sat(monkeypatch)
    artifact = plan_artifact(result)

    assert isinstance(artifact, dict)
    _assert_artifact_keys(artifact)
    json.dumps(artifact)

    assert artifact["source"] in {DataSource.CELESTRAK, DataSource.SYNTHETIC}
    assert artifact["source"] == DataSource.SYNTHETIC
    assert artifact["source"] == result.source
    assert artifact["covariance_source"] == CovarianceSource.SYNTHETIC_TLE
    assert artifact["covariance_source"] == result.covariance_source
    assert artifact["object_count"] == len(result.catalog)
    assert artifact["object_count"] == 3
    assert artifact["plan_id"] == result.plan.plan_id
    assert artifact["summary"] == result.plan.summary()
    assert artifact["converged"] is result.plan.converged
    assert artifact["iterations"] == result.plan.iterations
    assert isinstance(artifact["converged"], bool)
    assert isinstance(artifact["iterations"], int)
    _parse_iso_utc(artifact["generated_at"])

    assert isinstance(artifact["burns"], list)
    for burn in artifact["burns"]:
        assert _BURN_KEYS <= set(burn)
        dv = burn["delta_v_rtn_km_s"]
        assert isinstance(dv, list)
        assert len(dv) == 3
        assert all(isinstance(component, (int, float)) for component in dv)

    assert isinstance(artifact["resolved"], list)
    for outcome in artifact["resolved"]:
        assert _RESOLVED_KEYS <= set(outcome)

    assert isinstance(artifact["warnings"], list)
    assert all(isinstance(item, str) for item in artifact["warnings"])


def test_unresolved_sorted_by_shortfall_km_descending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_three_sat(monkeypatch)
    live = plan_artifact(result)
    _assert_unresolved_sorted(live["unresolved"])

    constructed = _constructed_result_unsorted_unresolved()
    artifact = plan_artifact(constructed)
    unresolved = artifact["unresolved"]
    _assert_unresolved_sorted(unresolved)
    assert len(unresolved) == 2
    assert unresolved[0]["shortfall_km"] == pytest.approx(5.0)
    assert unresolved[1]["shortfall_km"] == pytest.approx(0.5)
    assert [item["conjunction_id"] for item in unresolved] == ["c-large", "c-small"]


def test_synthetic_tle_warning_not_for_operational_maneuvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_three_sat(monkeypatch)
    assert result.covariance_source == CovarianceSource.SYNTHETIC_TLE
    artifact = plan_artifact(result)
    assert artifact["covariance_source"] == CovarianceSource.SYNTHETIC_TLE
    assert artifact["warnings"], "SYNTHETIC_TLE must produce honesty warnings"
    assert _mentions_synthetic_tle_honesty(_honesty_blob(artifact["warnings"]))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_write_plan_json_valid_with_required_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run_three_sat(monkeypatch)
    path = write_plan_json(result, tmp_path / "plan.json")
    assert path == tmp_path / "plan.json"
    assert path.is_file()

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    _assert_artifact_keys(payload)
    assert payload == plan_artifact(result)
    assert text.lstrip().startswith("{")
    assert json.dumps(payload, indent=2).strip() == text.strip()


def test_write_plan_text_contains_source_and_honesty_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run_three_sat(monkeypatch)
    path = write_plan_text(result, tmp_path / "plan.txt")
    assert path == tmp_path / "plan.txt"
    assert path.is_file()

    text = path.read_text(encoding="utf-8")
    assert result.source in text
    assert DataSource.SYNTHETIC in text
    assert _mentions_synthetic_tle_honesty(text)


# ---------------------------------------------------------------------------
# LIMITATIONS.md (repo docs/, not package docs/)
# ---------------------------------------------------------------------------


def test_limitations_md_exists_and_covers_required_ideas() -> None:
    assert LIMITATIONS_MD == REPO_ROOT / "docs" / "LIMITATIONS.md"
    assert LIMITATIONS_MD.is_file(), f"missing {LIMITATIONS_MD}"

    text = LIMITATIONS_MD.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "synthetic_tle" in lowered
    assert "operational" in lowered
    assert "aegis_allow_synthetic" in lowered
    assert "celestrak" in lowered
    assert "prototype" in lowered or "flight" in lowered

    assert "covariance" in lowered
    assert "triage" in lowered
    assert "cdm" in lowered
    assert "intra-fleet" in lowered or "intra fleet" in lowered
    assert "fail" in lowered or "failure" in lowered or "silently" in lowered
