"""Pipeline proof and modest scale contract (Step 6).

Writes against the public ``aegis.pipeline`` surface plus allowed imports
(core types, ingest). Does not import ``aegis.pipeline`` submodules.
Synthetic opt-in is set only inside these tests.
"""

from __future__ import annotations

import math

import pytest

from aegis.constants import MU_EARTH_KM3_S2, R_EARTH_KM
from aegis.core.maneuver import ManeuverPlan
from aegis.core.state import CovarianceSource
from aegis.ingest import DataSource, SyntheticAuthorization, SyntheticSpec
from aegis.pipeline import PipelineConfig, PipelineResult, run_pipeline

_THREE_SAT = SyntheticSpec(
    n_planes=1,
    sats_per_plane=3,
    include_known_conjunction_triple=True,
)
_TWENTY_FOUR_SAT = SyntheticSpec(
    n_planes=3,
    sats_per_plane=8,
    include_known_conjunction_triple=True,
)


def _auth() -> SyntheticAuthorization:
    return SyntheticAuthorization(acknowledge_synthetic=True)


def _period_s(altitude_km: float) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    return 2.0 * math.pi * math.sqrt(semi_major_km**3 / MU_EARTH_KM3_S2)


def _pair_ids(primary_id: str, secondary_id: str) -> set[str]:
    return {primary_id, secondary_id}


def _mentions_pair(result: PipelineResult, pair: set[str]) -> bool:
    for entry in result.assessed.entries:
        conjunction = entry.conjunction
        if _pair_ids(conjunction.primary.object_id, conjunction.secondary.object_id) == pair:
            return True
    for outcome in result.plan.resolved:
        if all(object_id in outcome.conjunction_id for object_id in pair):
            return True
    notes = " ".join(result.plan.notes)
    return all(object_id in notes for object_id in pair)


def test_known_triple_through_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    duration_s = 1.5 * _period_s(_THREE_SAT.altitude_km)
    result = run_pipeline(
        source=DataSource.SYNTHETIC,
        authorization=_auth(),
        synthetic_spec=_THREE_SAT,
        config=PipelineConfig(duration_s=duration_s),
    )

    assert isinstance(result, PipelineResult)
    assert result.source == DataSource.SYNTHETIC
    assert len(result.catalog) == 3
    assert isinstance(result.plan, ManeuverPlan)
    assert result.covariance_source == CovarianceSource.SYNTHETIC_TLE

    first, second = result.catalog.objects[0], result.catalog.objects[1]
    pair = {first.object_id, second.object_id}
    assert _mentions_pair(result, pair), (
        "assessed entries or plan must mention the first two catalog objects"
    )


def test_twenty_four_sat_pipeline_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    result = run_pipeline(
        source=DataSource.SYNTHETIC,
        authorization=_auth(),
        synthetic_spec=_TWENTY_FOUR_SAT,
        config=PipelineConfig(duration_s=5400, max_objects=24),
    )

    assert isinstance(result, PipelineResult)
    assert len(result.catalog) == 24
    assert len(result.catalog.objects) == 24
    assert isinstance(result.plan, ManeuverPlan)
    assert isinstance(result.plan.converged, bool)
    for outcome in result.plan.unresolved:
        assert outcome.resolved is False
        assert outcome.shortfall_km >= 0


def test_max_objects_caps_twenty_four_sat_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    result = run_pipeline(
        source=DataSource.SYNTHETIC,
        authorization=_auth(),
        synthetic_spec=_TWENTY_FOUR_SAT,
        config=PipelineConfig(duration_s=600.0, step_s=60.0, max_objects=5),
    )

    assert isinstance(result, PipelineResult)
    assert len(result.catalog) == 5
    assert result.source == DataSource.SYNTHETIC
    assert result.catalog.source == DataSource.SYNTHETIC


def test_huge_target_pc_still_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    result = run_pipeline(
        source=DataSource.SYNTHETIC,
        authorization=_auth(),
        synthetic_spec=_THREE_SAT,
        config=PipelineConfig(duration_s=600.0, step_s=60.0, target_pc=0.5),
    )

    assert isinstance(result, PipelineResult)
