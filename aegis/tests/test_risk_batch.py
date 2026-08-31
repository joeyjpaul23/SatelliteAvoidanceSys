"""``assess_catalog`` contract (Step 3).

Writes against the public risk surface and allowed fixtures only.
Does not import ``aegis.risk.batch``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from aegis.core.conjunction import Conjunction, RiskAssessment, RiskLevel
from aegis.core.objects import ObjectType, SpaceObject
from aegis.core.state import CovarianceSource, StateVector
from aegis.core.timebase import seconds_between
from aegis.ingest import (
    DataSource,
    MixedDataSourceError,
    SyntheticAuthorization,
    SyntheticSpec,
    generate_synthetic,
)
from aegis.propagation.covariance import default_covariance_model
from aegis.risk import (
    ALFANO_METHOD,
    AssessedCatalog,
    RankedConjunction,
    assess,
    assess_catalog,
    assess_projection,
)
from aegis.screening import screen

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)


def _object(object_id: str, data_source: str) -> SpaceObject:
    return SpaceObject(
        object_id=object_id,
        name=f"SAT-{object_id}",
        object_type=ObjectType.PAYLOAD,
        data_source=data_source,
    )


def _state(position_km: list[float], velocity_km_s: list[float]) -> StateVector:
    return StateVector(
        epoch=_EPOCH,
        position_km=np.asarray(position_km, dtype=float),
        velocity_km_s=np.asarray(velocity_km_s, dtype=float),
    )


def _dummy_conjunction(
    primary: SpaceObject,
    secondary: SpaceObject,
    conjunction_id: str = "c-1",
    *,
    miss_km: float = 1.0,
) -> Conjunction:
    """Geometric close approach that ``assess`` can consume without screening."""
    primary_state = _state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    secondary_state = _state([7000.0 + miss_km, 0.0, 0.0], [0.0, 7.5, 7.5])
    return Conjunction(
        conjunction_id=conjunction_id,
        primary=primary,
        secondary=secondary,
        tca=_EPOCH,
        miss_distance_km=miss_km,
        relative_speed_km_s=7.5,
        relative_position_rtn_km=np.array([miss_km, 0.0, 0.0]),
        relative_velocity_rtn_km_s=np.array([0.0, 0.0, 7.5]),
        primary_state=primary_state,
        secondary_state=secondary_state,
    )


def _propagation_days(obj: SpaceObject, tca: datetime) -> float:
    if obj.elements is None:
        return 0.0
    return seconds_between(obj.elements.epoch, tca) / 86400.0


def _known_catalog(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    spec = SyntheticSpec(
        n_planes=1,
        sats_per_plane=3,
        include_known_conjunction_triple=True,
    )
    return generate_synthetic(
        SyntheticAuthorization(acknowledge_synthetic=True),
        spec,
    )


def _screen_known(monkeypatch: pytest.MonkeyPatch) -> tuple[list, list[Conjunction]]:
    catalog = _known_catalog(monkeypatch)
    first = catalog.objects[0]
    assert first.elements is not None
    duration_s = 1.5 * first.elements.period_s
    conjunctions = screen(list(catalog), first.elements.epoch, duration_s)
    assert conjunctions, "known-conjunction triple must produce at least one close approach"
    return list(catalog), conjunctions


def test_risk_batch_public_imports() -> None:
    assert callable(assess_catalog)
    assert callable(assess)
    assert callable(assess_projection)
    assert isinstance(RankedConjunction, type)
    assert isinstance(AssessedCatalog, type)


def test_assess_catalog_empty() -> None:
    catalog = assess_catalog([])
    assert isinstance(catalog, AssessedCatalog)
    assert catalog.entries == []
    assert catalog.source == ""


def test_assess_catalog_mixed_objects_raises() -> None:
    primary = _object("1", DataSource.SYNTHETIC)
    secondary = _object("2", DataSource.SYNTHETIC)
    conjunctions = [_dummy_conjunction(primary, secondary)]
    objects = [
        _object("1", DataSource.SYNTHETIC),
        _object("2", DataSource.CELESTRAK),
    ]
    with pytest.raises(MixedDataSourceError):
        assess_catalog(conjunctions, objects=objects)
    with pytest.raises(MixedDataSourceError):
        assess_catalog([], objects=objects)


def test_assess_catalog_mixed_conjunction_sources_across_list_raises() -> None:
    synthetic = _dummy_conjunction(
        _object("1", DataSource.SYNTHETIC),
        _object("2", DataSource.SYNTHETIC),
        "syn",
    )
    celestrak = _dummy_conjunction(
        _object("3", DataSource.CELESTRAK),
        _object("4", DataSource.CELESTRAK),
        "cel",
    )
    with pytest.raises(MixedDataSourceError):
        assess_catalog([synthetic, celestrak])


def test_assess_catalog_mixed_sources_within_conjunction_raises() -> None:
    mixed = _dummy_conjunction(
        _object("1", DataSource.SYNTHETIC),
        _object("2", DataSource.CELESTRAK),
        "mixed",
    )
    with pytest.raises(MixedDataSourceError):
        assess_catalog([mixed])


def test_assess_catalog_synthetic_screen_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    objects, conjunctions = _screen_known(monkeypatch)
    catalog = assess_catalog(conjunctions, objects=objects)

    assert isinstance(catalog, AssessedCatalog)
    assert catalog.source == DataSource.SYNTHETIC
    assert catalog.covariance_source == CovarianceSource.SYNTHETIC_TLE
    assert catalog.entries
    assert all(isinstance(entry, RankedConjunction) for entry in catalog.entries)

    probabilities = [entry.assessment.probability for entry in catalog.entries]
    assert probabilities == sorted(probabilities, reverse=True)
    sort_key = [
        (-entry.assessment.probability, entry.conjunction.conjunction_id)
        for entry in catalog.entries
    ]
    assert sort_key == sorted(sort_key)

    assert catalog.entries[0].rank == 1
    assert catalog.entries[0].assessment.probability == max(probabilities)
    assert [entry.rank for entry in catalog.entries] == list(range(1, len(catalog.entries) + 1))

    for entry in catalog.entries:
        looked_up = catalog.by_id(entry.conjunction.conjunction_id)
        assert looked_up.conjunction.conjunction_id == entry.conjunction.conjunction_id
        assert looked_up.rank == entry.rank
        assert looked_up.assessment.probability == entry.assessment.probability

    monitor = catalog.above("MONITOR")
    act = catalog.above("ACT")
    monitor_floor = RiskLevel.rank(RiskLevel.MONITOR)
    act_floor = RiskLevel.rank(RiskLevel.ACT)
    assert all(
        RiskLevel.rank(entry.assessment.risk_level) >= monitor_floor for entry in monitor
    )
    assert all(RiskLevel.rank(entry.assessment.risk_level) >= act_floor for entry in act)
    monitor_ids = {entry.conjunction.conjunction_id for entry in monitor}
    act_ids = {entry.conjunction.conjunction_id for entry in act}
    for entry in catalog.entries:
        cid = entry.conjunction.conjunction_id
        severity = RiskLevel.rank(entry.assessment.risk_level)
        assert (cid in monitor_ids) == (severity >= monitor_floor)
        assert (cid in act_ids) == (severity >= act_floor)
    assert act_ids <= monitor_ids


def test_assess_catalog_uses_existing_assess(monkeypatch: pytest.MonkeyPatch) -> None:
    _objects, conjunctions = _screen_known(monkeypatch)
    catalog = assess_catalog(conjunctions)
    assert catalog.entries
    for entry in catalog.entries:
        assert 0.0 <= entry.assessment.probability <= 1.0
        assert entry.assessment.method
        assert entry.assessment.method == ALFANO_METHOD

    conjunction = catalog.entries[0].conjunction
    model = default_covariance_model()
    direct = assess(
        conjunction,
        model.covariance(
            conjunction.primary, _propagation_days(conjunction.primary, conjunction.tca)
        ),
        model.covariance(
            conjunction.secondary, _propagation_days(conjunction.secondary, conjunction.tca)
        ),
    )
    assert isinstance(direct, RiskAssessment)
    batched = catalog.entries[0].assessment
    assert batched.probability == pytest.approx(direct.probability, rel=0, abs=0)
    assert batched.method == direct.method


def test_assess_catalog_ties_broken_by_conjunction_id() -> None:
    primary = _object("1", DataSource.SYNTHETIC)
    secondary = _object("2", DataSource.SYNTHETIC)
    later = _dummy_conjunction(primary, secondary, "z-id")
    earlier = _dummy_conjunction(primary, secondary, "a-id")
    catalog = assess_catalog([later, earlier])
    assert [entry.conjunction.conjunction_id for entry in catalog.entries] == ["a-id", "z-id"]
    assert catalog.entries[0].assessment.probability == catalog.entries[1].assessment.probability
    assert catalog.entries[0].rank == 1
    assert catalog.entries[1].rank == 2


def test_assess_still_returns_risk_assessment() -> None:
    primary = _object("1", DataSource.SYNTHETIC)
    secondary = _object("2", DataSource.SYNTHETIC)
    conjunction = _dummy_conjunction(primary, secondary)
    model = default_covariance_model()
    assessment = assess(
        conjunction,
        model.covariance(primary, 0.0),
        model.covariance(secondary, 0.0),
    )
    assert isinstance(assessment, RiskAssessment)
    assert 0.0 <= assessment.probability <= 1.0
    assert assessment.method
    assert assessment.method == ALFANO_METHOD
    assert assessment.conjunction_id == conjunction.conjunction_id
