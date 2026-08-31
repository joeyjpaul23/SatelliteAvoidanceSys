"""``plan_maneuvers`` / ``rescreen_until_stable`` contract (Step 4).

Writes against the public maneuver, ingest, screening, and risk surfaces.
Does not import ``aegis.maneuver`` submodules.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from aegis.constants import (
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    REV_PER_DAY_TO_RAD_PER_S,
)
from aegis.core.conjunction import Conjunction, RiskAssessment
from aegis.core.maneuver import Maneuver, ManeuverPlan
from aegis.core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from aegis.core.state import StateVector
from aegis.core.timebase import shift
from aegis.ingest import (
    DataSource,
    MixedDataSourceError,
    SyntheticAuthorization,
    SyntheticSpec,
    generate_synthetic,
)
from aegis.maneuver import (
    plan_maneuvers,
    required_miss_distance_km,
    rescreen_until_stable,
)
from aegis.risk import AssessedCatalog, RankedConjunction, assess_catalog
from aegis.screening import screen

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)
_FLEET = Operator(identifier="FLEET", name="Fleet", maneuverable=True)
_EXTERNAL = Operator(identifier="EXT", name="External", maneuverable=False)


def _mean_motion_rev_per_day(altitude_km: float) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    mean_motion_rad_s = math.sqrt(MU_EARTH_KM3_S2 / semi_major_km**3)
    return mean_motion_rad_s / REV_PER_DAY_TO_RAD_PER_S


def _circular(
    object_id: str,
    altitude_km: float = 550.0,
    *,
    mean_anomaly_deg: float = 0.0,
    data_source: str = DataSource.SYNTHETIC,
    operator: Operator | None = None,
) -> SpaceObject:
    return SpaceObject(
        object_id=object_id,
        name=f"SAT-{object_id}",
        object_type=ObjectType.PAYLOAD,
        elements=OrbitalElements(
            epoch=_EPOCH,
            mean_motion_rev_per_day=_mean_motion_rev_per_day(altitude_km),
            eccentricity=0.0,
            inclination_deg=53.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=mean_anomaly_deg,
        ),
        data_source=data_source,
        operator=operator,
    )


def _state(position_km: list[float], velocity_km_s: list[float], epoch: datetime) -> StateVector:
    return StateVector(
        epoch=epoch,
        position_km=np.asarray(position_km, dtype=float),
        velocity_km_s=np.asarray(velocity_km_s, dtype=float),
    )


def _dummy_conjunction(
    primary: SpaceObject,
    secondary: SpaceObject,
    conjunction_id: str = "c-1",
    *,
    miss_km: float = 0.01,
    relative_position_rtn_km: np.ndarray | None = None,
    tca: datetime | None = None,
) -> Conjunction:
    epoch = tca if tca is not None else _EPOCH
    if relative_position_rtn_km is None:
        relative_position_rtn_km = np.array([miss_km, 0.0, 0.0])
    window_start = primary.elements.epoch if primary.elements is not None else _EPOCH
    return Conjunction(
        conjunction_id=conjunction_id,
        primary=primary,
        secondary=secondary,
        tca=epoch,
        miss_distance_km=miss_km,
        relative_speed_km_s=7.5,
        relative_position_rtn_km=relative_position_rtn_km,
        relative_velocity_rtn_km_s=np.array([0.0, 0.0, 7.5]),
        primary_state=_state([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0], epoch),
        secondary_state=_state([7000.0 + miss_km, 0.0, 0.0], [0.0, 7.5, 7.5], epoch),
        screening_window_start=window_start,
        screening_window_end=shift(window_start, 2.0 * 5800.0),
    )


def _pair_ids(conjunction: Conjunction) -> set[str]:
    return {conjunction.primary.object_id, conjunction.secondary.object_id}


def _known_pair(monkeypatch: pytest.MonkeyPatch) -> tuple[list[SpaceObject], datetime, float]:
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    catalog = generate_synthetic(
        SyntheticAuthorization(acknowledge_synthetic=True),
        SyntheticSpec(
            n_planes=1,
            sats_per_plane=3,
            include_known_conjunction_triple=True,
        ),
    )
    first, second = catalog.objects[0], catalog.objects[1]
    assert first.elements is not None
    assert first.is_maneuverable
    assert second.is_maneuverable
    duration_s = 1.5 * first.elements.period_s
    return [first, second], first.elements.epoch, duration_s


def _along_track_only(maneuver: Maneuver) -> None:
    dv = np.asarray(maneuver.delta_v_rtn_km_s, dtype=float).reshape(3)
    assert dv[0] == pytest.approx(0.0, abs=1e-12)
    assert dv[2] == pytest.approx(0.0, abs=1e-12)
    assert abs(dv[1]) == pytest.approx(maneuver.magnitude_km_s, abs=1e-12)
    assert maneuver.is_along_track_only


def test_plan_maneuvers_mixed_sources_raises() -> None:
    primary = _circular("1", operator=_FLEET, data_source=DataSource.SYNTHETIC)
    secondary = _circular(
        "2", mean_anomaly_deg=0.2, operator=_FLEET, data_source=DataSource.SYNTHETIC
    )
    extra = _circular("3", data_source=DataSource.CELESTRAK, operator=_FLEET)
    conjunction = _dummy_conjunction(primary, secondary)
    assessed = assess_catalog([conjunction])

    with pytest.raises(MixedDataSourceError):
        plan_maneuvers(assessed, [primary, secondary, extra], now=_EPOCH)


def test_non_maneuverable_objects_receive_no_burns() -> None:
    fleet = _circular("9001", operator=_FLEET)
    obstacle = _circular("9002", mean_anomaly_deg=0.2, operator=_EXTERNAL)
    assert fleet.is_maneuverable
    assert not obstacle.is_maneuverable

    period_s = fleet.elements.period_s
    tca = shift(_EPOCH, 2.0 * period_s)
    conjunction = _dummy_conjunction(
        fleet,
        obstacle,
        miss_km=0.05,
        relative_position_rtn_km=np.array([0.0, 0.05, 0.0]),
        tca=tca,
    )
    assessed = assess_catalog([conjunction], objects=[fleet, obstacle])
    plan = plan_maneuvers(assessed, [fleet, obstacle], now=_EPOCH)

    assert isinstance(plan, ManeuverPlan)
    obstacle_burns = [
        maneuver
        for maneuver in plan.all_maneuvers
        if maneuver.satellite_id == obstacle.object_id
    ]
    assert obstacle_burns == []
    if obstacle.object_id in plan.satellite_sets:
        assert plan.satellite_sets[obstacle.object_id].burn_count == 0


def test_plan_maneuvers_two_sat_along_track_separable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects, start, duration_s = _known_pair(monkeypatch)
    conjunctions = screen(objects, start, duration_s)
    pair = {objects[0].object_id, objects[1].object_id}
    matches = [c for c in conjunctions if _pair_ids(c) == pair]
    assert matches, "known-conjunction pair must screen"

    assessed = assess_catalog(matches, objects=objects)
    assert assessed.entries
    plan = plan_maneuvers(assessed, objects, now=start)

    assert isinstance(plan, ManeuverPlan)
    # Step 10: an already-safe pair must not emit a token burn.
    assert plan.total_burns == 0
    assert plan.all_maneuvers == []

    summary = plan.summary()
    assert isinstance(summary, dict)
    assert summary["plan_id"] == plan.plan_id
    assert summary["total_burns"] == plan.total_burns
    assert summary["conjunctions_addressed"] == len(plan.resolved)
    assert "converged" in summary

    target = assessed.entries[0]
    outcomes = [
        outcome
        for outcome in plan.resolved
        if outcome.conjunction_id == target.conjunction.conjunction_id
    ]
    assert outcomes, "plan must address the known pair conjunction"
    outcome = outcomes[0]

    hard_body_km = target.conjunction.combined_hard_body_radius_m / 1000.0
    required = required_miss_distance_km(
        hard_body_km,
        target.assessment.sigma_major_km,
        target.assessment.sigma_minor_km,
    )
    assert target.conjunction.miss_distance_km >= required
    assert outcome.resolved is True
    assert outcome.shortfall_km == pytest.approx(0.0, abs=1e-12)


def test_plan_maneuvers_infeasible_returns_slack() -> None:
    primary = _circular("8001", operator=_FLEET)
    secondary = _circular("8002", mean_anomaly_deg=0.2, operator=_FLEET)
    period_s = primary.elements.period_s
    tca = shift(_EPOCH, 2.0 * period_s)
    conjunction = _dummy_conjunction(
        primary,
        secondary,
        conjunction_id="infeasible-radial",
        miss_km=0.001,
        relative_position_rtn_km=np.array([0.001, 0.0, 0.0]),
        tca=tca,
    )
    assessment = RiskAssessment(
        conjunction_id=conjunction.conjunction_id,
        probability=0.2,
        method="ALFANO-2005-GAUSSCHEBYSHEV",
        hard_body_radius_m=10.0,
        miss_distance_km=0.001,
        max_probability=1.0,
        mahalanobis_distance=0.01,
        sigma_major_km=5.0,
        sigma_minor_km=1.0,
    )
    assessed = AssessedCatalog(
        source=DataSource.SYNTHETIC,
        entries=[
            RankedConjunction(conjunction=conjunction, assessment=assessment, rank=1)
        ],
    )

    plan = plan_maneuvers(
        assessed,
        [primary, secondary],
        now=_EPOCH,
        target_pc=1e-12,
        dv_budget_km_s=1e-12,
    )

    assert isinstance(plan, ManeuverPlan)
    assert plan.resolved
    unresolved = [outcome for outcome in plan.resolved if not outcome.resolved]
    assert unresolved, "tiny budget vs huge required miss must leave slack"
    for outcome in unresolved:
        assert outcome.resolved is False
        assert outcome.shortfall_km > 0.0


def test_rescreen_until_stable_two_sat_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    objects, start, duration_s = _known_pair(monkeypatch)
    plan = rescreen_until_stable(
        objects,
        start,
        duration_s,
        max_iterations=3,
    )

    assert isinstance(plan, ManeuverPlan)
    assert isinstance(plan.iterations, int)
    assert plan.iterations >= 1
    assert isinstance(plan.converged, bool)
    _ = plan.summary()
