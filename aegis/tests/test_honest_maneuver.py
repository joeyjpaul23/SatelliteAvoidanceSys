"""Physically honest maneuvers (Step 10).

Writes against the public maneuver, ingest, screening, and risk surfaces.
Does not import ``aegis.maneuver`` submodules.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from aegis.constants import (
    DEFAULT_DV_BUDGET_KM_S,
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    REV_PER_DAY_TO_RAD_PER_S,
)
from aegis.core.conjunction import Conjunction, RiskAssessment
from aegis.core.maneuver import ManeuverPlan
from aegis.core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from aegis.core.state import StateVector
from aegis.core.timebase import seconds_between, shift
from aegis.ingest import (
    DataSource,
    SyntheticAuthorization,
    SyntheticSpec,
    generate_synthetic,
)
from aegis.maneuver import (
    along_track_response_km,
    apply_along_track_burns,
    plan_maneuvers,
    required_miss_distance_km,
)
from aegis.propagation.propagator import Sgp4Propagator
from aegis.risk import AssessedCatalog, RankedConjunction, assess_catalog
from aegis.screening import screen

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)
_FLEET = Operator(identifier="FLEET", name="Fleet", maneuverable=True)
_CLOSE_MISS_KM = 0.05


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
    hard_body_radius_m: float | None = None,
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
        hard_body_radius_m=hard_body_radius_m,
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


def _along_track_close_pair(miss_km: float = _CLOSE_MISS_KM) -> list[SpaceObject]:
    """Two maneuverable circular sats with a ~``miss_km`` along-track offset."""
    phase_deg = math.degrees(miss_km / (R_EARTH_KM + 550.0))
    return [
        _circular("7001", operator=_FLEET, mean_anomaly_deg=0.0, hard_body_radius_m=5.0),
        _circular("7002", operator=_FLEET, mean_anomaly_deg=phase_deg, hard_body_radius_m=5.0),
    ]


def _along_track_only(maneuver) -> None:
    dv = np.asarray(maneuver.delta_v_rtn_km_s, dtype=float).reshape(3)
    assert dv[0] == pytest.approx(0.0, abs=1e-12)
    assert dv[2] == pytest.approx(0.0, abs=1e-12)
    assert abs(dv[1]) == pytest.approx(maneuver.magnitude_km_s, abs=1e-12)
    assert maneuver.is_along_track_only
    assert abs(float(dv[1])) > 1e-12


def _primary_rtn_y(primary: StateVector, secondary: StateVector) -> float:
    rotation = primary.rtn_to_eci()
    relative = rotation.T @ (secondary.position_km - primary.position_km)
    return float(relative[1])


def _states_at(objects: list[SpaceObject], epoch: datetime) -> dict[str, StateVector]:
    propagator = Sgp4Propagator(objects)
    index = {object_id: i for i, object_id in enumerate(propagator.object_ids)}
    return {
        object_id: propagator.propagate_one(index[object_id], epoch)
        for object_id in index
    }


def _intended_relative_dy_km(
    plan: ManeuverPlan,
    objects: list[SpaceObject],
    primary_id: str,
    tca: datetime,
) -> float:
    """CW along-track change of secondary-minus-primary in primary RTN."""
    by_id = {obj.object_id: obj for obj in objects}
    intended = 0.0
    for maneuver in plan.all_maneuvers:
        sat = by_id[maneuver.satellite_id]
        assert sat.elements is not None
        dt_s = seconds_between(maneuver.epoch, tca)
        dv = float(np.asarray(maneuver.delta_v_rtn_km_s, dtype=float).reshape(3)[1])
        dy = float(along_track_response_km(sat.elements.mean_motion_rad_s, dt_s, dv))
        if maneuver.satellite_id == primary_id:
            intended -= dy
        else:
            intended += dy
    return intended


def test_known_triple_already_safe_emits_zero_burns(
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
    assert plan.total_burns == 0
    assert plan.all_maneuvers == []

    target = assessed.entries[0]
    outcomes = [
        outcome
        for outcome in plan.resolved
        if outcome.conjunction_id == target.conjunction.conjunction_id
    ]
    assert outcomes, "plan must still list the already-safe pair"
    for outcome in outcomes:
        assert outcome.resolved is True
        assert outcome.shortfall_km == pytest.approx(0.0, abs=1e-12)


def test_constructed_along_track_close_pair_gets_real_burn() -> None:
    objects = _along_track_close_pair()
    assert objects[0].is_maneuverable
    assert objects[1].is_maneuverable
    assert objects[0].elements is not None
    period_s = objects[0].elements.period_s
    start = objects[0].elements.epoch
    duration_s = 1.5 * period_s

    conjunctions = screen(objects, start, duration_s)
    pair = {objects[0].object_id, objects[1].object_id}
    matches = [c for c in conjunctions if _pair_ids(c) == pair]
    assert matches, "constructed along-track pair must screen"
    closest = min(matches, key=lambda c: c.miss_distance_km)
    assert closest.miss_distance_km == pytest.approx(_CLOSE_MISS_KM, rel=0.75, abs=0.05)

    assessed = assess_catalog(matches, objects=objects)
    assert assessed.entries
    target = next(
        entry
        for entry in assessed.entries
        if entry.conjunction.conjunction_id == closest.conjunction_id
    )
    hard_body_km = target.conjunction.combined_hard_body_radius_m / 1000.0
    required = required_miss_distance_km(
        hard_body_km,
        target.assessment.sigma_major_km,
        target.assessment.sigma_minor_km,
    )
    assert closest.miss_distance_km < required

    now = shift(start, -2.0 * period_s)
    plan = plan_maneuvers(assessed, objects, now=now)

    assert isinstance(plan, ManeuverPlan)
    assert plan.total_burns >= 1
    assert plan.all_maneuvers
    for maneuver in plan.all_maneuvers:
        _along_track_only(maneuver)

    outcomes = [
        outcome
        for outcome in plan.resolved
        if outcome.conjunction_id == closest.conjunction_id
    ]
    assert outcomes, "plan must address the constructed close pair"
    outcome = outcomes[0]
    n = objects[0].elements.mean_motion_rad_s
    lead_s = 2.0 * period_s
    available = 2.0 * abs(along_track_response_km(n, lead_s, DEFAULT_DV_BUDGET_KM_S))
    gap = max(0.0, required - closest.miss_distance_km)
    if available >= gap:
        assert outcome.resolved is True


def test_apply_along_track_burns_increases_rescreen_miss() -> None:
    objects = _along_track_close_pair()
    assert objects[0].elements is not None
    period_s = objects[0].elements.period_s
    start = objects[0].elements.epoch
    duration_s = 1.5 * period_s
    pair = {objects[0].object_id, objects[1].object_id}

    before = screen(objects, start, duration_s)
    before_matches = [c for c in before if _pair_ids(c) == pair]
    assert before_matches, "constructed pair must screen before the burn"
    pre = min(before_matches, key=lambda c: c.miss_distance_km)
    pre_miss = pre.miss_distance_km

    assessed = assess_catalog(before_matches, objects=objects)
    now = shift(start, -2.0 * period_s)
    plan = plan_maneuvers(assessed, objects, now=now)
    assert plan.total_burns >= 1
    assert any(abs(float(m.delta_v_rtn_km_s[1])) > 1e-12 for m in plan.all_maneuvers)

    updated = apply_along_track_burns(objects, plan, assessed)
    assert isinstance(updated, list)
    assert {obj.object_id for obj in updated} == pair

    after = screen(updated, start, duration_s)
    after_matches = [c for c in after if _pair_ids(c) == pair]
    if after_matches:
        post_miss = min(c.miss_distance_km for c in after_matches)
        assert post_miss > pre_miss
    # else: pair dropped out of the screening box — also acceptable

    tca = pre.tca
    before_states = _states_at(objects, tca)
    after_states = _states_at(updated, tca)
    primary_id = pre.primary.object_id
    secondary_id = pre.secondary.object_id
    y_before = _primary_rtn_y(before_states[primary_id], before_states[secondary_id])
    y_after = _primary_rtn_y(after_states[primary_id], after_states[secondary_id])
    measured = y_after - y_before
    intended = _intended_relative_dy_km(plan, objects, primary_id, tca)
    assert measured * intended > 0.0 or abs(intended) < 1e-9
    within_relative = abs(abs(measured) - abs(intended)) <= 0.5 * abs(intended)
    within_absolute = abs(abs(measured) - abs(intended)) <= 2.0
    assert within_relative or within_absolute


def test_plan_maneuvers_infeasible_radial_still_slack() -> None:
    primary = _circular("8001", operator=_FLEET)
    secondary = _circular("8002", mean_anomaly_deg=0.2, operator=_FLEET)
    assert primary.elements is not None
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
