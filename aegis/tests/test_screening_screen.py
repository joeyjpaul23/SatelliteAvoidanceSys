"""``screen`` / ``refine_tca`` contract (Step 2).

Uses constructed ``SpaceObject``s and the public synthetic ingest gate.
Does not import screening internals.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from aegis.constants import LOW_RELATIVE_VELOCITY_KM_S, MU_EARTH_KM3_S2, R_EARTH_KM, REV_PER_DAY_TO_RAD_PER_S
from aegis.core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from aegis.core.timebase import ensure_utc, shift
from aegis.ingest import (
    DataSource,
    MixedDataSourceError,
    SyntheticAuthorization,
    SyntheticSpec,
    generate_synthetic,
)
from aegis.propagation.propagator import Sgp4Propagator
from aegis.screening import ScreeningError, broadphase, prefilter_pairs, refine_tca, screen

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)
_FLEET = Operator(identifier="FLEET", name="Fleet", maneuverable=True)
_EXTERNAL = Operator(identifier="EXT", name="External", maneuverable=False)


def _mean_motion_rev_per_day(altitude_km: float) -> float:
    semi_major_km = R_EARTH_KM + altitude_km
    mean_motion_rad_s = math.sqrt(MU_EARTH_KM3_S2 / semi_major_km**3)
    return mean_motion_rad_s / REV_PER_DAY_TO_RAD_PER_S


def _circular(
    object_id: str,
    altitude_km: float,
    *,
    inclination_deg: float = 53.0,
    raan_deg: float = 0.0,
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
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=mean_anomaly_deg,
        ),
        data_source=data_source,
        operator=operator,
    )


def _formation_pair(
    id_a: str,
    id_b: str,
    *,
    operator_a: Operator | None = None,
    operator_b: Operator | None = None,
    data_source: str = DataSource.SYNTHETIC,
    phase_deg: float = 0.2,
) -> list[SpaceObject]:
    """Two ~550 km circular sats, ~24 km along-track — inside the Starlink box."""
    return [
        _circular(
            id_a,
            550.0,
            mean_anomaly_deg=0.0,
            data_source=data_source,
            operator=operator_a,
        ),
        _circular(
            id_b,
            550.0,
            mean_anomaly_deg=phase_deg,
            data_source=data_source,
            operator=operator_b,
        ),
    ]


def _pair_ids(conjunction) -> set[str]:
    return {conjunction.primary.object_id, conjunction.secondary.object_id}


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


def test_screening_public_exports() -> None:
    assert callable(screen)
    assert callable(prefilter_pairs)
    assert callable(broadphase)
    assert callable(refine_tca)
    assert isinstance(ScreeningError, type)


def test_screen_rejects_mixed_data_source() -> None:
    objects = [
        _circular("1", 550.0, data_source=DataSource.SYNTHETIC, mean_anomaly_deg=0.0),
        _circular("2", 550.0, data_source=DataSource.CELESTRAK, mean_anomaly_deg=0.2),
    ]
    with pytest.raises(MixedDataSourceError):
        screen(objects, _EPOCH, 600.0)


def test_screen_known_conjunction_triple(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _known_catalog(monkeypatch)
    first, second = catalog.objects[0], catalog.objects[1]
    assert first.elements is not None
    duration_s = 1.5 * first.elements.period_s
    results = screen(list(catalog), catalog.objects[0].elements.epoch, duration_s)
    pair = {first.object_id, second.object_id}
    matches = [c for c in results if _pair_ids(c) == pair]
    assert matches, (
        "screen must report the known-conjunction pair "
        f"{first.object_id}/{second.object_id} within 1.5 periods"
    )
    assert any(c.miss_distance_km < 44.0 for c in matches)


def test_screen_altitude_mismatch_no_conjunction() -> None:
    objects = [
        _circular("1", 550.0),
        _circular("2", 20000.0, inclination_deg=0.0),
    ]
    results = screen(objects, _EPOCH, 7200.0)
    assert results == []


def test_screen_conjunction_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _known_catalog(monkeypatch)
    first = catalog.objects[0]
    assert first.elements is not None
    start = first.elements.epoch
    duration_s = 1.5 * first.elements.period_s
    results = screen(list(catalog), start, duration_s)
    pair = {catalog.objects[0].object_id, catalog.objects[1].object_id}
    matches = [c for c in results if _pair_ids(c) == pair]
    assert matches
    conjunction = min(matches, key=lambda c: c.miss_distance_km)

    assert conjunction.tca.tzinfo is not None
    assert conjunction.tca.utcoffset() == timezone.utc.utcoffset(conjunction.tca)
    assert conjunction.primary_state is not None
    assert conjunction.secondary_state is not None
    assert conjunction.primary_state.position_km.shape == (3,)
    assert conjunction.secondary_state.position_km.shape == (3,)
    assert conjunction.primary_state.velocity_km_s.shape == (3,)
    assert conjunction.secondary_state.velocity_km_s.shape == (3,)
    assert np.asarray(conjunction.relative_position_rtn_km).shape == (3,)
    assert np.asarray(conjunction.relative_velocity_rtn_km_s).shape == (3,)
    assert conjunction.screening_window_start is not None
    assert conjunction.screening_window_end is not None
    assert ensure_utc(conjunction.screening_window_start) == ensure_utc(start)
    assert ensure_utc(conjunction.screening_window_end) == shift(start, duration_s)
    assert conjunction.miss_distance_km < 44.0
    assert conjunction.relative_speed_km_s >= 0.0


def test_screen_conjunction_id_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _known_catalog(monkeypatch)
    first = catalog.objects[0]
    assert first.elements is not None
    start = first.elements.epoch
    duration_s = 1.5 * first.elements.period_s
    objects = list(catalog)
    first_pass = screen(objects, start, duration_s)
    second_pass = screen(objects, start, duration_s)
    assert [c.conjunction_id for c in first_pass] == [c.conjunction_id for c in second_pass]
    assert all(c.conjunction_id for c in first_pass)


def test_screen_primary_exactly_one_maneuverable() -> None:
    objects = _formation_pair(
        "3001",
        "3000",
        operator_a=_FLEET,
        operator_b=_EXTERNAL,
    )
    results = screen(objects, _EPOCH, 600.0)
    assert results
    for conjunction in results:
        if _pair_ids(conjunction) != {"3001", "3000"}:
            continue
        assert conjunction.primary.object_id == "3001"
        assert conjunction.secondary.object_id == "3000"
        return
    raise AssertionError("expected a conjunction between 3001 and 3000")


@pytest.mark.parametrize(
    "operator_a,operator_b",
    [
        (None, None),
        (_EXTERNAL, _EXTERNAL),
        (_FLEET, _FLEET),
    ],
)
def test_screen_primary_lower_object_id_when_not_exactly_one_maneuverable(
    operator_a: Operator | None,
    operator_b: Operator | None,
) -> None:
    objects = _formation_pair(
        "5001",
        "5000",
        operator_a=operator_a,
        operator_b=operator_b,
    )
    results = screen(objects, _EPOCH, 600.0)
    assert results
    for conjunction in results:
        if _pair_ids(conjunction) != {"5000", "5001"}:
            continue
        assert conjunction.primary.object_id == "5000"
        assert conjunction.secondary.object_id == "5001"
        return
    raise AssertionError("expected a conjunction between 5000 and 5001")


def test_screen_low_relative_velocity_metadata() -> None:
    objects = _formation_pair("6000", "6001", operator_a=_FLEET, operator_b=_FLEET)
    results = screen(objects, _EPOCH, 600.0)
    assert results
    saw_low = False
    for conjunction in results:
        speed = float(conjunction.relative_speed_km_s)
        flag = conjunction.metadata.get("low_relative_velocity")
        if speed < LOW_RELATIVE_VELOCITY_KM_S:
            assert flag is True
            saw_low = True
        else:
            assert flag is not True
    assert saw_low, (
        "formation pair at ~24 km along-track should have relative speed "
        f"below {LOW_RELATIVE_VELOCITY_KM_S} km/s"
    )


def test_refine_tca_miss_not_worse_than_guess() -> None:
    """Crossing LEO pair: equatorial vs high inclination, same node at epoch."""
    objects = [
        _circular("1", 550.0, inclination_deg=0.0, raan_deg=0.0, mean_anomaly_deg=0.0),
        _circular("2", 550.0, inclination_deg=80.0, raan_deg=0.0, mean_anomaly_deg=0.0),
    ]
    propagator = Sgp4Propagator(objects)
    start = shift(_EPOCH, -90.0)
    grid = propagator.propagate_grid(start, duration_s=180.0, step_s=15.0)
    t_guess = None
    for k in range(grid.n_times):
        if not (grid.valid[0, k] and grid.valid[1, k]):
            continue
        relative_position = grid.positions_km[0, k] - grid.positions_km[1, k]
        relative_velocity = grid.velocities_km_s[0, k] - grid.velocities_km_s[1, k]
        if float(relative_position @ relative_velocity) < 0.0:
            t_guess = grid.epoch_at(k)
            break
    assert t_guess is not None, "crossing pair must have an approaching sample"

    state_a, state_b = propagator.propagate_pair(0, 1, t_guess)
    miss_at_guess = float(np.linalg.norm(state_a.position_km - state_b.position_km))
    conjunction = refine_tca(propagator, 0, 1, t_guess)
    assert conjunction.miss_distance_km <= miss_at_guess


def test_refine_tca_relative_state_orthogonal_when_fast() -> None:
    objects = [
        _circular("1", 550.0, inclination_deg=0.0, raan_deg=0.0, mean_anomaly_deg=0.0),
        _circular("2", 550.0, inclination_deg=80.0, raan_deg=0.0, mean_anomaly_deg=0.0),
    ]
    propagator = Sgp4Propagator(objects)
    conjunction = refine_tca(propagator, 0, 1, shift(_EPOCH, -20.0))
    relative_position = (
        conjunction.primary_state.position_km - conjunction.secondary_state.position_km
    )
    relative_velocity = (
        conjunction.primary_state.velocity_km_s - conjunction.secondary_state.velocity_km_s
    )
    speed = float(np.linalg.norm(relative_velocity))
    assert speed > LOW_RELATIVE_VELOCITY_KM_S
    miss = float(np.linalg.norm(relative_position))
    assert miss > 0.0
    cosine = abs(float(relative_position @ relative_velocity)) / (miss * speed)
    assert cosine < 1e-3


def test_screen_includes_intra_fleet_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _known_catalog(monkeypatch)
    first, second = catalog.objects[0], catalog.objects[1]
    assert first.operator is not None and second.operator is not None
    assert first.operator.maneuverable is True
    assert second.operator.maneuverable is True
    assert first.elements is not None
    duration_s = 1.5 * first.elements.period_s
    results = screen(list(catalog), first.elements.epoch, duration_s)
    pair = {first.object_id, second.object_id}
    matches = [c for c in results if _pair_ids(c) == pair]
    assert matches, "intra-fleet pairs must not be excluded from screening"
    assert any(c.is_intra_fleet for c in matches)
    assert any(c.miss_distance_km < 44.0 for c in matches)
