"""Apogee/perigee prefilter contract (Step 2).

Constructs ``SpaceObject`` + ``OrbitalElements`` only — no screening internals.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from aegis.constants import MU_EARTH_KM3_S2, R_EARTH_KM, REV_PER_DAY_TO_RAD_PER_S
from aegis.core.objects import ObjectType, OrbitalElements, SpaceObject
from aegis.ingest import DataSource
from aegis.screening import prefilter_pairs

_EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)


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
        data_source=DataSource.SYNTHETIC,
    )


def test_prefilter_two_leo_550_survive() -> None:
    objects = [
        _circular("1001", 550.0, mean_anomaly_deg=0.0),
        _circular("1002", 550.0, mean_anomaly_deg=15.0),
    ]
    pairs = prefilter_pairs(objects)
    assert (0, 1) in pairs
    assert len(pairs) == 1


def test_prefilter_leo_vs_high_altitude_excluded() -> None:
    objects = [
        _circular("2001", 550.0),
        _circular("2002", 20000.0, inclination_deg=0.0),
    ]
    assert prefilter_pairs(objects) == []


def test_prefilter_pairs_are_ordered_index_tuples() -> None:
    objects = [
        _circular("1", 550.0, mean_anomaly_deg=0.0),
        _circular("2", 550.0, mean_anomaly_deg=10.0),
        _circular("3", 550.0, mean_anomaly_deg=20.0),
    ]
    pairs = prefilter_pairs(objects)
    assert pairs
    for pair in pairs:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        i, j = pair
        assert isinstance(i, int)
        assert isinstance(j, int)
        assert i < j
        assert 0 <= i < j < len(objects)


def test_prefilter_mixed_leo_and_high_altitude_keeps_only_meeting_bands() -> None:
    objects = [
        _circular("1", 550.0, mean_anomaly_deg=0.0),
        _circular("2", 550.0, mean_anomaly_deg=5.0),
        _circular("3", 20000.0, inclination_deg=0.0),
    ]
    pairs = prefilter_pairs(objects)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs


def test_prefilter_single_object_is_empty() -> None:
    assert prefilter_pairs([_circular("1", 550.0)]) == []
