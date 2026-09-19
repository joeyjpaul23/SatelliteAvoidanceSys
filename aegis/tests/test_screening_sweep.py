"""Streaming sweep: event splitting, block-edge stitching, parallel = serial."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aegis.constants import MU_EARTH_KM3_S2, R_EARTH_KM, REV_PER_DAY_TO_RAD_PER_S
from aegis.core.objects import ObjectType, Operator, OrbitalElements, SpaceObject
from aegis.propagation.propagator import Sgp4Propagator
from aegis.screening import screen
from aegis.screening import sweep as sweep_module
from aegis.screening.sweep import _events_from_rows, _stitch, make_spec, sweep

EPOCH = datetime(2026, 9, 18, tzinfo=timezone.utc)
FLEET = Operator(identifier="TEST", name="Test", organization="Test", maneuverable=True)


def _satellite(object_id: str, *, mean_anomaly_deg: float, inclination_deg: float = 53.0, raan_deg: float = 0.0):
    semi_major = R_EARTH_KM + 550.0
    rev_per_day = math.sqrt(MU_EARTH_KM3_S2 / semi_major**3) / REV_PER_DAY_TO_RAD_PER_S
    return SpaceObject(
        object_id=object_id,
        name=f"SAT-{object_id}",
        object_type=ObjectType.PAYLOAD,
        elements=OrbitalElements(
            epoch=EPOCH,
            mean_motion_rev_per_day=rev_per_day,
            eccentricity=0.0,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=mean_anomaly_deg,
        ),
        data_source="SYNTHETIC",
        operator=FLEET,
    )


def _crowd(n: int = 40) -> list[SpaceObject]:
    """Satellites in two nearby planes: repeated crossings plus a close co-orbiting pair."""
    objects = [
        _satellite(f"{8000 + k}", mean_anomaly_deg=k * 0.35, raan_deg=0.0 if k % 2 else 0.6)
        for k in range(n)
    ]
    objects.append(_satellite("7001", mean_anomaly_deg=100.0))
    objects.append(_satellite("7002", mean_anomaly_deg=100.0 + math.degrees(0.05 / (R_EARTH_KM + 550.0))))
    return objects


# ---------------------------------------------------------------------------
# Event splitting and stitching (logic, on hand-built rows)
# ---------------------------------------------------------------------------


def _rows(ranges: list[float], boxes: list[bool], start: int = 0):
    """One pair's kept intervals over consecutive samples with given ranges."""
    rows = []
    for offset in range(len(ranges) - 1):
        rows.append(
            (
                np.array([0]),
                np.array([1]),
                np.array([ranges[offset]]),
                np.array([ranges[offset + 1]]),
                np.array([boxes[offset]]),
                np.array([boxes[offset + 1]]),
                np.array([True]),
                start + offset,
            )
        )
    return rows


def test_leaving_the_box_between_approaches_makes_two_events() -> None:
    ranges = [30.0, 5.0, 30.0, 60.0, 30.0, 8.0, 30.0]
    boxes = [False, True, False, False, False, True, False]
    seeds = _stitch([_events_from_rows(_rows(ranges, boxes), 2, (0, 6), 7)], 2)
    assert seeds.sample.tolist() == [1, 5]
    assert seeds.entered.tolist() == [True, True]


def test_staying_inside_the_box_is_one_event_however_the_range_wiggles() -> None:
    ranges = [0.050, 0.049, 0.051, 0.048, 0.052, 0.047, 0.050]
    seeds = _stitch([_events_from_rows(_rows(ranges, [True] * 7), 2, (0, 6), 7)], 2)
    assert seeds.sample.tolist() == [5]


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5])
def test_block_edges_do_not_change_events(cut: int) -> None:
    ranges = [30.0, 5.0, 30.0, 60.0, 30.0, 8.0, 30.0]
    boxes = [False, True, False, False, False, True, False]
    whole = _stitch([_events_from_rows(_rows(ranges, boxes), 2, (0, 6), 7)], 2)
    left = _events_from_rows(_rows(ranges[: cut + 1], boxes[: cut + 1]), 2, (0, cut), 7)
    right = _events_from_rows(_rows(ranges[cut:], boxes[cut:], start=cut), 2, (cut, 6), 7)
    split = _stitch([left, right], 2)
    assert split.sample.tolist() == whole.sample.tolist()
    assert split.entered.tolist() == whole.entered.tolist()


# ---------------------------------------------------------------------------
# Real propagation: block sizes and worker processes do not change results
# ---------------------------------------------------------------------------


def _seed_set(seeds) -> set[tuple[int, int, int]]:
    return set(zip(seeds.index_a.tolist(), seeds.index_b.tolist(), seeds.sample.tolist(), strict=True))


def test_sweep_is_independent_of_block_size(monkeypatch) -> None:
    objects = _crowd()
    propagator = Sgp4Propagator(objects)
    spec = make_spec(propagator, EPOCH, 3 * 3600.0, 30.0, (2.0, 44.0, 51.0))
    one_block = _seed_set(sweep(propagator, spec))
    monkeypatch.setattr(sweep_module, "_BLOCK_BYTES", 1)  # forces 16-sample blocks
    many_blocks = _seed_set(sweep(propagator, spec))
    assert one_block, "the crowd must produce approaches"
    assert many_blocks == one_block


def _ids(conjunctions) -> list[str]:
    return [conjunction.conjunction_id for conjunction in conjunctions]


def test_parallel_screen_matches_serial(monkeypatch) -> None:
    objects = _crowd()
    serial = screen(objects, EPOCH, 3 * 3600.0, workers=1)
    monkeypatch.setattr(sweep_module, "_PARALLEL_MIN_WORK", 0)
    monkeypatch.setattr(sweep_module, "_BLOCK_BYTES", 1)
    parallel = screen(objects, EPOCH, 3 * 3600.0, workers=2)
    assert serial
    assert _ids(parallel) == _ids(serial)


def test_co_orbiting_pair_is_one_conjunction_per_window() -> None:
    pair = [obj for obj in _crowd() if obj.object_id in {"7001", "7002"}]
    period_s = pair[0].elements.period_s
    conjunctions = screen(pair, EPOCH + timedelta(minutes=5), 1.5 * period_s)
    assert len(conjunctions) == 1
    assert conjunctions[0].miss_distance_km < 0.1


# ---------------------------------------------------------------------------
# Window edges and duplicate element sets
# ---------------------------------------------------------------------------


def _fast_passes(conjunctions) -> dict[tuple[str, str], list[float]]:
    passes: dict[tuple[str, str], list[float]] = {}
    for c in conjunctions:
        if not c.metadata.get("low_relative_velocity"):
            key = (c.primary.object_id, c.secondary.object_id)
            passes.setdefault(key, []).append((c.tca - EPOCH).total_seconds())
    return passes


def test_an_approach_in_the_final_partial_step_is_found() -> None:
    # A window that is not a whole number of steps still screens its last
    # partial step, and keeps no fast pass whose TCA is past the end.
    objects = _crowd() + [  # steeper planes crossing the crowd at 1-2 km/s
        _satellite(f"{9100 + k}", mean_anomaly_deg=anomaly, inclination_deg=inclination)
        for k, (inclination, anomaly) in enumerate([(63.0, 100.0), (63.0, 100.02), (43.0, 100.05), (70.0, 99.97)])
    ]
    reference = _fast_passes(screen(objects, EPOCH, 3 * 3600.0, step_s=60.0))
    # The pass furthest past a grid sample: at that sample the pair is still far apart.
    late = [(tca % 60.0, pair, tca) for pair, tcas in reference.items() for tca in tcas if tca % 60.0 < 57.0]
    assert late, "the crowd must have a pass away from the 60 s grid"
    _phase, pair, tca = max(late)
    duration = tca + 2.0  # the pass sits in the final, partial step
    found = _fast_passes(screen(objects, EPOCH, duration, step_s=60.0))
    assert any(abs(t - tca) < 1.0 for t in found.get(pair, []))
    assert all(0.0 <= t <= duration + 1e-3 for tcas in found.values() for t in tcas)


def test_duplicate_element_sets_for_one_object_never_pair() -> None:
    first = _satellite("5", mean_anomaly_deg=10.0)
    second = _satellite("5", mean_anomaly_deg=10.001)
    assert screen([first, second], EPOCH, 3600.0) == []
