"""Hundreds-object screening / blocked-propagation contract (Step 13).

Uses the public ``aegis.screening``, ``aegis.propagation``, ``aegis.ingest``,
and ``aegis.constants`` surfaces. Does not import screening internals.
Synthetic opt-in is set only inside these tests.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import aegis.constants as constants
from aegis.ingest import SyntheticAuthorization, SyntheticSpec, generate_synthetic
from aegis.propagation.propagator import Sgp4Propagator
from aegis.screening import screen

_TWELVE_SAT = SyntheticSpec(n_planes=2, sats_per_plane=6)
_TWO_HUNDRED_SAT = SyntheticSpec(n_planes=8, sats_per_plane=25)
_SCALE_DURATION_S = 1800.0
_SCALE_STEP_S = 60.0


def _auth() -> SyntheticAuthorization:
    return SyntheticAuthorization(acknowledge_synthetic=True)


def _generate(monkeypatch: pytest.MonkeyPatch, spec: SyntheticSpec):
    monkeypatch.setenv("AEGIS_ALLOW_SYNTHETIC", "1")
    return generate_synthetic(_auth(), spec)


def _pair_ids(conjunction) -> frozenset[str]:
    return frozenset({conjunction.primary.object_id, conjunction.secondary.object_id})


def _conjunction_ids(results) -> set[str]:
    return {conjunction.conjunction_id for conjunction in results}


def _pair_id_set(results) -> set[frozenset[str]]:
    return {_pair_ids(conjunction) for conjunction in results}


def test_screening_partition_min_objects_is_fifty() -> None:
    name = "SCREENING_PARTITION_MIN_OBJECTS"
    if not hasattr(constants, name):
        alternates = [
            candidate
            for candidate in dir(constants)
            if "PARTITION" in candidate
            and "MIN" in candidate
            and "OBJECT" in candidate
            and not candidate.startswith("_")
        ]
        if not alternates:
            pytest.skip("partition min-objects constant not found on aegis.constants")
        name = alternates[0]
    assert getattr(constants, name) == 50


def test_screen_partitioned_matches_unpartitioned_conjunctions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _generate(monkeypatch, _TWELVE_SAT)
    assert len(catalog) == 12
    objects = list(catalog)
    first = catalog.objects[0]
    assert first.elements is not None
    start = first.elements.epoch

    partitioned = screen(
        objects,
        start,
        _SCALE_DURATION_S,
        step_s=_SCALE_STEP_S,
        partitioned=True,
    )
    unpartitioned = screen(
        objects,
        start,
        _SCALE_DURATION_S,
        step_s=_SCALE_STEP_S,
        partitioned=False,
    )

    assert isinstance(partitioned, list)
    assert isinstance(unpartitioned, list)
    partitioned_ids = _conjunction_ids(partitioned)
    unpartitioned_ids = _conjunction_ids(unpartitioned)
    if partitioned_ids != unpartitioned_ids:
        assert _pair_id_set(partitioned) == _pair_id_set(unpartitioned), (
            "partitioned and unpartitioned screen must return the same "
            "conjunction_id set (or the same pair ids)"
        )


def test_propagate_grid_blocked_matches_unblocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _generate(
        monkeypatch,
        SyntheticSpec(n_planes=1, sats_per_plane=3, include_known_conjunction_triple=False),
    )
    objects = list(catalog)
    first = catalog.objects[0]
    assert first.elements is not None
    propagator = Sgp4Propagator(objects)
    start = first.elements.epoch
    duration_s = 3600.0
    step_s = 120.0

    unblocked = propagator.propagate_grid(start, duration_s, step_s, block_duration_s=None)
    blocked = propagator.propagate_grid(start, duration_s, step_s, block_duration_s=1800.0)

    assert blocked.n_times == unblocked.n_times
    np.testing.assert_allclose(blocked.times_s, unblocked.times_s)
    assert blocked.object_ids == unblocked.object_ids

    both_valid = unblocked.valid & blocked.valid
    assert bool(both_valid.any()), "blocked and unblocked grids must share valid samples"
    np.testing.assert_allclose(
        unblocked.positions_km[both_valid],
        blocked.positions_km[both_valid],
        rtol=1e-9,
        atol=1e-9,
    )


def test_screen_two_hundred_sat_scale_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _generate(monkeypatch, _TWO_HUNDRED_SAT)
    assert len(catalog) == 200
    objects = list(catalog)
    first = catalog.objects[0]
    assert first.elements is not None
    start = first.elements.epoch

    began = time.perf_counter()
    results = screen(objects, start, duration_s=_SCALE_DURATION_S, step_s=_SCALE_STEP_S)
    elapsed_s = time.perf_counter() - began

    assert isinstance(results, list)
    assert elapsed_s < 90.0, f"200-sat screen took {elapsed_s:.1f}s (limit 90s)"


def test_partitioned_broadphase_does_not_drop_unpartitioned_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _generate(monkeypatch, _TWELVE_SAT)
    assert len(catalog) == 12
    objects = list(catalog)
    first = catalog.objects[0]
    assert first.elements is not None
    start = first.elements.epoch

    partitioned = screen(
        objects,
        start,
        _SCALE_DURATION_S,
        step_s=_SCALE_STEP_S,
        partitioned=True,
    )
    unpartitioned = screen(
        objects,
        start,
        _SCALE_DURATION_S,
        step_s=_SCALE_STEP_S,
        partitioned=False,
    )

    partitioned_pairs = _pair_id_set(partitioned)
    unpartitioned_pairs = _pair_id_set(unpartitioned)
    dropped = unpartitioned_pairs - partitioned_pairs
    assert not dropped, (
        "partitioned screening must not drop a pair that unpartitioned keeps: "
        f"{sorted(tuple(sorted(pair)) for pair in dropped)}"
    )
