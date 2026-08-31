"""Time-of-closest-approach refinement."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..constants import LOW_RELATIVE_VELOCITY_KM_S, SCREENING_STEP_S
from ..core.conjunction import Conjunction
from ..core.frames import enforce_tca
from ..core.objects import SpaceObject
from ..core.state import Ephemeris, StateVector
from ..core.timebase import ensure_utc, shift
from ..propagation.propagator import PropagationError, Sgp4Propagator
from .errors import ScreeningError
from .geometry import assign_primary, conjunction_id, relative_rtn

__all__ = ["refine_tca"]

#: Coarse bracket half-width, in screening steps.
_COARSE_HALF_STEPS = 4
#: Times the coarse bracket may expand if the minimum sits on the edge.
_COARSE_EXPAND = 3
#: Hermite search spacing inside the coarse bracket, seconds.
_HERMITE_STEP_S = 0.5
#: Linear-CA iterations after the sampled minimum.
_LINEAR_ITERS = 3
#: Clamp on a single linear-CA step, in screening steps.
_LINEAR_CLAMP_STEPS = 2


def _propagate_pair(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    epoch: datetime,
) -> tuple[StateVector, StateVector]:
    try:
        return propagator.propagate_pair(index_a, index_b, epoch)
    except PropagationError as error:
        raise ScreeningError(str(error)) from error
    except Exception as error:  # noqa: BLE001 - sgp4 / index failures
        raise ScreeningError(
            f"could not propagate pair {index_a}/{index_b} at {epoch}: {error}"
        ) from error


def _range_km(state_a: StateVector, state_b: StateVector) -> float:
    return float(np.linalg.norm(state_a.position_km - state_b.position_km))


def _try_propagate_pair(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    epoch: datetime,
) -> tuple[StateVector, StateVector] | None:
    try:
        return _propagate_pair(propagator, index_a, index_b, epoch)
    except ScreeningError:
        return None


def _sample_offsets(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    center: datetime,
    offsets_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[StateVector, StateVector]]]:
    """Propagate at ``center + offset``; skip samples that fail SGP4."""
    ranges: list[float] = []
    kept: list[float] = []
    states: list[tuple[StateVector, StateVector]] = []
    for offset in offsets_s:
        pair = _try_propagate_pair(propagator, index_a, index_b, shift(center, float(offset)))
        if pair is None:
            continue
        kept.append(float(offset))
        states.append(pair)
        ranges.append(_range_km(*pair))
    if not ranges:
        raise ScreeningError(
            f"could not propagate pair {index_a}/{index_b} near {center}"
        )
    return np.asarray(kept), np.asarray(ranges), states


def _coarse_minimum(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    t_guess: datetime,
    step_s: float,
) -> tuple[datetime, float, StateVector, StateVector]:
    """Bracket a few screening steps and return the sampled minimum."""
    half_steps = _COARSE_HALF_STEPS
    for _ in range(_COARSE_EXPAND):
        offsets = np.arange(-half_steps, half_steps + 1, dtype=float) * step_s
        kept, ranges, states = _sample_offsets(propagator, index_a, index_b, t_guess, offsets)
        best = int(np.argmin(ranges))
        on_edge = kept[best] in (kept[0], kept[-1]) and abs(kept[best]) >= half_steps * step_s - 1e-9
        if not on_edge:
            state_a, state_b = states[best]
            return shift(t_guess, float(kept[best])), float(ranges[best]), state_a, state_b
        half_steps *= 2

    best = int(np.argmin(ranges))
    state_a, state_b = states[best]
    return shift(t_guess, float(kept[best])), float(ranges[best]), state_a, state_b


def _hermite_minimum_offset(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    t_center: datetime,
    kept: np.ndarray,
    states: list[tuple[StateVector, StateVector]],
) -> float:
    """Offset of minimum interpolated range, seconds from ``t_center``."""
    if len(kept) < 2:
        return 0.0
    eph_a = Ephemeris(
        object_id=propagator.object_ids[index_a],
        epochs=kept,
        positions_km=np.stack([pair[0].position_km for pair in states]),
        velocities_km_s=np.stack([pair[0].velocity_km_s for pair in states]),
        reference_epoch=t_center,
    )
    eph_b = Ephemeris(
        object_id=propagator.object_ids[index_b],
        epochs=kept,
        positions_km=np.stack([pair[1].position_km for pair in states]),
        velocities_km_s=np.stack([pair[1].velocity_km_s for pair in states]),
        reference_epoch=t_center,
    )
    span = float(kept[-1] - kept[0])
    n_fine = max(int(np.ceil(span / _HERMITE_STEP_S)) + 1, 3)
    fine = np.linspace(float(kept[0]), float(kept[-1]), n_fine)
    best_offset = float(kept[0])
    best_range = np.inf
    for offset in fine:
        position_a, _ = eph_a.interpolate(float(offset))
        position_b, _ = eph_b.interpolate(float(offset))
        separation = float(np.linalg.norm(position_a - position_b))
        if separation < best_range:
            best_range = separation
            best_offset = float(offset)
    return best_offset


def _fine_minimum(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    t_center: datetime,
    step_s: float,
) -> tuple[datetime, float, StateVector, StateVector]:
    """Hermite refine around ``t_center``, then linear closest-approach."""
    half = max(step_s, SCREENING_STEP_S)
    n_each = 2
    offsets = np.arange(-n_each, n_each + 1, dtype=float) * half
    kept, ranges, states = _sample_offsets(propagator, index_a, index_b, t_center, offsets)
    hermite_offset = _hermite_minimum_offset(
        propagator, index_a, index_b, t_center, kept, states
    )
    sample_best = int(np.argmin(ranges))
    epoch = shift(t_center, hermite_offset)
    polished = _try_propagate_pair(propagator, index_a, index_b, epoch)
    if polished is None:
        epoch = shift(t_center, float(kept[sample_best]))
        state_a, state_b = states[sample_best]
        miss = float(ranges[sample_best])
    else:
        state_a, state_b = polished
        miss = _range_km(state_a, state_b)
        if miss > float(ranges[sample_best]):
            epoch = shift(t_center, float(kept[sample_best]))
            state_a, state_b = states[sample_best]
            miss = float(ranges[sample_best])

    for _ in range(_LINEAR_ITERS):
        relative_velocity = state_a.velocity_km_s - state_b.velocity_km_s
        speed = float(np.linalg.norm(relative_velocity))
        if speed < 1e-6:
            break
        _, _, _, _, dt = enforce_tca(
            state_a.position_km,
            state_a.velocity_km_s,
            state_b.position_km,
            state_b.velocity_km_s,
        )
        if abs(dt) < 1e-9:
            break
        dt = float(np.clip(dt, -_LINEAR_CLAMP_STEPS * step_s, _LINEAR_CLAMP_STEPS * step_s))
        candidate_epoch = shift(epoch, dt)
        try:
            candidate_a, candidate_b = _propagate_pair(
                propagator, index_a, index_b, candidate_epoch
            )
        except ScreeningError:
            break
        candidate_miss = _range_km(candidate_a, candidate_b)
        if candidate_miss > miss:
            break
        epoch = candidate_epoch
        state_a, state_b = candidate_a, candidate_b
        miss = candidate_miss

    return epoch, miss, state_a, state_b


def _build_conjunction(
    primary: SpaceObject,
    secondary: SpaceObject,
    primary_state: StateVector,
    secondary_state: StateVector,
    tca: datetime,
) -> Conjunction:
    position_rtn, velocity_rtn = relative_rtn(primary_state, secondary_state)
    miss = float(np.linalg.norm(primary_state.position_km - secondary_state.position_km))
    speed = float(np.linalg.norm(primary_state.velocity_km_s - secondary_state.velocity_km_s))
    metadata: dict = {}
    if speed < LOW_RELATIVE_VELOCITY_KM_S:
        metadata["low_relative_velocity"] = True
    return Conjunction(
        conjunction_id=conjunction_id(primary.object_id, secondary.object_id, tca),
        primary=primary,
        secondary=secondary,
        tca=tca,
        miss_distance_km=miss,
        relative_speed_km_s=speed,
        relative_position_rtn_km=position_rtn,
        relative_velocity_rtn_km_s=velocity_rtn,
        primary_state=primary_state,
        secondary_state=secondary_state,
        metadata=metadata,
    )


def refine_tca(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    t_guess: datetime,
) -> Conjunction:
    """Refine closest approach near ``t_guess``.

    Brackets a few screening steps, samples densely around the coarse
    minimum, then applies a clamped linear closest-approach correction
    via :func:`~aegis.core.frames.enforce_tca` and ``propagate_one``.
    Miss at the returned TCA is at most the miss at ``t_guess`` for a
    pair that is approaching. When relative speed is above
    :data:`LOW_RELATIVE_VELOCITY_KM_S`, relative ``r · v`` is driven
    near zero.
    """
    if index_a == index_b:
        raise ScreeningError("refine_tca requires two distinct object indices")

    try:
        objects = propagator.objects
        object_a = objects[index_a]
        object_b = objects[index_b]
    except (AttributeError, IndexError) as error:
        raise ScreeningError(f"invalid object indices {index_a}, {index_b}") from error

    t_guess = ensure_utc(t_guess)
    step_s = float(SCREENING_STEP_S)

    state_guess_a, state_guess_b = _propagate_pair(propagator, index_a, index_b, t_guess)
    miss_guess = _range_km(state_guess_a, state_guess_b)

    t_coarse, _, _, _ = _coarse_minimum(propagator, index_a, index_b, t_guess, step_s)
    t_refined, miss, state_a, state_b = _fine_minimum(
        propagator, index_a, index_b, t_coarse, step_s
    )

    if miss > miss_guess:
        t_refined = t_guess
        state_a, state_b = state_guess_a, state_guess_b

    primary, secondary = assign_primary(object_a, object_b)
    if primary is object_a:
        primary_state, secondary_state = state_a, state_b
    else:
        primary_state, secondary_state = state_b, state_a

    return _build_conjunction(primary, secondary, primary_state, secondary_state, t_refined)
