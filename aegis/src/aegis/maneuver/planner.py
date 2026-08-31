"""Fleet-wide along-track maneuver planning.

Decision variables are signed along-track delta-v at a handful of candidate
burn epochs per maneuverable satellite. Absolute value is represented with
a plus/minus split so the problem stays a linear program. Slack on each
conjunction is mandatory: infeasible geometries return a plan with
unresolved entries rather than raising.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
from scipy.optimize import linprog

from ..constants import (
    DEFAULT_BURN_SLOTS,
    DEFAULT_DV_BUDGET_KM_S,
    LP_SLACK_PENALTY,
    MIN_LEAD_TIME_ORBITS,
    MU_EARTH_KM3_S2,
    PC_TARGET_POST_MANEUVER,
    STATION_KEEPING_BOX_KM,
)
from ..core.frames import rtn_basis
from ..core.maneuver import (
    Maneuver,
    ManeuverPlan,
    ResolvedConjunction,
    SatelliteManeuverSet,
)
from ..core.objects import SpaceObject
from ..core.state import StateVector
from ..core.timebase import ensure_utc, seconds_between, shift, utc_now
from ..ingest import MixedDataSourceError
from ..risk.alfano import collision_probability
from ..risk.batch import AssessedCatalog
from .cw import along_track_response_km
from .errors import ManeuverSolverError
from .miss import required_miss_distance_km

__all__ = ["plan_maneuvers"]

_DV_EMIT_FLOOR_KM_S = 1e-12
_ZERO_MISS_KM = 1e-12
_SLACK_RESOLVED_KM = 1e-9


def _reject_mixed_sources(objects: list[SpaceObject]) -> None:
    if not objects:
        return
    sources = {obj.data_source for obj in objects}
    if len(sources) > 1:
        raise MixedDataSourceError(
            "cannot plan maneuvers for objects from mixed data sources: "
            + ", ".join(sorted(repr(s) for s in sources))
        )


def _mean_motion_rad_s(
    obj: SpaceObject, state: StateVector | None = None
) -> float | None:
    if obj.elements is not None and obj.elements.mean_motion_rev_per_day > 0.0:
        return obj.elements.mean_motion_rad_s
    if state is not None and state.radius_km > 0.0:
        return float(np.sqrt(MU_EARTH_KM3_S2 / state.radius_km**3))
    return None


def _along_track_alignment(primary_state: StateVector, secondary_state: StateVector) -> float:
    _, t_primary, _ = rtn_basis(primary_state.position_km, primary_state.velocity_km_s)
    _, t_secondary, _ = rtn_basis(secondary_state.position_km, secondary_state.velocity_km_s)
    return float(np.dot(t_primary, t_secondary))


def _new_plan_id() -> str:
    return f"plan-{uuid.uuid4().hex}"


def _default_now(assessed: AssessedCatalog) -> datetime:
    starts = [
        entry.conjunction.screening_window_start
        for entry in assessed.entries
        if entry.conjunction.screening_window_start is not None
    ]
    if starts:
        return min(starts)
    return utc_now()


def _catalog_by_id(
    objects: list[SpaceObject], assessed: AssessedCatalog
) -> dict[str, SpaceObject]:
    by_id: dict[str, SpaceObject] = {}
    for entry in assessed.entries:
        by_id[entry.conjunction.primary.object_id] = entry.conjunction.primary
        by_id[entry.conjunction.secondary.object_id] = entry.conjunction.secondary
    for obj in objects:
        by_id[obj.object_id] = obj
    return by_id


def _state_for(object_id: str, entry) -> StateVector | None:
    conjunction = entry.conjunction
    if conjunction.primary.object_id == object_id:
        return conjunction.primary_state
    if conjunction.secondary.object_id == object_id:
        return conjunction.secondary_state
    return None


def _empty_plan(generated_at: datetime, notes: list[str] | None = None) -> ManeuverPlan:
    return ManeuverPlan(
        plan_id=_new_plan_id(),
        generated_at=generated_at,
        notes=list(notes or []),
    )


def plan_maneuvers(
    assessed: AssessedCatalog,
    objects: list[SpaceObject],
    *,
    now: datetime | None = None,
    target_pc: float = PC_TARGET_POST_MANEUVER,
    dv_budget_km_s: float = DEFAULT_DV_BUDGET_KM_S,
    slack_penalty: float = LP_SLACK_PENALTY,
    burn_slots: int = DEFAULT_BURN_SLOTS,
    min_lead_orbits: float = MIN_LEAD_TIME_ORBITS,
) -> ManeuverPlan:
    """Solve a fleet LP for along-track avoidance burns.

    Only ``is_maneuverable`` objects receive burns. Slack is always present,
    so a purely radial or cross-track geometry returns unresolved entries
    instead of raising. Conjunctions that already meet the required miss
    stay listed as ``resolved=True`` with zero burns -- the LP is not
    forced to emit a token ``dv``.
    """
    _reject_mixed_sources(objects)
    _reject_mixed_sources(
        [
            obj
            for entry in assessed.entries
            for obj in (entry.conjunction.primary, entry.conjunction.secondary)
        ]
    )
    if burn_slots < 0:
        raise ManeuverSolverError("burn_slots must be non-negative")
    if dv_budget_km_s < 0.0:
        raise ManeuverSolverError("dv_budget_km_s must be non-negative")
    if min_lead_orbits < 0.0:
        raise ManeuverSolverError("min_lead_orbits must be non-negative")

    generated_at = utc_now()
    now = ensure_utc(now) if now is not None else _default_now(assessed)

    if not assessed.entries:
        return _empty_plan(generated_at)

    by_id = _catalog_by_id(objects, assessed)

    earliest_tca: dict[str, datetime] = {}
    mean_motion: dict[str, float] = {}
    for entry in assessed.entries:
        conjunction = entry.conjunction
        for object_id, state in (
            (conjunction.primary.object_id, conjunction.primary_state),
            (conjunction.secondary.object_id, conjunction.secondary_state),
        ):
            obj = by_id.get(object_id)
            if obj is None or not obj.is_maneuverable:
                continue
            tca = conjunction.tca
            previous = earliest_tca.get(object_id)
            if previous is None or tca < previous:
                earliest_tca[object_id] = tca
            if object_id not in mean_motion:
                n_rad_s = _mean_motion_rad_s(obj, state)
                if n_rad_s is not None and n_rad_s > 0.0:
                    mean_motion[object_id] = n_rad_s

    sat_ids = [sid for sid in earliest_tca if sid in mean_motion and burn_slots > 0]
    slots: dict[str, list[datetime]] = {}
    for sat_id in sat_ids:
        n_rad_s = mean_motion[sat_id]
        period_s = 2.0 * np.pi / n_rad_s
        half_orbit_s = 0.5 * period_s
        lead_s = min_lead_orbits * period_s
        tca = earliest_tca[sat_id]
        slots[sat_id] = [shift(tca, -(lead_s + j * half_orbit_s)) for j in range(burn_slots)]

    notes: list[str] = []
    if sat_ids and all(slots[sid][0] < now for sid in sat_ids):
        notes.append("all candidate burn slots precede the planning epoch")

    sat_ids = [sid for sid in sat_ids if slots.get(sid)]
    n_slot_pairs = len(sat_ids) * burn_slots
    n_conj = len(assessed.entries)
    n_vars = 2 * n_slot_pairs + n_conj

    def plus_index(sat_index: int, slot_index: int) -> int:
        return (sat_index * burn_slots + slot_index) * 2

    def minus_index(sat_index: int, slot_index: int) -> int:
        return plus_index(sat_index, slot_index) + 1

    def slack_index(conj_index: int) -> int:
        return 2 * n_slot_pairs + conj_index

    sat_index = {sat_id: i for i, sat_id in enumerate(sat_ids)}

    required: list[float] = []
    miss_before: list[float] = []
    sensitivity: list[float] = []
    hbr_km: list[float] = []
    along_coeffs: list[np.ndarray] = []

    for entry in assessed.entries:
        conjunction = entry.conjunction
        assessment = entry.assessment
        hbr = assessment.hard_body_radius_m / 1000.0
        need = required_miss_distance_km(
            hbr,
            assessment.sigma_major_km,
            assessment.sigma_minor_km,
            target_pc,
        )
        y_rtn = float(conjunction.relative_position_rtn_km[1])
        miss = float(conjunction.miss_distance_km)
        sens = y_rtn / miss if miss > _ZERO_MISS_KM else 0.0
        align = _along_track_alignment(conjunction.primary_state, conjunction.secondary_state)

        coeff = np.zeros(n_slot_pairs, dtype=float)
        for object_id, sign in (
            (conjunction.primary.object_id, -1.0),
            (conjunction.secondary.object_id, align),
        ):
            if object_id not in sat_index:
                continue
            n_rad_s = mean_motion[object_id]
            si = sat_index[object_id]
            for j, epoch in enumerate(slots[object_id]):
                dt_s = seconds_between(epoch, conjunction.tca)
                response = along_track_response_km(n_rad_s, dt_s, 1.0)
                coeff[si * burn_slots + j] = sign * response

        required.append(need)
        miss_before.append(miss)
        sensitivity.append(sens)
        hbr_km.append(hbr)
        along_coeffs.append(coeff)

    dv: np.ndarray
    slacks: np.ndarray

    if n_vars == 0:
        dv = np.zeros(0)
        slacks = np.zeros(0)
    else:
        c = np.zeros(n_vars)
        for k in range(n_slot_pairs):
            c[2 * k] = 1.0
            c[2 * k + 1] = 1.0
        for k in range(n_conj):
            c[slack_index(k)] = slack_penalty

        a_rows: list[list[float]] = []
        b_rows: list[float] = []

        for k in range(n_conj):
            row = [0.0] * n_vars
            lever = sensitivity[k] * along_coeffs[k]
            for slot_flat, value in enumerate(lever):
                row[2 * slot_flat] = -value
                row[2 * slot_flat + 1] = value
            row[slack_index(k)] = -1.0
            a_rows.append(row)
            b_rows.append(miss_before[k] - required[k])

        for si, sat_id in enumerate(sat_ids):
            row = [0.0] * n_vars
            for j in range(burn_slots):
                row[plus_index(si, j)] = 1.0
                row[minus_index(si, j)] = 1.0
            a_rows.append(row)
            b_rows.append(dv_budget_km_s)

            n_rad_s = mean_motion[sat_id]
            period_s = 2.0 * np.pi / n_rad_s
            last_epoch = slots[sat_id][0]
            eval_epoch = shift(last_epoch, period_s)
            sk_row = [0.0] * n_vars
            for j, epoch in enumerate(slots[sat_id]):
                dt_s = seconds_between(epoch, eval_epoch)
                response = along_track_response_km(n_rad_s, dt_s, 1.0)
                sk_row[plus_index(si, j)] = response
                sk_row[minus_index(si, j)] = -response
            box = STATION_KEEPING_BOX_KM[0]
            a_rows.append(sk_row)
            b_rows.append(box)
            a_rows.append([-v for v in sk_row])
            b_rows.append(box)

        bounds = [(0.0, dv_budget_km_s) for _ in range(2 * n_slot_pairs)] + [
            (0.0, None) for _ in range(n_conj)
        ]
        a_ub = np.asarray(a_rows, dtype=float) if a_rows else None
        b_ub = np.asarray(b_rows, dtype=float) if b_rows else None

        result = None
        for method in ("highs", "highs-ds", "interior-point"):
            try:
                result = linprog(
                    c,
                    A_ub=a_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method=method,
                )
            except Exception:  # noqa: BLE001 - try the next backend
                result = None
                continue
            if result.success and result.x is not None:
                break

        if result is None or not result.success or result.x is None:
            notes.append(
                "linear program did not return a success status; "
                "emitting a slack-only plan"
            )
            dv = np.zeros(n_slot_pairs)
            slacks = np.array(
                [max(0.0, required[k] - miss_before[k]) for k in range(n_conj)],
                dtype=float,
            )
        else:
            x = np.asarray(result.x, dtype=float)
            dv = np.zeros(n_slot_pairs)
            for k in range(n_slot_pairs):
                dv[k] = x[2 * k] - x[2 * k + 1]
            slacks = x[2 * n_slot_pairs : 2 * n_slot_pairs + n_conj]

    satellite_sets: dict[str, SatelliteManeuverSet] = {}
    for si, sat_id in enumerate(sat_ids):
        maneuvers: list[Maneuver] = []
        involved = [
            entry.conjunction.conjunction_id
            for entry in assessed.entries
            if sat_id in (
                entry.conjunction.primary.object_id,
                entry.conjunction.secondary.object_id,
            )
        ]
        for j, epoch in enumerate(slots[sat_id]):
            value = float(dv[si * burn_slots + j])
            if abs(value) < _DV_EMIT_FLOOR_KM_S:
                continue
            maneuvers.append(
                Maneuver(
                    satellite_id=sat_id,
                    epoch=epoch,
                    delta_v_rtn_km_s=np.array([0.0, value, 0.0]),
                    rationale=list(involved),
                )
            )
        if maneuvers:
            maneuvers.sort(key=lambda item: item.epoch)
            satellite_sets[sat_id] = SatelliteManeuverSet(
                satellite_id=sat_id, maneuvers=maneuvers
            )

    resolved: list[ResolvedConjunction] = []
    for k, entry in enumerate(assessed.entries):
        assessment = entry.assessment
        delta_y = float(along_coeffs[k] @ dv) if n_slot_pairs else 0.0
        miss_after = miss_before[k] + sensitivity[k] * delta_y
        if miss_after < 0.0:
            miss_after = 0.0
        probability_after = collision_probability(
            assessment.sigma_major_km,
            assessment.sigma_minor_km,
            miss_after,
            0.0,
            hbr_km[k],
        )
        slack = float(slacks[k]) if k < len(slacks) else max(
            0.0, required[k] - miss_after
        )
        shortfall = max(0.0, required[k] - miss_after)
        is_resolved = shortfall <= _SLACK_RESOLVED_KM and slack <= 1e-6
        resolved.append(
            ResolvedConjunction(
                conjunction_id=entry.conjunction.conjunction_id,
                probability_before=assessment.probability,
                probability_after=probability_after,
                miss_distance_before_km=miss_before[k],
                miss_distance_after_km=miss_after,
                required_miss_distance_km=required[k],
                resolved=is_resolved,
                shortfall_km=shortfall if not is_resolved else 0.0,
            )
        )

    return ManeuverPlan(
        plan_id=_new_plan_id(),
        generated_at=generated_at,
        satellite_sets=satellite_sets,
        resolved=resolved,
        notes=notes,
    )
