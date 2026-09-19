# Step 4 contract: maneuver optimizer

Tester writes tests from this document only and must not read
`src/aegis/maneuver/` while writing tests. Builder must not edit `tests/`.

## Package

`aegis.maneuver` in `src/aegis/maneuver/`. This is the optimizer, not
`aegis.core.maneuver` (value types). Do not change the public behavior of
the core value types.

Public exports:

- `clohessy_wiltshire_state`
- `along_track_response_km`
- `required_miss_distance_km`
- `plan_maneuvers`
- `apply_along_track_burns` (Step 10)
- `rescreen_until_stable`
- `ManeuverSolverError`

## clohessy_wiltshire_state

```
clohessy_wiltshire_state(
    n_rad_s: float,
    dt_s: float,
    r0_rtn_km: np.ndarray,   # shape (3,) radial, along-track, cross-track
    v0_rtn_km_s: np.ndarray, # shape (3,)
) -> tuple[np.ndarray, np.ndarray]  # (r_rtn, v_rtn) at dt
```

Exact linear CW. Along-track position contribution from an along-track
velocity impulse applied at the origin must use

`(4 * sin(n*dt) - 3*n*dt) / n`

Never the shortcut `3 * dv * dt` (wrong by ~85% at quarter-orbit lead).

Checks a tester will run:

- At `dt = 0`, state is unchanged.
- Pure along-track `dv` from the origin: radial =
  `(2/n) * (1 - cos(n dt)) * dv`, along-track =
  `(4*sin(n dt) - 3*n*dt)/n * dv`, cross-track = 0.
- Pure cross-track `dv` from the origin: only the normal axis moves,
  `z = (sin(n dt)/n) * dv`.
- Quarter-orbit (`dt = 0.25 * 2π/n`) along-track displacement from unit `dv`
  is **not** within 10% of `3 * dv * dt`.

## along_track_response_km

```
along_track_response_km(n_rad_s: float, dt_s: float, dv_along_track_km_s: float) -> float
```

Equals the along-track component of CW from the origin with a pure
tangential impulse. Same formula as above.

## required_miss_distance_km

```
required_miss_distance_km(
    hard_body_radius_km: float,
    sigma_major_km: float,
    sigma_minor_km: float,
    target_pc: float = PC_TARGET_POST_MANEUVER,
) -> float
```

Smallest miss distance (along the major-axis-aligned miss used by Alfano,
i.e. miss on x, z=0) such that `alfano.collision_probability` is `<=
target_pc`. Must be monotonically non-decreasing as `target_pc` decreases.
If even a very large miss (e.g. 100 km) cannot reach the target, return
that large sentinel and do not raise.

## plan_maneuvers

```
plan_maneuvers(
    assessed: AssessedCatalog,
    objects: list[SpaceObject],
    *,
    now: datetime | None = None,
    target_pc: float = PC_TARGET_POST_MANEUVER,
    dv_budget_km_s: float = DEFAULT_DV_BUDGET_KM_S,
    slack_penalty: float = LP_SLACK_PENALTY,
    burn_slots: int = DEFAULT_BURN_SLOTS,
    min_lead_orbits: float = MIN_LEAD_TIME_ORBITS,
) -> ManeuverPlan
```

- Mixed object sources raise `MixedDataSourceError`.
- Only `SpaceObject.is_maneuverable` objects receive burns. Others are
  obstacles.
- Decision variables: along-track delta-v at `burn_slots` candidate epochs
  per maneuverable satellite, spaced half an orbit apart, working backwards
  from the earliest TCA that involves that satellite, each slot at least
  `min_lead_orbits` before that TCA.
- Objective: minimize sum of absolute along-track delta-v plus
  `slack_penalty * sum(slack_k)` for each conjunction slack.
- Constraint (linearized via CW): for each conjunction, the change in
  along-track miss contributed by planned burns, plus slack, must make
  predicted miss `>= required_miss_distance_km(...)` computed from that
  event's assessment sigmas and combined HBR.
- Per-satellite sum of |dv| `<= dv_budget_km_s`.
- Station-keeping: predicted along-track displacement magnitude at one
  orbit after the last burn `<= STATION_KEEPING_BOX_KM[0]`.
- Slack is mandatory. An infeasible geometry (needed displacement is
  purely radial/cross-track with no along-track lever) must still return a
  `ManeuverPlan` with `resolved` entries whose `resolved` flag is False and
  `shortfall_km > 0` rather than raising.
- Returned type is `aegis.core.maneuver.ManeuverPlan`.
- Burns are stored as `Maneuver` with `delta_v_rtn_km_s` along-track only
  (`[0, dv, 0]`).
- `plan.summary()` remains valid.
- A conjunction that already meets the required miss gets no burn and is
  listed `resolved=True` with `shortfall_km == 0`. An along-track-separable
  conjunction below the required miss gets a real LP burn and is marked
  `resolved=True` when the required miss is achievable within budget
  (Step 10).

`now` defaults to the earliest screening window start among conjunctions, or
`utc_now()` if absent.

## rescreen_until_stable

```
rescreen_until_stable(
    objects: list[SpaceObject],
    start: datetime,
    duration_s: float,
    *,
    max_iterations: int = 5,
    **screen_and_plan_kwargs,
) -> ManeuverPlan
```

Loop: screen → assess_catalog → plan_maneuvers → apply impulsive burns to
copies of the objects (adjust along-track mean anomaly / TLE-equivalent by
the CW-implied phase, or rebuild elements so that a subsequent
`Sgp4Propagator` run reflects an along-track timing shift consistent with
the planned dv; a documented approximation is allowed if it is deterministic
and reduces the original pair's miss for the separable case) → screen again.
Stop when no new `ACT`/`WATCH` conjunctions appear that were not in the
previous set of pair-ids, or `max_iterations` is hit.

Set `ManeuverPlan.iterations` and `converged` accordingly. `converged` is
True when the last re-screen introduced no new pair above `WATCH`.

On a 2-sat along-track-separable synthetic pair, after this loop the
re-assessed Pc of that pair is below `target_pc` or the plan lists the
conjunction as unresolved with slack — never a crash.
