# Step 10 contract: physically honest maneuvers

Tester writes tests from this document. Builder must not edit `tests/`.

## 1. No token burns

The planner never emits a token burn (e.g. 1e-6 km/s) when the LP chose
dv=0.

Rules:

- If every conjunction already has miss >= required miss, `plan_maneuvers`
  must emit **zero** burns (`total_burns == 0`). The conjunctions are
  still listed in `plan.resolved` with `resolved=True` and
  `shortfall_km == 0`.
- Burns appear only when the LP assigns |dv| above the emit floor
  because a constraint is not met at dv=0.
- Radial / cross-track infeasible cases still return a `ManeuverPlan`
  with `resolved=False` and `shortfall_km > 0`, no raise.

The known-conjunction synthetic triple (first two objects, ~10 km miss)
already clears a 1e-6 Pc target. After this step it must get **0 burns**.

## 2. Unsafe along-track pair must get a real burn

When a constructed along-track conjunction has miss **well below**
required miss (e.g. miss 0.05 km, HBR 10 m, typical TLE sigmas, target
1e-6), `plan_maneuvers` must:

- emit at least one along-track burn `[0, dv, 0]` with `|dv|` above 1e-12
- mark that conjunction `resolved=True` if the linearized miss can reach
  required within `dv_budget_km_s`
- not use a hardcoded 1e-6 token; dv must come from the LP

## 3. Re-screen miss must move with CW

`apply_along_track_burns` must apply a change that SGP4 can see and that
increases along-track separation consistently with CW.

Required behavior:

```
apply_along_track_burns(objects, plan, assessed) -> list[SpaceObject]
```

After applying a **non-zero** along-track plan to an along-track-separable
pair, a new `screen` (same window) of the copies must show that pair’s
miss distance **greater than** the pre-burn miss (strict), or the pair
must drop out of the screening box (also acceptable: no longer a
conjunction).

Additionally, for a two-object case where you can evaluate both states at
the original TCA after the burn:

- Let `Δy_cw` be `along_track_response_km(n, dt, dv)` summed over burns
  of the maneuvering sat.
- The change in along-track RTN miss (primary RTN y) must have the same
  sign as the intended separation and magnitude within **50%** of
  `|Δy_cw|` or within **2 km**, whichever is looser.

Implementation note (builder): a pure mean-anomaly shift `ΔM = Δy/a` is
acceptable **if** it satisfies the tests above. Changing mean motion /
semi-major axis to capture the secular CW term is also acceptable.
Document the chosen mapping in the `rescreen` module docstring.

## 4. Slack path unchanged

`test_plan_maneuvers_infeasible_returns_slack` must still pass.

## Out of scope

CDM/OEM, Starlink fixture, spatial partition.
