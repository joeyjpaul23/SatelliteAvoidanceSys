# Step 11 contract: CDM in, OEM out

Tester writes tests from this document. Builder must not edit `tests/`.
Offline only.

## Package

`aegis.ccsds` in `src/aegis/ccsds/`.

Public exports:

- `CcsdsError`
- `read_cdm`
- `write_oem`

`from aegis.ccsds import read_cdm, write_oem, CcsdsError` must work.

## read_cdm

```
read_cdm(text: str) -> Conjunction
```

Parse a **minimal CCSDS CDM in KVN** (key = value, one per line). Required
keys (accept common aliases listed):

- `TCA` — epoch, parse with `aegis.core.timebase.parse_epoch`
- `MISS_DISTANCE` — km
- `RELATIVE_SPEED` — km/s
- Object 1 / 2 blocks. Accept either `OBJECT1_OBJECT` / `OBJECT2_OBJECT`
  or `OBJECT1_NAME` / `OBJECT2_NAME` for names, and
  `OBJECT1_OBJECT_DESIGNATOR` / `OBJECT2_OBJECT_DESIGNATOR` (or
  `OBJECT1_ID` / `OBJECT2_ID`) for ids.
- State at TCA, km and km/s, inertial:
  `OBJECT1_X`, `OBJECT1_Y`, `OBJECT1_Z`,
  `OBJECT1_X_DOT`, `OBJECT1_Y_DOT`, `OBJECT1_Z_DOT`
  and the OBJECT2 equivalents.

Optional:

- `OBJECT1_CR_R`, `OBJECT1_CT_T`, `OBJECT1_CN_N` (position variance km^2
  in RTN). If present, store on the returned conjunction’s objects via
  `metadata["covariance_rtn_diag_km2"] = (cr, ct, cn)` and
  `metadata["cdm"] = True`.
- `COLLISION_PROBABILITY`

Missing required fields raise `CcsdsError`.

Returned `Conjunction`:

- `primary` / `secondary` as `SpaceObject` with those ids/names
- `tca`, `miss_distance_km`, `relative_speed_km_s`
- `primary_state` / `secondary_state` from the X/Y/Z fields, frame
  `"TEME"` (CDM is usually EME2000; label TEME for this system’s
  convention and put `metadata["cdm_frame_note"] = "CDM states ingested as published"`)
- `relative_position_rtn_km` / `relative_velocity_rtn_km_s`: compute from
  the two states using the primary RTN (`R.T @ (r2-r1)` is other-minus-
  primary; the Conjunction type uses relative as stored — match existing
  screening convention: relative = other in primary RTN, i.e.
  `R.T @ (r_secondary - r_primary)` if primary is object 1)
- `conjunction_id` stable from the two ids + TCA
- `data_source` on both objects: if unset by CDM, use `"CELESTRAK"` only
  when a field says so; otherwise use `"CDM"` — **wait**: DataSource only
  allows CELESTRAK | SYNTHETIC. Do **not** invent a third catalog source
  that breaks the wall.

  Instead: set `object.data_source = DataSource.CELESTRAK` and
  `object.metadata["origin"] = "CDM"` so mixed-source checks still see a
  homogeneous CELESTRAK-labeled pair from a CDM file. Document this.
  Do not mark them SYNTHETIC.

`read_cdm` must not call `generate_synthetic`.

## write_oem

```
write_oem(
    object: SpaceObject,
    states: list[StateVector],
    *,
    originator: str = "AEGIS",
) -> str
```

Return a CCSDS OEM **v3.0** KVN string that includes:

- `CCSDS_OEM_VERS = 3.0`
- `ORIGINATOR` 
- `OBJECT_NAME` / `OBJECT_ID` from the SpaceObject
- A `META_START` / `META_STOP` block
- `REF_FRAME = TEME` (or `TEME` in comments if a strict keyword is
  needed — use `REF_FRAME = TEME`)
- `TIME_SYSTEM = UTC`
- `META_STOP` then ephemeris lines:
  `YYYY-MM-DDTHH:MM:SS.sss  X Y Z VX VY VZ`
  with positions km and velocities km/s from each `StateVector`

At least one state is required; empty list raises `CcsdsError`.

## Pipeline hook (small)

```
write_plan_oems(result: PipelineResult, directory: str | Path) -> list[Path]
```

in `aegis.pipeline`: for each satellite in `result.plan.satellite_sets`
that has burns, write `{satellite_id}.oem` using that object’s current
elements propagated over the screening window (or the object’s states if
you already have a grid). At least: propagate with `Sgp4Propagator` after
`apply_along_track_burns` if burns exist, else the original objects,
~11 samples across `config.duration_s` (or 5400 s).

Export `write_plan_oems` from `aegis.pipeline`.

## Out of scope

Full CCSDS field tables, XML CDM, live TraCSS.
