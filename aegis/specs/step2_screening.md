# Step 2 contract: screening

Tester writes tests from this document only and must not read
`src/aegis/screening/` while writing tests. Builder must not edit `tests/`.

## Package

`aegis.screening` in `src/aegis/screening/`.

Public exports:

- `screen`
- `prefilter_pairs`
- `broadphase`
- `refine_tca`
- `ScreeningError`

## screen

```
screen(
    objects: list[SpaceObject],
    start: datetime,
    duration_s: float,
    *,
    step_s: float = SCREENING_STEP_S,
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM,
    propagator: Sgp4Propagator | None = None,
) -> list[Conjunction]
```

- Rejects mixed `data_source` values among `objects` with
  `aegis.ingest.MixedDataSourceError` (import the existing exception).
- Propagates with `Sgp4Propagator` (existing) unless a propagator is passed.
- Returns zero or more `Conjunction` objects for pairs that enter the RTN
  screening box during the window.
- Each conjunction has: both objects, TCA, miss distance, relative speed,
  relative RTN state on the primary, both inertial states at TCA,
  `screening_window_start` / `end` set from the requested window.
- Primary is the maneuverable object if exactly one of the pair is
  maneuverable; otherwise the lower `object_id` is primary.
- Intra-fleet pairs are not excluded (this is the point of the module).
- Pairs whose apogee/perigee bands cannot meet, after
  `PERIGEE_APOGEE_PAD_KM`, never appear.
- A pair that never enters the box does not appear.
- If relative speed at TCA is below `LOW_RELATIVE_VELOCITY_KM_S`, set
  `conjunction.metadata["low_relative_velocity"] = True`.
- `conjunction_id` is stable for the same pair + TCA (same inputs → same id).

## prefilter_pairs

```
prefilter_pairs(objects: list[SpaceObject], *, pad_km: float = PERIGEE_APOGEE_PAD_KM) -> list[tuple[int, int]]
```

Returns index pairs `(i, j)` with `i < j` that survive the
apogee/perigee filter. Two circular LEO satellites at 550 km survive. A
550 km object and a 20000 km object do not.

## broadphase

```
broadphase(
    grid: PropagationGrid,
    *,
    box_km: tuple[float, float, float] = SCREENING_BOX_STARLINK_KM,
    max_relative_speed_km_s: float = MAX_RELATIVE_SPEED_KM_S,
) -> list[tuple[int, int, int]]
```

Returns `(i, j, time_index)` candidates. A pair is a candidate at an epoch
when the relative position expressed in the primary's RTN frame falls inside
the box half-widths `(radial, transverse, normal)`, **or** the no-miss gate
cannot rule the step out: if relative speed could close the remaining
distance within `step` (use `grid.times_s` spacing; if a single step, use
`SCREENING_STEP_S`) at `max_relative_speed_km_s`, keep the candidate.
Invalid SGP4 samples (`grid.valid == False`) are skipped.

## refine_tca

```
refine_tca(
    propagator: Sgp4Propagator,
    index_a: int,
    index_b: int,
    t_guess: datetime,
) -> Conjunction
```

Refines closest approach near `t_guess` (bracket a few screening steps,
then linear/Hermite refine). Miss distance at the returned TCA must be
strictly smaller than, or equal to, the miss at `t_guess` for a pair that
is actually approaching. Relative position · relative velocity at TCA is
near zero compared to the product of the magnitudes (dot product / (|r||v|)
absolute value < 1e-3) when relative speed is above
`LOW_RELATIVE_VELOCITY_KM_S`.

## Known-conjunction synthetic fixture

When Step 1's `generate_synthetic` is used with a valid gate and
`include_known_conjunction_triple=True`, `sats_per_plane>=3`, `n_planes>=1`,
`screen` over `1.5` orbital periods from the spec epoch must return at least
one conjunction whose two object ids are the first two objects of the
catalog (plane 0, first two), with miss distance < 44 km (inside the
along-track half-width of the Starlink box). The third object of that
triple must not form a conjunction with the first two inside that box in
the same window if the generator kept it well clear (tester asserts the
close pair is present; may allow extra pairs).
