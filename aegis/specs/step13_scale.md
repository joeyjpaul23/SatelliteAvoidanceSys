# Step 13 contract: hundreds-object screening

Tester writes tests from this document. Builder must not edit `tests/`.
No live CelesTrak.

## Spatial partition

`aegis.screening.broadphase` (or a new `aegis.screening.partition`)
must not require an all-pairs Python loop over objects × times as the
only path when N is large.

Add:

```
broadphase(
    grid,
    *,
    box_km=...,
    max_relative_speed_km_s=...,
    cell_km: float | None = None,
) -> list[tuple[int, int, int]]
```

or a separate `broadphase_partitioned(...)` that `screen` uses when
`n_objects >= 50` (threshold may be a constant
`SCREENING_PARTITION_MIN_OBJECTS = 50` in `constants.py`).

Partition idea (builder may vary): grid cells of size ~ max(box) +
reach, hash objects per epoch, only test pairs that share a cell or
neighbor cells. Must be **conservative**: any pair the naive box/no-miss
gate would keep must still be a candidate (no false negatives).

## Correctness

On a synthetic catalog of **12** objects (gated generate_synthetic,
n_planes=2, sats_per_plane=6), `screen` with and without partition
(expose a `screen(..., partitioned: bool | None = None)` or compare
`broadphase` vs `broadphase_partitioned` on the same grid) must produce
the **same set of (i, j, time_index)** after sorting, or the same set of
`conjunction_id`s from `screen`.

If you only add an optional flag:

```
screen(..., partitioned: bool | None = None)
```

`None` means auto (on when N >= 50). Tests will force both True and
False on N=12.

## Blocked propagation

`Sgp4Propagator.propagate_grid` (or a new `propagate_grid_blocked`)
must support splitting the time axis into blocks so a long window does
not allocate one giant `(N, T, 3)` array.

```
propagate_grid(start, duration_s, step_s, *, block_duration_s: float | None = None)
```

When `block_duration_s` is set (e.g. 1800), propagate in chunks and
**concatenate** (or yield) a `PropagationGrid` equivalent to the
unblocked call (same times, positions agree where valid).

`screen` on N>=50 should pass a block duration (e.g. 1800 s) so memory
stays bounded. Positions at overlapping block edges must match.

## Scale smoke

`generate_synthetic` with `n_planes=8`, `sats_per_plane=25` → **200**
objects, `AEGIS_ALLOW_SYNTHETIC=1`.

`screen(objects, epoch, duration_s=1800, step_s=60)` must:

- return (no exception)
- finish in **under 90 seconds**
- return a list (possibly empty) of `Conjunction`

Do not require the full maneuver LP on 200 objects in this step.

## Out of scope

Catalog-scale (10k+), GNN, api/ui.
