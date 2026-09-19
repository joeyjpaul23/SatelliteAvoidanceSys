# Step 13 contract: hundreds-object screening

Tester writes tests from this document. Builder must not edit `tests/`.
No live CelesTrak.

## Candidate pairs

`screen` must not require an all-pairs Python loop over objects × times
as the only path when N is large.

```
screen(..., partitioned: bool | None = None, workers: int | None = None)
```

- `partitioned=True` generates candidate pairs from a k-d tree;
  `partitioned=False` uses all pairs. `None` means auto: the tree is on
  when N >= `SCREENING_PARTITION_MIN_OBJECTS` (50, in `constants.py`).
- The tree must be **conservative**: any pair the all-pairs path would
  keep must still be a candidate (no false negatives).
- `workers` caps parallel worker processes (default `AEGIS_WORKERS` or
  the CPU count). Parallel and serial runs return the same result.

## Correctness

On a synthetic catalog of **12** objects (gated generate_synthetic,
n_planes=2, sats_per_plane=6), `screen` with `partitioned=True` and
`partitioned=False` must produce the same set of `conjunction_id`s (or,
failing that, the same set of pair ids). Partitioned screening must never
drop a pair that unpartitioned screening keeps.

## Bounded memory

`screen` streams the window in time blocks and never holds the full
`(N, T, 3)` grid. Where the block edges fall must not change the result:
an approach cut by a block edge is stitched back into one conjunction.

## Scale smoke

`generate_synthetic` with `n_planes=8`, `sats_per_plane=25` → **200**
objects, `AEGIS_ALLOW_SYNTHETIC=1`.

`screen(objects, epoch, duration_s=1800, step_s=60)` must:

- return (no exception)
- finish in **under 90 seconds**
- return a list (possibly empty) of `Conjunction`

Do not require the full maneuver LP on 200 objects in this step.

## Out of scope

GNN, api/ui.
