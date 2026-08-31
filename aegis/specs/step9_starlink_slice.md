# Step 9 contract: cached Starlink slice through the pipeline

Tester writes tests from this document. Builder must not edit `tests/`.
Tester must not hit celestrak.org.

## Deliverable

A **committed** offline TLE extract of real Starlink (or Starlink-named)
objects, plus a thin helper so the existing pipeline can run on it.

### Fixture

Path (exact):

`/Users/joeypaul/PIGNN-SAT/aegis/tests/fixtures/starlink_slice.tle`

- 3-line TLE groups (name, line1, line2)
- At least **20** objects, at most **80**
- Names contain `STARLINK` (case-insensitive) on the name line
- Valid enough that `catalog_from_tle_file` returns
  `source == CELESTRAK` and `Sgp4Propagator` can initialise every object
- No HTTP at test time

Builder creates this file from a one-shot CelesTrak download (or an
already-local extract). It is test data, not live ingest.

### Helper

Add to `aegis.pipeline` (or `aegis.ingest`):

```
load_starlink_slice(
    path: str | Path | None = None,
    *,
    max_objects: int | None = None,
) -> Catalog
```

- Default path is the fixture above (relative to the package or the
  exact tests/fixtures path resolved from a well-defined location —
  must work when cwd is `aegis/` with `PYTHONPATH=src`).
- Prefer: resolve relative to this file or accept the absolute fixture
  path. Document the default in the docstring.
- `data_source` / catalog `source` are `CELESTRAK`.
- `max_objects` keeps the first N (same as pipeline cap).
- No HTTP. No `generate_synthetic`.

`run_pipeline(source=CELESTRAK, tle_path=<fixture>, config=PipelineConfig(duration_s=2700, max_objects=20, step_s=60))`
must return a `PipelineResult` with:

- `source == CELESTRAK`
- `covariance_source == SYNTHETIC_TLE`
- `len(catalog) <= 20` when max_objects=20
- `plan` is a `ManeuverPlan` (0 burns allowed)
- Completes in under 60 seconds

## CLI

`--tle-path tests/fixtures/starlink_slice.tle --max-objects 20 --duration-s 2700 --output …`
already exists; no new flag required unless the helper needs one.
Do not add `--source SYNTHETIC` fallback.

## Honesty

Do not tag these objects `SYNTHETIC`. Do not invent Starlink TLEs by
copying the synthetic generator and relabeling.

## Out of scope

Spatial partition (Step 13), CDM/OEM (Step 11), token-burn removal
(Step 10).
