# Step 5 contract: end-to-end pipeline

Tester writes tests from this document only and must not read
`src/aegis/pipeline/` while writing tests. Builder must not edit `tests/`.

## Package

`aegis.pipeline` in `src/aegis/pipeline/`.

Public exports (`from aegis.pipeline import ...` and `__all__`):

- `PipelineConfig`
- `PipelineResult`
- `PipelineError`
- `run_pipeline`

`python -m aegis.pipeline` must also work (package `__main__` or
`pipeline/__main__.py` calling the same `main`).

## Source wall (non-negotiable)

`run_pipeline` takes an explicit `source` argument. Allowed values are
exactly `DataSource.CELESTRAK` (`"CELESTRAK"`) and `DataSource.SYNTHETIC`
(`"SYNTHETIC"`). Any other string raises `PipelineError`.

Rules:

- Default `source` is `DataSource.CELESTRAK`. Callers who want synthetic
  must pass `source=DataSource.SYNTHETIC` (or `"SYNTHETIC"`).
- If `source` is CelesTrak, the pipeline must not import or call
  `generate_synthetic`, must not construct `SyntheticAuthorization` /
  `SyntheticSpec`, and must not consult `AEGIS_ALLOW_SYNTHETIC`.
- If CelesTrak ingest fails, raise `PipelineError` (chaining the
  underlying error). **Never** fall back to a synthetic catalog.
- If `source` is SYNTHETIC, require a `SyntheticAuthorization` instance
  **and** `AEGIS_ALLOW_SYNTHETIC=1` (the existing dual gate). Missing
  either raises `SyntheticNotAuthorizedError` (or `PipelineError` that
  wraps it). No other bypass.
- Passing `source=CELESTRAK` together with a `synthetic_spec` or
  `authorization` raises `PipelineError` (those arguments are illegal on
  the real path).
- Mixed catalogs still raise `MixedDataSourceError`.

## PipelineConfig

Dataclass (all fields have defaults):

- `duration_s: float` — screening window length. Default `1.5` orbital
  periods is **not** assumed globally; default `duration_s = 5400.0`
  (90 minutes) is acceptable, or `None` meaning "1.5 periods of the first
  object's mean motion when elements exist, else 5400". Document which.
- `step_s: float` default `SCREENING_STEP_S`
- `box_km` default `SCREENING_BOX_STARLINK_KM`
- `target_pc: float` default `PC_TARGET_POST_MANEUVER`
- `dv_budget_km_s: float` default `DEFAULT_DV_BUDGET_KM_S`
- `max_iterations: int` default `5`
- `max_objects: int | None` default `None` (no cap). If set, keep the
  first N objects after ingest (stable order). Used later for Starlink
  slices; must not change data_source.

Starlink-specific numbers (inclination, sat count) must not be hardcoded
inside the optimizer call — only these config fields.

## run_pipeline

```
run_pipeline(
    source: str = DataSource.CELESTRAK,
    *,
    group: str = "starlink",
    catalog: Catalog | None = None,
    authorization: SyntheticAuthorization | None = None,
    synthetic_spec: SyntheticSpec | None = None,
    session=None,
    cache_dir=None,
    config: PipelineConfig | None = None,
) -> PipelineResult
```

Ingest:

- If `catalog` is provided, use it. Its `source` must equal the `source`
  argument or raise `MixedDataSourceError` / `PipelineError`. Do not
  re-fetch. This is the offline path (tests inject a catalog).
- Else if `source` is CELESTRAK: `fetch_celestrak(group, session=session,
  cache_dir=cache_dir)`.
- Else if `source` is SYNTHETIC: `generate_synthetic(authorization,
  synthetic_spec or SyntheticSpec())`.

Then, in order: `screen` → `assess_catalog` → `rescreen_until_stable`
(preferred) or `plan_maneuvers` if the catalog has zero objects.

- Empty catalog: return a result with empty assessed entries and an empty
  `ManeuverPlan` (0 burns, 0 resolved). Do not raise.
- `now` for planning is the screening start (catalog epoch / first object
  elements.epoch / `catalog.fetched_at`), **not** wall-clock, when that
  epoch is available.
- Screening start: if objects have `elements.epoch`, use the earliest;
  else `catalog.fetched_at`.

## PipelineResult

- `source: str` — `CELESTRAK` or `SYNTHETIC`
- `catalog: Catalog`
- `assessed: AssessedCatalog`
- `plan: ManeuverPlan`
- `covariance_source: str` — `CovarianceSource.SYNTHETIC_TLE` when the
  default TLE model was used
- `config: PipelineConfig`

`result.plan.summary()` must remain valid. Result must expose enough to
print: source, object count, conjunction count, plan summary,
covariance_source.

## CLI (`python -m aegis.pipeline`)

Arguments (argparse is fine):

- `--source` default `CELESTRAK` (accept `CELESTRAK` / `SYNTHETIC`,
  case-insensitive)
- `--group` default `starlink` (CelesTrak only)
- `--acknowledge-synthetic` flag, required for SYNTHETIC together with
  the env var
- `--n-planes`, `--sats-per-plane` (synthetic spec only; ignored on
  CelesTrak)
- `--duration-s`, `--max-objects`
- `--output` optional path; if omitted, print `plan.summary()` as JSON
  to stdout

If `--source SYNTHETIC` without `--acknowledge-synthetic` or without
env `AEGIS_ALLOW_SYNTHETIC=1`, exit code != 0 and stderr mentions
synthetic was refused. Must not print a fake constellation.

CLI that cannot reach the network and is not given a catalog may fail
on CelesTrak; tests will invoke `run_pipeline` with an injected
`catalog=` rather than live CLI network.

## What this step does not do

No JSON plan artifact schema (Step 8). No TLE-file loader (Step 7).
No LIMITATIONS.md. No api/ui.
