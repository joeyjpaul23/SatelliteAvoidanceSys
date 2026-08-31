# Step 7 contract: real Starlink path (offline-capable)

Tester writes tests from this document only. Builder must not edit `tests/`.
Tester must not hit celestrak.org. Builder must not require live network
for unit tests.

## Offline CelesTrak catalog from a TLE file

Add to `aegis.ingest` (implement in `celestrak.py`, export from
`aegis.ingest` via the existing lazy `__getattr__` so that
`import aegis.ingest.synthetic` still does not load this if possible;
loading `celestrak` for this export is correct):

```
catalog_from_tle_file(path: str | Path) -> Catalog
```

- Reads a local TLE file (2-line or 3-line groups).
- Every object `data_source == DataSource.CELESTRAK`.
- Catalog `source == CELESTRAK`.
- `query` includes the file path name.
- No HTTP. Must not call `generate_synthetic`.
- Invalid / empty parse raises `CelesTrakError`.
- Objects with TLE lines must be usable by `Sgp4Propagator`.

Also acceptable equivalent name if documented in `__all__`: the name
**must** be `catalog_from_tle_file`.

## Pipeline accepts a local TLE path

Extend `run_pipeline`:

```
tle_path: str | Path | None = None
```

- If `tle_path` is set, `source` must be CELESTRAK (default). Using
  `tle_path` with `source=SYNTHETIC` raises `PipelineError`.
- Loads via `catalog_from_tle_file`; does not open HTTP.
- `group` is ignored when `tle_path` is set.
- `config.max_objects` still applies after load.

CLI: `--tle-path PATH` on `python -m aegis.pipeline`.

## Fleet config, not hardcoded Starlink physics

`PipelineConfig` may gain:

- `group: str = "starlink"` (ingest hint only)
- `station_keeping_box_km` already lives in constants; do not bake
  Starlink sat counts or inclinations into `plan_maneuvers`

A 2-object TLE fixture run must produce `source == CELESTRAK` and
`covariance_source == SYNTHETIC_TLE`.

## Test fixture (tester-owned)

Tester will write a tiny TLE fixture (e.g. two valid LEO TLEs) under
`tests/fixtures/`. Builder must not depend on a particular filename
beyond accepting any TLE path. Do not check in a full Starlink dump.

## Honesty

Results from this path are still TLE-grade. Do not set
`CovarianceSource.CALCULATED`. Do not claim operator CDMs.

## What this step does not do

Live download in CI. Full-constellation screen. api/ui. CDM/OEM writers.
