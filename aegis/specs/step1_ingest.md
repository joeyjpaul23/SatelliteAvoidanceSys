# Step 1 contract: ingest (CelesTrak vs synthetic)

This is the public contract. Builder implements it. Tester writes tests from
this document only and must not read `src/aegis/ingest/` while writing tests.
Analyzer and builder must not edit tester-owned files under `tests/`.

## Package

`aegis.ingest` lives in `src/aegis/ingest/`.

Public exports (must appear in `aegis.ingest.__all__` and be importable as
`from aegis.ingest import ...`):

- `DataSource`
- `Catalog`
- `CatalogError`
- `MixedDataSourceError`
- `SyntheticNotAuthorizedError`
- `SyntheticAuthorization`
- `SyntheticSpec`
- `CelesTrakClient`
- `CelesTrakError`
- `fetch_celestrak`
- `generate_synthetic`

## Data-source wall (non-negotiable)

Two acquisition paths exist. They must not share objects, caches, parsers, or
network calls.

1. **Real path** — CelesTrak GP / SupGP only.
2. **Synthetic path** — in-memory generated constellation only.

Rules:

- `src/aegis/ingest/celestrak.py` must not import anything from
  `aegis.ingest.synthetic` (or a `synthetic` submodule).
- `src/aegis/ingest/synthetic.py` must not import anything from
  `aegis.ingest.celestrak` (or a `celestrak` submodule).
- Synthetic code must not open HTTP connections and must not read or write
  any CelesTrak cache directory.
- CelesTrak code must not call `generate_synthetic` or construct
  `SyntheticSpec` / `SyntheticAuthorization`.
- Default ingest is CelesTrak. Synthetic is opt-in only.

`DataSource` is a class (or enum-like namespace) with string constants:

- `DataSource.CELESTRAK == "CELESTRAK"`
- `DataSource.SYNTHETIC == "SYNTHETIC"`

Every `SpaceObject` leaving ingest must have `data_source` set to exactly one
of those two strings. Never empty. Never a third value from these two
functions.

## Catalog

`Catalog` is a homogeneous collection:

- `source: str` — `DataSource.CELESTRAK` or `DataSource.SYNTHETIC`
- `objects: list[SpaceObject]`
- `fetched_at` — timezone-aware UTC datetime
- `query` — string describing what was requested (group name, or synthetic
  spec summary)

Construction must reject a mixed list: if any object's `data_source` differs
from `source` or from another object, raise `MixedDataSourceError` (a
`CatalogError`). An empty catalog is allowed and still carries a `source`.

`Catalog` must not provide a silent merge of two catalogs with different
sources. If a `merge` / `+` / `extend` helper exists, mixed sources raise
`MixedDataSourceError`. Same-source merge is allowed.

`len(catalog)` equals the number of objects. Iteration yields the objects.

## Synthetic gate

`generate_synthetic` must not run unless **both** of the following are true:

1. Caller passes a `SyntheticAuthorization` instance.
2. Environment variable `AEGIS_ALLOW_SYNTHETIC` is the exact string `"1"`.

`SyntheticAuthorization` construction:

```
SyntheticAuthorization(acknowledge_synthetic=True)
```

- `acknowledge_synthetic` is a required keyword-only argument.
- Any value other than the boolean `True` raises `SyntheticNotAuthorizedError`.
- Positional construction `SyntheticAuthorization(True)` must raise `TypeError`.
- There is no other public constructor, factory, or "dev bypass" that skips
  the env var.

If the env var is missing, empty, `"true"`, `"yes"`, `"0"`, or anything other
than `"1"`, `generate_synthetic(...)` raises `SyntheticNotAuthorizedError`
even if a valid authorization object is passed.

The error message must mention that synthetic data was refused and that
`AEGIS_ALLOW_SYNTHETIC=1` plus an explicit authorization are required.

`fetch_celestrak` must work without that env var and must never consult it.

## SyntheticSpec and generate_synthetic

`SyntheticSpec` fields (all have defaults so a tester can construct one):

- `n_planes: int` default 1
- `sats_per_plane: int` default 3
- `altitude_km: float` default 550.0
- `inclination_deg: float` default 53.0
- `epoch` — timezone-aware UTC datetime; default a fixed documented epoch
  `2010-01-01T00:00:00+00:00` (far enough in the past that SGP4 accepts it)
- `seed: int` default 0
- `operator_id: str` default `"SYNTHETIC-OP"`
- `operator_name: str` default `"Synthetic Operator"`
- `include_known_conjunction_triple: bool` default True

`generate_synthetic(authorization, spec) -> Catalog`:

- Returns `Catalog` with `source == DataSource.SYNTHETIC`.
- Object count is `n_planes * sats_per_plane`. When
  `include_known_conjunction_triple` is True, the catalog still has exactly
  that many objects; the first three objects of plane 0 are arranged so that
  two of them have a close approach (miss distance well inside the Starlink
  screening box) within 1.5 orbital periods of `spec.epoch`, and the third
  is phased so it does not participate in that close approach.
- Every object has:
  - unique `object_id` (numeric-looking string, unique within the catalog)
  - `data_source == DataSource.SYNTHETIC`
  - `object_type` payload
  - `elements` populated and SGP4-initializable via existing
    `aegis.propagation.Sgp4Propagator` (TLE lines optional if elements work)
  - `operator` set, `operator.maneuverable is True`,
    `operator.identifier == spec.operator_id`
- Same `spec` + same `seed` is deterministic (identical ids and elements).
- Different seeds produce different phasing (not an identical catalog).

## CelesTrak

`CelesTrakClient(cache_dir=None, session=None)`:

- `cache_dir` defaults to a directory named `celestrak` under a cache root
  that is **not** used by synthetic code. Cache files are only written here.
- Uses `CELESTRAK_CACHE_TTL_S`, `CELESTRAK_MAX_RETRIES`,
  `CELESTRAK_USER_AGENT` from `aegis.constants`.
- `session` is an optional `requests`-like object (must provide `.get(url,
  timeout=..., headers=...)`). Tests inject a fake session; the client must
  not require the network when `session` is supplied.
- User-Agent header must equal `CELESTRAK_USER_AGENT`.

`CelesTrakClient.fetch_gp(group: str, *, fmt: str = "tle") -> Catalog`
and module function `fetch_celestrak(group, *, fmt="tle", supplement=False,
session=None, cache_dir=None) -> Catalog`:

- `fmt="tle"`: response body is standard 3-line TLE groups (name, line1,
  line2) or 2-line (line1, line2). Parse into `SpaceObject` with
  `tle_line1`, `tle_line2`, `name`, `object_id` from the NORAD number on
  line 1, `elements` populated from the TLE, `data_source=CELESTRAK`.
- `fmt="json"`: CelesTrak OMM-JSON array (objects with at least
  `OBJECT_NAME`, `OBJECT_ID`, `NORAD_CAT_ID`, `EPOCH`, `MEAN_MOTION`,
  `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`,
  `MEAN_ANOMALY`, `BSTAR`, and optionally `TLE_LINE1` / `TLE_LINE2`).
- `supplement=True` uses the supplemental GP URL
  `https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE={group}&FORMAT={fmt}`
  otherwise
  `https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT={fmt}`.
- Failed HTTP (status >= 400 after retries) raises `CelesTrakError`.
- Cache key includes group, fmt, and whether supplemental. Fresh cache is
  reused without calling `session.get`. Stale cache is refreshed.
- Returned catalog `source` is `CELESTRAK`. `query` includes the group name.
- Operator may be None (CelesTrak is not an operator assignment).
- Objects must be usable by `Sgp4Propagator` when TLE lines are present.

TLE checksums on synthetic objects are not required if elements-only
initialization is used. CelesTrak-parsed objects that include TLE lines
must keep those lines verbatim.

## Errors

- `CatalogError` — base for catalog/ingest structural errors.
- `MixedDataSourceError` — subclass of `CatalogError`.
- `SyntheticNotAuthorizedError` — subclass of `CatalogError`.
- `CelesTrakError` — network/parse failures on the real path.

## What this step does not do

No screening, no risk batch API, no maneuver solver, no live network in
unit tests (tester will mock `session`).
