# Starlink Full-Catalog Screening Report

**Status:** Done with concerns  
**Verification date:** September 10, 2026

## Outcome

AEGIS can retrieve the free CelesTrak SOCRATES catalog and identify every published conjunction candidate where either object is a Starlink.

A cloud database was not the primary blocker. The main limitations were catalog coverage and the scope of AEGIS's local screening process:

- AEGIS screened only 40 objects by default and allowed at most 200.
- It downloaded Starlink and selected debris rather than the complete public catalog.
- Legacy TLE ingestion omitted newer six-digit catalog IDs.
- Local propagation was being treated as a full-catalog screening system when it was designed as a small-sample demonstrator.

## Implementation

- Added complete SOCRATES ingestion and filtering that retains every row where either object is a Starlink.
- Added free local caching with a stale-cache fallback for temporary network failures.
- Added `GET /api/starlink-alerts` with probability and miss-distance filters, pagination, source metadata, prediction-horizon metadata, and completeness counts.
- Changed live Starlink and debris ingestion to OMM JSON so six-digit NORAD catalog IDs are retained.
- Added parser, cache, schema-drift, API, and six-digit-ID tests.

## Verification Results

The public feed used for verification contained:

- 147,838 total conjunction rows.
- 108,123 conjunctions involving at least one Starlink.
- 10,950 unique Starlink catalog objects.
- A prediction horizon from September 10 through September 17, 2026.
- All seven example conjunction pairs supplied for the investigation.

The complete feed parsed in approximately 0.4 seconds. The full automated test suite passed with 224 tests.

## Cost and Storage Recommendation

No paid cloud service is currently necessary. AEGIS downloads one public feed of approximately 16 MB and caches it locally for two hours.

Cloud storage or a database becomes useful only if AEGIS needs to:

- Retain historical feed snapshots.
- Compare how predictions change over time.
- Serve multiple users or application instances continuously.
- Run long-term analytics and alert-delivery workflows.

For the present use case, the lowest-cost architecture is the public SOCRATES feed plus the local cache. A free object-storage or database tier can be added later for history without changing the screening source.

## Important Limitations

The returned events are public-data conjunction candidates. They are not confirmations that two objects will physically collide, and they are not operator-grade maneuver recommendations.

SOCRATES maximum probability is a conservative public screening metric and is not identical to an operator's covariance-based collision probability or AEGIS's Alfano probability calculation. The API labels the probability type explicitly to prevent those values from being confused.

Coverage is complete relative to the conjunction rows published in the current SOCRATES feed, not every possible conjunction that could be derived from private operator ephemerides. High-confidence operational decisions would still require current precision ephemerides, covariance data, and coordination with the relevant satellite operators.

## Relevant Source Files

- `aegis/src/aegis/ingest/socrates.py`
- `aegis/src/aegis/ingest/ops.py`
- `aegis/src/aegis/api/app.py`
- `aegis/tests/test_socrates.py`
- `aegis/tests/test_ops_catalog.py`
