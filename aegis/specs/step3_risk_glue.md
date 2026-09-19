# Step 3 contract: risk glue

Tester writes tests from this document only and must not read the new
implementation module(s) while writing tests. Builder must not edit `tests/`.

## Package

Add `aegis.risk.batch` (file `src/aegis/risk/batch.py`) and export from
`aegis.risk`:

- `assess_catalog`
- `RankedConjunction`
- `AssessedCatalog`

`from aegis.risk import assess_catalog, RankedConjunction, AssessedCatalog`
must work. Existing `assess` / `assess_projection` remain unchanged in
behavior.

## assess_catalog

```
assess_catalog(
    conjunctions: list[Conjunction],
    *,
    covariance_model: TleCovarianceModel | None = None,
    cross_check: bool = True,
    objects: list[SpaceObject] | None = None,
) -> AssessedCatalog
```

- If `objects` is provided, mixed `data_source` raises
  `aegis.ingest.MixedDataSourceError`.
- If conjunctions themselves mix primary/secondary `data_source` values
  across the list (any object's source differs from any other), raise
  `MixedDataSourceError`.
- For each conjunction, build primary and secondary covariances with
  `TleCovarianceModel` (default `default_covariance_model()`). Propagation
  age in days is (TCA − object.elements.epoch) / 86400 when elements exist,
  otherwise 0.
- Call existing `assess(...)`.
- Do not invent a new Pc method.

`AssessedCatalog` fields:

- `source: str` — the homogeneous `data_source` (`CELESTRAK`, `SPACETRACK` or
  `SYNTHETIC`),
  or empty string if there were zero conjunctions.
- `entries: list[RankedConjunction]` sorted by `assessment.probability`
  descending (highest risk first). Ties broken by `conjunction_id`.
- `covariance_source: str` — `CovarianceSource.SYNTHETIC_TLE` when the
  default TLE model was used.

`RankedConjunction` fields:

- `conjunction: Conjunction`
- `assessment: RiskAssessment`
- `rank: int` — 1 for the highest probability, then 2, ...

Every assessment produced with the default model must carry the synthetic
TLE honesty path: the covariances used are `CovarianceSource.SYNTHETIC_TLE`.
`AssessedCatalog.covariance_source` equals that constant. Do not strip or
relabel it as calculated.

`AssessedCatalog` helpers:

- `above(level: str) -> list[RankedConjunction]` where `level` is a
  `RiskLevel` name (`MONITOR`, `WATCH`, `ACT`). Returns entries whose
  `assessment.risk_level` is at least that severity (`RiskLevel.rank`).
- `by_id(conjunction_id: str) -> RankedConjunction`

Empty input yields an `AssessedCatalog` with empty entries and empty source.
