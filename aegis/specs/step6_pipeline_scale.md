# Step 6 contract: pipeline proof and modest scale

Tester writes tests from this document only. Builder must not edit `tests/`.
Do not read `src/aegis/pipeline/` if you are the tester (you may import
`aegis.pipeline` in test code).

This step extends Step 5. If `run_pipeline` already satisfies these
behaviors, builder only adds what is missing (e.g. a documented duration
helper). Prefer not to break Step 5.

## 3-sat known pair through the pipeline

`run_pipeline(source=SYNTHETIC, authorization=..., synthetic_spec=
SyntheticSpec(n_planes=1, sats_per_plane=3, include_known_conjunction_triple=True),
config=PipelineConfig with duration covering 1.5 periods of the spec epoch)`

must:

- Return `PipelineResult` with `source == SYNTHETIC`
- `len(result.catalog) == 3`
- `result.assessed` has at least one entry involving the first two objects
  (or `result.plan.resolved` addresses that pair)
- `result.plan` is a `ManeuverPlan`
- `result.covariance_source == CovarianceSource.SYNTHETIC_TLE`
- Complete in a few seconds on CPU (no 7-day screen)

## Modest fleet (tens, not thousands)

`SyntheticSpec(n_planes=3, sats_per_plane=8, include_known_conjunction_triple=True)`
→ 24 objects.

`run_pipeline` with `PipelineConfig(duration_s=5400, max_objects=24)` (or
equivalent 90-minute window) must:

- Return successfully (no uncaught exception)
- `len(result.catalog.objects) == 24`
- `result.plan` is a `ManeuverPlan` (0 burns is allowed if nothing is
  actionable)
- If any `plan.unresolved` entries exist, each has `shortfall_km >= 0`
  and `resolved is False`
- `result.plan.converged` is a bool
- Finish in under 60 seconds on a laptop for this 24-sat / 90-min case

Do not add a "thousands of objects" path in this step.

## Config stays outside the solver

`PipelineConfig.target_pc` and `dv_budget_km_s` must be forwarded to
planning. A test may set a huge `target_pc` (e.g. 0.5) and observe the
run still returns; a test may set `max_objects=5` on a 24-sat spec and
observe `len(catalog)==5` with source still SYNTHETIC.

## Regression

Existing Step 1–5 tests must keep passing. Mixed source and synthetic
gate behavior unchanged.
