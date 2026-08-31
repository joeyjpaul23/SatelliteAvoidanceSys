# Step 8 contract: honest plan artifact + limitations

Tester writes tests from this document only. Builder must not edit `tests/`.

## Package additions

In `aegis.pipeline` (or `aegis.pipeline.export`):

```
write_plan_json(result: PipelineResult, path: str | Path) -> Path
write_plan_text(result: PipelineResult, path: str | Path) -> Path
plan_artifact(result: PipelineResult) -> dict
```

Export `write_plan_json`, `write_plan_text`, `plan_artifact` from
`aegis.pipeline`.

## plan_artifact dict (JSON-serializable)

Required keys:

- `source` — `CELESTRAK` or `SYNTHETIC`
- `covariance_source` — must be `SYNTHETIC_TLE` for the default model
- `object_count`
- `generated_at` — ISO UTC string
- `plan_id`
- `summary` — `result.plan.summary()` 
- `burns` — list of `{satellite_id, epoch, delta_v_rtn_km_s: [r,t,n], magnitude_km_s}`
- `resolved` — list of conjunction outcomes:
  `{conjunction_id, probability_before, probability_after, miss_distance_before_km,
    miss_distance_after_km, resolved, shortfall_km}`
- `unresolved` — the subset with `resolved is False`, **sorted by
  `shortfall_km` descending** (largest shortfall first)
- `warnings` — list of strings; must include a sentence that covariance
  is synthetic / TLE-grade and not for operational maneuver decisions
  when `covariance_source` is `SYNTHETIC_TLE`
- `converged` — bool
- `iterations` — int

`write_plan_json` writes that dict as JSON (indent=2).
`write_plan_text` writes a human-readable report that includes source,
covariance honesty line, summary numbers, each burn, and unresolved
pairs ranked by shortfall.

CLI `--output FILE.json` writes JSON via `write_plan_json`. If `--output`
ends in `.txt` / `.md`, write text instead.

## LIMITATIONS.md

Create `/Users/joeypaul/PIGNN-SAT/docs/LIMITATIONS.md` (repo `docs/`,
the path referenced from `constants.py`).

Must state, in plain language (tester will search for these ideas, not
exact poetry):

1. TLEs carry no covariance; AEGIS synthesizes one tagged `SYNTHETIC_TLE`.
2. Synthetic covariance supports ranking/triage, not operational
   maneuver decisions.
3. Intra-fleet Pc from TLEs is not operator-grade CDM quality.
4. The maneuver plan is a prototype / research output, not flight
   commands.
5. Synthetic catalogs are opt-in (`AEGIS_ALLOW_SYNTHETIC` +
   authorization) and must not be confused with CelesTrak data.
6. CelesTrak failure must not silently become synthetic data.

## What this step does not do

api/ui, GNN, live TraCSS, multi-source CDM consensus.
