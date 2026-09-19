# aegis

Python package for ingest → screen → assess → plan, plus the local operations console.

```
src/aegis/     code (dependencies point downward)
  api/         FastAPI + static console
  ui/          console HTML / CSS / JS / Earth texture
  pipeline/    end-to-end orchestration; crosscheck vs 18 SDS public CDMs
  maneuver/    fleet LP
  risk/        Alfano / Chan Pc
  screening/   close approaches: streaming sweep, batched TCA refinement
  propagation/ SGP4 + TLE covariance
  ingest/      CelesTrak / Space-Track vs gated synthetic; ops catalog
  ccsds/       CDM / OEM
  core/        domain types
  constants.py every tunable number, with its source
  envfile.py   repo-root .env loader for the entry points
research layer (see the top-level README)
  fleetopt/    certified fleet maneuver optimization
  scenarios/   deterministic scenario families
  experiments/ benchmark sweeps and SGP4 validation
  ml/          physics-informed conjunction GNN
  store/       artifact and experiment storage
tests/         pytest (contracts in specs/); fixtures/ holds the committed TLE slices
specs/         builder/tester contracts per step
```

```bash
PYTHONPATH=src python3 -m aegis.api                        # console on 127.0.0.1:8000
PYTHONPATH=src python3 -m aegis.pipeline --tle-path tests/fixtures/starlink_slice.tle --output /tmp/plan.json
PYTHONPATH=src python3 -m aegis.ingest.spacetrack check    # verify Space-Track credentials (.env at repo root)
PYTHONPATH=src python3 -m aegis.pipeline.crosscheck        # AEGIS vs 18 SDS public CDMs
PYTHONPATH=src python3 -m pytest tests -q
```

`screening.screen_table()` is the catalog-scale entry point. It streams time
blocks through worker processes, so call it under `if __name__ ==
"__main__":` from your own scripts. `AEGIS_WORKERS` caps the worker count.
