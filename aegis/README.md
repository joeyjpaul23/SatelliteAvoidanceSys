# aegis

Python package for ingest → screen → assess → plan, plus the local operations console.

```
src/aegis/     code (dependencies point downward)
  api/         FastAPI + static console
  pipeline/    end-to-end orchestration
  maneuver/    fleet LP
  risk/        Alfano / Chan Pc
  screening/   close approaches
  propagation/ SGP4 + TLE covariance
  ingest/      CelesTrak vs gated synthetic
  ccsds/       CDM / OEM
  core/        domain types
  ui/          console HTML / CSS / JS / Earth texture
tests/         pytest (contracts in specs/)
specs/         builder/tester contracts per step
```

```bash
PYTHONPATH=src python3 -m aegis.api
PYTHONPATH=src python3 -m aegis.pipeline --tle-path tests/fixtures/starlink_slice.tle --output /tmp/plan.json
PYTHONPATH=src python3 -m pytest tests -q
```
