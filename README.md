# AEGIS

Fleet-wide conjunction assessment and fuel-optimal maneuver planning.

This repository used to be a PINN/GNN + Kelvins Collision Avoidance experiment (`PIGNN-SAT`). The live product is **AEGIS**. Older Kelvins methods, notebooks, and study notes are in `archive/`.

```
aegis/     Python package, tests, step contracts, operations console
docs/      What the system will and will not claim
archive/   Historical Kelvins / PINN work — not on the default path
```

## Run

```bash
cd aegis
PYTHONPATH=src python3 -m aegis.api
```

Open http://127.0.0.1:8000

Pipeline only (no UI):

```bash
cd aegis
PYTHONPATH=src python3 -m aegis.pipeline --tle-path tests/fixtures/starlink_slice.tle --output /tmp/plan.json
```

Tests:

```bash
cd aegis
PYTHONPATH=src python3 -m pytest tests -q
```

Default catalog is live CelesTrak Starlink. If that fetch fails, the committed slice is used. Synthetic catalogs require both `AEGIS_ALLOW_SYNTHETIC=1` and an explicit acknowledge flag. There is no silent fallback to fake data.

See `docs/LIMITATIONS.md` before treating any number as operational.
