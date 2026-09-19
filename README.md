# AEGIS

Fleet-wide conjunction assessment and **certified** fuel-optimal maneuver planning.

Given a network of simultaneous conjunctions across a maneuverable fleet,
AEGIS computes one coordinated set of impulsive maneuvers that resolves every
modelled conjunction to a target collision probability **while creating no new
ones**, and measures honestly what that costs.

That last clause is the research contribution. The multi-encounter collision
avoidance literature optimises one spacecraft against passive secondaries and
assumes, in Pavanello et al.'s words, that "there is no reason to suppose that
a close approach with a particular secondary is likely to promote a close
approach with some other secondary." At mega-constellation scale that is
measurably false: Chen et al. (arXiv:2406.06068) find **81.4 % of
Starlink-on-Starlink maneuvers are cascade-induced**, with one external event
triggering as many as 41 of them. Operationally the cascade is handled by an
outer re-screening loop — generate the ephemeris, resubmit, repeat — which can
discover an induced conjunction but cannot prevent one.

AEGIS puts the condition inside the optimizer, and proves the constraint set
is complete.

```
aegis/      Python package, tests, step contracts, operations console
docs/       What the system will and will not claim, and what was measured
deploy/     Always-on console on an Oracle Cloud Always Free VM
paper/      Whitepaper LaTeX source
orbitlab/   Separate C++20 orbital-dynamics sandbox
notebooks/  TLE propagation notebook
```

## The research layer

| Package | What it is |
|---|---|
| `aegis.fleetopt` | Three-axis B-plane maneuver optimization, certified induced-conjunction constraints, reachability bound, exact-penalty and Farkas analysis |
| `aegis.scenarios` | Ten deterministic, structurally-verified scenario families: nine synthetic and one replayed from committed TLE fixtures |
| `aegis.experiments` | The benchmark sweep, the safety-premium distribution, SGP4-measured validation |
| `aegis.ml` | A physics-informed conjunction GNN, used to accelerate the solve and never to decide it |
| `aegis.store` | Offline-first, cloud-optional artifact and experiment storage |

Read `aegis/specs/step14_fleet_optimization.md` for the contract,
`docs/prior-art-and-novelty-ledger.md` for what is and is not novel, and
`docs/model-fidelity-validation.md` for what was measured against SGP4.

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

Default catalog is live Starlink plus overlapping catalog debris, screened 3 days ahead. With `SPACETRACK_USER` and `SPACETRACK_PASS` set, both halves come from Space-Track (throttled, cached hourly); otherwise, or if Space-Track fails, from CelesTrak. If the live fetch fails, the committed slices are used. Credentials load automatically from a gitignored `.env` at the repo root. `python -m aegis.ingest.spacetrack check` verifies them. `python -m aegis.pipeline.crosscheck` compares AEGIS against 18 SDS on Space-Track's public CDM feed; the console shows the same comparison in its VALIDATION panel. To run the console always-on on an Oracle Cloud Always Free VM, see [`deploy/oracle/README.md`](deploy/oracle/README.md). Synthetic catalogs require both `AEGIS_ALLOW_SYNTHETIC=1` and an explicit acknowledge flag. There is no silent fallback to fake data.

Research benchmarks:

```bash
cd aegis
export AEGIS_ALLOW_SYNTHETIC=1
PYTHONPATH=src python3 -m aegis.experiments --help
PYTHONPATH=src python3 -m aegis.experiments scenario-list
# 2000 mm/s per satellite: the induced-cascade family has repeat encounters
# whose required miss under TLE-grade covariance exceeds what the 1000 mm/s
# default can reach, so the default leaves every scenario infeasible.
PYTHONPATH=src python3 -m aegis.experiments benchmark --acknowledge-synthetic \
    --family induced-cascade --seeds 8 --budget-mm-s 2000 --output /tmp/bench
PYTHONPATH=src python3 -m aegis.experiments frontier --acknowledge-synthetic \
    --family induced-cascade --seed 8
```

See `docs/LIMITATIONS.md` before treating any number as operational.
