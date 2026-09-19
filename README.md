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
is complete. **Caveat (2026-09-19):** a pre-push review found gaps between that
claim and the code — a plan can be reported as certified safe while violating a
constraint that thinning dropped. See section 12 of `docs/LIMITATIONS.md`. The
screening, risk and console paths are not affected.

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

Synthetic catalogs require both `AEGIS_ALLOW_SYNTHETIC=1` and an explicit
acknowledge flag. There is no silent fallback to fake data.

## Live data

| Source | Used when | What it provides |
|---|---|---|
| Space-Track.org | `SPACETRACK_USER` / `SPACETRACK_PASS` are set | GP elements for Starlink and LEO debris, plus the public CDM feed. Requests stay under the API limits and are cached for up to two hours; the hourly refresh keeps the cache under one. |
| CelesTrak | No credentials, or Space-Track fails | GP elements and the SOCRATES candidate feed |
| Committed TLE slices | Both live sources fail | `aegis/tests/fixtures/` |

Credentials load from a gitignored `.env` at the repo root:

```bash
cd aegis
PYTHONPATH=src python3 -m aegis.ingest.spacetrack check   # log in, count the fleet and upcoming public CDMs
PYTHONPATH=src python3 -m aegis.pipeline.crosscheck       # AEGIS vs 18 SDS on every upcoming public CDM
```

The console shows the same cross-check in its VALIDATION panel.

## Full-catalog screening

`aegis.screening.screen()` returns `Conjunction` objects. `screen_table()`
returns the same results as numpy columns, for catalog scale. Measured on this
laptop, using the Starlink screening box:

| Catalog | Window | 11 cores | 1 core |
|---|---|---|---|
| Starlink + every object crossing its altitudes (16,091) | 24 h | 42 s | — |
| Starlink + LEO debris (14,067) | 3 days | ~100 s | ~26 min, 1.5 GB |

The previous engine could not finish a 3-day screen of the fleet.

Close approaches use SpaceX Space Safety's screening box, which is the
18/19 SDS Early Orbit volume:

- **Size:** 2 km radial, 44 km along-track, 51 km out of plane.
- **Frame:** each object's own RTN frame, so the box turns with the orbit.
- **Direction:** checked from both objects.
- **Acceptance:** a pass counts when its closest approach falls inside the box.

Results do not depend on the grid step. The detection gate was validated
with no false negatives against 0.5 s propagation. With every object in the
band loaded, AEGIS reproduces 80% of CelesTrak SOCRATES's Starlink events.
Nearly all of the rest are differences between element sets.

[`docs/screening-engine-and-live-data-report.md`](docs/screening-engine-and-live-data-report.md)
has the evidence and the box study (where risk actually lives). It also has
the measured uncertainty: Starlink's own manoeuvres dominate it, at a
median 11 km along-track one day ahead.

## Always-on hosting

[`deploy/oracle/`](deploy/oracle/README.md) runs the console and an hourly
Space-Track refresh on an Oracle Cloud Always Free VM. It includes a script
that creates the VM and retries until Arm capacity frees up. The console is
reached over an SSH tunnel, with no public port.

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
