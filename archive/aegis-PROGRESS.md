# AEGIS build progress

Constellation conjunction assessment and collision avoidance. Paused mid-build
at the user's request; this file records exactly where things stand so the work
can resume without re-deriving anything.

## Complete and verified

**`constants.py`** — every tunable number in the system, each with its source.
Screening volumes (Starlink's published 2 x 44 x 51 km RTN box, the 19 SDS
regime table, NASA CARA's, and a TLE-grade volume), Pc thresholds, quadrature
order, propulsion parameters, ingest rate limits.

**`core/`** — domain types with no I/O and no algorithms. `frames.py` (RTN and
encounter-plane construction), `state.py` (state vectors, covariance, ephemeris
with Hermite interpolation), `objects.py`, `conjunction.py`, `maneuver.py`,
`timebase.py`.

*Verified:* the encounter-frame identity `x_hat == r/|r|` at closest approach
holds to 1.1e-16.

**`risk/`** — collision probability. `projection.py` (reduce a conjunction to
the canonical 2D problem, with NPD covariance remediation), `alfano.py` (the
method Starlink documents using, via Gauss-Chebyshev quadrature), `chan.py`
(independent analytic cross-check), `assessor.py` (orchestration plus dilution,
remediation and short-encounter flags).

*Verified two independent ways:*
1. Against a brute-force 2D polar quadrature written separately for the
   purpose — agreement to machine precision (2.7e-16) on all six test
   geometries, including a far-tail case at Pc ~ 1e-29 where naive `erf`
   differencing collapses to zero.
2. Against published reference values. **Note for the record:** the reference
   table initially obtained had a unit inconsistency (hard-body radius in km
   while sigmas were in metres). Our values disagreed by ~6 orders of
   magnitude until that was identified; correcting it reproduces the published
   numbers to 7-8 significant figures. The independent quadrature above is what
   established which side was wrong.

Confirmed empirically: Gauss-Chebyshev reaches machine precision at 8 nodes;
Simpson's rule is still at 1.6e-6 relative error with 4096 intervals. Chan is
exact for isotropic covariance and degrades as predicted with scaled radius,
and the applicability gate correctly refuses the bad regimes.

**`propagation/`** — `propagator.py` (vectorised SGP4 over a shared time grid,
blocked to control memory) and `covariance.py` (synthetic TLE covariance model).

*Verified:* propagating a real ISS element set gives altitude 406-426 km and
speed 7.646-7.669 km/s, both physically correct.

## Not yet built

- `ccsds/` — CDM and OEM readers/writers (field tables researched, not written)
- `ingest/` — CelesTrak GP/SupGP client, cache, synthetic constellation generator
- `screening/` — prefilter, broad phase, smart sieve, TCA refinement
- `maneuver/` — CW dynamics, Pc-to-miss-distance mapping, fleet LP, re-screen loop
- `pipeline/`, `api/`, `ui/`, `tests/`

## Decisions worth not relitigating

- **Frame:** everything stays in TEME. Conjunction geometry is relative and
  therefore frame-consistent; converting would cost time and add error while
  changing no answer.
- **Quadrature:** Gauss-Chebyshev, not Simpson. The integrand has an infinite
  derivative at the disc edge which GC absorbs exactly. This is also what NASA
  CARA ships.
- **Covariance honesty:** TLEs carry no covariance. Everything synthesised is
  tagged `SYNTHETIC_TLE` and that tag must propagate into every assessment,
  CDM, and UI surface. Synthetic covariance supports triage and ranking, not
  operational maneuver decisions.
- **Slack variables in the fleet LP are mandatory, not optional.** Without them
  the problem is infeasible whenever a conjunction needs radial or cross-track
  displacement, because those responses are bounded and periodic while only
  along-track grows secularly.
- **Use the exact CW response `(4 sin(nt) - 3nt)/n`,** never the `3*dv*dt`
  shortcut — the shortcut is wrong by 85% at quarter-orbit lead times.

## Environment note

Neither this session's cloud container nor the device shell can reach
celestrak.org, so the CelesTrak client will be built to spec and validated
against the synthetic constellation generator rather than live data. Your own
machine has normal network access, so it will work when you run it.

Local Python is 3.10 with numpy, pandas, matplotlib, requests and pyyaml but
**without** scipy, sgp4, or fastapi — `pip install -r requirements.txt` will be
needed before running.
