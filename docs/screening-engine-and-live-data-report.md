# Full-catalog screening and live data

**Dates:** 2026-09-18 to 2026-09-19
**Scope:** Space-Track ingest, the screening engine rebuild, the screening-box study, and measured uncertainty.

The numbers below come from the full Space-Track catalog of Starlink plus
every object whose orbit crosses Starlink's altitudes: 16,091 objects on
2026-09-18. "Starlink box" means SpaceX's screening box (section 3).

## 1. Summary

| | Before | After |
|---|---|---|
| Full-fleet screen, 3 days (14k objects) | Did not finish; the broad phase alone was projected at 2.2 h and ran out of memory | **~100 s** on 11 cores; about 26 min on one core at 1.5 GB |
| Full band catalog (16k), 24 h, all cores | — | **42 s** |
| Starlink conjunctions reproduced vs CelesTrak SOCRATES | 42% (Starlink + debris catalog) | **80%** with every object crossing the band |
| Data source | CelesTrak only | Space-Track (throttled, cached), then CelesTrak, then committed slices |

## 2. Live data

`aegis/src/aegis/ingest/spacetrack.py` downloads three things:

- GP elements for the Starlink fleet;
- GP elements for non-payload objects crossing the LEO band;
- the `cdm_public` feed.

It stays under Space-Track's API limits (below 30 requests per minute and
300 per hour) with a sliding-window throttle. Readers accept a cache up to
2 h old, and the scheduled refresh re-downloads after 50 min, so only one
download happens per hour. Credentials are read only from
`SPACETRACK_USER` / `SPACETRACK_PASS`, loaded from a gitignored `.env`.

The public CDM feed contains only emergency-reportable events: every event
with a Pc has Pc at or above 1e-4. On the first pull, none of those events
involved a Starlink.
`python -m aegis.pipeline.crosscheck` screens each event's pair with AEGIS and
compares the result with 18 SDS. On 35 events (142 CDM rows):

- AEGIS found all 35.
- Median TCA offset was 0.1 s; median miss error was 0.49 km.
- AEGIS's Pc was a median ~200x below 18 SDS, because AEGIS's TLE covariance
  dilutes it.

The VM archives this comparison hourly (`deploy/oracle/`).

## 3. The screening box

SpaceX Space Safety defines the box this way
(docs.space-safety.starlink.com, *Architecture and Internals*):

- **Shape.** A rectangular box with half-widths of 2 km radial, 44 km
  along-track and 51 km out of plane. It sits in each object's own RTN frame:
  radial along position, normal along orbital angular momentum, transverse
  completing the triad.
- **Orientation.** The box turns with the orbit, not with the satellite's
  body attitude.
- **Direction.** It is checked from both objects.
- **Close approach.** A local minimum of relative distance that falls inside
  the box. A CDM is generated for everything inside, even at Pc 0.
- **Decision.** Taken on 2D Alfano Pc.

The dimensions are exactly the 18/19 SDS **Early Orbit** near-Earth screening
volume (*Spaceflight Safety Handbook for Operators*, Table 7, 2 x 44 x 51 km,
7 days). That is the most generous standard near-Earth volume. For Starlink's
altitude, the 19 SDS standard volumes are 0.4 x 25 x 25 km (catalog,
Table 3) and 2 x 25 x 25 km (operator ephemeris, Table 4). The box is thin
radially because altitude is the best-known part of an orbit, and long
along-track because timing error grows every orbit.

AEGIS implements it the same way: both directions, acceptance at the refined
TCA, and the box in each object's own RTN frame. `tests/test_screening_box.py`
pins this.

### What the box holds, and where risk actually is

Starlink-involving conjunctions over 3 days, with AEGIS's Pc on a stratified
sample of 1,500 per band (hard-body radius 5 m per satellite, 1 m per debris):

| Miss distance | Conjunctions | Pc ≥ 1e-4 | Pc ≥ 1e-5 | Pc ≥ 1e-7 |
|---|---|---|---|---|
| < 0.2 km | 594 | 2.2% | 96.5% | 100% |
| 0.2–1 km | 11,923 | ≤1% | 45–84% | ~99% |
| 1–5 km | 88,154 | ≤0.1% | ~29% | ~76% |
| 5–10 km | 105,354 | 0 | 11.9% | 70.4% |
| 10–20 km | 232,410 | 0 | 0 | 51.4% |
| 20–40 km | 698,562 | 0 | 0 | 7.5% |
| > 40 km | 663,014 | 0 | 0 | **0** |

Three independent sources agree that action-level risk (Pc ≥ 1e-4) sits
within about 2.5 km:

- AEGIS: under 2 km.
- SOCRATES worst-case probability ≥ 1e-4: under 2.4 km.
- 18 SDS public CDMs, which use real covariance: all under 1 km.

About 82% of box conjunctions are below the 1e-7 level at which 19 SDS
reports a CDM.

### Sphere vs box, and the radial slab

A single 24 h screen covered both a 45 km sphere and the Starlink box:

| | Starlink passes | Highest Pc in a 1,500 sample |
|---|---|---|
| Inside both | 304,957 | 1.9e-4 |
| Sphere only | 3,926,052 | 1.6e-10 |
| Box only | 123,471 | 8.0e-9 |

The sphere needs about 10x the work for no added risk, because uncertainty is
anisotropic (section 5). Close passes excluded by the 2 km radial slab tell the
same story. Of 4,031 passes with a total miss under 2.5 km but a radial offset
of 2–2.5 km, the highest Pc was 3.6e-7 and none reached 1e-5. The thin
radial side is therefore sound.

## 4. Screening engine

These are the stages behind `screen()` / `screen_table()`, in
`aegis/src/aegis/screening/`:

1. **Candidates.** A k-d tree over positions at each sample, bounded by the
   19 km/s maximum closing speed.
2. **Chord gate.** A pair is kept for an interval if the segment between its
   sampled relative positions, minus a `(1/8) A dt^2` curvature bound, can
   reach the box's circumscribing sphere. The gate uses positions at both
   ends, not SGP4 velocity. For e≈0.3 objects, SGP4's velocity disagrees with
   its own positions by tens of m/s.
3. **Radial test.** Passes whose orbital radii cannot come within the box's
   radial reach are not refined.
4. **Events.** A pair's close run is split into approaches at range maxima
   that lie outside both boxes.
5. **Batched TCA refinement.** This runs the scalar `refine_tca` search,
   vectorised, with one `sgp4_array` call per object per stage.
6. **Acceptance.** A pass is kept when its TCA lies inside either object's
   box.

Time blocks stream (the full grid is never held in memory) and run in spawned
worker processes. Results are columnar (`ConjunctionTable`).

### Evidence that nothing was lost

| Check | Result |
|---|---|
| Gate vs 0.5 s propagation, 96,265 real pairs (incl. 35,043 dropped pairs that came within 20 km of the threshold) | 0 false negatives; worst curvature used 54% of the bound |
| New vs previous engine on real subsets (launch trains, 3-day windows) | Every previous conjunction found, identical TCA and miss; additions were repeat approaches |
| Batched vs scalar refinement, 6,000 real passes | Miss difference 0.000 m; TCA within 0.7 ms; no fallbacks |
| Radial test on vs off (4,000 objects, both boxes) | 0 missing, 0 extra |
| 60 s vs 30 s step, 4,000 objects, 12 h | 14,069 fast passes identical |
| Parallel vs serial, full catalog | Identical 1,834,847 conjunctions |

The comparison also exposed a bug in the previous engine. In its spatial hash,
swapping the loop variable in place dropped about 25% of candidate pairs in
crowded catalogs, a false negative its docstring said could not happen. That
code has since been removed.

## 5. Uncertainty, measured

Space-Track `gp_history` (8 days, 400 objects) gives the error of predicting
each object from an older element set to a newer one. These are along-track
errors, median with the 90th percentile in parentheses:

| | ~0.5 day | ~1 day | ~2 days | ~3 days |
|---|---|---|---|---|
| Starlink | 3.1 km (20) | 11 km (87) | 30 km (505) | 69 km (1,667) |
| Debris / rocket bodies | 0.2 km (1.0) | 0.5 km (2.7) | 1.1 km (7.2) | 2.1 km (20) |
| Other payloads | 0.2 km (1.0) | 0.5 km (2.8) | 1.0 km (7.9) | 1.8 km (26) |

Radial and cross-track errors are 0.05–0.5 km for stable objects.

Sources, ordered by contribution:

1. Starlink manoeuvres. These are not in TLEs: 15–19% of Starlink element-set
   pairs contain a semi-major-axis jump over 0.5 km.
2. Starlink along-track drift between burns.
3. Prediction horizon.
4. Drag on other objects.
5. Radial and cross-track error.
6. Element-set fit error at epoch.
7. AEGIS's numerics (metres, sub-millisecond).

For Pc, covariance realism and hard-body radius act as multipliers.
AEGIS's `TleCovarianceModel` is about 7x too small along-track for Starlink
and about 10x too large for debris.

## 6. Coverage vs SOCRATES

Over a 3-day window with 47,796 SOCRATES Starlink events:

| Catalog | Both objects in catalog | Matched by AEGIS |
|---|---|---|
| Starlink + LEO debris (the current ops catalog) | 53% | 42% of all, 79% of in-catalog |
| + every payload crossing the band (16,091 objects) | 99.5% | **80.1%** of all |

Match rate stays at 77–81% in every SOCRATES miss band. It drops to 67% for
SOCRATES worst-case probability ≥ 1e-4. Nearly all unmatched events are
element-set differences: AEGIS's newer GP puts the pair outside the box. Some
SOCRATES elements were 2–5 days old. Neither side is ground truth.

## 7. Slow and formation pairs

Over 24 h there were 3,218 Starlink conjunctions below 0.5 km/s (0.7%);
97% were Starlink–Starlink. Their median relative speed is 409 m/s. At that
speed two similar satellites carry about 80 J/g, twice the ~40 J/g
catastrophic-breakup threshold. These are slow crossings between neighbouring
planes, so they are real threats and not formation flight. True formation
flight (cm/s to a few m/s) is essentially absent from the catalog.

For such pairs the TCA is ill-defined, and the reported TCA wobble can move
with the grid step. They are flagged `low_relative_velocity`.

## 8. Open items

- Batched Pc and risk tiers:
  - Act ≥ 1e-4;
  - Watch ≥ 1e-5;
  - Monitor ≥ 1e-7;
  - a formation category below ~10 m/s.
  After that, move the console and pipeline onto full-fleet results.
- Load every object crossing the band in the ops catalog. So far it has only
  been measured through an experiment script.
- Recalibrate `TleCovarianceModel` from section 5.
- Replace `alfano.max_collision_probability`. It is a round-covariance bound,
  exceeded 10–30x by elongated covariances, so it is not a true ceiling.
- Use real hard-body radii (Pc scales as HBR²).
- SpaceX's own Starlink ephemerides. Space-Track lists a
  `public-data-files-552-spacex-prod` directory, not yet examined, and
  public-file downloads are limited to 3 per day.
