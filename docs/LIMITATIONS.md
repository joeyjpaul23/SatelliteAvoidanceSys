# AEGIS limitations

AEGIS is a research prototype for constellation conjunction assessment and
fleet maneuver planning. This page states what the system does **not** claim.

## 1. TLEs carry no covariance

Two-line elements do not include uncertainty. AEGIS synthesizes a covariance
and tags it `SYNTHETIC_TLE`. That tag is provenance, not a calculated
orbit-determination result.

## 2. Synthetic covariance is for ranking, not operations

The synthetic TLE covariance supports ranking and triage of close approaches.
It is **not** for operational maneuver decisions. Do not treat a planned burn
as flight-ready because a synthesized Pc dropped below a threshold.

## 3. Intra-fleet Pc from TLEs is not CDM quality

Collision probabilities computed from TLE-grade ephemeris and synthetic
covariance are not operator-grade Conjunction Data Message quality. They are
a screening and research signal, not a replacement for owner/operator CDMs.

This gap is now measured, not just stated. `python -m aegis.pipeline.crosscheck`
(and the console's VALIDATION panel) re-screens every event in Space-Track's
public CDM feed from GP elements and compares AEGIS with 18 SDS. On the first
live run (2026-09-18: 35 events from 142 CDM rows), the geometry agreed well:
AEGIS found every event, with a median TCA offset of 0.1 s and a median
miss-distance error of 0.49 km. The Pc did not agree: AEGIS was a median 10^2.3
(about 200x) below 18 SDS. AEGIS flagged covariance dilution on all 35 events,
and even the "worst-case" Pc reached the 18 SDS value on only 3 of 31, though that
worst case assumes round covariance and understates the true ceiling (section 9).
A half-kilometre
TLE miss error is as large as the misses being assessed. The VM archives every
check hourly, so these numbers become a growing dataset rather than a snapshot.

## 4. The maneuver plan is not flight commands

The maneuver plan is a prototype / research output. It is not a set of flight
commands. Burns are along-track impulses from a linear program on Clohessy–
Wiltshire responses; they are not validated guidance products.

## 5. Synthetic catalogs are opt-in and distinct from CelesTrak

Generated catalogs require `AEGIS_ALLOW_SYNTHETIC=1` plus an explicit
authorization. That path is opt-in. Synthetic objects must not be confused
with CelesTrak data.

## 6. Live-data failure is not a silent fallback

If Space-Track or CelesTrak ingest fails, the pipeline raises. It does **not**
silently become synthetic data. A refused or failed live fetch never
substitutes a generated catalog. The ops catalog falls back from Space-Track
to CelesTrak to the committed CelesTrak slices, and reports which one it used.

## 7. "Zero induced conjunctions" is a measurement, and it has conditions

The fleet planners reach zero SGP4-measured induced conjunctions across every
configuration tested **at the default 30 s enumeration step** — 900 plannable
configurations (nine families, sixteen seeds, four Δv budgets, three planners),
plus 366 more varying burn slots, refinement iterations, control axes, cost
model, probability threshold and risk level. That is a measurement on a
synthetic benchmark, not a theorem about any catalog.

**It does not hold at a coarser enumeration step.** At 60 s, 4 of 900 checks
produced a measured induced conjunction; at 120 s, 24 of 900, and most of those
still reported a safe certificate. `latent_step_s` is therefore a correctness
parameter, not a performance knob: coarsening it changes which close approaches
are visible at all. The enumeration sets `grid_coarser_than_validated` and
emits a note above 30 s, so a coarse run is reported as unvalidated rather than
passing for the validated one.

Two conditions are worth stating explicitly, because both were found the hard
way:

* **The constraint set is only as good as its search.** Constraints are placed
  at minima of the separation, and a minimum between two grid samples is
  invisible to any test on the samples themselves. A pair crossing at
  7.24 km/s sweeps 217 km per 30 s step; one measured case had a 143 m minimum
  bracketed by samples of 76 km and 141 km. Minima are now bracketed by
  range-rate sign change and refined by interpolation, which removes the
  dependence on the step — but a new geometry should be checked, not assumed.
* **The inflation bound is conditional on relative velocity.** The second-order
  inflation bounds the *curvature* of the relative trajectory. That is the
  governing quantity only when the relative velocity at the constrained
  minimum is near zero. For crossing geometry the governing quantity is
  linear in relative speed and vastly larger.
* **A pair has to be admitted before it can be constrained.** Candidate
  admission is decided on sampled separations, so the gate carries the
  inter-sample sweep ½·v_rel·h — the term that would be vacuous on the
  constraint floor but is nearly free on the gate. Widening the gate admits
  pairs that turn out not to bind, which costs enumeration time and nothing
  else; narrowing it silently drops fast crossings.

The certificate is evaluated on the closed row set and against the true
non-linear constraint, so a plan the optimizer could not make safe is reported
as unsafe. **A 2026-09-19 review found cases where that is not what the code
does; see section 12 before using any of these numbers.** It certifies the plan against the **linear dynamics model**, never
against SGP4; the SGP4 re-screen is a separate, empirical measurement, and the
two numbers are always reported together.

## 8. Starlink positions are the dominant uncertainty, and the covariance model is not calibrated to it

Measured on Space-Track element-set history, predicting from an older element
set to a newer one:

| Horizon | Starlink along-track error, median | Debris along-track error, median |
|---|---|---|
| 1 day | 11 km | 0.5 km |
| 3 days | 69 km (90th percentile 1,667 km) | 2 km |

Starlinks manoeuvre often, and TLEs cannot know a future burn. A Starlink
conjunction predicted more than about a day ahead is low-confidence whatever
the screening does. AEGIS's `TleCovarianceModel` does not reflect this yet: it
is about 7x too small along-track for Starlink and about 10x too large for
debris. See `docs/screening-engine-and-live-data-report.md` section 5.

## 9. The "worst-case" Pc is not a ceiling

`RiskAssessment.max_probability` is `R^2 / (e d^2)`, the maximum over
covariance scale for a *round* covariance. TLE covariances are elongated, 20
to 100 times longer along-track than radially. Their actual Pc can exceed that
number by 10 to 30 times. Treat it as a round-covariance reference, not a
bound, until it is replaced.

## 10. The ops catalog does not yet hold every object near Starlink

The console's catalog is Starlink plus non-payload LEO debris. About 47% of
CelesTrak SOCRATES Starlink events involve other operators' satellites, which
it does not load. With every object crossing the band (16,091 objects), AEGIS
reproduces 80% of SOCRATES Starlink events. Most of the rest are element-set
differences; neither side is ground truth.

## 11. Slow and co-orbiting pairs

Below 0.5 km/s the 2D Pc short-encounter assumption weakens. Conjunctions
there are flagged `low_relative_velocity`, and for co-orbiting pairs the
reported TCA is one of several near-equal minima and can move with the grid
step. Most such pairs in the catalog are Starlink–Starlink crossings at about
400 m/s, which is still well above breakup energy. They are intra-fleet events,
not formation flight, and remain risks.

## 12. The fleet optimizer's certificate has known gaps (found 2026-09-19)

A pre-push review (two independent Claude reviewers and Codex, each
reproducing its own findings) showed that `certified_safe=True` does **not**
currently mean what sections 7 and 11.x of `aegis/specs/step14_fleet_optimization.md`
say it means. Until these are fixed, treat every number from
`aegis.experiments` and from `paper/` as provisional. Nothing here touches the
screening, risk or console path: `aegis.fleetopt` is research code and the
console plans with `aegis.maneuver`.

| # | Gap | Where |
|---|---|---|
| 1 | Rows dropped when the latent set is thinned (on by default, 2,000–4,000 rows) are never re-checked, and the verifier never sees them. Reproduced: induced-cascade seed 8 certified safe while violating a dropped pair by 1.1 km and 3.2 km. | `fleetopt/latent.py` (`thin_constraints`) |
| 2 | A pair admitted but holding no row gets a floor of 0 km, so the perturbed-minimum search can never flag it. Reproduced: 2 minima below floor on a plan certified safe, one at 0.495 km. | `fleetopt/latent.py` (`perturbed_minimum_epochs`), `fleetopt/planners.py` (`build_context`) |
| 3 | `pignn-warm-start` runs one `sequential_solve`, skipping the two-stage polish and the rows added at perturbed minima, then certifies against the un-closed set. Reproduced: an induced minimum inside the floor on 6 of 8 seeds, all reported safe. | `fleetopt/planners.py` |
| 4 | `pignn-active-set` verifies against the problem as it stood *before* the row set closed, so rows added by polishing are not certified. | `fleetopt/planners.py` |
| 5 | A latent row whose violation is absorbed as slack is not counted as a violation, so a plan that deliberately induces a conjunction can still report `linearized_safe=True`. | `fleetopt/certify.py` (`verify_plan`) |
| 6 | The benchmark never checks solver status. A failed solve becomes a no-burn plan and a defined −100 % premium instead of an error. | `experiments/runner.py` |
| 7 | "Not measured" is stored as "0 induced", which is also how the headline zero is produced when measurement fails. | `experiments/runner.py`, `store/db.py` |
| 8 | `legacy-lp` burns are snapped to the fleetopt grid (measured: up to 3.5 orbits), so its measured induced count is not a measurement of the legacy planner. | `fleetopt/planners.py` |

Also found, and equally unfixed: the MILP charges `ops_cost_per_burn` once per
manoeuvring satellite rather than once per burn (`fleetopt/problem.py`); the
SGP4 "truth" miss in `experiments/metrics.py` is a 3 s sampled minimum with no
refinement, which overstated one linearisation error by about 70×; the
`debris-shower` scenario family ignores its seed, so 44 identical scenarios run
as if distinct; and `star` / `chain` scenarios place most designed crossings
after the screening window.
