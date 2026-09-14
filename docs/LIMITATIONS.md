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

## 4. The maneuver plan is not flight commands

The maneuver plan is a prototype / research output. It is not a set of flight
commands. Burns are along-track impulses from a linear program on Clohessy–
Wiltshire responses; they are not validated guidance products.

## 5. Synthetic catalogs are opt-in and distinct from CelesTrak

Generated catalogs require `AEGIS_ALLOW_SYNTHETIC=1` plus an explicit
authorization. That path is opt-in. Synthetic objects must not be confused
with CelesTrak data.

## 6. CelesTrak failure is not a silent fallback

If CelesTrak ingest fails, the pipeline raises. It does **not** silently
become synthetic data. A refused or failed live fetch never substitutes a
generated catalog.

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
as unsafe. It certifies the plan against the **linear dynamics model**, never
against SGP4; the SGP4 re-screen is a separate, empirical measurement, and the
two numbers are always reported together.
