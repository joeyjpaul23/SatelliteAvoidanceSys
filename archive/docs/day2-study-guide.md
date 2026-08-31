# Day 2 Study Guide — TLEs, SGP4, and How Satellites Are Actually Tracked

*Phase 1 of the 2-week build plan. Target: ~6–8 hrs. Assumes Day 1's orbital elements (a, e, i, ω, Ω, true/mean anomaly) are already fluent — today reuses them constantly.*

---

## Why this day matters

Day 1 gave you the physics of an orbit in the abstract. Today you meet the actual data format every downstream phase consumes: the **TLE**. Kessler's CDMs reference objects by catalog number, your live-data pipeline in Phase 5 pulls TLEs directly from CelesTrak/Space-Track, and your Pc calculator in Phase 7 needs propagated position/velocity — which comes from running a TLE through **SGP4**. If you don't understand what a TLE actually encodes and what SGP4 does to it, Phase 5's "pull TLEs, propagate, screen for close approaches" will feel like calling a magic function instead of engineering a pipeline.

Two things happen today, in order: (1) understand the TLE format itself, (2) understand what SGP4 does to a TLE and why it's only approximately right — then (3) prove it to yourself with a real script.

---

## Hour-by-hour plan

### Hours 1–2: The Two-Line Element format, field by field

**Read:** CelesTrak's TLE format explainer (celestrak.org — Dr. T.S. Kelso's "Frequently Asked Questions: Two-Line Element Set Format," plus the NORAD TLE format reference page).

**Core concepts:**

- A TLE is **two 69-character lines** (sometimes preceded by a name line — "three-line element," or 3LE) containing **mean orbital elements** at a specific reference time, formatted for a specific propagation model (SGP4/SDP4). It is *not* a raw state vector — it's a fitted set of elements meant to be fed into that specific model, and using it with a different propagator gives wrong answers.
- **Line 1 fields** — know these cold, you'll parse them by hand at least once today:
  - **Satellite (catalog) number** — NORAD's unique ID for the object, assigned in observation order (not launch order).
  - **Classification** — 'U' for unclassified; all public data is 'U'.
  - **International Designator** — launch year + launch-of-year + piece-of-launch (e.g., `98067A` = 1998, 67th launch, piece A — this is how the ISS is identified). Different from the catalog number: catalog number is assigned when first *observed*, international designator is tied to the original *launch*.
  - **Epoch** — the reference time all elements are given for, encoded as 2-digit year + fractional day-of-year (e.g., `98001.00000000` = 1998-01-01 00:00 UT). Note the two-digit-year quirk: 57–99 → 1957–1999, 00–56 → 2000–2056. This matters the moment you write a parser — off-by-a-century bugs are a classic first mistake here.
  - **First/second derivative of mean motion** — used only by the simpler SGP model, not SGP4/SDP4; largely vestigial for your purposes but you should recognize the fields.
  - **BSTAR drag term** — an SGP4-specific drag coefficient (B* = B·ρ₀/2, where B is the ballistic coefficient CD·A/m). This is the field that captures atmospheric drag effects — directly relevant later since drag is a major source of propagation uncertainty for LEO objects, which is part of why conjunction risk assessments degrade the further out from "now" you propagate.
  - **Element number, checksum** — bookkeeping; the checksum is a simple modulo-10 sum, good enough to catch most transcription errors but not authoritative validation (format/range checks matter more).
- **Line 2 fields** — the actual orbital elements you learned Day 1, in TLE units:
  - **Inclination** (degrees, 0–180)
  - **Right ascension of the ascending node (RAAN)** — this is the TLE's name for Day 1's "longitude of ascending node" (Ω)
  - **Eccentricity** — unitless, leading decimal point assumed (e.g., `0008546` → 0.0008546)
  - **Argument of perigee** (degrees) — Day 1's ω
  - **Mean anomaly** — *not* true anomaly; mean anomaly is a uniformly-varying-with-time angle used for the elliptical position calculation, converted to true anomaly internally by solving Kepler's equation. Know that this conversion step exists even if you don't derive it by hand.
  - **Mean motion** (revolutions/day) — this is a proxy for semi-major axis (via Kepler's third law); TLEs give you rev/day instead of a directly a distance.
  - **Revolution number at epoch** — a rev counter, useful for identifying *which* orbit pass a given epoch refers to.

**Hands-on exercise (do this, don't just read):** take one real TLE (pull any current one from CelesTrak, e.g. the ISS or a Starlink satellite), and manually pick out each field by counting columns. Confirm the modulo-10 checksum by hand on one line. This forces you to actually look at column positions instead of trusting a library to have parsed it correctly — useful the first time your own pipeline produces a garbage result and you need to check whether the bug is in your parsing or your propagation.

**Self-check:** given a raw TLE, can you state the object's catalog number, epoch (as a calendar date/time), inclination, and eccentricity without running any code?

---

### Hours 3–4: What SGP4 actually does (and doesn't do)

**Skim (don't fully absorb the derivation):** Vallado, Crawford, Hujsak & Kelso, *"Revisiting Spacetrack Report #3"* (AIAA 2006-6753) — the canonical modern reference implementation and description of SGP4/SDP4.

**What to take away — the conceptual model, not the equations:**

- **Input → output.** SGP4 takes a TLE (mean elements at epoch) plus a target time, and outputs a **position and velocity vector** (state vector) at that time, in a specific inertial reference frame (TEME — True Equator, Mean Equinox — not the same frame as J2000/ECI used elsewhere; this matters if you ever combine SGP4 output with other coordinate data and get confusing offset errors).
- **SGP4 vs SDP4.** SGP4 is used for "near-Earth" objects (orbital period < 225 minutes); SDP4 extends it for "deep-space" objects (period ≥ 225 minutes, e.g., GEO, Molniya) with additional lunar/solar gravity and resonance terms. Modern implementations (including Python's `sgp4` package) select the right one automatically based on the TLE's mean motion — you generally call one API and it dispatches internally.
- **Why it's a *simplified* perturbation model.** SGP4 analytically approximates the dominant perturbations — Earth's oblateness (the J2, J3, J4 zonal harmonic terms — Earth isn't a perfect sphere, it bulges at the equator, which drags orbital planes around over time) and atmospheric drag (via the BSTAR term) — using closed-form equations rather than numerically integrating the full force model. This is *why* it's fast enough to propagate thousands of objects but also *why* its accuracy degrades: real perturbations (solar/lunar third-body effects for higher orbits, real time-varying atmospheric density, unmodeled maneuvers) aren't fully captured.
- **Accuracy degrades with propagation time.** A TLE is a snapshot fit to recent tracking data; the further you propagate from its epoch, the more the real trajectory diverges from SGP4's prediction — typically kilometers of error after a few days for LEO objects, worse for higher-drag objects. This is *the* reason conjunction screening has to keep re-pulling fresh TLEs rather than propagating one old TLE for weeks, and it's also why real CDMs come with **covariance** (uncertainty), not just a point prediction — foreshadowing Day 3.
- **You do not need to hand-derive the equations.** The goal is to be able to say, accurately: "SGP4 takes mean elements and a time, analytically propagates through an approximate Earth-oblateness-plus-drag model, and outputs a state vector in TEME — and it's an approximation that gets worse the further from epoch you push it." That sentence is the entire conceptual payload of the report for this project's purposes.

**Self-check:** if someone asked you "why not just use the TLE's raw elements directly instead of running SGP4?", could you answer in terms of mean vs. instantaneous elements and the mean-anomaly-to-true-anomaly/perturbation step? If not, re-skim the report's introduction section.

---

### Hours 5–6 (hands-on): Propagate a real TLE yourself

This is the moment orbital mechanics stops being abstract — budget real time for it, don't rush.

**Setup:**
```bash
pip install sgp4 --break-system-packages
```

**Exercise:**
1. Pull one real, current TLE from CelesTrak (celestrak.org has plain-text TLE group files, e.g. the active satellites or Starlink group — no auth required for basic groups).
2. Parse it with the `sgp4` package's `Satrec.twoline2rv(line1, line2)`.
3. Propagate it forward using `.sgp4(jd, fr)` (Julian date + fractional day) for a handful of times spread over the next few hours, and print out the resulting position/velocity vectors.
4. Propagate the *same* TLE for a time a week or two past its epoch and compare — notice that you get an answer with no built-in warning that it's now much less trustworthy. This is worth doing deliberately: it makes concrete the "accuracy degrades with propagation time" point from hours 3–4, and previews why your Phase 5 pipeline will need to care about TLE freshness (age since epoch) when deciding how much to trust a screening result.
5. Sanity-check your output: does the position vector's magnitude (distance from Earth's center) roughly match altitude + Earth's radius (~6378 km) for the object you picked? Does the velocity magnitude look like a plausible orbital speed (~7.5 km/s for LEO)? This kind of order-of-magnitude sanity check is a habit worth building now — you'll want it again when your ML model and Pc calculator start producing numbers you need to trust or distrust.

**Stretch (if time allows):** pull two TLEs for objects you'd expect to be reasonably close (e.g., two Starlink satellites from the same launch batch), propagate both over the same time window, and manually compute the distance between them at each time step. This is a hand-rolled, tiny preview of the "screening" step you'll build for real in Phase 5 — worth doing now while the mechanics are fresh, even in a throwaway five-line script.

---

## End-of-day checkpoint (from the build plan)

You should be able to:

1. Read a raw TLE and identify catalog number, epoch, inclination, eccentricity, and RAAN by eye.
2. Explain what SGP4 takes as input, what it outputs, and in one sentence, why its accuracy is approximate rather than exact.
3. Have a working script that pulls a real TLE and propagates it to get a position/velocity at a chosen time — and have actually run it, not just read about how to.

If step 3 isn't done, that's the one to prioritize before Day 3 — Day 3's CDM work assumes you already have a working propagation script to build on, not just conceptual understanding.

---

## Reference links for today

- CelesTrak, TLE format FAQ — https://celestrak.org/columns/v04n03/
- CelesTrak, NORAD TLE format reference — https://celestrak.org/NORAD/documentation/tle-fmt.php
- CelesTrak, current TLE data (by group) — https://celestrak.org/NORAD/elements/
- Vallado, Crawford, Hujsak & Kelso, "Revisiting Spacetrack Report #3" — https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev1.pdf
- Python `sgp4` package docs — https://pypi.org/project/sgp4/
