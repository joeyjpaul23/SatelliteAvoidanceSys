# Day 1 Study Guide — Orbital Mechanics Fundamentals

*Phase 1 of the 2-week build plan (PIGNN-SAT / AI Space Debris Collision-Risk Copilot). Target: ~6–8 hrs, zero prior space-domain knowledge assumed.*

---

## Why this day matters

Every later phase — the Bayesian LSTM, the Pc calculator, the agent's tool calls, the live TLE screening — talks about orbits using a shared vocabulary (apogee, inclination, TCA, propagation). If that vocabulary is shaky, Days 4–13 turn into copy-pasting tutorial code without understanding what it computes. Today's only job is to make that vocabulary solid and intuitive.

Two sources, used in sequence, not in parallel:
1. **NASA JPL, "Basics of Space Flight"** (solarsystem.nasa.gov/basics) — conceptual, narrative, written for new JPL ops staff. Read this *first* per topic.
2. **Braeunig, "Orbital Mechanics"** (braeunig.us/space/orbmech.htm) — the same topics with the actual math. Read this *second*, once JPL has given you the mental picture.

---

## Hour-by-hour plan

### Hours 1–2: Reference systems + why orbits need six numbers

**Read (JPL):** Section I, "Reference Systems" chapter.

**Read (Braeunig):** the "Conic Sections" and "Orbital Elements" sections at the top of the page.

**Core concepts to absorb:**

- **Reference frames.** An orbit's numbers are meaningless without a coordinate system to measure them in. Learn the idea of a frame anchored to the Earth's equator and the vernal equinox direction — this is what "longitude of ascending node" is measured against later.
- **Conic sections.** All orbits under a single dominant gravity source are one of four shapes: circle, ellipse, parabola, hyperbola. Which one you get is determined by *eccentricity (e)*:
  - e = 0 → circle
  - 0 < e < 1 → ellipse (this is what nearly every satellite and all debris conjunctions you'll model use)
  - e = 1 → parabola (escape trajectory, borderline)
  - e > 1 → hyperbola (interplanetary flybys, not relevant to this project)
- **The six orbital elements.** Memorize these cold — you'll type these variable names into code by Day 5:
  1. **Semi-major axis (a)** — half the long axis of the ellipse; effectively the orbit's mean size/distance from Earth.
  2. **Eccentricity (e)** — how stretched the ellipse is (0 = circle, closer to 1 = very elongated).
  3. **Inclination (i)** — tilt of the orbital plane relative to Earth's equator. 0° = equatorial prograde, 90° = polar, 180° = equatorial retrograde.
  4. **Argument of periapsis (ω)** — angle from the ascending node to the point of closest approach (perigee), measured within the orbital plane.
  5. **Longitude of ascending node (Ω)** — angle from a fixed reference direction (vernal equinox) to the point where the satellite crosses the equator moving north.
  6. **Time of periapsis passage (T)** — when the satellite is actually at perigee (this is the "where on the ellipse is it *right now*" piece; often replaced in practice by **true anomaly (ν)**, the satellite's current angular position past perigee).

**Self-check before moving on:** can you draw an ellipse, label the two foci (Earth sits at one), mark perigee and apogee, and explain in one sentence what each of the six elements controls? If not, re-read before continuing — this is the single most load-bearing concept of the day.

---

### Hours 3–4: Gravity, Newton's laws, and why orbits are stable

**Read (JPL):** Section I, "Gravity & Mechanics" chapter.

**Read (Braeunig):** "Newton's Laws of Motion and Universal Gravitation" and "Uniform Circular Motion" sections.

**Core concepts:**

- **Newton's three laws**, specifically as they apply to a satellite: no force → straight line motion (law 1); force → acceleration, F = ma (law 2); gravity pulls the satellite toward Earth while the satellite's own inertia "pulls" it in a straight line — orbit is the compromise between the two (this *is* orbital motion, intuitively).
- **Universal gravitation:** F = G·m₁·m₂/r². The key derived constant you'll see constantly in code and papers is **μ (mu)** = G·M, Earth's standard gravitational parameter (≈ 3.986005×10¹⁴ m³/s²). Most orbital mechanics formulas are written in terms of μ, not G and M separately — get comfortable with that substitution now.
- **Why higher orbits are slower.** Derive (or at least trace through) why orbital velocity decreases as semi-major axis increases. This explains later, without hand-waving, why conjunctions in LEO evolve faster than in GEO — directly relevant to why your Pc calculator's time window matters.
- **Escape velocity vs. orbital velocity** — the distinction between "moving fast enough to stay in orbit" and "moving fast enough to leave entirely." Not core to this project, but it cements the energy intuition (negative energy = bound ellipse, zero = parabola/escape, positive = hyperbola — ties back to the conic section table from hours 1–2).

**Self-check:** explain, without looking, why a satellite doesn't fall into the Earth despite gravity constantly pulling on it. If your explanation doesn't mention *tangential velocity* and *free fall*, revisit.

---

### Hours 5–6: Trajectories, orbit types, and the vocabulary you'll see in TLEs/CDMs

**Read (JPL):** Section I, "Trajectories" chapter.

**Read (Braeunig):** "Types of Orbits" section.

**Core concepts — know what each of these is and why it exists, since you'll see them in real catalog data starting Day 2:**

- **LEO (low Earth orbit)** — where most debris and conjunctions you'll be working with live (roughly 160–2,000 km altitude). Fast orbital periods (~90 min), which is why conjunction geometry changes quickly and screening needs to run often.
- **Geosynchronous / geostationary (GEO)** — 24-hour period; geostationary specifically means i≈0° so the satellite appears fixed over one ground point. Different collision-risk character than LEO (much slower relative geometry, much larger separations).
- **Polar orbits** — i≈90°, used for mapping/surveillance; matters because polar-orbit debris crosses many other orbital planes, making it geometrically prone to conjunctions.
- **Sun-synchronous orbits (SSO)** — a walking/precessing orbit tuned so the satellite crosses a given latitude at the same local solar time every pass. Extremely common for Earth-observation satellites — expect to see many of these in real TLE data during Phase 5.
- **Molniya orbits** — highly eccentric, ~12-hour period, inclination fixed at 63.4° or 116.6° so the argument of perigee doesn't drift. Good example of *why* inclination and argument of perigee interact — sets up "orbit perturbations" conceptually for later.
- **Hohmann transfer orbits** — least-propellant transfer between two orbits. Not central to collision-risk modeling, but useful vocabulary since it appears throughout the source literature.
- **Perigee / apogee (and periapsis/apoapsis generally)** — closest/farthest point from the primary body. You already met these in hours 1–2; here, connect them to *why* they matter operationally: relative velocity between two objects is generally highest near perigee, which affects both collision energy and how fast risk geometry evolves.
- **True anomaly (ν)** — the satellite's actual angular position along the ellipse right now, measured from perigee. This is what changes moment-to-moment as the satellite moves; the other five elements are comparatively slow-changing (absent perturbations).

**Self-check:** given an orbit description (e.g., "e=0.001, i=53°, a=6900 km"), can you say roughly what kind of orbit this is (near-circular LEO, moderate inclination — sounds like a Starlink-class orbit) and what that implies about how often it might have conjunctions? You don't need a precise answer — the point is pattern recognition, which is what lets you sanity-check model outputs later instead of trusting them blindly.

---

### Hour 7 (buffer / synthesis): TCA and the collision-risk framing

The plan's own checkpoint asks you to explain **TCA — time of closest approach** — in a conjunction context. This isn't heavily covered by either source today (it's really a Day 3 CDM concept), but ground it now while orbital elements are fresh:

- Two objects each follow their own ellipse, each described by six elements evolving over time.
- A **conjunction** is a predicted event where two objects' propagated positions come close to each other at some future time.
- **TCA** is that specific future time — the moment the predicted miss distance between the two objects is smallest.
- Everything from Day 3 onward (CDMs, Pc, the ML model) is fundamentally about characterizing *how uncertain* and *how risky* that TCA moment is, given that neither object's position is known perfectly (measurement error, propagation error compounding over the days between "we spotted this" and "TCA actually happens").

This is the conceptual bridge from today's static orbital-geometry knowledge to the rest of the project, which is entirely about *two orbits interacting*.

---

## End-of-day checkpoint (from the build plan)

You should be able to explain, without notes:

1. What the six orbital elements are and what each one physically represents.
2. What apogee and perigee mean, and why objects at different altitudes move at different speeds (tie back to μ and the energy/semi-major-axis relationship).
3. What TCA means in a conjunction context.

If any of these three feel shaky, that's a signal to revisit the relevant hour block above before starting Day 2 — Day 2 (TLEs and SGP4) assumes all of this is fluent, not just recognized.

---

## Reference links for today

- NASA JPL, *Basics of Space Flight* — https://solarsystem.nasa.gov/basics/index.php
- Braeunig, *Orbital Mechanics* — http://www.braeunig.us/space/orbmech.htm
