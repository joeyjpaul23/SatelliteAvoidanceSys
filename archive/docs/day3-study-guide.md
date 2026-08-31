# Day 3 Study Guide — CDMs, Conjunctions, Probability of Collision + Environment Setup

*Phase 1 of the 2-week build plan (PIGNN-SAT / AI Space Debris Collision-Risk Copilot). Target: ~6–8 hrs. Assumes Day 1 (orbital elements) and Day 2 (TLEs, SGP4, working propagation script) are solid — today builds directly on both.*

---

## Why this day matters

Day 1 gave you the physics of a single orbit; Day 2 gave you how one object's position is tracked and propagated. Today is where the project's actual subject — **risk between two objects** — enters the picture. A CDM (Conjunction Data Message) is the real-world artifact everything downstream consumes: your Phase 2 LSTM is trained on sequences of CDMs, your Phase 3 Pc calculator reproduces the number a CDM reports, and your Phase 4 agent talks about CDM fields by name. This is also the last "pure learning" day of Phase 1 — by the end of today your dev environment needs to be live, not just your understanding.

Three things happen today, in order: (1) understand what a CDM is and what Pc means, (2) understand the Foster 2D method conceptually (you'll implement it for real on Day 7), (3) get Kessler installed and the Kelvins dataset loading.

---

## Hour-by-hour plan

### Hours 1–2: CDMs and the collision-risk framing

**Read:** Space-Track.org, **"How the JSpOC Calculates Probability of Collision"** — https://www.space-track.org/documents/How_the_JSpOC_Calculates_Probability_of_Collision.pdf. This is short and operationally written — read it in full before touching anything denser.

**Core concepts:**

- **What a CDM is.** A Conjunction Data Message is a standardized (CCSDS `.kvn` format) report issued when two tracked objects are predicted to pass close to each other. It's not a one-time alert — the same conjunction event typically gets *multiple* CDMs issued as the TCA approaches and tracking data improves, which is exactly why Kessler groups them into `Event` objects (a sequence of CDMs about one close approach) rather than treating each CDM as independent.
- **Key fields you'll use directly** (you already met TCA on Day 1; these build on it):
  - **TCA** — time of closest approach, same concept as Day 1, now with a real predicted value attached.
  - **MISS_DISTANCE** — the predicted separation between the two objects at TCA.
  - **RELATIVE_SPEED** — how fast the two objects are closing at TCA; combined with miss distance, this is what makes an event "risky" versus "close but slow."
  - **COLLISION_PROBABILITY (Pc)** — the actual number everything is building toward. A single scalar summarizing risk, but one that depends heavily on assumptions (object size, covariance quality) that aren't visible if you only look at the number itself.
  - **Covariance** — each object's position/velocity uncertainty, usually given as a 3x3 or 6x6 covariance matrix in a relative encounter frame. This is *why* Pc isn't just "did the miss distance go below some threshold" — a tiny miss distance with tight covariance can be less risky than a larger miss distance with sloppy tracking data.
- **Why Pc, not just miss distance.** Two objects with identical miss distance can have very different collision probability depending on their combined position uncertainty and physical size (hard-body radius). This is the conceptual jump from Day 1–2's "where are the objects" to today's "how confident are we, and does that confidence matter more than the raw distance."
- **The Foster 2D method (conceptual pass only — you implement this on Day 7).** The operationally standard approach: project both objects' combined position covariance onto the 2D plane perpendicular to their relative velocity at TCA (the "encounter plane"), collapse the problem to a 2D Gaussian integral over a combined hard-body-radius circle. It's called "2D" because it deliberately ignores the along-track (time) dimension — a simplification that's accurate enough operationally and much cheaper than a full 3D numerical integration. You don't need the integral itself today; you need to be able to say *why* the problem reduces to 2D and *what* is being integrated (probability mass of the combined position-uncertainty ellipse falling within the combined object-size circle).

**Self-check:** given a CDM with a small miss distance but tight covariance, and another with a larger miss distance but loose covariance, can you explain in your own words which one is *not* necessarily the riskier event, and why?

---

### Hours 3–4: The CDM field reference, via Kessler's own tutorials

- Read through the CDM field reference as presented in Kessler's tutorials (real fields like `MISS_DISTANCE`, `RELATIVE_SPEED`, `COLLISION_PROBABILITY` appear directly in the `Event`/`EventDataset` objects you'll use starting Day 4) — this connects the conceptual fields from hours 1–2 to the actual data structure you'll load in code today.
- Note which fields are *reported* (came directly off a real CDM) versus *derived* (computed by Kessler or by you later) — this distinction matters once you're deciding what your Pc calculator (Day 7) should take as input versus what it should independently compute and compare against.

**Self-check:** could you point to where `COLLISION_PROBABILITY` and `MISS_DISTANCE` would live in a raw `.kvn` file versus in a loaded `Event` object, conceptually?

---

### Hours 5–7 (hands-on): Environment setup — Kessler + Kelvins dataset

This is the environment-setup half of today — budget real time, since Days 4–6 assume this all works.

**Setup:**
```bash
pip install kessler --break-system-packages
```
If `pip install kessler` gives you trouble, fall back to the conda/mamba install path (`conda install conda-forge::kessler`) — check the actual installed `LICENSE` file either way (GPL-3.0 per GitHub vs. BSD-3-Clause per conda-forge is a real discrepancy worth resolving now rather than at Day 14 writeup time).

**Exercise:**
1. Work through Kessler's **"Basics: loading CDMs"** tutorial — https://kesslerlib.github.io/kessler/notebooks/basics.html. Confirm you can load a `.kvn` file into a `ConjunctionDataMessage`/`Event` object and inspect its fields.
2. Download the **ESA Kelvins Collision Avoidance Challenge dataset** — https://kelvins.esa.int/collision-avoidance-challenge/data/ — you'll need the full training set for Phase 2 (Day 4 onward), so pull it now rather than partway through Day 4.
3. Confirm the Kelvins CSV loads via `kessler.data.kelvins_to_event_dataset` and check the row-to-event collapse (~24,000 raw CDM rows → ~1,700 grouped events is the expected order of magnitude from the docs) — if your numbers are wildly different, something's off in the load step, not in your understanding.
4. Sanity-check one loaded event by eye: pick one `Event`, print its sequence of CDMs, and confirm `MISS_DISTANCE`/`RELATIVE_SPEED`/`COLLISION_PROBABILITY` look like plausible real values (miss distances on the order of meters to kilometers, Pc values mostly very small with occasional higher-risk outliers) — the same order-of-magnitude sanity-check habit from Day 2's propagation exercise applies here.

**Stretch (if time allows):** open the CDM field list next to your Day 2 propagation script's output and note which fields (position/velocity-derived ones) you could, in principle, reproduce yourself from a TLE + SGP4 — this previews the Day 7 "check the CDM's claimed risk rather than just trusting it" goal.

---

## End-of-day checkpoint (from the build plan)

You should be able to:

1. Explain, in your own words, what a CDM is and why one conjunction event produces a sequence of them over time.
2. Explain why Pc matters — specifically, why miss distance alone doesn't determine risk.
3. Explain what SGP4 does and doesn't do (Day 2 recap — today's Pc discussion depends on remembering that propagated positions carry uncertainty).
4. Have Kessler installed, with the Kelvins dataset downloaded and loading successfully via `kelvins_to_event_dataset`.

If step 4 isn't done, that's the one to prioritize before Day 4 — Phase 2 (Bayesian LSTMs) assumes `EventDataset` objects are already loading cleanly, not just conceptually understood.

---

## Reference links for today

- Space-Track.org, "How the JSpOC Calculates Probability of Collision" — https://www.space-track.org/documents/How_the_JSpOC_Calculates_Probability_of_Collision.pdf
- Kessler documentation, "Basics: loading CDMs" — https://kesslerlib.github.io/kessler/notebooks/basics.html
- ESA Kelvins Collision Avoidance Challenge (data) — https://kelvins.esa.int/collision-avoidance-challenge/data/
- Foster & Estes (1992), "A Parametric Analysis of Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles," NASA/JSC-25898 — reference for Day 7, not needed in full today
- Kessler license question — check the repo's actual `LICENSE` file (GPL-3.0 per GitHub README vs. BSD-3-Clause per conda-forge; see `kessler-report-and-integration-plan.md` in this repo for detail)
