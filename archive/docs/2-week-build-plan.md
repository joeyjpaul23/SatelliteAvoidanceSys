# Build Plan: AI Space Debris Collision-Risk Copilot (2-Week Roadmap)

## Before the schedule: a few honest calibrations

**On timeline:** This plan assumes **full-time, focused work (~6-8 hrs/day)**. Given you're starting with zero space-domain knowledge and need to genuinely absorb orbital mechanics, CDM/Pc concepts, and Bayesian deep learning before you can build on them meaningfully — not just skim past them — 14 days is already tight, not padded. I'm not compressing it further, because the whole point you stated is for this to be a real "gateway to learning," not a checkbox exercise. If you're working part-time (a few hours/evening), stretch this to 3-4 weeks at the same daily task granularity.

**On graph neural networks (GNNs):** You don't need them. Kessler — the library this plan builds on — uses Bayesian LSTMs, not GNNs. GNNs show up in the *broader* collision-avoidance research literature (e.g., "all-vs-all conjunction screening" papers that need to reason about relationships between many objects at once), but they're not required for this project's core scope. I've marked one **optional Day 15+ extension** at the end if you want to pick GNNs up later — don't let it block the 2-week core plan.

**On scope discipline:** The plan below gets you to a working end-to-end system: real historical data → trained ML model → physics-grounded agent → live current conjunctions → a usable interface. It does not get you to a polished, production-hardened product — that's correct for a 2-week learning-and-portfolio project, and worth saying explicitly rather than overselling later.

---

## Phase 1: Foundations (Days 1-3)

**Goal:** Understand the domain well enough that everything after this doesn't feel like magic. This is the highest-leverage phase — rushing it will cost you more time later than it saves now.

### Day 1 — Orbital mechanics fundamentals
- Read: NASA JPL's **"Basics of Space Flight"** (free, self-paced, written for engineers learning this for the first time — this is literally what JPL uses to train new operations staff) — https://solarsystem.nasa.gov/basics/index.php — work through Section I chapters on Reference Systems, Gravity & Mechanics, and Trajectories.
- Read: **Braeunig's "Orbital Mechanics"** page — http://www.braeunig.us/space/orbmech.htm — for the actual math (orbital elements, ellipses, the six numbers that define an orbit). More formula-dense than JPL's tutorial; use it as the companion reference once JPL's version gives you the concepts.
- **Concretely, by end of day you should be able to explain:** what the six orbital elements are, what "apogee/perigee" mean, why objects at different altitudes move at different speeds, and what "TCA" (time of closest approach) means in a conjunction context.

### Day 2 — TLEs, SGP4, and how satellites are actually tracked
- Read: **CelesTrak's own explainer pages** on TLEs (celestrak.org) to understand the Two-Line Element format itself — what each field means.
- Skim (don't fully absorb the math yet): Vallado, Crawford, Hujsak & Kelso, **"Revisiting Spacetrack Report #3"** — the canonical modern reference for the SGP4 propagation algorithm — free PDF at https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev1.pdf. You don't need to hand-derive SGP4; you need to understand *what it does* (takes a TLE + a time, outputs a position/velocity) and *why it's approximate* (simplified perturbation model, degrades in accuracy over days/weeks).
- Hands-on: install Python's `sgp4` package, pull one real TLE from CelesTrak, and propagate it forward a few hours yourself. Small script, but this is the moment orbital mechanics stops being abstract.

### Day 3 — CDMs, conjunctions, and probability of collision + environment setup
- Read: **Space-Track.org's "How the JSpOC Calculates Probability of Collision"** — https://www.space-track.org/documents/How_the_JSpOC_Calculates_Probability_of_Collision.pdf — this is the actual operational document explaining the 2D Pc method (Foster's method) used by the U.S. Space Force. It's short, practically written, and is a much better starting point than the dense academic papers (Chan's textbook, Alfano's papers) — save those for reference once you need more depth.
- Read: the CDM field reference (via the Kessler tutorials, which show real CDM fields like `MISS_DISTANCE`, `RELATIVE_SPEED`, `COLLISION_PROBABILITY` — you'll use these directly in Phase 2).
- Hands-on setup: install Python environment, `pip install kessler`, work through Kessler's **"Basics: loading CDMs"** tutorial (https://kesslerlib.github.io/kessler/notebooks/basics.html) and confirm you can load both `.kvn` files and the Kelvins dataset format.
- Download the **ESA Kelvins Collision Avoidance Challenge dataset** — https://kelvins.esa.int/collision-avoidance-challenge/data/ — you'll need this for Phase 2.

**End of Phase 1 checkpoint:** you should be able to explain, in your own words, what a CDM is, why Pc matters, what SGP4 does and doesn't do, and you should have Kessler installed with the Kelvins dataset loading successfully.

---

## Phase 2: ML Layer (Days 4-6)

**Goal:** A trained model that predicts how a conjunction's risk evolves, benchmarked against real published results.

### Day 4 — Understanding Bayesian LSTMs (the ML concept, not just the API call)
- You have intermediate ML background but likely haven't worked with Bayesian neural networks specifically — the key new idea is that instead of learning single fixed weights, the network learns a *distribution* over weights, which lets it output *uncertainty* along with predictions (important for a domain where "how confident is this?" matters as much as the prediction itself).
- Read the original Kessler paper's ML section: Acciarini et al., **"Kessler: a Machine Learning Library for Spacecraft Collision Avoidance"** (2021) and the companion NeurIPS 2020 workshop paper, Pinto et al., **"Towards Automated Satellite Conjunction Management with Bayesian Deep Learning"** — arXiv:2012.12450. These explain exactly what the LSTM is predicting and why the Bayesian framing was chosen.
- Work through Kessler's **"Data Loading"** tutorial to load the full Kelvins dataset into `EventDataset` objects: https://kesslerlib.github.io/kessler/notebooks/kelvins_dataset.html

### Day 5 — Train the model
- Work through Kessler's **"LSTM training"** tutorial end-to-end: https://kesslerlib.github.io/kessler/notebooks/LSTM_training.html — set up `LSTMPredictor`, train on the Kelvins training split, evaluate on the held-out test split.
- Do a real training run (not just the 1-epoch demo in the tutorial) — increase epochs, tune `lstm_size`/`lstm_depth`, and try to get a stable validation loss curve.
- Save the trained model — you'll wrap this as an agent tool in Phase 4.

### Day 6 — Benchmark and understand your results
- Compare your model's predictive performance against the numbers reported in the Kessler and Pinto et al. papers, and against the original **ESA Kelvins Challenge leaderboard/report** — this is your "I reproduced a real benchmark" story.
- Explore Kessler's **probabilistic programming module** (Pyro-based) at a conceptual level: https://kesslerlib.github.io/kessler/notebooks/probabilistic_programming_module.html — you don't need to master Pyro this week, but understanding what a generative model of the CDM process buys you (synthetic data generation, Bayesian inference on a single event) sets up a strong differentiation point if you have spare time later.
- If you want a light primer on Pyro specifically before diving in: https://pyro.ai/examples/ has short, runnable examples.

**End of Phase 2 checkpoint:** a trained, saved LSTM model that predicts conjunction evolution with uncertainty estimates, with results you can compare against a published benchmark.

---

## Phase 3: Physics Layer (Day 7)

**Goal:** A working probability-of-collision calculator and confirmed propagation pipeline, independent of what's baked into any CDM you're given.

### Day 7 — Build the Pc calculator
- Using the Foster 2D method from the Space-Track document you read on Day 3, implement your own Pc calculator from position/velocity/covariance inputs. This is the piece that lets your system *check* a CDM's claimed risk rather than just trusting it.
- Reference for the math details as needed: Foster & Estes (1992), **"A Parametric Analysis of Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles"** (NASA/JSC-25898) — the original derivation; Alfano (2005), **"A Numerical Implementation of Spherical Object Collision Probability"** (Journal of the Astronautical Sciences) — a cleaner numerical treatment if the original derivation is hard to follow.
- Confirm your implementation against Kessler's bundled `dsgp4` propagator (differentiable SGP4) — check `dsgp4`'s own repo/docs for usage examples, since Kessler's tutorials don't demonstrate it directly.
- Validate: run your Pc calculator against a few of Alfano's published test cases (his 2009 paper, "Satellite Conjunction Monte Carlo Analysis," provides 12 standard test cases with known answers) — if your numbers match, you've confirmed your implementation is correct.

**End of Phase 3 checkpoint:** you can independently compute Pc from raw position/covariance data and propagate an orbit forward via SGP4, without depending on a CDM's own stated risk number.

---

## Phase 4: Agent Layer (Days 8-9)

**Goal:** An LLM agent that answers questions about conjunctions by calling your physics/ML layer — never by reasoning about orbital mechanics on its own.

### Day 8 — Design the tool interface
- Define a small set of tools the agent can call: something like `predict_event_evolution(event)` (wraps your trained `LSTMPredictor`), `calculate_pc(position, velocity, covariance)` (wraps your Day 7 calculator), `propagate_orbit(tle, time)` (wraps SGP4/dsgp4), and `get_event_history(object_id)` (looks up past CDMs for context).
- Read: Anthropic's tool use documentation — https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview — covers how to define tools, handle tool calls, and the general pattern for building a tool-using agent. (If you're using a different model provider, the same conceptual pattern — function/tool calling — applies; OpenAI's function-calling docs cover the equivalent.)
- Write the tool definitions and a basic tool-execution loop (call the model, detect a tool-use request, execute your function, return the result, let the model continue).

### Day 9 — Ground the agent, prevent hallucination
- This is the core hard design problem of the whole project: write a system prompt that **requires** the agent to call your tools for any factual claim about risk, trajectory, or probability — and explicitly forbids it from stating numbers it hasn't retrieved from a tool call.
- Test this adversarially: ask questions designed to tempt the model into guessing ("roughly how risky is a typical LEO conjunction?") and confirm it either calls a tool or declines rather than fabricating a plausible-sounding number.
- Get the core conversational loop working: "why is this riskier than yesterday's CDM," "what would a 2cm/s burn do to this" (this second one requires you to add a simple maneuver-effect calculation as another tool — a good stretch task if Day 9 goes faster than expected).

**End of Phase 4 checkpoint:** a working chat loop where the agent answers real questions about a specific conjunction event, grounded in your ML/physics layer.

---

## Phase 5: Live Data Layer (Days 10-11)

**Goal:** Move from static historical data to real, current conjunctions.

### Day 10 — TLE ingestion
- Pull live TLE data from **CelesTrak** (celestrak.org has documented, free APIs/data files for current TLEs by catalog number or satellite group). Space-Track.org also provides TLE and (with registration) CDM data, with more complete catalog coverage but a login-gated API — worth setting up an account here since it's the authoritative source.
- Build a small pipeline: pull TLEs for a set of objects you care about (e.g., Starlink satellites, or any active constellation), propagate them via SGP4, and screen pairs for close approaches over the next 24-72 hours.

### Day 11 — Live conjunction screening
- This is a genuinely new integration step — Kessler's tutorials all work from static files, so you're doing real engineering here, not just following documentation.
- Implement a basic "screening" pass: for your tracked object set, find pairs whose propagated positions come within some threshold distance, and flag them as candidate conjunctions.
- Feed a real, current flagged conjunction through your Phase 3 Pc calculator and Phase 2 trained model, then through your Phase 4 agent — confirm the whole pipeline works end-to-end on live data, not just the Kelvins test set.

**End of Phase 5 checkpoint:** the system can identify a real, current close approach from live TLE data and reason about it through your full pipeline.

---

## Phase 6: Interface (Days 12-13)

**Goal:** Something demoable — a person unfamiliar with the project should be able to look at it and understand what it does in under a minute.

### Day 12 — Basic dashboard
- Build a minimal interface (a simple web app is fine — Streamlit or a lightweight Flask/React app, whichever you're faster with) showing: a list of current flagged conjunctions, key details (objects involved, TCA, miss distance, your calculated Pc), and a chat panel to talk to the agent about a selected event.
- Don't over-invest in visual polish here — function over form for a 2-week timeline.

### Day 13 — Integration pass
- Wire everything together: live data → screening → physics/ML layer → agent → interface, as one running application.
- Fix the inevitable integration bugs (data format mismatches between phases are the most likely culprit — e.g., the units/format your Phase 5 screening outputs may not exactly match what your Phase 2 model or Phase 3 calculator expects).

**End of Phase 6 checkpoint:** one application, running end-to-end, that a stranger could open and understand.

---

## Phase 7: Polish & Documentation (Day 14)

- Write a clear README: what the project does, why (the CDM-fatigue problem), the architecture (the four layers), what's novel about it (the grounded-agent layer — reference back to your novelty research), and how to run it.
- Prepare a short demo script/walkthrough for interviews: pick one real conjunction event and be ready to walk through the whole pipeline on it — this is more convincing than a live open-ended demo.
- Be honest in the README about limitations (not production-hardened, screening logic is basic, etc.) — this reads as engineering maturity, not weakness.
- Resolve and document the Kessler license question (GPL-3.0 per the repo vs. BSD-3-Clause per conda-forge — check the actual LICENSE file) and state clearly how that affects your own project's license.

---

## Optional Day 15+ extension: Graph Neural Networks

Not part of the core 2-week plan, but if you want to extend the project afterward: the "all-vs-all conjunction screening" problem (checking every object against every other object, not just a preselected set) is naturally suited to GNNs, since it's fundamentally a relational problem over a graph of objects. A reasonable starting point once you're ready: PyTorch Geometric's documentation and tutorials (the standard GNN library in the PyTorch ecosystem you're already using via Kessler), applied to reframe your Phase 5 screening step as a learned graph problem instead of brute-force pairwise propagation.

---

## Full source list (for reference)

- NASA JPL, *Basics of Space Flight* — https://solarsystem.nasa.gov/basics/index.php
- Braeunig, *Orbital Mechanics* — http://www.braeunig.us/space/orbmech.htm
- Vallado, Crawford, Hujsak & Kelso, "Revisiting Spacetrack Report #3" — https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev1.pdf
- Space-Track.org, "How the JSpOC Calculates Probability of Collision" — https://www.space-track.org/documents/How_the_JSpOC_Calculates_Probability_of_Collision.pdf
- Foster & Estes (1992), "A Parametric Analysis of Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles," NASA/JSC-25898
- Alfano (2005), "A Numerical Implementation of Spherical Object Collision Probability," Journal of the Astronautical Sciences
- Alfano (2009), "Satellite Conjunction Monte Carlo Analysis" (test cases for validating your Pc calculator)
- Acciarini et al. (2021), "Kessler: a Machine Learning Library for Spacecraft Collision Avoidance," 8th European Conference on Space Debris
- Pinto et al. (2020), "Towards Automated Satellite Conjunction Management with Bayesian Deep Learning," arXiv:2012.12450
- Kessler documentation — https://kesslerlib.github.io/kessler/
- ESA Kelvins Collision Avoidance Challenge — https://kelvins.esa.int/collision-avoidance-challenge/
- Pyro examples — https://pyro.ai/examples/
- CelesTrak (TLE data and format reference) — https://celestrak.org
- Anthropic tool use documentation — https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- (Optional extension) PyTorch Geometric documentation
