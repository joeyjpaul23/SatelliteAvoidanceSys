# Compact 1-Week Schedule — PIGNN-SAT (Remaining Work: Days 4–14 → 7 Days)

**Status:** Days 1–3 (orbital mechanics, TLEs/SGP4, CDMs + Kessler/Kelvins setup) are done, per `day1-study-guide.md` / `day2-study-guide.md` / `day3-study-guide.md`. Everything below is the remaining content from `2-week-build-plan.md` (original Days 4–14, Phases 2–7), repacked into 7 days. **No task, reading, or checkpoint was cut** — days were only paired up, based on balancing a lighter reading/conceptual block against a heavier hands-on block on the same day.

**Dates:** Monday Aug 10 → Sunday Aug 16, 2026.

**Load:** Paired days below run long — budget ~9–11 focused hours. The three single-topic days (5, 6, 7) stay at the original ~6–8 hours.

---

## New Day 1 — Mon Aug 10: Bayesian LSTMs + Train the Model
*(orig. Day 4 + Day 5)*

**Part A — Understand Bayesian LSTMs (concept, not just API):**
- Key idea: instead of learning fixed weights, the network learns a *distribution* over weights, giving you uncertainty alongside predictions — important when "how confident is this?" matters as much as the prediction.
- Read: Acciarini et al., "Kessler: a Machine Learning Library for Spacecraft Collision Avoidance" (2021).
- Read: Pinto et al., "Towards Automated Satellite Conjunction Management with Bayesian Deep Learning" — arXiv:2012.12450.
- Hands-on: Kessler's "Data Loading" tutorial — load the full Kelvins dataset into `EventDataset` objects: https://kesslerlib.github.io/kessler/notebooks/kelvins_dataset.html

**Part B — Train the model:**
- Work through Kessler's "LSTM training" tutorial end-to-end: https://kesslerlib.github.io/kessler/notebooks/LSTM_training.html — set up `LSTMPredictor`, train on the Kelvins training split, evaluate on the held-out test split.
- Do a real training run (not the 1-epoch demo) — increase epochs, tune `lstm_size`/`lstm_depth`, aim for a stable validation loss curve.
- Save the trained model — wrapped as an agent tool in New Day 3.

**Efficiency tip:** start the real training run first, then read the papers (Part A) while it trains in the background.

**Checkpoint:** trained, saved LSTM model producing uncertainty-aware predictions.

---

## New Day 2 — Tue Aug 11: Benchmark Results + Build the Pc Calculator
*(orig. Day 6 + Day 7)*

**Part A — Benchmark and understand your results:**
- Compare your model's performance against the numbers in the Kessler/Pinto papers and the ESA Kelvins Challenge leaderboard — your "reproduced a real benchmark" story.
- Explore Kessler's probabilistic programming module (Pyro-based) at a conceptual level: https://kesslerlib.github.io/kessler/notebooks/probabilistic_programming_module.html — understand what a generative model of the CDM process buys you (synthetic data, Bayesian inference on a single event). Light primer if needed: https://pyro.ai/examples/

**Part B — Build the Pc calculator:**
- Using the Foster 2D method (read Day 3), implement your own Pc calculator from position/velocity/covariance inputs — this is what lets your system *check* a CDM's claimed risk instead of trusting it.
- References as needed: Foster & Estes (1992), "A Parametric Analysis of Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles" (NASA/JSC-25898); Alfano (2005), "A Numerical Implementation of Spherical Object Collision Probability."
- Confirm your implementation against Kessler's bundled `dsgp4` propagator (check `dsgp4`'s own repo/docs — Kessler's tutorials don't demonstrate it directly).
- Validate against Alfano (2009) "Satellite Conjunction Monte Carlo Analysis" — 12 standard test cases with known answers. Matching numbers = confirmed implementation.

**Checkpoint:** benchmarked model results in hand, plus a standalone, validated Pc calculator independent of any CDM's self-reported risk number.

---

## New Day 3 — Wed Aug 12: Agent Tool Design + Grounding
*(orig. Day 8 + Day 9)*

**Part A — Design the tool interface:**
- Define the tool set: `predict_event_evolution(event)` (wraps `LSTMPredictor`), `calculate_pc(position, velocity, covariance)` (wraps your Day 2 calculator), `propagate_orbit(tle, time)` (wraps SGP4/dsgp4), `get_event_history(object_id)` (past CDMs for context).
- Read: Anthropic tool use docs — https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview (or the equivalent function-calling docs if using another provider).
- Write the tool definitions and a basic tool-execution loop (call model → detect tool-use request → execute function → return result → continue).

**Part B — Ground the agent, prevent hallucination:**
- Core hard design problem of the whole project: write a system prompt that **requires** the agent to call your tools for any factual claim about risk, trajectory, or probability, and explicitly forbids stating numbers it hasn't retrieved from a tool call.
- Adversarial test: ask questions designed to tempt guessing ("roughly how risky is a typical LEO conjunction?") — confirm the agent calls a tool or declines, never fabricates.
- Get the core conversational loop working: "why is this riskier than yesterday's CDM," "what would a 2cm/s burn do to this" (stretch: add a simple maneuver-effect tool for the second one, if time allows).

**Checkpoint:** working chat loop, grounded in your ML/physics layer, that resists guessing.

---

## New Day 4 — Thu Aug 13: Live TLE Ingestion + Conjunction Screening
*(orig. Day 10 + Day 11)*

**Part A — TLE ingestion:**
- Pull live TLE data from CelesTrak (documented, free, by catalog number or satellite group). Set up a Space-Track.org account too — login-gated but more complete catalog coverage, and you'll want it as the authoritative source.
- Build a pipeline: pull TLEs for a tracked object set (e.g., Starlink satellites), propagate via SGP4, screen pairs for close approaches over the next 24–72 hours.

**Part B — Live conjunction screening:**
- New engineering territory — Kessler's tutorials are all static-file based, so this integration is on you.
- Implement a basic screening pass: for your tracked set, find pairs whose propagated positions come within a threshold distance, flag as candidate conjunctions.
- Feed one real, current flagged conjunction through your Pc calculator, trained model, and agent — confirm the full pipeline works end-to-end on live data, not just the Kelvins test set.

**Checkpoint:** system identifies a real, current close approach from live TLE data and reasons about it through the full pipeline.

---

## New Day 5 — Fri Aug 14: Basic Dashboard
*(orig. Day 12, unchanged)*

- Build a minimal interface (Streamlit, or a lightweight Flask/React app — whichever you're faster with) showing: current flagged conjunctions, key details (objects involved, TCA, miss distance, calculated Pc), and a chat panel to talk to the agent about a selected event.
- Don't over-invest in visual polish — function over form.

**Checkpoint:** a bare but functional dashboard surfacing live conjunctions and the agent chat.

---

## New Day 6 — Sat Aug 15: Integration Pass
*(orig. Day 13, unchanged)*

- Wire everything together: live data → screening → physics/ML layer → agent → interface, as one running application.
- Fix the inevitable integration bugs — data format mismatches between phases are the most likely culprit (units/format your screening step outputs vs. what your model/calculator expects).

**Checkpoint:** one application, running end-to-end.

---

## New Day 7 — Sun Aug 16: Polish & Documentation
*(orig. Day 14, unchanged)*

- Write a clear README: what the project does, why (the CDM-fatigue problem), the architecture (four layers), what's novel (the grounded-agent layer), how to run it.
- Prepare a short demo script/walkthrough for interviews: pick one real conjunction event, walk through the whole pipeline on it.
- Be honest about limitations (not production-hardened, screening logic is basic, etc.) — reads as engineering maturity.
- Resolve and document the Kessler license question (GPL-3.0 per repo vs. BSD-3-Clause per conda-forge — check the actual `LICENSE` file; see `kessler-report-and-integration-plan.md`) and state how it affects your own project's license.

**Checkpoint:** portfolio-ready project with README, demo script, and license resolved.

---

## Optional (unscheduled): Graph Neural Network extension

Still not part of the core plan — Kessler uses Bayesian LSTMs, not GNNs. If you want it later: the "all-vs-all conjunction screening" problem (checking every object against every other, not a preselected set) suits GNNs naturally. Starting point: PyTorch Geometric docs/tutorials, applied to reframe New Day 4's screening step as a learned graph problem.

---

## How this was compacted

Original Days 4–14 (11 day-units across Phases 2–7) → 7 days, by pairing one lighter/conceptual day with one heavier/hands-on day where the two were already adjacent in the same phase:

- Day 4 (concept-heavy) + Day 5 (hands-on-heavy) → New Day 1
- Day 6 (concept-heavy, has optional/trim-able depth) + Day 7 (hands-on-heavy) → New Day 2
- Day 8 (design) + Day 9 (testing) → New Day 3 — these were already a tight pair
- Day 10 (ingestion) + Day 11 (screening) → New Day 4 — already a tight pair
- Days 12, 13, 14 kept as single days — each is already a full-day scope (build, integrate, document) that doesn't compress well without cutting content

If a paired day runs long, the natural overflow valve is starting the heavier hands-on half (training run, screening pipeline) first and doing the reading half while it runs in the background — several pairs above have exactly this shape.
