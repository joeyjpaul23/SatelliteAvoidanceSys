# Kessler Library: In-Depth Report + Integration Plan

## Part 1: What Kessler Is

**Origin:** Built by the FDL (Frontier Development Lab) Europe Constellations team in 2020 — a public-private partnership between ESA, Trillium Technologies, and the University of Oxford, done in collaboration with ESA's own operations center (ESOC). Main developer: Giacomo Acciarini. It's a genuine research output, not a hobby project — it has three associated peer-reviewed papers (8th European Conference on Space Debris 2021; two NeurIPS 2020 workshop papers on Bayesian deep learning and probabilistic programming for collision risk).

**Maturity/community signal (be realistic about this):** 66 GitHub stars, 19 forks, one repository under the `kesslerlib` org. This is a small, academic-adjacent project — not a widely-adopted industry tool. Documentation is five tutorial notebooks plus an API reference; there's no large community, Discord, or Stack Overflow presence to lean on. Expect to read source code directly when the tutorials don't cover what you need.

**License — verify before building on it:** The GitHub README states GPL-3.0 ("Kessler is distributed under the GNU General Public License version 3. Get in touch with the authors for other licensing options.") but the conda-forge package listing separately shows BSD-3-Clause. This is a real discrepancy — **check the actual LICENSE file in the repo before deciding how you'll license your own project**, since GPL-3.0 has copyleft implications (derivative works generally must also be GPL-3.0) that matter if you ever want your project under a more permissive license.

**Install options:**
```
conda install conda-forge::kessler
# or
mamba install kessler
# or
pip install kessler
# or, for local dev:
git clone https://github.com/kesslerlib/kessler.git
cd kessler
pip install -e .
```

**Dependencies (from setup.py):** `pyro-ppl`, `numpy`, `matplotlib`, `torch>=1.5.1`, `dsgp4`, `skyfield>=1.26`, `pandas`. Two of these matter a lot for your project:
- **`dsgp4`** — a differentiable SGP4 orbit propagator (also from the same research group). This is actually more capable than a plain SGP4 implementation since it supports gradient-based methods, though the Kessler tutorials don't walk through using it directly — you'd need to go to `dsgp4`'s own docs/source.
- **`skyfield`** — a well-established, well-documented astronomy library for computing satellite positions and handling time/coordinate systems.

## Part 2: What It Actually Does (Capabilities)

Kessler has four documented capability areas:

1. **CDM import/export and event grouping.** Load Conjunction Data Messages from the standard `.kvn` (CCSDS) format, or from a pandas DataFrame, or directly from the ESA Kelvins Collision Avoidance Challenge CSV format via a built-in converter (`kessler.data.kelvins_to_event_dataset`). CDMs are automatically grouped into `Event` objects (a conjunction event = the sequence of CDMs issued about one specific close approach over time). Example from the docs: loading the Kelvins test set converts ~24,000 raw CDM rows into ~1,700 grouped events after cleaning.

2. **Plotting.** Visualize how an event's features (e.g., `RELATIVE_SPEED`, `MISS_DISTANCE`) evolve across the sequence of real CDMs, or overlay model-predicted evolution against ground truth.

3. **The ML module — this is the strongest, best-documented piece.** A stack of **Bayesian LSTM neural networks** (`kessler.nn.LSTMPredictor`) trainable on your own CDM collection. Key config: `lstm_size`, `lstm_depth`, `dropout`, and a `features` list (you can pull all available numeric fields via `events.common_features(only_numeric=True)` — this can include things like miss distance, relative speed, and collision probability). Training via `.learn()` takes epochs, learning rate, batch size, device (CPU/CUDA), and validation split. Inference via `.predict_event()` gives you a **distribution of predicted future CDM values with uncertainty**, not a single point estimate — genuinely useful for a "how confident is this prediction" story.

4. **A probabilistic programming module (built on Pyro).** A generative model that simulates the CDM-generation process itself. Useful two ways: (a) Bayesian inference on a real observed event, and (b) generating synthetic CDM sequences/datasets sampled from the model. This is the least-used, least-documented part of Kessler — which makes it a good place to differentiate your project if you want to go deeper than most people would bother to.

## Part 3: What Kessler Does NOT Do (the gaps you'd need to fill)

- **No live tracking data ingestion.** Everything in the tutorials works from static files (`.kvn` files or the Kelvins CSV) — there's no built-in connector to CelesTrak or Space-Track for live TLEs or live CDMs.
- **No standalone documented Pc (probability-of-collision) calculator API.** The LSTM predicts future CDM *feature values* (which can include the collision probability field if you include it in your feature list), but I found no clean, separately-documented function that computes Pc from covariance data the way a dedicated physics library would. Worth checking the source directly (`kessler.model`, `kessler.observation_model`) rather than assuming it isn't there.
- **No agent/LLM/conversational layer of any kind.** It's a pure Python ML library — everything is code you call from a script or notebook. This is exactly the gap your project fills.
- **No dashboard or UI.** Output is Jupyter plots and saved model files.

## Part 4: How to Use It for Your Project

Mapping Kessler onto the four-layer architecture from the earlier brief:

**Layer 2 (ML) — Kessler gets you most of the way there immediately.**
1. `pip install kessler`, work through the "Basics: loading CDMs" and "Data Loading" (Kelvins) tutorials to get comfortable with `EventDataset`/`Event` objects.
2. Load the *full* Kelvins Collision Avoidance Challenge dataset via `kelvins_to_event_dataset` and train an `LSTMPredictor` on it. Try to reproduce something close to the benchmark numbers from the original Kessler/Pinto papers — this gives you a legitimate, citable "I trained a real model on a real benchmark and got comparable results" story for interviews, on day one of the ML layer.
3. Once the basic LSTM works, explore the Pyro-based probabilistic programming module — using it for Bayesian inference on a specific event, or to generate synthetic conjunction scenarios for stress-testing. This part is used by almost nobody outside the original authors, so going deep here is genuine differentiation.

**Layer 1 (physics) — partially covered, needs your own work.**
- `dsgp4` (bundled dependency) gives you differentiable SGP4 propagation, but you'll need to read its docs separately since Kessler's tutorials don't demonstrate it.
- You'll likely need to write your own Pc calculation from covariance data (standard formulas exist in the CDM/conjunction-assessment literature — this is a well-documented piece of orbital mechanics, just not one Kessler hands you pre-built).

**Layer 3 (agent) — entirely yours to build; Kessler offers nothing here.**
- Wrap `LSTMPredictor.predict_event()` and your Pc/physics functions as callable tools.
- Give an LLM tool-calling access to them (function calling via the Anthropic or OpenAI API, or MCP if you want to reuse the pattern you've already been exploring for dev tooling).
- The system prompt needs to explicitly require the model to call these tools for any factual claim about risk or trajectory, rather than reasoning about orbital mechanics from its own training — this is the core hard design problem of the whole project.

**Layer 4 (live data/interface) — entirely yours to build.**
- Write your own CelesTrak/Space-Track ingestion for current TLEs and, if accessible, current CDMs.
- Kessler's tutorials all use static datasets, so connecting it to a live feed is genuinely new integration work, not something to expect help with from the library.

## Suggested Build Order

1. Install Kessler, run the existing tutorials end-to-end, confirm you understand `EventDataset`/`Event`/`LSTMPredictor`.
2. Train `LSTMPredictor` on the full Kelvins dataset; benchmark against the published results.
3. Resolve the license question (check the actual LICENSE file in the repo) before deciding your own project's license.
4. Build (or find) a Pc calculator and confirm/extend orbit propagation via `dsgp4`.
5. Wrap the trained model + physics functions as agent tools; build the grounded LLM layer.
6. Add live TLE ingestion and a minimal dashboard last, once the core pipeline is proven on static data.
