# Day 4–5 Study Guide — Bayesian LSTMs + Training the Model

*Phase 2 of the build plan (PIGNN-SAT / AI Space Debris Collision-Risk Copilot). Corresponds to "New Day 1" (Mon Aug 10, 2026) in `compact-1-week-schedule.md`. Assumes Days 1–3 (orbital mechanics, TLEs/SGP4, CDMs + Kessler/Kelvins setup) are done.*

---

## Why this day matters

Days 1–3 gave you the physics and the data format. Today you build the actual ML layer: a model that looks at the CDMs issued so far for a conjunction event and predicts how the risk (miss distance, relative speed, Pc, ...) will evolve as TCA approaches — with an honest uncertainty estimate attached, not just a point guess. This is the piece your Phase 4 agent will call as a tool, and the piece you'll benchmark against a published paper.

Everything below was verified directly against the installed `kessler==1.0.1` source (not just the tutorial docs), so the API signatures are exact, not paraphrased.

---

## Part A: Understanding Bayesian LSTMs (concept first)

### The problem with a normal LSTM here

A standard LSTM trained to predict "what will the next CDM's fields look like" gives you one number per field. For a domain where the whole point is *how much should I trust this number*, a single point estimate throws away exactly the information you need — two predictions of "Pc = 1e-5" are very different if one comes from a model that's seen 50 similar events and the other from one extrapolating wildly outside its training distribution.

### The trick Kessler actually uses: MC Dropout

"Bayesian LSTM" sounds like it means the network learns a full probability distribution over every weight (variational Bayesian neural networks do this, and it's expensive). Kessler uses a cheaper, well-established approximation instead: **Monte Carlo Dropout**.

- Dropout is normally a *training-only* regularizer — you randomly zero out neurons during training, then turn it off at inference for a clean deterministic output.
- MC Dropout's trick: **leave dropout switched on at inference time**, and run the same input through the network many times. Each pass randomly drops a different set of neurons, so each pass is like sampling from a slightly different sub-network. The spread across those samples approximates a distribution over predictions.

I confirmed this is exactly what `kessler.nn.LSTMPredictor.predict_event()` does — its source (`num_samples` parameter) runs the *same* event forward `num_samples` times, each time sampling a full stochastic trajectory, and returns them as an `EventDataset` of alternate futures. The mean and spread across those samples is your prediction and its uncertainty. No separate "uncertainty head" or Bayesian weight machinery — it's dropout, kept on, sampled repeatedly.

### Epistemic vs. aleatoric — which one does this give you?

- **Epistemic uncertainty** = "the model doesn't know" (sparse/unfamiliar training data). MC Dropout is fundamentally an epistemic-uncertainty tool — the spread grows in regions the network hasn't seen much of.
- **Aleatoric uncertainty** = irreducible noise in the data itself (e.g. real sensor/tracking noise in the CDMs). MC Dropout doesn't directly model this; it's a separate concept some architectures add via a learned noise output.

**Self-check:** if you fed the model an event that looks nothing like anything in the Kelvins training set, would you expect the spread across `num_samples` predictions to be wide or narrow? (Wide — that's epistemic uncertainty doing its job.)

---

## Hands-on warm-up: MC Dropout in 15 lines (runnable now, no dataset needed)

This is a minimal stand-in for what `LSTMPredictor` does internally — small enough to run before you touch the real dataset, so the concept isn't abstract when you get to it. Saved as `notebooks/bayesian_lstm_demo.py`.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class TinyBayesianLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(self.drop(h[:, -1]))

model = TinyBayesianLSTM()
model.train()  # keep dropout ACTIVE even during "inference" — this is the whole trick

x = torch.randn(1, 5, 4)  # one fake 5-timestep, 4-feature sequence
samples = torch.stack([model(x) for _ in range(200)])  # 200 stochastic forward passes

print("Mean prediction:", samples.mean(dim=0).squeeze().tolist())
print("Std (uncertainty):", samples.std(dim=0).squeeze().tolist())
```

I ran this in a sandbox to confirm it works. Output (untrained weights, so the numbers themselves are meaningless — it's the *shape* of the result that matters):

```
Mean prediction: [-0.0205, 0.2481, 0.0581, -0.1789]
Std (uncertainty): [0.0271, 0.0319, 0.0369, 0.0367]
```

One mean vector and one std vector, per feature — that std vector *is* your uncertainty estimate. Scale this up (real features, real trained weights, `num_samples` rollouts per event) and you have `LSTMPredictor.predict_event()`.

---

## Part B: Train the real model

### Setup

```bash
pip install kessler --break-system-packages
```

Note: this pulls in `torch`, `pyro-ppl`, and `dsgp4` as dependencies — it's a sizeable install (~500MB+ with torch). I hit a "no space left on device" error on first try in a constrained sandbox; if you see that, clear `pip cache purge` first or free disk space, then retry — it's a disk issue, not a package problem.

I confirmed `kessler==1.0.1` from PyPI installs cleanly and exposes `kessler.nn.LSTMPredictor` and `kessler.data.kelvins_to_event_dataset` (neither is re-exported at the top-level `kessler` namespace — import from the submodule directly, e.g. `from kessler.nn import LSTMPredictor`, `from kessler.data import kelvins_to_event_dataset`). This matches what `kessler-report-and-integration-plan.md` describes, so the PyPI package is the right one for this step regardless of how the GPL/BSD license question in that doc eventually resolves.

### Load the Kelvins dataset

```python
from kessler.data import kelvins_to_event_dataset

event_set = kelvins_to_event_dataset("path/to/kelvins_train.csv")
print(len(event_set))  # expect ~1,700 grouped events from ~24,000 raw CDM rows
```

You still need to download the actual CSV from https://kelvins.esa.int/collision-avoidance-challenge/data/ yourself (per Day 3) — it's not bundled with the package, and I couldn't find a programmatic download path in the source, so this is a manual step.

### Build the predictor

Verified constructor signature:

```python
LSTMPredictor(lstm_size=256, lstm_depth=2, dropout=0.2, features=None)
```

If you don't pass `features`, it defaults to a fixed list of 62 CDM fields — everything from `MISS_DISTANCE` and `RELATIVE_SPEED` through the full 6x6 covariance components for both objects (`OBJECT1_CR_R`, `OBJECT1_CRDOT_T`, etc.) — the same fields Day 3 introduced. You can also pass a custom subset if you want a smaller model to start with.

```python
from kessler.nn import LSTMPredictor

model = LSTMPredictor(lstm_size=64, lstm_depth=2, dropout=0.2)  # smaller than default for a first run
```

### Train

Verified `.learn()` signature:

```python
model.learn(
    event_set,
    epochs=2,              # tutorial default — bump this for a real run
    lr=1e-3,
    batch_size=8,
    device='cpu',           # or 'cuda' if available
    valid_proportion=0.15,
    num_workers=4,
    event_samples_for_stats=250,
    file_name_prefix=None,  # set this to auto-save checkpoints per epoch
)
```

`.learn()` internally computes feature normalization stats from a sample of events, splits train/valid, and reports train/valid loss per epoch (`model.plot_loss()` after training gives you the curve). Do a real run here — not the 1-epoch tutorial demo — and watch for a validation loss that stabilizes rather than diverges. Save the model (`model.save(path)`) once you're happy with it; you'll load it back in Phase 4 as an agent tool.

### Predict — this is where "Bayesian" pays off

```python
predicted = model.predict_event(some_event, num_samples=50)
# num_samples=1  -> single Event (one stochastic rollout)
# num_samples>1  -> EventDataset of 50 alternate future trajectories for the same starting event
```

I traced this through the source: each of the 50 samples is a full autoregressive rollout (predict next CDM, feed it back in, repeat) with dropout active throughout, stopping when `TCA` is reached or a max length is hit. To get a usable "risk + confidence" number, you'd aggregate across the 50 rollouts yourself (e.g. mean and std of the predicted `COLLISION_PROBABILITY` at the final step) — Kessler hands you the raw samples, not a pre-aggregated summary.

**Self-check:** if you ran `predict_event(event, num_samples=50)` twice in a row on the same event, would you get identical results both times? (No — dropout is stochastic each call, which is exactly the point.)

---

## End-of-day checkpoint

You should be able to:

1. Explain why MC Dropout gives you an uncertainty estimate without a separate "Bayesian" training procedure — and that the mechanism is dropout left on at inference, sampled repeatedly.
2. Distinguish epistemic uncertainty (what MC Dropout captures) from aleatoric uncertainty (what it doesn't).
3. Have a trained, saved `LSTMPredictor` on the real Kelvins training split, with a validation loss curve that's stabilized rather than diverged.
4. Call `predict_event(event, num_samples=N)` on a held-out event and describe, in your own words, what the spread across the N samples is telling you.

If you don't have the Kelvins CSV downloaded yet, that's the blocker to clear first — everything else in this guide works once it's loaded.

---

## Reference links

- Acciarini et al. (2021), "Kessler: a Machine Learning Library for Spacecraft Collision Avoidance"
- Pinto et al. (2020), "Towards Automated Satellite Conjunction Management with Bayesian Deep Learning" — arXiv:2012.12450
- Kessler docs, "Data Loading" (Kelvins) — https://kesslerlib.github.io/kessler/notebooks/kelvins_dataset.html
- Kessler docs, "LSTM training" — https://kesslerlib.github.io/kessler/notebooks/LSTM_training.html
- ESA Kelvins Collision Avoidance Challenge (data) — https://kelvins.esa.int/collision-avoidance-challenge/data/
- `kessler-report-and-integration-plan.md` in this repo — background on the library, its gaps, and the license question
- Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation" — the original MC Dropout paper, useful if you want the theory behind what Kessler is doing
