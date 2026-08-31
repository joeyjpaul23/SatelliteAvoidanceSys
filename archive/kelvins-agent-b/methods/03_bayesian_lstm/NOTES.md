# Method 3: Bayesian LSTM

## What this reproduces
Pinto et al., "Towards Automated Satellite Conjunction Management with Bayesian Deep
Learning," arXiv:2012.12450, using `kessler.nn.LSTMPredictor`
(github.com/kesslerlib/kessler) as the contract specifies, rather than
re-deriving the architecture from the paper.

## License: resolved, GPL-3.0 confirmed first-hand
The contract flagged a licensing discrepancy (GPL-3.0 per kessler's GitHub README vs.
BSD-3-Clause per its conda-forge listing) and asked that the actual LICENSE file be
checked before relying on it. I did: the PyPI package's own `dist-info/LICENSE.md` is
the complete, unmodified GPL-3.0 text (verified by reading it in full, not just
grepping a header line) -- while that same package's `METADATA` file simultaneously
declares `License: BSD` and an OSI BSD classifier. This is a genuine, first-hand-
confirmed internal contradiction in kessler's own packaging, not a stale rumor. Per
explicit user decision, this project proceeds using kessler and accepting GPL-3.0
implications; this is noted here for the record since it affects how any code in this
method's directory (not the prediction CSV itself) could be redistributed.

## kessler bugs and gaps found while building this (real findings, not workarounds I invented casually)
Building this method surfaced three concrete problems with kessler 1.0.1 that aren't
mentioned in its own tutorials:

1. **Progress bar crashes on import-clean use.** `kessler.util.progress_bar_update`
   calls an undefined name, `days_hours_mins_secs_str`, which does not exist anywhere
   in `kessler/util.py` (confirmed by grepping the installed source). Every call to
   `kelvins_to_event_dataset()` -- literally the library's own documented entry point
   for this exact dataset -- crashes with a `NameError` partway through, before
   producing any output. Worked around by monkey-patching
   `progress_bar_update`/`progress_bar_init`/`progress_bar_end` to no-ops before
   calling into kessler; this only silences a print, it does not touch any data or
   model logic.
2. **Default `LSTMPredictor` feature list is incompatible with kessler's own Kelvins
   loader.** The default `features` argument includes `OBJECT1_X`, `OBJECT1_Y`,
   `OBJECT1_Z`, `OBJECT1_X_DOT`, etc. (Cartesian state for both objects), but
   `kelvins_to_event_dataset()` never populates those fields on the CDMs it
   constructs -- it only sets relative-geometry fields (`RELATIVE_POSITION_R/T/N`,
   `RELATIVE_VELOCITY_R/T/N`, `MISS_DISTANCE`, `RELATIVE_SPEED`), raw covariance
   terms (`OBJECT{1,2}_C*`), and OD-quality fields. Calling `LSTMPredictor()` with its
   own defaults against kessler's own Kelvins-loaded data raises
   `RuntimeError: Feature OBJECT1_X is not present in the dataset` immediately. This
   means the library's documented "load Kelvins data, train an LSTM" path does not
   work together out of the box with default arguments -- a genuine reproducibility
   gap, not user error. Worked around by building a custom `features` list (see
   `predict.py`) restricted to fields the loader actually sets: the same relative
   geometry, the full raw covariance for both objects (21 terms each, all of
   `CR_R`/`CT_R`/`CT_T`/`CN_R`/`CN_T`/`CN_N` plus their rate-of-change counterparts),
   and the OD-quality fields, in place of the unavailable Cartesian state -- 66
   features total.
3. **`kelvins_to_event_dataset()` output loses `event_id`.** The returned
   `EventDataset`'s `Event`/`CDM` objects retain no field identifying which original
   Kelvins `event_id` they came from -- it's used only internally as a pandas
   `groupby` key during construction and then discarded. Since the contract's output
   format requires `event_id,predicted_risk`, this had to be recovered independently:
   `predict.py`'s `surviving_event_ids()` re-implements kessler's own row-filtering
   (column drop, `dropna()`, the six hardcoded outlier bounds) in a standalone
   function, copied from reading kessler's actual `data.py` source rather than
   guessed, and relies on pandas' `groupby` producing the same sort-ascending key
   order kessler's own iteration uses. Verified with an assertion
   (`len(recovered_ids) == len(events)`) before trusting the alignment, for both the
   train and test splits.

## Data loss from kessler's own cleaning -- and how the contract's "every event_id gets a row" requirement was preserved
`kelvins_to_event_dataset()`'s default `remove_outliers=True` applies six hardcoded
per-object sigma thresholds (e.g. `t_sigma_r <= 20`, `c_sigma_t <= 100000`) plus an
unconditional `dropna()` across *all* raw Kelvins columns (not just the ones this
method uses -- e.g. a missing `SSN`/`F10`/`AP` space-weather value on an otherwise
fine CDM row is enough to drop that row). On the test set this drops
**441 of 2167 events entirely (1726 survive, ~20% loss)**; on the training set,
**3568 of 13154 (9586 survive)**. I used kessler's own defaults as-is rather than
loosening them, since they're presumably calibrated by the library's authors for
their own results, and changing them would be a bigger, less-defensible deviation
than accepting the data loss.

Since the contract requires exactly one row per test `event_id` regardless, the 441
events kessler's own cleaning drops are filled in with **method 1's naive persistence
baseline** (last available risk value), computed directly and independently from the
raw `test_data.csv`, not from anything kessler touched. This is disclosed, not hidden
-- Agent T/Agent A should know that ~20% of this method's "predictions" are actually
the naive baseline in disguise. See `predictions.csv` combined with the counts here
for exactly how many of each.

## The "Bayesian" part: MC-Dropout, not a modification I added
`LSTMPredictor.predict()` calls `self.train()` (not `.eval()`) before its forward
pass -- i.e. dropout stays active at inference time. This is Monte Carlo Dropout, a
standard cheap approximate-Bayesian technique, and it's kessler's own design choice,
not something added here. `predict_event(event, num_samples=K)` autoregressively
rolls a single event forward K times, each with independent dropout masks, giving K
different trajectories to (predicted) TCA. I average `log10(Pc)` across the K
rollouts per event (a geometric-mean-style average in probability space, which is
more robust than averaging raw Pc directly given how many orders of magnitude Pc can
span and how often it saturates at the -30 floor).

## Compute budget: initially scaled down, then expanded to full scale by explicit request
Benchmarked `predict_event()` at **~0.65s per MC-Dropout sample per event** (each
single-step `.predict()` call reprocesses the *entire* growing sequence through the
LSTM from scratch each time -- `reset()` zeroes hidden state on every call, so there's
no state reuse across autoregressive steps; this is a property of kessler's own
`predict()` implementation, not something changed here). At `num_samples=20` across
all 1726 surviving test events, that's 6+ hours of prediction alone.

The run actually used for the delivered `predictions.csv` is the **full paper-scale
configuration**, run at the user's explicit request accepting that runtime:
- `MC_SAMPLES = 20`
- `MAX_ROLLOUT_LENGTH = 22` (kessler's own `predict_event()` default, unmodified)
- `EPOCHS = 6` for training (no epoch count is specified in the paper or kessler's
  own tutorials; chosen as a reasonable full-convergence-attempt budget)

An earlier, compute-reduced trial run (`MC_SAMPLES=5`, `MAX_ROLLOUT_LENGTH=15`,
`EPOCHS=4`) was started first to confirm the pipeline worked end-to-end before
committing to the multi-hour run; it was stopped before completion once the full-scale
run was requested, and none of its (partial, never-completed) output was used.

## A concern surfaced during benchmarking, not yet resolved
On a lightly-trained model (1 epoch, 300-event subset), every test rollout hit the
max-length cap (22 steps, kessler's default) without the autoregressively-*predicted*
`__TCA` feature ever being overtaken by the predicted `__CREATION_DATE` -- i.e. the
model's own running estimate of "how many days until TCA" wasn't converging toward
zero as the rollout progressed. This could mean: (a) more training resolves it, (b)
the `__TCA` feature (itself autoregressively re-predicted at every step, not fixed)
is intrinsically unstable in this setup, or (c) the normalization/feature-scaling
interacts badly with a feature that should be *shrinking* over the rollout. Whichever
it is, it means many rollouts in the actual run below are truncated by
`MAX_ROLLOUT_LENGTH` rather than reaching a natural stopping point -- worth Agent T
and Agent A treating this method's risk estimates as coming from "best available
prediction after N forced steps," not "prediction genuinely converged to TCA," until
someone has time to dig into the `__TCA` divergence question directly.

## Results
Full run completed: training took 1,376s (~23 min) for 6 epochs, final normalized
MSE loss ~0.15 (train) / 0.196 (validation, held out 15% of training events). The
prediction phase (1,726 events x 20 MC-Dropout samples, autoregressive rollout per
sample) took ~10,862s (~3.0 hours) -- faster than the ~6-hour worst-case estimate
from benchmarking, likely because per-event cost didn't grow as much as feared once
the model was actually trained (better-converged rollouts may reach the stopping
condition sooner than the lightly-trained benchmark model did).

Final `predictions.csv`: 2,167 rows (1,726 genuine LSTM/MC-Dropout predictions + 441
naive-persistence fallbacks for events kessler's own cleaning dropped, see above).

| | count | floor rate (-30) | mean | std |
|---|---|---|---|---|
| All 2,167 predictions | 2167 | 23.4% | -25.20 | 6.27 |
| LSTM-derived only | 1726 | 18.8% | -26.72 | 3.48 |
| Naive-fallback only | 441 | 41.0% | (n/a, see method 1) | (n/a) |

Worth flagging alongside method 2's own flagged concern: this method's floor rate
(18.8% on the genuinely LSTM-derived subset, 23.4% overall) is *much lower* than
both the naive baseline (53.3% on the full test set) and method 2's GBM-covariance
approach (77.7%). Since the test set is deliberately risk-enriched, a lower floor
rate isn't automatically "better" -- it could equally mean this method is
systematically over-predicting risk (biased toward higher Pc) rather than genuinely
discriminating high-risk events better. I have not attempted to determine which,
since doing so would require exactly the true-label comparison this method's builder
isn't allowed to make. Flagging the three methods' very different floor rates
(53.3% / 77.7% / 18.8%) as a cluster worth Agent T's calibration diagnostics and
Agent A's cross-method review, rather than trusting any one of them in isolation.

## Reproducibility flag
Between the two library bugs above and the multi-hour compute cost even after
scaling to what fits reasonably in a working session, I don't believe this
reproduction should be read as "Pinto et al.'s method, faithfully reproduced
end-to-end and fully converged." It's a genuine best-effort built on top of their
actual released code, run at full paper-scale sampling (20 MC-Dropout samples,
kessler's own default rollout length) at the user's explicit request, with every
deviation from what an unconstrained reproduction would look like documented above --
most notably the ~20% of test events kessler's own data cleaning drops outright, the
unresolved `__TCA`-convergence question noted above, and only 6 training epochs
against no stated reference epoch count in the source material.
