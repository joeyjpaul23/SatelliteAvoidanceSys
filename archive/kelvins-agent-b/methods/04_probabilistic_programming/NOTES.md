# Method 4: Probabilistic Programming (Pyro) -- Original Model, Not a Reproduction

## What this is and isn't
The contract points at Acciarini et al., "Spacecraft Collision Risk Assessment with
Probabilistic Programming" (arXiv:2012.10260), and kessler's Pyro-based `model.py`, as
the starting point for this method. Having read the paper in full and read
`kessler/model.py` end to end, I concluded this method cannot be a faithful
reproduction of a published risk-prediction result, because no such result exists in
the source material. This is documented in detail below, per the contract's explicit
allowance to flag non-reproducibility as a valid finding. What follows this section is
instead an **original probabilistic model**, built for this benchmark, informed by the
paper's general philosophy but not claiming to reproduce its specific experiment.

## Why the paper's own experiment doesn't map onto the Kelvins task
Read directly from arXiv:2012.10260 (NeurIPS 2020 workshop paper, self-described as
"preliminary work"):

- **Their one experiment**: sample a *synthetic* conjunction event from their own
  generative model (known ground-truth latents, since they generated it), treat its
  first CDM as "observed data," then run importance sampling (70,000 samples) to infer
  posterior distributions over 6 initial orbital elements (mean motion, eccentricity,
  inclination, mean anomaly, argument of perigee, RAAN) for target and chaser, and
  compare the posteriors to the known synthetic ground truth (their Figure 1).
- **They never predict a risk/Pc value.** The inference target is orbital elements,
  not collision risk. There is no `predicted_risk` analog anywhere in what they built
  or reported.
- **They never condition on a real Kelvins test event.** Real Kelvins CDM data was
  used only to calibrate the generative model's *covariance/measurement-noise priors*,
  "by inspection" (Section 3: "checking whether the distributions of covariances
  overlapped") -- a qualitative calibration step, not a per-event prediction pipeline.
- **They explicitly say real-CDM evaluation hadn't been done yet**: "due to the
  unavailability of public data, we are also working with space operators on
  evaluating this technique with real CDMs" (Section 4) -- stated as unstarted future
  work, not a completed part of the paper.
- **Their implementation used PyProb, not Pyro** (Section 4, footnote: integrated with
  "Pyprob, a universal probabilistic programming framework"). kessler's `model.py` is
  a later, separate reimplementation of similar concepts in Pyro by overlapping
  authors -- not the original paper's codebase.

This explains why `kessler/model.py`'s `generate_cdm()` method has `pyro.sample(...,
obs=...)` "likelihood" statements that condition on values computed *within the same
forward pass* rather than on external data (see lines 726-732 of that file) -- the
real-data conditioning the paper describes as future work was apparently never
finished in the shipped library either. `kessler.model.Conjunction` also requires
actual TLEs as input (mean motion, RAAN, argument of perigee, mean anomaly, B*); the
Kelvins CSV only provides `sma`/`ecc`/`inc` per object, missing exactly the elements
needed to construct one, so there is no way to feed real Kelvins events into it
without fabricating the missing orbital elements outright.

Given all this, there is no published number for "Acciarini et al.'s method scores X
on the Kelvins risk-prediction task" to reproduce, regardless of implementation
effort. This was confirmed with the user before proceeding, who chose to have an
original model built instead of stopping at the documentation-only finding.

## The original model actually built
**Physical motivation**: an object's orbit-determination (OD) uncertainty generally
*shrinks* CDM-to-CDM as TCA approaches, since more tracking data accumulates over
time. This is a real, physically grounded trend visible directly in the Kelvins CDM
history (not an arbitrary curve-fit choice) -- confirmed empirically: the
population-level fit below found a *positive* slope of log(sigma) vs. `time_to_tca`
(0.39, see prior below), meaning uncertainty is larger further from TCA and shrinks
as `time_to_tca -> 0`, exactly as expected.

**Per-event Bayesian regression**: For each event, take the combined position
uncertainty scale per CDM (`sigma_scale = sqrt(sum of both objects' 6 RTN sigma^2
terms)`), and fit `log(sigma_scale) ~ intercept + slope * time_to_tca` via Pyro SVI,
using only that event's own CDMs with `time_to_tca >= 2` (same cutoff as every other
method). Because `time_to_tca = 0` at TCA, the posterior intercept *is* the predicted
log(combined sigma) at TCA by construction -- no extra step needed to "extrapolate to
TCA" once the regression is fit.

**Population-calibrated priors (the actual point of doing this Bayesian)**: most test
events have only a handful of CDMs in the >=2-day window (median subset sizes are
small), too few to fit a reliable per-event trend from scratch. The regression's
priors on intercept and slope are calibrated from a population-level fit across all
10,189 usable training events (robust median/MAD, to resist the same kind of
sentinel-value contamination method 2 found in this data):
```
intercept_mean=7.14, intercept_std=1.86, slope_mean=0.39, slope_std=0.23, obs_noise=0.15
```
This gives genuine partial pooling: events with few CDMs shrink toward the population
trend instead of overfitting 1-2 noisy points, while events with a longer CDM history
can pull further from the prior. This is the one substantive methodological echo of
Acciarini et al.'s actual approach -- calibrating a probabilistic model's priors from
real Kelvins data, same as their covariance-noise calibration -- even though the
target being predicted is completely different.

**From posterior to risk**: draw 30 posterior samples of the intercept, convert each
to a sigma-scaling ratio relative to the last known CDM's covariance, scale that CDM's
target/chaser covariance matrices by the ratio (correlations and relative geometry
left at their last-known values, same simplification method 2's paper-following
design uses for what it doesn't predict), linearly extrapolate relative position to
TCA via the last known relative velocity, and run each of the 30 draws through the
*same* 2D Pc calculator (`pc_calculator.py`, copied from method 2 for self-containment
rather than re-derived) used elsewhere in this project. Final `predicted_risk =
log10(mean of the 30 draws' Pc values)` -- i.e., the posterior-predictive mean of Pc
in linear space, logged once at the end, which is the statistically correct way to
summarize a posterior predictive risk (not an average of already-logged values, which
would understate the influence of higher-risk posterior draws).

## Documented design choices / simplifications
- **Single combined uncertainty-growth factor**, applied uniformly to scale both
  objects' full covariance matrices. A more faithful model would fit target and
  chaser uncertainty growth separately (and possibly per-axis, R/T/N growing at
  different rates) -- simplified here to keep the per-event regression a genuinely
  small, fast Bayesian model (2 latent parameters) rather than a much larger one that
  would need more data per event than most events actually have.
- **Fixed (not inferred) observation noise** in the per-event regression, set from the
  population-level residual std. With typically only a handful of data points per
  event, inferring per-event noise as a third latent parameter would be poorly
  identified; using a fixed, data-calibrated value is a standard simplification for
  small-N Bayesian regression.
- **Naive fallback** for the small number of test events with NaN/missing required
  fields after sentinel-cleaning (see run results below for count) -- same fallback
  pattern used in methods 1 and 3, for the same reason: the contract requires one row
  per test `event_id` regardless of any individual method's coverage gaps.

## Results
Full run completed in ~514s (~8.6 min): population prior calibration from 10,189
training events (2.5s), then per-event Pyro SVI regression (300 steps/event, ~0.25s
each) + 30-draw posterior-predictive Pc for 2,088 test events, plus naive fallback for
79 events with missing/sentinel-cleaned required fields.

Final `predictions.csv`: 2,167 rows (2,088 genuine model predictions + 79
naive-persistence fallback).

| | count | floor rate (-30) | mean | std |
|---|---|---|---|---|
| All 2,167 predictions | 2167 | 73.5% | -24.90 | 9.01 |
| Model-derived only | 2088 | 76.1% | -25.35 | 8.80 |
| Naive-fallback only | 79 | 6.3% | (n/a, see method 1) | (n/a) |

This method's floor rate (76.1% model-derived) sits close to method 2's (77.7%) and
far from method 3's (18.8% LSTM-derived) -- adding a third data point to the
cross-method floor-rate spread already flagged in method 3's NOTES.md (53.3% / 77.7%
/ 18.8%, now also 76.1%). Two of four methods built so far cluster near ~75-78%,
while the LSTM sits as a clear outlier. Still not something I can resolve without true
labels, but worth Agent T/A weighing this clustering pattern rather than treating each
method's floor rate as an independent, equally-likely-correct signal.

## Reproducibility flag
This is not a reproducibility flag in the usual sense (no published number was ever
generated for the actual thing this method predicts), so there's nothing here to
compare against and no "gap" to explain. The honest framing: this is an original
Bayesian model built to satisfy this benchmark's requirement for a probabilistic-
programming-based method, informed by Acciarini et al.'s general approach (physics-
motivated generative model, priors calibrated from real Kelvins data, Pyro-based) but
evaluated as a novel contribution on its own terms, not graded against a paper result
that doesn't exist.
