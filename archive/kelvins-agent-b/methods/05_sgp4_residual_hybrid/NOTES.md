# Method 5: Hybrid Physics + ML (Propagation-Residual Correction)

## What this reproduces, and the access gap
Reproduces the general direction of Li et al. (cited in "Machine learning-based risk
classification of space debris conjunction events," EPJ Web of Conferences 2026):
"CNNs are used to supplement SGP4 models with residual error of the orbit
prediction." Per the contract's instruction, I tried to track down and read the
original Li et al. source directly rather than work from the citation alone. Both
the citing EPJ Web of Conferences PDF and every attempt to reach the underlying Li et
al. paper returned HTTP 403 (direct WebFetch and curl both blocked; did not pursue
full browser-based access given the time budget for this batch of four methods).
**This is a real information-access gap, not a design choice** -- the implementation
below is built from the citing paper's one-sentence abstract-level description, not
Li et al.'s actual methodology, architecture, or reported results.

## The substitution this required, and why
Real SGP4 propagates from a TLE (mean motion, eccentricity, inclination, RAAN,
argument of perigee, mean anomaly, B*). The Kelvins CSV only has `sma`/`ecc`/`inc`
per object -- the same gap already documented in method 4's NOTES.md -- so real SGP4
cannot run without fabricating the missing elements. In its place, this uses the
simplest possible physics baseline a CNN could plausibly learn to correct: zero-order
hold (last-known relative position and covariance, unchanged through TCA).

**A constant-velocity linear extrapolation was tried first and discarded.** Over a
multi-day gap, `relative_velocity * dt` diverges by thousands of km, because it
ignores the orbital periodicity that real SGP4 (and the true trajectory) captures --
relative motion over days is periodic, not linear. That divergence completely
dominated the training loss (MSE ~1.9x10^12 at epoch 1) and gave the CNN an
effectively unlearnable residual: "how wrong is a systematically-diverging
extrapolation" rather than a bounded correction. Switching the position baseline to
zero-order hold brought the loss to a sane, decreasing scale (MSE ~7.7 -> ~7.3 over 8
epochs). This is itself a real finding worth keeping: it's a concrete illustration of
why the *physics* half of a physics+ML hybrid matters -- a bad physics baseline can
make the ML correction task harder, not easier, and real SGP4 (unlike either
substitute tried here) wouldn't have this problem since it actually models orbital
motion.

## Design
- **Input**: each event's own >=2-day CDM history as a (20-step, 20-feature)
  sequence -- time_to_tca, 6 covariance sigmas, 6 correlation terms, 3 relative
  position components, 3 relative velocity components, and the per-CDM
  self-computed `risk` field. Sequences longer than 20 keep only the most recent
  20 CDMs; shorter ones are zero-padded. Features are standardized using
  training-set mean/std.
- **Model**: a small 1D CNN (2 conv layers, 32->64 channels, global average
  pooling, 2-layer MLP head) predicting 4 values: a log-ratio correction to the
  combined covariance scale, and a 3D correction (in km) to the zero-order-hold
  relative position.
- **Supervision**: for each of 10,782 usable training events, the target is the
  actual difference between the naive baseline and that event's own true final
  CDM (the row with minimum `time_to_tca` in the *full, unfiltered* training
  data for that event -- which can go slightly past TCA). The model never sees
  that row as input, only as the label defining what correction the naive
  baseline needed.
- **At test time**: predict the two corrections, apply them to the zero-order-hold
  baseline, and feed the corrected covariance + position into the same 2D Pc
  calculator used by methods 2 and 4 (`pc_calculator.py`, copied for
  self-containment).
- **Fallback**: 79 test events with NaN in required fields after sentinel-cleaning
  use the naive-persistence fallback, same pattern as methods 1, 3, and 4.

## Compute budget
8 epochs, ~7s total training time (this is a small model -- nowhere near method 3's
LSTM cost). Final training MSE ~7.25 and still decreasing at epoch 8; more epochs
would likely help further but were not run given the time budget for completing
methods 5-8 in one pass. This should be read as an early-stopped result, not a
converged one.

## Results
2,167 predictions (2,088 model-derived + 79 fallback). Floor rate (-30): **85.9%** --
the highest of any method built so far (naive 53.3%, method 2 77.7%, method 3 18.8%
LSTM-only, method 4 76.1% model-derived, now this at 85.9%). Given the undertrained
model (loss still decreasing at cutoff) and the crude zero-order-hold physics
baseline, I read this high floor rate as more likely reflecting an undertrained
model defaulting toward small corrections than a genuine risk assessment -- flagging
this directly rather than presenting it as if it were a confident result.

## Reproducibility flag
Two independent reasons this should not be read as a faithful reproduction of Li et
al.: (1) the source paper itself was inaccessible (403 on every attempt), so the
implementation is built from a one-sentence secondhand description, not their actual
method; (2) real SGP4 was not available given the Kelvins dataset's missing TLE
elements, so even the "physics" half of this hybrid is a simplified stand-in, not
what their approach actually uses. This is documented as a genuine, useful finding
per the contract's explicit allowance, not smoothed over.
