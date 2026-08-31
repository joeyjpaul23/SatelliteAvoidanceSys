# Method 2: Gradient Boosting / Ensemble

## What this reproduces, and a naming discrepancy worth flagging up front
The shared contract labels this method "Gradient boosting / ensemble" and points to
Metz's 2020 TU Darmstadt thesis. I could not access the thesis itself (ResearchGate
blocked the direct fetch), but I did get the companion peer-reviewed conference paper
covering the same work: Metz, Letizia & Simon, "Implementation and Comparison of
Data-Based Methods for Collision Avoidance in Satellite Operations," 8th European
Conference on Space Debris (2021), ESA proceedings PDF.

**That paper does not use gradient boosting.** It compares a Random Forest
(`sklearn.ensemble.RandomForestRegressor`, 100 trees, `min_samples_split=2`,
`min_samples_leaf=1`, unregularized) against an LSTM. There is no GBM anywhere in the
paper. I don't know whether the contract's "gradient boosting" label reflects a
different part of Metz's actual thesis (which is longer than the conference paper and
may cover more methods) or a mix-up with a different paper. Rather than guess, I kept
Metz's actual, verifiable methodology (feature engineering, target design, "naive
completion" of what isn't predicted) faithful to the source, and swapped only the
regressor family to `sklearn.ensemble.GradientBoostingRegressor` to satisfy the
contract's explicit naming. This is a deliberate, documented deviation from the
source paper's own algorithm choice -- everything else about the pipeline follows the
paper as closely as the data allows.

## The key design choice this method reproduces: predicting covariance, not risk
The paper's central idea (and the reason it's methodologically interesting, not just
"GBM on the risk column") is that it does **not** regress risk directly. It predicts
the chaser object's position-uncertainty standard deviations in RTN coordinates
(`c_sigma_r`, `c_sigma_t`, `c_sigma_n`) -- one of the physical inputs that determines
Pc, not Pc itself -- and then computes a probability of collision from those
predictions via a real physics formula. I reproduced this exactly: three
`GradientBoostingRegressor` models (one per sigma component), and a hand-written 2D
Pc calculator (`pc_calculator.py`) rather than any shortcut proxy.

## Feature engineering ("event aggregation setup")
Matches the paper's Section 2.3: each event's CDM history (filtered to
`time_to_tca >= 2.0`, per the contract) is condensed into one row via `last` (value
at the row closest to the cutoff), `min`, `max`, and `mean` of every numeric CDM
field, plus CDM count and observation time span. The paper also engineers
"time since last CDM" / "time since last observation update" and EWMA features from
irregularly-spaced real-world CDM issue timestamps; I did not reproduce those two
specifically, since the Kelvins feature set gives `time_to_tca` (relative to TCA) but
not absolute CDM generation timestamps, so "time since last CDM" isn't directly
computable the way the paper (working from ESA's raw, non-anonymized internal data)
could. `num_cdms` and `obs_span` (max minus min `time_to_tca`) are the closest
available substitutes and are included.

## Target/label construction
For training, the "true final" chaser sigma per event is taken from the row with the
smallest `time_to_tca` in the **unfiltered** training data (i.e. the closest-to-TCA
CDM actually available), which can be inside the 2-day window or even slightly past
TCA (`time_to_tca` as low as -0.15 was observed) -- exactly the information the
contract permits using for training labels but not as model input. Training events
were kept only if (a) at least one CDM exists at `time_to_tca >= 2` to build input
features from, and (b) the label's `time_to_tca` is strictly less than 2, so the
label is never the same row as the most-recent input feature (would otherwise leak /
be a no-op regression). This dropped training from 13,154 to 8,589 usable events;
see below for where the rest went.

## What the model doesn't predict (matches the paper's own "naive completion")
Section 4.1 of the paper is explicit that not everything is predicted: the chaser's
correlation coefficients and its own position (for the relative-geometry input) use
"the last recorded value available at prediction time," and the target object is
"assumed to be known" (well-tracked, so its final covariance ≈ its last available
one). I reproduced this design choice as-is, not as a shortcut of my own:
- Target covariance: last available (`time_to_tca >= 2`) `t_sigma_r/t/n` +
  `t_ct_r`/`t_cn_r`/`t_cn_t`, used directly as the "known" final value. This is
  itself an adaptation forced by the contract -- the paper's authors had access to
  the real (non-anonymized, unrestricted) final target values since target
  covariance isn't the sensitive part of the challenge; the contract's >=2-day
  cutoff doesn't let me use anything closer to TCA, even for the well-tracked
  target, so "last available" is the best legitimate substitute.
- Chaser correlation coefficients and relative position/velocity (encounter
  geometry): last available values, per the paper's own stated method, not
  predicted by the ensemble.
- Only chaser `sigma_r`, `sigma_t`, `sigma_n` are genuinely learned.

## Data-quality issue found and handled: sentinel values, not real outliers
A subset of raw CDM rows contain exact, repeated sentinel values -- 1.5% of rows had
`t_position_covariance_det == 6.732289e46` exactly, and the corresponding
`t_sigma_t == 6.378136e7` (63,780 km, physically absurd for a position uncertainty),
both suspiciously exact and repeated -- almost certainly a degenerate-OD-fit error
code baked into ESA's data, not organic variance. This crashed `GradientBoostingRegressor`
outright (float32 overflow) before I identified it. The source paper's own Section 2.1
describes removing "physically meaningless values" during data prep, which this
matches, so treating it as an error code (converted to NaN, not clipped/winsorized)
is a faithful response, not an invented workaround. Sanitization is applied at the
raw-row level before aggregation, for both train and test, using generous sanity
bounds (sigma > 1e5 m or covariance determinant > 1e15 flagged) chosen to sit far
above legitimate values (95th percentile of `t_sigma_t` is ~1,892 m) and far below the
sentinel. NaNs from this step are median-imputed downstream (median computed on
training data only, applied to both splits) rather than dropped, except where an
event's *label* itself was sentinel-corrupted, in which case the whole training event
is excluded (see above).

## Pc calculator: reused this project's own prior research, not re-derived
`pc_calculator.py` implements the 2D Pc method (rectilinear-motion assumption,
combine both objects' covariances into one Gaussian on the plane perpendicular to
relative velocity, integrate over the touching disk via JSpOC's circumscribing-square
erf approximation) exactly as worked out in this project's own
`docs/probability-of-collision-deep-dive.md` (a companion to the Foster/Chan/JSpOC
method). Reusing it rather than re-deriving from scratch keeps this method's physics
consistent with the rest of the project and with a well-documented, independently
validated reference method (see Validation below).

Two simplifications relative to that document's full method, both documented there as
required steps I did not implement:
1. **No per-object frame rotation before summing covariances.** The full method
   requires rotating each object's covariance from its own RTN frame into a common
   frame before summing. I summed target and chaser covariance directly, since both
   are reported with the same R/T/N field names and the relative position/velocity
   fields are already given in what appears to be a shared frame. For a genuine
   close conjunction the two objects' local RTN frames are nearly aligned, so this is
   a small-angle approximation, not obviously wrong -- but it is unverified against
   the "correct" rotation, since the flat CDM feature columns used here don't expose
   enough orbital-element information to construct that rotation without additional
   processing.
2. **Hard-body radius (HBR) is not in the dataset** -- Kelvins's feature set has no
   object-size field. I used the same default sizes cited in this project's own
   deep-dive doc (5 m payload/platform, 3 m rocket body/unknown, 1 m debris),
   sourced from the JSpOC Pc memo (Kopke, Snow & Hejduk), keyed off `c_object_type`
   (target assumed always payload, 5 m). This is a real, unavoidable source of error
   in the absolute Pc scale, not a modeling choice I'd defend as "correct."

## Validation performed (using only training data -- no test labels touched)
Before trusting the Pc calculator, I fed it the **true** (not model-predicted) final
sigma/correlation/geometry values from 500-1000 randomly sampled training events and
compared its output against ESA's own reported `risk` value for those same events
(which the calculator never sees as input):
- Correlation between computed and actual risk: **0.947**
- Median difference: **0.0** (exactly), mean difference -0.34, MAE 0.86 (in log10
  units, i.e. typically within less than one order of magnitude)
- Floor-rate (fraction of events at exactly -30) using true sigmas: **75.5%**, vs.
  **70.3%** for ESA's own reported risk on the same sample -- close, not exact,
  consistent with the frame/HBR simplifications above.

This gives real confidence the Pc calculator itself is sound, independent of how well
the GBM predicts sigma.

## A concern worth flagging to Agent T/Agent A, not resolved here
On the actual test set, this method's predictions are floored at -30 for **77.7%** of
events. The naive baseline (method 1) floors only **53.3%** of the *same* test set.
Per the Kelvins documentation, the test set is deliberately "hand picked... to
over-represent high risk events" relative to the general population -- so a method
that's actually adding value should, if anything, floor *less* often than a
population-average rate, not more, and certainly not floor meaningfully more than the
naive persistence baseline on the same risk-enriched set. The validation above shows
the *calculator* reproduces a population-level floor rate close to ESA's own
(75.5% vs 70.3%, on a *random*, not risk-enriched, training sample) -- so I don't
believe the discrepancy is a Pc-calculator bug. That leaves the GBM sigma predictions
as the more likely source: they may be systematically over-predicting chaser
uncertainty specifically for the kind of events the test set over-represents (i.e.
underperforming exactly where it matters most for this benchmark). I have not
hand-tuned anything in response to this observation -- doing so without access to
true test labels would risk exactly the kind of guessing the contract prohibits.
Flagging it here for Agent T's calibration diagnostics and Agent A's review, since a
77.7%-vs-53.3% gap on the population the challenge cares most about is a legitimate
red flag about this method's real-world usefulness, not a data or metric artifact.

## Reproducibility flag
The Metz thesis itself (not just the conference paper) may contain a gradient
boosting comparison the conference paper omits -- I was unable to access it directly
(PDF host blocked automated fetch) to confirm one way or the other. If the full thesis
is available by other means, it would be worth checking whether it changes the
"gradient boosting" naming discrepancy noted above.
