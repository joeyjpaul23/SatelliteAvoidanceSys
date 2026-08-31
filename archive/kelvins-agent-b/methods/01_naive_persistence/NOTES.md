# Method 1: Naive Persistence Baseline

## What this reproduces
The mandatory baseline defined in the shared contract, not a published method:
predict the final risk (log10 Pc) at TCA as the risk value from the most
recent CDM available before the 2-day-prior-to-TCA cutoff. This is the
standard "do nothing clever" sanity-check baseline used in the original
Kelvins challenge writeup (Uriot et al., 2022) and is required reading before
trusting any more sophisticated method's score.

## Implementation
- Input: `test_data.csv` only (no training needed).
- The Kelvins dataset already excludes CDMs within 2 days of TCA for both
  train and test (confirmed empirically: `time_to_tca.min()` ~= 2.0003 days
  in the test set) -- so no additional filtering was needed to respect the
  contract's cutoff rule. The predict.py script asserts this invariant rather
  than assuming it, in case a future data pull differs.
- For each `event_id`, select the row with minimum `time_to_tca` (the CDM
  closest to, but still >=2 days before, TCA) and use its `risk` field
  directly as `predicted_risk`.

## Design choices / ambiguity
- None -- this method is fully specified by the contract, no judgment calls
  needed.

## Known limitations (expected, not a bug)
- This baseline cannot react to risk trends within an event's CDM history
  (e.g. rapidly increasing miss-probability over successive CDMs) -- it's
  intentionally naive. Its purpose is solely to give every other method a
  floor to beat.

## Data observation worth flagging to Agent T
`risk` in the raw Kelvins data hits an exact floor of -30.0 for ~28% of all
CDM rows (6,884 / 24,484 in the full test_data.csv) -- this is a genuine
sentinel in the source dataset (used when the self-computed collision risk
is degenerate/near-zero), not an artifact of this pipeline. `max_risk_estimate`
stays populated with a real (non-floored) value on those same rows, confirming
`risk` and `max_risk_estimate` are computed differently and only the former
floors out. Downstream: 1,083 of 2,167 (50%) of this baseline's predictions
are exactly -30.0. Any calibration diagnostic Agent T builds should be aware
this is a real, repeated value in ground-truth-adjacent data, not a constant-
output bug -- though if a *learned* method (2-8) produces suspiciously many
exact -30.0 outputs, that would be worth scrutinizing since a model has no
reason to reproduce this exact floor unless it's directly copying an input
feature.

## Reproducibility flag
N/A -- this is a mechanical baseline defined by the contract itself, not a
literature reproduction, so there's no published number to compare against
directly. Its own combined score (to be computed by Agent T) is the
reference point for judging whether methods 2-8 add value.
