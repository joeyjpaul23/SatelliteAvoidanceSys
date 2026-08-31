# Shared Contract — Collision-Risk Prediction Benchmark

Fixed by the human before any agent started work. Identical copy given to Agent B and Agent T.

## Dataset
ESA Kelvins Collision Avoidance Challenge dataset (kelvins.esa.int/collision-avoidance-challenge/data).
Use the original train/test split as published, not a custom split — this is what makes results
comparable to the published literature.

## Prediction task
For each `event_id` in the test set, predict the final risk value (log10 of collision probability)
at time of closest approach (TCA), using only CDM data available >= 2 days prior to TCA — matching
the original competition's rules exactly.

## Output format (Agent B's only deliverable per method)
A CSV with exactly two columns: `event_id`, `predicted_risk` — one row per test event.
No model files, no code, no intermediate features handed to the testing team.

## Official metric
Kelvins challenge combined score = MSE on high-risk events (MSE_HR) scaled by (1 / F2), where F2 is
the classification F-score (beta=2, weighting recall over precision) at the risk threshold of -6
(i.e. Pc = 1e-6). Lower score is better.
Source: Uriot et al., "Spacecraft Collision Avoidance Challenge: design and results of a machine
learning competition," Astrodynamics 6, 121-140 (2022); and the Kelvins challenge site's own
scoring page.

Known quirk (per Uriot et al.): the metric's optimal strategy for a confidently-low-risk prediction
is r = -6-epsilon, not the true value. Agent T's evaluation methodology should account for this.

## Mandatory baseline
Naive "persistence" baseline: use the latest available risk value before the 2-day cutoff as the
final prediction. Every method's score is reported alongside this baseline's score.

## Methods (given to Agent B)
1. Naive baseline — persistence of last known risk value (mandatory, build first).
2. Gradient boosting / ensemble — reproduce approach direction from Metz & Dart (Metz's 2020 TU
   Darmstadt thesis, "Implementation and comparison of data-based methods for collision avoidance
   in satellite operations").
3. Bayesian LSTM — reproduce Pinto et al., "Towards Automated Satellite Conjunction Management with
   Bayesian Deep Learning," arXiv:2012.12450. Starting point: kessler.nn.LSTMPredictor
   (github.com/kesslerlib/kessler). Verify Kessler's actual LICENSE file (GPL-3.0 per GitHub vs.
   BSD-3-Clause per conda-forge — discrepancy must be resolved before relying on it).
4. Probabilistic programming model — reproduce Acciarini et al., "Spacecraft Collision Risk
   Assessment with Probabilistic Programming," arXiv:2012.10260 (in Kessler, built on Pyro).
5. Hybrid physics + ML (SGP4 residual correction) — CNN or similar correcting SGP4 propagation
   error, feeding cleaner features into a downstream risk predictor. Referenced in a 2026 paper
   (Li et al., cited in "Machine learning-based risk classification of space debris conjunction
   events," EPJ Web of Conferences, 2026) — track down and read the original Li et al. source.
6. Physics-informed GAN — reproduce the general approach of "Improving Satellite Collision Risk
   Prediction via Physics-Informed Generative Adversarial Networks" (2025).
7. Novel — Transformer/attention-based sequence model over the CDM time series (token sequence with
   time/feature encodings). No published work applies this to this benchmark.
8. Novel — Combined hybrid — merge method 5 (physics-based propagation-error correction) with
   method 7 or 3 (learned sequence modeling on corrected features). No published work does both.

Methods 1-6: faithful reproductions first. Methods 7-8: genuinely exploratory, underperformance
expected and fine.

## Builder rules (Agent B)
- Build each method as an independent, self-contained pipeline.
- No access to true test-set labels — only training set and unlabeled test set inputs.
- Do not write or see evaluation/scoring code.
- Only deliverable per method: a CSV of predictions in the exact format above.
- Do not infer/guess true test labels through any means (no hand-tuning to "what seems right").
- Document per method: which paper/approach is being reproduced, what was changed or couldn't be
  fully reproduced (missing implementation details), reasoning for design choices left ambiguous
  by the source material.
- Flag clearly if a method's published results seem non-reproducible from available information.
- Hand off only prediction CSVs + method-documentation notes. No training code, weights, or
  feature-engineering scripts to the testing team.
