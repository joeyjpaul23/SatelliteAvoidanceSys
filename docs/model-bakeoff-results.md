# The model bake-off: one real predictor, one rejected gate

Twenty-plus model families were trained on two targets where learning could
plausibly pay, after a profile showed that the LP the first session had been
accelerating was 1 % of the runtime. Both results are reported; one of them is
a rejection.

Dataset: 423 scenarios across ten families and four delta-v budgets, split
60/20/20 by scenario digest, built in 1 500 s of which 1 201 s was screening.

---

## 1. Pre-screening triage — **rejected**

**The target.** 53 % of a benchmark sweep needed no maneuver at all and paid
full screening cost anyway. Predicting `needs_maneuver` from mean elements
alone, before any propagation, would skip that cost.

**The apparent result.** Every model family reached **ROC-AUC 1.0 and
specificity 1.0 at recall 1.0** on the held-out test split: gradient boosting,
random forest, logistic regression (L1 and L2), k-NN, MLPs, deep ensembles,
MC-dropout, Bayes-by-backprop, a Bayesian LSTM, DeepSets, a Set Transformer, a
1D CNN, and a two-feature closed-form rule at 0.57 µs. An L1 fit needed 7 of 66
features. A depth-2 tree needed two.

Five unrelated model families all hitting a ceiling is a warning, not a win.

**The rejection.** The label is constant within **nine of the ten** scenario
families:

| family | n | `needs_maneuver` |
|---|---:|---|
| crossing-planes, induced-cascade, isolated-pair, star | 220 | **100 % True** |
| clique, debris-shower, dense-shell, intra-plane, replay-tle | 163 | **0 % True** |
| chain | 40 | 60 % — the only mixed family |

So "triage" is family recognition, and two pooled statistics accomplish it.
A leave-one-family-out refit of the same rule shape, at recall 1.0 on the
remaining families, confirms it:

| held-out family | positives | skipped in error | recall |
|---|---:|---:|---:|
| chain | 24 | 0 | 1.000 |
| crossing-planes | 40 | 0 | 1.000 |
| isolated-pair | 40 | 0 | 1.000 |
| star | 40 | 0 | 1.000 |
| **induced-cascade** | **100** | **100** | **0.000** |

Held out the family the whole project is about, the rule skips **every single
scenario that needed a maneuver**, silently. The published rule also has three
false negatives across all 423 rows (recall 0.988), not the 1.000 its test
split showed.

The one error this system cannot tolerate is a false negative — skipping a
scenario that needed a burn. **No triage gate is implemented.**

Two rules survive as honest curiosities. `skip iff every pair is co-planar`
has recall 1.0 on every split at specificity 0.42–0.55, and is the only
non-vacuous rule with a physical reading. The genuinely *sound* physics rule —
`skip iff no pair passes the perigee/apogee prefilter` — fires on **0 of 423**
scenarios at every pad tested, because every synthetic family lives on one
550 km shell. Sound and useless.

**What would change the answer**: a benchmark whose families vary their label
*within* the family — different altitude shells, Pc margins and crossing speeds
inside each generator — so that the task stops being family lookup. As built,
the benchmark cannot distinguish "this generalises" from "this memorised the
generator", and no amount of model capacity fixes that.

### The architectures, and what condemned each

Every one reached the same ROC 1.0 / specificity 1.0 ceiling as a 0.57 µs
closed-form rule, so cost decided:

| architecture | inference | verdict |
|---|---:|---|
| closed-form 2-feature rule | 0.57 µs | ceiling, zero parameters |
| logistic regression | 0.70 µs | ceiling |
| depth-2 decision tree | 29 µs | ceiling |
| MLP / CNN1d | 18–75 µs | ceiling, no gain |
| deep ensemble (5×) | 127 µs | ceiling, no gain |
| Set Transformer | 332–587 µs | ceiling, 11–1030× cost |
| MC-dropout, Bayes-by-backprop | 1.4–5.2 ms | ceiling, no gain |
| **Bayesian LSTM / BiLSTM** | 2.0–4.7 ms | ceiling, 70–160× cost, no gain |
| DeepSets (sum / sqrt-sum pooling) | 73–141 µs | **ROC 0.56 — failed**, and extrapolated catastrophically to the 132-object catalogs it had never seen |
| physics-informed DeepSets | 73–141 µs | **failed to internalise its own sufficient condition** — predicted p = 0.59 on catalogs where every pair is provably separated |

**Does Bayesian uncertainty earn its cost?** No. The ensemble spread and the
variational LSTM's standard deviation do rank the borderline cases correctly,
but the "failures" they catch are threshold-transfer noise on five `chain`
rows, and choosing the threshold from train+val instead of val alone removes
them for free. Error-detection AUROC is undefined for every Bayesian variant
because there are zero misclassifications to detect.

---

## 2. The safety-premium predictor — **adopted**

The first session proposed the *coupling number*, `cond(BBᵀ)` of the
row-normalised projected sensitivities, and measured its rank correlation with
the premium at **0.074**. This is the replacement.

```
                    ( f_j  -  u_j · (d_j + B_j x_fuel) )_+
    varrho  =  max  ---------------------------------------
                j                  reach_j

    reach_j  =  sum_s  D_s · || row_j restricted to satellite s ||_inf
```

The numerator is the violation the **cheapest** plan commits on latent row *j*;
the denominator is the most any admissible plan could move that row. The ratio
is dimensionless and reads as *"what fraction of the fleet's reach is already
spoken for by the fuel-optimal plan's worst induced conjunction."*

Measured independently of the bake-off, on 70 `induced-cascade` scenarios:

| predictor | Spearman | univariate R² |
|---|---:|---:|
| **penetration index** | **0.989** | **0.866** |
| neighbour spacing (log1p) | −0.855 | 0.625 |
| conflict dimension | 0.462 | 0.085 |
| **coupling number `cond(BBᵀ)`** | **constant — predicts nothing** | — |

The bake-off's own run reported Spearman 0.943 / R² 0.899 on a different
budget mix, and found nothing better across thirteen hand-constructed scalars
and a random forest on sixty-six pre-screening features (R² 0.898, worse rank
correlation, not interpretable).

**The sharper property is the exact one.** The index is zero *if and only if*
the premium is zero — verified on **70 of 70** scenarios here and 71 of 71 in
the bake-off. If the fuel-optimal plan violates no latent row, coordination is
free. That is an equivalence, not a correlation, and it answers the operational
question ("is there a premium at all?") exactly rather than approximately.

**What is deliberately not claimed.** The fitted slope is not a calibrated
magnitude: 2.72 on one dataset, 5.73 on another with a different budget,
because it absorbs the scenario's own delta-v scale. The index ranks and
detects; it does not predict a percentage.

Cost: one pass over the latent rows using the fuel-only solution the planner
already computes. Implemented as
`aegis.fleetopt.pareto.penetration_index`, surfaced on every `FleetPlan` and
recorded by the benchmark.
