# Certified acceleration: what worked and what did not

Two things were built here and they came out differently. Both numbers are
reported because the second one is the interesting one.

## 1. The certified lazy-constraint closure — works, and scales

`aegis.fleetopt.solver.lazy_solve` solves the fleet program with a *subset* of
the induced-conjunction rows, checks every omitted row against the solution,
adds back any that is violated, and repeats until none is. At termination the
point is feasible for the full program and its cost is a lower bound on the
full optimum (every intermediate program is a relaxation), so **the returned
solution is exactly optimal for the full problem, whatever subset was
guessed.**

Measured over 27 scenarios spanning five families, for four kinds of guess:

| guess | max relative objective gap | mismatches (> 1e-7) | rows used (median) | rounds |
|---|---:|---:|---:|---:|
| empty | 1.4e-16 | 0/27 | 8.3 % | 2 |
| model-predicted | 0.0 | 0/27 | 41.7 % | 1 |
| adversarial (worst-margin rows) | 1.4e-16 | 0/27 | 18.8 % | 2 |
| all rows | 0.0 | 0/27 | 100 % | 1 |

Machine epsilon, for every guess including an empty and an adversarial one.

**One bug had to be fixed to make that true**, and it is worth recording
because it is easy to get wrong. The loop originally tested each omitted row
with its *physical* residual, `||d + Bx|| - floor`. The program enforces the
*linearized* row, `u·(d + Bx) - floor`, and by Cauchy-Schwarz the linear form
is never larger. So a point could satisfy the true separation while violating
the row, the loop would terminate believing everything was fine, and the
returned objective would be strictly below the full problem's. It disagreed on
6 of 27 scenarios. `LatentConstraint.linear_residual_km` exists so the
distinction cannot be made again by accident.

### Speedup scales with the problem

With an oracle guess (the rows that actually bind), on the `induced-cascade`
family with the neighbour count and latent grid step varied:

| latent rows | rows used | rounds | full solve | lazy solve | speedup |
|---:|---:|---:|---:|---:|---:|
| 68 | 1 | 1 | 4.5 ms | 2.5 ms | **1.8×** |
| 358 | 2 | 1 | 18.5 ms | 5.4 ms | **3.5×** |
| 820 | 2 | 1 | 48.5 ms | 8.3 ms | **5.9×** |
| 1716 | 2 | 1 | 145.8 ms | 16.1 ms | **9.0×** |

Objective gap exactly zero at every size. The active set is tiny — typically
**two** binding rows out of 1716 — which is precisely why the reduction pays.

## 2. The learned active-set predictor — did not pay off

`ConjunctionPIGNN` (892k parameters) was trained on 320 scenarios across eight
families, split by scenario digest, 258/39/23 train/val/test. Training is
exactly reproducible: two runs with the same seed gave identical validation
loss to the last printed digit (difference 0.00e+00).

Calibrated on the validation split:

| target recall | threshold | achieved recall | precision | rows retained |
|---|---:|---:|---:|---:|
| 0.999 | 0.123 | 1.000 | 0.117 | 64.6 % |
| 0.95 | 0.316 | 0.962 | 0.285 | 25.5 % |

On 18 held-out large scenarios, end to end and including inference time:

| metric | value |
|---|---|
| objective gap vs the full solve | **0.00e+00** (0/18 mismatches) |
| median active-set recall on the binding rows | **0.00** |
| median end-to-end speedup vs the full solve | **0.77×** |
| median speedup of an **empty** guess instead | **~4×** (range 1.7–8.6×) |

**The empty guess beats the trained network.** Two reasons, both measurable:

1. **The active set is far smaller than the network's output.** Typically two
   rows bind out of 1716; the network at its calibrated threshold proposed
   between 2 and 146. A reduced program of 146 rows is slower than one of 9,
   which is what the empty start converges to in two rounds.
2. **Inference costs as much as the solve.** 8–56 ms of forward pass against a
   14–180 ms solve. At this problem size there is no room for a learned prior
   to earn its keep.

### What this means for the contribution

The claim that survives is about the *mechanism*, not the model: an active-set
guess can be turned into an exactly-certified speedup, and the speedup grows
with problem size. The claim that does not survive, on this data, is that a
learned predictor supplies a better guess than the trivial one. Claim C6 in
`docs/prior-art-and-novelty-ledger.md` is narrowed accordingly.

Two honest caveats on the negative result. The training set is small (320
scenarios, 489 active edges in total), and the label is extremely imbalanced —
learning to find two rows in seventeen hundred is a needle-in-a-haystack
problem that the loss weighting only partly compensates for. A larger, harder
corpus might change the answer. It is reported as measured, not as settled.
