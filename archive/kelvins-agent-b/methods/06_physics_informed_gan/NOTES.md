# Method 6: Physics-Informed GAN (Data Augmentation)

## What this reproduces, and the access gap
Reproduces the general approach of "Improving Satellite Collision Risk Prediction via
Physics-Informed Generative Adversarial Networks" (2025). Every fetch attempt
(ResearchGate page, direct search-result links) returned HTTP 403. Implementation is
built entirely from the search-result abstract snippet, quoted here in full since
it's the only source material available:

> "A stabilized Physics-Informed Generative Adversarial Network (P-GAN) was
> developed for augmenting imbalanced satellite conjunction data, utilizing a
> conditional sequence-to-sequence architecture that generates high-fidelity,
> physically consistent time-series of Conjunction Data Messages (CDMs)... Key
> innovations include a stabilized generator with bounded outputs, a comprehensive
> physics loss function incorporating twelve orbital mechanics constraints, and a
> robust post-generation filtering pipeline... training state-of-the-art models on
> augmented data significantly improves detection of critical high-risk events as
> measured by the official ESA challenge metric."

Not reproduced/confirmable from this snippet alone: the exact architecture, the full
list of twelve physics constraints, the "post-generation filtering pipeline," or any
reported numbers. What follows is a faithful-in-*structure* implementation of what
the abstract describes, not a faithful-in-*detail* reproduction.

## The key structural decision this required
A plain GAN has no natural way to produce a risk prediction for one specific real
test event -- it's a generator, not a conditional point-predictor. Forcing it into
that role would misrepresent both the source paper (which explicitly describes it as
an *augmentation* technique) and the architecture itself. So this implementation
follows the paper's actual stated use: **the GAN generates synthetic high-risk-like
CDM sequences to augment the training set for a separate downstream predictor**,
which is then evaluated on the real test set. Method 5's residual-correction CNN
architecture is reused as that downstream predictor (retrained from scratch on
augmented data) rather than designing a second full architecture from nothing --
the thing actually being tested here is whether GAN-based augmentation helps, not
which downstream architecture is best.

## Design
- **High-risk class definition**: training events with true final risk > -6, using
  the contract's own official-metric threshold (Pc = 1e-6) rather than an arbitrary
  cut. 285 of 10,782 usable training events (2.6%) qualify -- a real, severe class
  imbalance, matching the paper's stated motivation.
- **Generator**: conditioned on a noise vector (16-dim) plus a scalar target
  log(sigma_scale at TCA), sampled from real high-risk events' own final values (so
  generation is explicitly biased toward the minority class). A GRU decoder unrolls
  20 steps to produce a synthetic 20-feature-per-step CDM sequence, with bounded
  (tanh-scaled) output -- the "stabilized generator with bounded outputs" the
  abstract describes.
- **Discriminator**: a GRU encoder + linear head classifying real vs. generated
  sequences.
- **Physics loss** (three constraints; "twelve" from the source paper could not be
  confirmed, see access note above):
  1. Non-negativity pressure on the covariance-sigma channels.
  2. Smoothness -- penalizes implausibly large step-to-step feature jumps, since
     real CDM sequences evolve gradually.
  3. Monotone covariance shrinkage toward the sequence's end, matching the same
     real, data-confirmed trend method 4's NOTES.md documents and calibrates
     against actual Kelvins data.
- **Training**: standard alternating GAN updates, 40 epochs (~1s -- this is a small
  model). Generator loss = adversarial term + 0.5x physics-loss term.
- **Augmentation**: 2,000 synthetic sequences generated (conditioned on real
  high-risk events' target sigma-scale values), added to the 10,782 real training
  sequences. Synthetic examples get a covariance-correction target derived from
  their own conditioning value (biasing the downstream model's correction pathway
  toward the high-risk regime); no position-correction target exists for pure
  synthetic data, so that component is set to 0 for synthetic rows only.
- **Downstream + prediction**: identical architecture, training procedure, and Pc
  pipeline to method 5, just trained on the augmented set instead of the original
  training set alone -- the one deliberate controlled variable between these two
  methods' results.

## Results
Downstream training MSE: 6.32 after 8 epochs (vs. method 5's 7.25 on the same
architecture without augmentation, same epoch count). Some improvement, though I
want to be honest about what this comparison can and can't support: it's one run
each, no repeated trials, and the two methods' training data differ in more than
just "augmented or not" (this run also re-initializes the CNN's weights from a
different random seed state, since it runs after the GAN training consumed RNG
draws). Not claiming this demonstrates the augmentation helped -- flagging it as a
directionally consistent, single-run observation for Agent T/A to weigh, not a
result.

2,167 predictions (2,088 model-derived + 79 fallback, identical fallback set to
methods 4 and 5 since it's driven by the same missing-field pattern in the raw
test data). Floor rate (-30): **87.1%** -- the highest yet across all six methods
built so far (53.3% / 77.7% / 18.8% / 76.1% / 85.9% / 87.1%). Given this sits even
higher than method 5's already-undertrained result, and both share the same
downstream architecture, I read this as consistent with the same undertrained-model
explanation flagged in method 5's NOTES.md, not a signal that this method is more
discriminating.

## Reproducibility flag
Same two-part caveat as method 5: the source paper was inaccessible (403 throughout),
so this is built from one abstract paragraph, not the actual paper; and the exact
physics-constraint list ("twelve") could not be confirmed, so three defensible
substitutes were used instead and documented as such rather than presented as a
match.
