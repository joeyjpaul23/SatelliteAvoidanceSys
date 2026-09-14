# Model fidelity validation

Numbers on this page were measured, not assumed. Each is reproducible with
`python -m aegis.experiments validate` (see `aegis/src/aegis/experiments/`).
Ground truth throughout is SGP4 itself: a plan is applied to the catalog,
both catalogs are propagated, and the realised geometry is measured.

## 1. Burn application: does SGP4 see what the optimizer intended?

The optimizer's dynamics model is the exact Clohessy-Wiltshire
position-from-impulse map `Phi(n, sigma)`
(`aegis.fleetopt.dynamics.cw_impulse_matrix`). For that model to mean
anything, applying a burn to an SGP4 object must reproduce it.

Two mappings were compared on a 550 km synthetic satellite
(n = 1.09482e-3 rad/s, period 5739 s), burning at 600 s after epoch and
measuring the displacement in the unmaneuvered satellite's RTN frame at
lead times of 0.23, 0.61, 1.37, 2.84 and 5.19 orbits.

### 1a. Legacy mapping — constant phase offset

`aegis.maneuver.rescreen.apply_along_track_burns` converts the total CW
along-track displacement at one reference epoch into a mean-anomaly shift
and leaves mean motion unchanged. Measured against SGP4 at a *different*
epoch than the reference, the predicted miss change was wrong by up to
**24.8 %**, because a constant phase offset has no secular drift while the
CW response is dominated by the secular term `-3 dv sigma`. The TLE
mean-anomaly field additionally quantises the shift at 1e-4 deg = 12 m of
along-track position, which is a quarter of the displacement a 2 mm/s burn
produces one and a half orbits ahead.

This is a documented property of that function, not a defect against its own
contract (step 10); it is simply not a mapping an optimizer can be held to.

### 1b. New mapping — Gauss update plus mean-element refit

`aegis.fleetopt.apply.apply_impulse` works in two steps.

1. **Analytic.** `gauss_element_update` applies the first-order Gauss
   variational equations for a near-circular orbit, carrying the
   eccentricity as the vector `(e cos w, e sin w)` so the `e = 0` singularity
   never arises. This is consistent with the optimizer's model by
   construction: the semi-major-axis change `da = 2 dv_T / n` shifts mean
   motion by `dn = -3 dv_T / a`, whose along-track effect grows as
   `-3 dv_T (t - tau)` — exactly the secular term of the CW response — and the
   eccentricity change supplies the periodic `4 sin(n sigma)/n` term.
2. **Numerical.** `fit_mean_elements`, seeded with that result, matches the
   exact post-burn state with a trust-region least-squares solve over the six
   mean elements, using an **explicit Jacobian with absolute element steps**.
   A relative step cannot work here: 1e-6 of a 50 degree angle is six metres
   of position, and the fit stalls there.

The refinement is accepted **only when it converges**. A partial fit is worse
than none: for a 2 mm/s impulse the solver settled a metre from target at the
burn epoch, and because that metre lands mostly in the semi-major axis it
grows secularly, making the resulting displacement wrong by a factor of ten
against 1.6 % for the analytic update it would have replaced.

**Measured against SGP4**, 40 randomly drawn along-track-dominant impulses
(seed 5) spanning 1–500 mm/s, each evaluated at 0.5, 1, 2 and 4 orbits of
lead — 160 samples:

| Impulse magnitude | n | refit converged | abs error, median | rel error, median |
|---|---:|---:|---:|---:|
| ≥ 20 mm/s | 88 | **100 %** | 2.0–7.0 m | **0.22–0.24 %** |
| < 20 mm/s | 72 | 17 % | 9.8 m | 11.6 % |
| all | 160 | 62 % | 7.0 m | 0.28 % |

Pure single-axis impulses are the hard case; mixed three-axis impulses of
10 mm/s converge and land at 0.19 %. The relevant comparison is **absolute**:
about ten metres of disagreement, against an exclusion radius of 2 km and
required miss distances of hundreds of metres to kilometres. That is a
0.5 % effect on the quantities the optimizer constrains, and it is reported
with every result rather than assumed away.

Per-axis relative agreement for converged fits, measured at leads of 0.23,
0.61, 1.37, 2.84 and 5.19 orbits:

| Impulse axis | median | max |
|---|---:|---:|
| Transverse (along-track) | **0.20 %** | 0.42 % |
| Radial | **0.43 %** | 4.2 % |
| Normal (cross-track) | **2.6 %** | 7.6 % |

Cross-track error grows with lead time (0.5 % at a quarter orbit to 7.6 % at
five orbits): that is differential J2 nodal regression from the inclination
change, which CW does not model. It is a real limitation and is why the
optimizer prices cross-track burns at five times along-track
(`DEFAULT_AXIS_WEIGHTS` in `aegis.fleetopt.norms`) rather than relying on
them.

**Consequence.** Replacing the legacy phase-offset mapping with this one
improves optimizer/simulator consistency from ~25 % to ~0.2 % for the
along-track burns that do the work — more than two orders of magnitude.

### 1c. Placing a conjunction where you want it

The same `fit_mean_elements` machinery, run with `reseed=True`, constructs a
satellite whose SGP4 state at a chosen epoch is a chosen state — which is how
the scenario generator places a close approach at an exact miss distance
rather than searching for one. From a far seed the solve can stall in a local
minimum a few kilometres out (suspiciously close to the J2 short-period
radial amplitude, which is presumably what it is); a **deterministic**
multi-start escapes it. Measured over seven placements spanning 0.05–2.0 km
of target miss and 5–170 degrees of plane angle: **every one lands within
1e-5 km of target with a residual below 1e-8 m**.

## 2. Miss-distance sensitivity: scalar versus B-plane

`aegis.maneuver.planner.plan_maneuvers` models the post-maneuver miss as

```
miss_after = miss_before + (y_rtn / miss) * delta_y_along_track
```

with the secondary's contribution scaled by `align = t_primary . t_secondary`.
`aegis.fleetopt.bplane` instead uses the full three-axis CW response, rotated
through each satellite's own RTN-to-ECI matrix, projected onto the encounter
B-plane.

### 2a. Primary maneuvers only

39 cases across six synthetic catalogs, burns of 5 / 25 / 100 mm/s at 1.5
orbits of lead. Error in the **predicted change** of miss distance:

| Model | median | p90 | max |
|---|---:|---:|---:|
| B-plane | 12.3 % | 18.2 % | 18.2 % |
| Scalar | 9.9 % | 18.2 % | 18.2 % |

**The two models are equivalent here, and that is expected.** At the true
TCA the miss vector is orthogonal to the relative velocity, so
`d_hat^T P = d_hat^T`: the B-plane projector changes nothing at first order.
Both models' residual error is second order and is dominated by one case
family — the co-planar pair with 0.01 km/s relative speed, where a 2.6 km
displacement shifts TCA by 258 s and the neglected relative acceleration
(`|d| n^2 = 1.17e-5 km/s^2`) contributes 0.39 km, matching the observed
0.398 km discrepancy. That is the low-relative-velocity regime the codebase
already flags through `LOW_RELATIVE_VELOCITY_KM_S`.

No improvement is claimed for this case.

### 2b. Secondary maneuvers — cross-plane intra-fleet conjunctions

This is the case the project exists for: both objects belong to the
maneuverable fleet, their planes differ, and the *secondary* burns. 24 cases,
burns of 10 / 50 mm/s.

| Model | median | p90 | max |
|---|---:|---:|---:|
| B-plane | **1.0 %** (cross-plane subset) | 9.9 % | 9.9 % |
| Scalar | **128 %** (cross-plane subset) | 130 % | **132 %** |

And the error is not merely large — **it has the wrong sign**. For a pair at
106.04 deg plane separation and 12.12 km/s relative speed:

| dv | true change | B-plane | scalar |
|---|---:|---:|---:|
| 10 mm/s | **−0.1450 km** | −0.1473 km | **+0.0424 km** |
| 50 mm/s | **−0.6660 km** | −0.7317 km | **+0.2119 km** |

The scalar model predicts the burn *increases* separation when it actually
*decreases* it. An optimizer using it would spend fuel making an intra-fleet
conjunction worse, and would report the conjunction resolved.

The cause is structural, not a coding slip. The scalar model estimates the
secondary's contribution as `(d_hat . t_primary) * (t_primary . t_secondary)`,
when the correct coefficient is `d_hat . t_secondary`. Those agree only when
the two orbit planes nearly coincide; at 106 deg separation the double
projection through the primary's frame inverts the sign. Co-planar pairs in
the same experiment show the two models agreeing to 0.01 %, which is exactly
the predicted behaviour.

**This is the quantitative justification for the B-plane formulation.** It is
not an accuracy refinement; for coordinated multi-satellite maneuvering it is
a correctness fix.

## 3. What is still approximate

- Linearised relative motion. Second-order error scales as
  `||displacement||^2 / (2 * miss)` and is reported with every delta-v claim.
- Low relative velocity (below `LOW_RELATIVE_VELOCITY_KM_S`): the TCA shift
  grows as `1/|v_rel|` and the neglected relative acceleration dominates.
  These events are flagged `short_encounter_valid=False` upstream and are
  reported separately.
- Cross-track burns beyond ~2 orbits of lead: differential J2 nodal
  regression, up to 5.6 % at 5 orbits.
- Every probability inherits TLE-grade covariance
  (`CovarianceSource.SYNTHETIC_TLE`). See `docs/LIMITATIONS.md`.
