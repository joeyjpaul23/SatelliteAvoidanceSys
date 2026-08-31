"""
Method 3: Bayesian LSTM (reproducing Pinto et al., "Towards Automated
Satellite Conjunction Management with Bayesian Deep Learning," arXiv:2012.12450),
using kessler.nn.LSTMPredictor (github.com/kesslerlib/kessler) as the
starting point per the shared contract.

kessler's LSTMPredictor doesn't predict risk directly -- it autoregressively
predicts the *next CDM* in an event's time series (full state + raw
covariance terms for both objects), one step at a time. Its "Bayesian"
character comes from Monte Carlo Dropout: `LSTMPredictor.predict()` keeps
the model in `.train()` mode (dropout active) during inference, so repeated
calls sample from an approximate posterior over predictions rather than
returning one fixed output -- this is a design choice in kessler's own code,
not something added here. We roll each test event forward autoregressively
to (predicted) TCA multiple times per event (`predict_event(num_samples=K)`)
and average the resulting risk across samples.

See NOTES.md for kessler bugs/gaps found and worked around while building
this (a crashing progress bar, and a default feature list incompatible with
kessler's own Kelvins data loader), the event_id recovery workaround (the
loader doesn't retain event_id on its output objects), and the fallback path
for the ~20% of events kessler's own data cleaning drops.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

import kessler.util as kutil
# kessler 1.0.1's progress bar calls an undefined helper function and
# crashes outright (see NOTES.md) -- neutered here, not part of the
# reproduction itself.
kutil.progress_bar_update = lambda *a, **kw: None
kutil.progress_bar_end = lambda *a, **kw: None
kutil.progress_bar_init = lambda *a, **kw: None

import kessler.data as kdata
from kessler.nn import LSTMPredictor

from pc_calculator import compute_risk_from_raw

CUTOFF = 2.0
# Full paper-scale settings, run explicitly at the user's request accepting
# the ~6+ hour runtime (benchmarked at ~0.65s per MC-Dropout sample per
# event; each `.predict()` call reprocesses the whole growing sequence
# through the LSTM from scratch -- an O(n^2)-per-rollout cost that's a
# property of kessler's own implementation, not something changed here).
# MAX_ROLLOUT_LENGTH left at kessler's own predict_event() default (22).
MC_SAMPLES = 20
MAX_ROLLOUT_LENGTH = 22
EPOCHS = 6
TRAIN_CSV = "../../data/extracted/train_data.csv"
TEST_CSV = "../../data/test_data.csv"

# kessler's default LSTMPredictor.features list includes OBJECT1_X/Y/Z etc.
# (Cartesian state), which kelvins_to_event_dataset() never populates -- the
# library's own tutorial combination doesn't work out of the box (see
# NOTES.md). This list is restricted to fields the Kelvins loader actually
# sets: relative geometry + full raw covariance (both objects) + OD-quality
# fields, in place of the unavailable Cartesian state.
FEATURES = [
    "__CREATION_DATE", "__TCA",
    "MISS_DISTANCE", "RELATIVE_SPEED",
    "RELATIVE_POSITION_R", "RELATIVE_POSITION_T", "RELATIVE_POSITION_N",
    "RELATIVE_VELOCITY_R", "RELATIVE_VELOCITY_T", "RELATIVE_VELOCITY_N",
]
for obj in ["OBJECT1", "OBJECT2"]:
    FEATURES += [
        f"{obj}_CR_R", f"{obj}_CT_R", f"{obj}_CT_T", f"{obj}_CN_R", f"{obj}_CN_T", f"{obj}_CN_N",
        f"{obj}_CRDOT_R", f"{obj}_CRDOT_T", f"{obj}_CRDOT_N", f"{obj}_CRDOT_RDOT",
        f"{obj}_CTDOT_R", f"{obj}_CTDOT_T", f"{obj}_CTDOT_N", f"{obj}_CTDOT_RDOT", f"{obj}_CTDOT_TDOT",
        f"{obj}_CNDOT_R", f"{obj}_CNDOT_T", f"{obj}_CNDOT_N", f"{obj}_CNDOT_RDOT", f"{obj}_CNDOT_TDOT", f"{obj}_CNDOT_NDOT",
        f"{obj}_RECOMMENDED_OD_SPAN", f"{obj}_ACTUAL_OD_SPAN", f"{obj}_OBS_AVAILABLE", f"{obj}_OBS_USED",
        f"{obj}_RESIDUALS_ACCEPTED", f"{obj}_WEIGHTED_RMS", f"{obj}_SEDR",
    ]


def surviving_event_ids(file_name, drop_features=("c_rcs_estimate", "t_rcs_estimate"), remove_outliers=True):
    """Independently replicates kelvins_to_event_dataset()'s row-filtering
    (same code path, copied here) so we can recover which event_id each
    entry of the returned EventDataset corresponds to -- kessler's own
    output objects don't retain event_id anywhere. Verified (see NOTES.md)
    that pandas groupby order matches kessler's own iteration order and that
    len() matches exactly."""
    kelvins = pd.read_csv(file_name)
    kelvins = kelvins.drop(list(drop_features), axis=1)
    kelvins = kelvins.dropna()
    if remove_outliers:
        kelvins = kelvins[kelvins["t_sigma_r"] <= 20]
        kelvins = kelvins[kelvins["c_sigma_r"] <= 1000]
        kelvins = kelvins[kelvins["t_sigma_t"] <= 2000]
        kelvins = kelvins[kelvins["c_sigma_t"] <= 100000]
        kelvins = kelvins[kelvins["t_sigma_n"] <= 10]
        kelvins = kelvins[kelvins["c_sigma_n"] <= 450]
    return list(kelvins.groupby("event_id").groups.keys())


def naive_fallback(test_csv, event_ids):
    """For test events kessler's own cleaning drops entirely, fall back to
    method 1's naive persistence (last available risk value) so every
    test event_id still gets a prediction, per the contract."""
    df = pd.read_csv(test_csv)
    df = df[df.event_id.isin(event_ids)]
    latest = df.loc[df.groupby("event_id")["time_to_tca"].idxmin()]
    return dict(zip(latest.event_id, latest.risk))


def cdm_cov_terms(cdm, obj):
    return (
        cdm[f"{obj}_CR_R"], cdm[f"{obj}_CT_R"], cdm[f"{obj}_CT_T"],
        cdm[f"{obj}_CN_R"], cdm[f"{obj}_CN_T"], cdm[f"{obj}_CN_N"],
    )


def main():
    torch.manual_seed(0)

    print("=== Loading data ===")
    train_events = kdata.kelvins_to_event_dataset(TRAIN_CSV)
    test_events = kdata.kelvins_to_event_dataset(TEST_CSV)

    train_ids = surviving_event_ids(TRAIN_CSV)
    test_ids = surviving_event_ids(TEST_CSV)
    assert len(train_ids) == len(train_events), "train event_id recovery mismatch"
    assert len(test_ids) == len(test_events), "test event_id recovery mismatch"
    print(f"train: {len(train_events)} events survived kessler's cleaning")
    print(f"test: {len(test_events)} of 2167 events survived kessler's cleaning")

    print("=== Training LSTMPredictor ===")
    predictor = LSTMPredictor(features=FEATURES)
    t0 = time.time()
    predictor.learn(train_events, epochs=EPOCHS, batch_size=16, device="cpu", num_workers=0)
    print(f"\ntraining took {time.time()-t0:.0f}s")

    print("=== Predicting test events (MC-Dropout rollout to TCA) ===")
    results = {}
    t0 = time.time()
    for i, event in enumerate(test_events):
        event_id = test_ids[i]
        rollouts = predictor.predict_event(event, num_samples=MC_SAMPLES, max_length=MAX_ROLLOUT_LENGTH)
        risks = []
        for sample_event in rollouts:
            final_cdm = sample_event[-1]
            target_terms = cdm_cov_terms(final_cdm, "OBJECT1")
            chaser_terms = cdm_cov_terms(final_cdm, "OBJECT2")
            rel_pos = [final_cdm["RELATIVE_POSITION_R"], final_cdm["RELATIVE_POSITION_T"], final_cdm["RELATIVE_POSITION_N"]]
            rel_vel = [final_cdm["RELATIVE_VELOCITY_R"], final_cdm["RELATIVE_VELOCITY_T"], final_cdm["RELATIVE_VELOCITY_N"]]
            # object type isn't part of the predicted feature set (categorical,
            # constant for the event) -- read it from the event's own last
            # real (non-predicted) CDM instead.
            c_object_type = event[-1]["OBJECT2_OBJECT_TYPE"]
            risk = compute_risk_from_raw(target_terms, chaser_terms, rel_pos, rel_vel, c_object_type)
            risks.append(risk)
        results[event_id] = float(np.mean(risks))
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(test_events)} events, {elapsed:.0f}s elapsed, ~{elapsed/(i+1)*len(test_events):.0f}s total est.")
            sys.stdout.flush()

    print("=== Naive fallback for events kessler's cleaning dropped ===")
    missing_ids = sorted(set(range(2167)) - set(test_ids))
    print(f"{len(missing_ids)} test events need fallback: {missing_ids[:20]}{'...' if len(missing_ids) > 20 else ''}")
    fallback = naive_fallback(TEST_CSV, missing_ids)
    results.update(fallback)

    out = pd.DataFrame(sorted(results.items()), columns=["event_id", "predicted_risk"])
    assert len(out) == 2167, f"expected 2167 predictions, got {len(out)}"
    out.to_csv("predictions.csv", index=False)
    print(f"wrote {len(out)} predictions to predictions.csv")
    print(out.describe())


if __name__ == "__main__":
    main()
