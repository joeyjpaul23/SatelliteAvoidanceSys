"""
Method 4: original Pyro probabilistic-programming model.

NOT a reproduction of Acciarini et al., "Spacecraft Collision Risk Assessment
with Probabilistic Programming" (arXiv:2012.10260) -- see NOTES.md for why.
In short: that paper's own reported experiment infers posteriors over orbital
elements for a single event it generated itself (synthetic ground truth,
known latents), never predicts a risk value, was implemented in PyProb (not
Pyro), and explicitly states real-CDM evaluation was unstarted future work.
There is no published risk-prediction result for that method to reproduce.

This is an original probabilistic model built for this benchmark, following
the same general philosophy (a physics-motivated model of uncertainty
evolution, fit with Pyro, used for Bayesian risk assessment) without
claiming to reproduce their specific experiment.

Design:
  Orbit-determination uncertainty about an object's *current* state
  generally shrinks CDM-to-CDM as TCA approaches (more tracking data
  accumulates). For each event, treat the combined position-uncertainty
  scale (RSS of both objects' RTN sigmas) as log-linear in time_to_tca,
  and fit a per-event Bayesian linear regression via Pyro SVI using only
  that event's own >=2-day CDM history. Priors on the regression's
  intercept/slope are calibrated from the population-level trend across
  the full training set, so events with few CDMs (most test events have
  only a handful in the >=2-day window) shrink toward the population
  trend rather than overfitting 1-2 noisy points -- this partial pooling
  is the actual reason to do this Bayesian rather than as a plain
  per-event OLS fit.

  Sample from each event's posterior, scale the last-known covariance
  matrices by the resulting sigma ratio (predicted log-sigma at TCA is
  exactly the intercept, by construction of the t=0 parameterization),
  linearly extrapolate relative position via last-known relative
  velocity, and run each posterior draw through the same 2D Pc calculator
  method 2 uses (reused here for consistency, not re-derived). Final
  predicted_risk = log10(posterior-predictive mean Pc) -- i.e. E[Pc]
  averaged in Pc-space across draws, then logged once at the end (the
  statistically correct way to summarize a posterior predictive risk,
  not an average of logs).
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from pc_calculator import compute_risk, HBR_BY_TYPE

CUTOFF = 2.0
SIGMA_COLS = ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]
SIGMA_SANITY_BOUND = 1e5  # meters; same sentinel-cleaning bound method 2 uses
CORR_COLS = ["t_ct_r", "t_cn_r", "t_cn_t", "c_ct_r", "c_cn_r", "c_cn_t"]
SECONDS_PER_DAY = 86400.0
RISK_FLOOR = -30.0
N_SVI_STEPS = 300
N_POSTERIOR_DRAWS = 30
TRAIN_CSV = "../../data/extracted/train_data.csv"
TEST_CSV = "../../data/test_data.csv"


def sanitize(df):
    df = df.copy()
    for c in SIGMA_COLS:
        df.loc[df[c].abs() > SIGMA_SANITY_BOUND, c] = np.nan
    return df


def sigma_scale(row):
    return float(np.sqrt(sum(row[c] ** 2 for c in SIGMA_COLS)))


def fit_population_prior(train_csv):
    """OLS log(sigma_scale) ~ time_to_tca per training event (>=2 CDMs in the
    >=2-day window), pooled across events to get prior mean/std for the
    per-event Pyro regression's intercept and slope, plus a fixed
    observation-noise scale (population residual std)."""
    print("Fitting population-level prior from training data...")
    df = pd.read_csv(train_csv)
    df = sanitize(df)
    df = df[df.time_to_tca >= CUTOFF].dropna(subset=SIGMA_COLS)
    df["log_sigma_scale"] = df[SIGMA_COLS].pow(2).sum(axis=1).pow(0.5).apply(
        lambda s: np.log(max(s, 1e-6))
    )

    intercepts, slopes, resid_stds = [], [], []
    for _, g in df.groupby("event_id"):
        if len(g) < 2:
            continue
        t = g["time_to_tca"].to_numpy()
        y = g["log_sigma_scale"].to_numpy()
        if np.ptp(t) < 1e-6:
            continue
        slope, intercept = np.polyfit(t, y, 1)
        resid = y - (intercept + slope * t)
        intercepts.append(intercept)
        slopes.append(slope)
        resid_stds.append(resid.std())

    intercepts, slopes, resid_stds = map(np.array, (intercepts, slopes, resid_stds))
    print(f"  {len(intercepts)} training events used for prior calibration")

    prior = {
        "intercept_mean": float(np.median(intercepts)),
        "intercept_std": float(1.4826 * np.median(np.abs(intercepts - np.median(intercepts)))) + 1e-3,
        "slope_mean": float(np.median(slopes)),
        "slope_std": float(1.4826 * np.median(np.abs(slopes - np.median(slopes)))) + 1e-3,
        "obs_noise": float(np.median(resid_stds)) + 1e-3,
    }
    print(f"  prior: {prior}")
    return prior


def svi_regression(t_obs, y_obs, prior):
    """Fit the per-event Bayesian linear regression via Pyro SVI. Returns
    posterior mean/std of the intercept (= predicted log(sigma_scale) at
    TCA, since time_to_tca=0 there)."""
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32)
    obs_noise = prior["obs_noise"]

    def model(t, y):
        intercept = pyro.sample("intercept", dist.Normal(prior["intercept_mean"], prior["intercept_std"]))
        slope = pyro.sample("slope", dist.Normal(prior["slope_mean"], prior["slope_std"]))
        mean = intercept + slope * t
        with pyro.plate("data", len(t)):
            pyro.sample("obs", dist.Normal(mean, obs_noise), obs=y)

    def guide(t, y):
        intercept_loc = pyro.param("intercept_loc", torch.tensor(float(prior["intercept_mean"])))
        intercept_scale = pyro.param(
            "intercept_scale", torch.tensor(float(prior["intercept_std"]) * 0.5),
            constraint=dist.constraints.positive,
        )
        slope_loc = pyro.param("slope_loc", torch.tensor(float(prior["slope_mean"])))
        slope_scale = pyro.param(
            "slope_scale", torch.tensor(float(prior["slope_std"]) * 0.5),
            constraint=dist.constraints.positive,
        )
        pyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))
        pyro.sample("slope", dist.Normal(slope_loc, slope_scale))

    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": 0.05}), loss=Trace_ELBO())
    for _ in range(N_SVI_STEPS):
        svi.step(t_obs_t, y_obs_t)

    intercept_loc = pyro.param("intercept_loc").item()
    intercept_scale = pyro.param("intercept_scale").item()
    return intercept_loc, intercept_scale


def predict_event_risk(event_df, prior, rng):
    """event_df: this event's own >=2-day-filtered, sanitized CDM rows."""
    event_df = event_df.sort_values("time_to_tca")
    t_obs = event_df["time_to_tca"].to_numpy()
    y_obs = event_df[SIGMA_COLS].pow(2).sum(axis=1).pow(0.5).apply(
        lambda s: np.log(max(s, 1e-6))
    ).to_numpy()

    intercept_loc, intercept_scale = svi_regression(t_obs, y_obs, prior)

    last = event_df.iloc[0]  # smallest time_to_tca = latest available CDM
    last_sigma_scale = np.exp(y_obs[0])
    dt_seconds = last["time_to_tca"] * SECONDS_PER_DAY

    rel_pos_last = np.array([last["relative_position_r"], last["relative_position_t"], last["relative_position_n"]])
    rel_vel = np.array([last["relative_velocity_r"], last["relative_velocity_t"], last["relative_velocity_n"]])
    rel_pos_tca = rel_pos_last + rel_vel * dt_seconds

    intercept_draws = rng.normal(intercept_loc, max(intercept_scale, 1e-6), size=N_POSTERIOR_DRAWS)
    pc_draws = []
    for intercept in intercept_draws:
        ratio = np.exp(intercept) / max(last_sigma_scale, 1e-6)
        try:
            log_pc = compute_risk(
                t_sigma_r=last["t_sigma_r"] * ratio, t_sigma_t=last["t_sigma_t"] * ratio, t_sigma_n=last["t_sigma_n"] * ratio,
                t_ct_r=last["t_ct_r"], t_cn_r=last["t_cn_r"], t_cn_t=last["t_cn_t"],
                c_sigma_r=last["c_sigma_r"] * ratio, c_sigma_t=last["c_sigma_t"] * ratio, c_sigma_n=last["c_sigma_n"] * ratio,
                c_ct_r=last["c_ct_r"], c_cn_r=last["c_cn_r"], c_cn_t=last["c_cn_t"],
                rel_pos_rtn=rel_pos_tca, rel_vel_rtn=rel_vel, c_object_type=last["c_object_type"],
                risk_floor=RISK_FLOOR,
            )
            pc_draws.append(10 ** log_pc)
        except Exception:
            continue

    if not pc_draws:
        return RISK_FLOOR
    mean_pc = float(np.mean(pc_draws))
    return max(np.log10(max(mean_pc, 10 ** RISK_FLOOR)), RISK_FLOOR)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=== Calibrating population prior ===")
    prior = fit_population_prior(TRAIN_CSV)

    print("=== Loading test data ===")
    test_df = pd.read_csv(TEST_CSV)
    test_df = sanitize(test_df)
    filtered = test_df[test_df.time_to_tca >= CUTOFF].dropna(subset=SIGMA_COLS + CORR_COLS + [
        "relative_position_r", "relative_position_t", "relative_position_n",
        "relative_velocity_r", "relative_velocity_t", "relative_velocity_n", "c_object_type",
    ])

    all_event_ids = sorted(test_df["event_id"].unique())
    usable_event_ids = set(filtered["event_id"].unique())
    fallback_ids = sorted(set(all_event_ids) - usable_event_ids)
    print(f"{len(usable_event_ids)}/{len(all_event_ids)} test events usable; "
          f"{len(fallback_ids)} need naive fallback (NaN in required fields after cleaning)")

    print("=== Predicting via per-event Pyro SVI regression + Pc ===")
    rows = []
    t0 = time.time()
    n_done = 0
    for event_id, g in filtered.groupby("event_id"):
        risk = predict_event_risk(g, prior, rng)
        rows.append((event_id, risk))
        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {n_done}/{len(usable_event_ids)} events, {elapsed:.0f}s elapsed, "
                  f"~{elapsed / n_done * len(usable_event_ids):.0f}s total est.")
            sys.stdout.flush()

    print("\n=== Naive fallback for events with unusable data ===")
    print(f"{len(fallback_ids)} test events need fallback: {fallback_ids[:20]}...")
    for event_id in fallback_ids:
        latest = test_df[test_df.event_id == event_id].sort_values("time_to_tca").iloc[0]
        rows.append((event_id, latest["risk"]))

    out = pd.DataFrame(rows, columns=["event_id", "predicted_risk"]).sort_values("event_id").reset_index(drop=True)
    out.to_csv("predictions.csv", index=False)
    print(f"wrote {len(out)} predictions to predictions.csv")
    print(out.describe())
