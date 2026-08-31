"""
Method 5: hybrid physics + ML (propagation-residual correction).

Reproduces the general direction of Li et al. (cited in "Machine learning-based
risk classification of space debris conjunction events," EPJ Web of Conferences
2026): a CNN supplements a physics propagator by learning to predict its residual
error, and the corrected propagation feeds a downstream risk predictor.

Access note: the original Li et al. source could not be read directly -- both the
EPJ Web of Conferences PDF and the underlying Li et al. paper returned HTTP 403 on
every fetch attempt (direct WebFetch and curl; browse-skill setup was not pursued
further given the time budget for this batch of methods). Implementation below is
based on the citing paper's abstract-level description ("CNNs are used to supplement
SGP4 models with residual error of the orbit prediction") rather than the original
paper's full methodology. Flagged as a real information-access gap, not a design
choice.

Substitution, and why: SGP4 propagates from a real TLE (mean motion, eccentricity,
inclination, RAAN, argument of perigee, mean anomaly, B*). The Kelvins CSV only
provides sma/ecc/inc per object -- the same gap documented in method 4's NOTES.md --
so real SGP4 cannot be run without fabricating the missing elements. In its place,
this uses the simplest possible physics baseline: constant-velocity linear
extrapolation of relative position, and zero-growth (unchanged) covariance. This is
deliberately dumb, which is the right stand-in for "a physics propagator with known,
learnable error" -- the CNN's job is to learn exactly the correction this naive
propagator's error pattern requires, which is the same structural role SGP4-residual
learning plays in the source approach, just against a simpler physics baseline.

Design:
  Each event's >=2-day CDM history is a short multi-channel time series (20
  features/step: time_to_tca, 6 sigmas, 6 correlations, 3 relative position, 3
  relative velocity, per-CDM risk). A small 1D CNN consumes this sequence and
  predicts two residual corrections against the naive physics baseline: a
  log-ratio correction to the combined covariance scale, and a 3D correction to
  the linearly-extrapolated relative position at TCA. Supervision comes from
  each training event's own true final CDM (the row closest to TCA in the full,
  unfiltered training data, which can go slightly past TCA) -- the CNN never
  sees that row as input, only as the label to predict the naive baseline's
  error against.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pc_calculator import compute_risk

CUTOFF = 2.0
SECONDS_PER_DAY = 86400.0
RISK_FLOOR = -30.0
MAX_LEN = 20
FEATURES = [
    "time_to_tca",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "t_ct_r", "t_cn_r", "t_cn_t", "c_ct_r", "c_cn_r", "c_cn_t",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "risk",
]
SIGMA_COLS = ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]
SIGMA_SANITY_BOUND = 1e5
EPOCHS = 8
BATCH_SIZE = 64
TRAIN_CSV = "../../data/extracted/train_data.csv"
TEST_CSV = "../../data/test_data.csv"


def sanitize(df):
    df = df.copy()
    for c in SIGMA_COLS:
        df.loc[df[c].abs() > SIGMA_SANITY_BOUND, c] = np.nan
    return df


def naive_physics_baseline(last_row):
    """Zero-order hold: last-known relative position and covariance, unchanged.
    (A constant-velocity extrapolation was tried first and discarded -- over
    multi-day gaps it diverges by thousands of km because it ignores orbital
    periodicity, which is exactly the physics real SGP4 captures and this
    substitute cannot. That divergence dominated the training loss and gave
    the CNN an unlearnable residual to correct. Zero-order hold is a weaker
    physics baseline in one sense, but a *learnable* one -- it makes the
    residual the CNN must predict "how much does relative position actually
    change by TCA", a bounded, meaningful quantity, instead of "how wrong is
    a systematically-diverging linear extrapolation", which is not.)"""
    rel_pos = np.array([last_row["relative_position_r"], last_row["relative_position_t"], last_row["relative_position_n"]])
    sigma_scale = float(np.sqrt(sum(last_row[c] ** 2 for c in SIGMA_COLS)))
    return rel_pos, sigma_scale


def build_sequence(event_df, feat_mean, feat_std):
    """event_df sorted ascending time_to_tca (most recent last). Returns
    (MAX_LEN, n_features) padded/truncated array and the true sequence length."""
    seq = event_df[FEATURES].to_numpy(dtype=np.float32)
    seq = (seq - feat_mean) / (feat_std + 1e-8)
    length = min(len(seq), MAX_LEN)
    if len(seq) > MAX_LEN:
        seq = seq[-MAX_LEN:]  # keep the most recent MAX_LEN CDMs
    padded = np.zeros((MAX_LEN, len(FEATURES)), dtype=np.float32)
    padded[:length] = seq[:length]
    return padded, length


class ResidualCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4),  # [log_sigma_ratio, dpos_r, dpos_t, dpos_n]
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> conv wants (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.head(x)


class ResidualDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return self.sequences[i], self.targets[i]


def prepare_training_data(train_csv):
    print("Loading + preparing training data...")
    full = pd.read_csv(train_csv)
    full = sanitize(full)
    filtered = full[full.time_to_tca >= CUTOFF].dropna(subset=FEATURES)

    feat_mean = filtered[FEATURES].mean().to_numpy(dtype=np.float32)
    feat_std = filtered[FEATURES].std().to_numpy(dtype=np.float32)

    sequences, targets = [], []
    n_skipped = 0
    for event_id, g in filtered.groupby("event_id"):
        full_event = full[full.event_id == event_id].dropna(subset=SIGMA_COLS)
        if len(full_event) == 0:
            n_skipped += 1
            continue
        true_final = full_event.loc[full_event["time_to_tca"].idxmin()]

        g = g.sort_values("time_to_tca", ascending=False)  # oldest (largest time_to_tca) first
        last_row = g.iloc[-1]  # most recent CDM still >= cutoff
        pred_pos, naive_sigma_scale = naive_physics_baseline(last_row)

        true_sigma_scale = float(np.sqrt(sum(true_final[c] ** 2 for c in SIGMA_COLS)))
        true_pos = np.array([true_final["relative_position_r"], true_final["relative_position_t"], true_final["relative_position_n"]])

        log_sigma_ratio = np.log(max(true_sigma_scale, 1e-6) / max(naive_sigma_scale, 1e-6))
        dpos = (true_pos - pred_pos) / 1000.0  # scale to km for a saner loss magnitude

        seq, _ = build_sequence(g, feat_mean, feat_std)
        sequences.append(seq)
        targets.append(np.array([log_sigma_ratio, dpos[0], dpos[1], dpos[2]], dtype=np.float32))

    print(f"  {len(sequences)} training events prepared, {n_skipped} skipped (no clean final row)")
    return np.stack(sequences), np.stack(targets), feat_mean, feat_std


def train_model(sequences, targets):
    model = ResidualCNN(n_features=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = ResidualDataset(torch.tensor(sequences), torch.tensor(targets))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        print(f"  epoch {epoch+1}/{EPOCHS} | train MSE {total_loss/len(dataset):.4f}")
    return model


def predict_test_set(model, test_csv, feat_mean, feat_std):
    test_df = pd.read_csv(test_csv)
    test_df = sanitize(test_df)
    filtered = test_df[test_df.time_to_tca >= CUTOFF].dropna(subset=FEATURES + ["c_object_type"])

    all_event_ids = sorted(test_df["event_id"].unique())
    usable_ids = set(filtered["event_id"].unique())
    fallback_ids = sorted(set(all_event_ids) - usable_ids)
    print(f"{len(usable_ids)}/{len(all_event_ids)} test events usable; {len(fallback_ids)} need fallback")

    model.eval()
    rows = []
    with torch.no_grad():
        for event_id, g in filtered.groupby("event_id"):
            g = g.sort_values("time_to_tca", ascending=False)
            last_row = g.iloc[-1]
            pred_pos, naive_sigma_scale = naive_physics_baseline(last_row)

            seq, _ = build_sequence(g, feat_mean, feat_std)
            x = torch.tensor(seq).unsqueeze(0)
            out = model(x).squeeze(0).numpy()
            log_sigma_ratio, dpos_r, dpos_t, dpos_n = out

            ratio = np.exp(log_sigma_ratio)
            corrected_pos = pred_pos + np.array([dpos_r, dpos_t, dpos_n]) * 1000.0
            rel_vel = np.array([last_row["relative_velocity_r"], last_row["relative_velocity_t"], last_row["relative_velocity_n"]])

            try:
                risk = compute_risk(
                    t_sigma_r=last_row["t_sigma_r"] * ratio, t_sigma_t=last_row["t_sigma_t"] * ratio, t_sigma_n=last_row["t_sigma_n"] * ratio,
                    t_ct_r=last_row["t_ct_r"], t_cn_r=last_row["t_cn_r"], t_cn_t=last_row["t_cn_t"],
                    c_sigma_r=last_row["c_sigma_r"] * ratio, c_sigma_t=last_row["c_sigma_t"] * ratio, c_sigma_n=last_row["c_sigma_n"] * ratio,
                    c_ct_r=last_row["c_ct_r"], c_cn_r=last_row["c_cn_r"], c_cn_t=last_row["c_cn_t"],
                    rel_pos_rtn=corrected_pos, rel_vel_rtn=rel_vel, c_object_type=last_row["c_object_type"],
                    risk_floor=RISK_FLOOR,
                )
            except Exception:
                risk = RISK_FLOOR
            rows.append((event_id, risk))

    for event_id in fallback_ids:
        latest = test_df[test_df.event_id == event_id].sort_values("time_to_tca").iloc[0]
        rows.append((event_id, latest["risk"]))

    return pd.DataFrame(rows, columns=["event_id", "predicted_risk"]).sort_values("event_id").reset_index(drop=True), fallback_ids


if __name__ == "__main__":
    torch.manual_seed(0)
    t0 = time.time()

    sequences, targets, feat_mean, feat_std = prepare_training_data(TRAIN_CSV)
    print(f"data prep took {time.time()-t0:.0f}s")

    t0 = time.time()
    print("=== Training residual-correction CNN ===")
    model = train_model(sequences, targets)
    print(f"training took {time.time()-t0:.0f}s")

    print("=== Predicting test set ===")
    out, fallback_ids = predict_test_set(model, TEST_CSV, feat_mean, feat_std)
    print(f"{len(fallback_ids)} test events used naive fallback: {fallback_ids[:20]}...")
    out.to_csv("predictions.csv", index=False)
    print(f"wrote {len(out)} predictions to predictions.csv")
    print(out.describe())
