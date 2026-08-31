"""
Method 7: novel transformer/attention-based sequence model.

No published work applies this architecture family to this specific benchmark, per
the contract -- full design freedom, documented here rather than following any
source paper.

Design: treat each event's >=2-day CDM history as a token sequence (one token per
CDM), project each token's raw features to an embedding, add a learned time-aware
encoding derived from time_to_tca (continuous, not a discrete position index --
CDMs are irregularly spaced in time, so a standard integer positional encoding
would be a poor fit), pass through a small self-attention Transformer encoder with
padding masking, mean-pool over the real (non-padded) tokens, and predict log10
risk directly end-to-end -- unlike methods 2/4/5/6, which predict a covariance/
position correction and route it through the shared Pc calculator, this method
predicts the risk value itself, since an end-to-end deep sequence model doing its
own feature learning is the more natural and standard use of this architecture
family, and contrasting it against the physics-routed methods is itself a useful
comparison point for Agent T/A.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CUTOFF = 2.0
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
N_FEATURES = len(FEATURES)
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
EPOCHS = 15
BATCH_SIZE = 64
TRAIN_CSV = "../../data/extracted/train_data.csv"
TEST_CSV = "../../data/test_data.csv"


def sanitize(df):
    df = df.copy()
    for c in SIGMA_COLS:
        df.loc[df[c].abs() > SIGMA_SANITY_BOUND, c] = np.nan
    return df


def build_sequence(event_df, feat_mean, feat_std):
    """Returns (padded_seq[MAX_LEN, N_FEATURES], mask[MAX_LEN] bool, real_length)."""
    seq = event_df[FEATURES].to_numpy(dtype=np.float32)
    time_raw = event_df["time_to_tca"].to_numpy(dtype=np.float32)
    seq = (seq - feat_mean) / (feat_std + 1e-8)
    length = min(len(seq), MAX_LEN)
    if len(seq) > MAX_LEN:
        seq = seq[-MAX_LEN:]
        time_raw = time_raw[-MAX_LEN:]
    padded = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    padded[:length] = seq[:length]
    time_padded = np.zeros(MAX_LEN, dtype=np.float32)
    time_padded[:length] = time_raw[:length]
    mask = np.zeros(MAX_LEN, dtype=bool)  # True = padding (for key_padding_mask)
    mask[length:] = True
    return padded, time_padded, mask, length


class TimeEncoding(nn.Module):
    """time2vec-style continuous time encoding: one linear ("trend") term plus
    sinusoidal ("periodic") terms, since CDMs are irregularly spaced in
    time_to_tca -- a discrete positional index would discard that spacing
    information entirely."""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, t):
        # t: (batch, seq_len, 1)
        trend = self.linear(t)
        periodic = torch.sin(self.periodic(t))
        return torch.cat([trend, periodic], dim=-1)


class TransformerRiskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_embed = nn.Linear(N_FEATURES, D_MODEL)
        self.time_encode = TimeEncoding(D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=128,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(nn.Linear(D_MODEL, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, time_raw, key_padding_mask):
        x = self.feature_embed(seq) + self.time_encode(time_raw.unsqueeze(-1))
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        real_mask = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (x * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def prepare_training_data(train_csv):
    print("Loading + preparing training data...")
    full = pd.read_csv(train_csv)
    full = sanitize(full)
    filtered = full[full.time_to_tca >= CUTOFF].dropna(subset=FEATURES)

    feat_mean = filtered[FEATURES].mean().to_numpy(dtype=np.float32)
    feat_std = filtered[FEATURES].std().to_numpy(dtype=np.float32)

    sequences, time_raws, masks, targets = [], [], [], []
    for event_id, g in filtered.groupby("event_id"):
        full_event = full[full.event_id == event_id].dropna(subset=["risk"])
        if len(full_event) == 0:
            continue
        true_final_risk = full_event.loc[full_event["time_to_tca"].idxmin(), "risk"]

        g = g.sort_values("time_to_tca", ascending=False)
        seq, time_raw, mask, _ = build_sequence(g, feat_mean, feat_std)
        sequences.append(seq)
        time_raws.append(time_raw)
        masks.append(mask)
        targets.append(max(true_final_risk, RISK_FLOOR))

    print(f"  {len(sequences)} training events prepared")
    return (np.stack(sequences), np.stack(time_raws), np.stack(masks),
            np.array(targets, dtype=np.float32), feat_mean, feat_std)


def train_model(sequences, time_raws, masks, targets):
    model = TransformerRiskPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(sequences), torch.tensor(time_raws), torch.tensor(masks), torch.tensor(targets))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for seq, t, mask, y in loader:
            optimizer.zero_grad()
            pred = model(seq, t, mask)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(seq)
        print(f"  epoch {epoch+1}/{EPOCHS} | train MSE {total_loss/len(dataset):.4f}")
    return model


def predict_test_set(model, test_csv, feat_mean, feat_std):
    test_df = pd.read_csv(test_csv)
    test_df = sanitize(test_df)
    filtered = test_df[test_df.time_to_tca >= CUTOFF].dropna(subset=FEATURES)

    all_event_ids = sorted(test_df["event_id"].unique())
    usable_ids = set(filtered["event_id"].unique())
    fallback_ids = sorted(set(all_event_ids) - usable_ids)
    print(f"{len(usable_ids)}/{len(all_event_ids)} test events usable; {len(fallback_ids)} need fallback")

    model.eval()
    rows = []
    with torch.no_grad():
        for event_id, g in filtered.groupby("event_id"):
            g = g.sort_values("time_to_tca", ascending=False)
            seq, time_raw, mask, _ = build_sequence(g, feat_mean, feat_std)
            pred = model(
                torch.tensor(seq).unsqueeze(0),
                torch.tensor(time_raw).unsqueeze(0),
                torch.tensor(mask).unsqueeze(0),
            ).item()
            rows.append((event_id, max(pred, RISK_FLOOR)))

    for event_id in fallback_ids:
        latest = test_df[test_df.event_id == event_id].sort_values("time_to_tca").iloc[0]
        rows.append((event_id, latest["risk"]))

    return pd.DataFrame(rows, columns=["event_id", "predicted_risk"]).sort_values("event_id").reset_index(drop=True), fallback_ids


if __name__ == "__main__":
    torch.manual_seed(0)
    t0 = time.time()

    sequences, time_raws, masks, targets, feat_mean, feat_std = prepare_training_data(TRAIN_CSV)
    print(f"data prep took {time.time()-t0:.0f}s")

    t0 = time.time()
    print("=== Training transformer risk predictor ===")
    model = train_model(sequences, time_raws, masks, targets)
    print(f"training took {time.time()-t0:.0f}s")

    print("=== Predicting test set ===")
    out, fallback_ids = predict_test_set(model, TEST_CSV, feat_mean, feat_std)
    print(f"{len(fallback_ids)} test events used naive fallback: {fallback_ids[:20]}...")
    out.to_csv("predictions.csv", index=False)
    print(f"wrote {len(out)} predictions to predictions.csv")
    print(out.describe())
