"""
Method 6: physics-informed GAN for data augmentation.

Reproduces the general approach of "Improving Satellite Collision Risk Prediction
via Physics-Informed Generative Adversarial Networks" (2025): per its abstract, a
"stabilized Physics-Informed GAN (P-GAN)... for augmenting imbalanced satellite
conjunction data, utilizing a conditional sequence-to-sequence architecture that
generates high-fidelity, physically consistent time-series of CDMs," with "a
comprehensive physics loss function incorporating twelve orbital mechanics
constraints," used to augment training data so that "training state-of-the-art
models on augmented data significantly improves detection of critical high-risk
events."

Access note: the ResearchGate page and every other source found returned HTTP 403
on fetch. Implementation below is built entirely from the search-result abstract
snippet quoted above -- not the paper's actual architecture, exact constraint list,
or reported results. In particular, "twelve orbital mechanics constraints" could not
be confirmed; three physically-motivated constraints are implemented instead (see
below), chosen for defensibility and speed rather than as a claimed match to the
paper's actual twelve.

Key structural point this design follows even without the full paper: this is a
GAN used for **data augmentation**, not a GAN used as a direct risk predictor. A
plain GAN has no natural mechanism to produce a point risk estimate for a specific
observed test event, so treating it as one would misrepresent both the paper and
the architecture. Instead: train a conditional sequence-to-sequence GAN to generate
synthetic high-risk-like CDM sequences (conditioned on a target final-covariance
scale drawn from real high-risk training events), physics-regularize the generator,
use the generated sequences to augment training data for method 5's residual-
correction CNN (reused as the downstream predictor, since re-deriving a second full
downstream architecture would not change what's being tested here), and predict the
real test set with that augmented-trained model.

High-risk threshold: training events with true final risk > -6 are treated as the
minority/high-risk class to augment. This directly matches the contract's own
official-metric threshold (Pc = 1e-6, i.e. risk = -6) rather than an arbitrary cut.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pc_calculator import compute_risk

CUTOFF = 2.0
SECONDS_PER_DAY = 86400.0
RISK_FLOOR = -30.0
HIGH_RISK_THRESHOLD = -6.0
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
SIGMA_IDX = [FEATURES.index(c) for c in SIGMA_COLS]
SIGMA_SANITY_BOUND = 1e5
N_FEATURES = len(FEATURES)
NOISE_DIM = 16
GAN_EPOCHS = 40
GAN_BATCH = 32
N_SYNTHETIC = 2000
DOWNSTREAM_EPOCHS = 8
DOWNSTREAM_BATCH = 64
TRAIN_CSV = "../../data/extracted/train_data.csv"
TEST_CSV = "../../data/test_data.csv"


def sanitize(df):
    df = df.copy()
    for c in SIGMA_COLS:
        df.loc[df[c].abs() > SIGMA_SANITY_BOUND, c] = np.nan
    return df


def naive_physics_baseline(last_row):
    rel_pos = np.array([last_row["relative_position_r"], last_row["relative_position_t"], last_row["relative_position_n"]])
    sigma_scale = float(np.sqrt(sum(last_row[c] ** 2 for c in SIGMA_COLS)))
    return rel_pos, sigma_scale


def build_sequence(event_df, feat_mean, feat_std):
    seq = event_df[FEATURES].to_numpy(dtype=np.float32)
    seq = (seq - feat_mean) / (feat_std + 1e-8)
    length = min(len(seq), MAX_LEN)
    if len(seq) > MAX_LEN:
        seq = seq[-MAX_LEN:]
    padded = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    padded[:length] = seq[:length]
    return padded


# ---------------------------------------------------------------------------
# Conditional sequence-to-sequence GAN
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Conditioned on a scalar target log(sigma_scale) at TCA + noise. Unrolls
    a GRU decoder for MAX_LEN steps to produce a synthetic CDM feature
    sequence -- the "sequence-to-sequence" structure the source paper
    describes, in a small, fast-to-train form."""

    def __init__(self):
        super().__init__()
        self.init_hidden = nn.Linear(NOISE_DIM + 1, 64)
        self.gru = nn.GRUCell(N_FEATURES, 64)
        self.out = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, N_FEATURES), nn.Tanh())

    def forward(self, noise, condition, seq_len=MAX_LEN):
        h = torch.tanh(self.init_hidden(torch.cat([noise, condition], dim=-1)))
        x = torch.zeros(noise.size(0), N_FEATURES, device=noise.device)
        outputs = []
        for t in range(seq_len):
            h = self.gru(x, h)
            x = self.out(h) * 3.0  # bounded output (stabilized generator, per the source paper's description)
            outputs.append(x)
        return torch.stack(outputs, dim=1)  # (batch, seq_len, n_features)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(N_FEATURES, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq):
        _, h = self.gru(seq)
        return self.head(h.squeeze(0)).squeeze(-1)


def physics_loss(seq):
    """Three physically-motivated constraints, chosen for defensibility given
    the source paper's exact "twelve orbital mechanics constraints" could not
    be confirmed (access blocked):
      1. Non-negativity of variance terms (sigma^2 columns): a covariance
         diagonal must be non-negative -- generated sigma-derived features
         (which are standardized, but the diagonal sigma features themselves
         should stay in a physically plausible positive range pre-standardization)
         are penalized for large negative excursions.
      2. Smoothness: consecutive-timestep feature deltas are penalized for
         implausibly large jumps (real CDM sequences evolve gradually as
         orbit determination improves; real data does not jump discontinuously
         step to step).
      3. Monotone covariance shrinkage: the sigma-scale-relevant channels
         should trend downward as the sequence progresses toward TCA (matches
         the same physical trend method 4's NOTES.md documents and calibrates
         against real data).
    """
    sigma_channels = seq[:, :, SIGMA_IDX]
    neg_penalty = torch.relu(-sigma_channels - 2.0).pow(2).mean()  # standardized units; -2 ~ implausible negative sigma

    deltas = seq[:, 1:, :] - seq[:, :-1, :]
    smoothness_penalty = deltas.pow(2).mean()

    sigma_scale_proxy = sigma_channels.pow(2).sum(dim=-1).sqrt()  # (batch, seq_len)
    trend = sigma_scale_proxy[:, 1:] - sigma_scale_proxy[:, :-1]
    monotonicity_penalty = torch.relu(trend).mean()  # penalize increases toward the sequence's end

    return neg_penalty + 0.1 * smoothness_penalty + monotonicity_penalty


def train_gan(train_sequences, high_risk_targets, feat_mean, feat_std):
    """high_risk_targets: array of standardized log-sigma-scale-at-TCA values
    from real high-risk training events, used as the conditioning distribution
    (i.e. the generator is trained to be able to produce sequences consistent
    with any of these targets, biasing generation toward the minority class)."""
    generator = Generator()
    discriminator = Discriminator()
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    real = torch.tensor(train_sequences)
    n = len(real)
    targets = torch.tensor(high_risk_targets, dtype=torch.float32)

    for epoch in range(GAN_EPOCHS):
        idx = torch.randperm(n)[:GAN_BATCH]
        real_batch = real[idx]
        cond_idx = torch.randint(0, len(targets), (GAN_BATCH,))
        cond = targets[cond_idx].unsqueeze(-1)
        noise = torch.randn(GAN_BATCH, NOISE_DIM)

        # Discriminator step
        opt_d.zero_grad()
        fake_batch = generator(noise, cond).detach()
        d_real = discriminator(real_batch)
        d_fake = discriminator(fake_batch)
        d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        opt_d.step()

        # Generator step
        opt_g.zero_grad()
        fake_batch = generator(noise, cond)
        d_fake = discriminator(fake_batch)
        adv_loss = bce(d_fake, torch.ones_like(d_fake))
        phys_loss = physics_loss(fake_batch)
        g_loss = adv_loss + 0.5 * phys_loss
        g_loss.backward()
        opt_g.step()

        if (epoch + 1) % 10 == 0:
            print(f"  GAN epoch {epoch+1}/{GAN_EPOCHS} | d_loss {d_loss.item():.3f} | "
                  f"g_adv {adv_loss.item():.3f} | g_phys {phys_loss.item():.3f}")

    return generator


# ---------------------------------------------------------------------------
# Downstream predictor (same architecture as method 5, retrained on augmented data)
# ---------------------------------------------------------------------------

class ResidualCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.head(x)


def prepare_training_data(train_csv):
    print("Loading + preparing training data...")
    full = pd.read_csv(train_csv)
    full = sanitize(full)
    filtered = full[full.time_to_tca >= CUTOFF].dropna(subset=FEATURES)

    feat_mean = filtered[FEATURES].mean().to_numpy(dtype=np.float32)
    feat_std = filtered[FEATURES].std().to_numpy(dtype=np.float32)

    sequences, targets, final_risks, final_sigma_scales = [], [], [], []
    for event_id, g in filtered.groupby("event_id"):
        full_event = full[full.event_id == event_id].dropna(subset=SIGMA_COLS)
        if len(full_event) == 0:
            continue
        true_final = full_event.loc[full_event["time_to_tca"].idxmin()]

        g = g.sort_values("time_to_tca", ascending=False)
        last_row = g.iloc[-1]
        pred_pos, naive_sigma_scale = naive_physics_baseline(last_row)

        true_sigma_scale = float(np.sqrt(sum(true_final[c] ** 2 for c in SIGMA_COLS)))
        true_pos = np.array([true_final["relative_position_r"], true_final["relative_position_t"], true_final["relative_position_n"]])

        log_sigma_ratio = np.log(max(true_sigma_scale, 1e-6) / max(naive_sigma_scale, 1e-6))
        dpos = (true_pos - pred_pos) / 1000.0

        sequences.append(build_sequence(g, feat_mean, feat_std))
        targets.append(np.array([log_sigma_ratio, dpos[0], dpos[1], dpos[2]], dtype=np.float32))
        final_risks.append(true_final["risk"])
        final_sigma_scales.append(np.log(max(true_sigma_scale, 1e-6)))

    print(f"  {len(sequences)} training events prepared")
    return (np.stack(sequences), np.stack(targets), np.array(final_risks),
            np.array(final_sigma_scales), feat_mean, feat_std)


def train_downstream(sequences, targets, epochs=DOWNSTREAM_EPOCHS):
    from torch.utils.data import TensorDataset, DataLoader
    model = ResidualCNN(N_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(sequences), torch.tensor(targets)), batch_size=DOWNSTREAM_BATCH, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        print(f"  downstream epoch {epoch+1}/{epochs} | train MSE {total/len(sequences):.4f}")
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

            x = torch.tensor(build_sequence(g, feat_mean, feat_std)).unsqueeze(0)
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
    np.random.seed(0)
    t0 = time.time()

    sequences, targets, final_risks, final_sigma_scales, feat_mean, feat_std = prepare_training_data(TRAIN_CSV)
    print(f"data prep took {time.time()-t0:.0f}s")

    high_risk_mask = final_risks > HIGH_RISK_THRESHOLD
    n_high_risk = high_risk_mask.sum()
    print(f"{n_high_risk}/{len(final_risks)} training events are high-risk (risk > {HIGH_RISK_THRESHOLD}), "
          f"used as the GAN's conditioning distribution")
    high_risk_targets = (final_sigma_scales[high_risk_mask] - feat_mean[SIGMA_IDX[0]]) / (feat_std[SIGMA_IDX[0]] + 1e-8)

    print("=== Training physics-informed conditional GAN ===")
    t0 = time.time()
    generator = train_gan(sequences, high_risk_targets, feat_mean, feat_std)
    print(f"GAN training took {time.time()-t0:.0f}s")

    print(f"=== Generating {N_SYNTHETIC} synthetic high-risk-like sequences ===")
    generator.eval()
    with torch.no_grad():
        cond_idx = np.random.randint(0, len(high_risk_targets), N_SYNTHETIC)
        cond = torch.tensor(high_risk_targets[cond_idx], dtype=torch.float32).unsqueeze(-1)
        noise = torch.randn(N_SYNTHETIC, NOISE_DIM)
        synthetic_seqs = generator(noise, cond).numpy()
        # Targets for synthetic examples: bias the covariance-correction pathway
        # toward the conditioned (high-risk, small-sigma) target; no position
        # ground truth exists for pure synthetic generation, so dpos target is 0.
        synthetic_targets = np.zeros((N_SYNTHETIC, 4), dtype=np.float32)
        naive_log_sigma = synthetic_seqs[:, -1, SIGMA_IDX].astype(np.float64)
        naive_log_sigma = np.log(np.sqrt((naive_log_sigma ** 2).sum(axis=-1)).clip(1e-6))
        synthetic_targets[:, 0] = (high_risk_targets[cond_idx] * feat_std[SIGMA_IDX[0]] + feat_mean[SIGMA_IDX[0]]) - naive_log_sigma

    augmented_sequences = np.concatenate([sequences, synthetic_seqs], axis=0)
    augmented_targets = np.concatenate([targets, synthetic_targets], axis=0)
    print(f"augmented training set: {len(sequences)} real + {N_SYNTHETIC} synthetic = {len(augmented_sequences)}")

    print("=== Training downstream residual-correction CNN on augmented data ===")
    t0 = time.time()
    model = train_downstream(augmented_sequences, augmented_targets)
    print(f"downstream training took {time.time()-t0:.0f}s")

    print("=== Predicting test set ===")
    out, fallback_ids = predict_test_set(model, TEST_CSV, feat_mean, feat_std)
    print(f"{len(fallback_ids)} test events used naive fallback: {fallback_ids[:20]}...")
    out.to_csv("predictions.csv", index=False)
    print(f"wrote {len(out)} predictions to predictions.csv")
    print(out.describe())
