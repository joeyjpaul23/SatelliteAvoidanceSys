"""
Minimal, runnable illustration of MC Dropout — the mechanism Kessler's
LSTMPredictor uses to turn a plain LSTM into a "Bayesian" one.

Verified: kessler.nn.LSTMPredictor.predict_event(event, num_samples=N) does
the same thing at full scale — N stochastic autoregressive rollouts of the
same event, with dropout left active, returned as an EventDataset. The
mean/std across the N rollouts is the model's prediction + uncertainty.

Run: python bayesian_lstm_demo.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


class TinyBayesianLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(self.drop(h[:, -1]))


def main():
    model = TinyBayesianLSTM()
    model.train()  # keep dropout ACTIVE even during "inference" — this is the whole trick

    x = torch.randn(1, 5, 4)  # one fake 5-timestep, 4-feature sequence
    num_samples = 200
    samples = torch.stack([model(x) for _ in range(num_samples)])

    mean = samples.mean(dim=0).squeeze()
    std = samples.std(dim=0).squeeze()

    print(f"{num_samples} stochastic forward passes on the same input:")
    print("Mean prediction:", mean.tolist())
    print("Std (uncertainty):", std.tolist())
    print()
    print("Untrained weights -> the numbers are meaningless, but the SHAPE")
    print("of the result (one mean + one std per feature) is exactly what")
    print("LSTMPredictor.predict_event() gives you at full scale.")


if __name__ == "__main__":
    main()
