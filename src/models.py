"""Score model m_{t, theta}(x): shared trunk with a T-dim head (Sec. 3.1 of
the paper — one representation shared across all treatments)."""

import torch
from torch import nn


class MLPScore(nn.Module):
    def __init__(self, d_in=30, hidden=64, T=15):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.head = nn.Linear(hidden, T)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.double()

    def forward(self, X):
        return self.head(self.trunk(X))
