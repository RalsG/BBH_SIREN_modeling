import torch
import torch.nn as nn
import math


class MLPBlock(nn.Module):
    """A simple MLP block using a conventional activation (ReLU by default)."""
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = activation

    def forward(self, x):
        return self.act(self.linear(x))


class SHAN(nn.Module):
    """Sin-Hybrid Activated Network (SHAN).

    Architecture:
    - MLP body with conventional activations (e.g. ReLU or Sigmoid)
    - Penultimate head that outputs three scalars per input: A, B, C
    - Final output computed as: A * sin(B * t + C)

    Note: Time should be one of the input features so the network can
    produce A,B,C that depend on the full input including time. The final
    sin uses the same time value (assumed to be the last input column).
    """
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int,
                 out_features=1, activation: str='relu', enforce_positive_B: bool=True):
        super().__init__()
        # Choose activation
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()
        elif activation == 'tanh':
            act = nn.Tanh()
        else:
            act = nn.ReLU()

        layers = []
        # Input layer
        layers.append(MLPBlock(in_features, hidden_features, activation=act))
        # Hidden MLP layers
        for _ in range(hidden_layers - 1):
            layers.append(MLPBlock(hidden_features, hidden_features, activation=act))

        self.body = nn.Sequential(*layers)

        # Penultimate head produces A, B, C
        self.head = nn.Linear(hidden_features, 3)

        # Optionally enforce positive frequency/amplitude as needed
        self.enforce_positive_B = enforce_positive_B
        if enforce_positive_B:
            # Use softplus to ensure B > 0
            self._pos = nn.Softplus()
        else:
            self._pos = None

    def forward(self, x):
        # x shape: (N, D) with time expected as the last column
        if x.ndim != 2:
            raise ValueError('Input x must be 2D: (N, D)')

        # Save time value (assume last column)
        t = x[:, -1].unsqueeze(1)  # shape (N,1)

        h = self.body(x)
        abc = self.head(h)  # shape (N,3)

        # Sliced to ensure correct shape (N,1)
        A = abc[:, 0:1]
        B = abc[:, 1:2]
        C = abc[:, 2:3]

        if self._pos is not None:
            # Make B positive (frequency) and add small epsilon for stability
            B = self._pos(B) + 1e-6

        out = A * torch.sin(B * t + C)
        return out


# Backwards-compatible alias
Siren = SHAN