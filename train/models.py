import torch
from torch import nn


class HitNN(nn.Module):
    """
    Binary classifier: predicts whether a ball in play results in a hit.
    """
    def __init__(self, n_input_features, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


class BasesNN(nn.Module):
    """
    Multi-class classifier: predicts number of bases (0=out, 1=single,
    2=double, 3=triple, 4=home run).
    """
    def __init__(self, n_input_features, n_classes=5, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.layers(x)