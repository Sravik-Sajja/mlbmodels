import torch
from torch import nn

class HitNN(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.layers(x)

class BasesNN(nn.Module):
    def __init__(self, n_input_features, n_classes=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )   
    def forward(self, x):
        return self.layers(x)