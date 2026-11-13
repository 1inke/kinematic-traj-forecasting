"""Only the new 4770-parameter regressor; never loads F16 or old heads."""
import torch
from torch import nn


class ParameterRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(40, 64), nn.GELU(),
                                    nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        if sum(p.numel() for p in self.parameters()) != 4770:
            raise RuntimeError('parameter contract changed')

    def forward(self, normalized_observed_features):
        return torch.tanh(self.layers(normalized_observed_features))


def window_mse(prediction, target):
    if prediction.shape != target.shape or prediction.shape[1:] != (20, 2):
        raise ValueError('loss must average full windows, 20 slots and 2 parameters')
    return ((prediction-target)**2).mean(dim=(1, 2)).mean()
