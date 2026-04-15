import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    """
    Deterministic policy:  state → action ∈ [-max_action, max_action]
    Tanh final activation squashes to (-1,1); we scale by max_action after.
    Orthogonal init keeps gradient magnitudes stable through depth.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        # Orthogonal init — stable gradients through ReLU stacks
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        # Tiny final-layer weights: policy starts near zero → safe exploration
        final_layer = self.net[-2]
        if isinstance(final_layer, nn.Linear): 
            nn.init.orthogonal_(final_layer.weight, gain=0.01)
 
    def forward(self, state):
        return self.max_action * self.net(state)