import torch.nn as nn
import numpy as np
import torch

class Critic(nn.Module):
    """
    Q-network:  (state ‖ action) → scalar Q-value
    Concatenating state and action at the input (vs. summing embeddings)
    lets the network learn cross-feature interactions from layer 1.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=1.0)
 
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))
 