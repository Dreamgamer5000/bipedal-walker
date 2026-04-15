import torch
import torch.nn as nn
import numpy as np

class ForwardModel(nn.Module):
    """
    Predicts next state given (state, action).
    Trained supervised on real transitions from the replay buffer.
    Loss: MSE between predicted s' and real s'.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),              
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.orthogonal_(l.weight, gain=np.sqrt(2))
                nn.init.zeros_(l.bias)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))