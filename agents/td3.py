 
import torch
from agents.actor import Actor
from agents.critic import Critic
import numpy as np
import torch.nn.functional as F
from agents.forward_model import ForwardModel



class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient.
 
    Three core tricks:
      Trick 1 — Twin critics:        use min(Q1, Q2) as target → kills overestimation
      Trick 2 — Delayed actor:       update actor every policy_freq critic steps
      Trick 3 — Target smoothing:    add clipped noise to target actions during update
    """
    def __init__(       
        
        self, state_dim, action_dim, max_action, device="cpu",
        lr_actor=3e-4, lr_critic=3e-4,
        gamma=0.99,           # discount — 0.99 ≈ cares about ~100 steps ahead
        tau=0.005,            # Polyak: target nets move 0.5% per update
        policy_noise=0.2,     # std of noise on target actions (trick 3)
        noise_clip=0.5,       # clip that noise so it doesn't blow up
        policy_freq=2,        # actor update period (trick 2)
        exploration_noise=0.1 # noise added during env interaction
    ):
        self.device            = device
        self.max_action        = max_action
        self.gamma             = gamma
        self.tau               = tau
        self.policy_noise      = policy_noise
        self.noise_clip        = noise_clip
        self.policy_freq       = policy_freq
        self.exploration_noise = exploration_noise
        self.total_updates     = 0
 
        # Online networks
        self.actor   = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)

        self.forward_model = ForwardModel(state_dim, action_dim).to(device)
        self.forward_opt   = torch.optim.Adam(
            self.forward_model.parameters(), lr=3e-4)
        self.fork_lambda   = 0.3   # tune this — 0.1 to 0.5
 
        # Target networks — start as exact copies, updated via Polyak
        self.actor_target   = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
 
        # Separate optimisers: actor and critics update at different frequencies
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
 
    # ── Action selection ──────────────────────────────────────────────────
    def select_action(self, state, add_noise=True):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
        if add_noise:
            noise  = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        return action
 
    # ── One update step ───────────────────────────────────────────────────
    def update(self, replay_buffer, batch_size=256):
        self.total_updates += 1
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
 
        with torch.no_grad():
            # Trick 3: noisy target actions
            noise = (torch.randn_like(actions) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise
                            ).clamp(-self.max_action, self.max_action)
 
            # Trick 1: pessimistic Q target
            q1_next  = self.critic1_target(next_states, next_actions)
            q2_next  = self.critic2_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(q1_next, q2_next)
 
        # Critic 1 update
        current_q1   = self.critic1(states, actions)
        td_error      = (current_q1 - target_q).abs().mean().item()   # for logging
        critic1_loss  = F.mse_loss(current_q1, target_q)
        self.critic1_opt.zero_grad(); critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_opt.step()
 
        # Critic 2 update
        current_q2  = self.critic2(states, actions)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_opt.zero_grad(); critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_opt.step()
 
        actor_loss_val = None


        # delayed actor update with forward looping knowledge(FORK)
        if self.total_updates % self.policy_freq == 0:

            # --- Forward model update (supervised, every actor step) ---
            predicted_next = self.forward_model(states, actions)
            forward_loss   = F.mse_loss(predicted_next, next_states)
            self.forward_opt.zero_grad()
            forward_loss.backward()
            self.forward_opt.step()

            # --- FORK actor update ---
            current_actions   = self.actor(states)
            predicted_next_s  = self.forward_model(states, current_actions)
            # stop gradient through forward model into actor —
            # we only want the actor to optimise its own weights
            predicted_next_s  = predicted_next_s.detach()

            future_actions    = self.actor(predicted_next_s)
            q_current         = self.critic1(states, current_actions)
            q_future          = self.critic1(predicted_next_s, future_actions)

            # FORK loss: current value + weighted future value
            actor_loss = -(q_current + self.fork_lambda * q_future).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            # Polyak updates — unchanged
            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss":   actor_loss_val,
            "q1_mean":      current_q1.mean().item(),
            "q2_mean":      current_q2.mean().item(),
            "td_error":     td_error,
        }
 
    def _soft_update(self, online, target):
        for op, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)
 
    def save(self, path):
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic1":      self.critic1.state_dict(),
            "critic2":      self.critic2.state_dict(),
            
            "actor_opt":    self.actor_opt.state_dict(),
            "critic1_opt":  self.critic1_opt.state_dict(),
            "critic2_opt":  self.critic2_opt.state_dict(),
            "total_updates": self.total_updates,   
            "forward_model": self.forward_model.state_dict(),
            "forward_opt":   self.forward_opt.state_dict(),
     
        }, path)
 
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # ← add these (with .get() fallback for old checkpoints that don't have them)
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
            self.critic2_opt.load_state_dict(ckpt["critic2_opt"])
        if "total_updates" in ckpt:
            self.total_updates = ckpt["total_updates"]
        if "forward_model" in ckpt:
            self.forward_model.load_state_dict(ckpt["forward_model"])
            self.forward_opt.load_state_dict(ckpt["forward_opt"])

        print(f"Loaded from {path}  (updates so far: {self.total_updates})")
    