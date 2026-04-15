import argparse
import time
import os
 
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
import gymnasium as gym


from agents.td3 import TD3Agent
from utils.dashboard import TrainingDashboard
from utils.replay_buffer import ReplayBuffer
from utils.live_render import LiveRenderer
import re
import typing


def evaluate_and_capture(agent, env_id, n_episodes=3, capture_episode=0):
    """
    Run n_episodes with no exploration noise.
    Returns (mean_reward, list_of_rgb_frames_from_capture_episode).
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.unwrapped.metadata["render_fps"] = 10000
    total_reward = 0.0
    capture_frames = []
 
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        frames = []
 
        while not done:
            action = agent.select_action(state, add_noise=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
            if ep == capture_episode:
                frames.append(env.render())
 
        total_reward += ep_reward
        if ep == capture_episode:
            capture_frames = frames
 
    env.close()
    return total_reward / n_episodes, capture_frames
 

 
def train(args):
    # ── Setup ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Environment: {args.env}\n")
 
    env = gym.make(args.env, render_mode="rgb_array") 
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
 
    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    assert isinstance(env.action_space, gym.spaces.Box)
    
    state_dim  = env.observation_space.shape[0]   # 24
    action_dim = env.action_space.shape[0]         # 4
    max_action = float(env.action_space.high[0])   # 1.0
 
    agent  = TD3Agent(state_dim, action_dim, max_action, device=device)
    buffer = ReplayBuffer(state_dim, action_dim, device=device)

    if args.load:
        if not os.path.isfile(args.load):
            raise FileNotFoundError(f"Checkpoint not found: {args.load}")
        agent.load(args.load)
 
    headless = not bool(os.environ.get("DISPLAY"))
    dash = TrainingDashboard(
        action_dim=action_dim,
        save_dir=args.plot_dir,
        headless=headless
    )
 
    # ── State ──────────────────────────────────────────────────────────────
    state, _ = env.reset()
    episode_reward = 0.0
    episode_num    = 0
    last_update_time = time.time()
    updates_since_last = 0
    updates_per_sec    = 0.0
    avg100 = 0.0
 
    print(f"{'Step':>8}  {'Episode':>7}  {'Reward':>9}  {'Avg100':>9}  {'Best':>8}")
    print("─" * 55)

    renderer = LiveRenderer(width=600, height=400, title=f"TD3 — {args.env}")
    renderer.start()

    # ── Resume step counter from checkpoint filename ──────────────────
    start_step = 0
    if args.load:
        match = re.search(r'step(\d+)', args.load)
        if match:
            start_step = int(match.group(1))
        print(f"Resuming from step {start_step:,}")

    warmup = args.warmup if not args.load else 0

    for step in range(start_step + 1, args.max_steps + 1):
        if step % 2 == 0:
            renderer.push(typing.cast(np.ndarray, env.render()), stats={
                "step": step, "episode": episode_num,
                "reward": episode_reward, "avg100": avg100,
            })
        # ── Collect experience ─────────────────────────────────────────────
        if step < start_step + warmup:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, add_noise=True)
 
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
 
        # Store terminated (not truncated) as the done flag so we don't
        # suppress bootstrapping when the episode ends by time limit.
        buffer.add(state, action, reward, next_state, float(terminated))
        state          = next_state
        episode_reward += float(reward)
 
        # ── Episode end ────────────────────────────────────────────────────
        if done:
            dash.log_episode(step, episode_reward)
            episode_num += 1
 
            avg100 = (np.mean(list(dash._recent_rewards))
                      if dash._recent_rewards else episode_reward)
            best   = max(dash.rewards) if dash.rewards else episode_reward
 
            if episode_num % 5 == 0:
                print(f"{step:>8,}  {episode_num:>7,}  "
                      f"{episode_reward:>9.1f}  {avg100:>9.1f}  {best:>8.1f}")
 
            state, _ = env.reset()
            episode_reward = 0.0
 
        # ── Learn ──────────────────────────────────────────────────────────
        if step >= start_step + warmup and len(buffer) >= args.batch_size:
            # Sample a batch of raw actions to log distributions
            with torch.no_grad():
                sb, ab, *_ = buffer.sample(min(256, len(buffer)))
                pred_actions = agent.actor(sb).cpu().numpy()
 
            metrics = agent.update(buffer, args.batch_size)
            dash.log_update(step, metrics, pred_actions)
            updates_since_last += 1
 
        # ── Update speed estimate ──────────────────────────────────────────
        now = time.time()
        if now - last_update_time >= 2.0:
            updates_per_sec    = updates_since_last / (now - last_update_time)
            updates_since_last = 0
            last_update_time   = now
 
        # ── Dashboard refresh ──────────────────────────────────────────────
        if step % args.plot_every == 0:
            dash.update(step, episode_num, len(buffer), int(updates_per_sec))
 
        # ── Eval + render capture ──────────────────────────────────────────
        if step % args.eval_every == 0:
            eval_reward, frames = evaluate_and_capture(agent, args.env)
            dash.log_render_frames(frames)
            dash.update(step, episode_num, len(buffer), int(updates_per_sec))
            snap = dash.save_snapshot(step)
            print(f"  → Eval: {eval_reward:.1f}  |  snapshot: {snap}")
 
        # ── Checkpoint ────────────────────────────────────────────────────
        if step % args.save_every == 0:
            path = os.path.join(args.plot_dir, f"td3_step{step:07d}.pt")
            agent.save(path)
            print(f"  → Saved checkpoint: {path}")
    
    renderer.stop()
    env.close()
    dash.save_snapshot(args.max_steps)
    print("\nTraining complete.")
    return agent, dash
 
 
def parse_args():

    p = argparse.ArgumentParser(description="TD3 from scratch — BipedalWalker")
    p.add_argument("--env",        default="BipedalWalker-v3", help="Gymnasium env ID")
    p.add_argument("--max-steps",  type=int, default=1_000_000)
    p.add_argument("--warmup",     type=int, default=10_000,  help="random steps before learning starts")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=25_000,  help="steps between eval+render captures")
    p.add_argument("--plot-every", type=int, default=2_000,   help="steps between dashboard redraws")
    p.add_argument("--save-every", type=int, default=100_000, help="steps between model checkpoints")
    p.add_argument("--plot-dir",   default="td3_output",      help="directory for plots and checkpoints")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--load",       default=None,              help="path to checkpoint to resume from")

    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
 
    train(args)
 