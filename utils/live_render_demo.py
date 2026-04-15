"""
Live renders the train environment
"""

from live_render import LiveRenderer
import gymnasium as gym
from collections import deque

def train_normal(args):
    env = gym.make(args.env, render_mode="rgb_array")

    renderer = LiveRenderer(
        width=600, height=400,
        title=f"TD3 — {args.env}",
        show_hud=True,
        hardcore=False,
    )
    renderer.start()
    
    state, _ = env.reset()
    episode_reward = 0.0
    episode_num = 0         

    for step in range(1, args.max_steps + 1):
        if step % 2 == 0:
            frame = env.render()
            
            assert isinstance(frame, np.ndarray) 
            
            avg100 = 0.0 # Placeholder rolling average
            renderer.push(frame, stats={
                "step":    step,
                "episode": episode_num,
                "reward":  episode_reward,
                "avg100":  avg100,
            })

    renderer.stop()
    env.close()


"""
FOR td3_hardcore.py — same changes, plus airborne/pit flags in stats
"""
def train_hardcore(args):
    env = gym.make("BipedalWalkerHardcore-v3", render_mode="rgb_array")

    renderer = LiveRenderer(
        width=600, height=400,
        title="TD3 Hardcore — BipedalWalkerHardcore-v3",
        show_hud=True,
        hardcore=True,
    )
    renderer.start()
    ep_num = 0      
    ep_reward = 0.0 
    step = 0
    airborne = 0
    pit_ahead = 0
    vel_x = 0
    recent_rewards = deque(maxlen=100)

    
    if step % 2 == 0:
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        
        renderer.push(frame, stats={
            "step":      step,
            "episode":   ep_num,
            "reward":    ep_reward,
            "avg100":    float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "airborne":  airborne,    
            "pit_ahead": pit_ahead,   
            "vel_x":     vel_x,       
        })

    renderer.stop()
    env.close()


import argparse, time
import numpy as np


def demo_renderer():
    """
    Runs a random-action agent for 500 steps just to test the renderer.
    You should see the walker falling repeatedly in the pygame window.
    """
    from live_render import LiveRenderer
    import gymnasium as gym

    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    obs, _ = env.reset()

    renderer = LiveRenderer(
        width=600, height=400,
        title="Renderer demo — random actions",
        show_hud=True,
    )
    renderer.start()
    print("Renderer started. Press Q in the window to quit early.")

    ep_reward = 0.0
    episode = 0

    for step in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += float(reward)
        
        frame = env.render()
        assert isinstance(frame, np.ndarray)

        renderer.push(frame, stats={
            "step":    step,
            "episode": episode,
            "reward":  ep_reward,
            "avg100":  ep_reward,
        })

        if terminated or truncated:
            obs, _ = env.reset()
            ep_reward = 0.0
            episode += 1

        if not renderer.is_alive():
            print("Window closed by user.")
            break

        time.sleep(0.01)   # slow down so you can actually see it

    renderer.stop()
    env.close()
    print(f"Demo done. Frames received: {renderer.frames_recv}, "
          f"dropped: {renderer.frames_drop}")


"""
──────────────────────────────────────────────────────────────────────
KEYBOARD CONTROLS in the pygame window
──────────────────────────────────────────────────────────────────────
  SPACE   pause / resume rendering  (training keeps running)
  Q       close window  (training keeps running — renderer just stops)
  X       close button  (same as Q)

──────────────────────────────────────────────────────────────────────
TUNING TIPS
──────────────────────────────────────────────────────────────────────
  renderer.push() every 2nd step    — good balance of smoothness vs CPU
  renderer.push() every 1st step    — smoothest, ~3-5% CPU overhead
  renderer.push() every 10th step   — barely visible motion, minimal cost

  fps_cap=30   — default, smooth
  fps_cap=15   — saves CPU if training is CPU-bound
  fps_cap=60   — smooth on fast machines, more CPU
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run the standalone demo")
    args = p.parse_args()
    if args.demo:
        demo_renderer()
    else:
        print(__doc__)