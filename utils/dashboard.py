
import os
from collections import deque
 
import numpy as np
import matplotlib
matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Colour palette — consistent across all panels
C_REWARD    = "#1D9E75"   # teal — primary reward line
C_AVG       = "#085041"   # dark teal — rolling average
C_Q1        = "#534AB7"   # purple — Q1
C_Q2        = "#D85A30"   # coral — Q2
C_CRITIC    = "#534AB7"   # purple — critic loss
C_ACTOR     = "#D85A30"   # coral — actor loss
C_TD        = "#BA7517"   # amber — TD error
C_SOLVED    = "#E24B4A"   # red — solved threshold line
JOINT_COLS  = ["#1D9E75", "#534AB7", "#D85A30", "#BA7517"]  # one per joint
PANEL_BG    = "#F8F8F6"
FIGURE_BG   = "#FFFFFF"
 
 
class TrainingDashboard:
    """
    Six-panel live dashboard.  Call .update() every N steps.
    Call .save_snapshot() to write the current figure to disk.
 
    Panel layout:
        [reward]          [Q-values]       [losses]
        [action dist]     [TD error]       [render strip]
    """
 
    def __init__(self, action_dim=4, save_dir="td3_plots", headless=False):
        self.action_dim = action_dim
        self.save_dir   = save_dir
        self.headless   = headless
        os.makedirs(save_dir, exist_ok=True)
 
        # ── Data stores ────────────────────────────────────────────────────
        self.steps          = []
        self.rewards        = []        # per-episode rewards
        self.reward_steps   = []        # step at which each episode ended
        self.avg_rewards    = []        # rolling 100-ep average at each episode end
        self.q1_history     = []
        self.q2_history     = []
        self.q_steps        = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.actor_losses   = []
        self.loss_steps     = []
        self.td_errors      = []
        self.td_steps       = []
        self.recent_actions = deque(maxlen=2000)  # last N actions for histogram
        self.render_frames  = []                  # frames from latest eval rollout
 
        self._recent_rewards = deque(maxlen=100)
 
        # ── Figure ─────────────────────────────────────────────────────────
        plt.ion()
        self.fig = plt.figure(figsize=(18, 10), facecolor=FIGURE_BG)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("TD3 Training — BipedalWalker")
 
        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            hspace=0.42,
            wspace=0.32,
            left=0.06, right=0.97,
            top=0.91,  bottom=0.08
        )
 
        self.ax_reward  = self.fig.add_subplot(gs[0, 0])
        self.ax_q       = self.fig.add_subplot(gs[0, 1])
        self.ax_loss    = self.fig.add_subplot(gs[0, 2])
        self.ax_action  = self.fig.add_subplot(gs[1, 0])
        self.ax_td      = self.fig.add_subplot(gs[1, 1])
        self.ax_render  = self.fig.add_subplot(gs[1, 2])
 
        self._style_all_axes()
        self._draw_static_labels()
 
        # Stats strip along the bottom
        self.stats_text = self.fig.text(
            0.5, 0.01, "", ha="center", va="bottom",
            fontsize=10, color="#444441",
            fontfamily="monospace"
        )
 
        if not headless:
            plt.show(block=False)
            plt.pause(0.1)
 
    # ── Styling helpers ────────────────────────────────────────────────────
 
    def _style_ax(self, ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#D3D1C7")
        ax.tick_params(labelsize=8, color="#888780", length=3)
        ax.set_title(title, fontsize=10, fontweight="medium",
                     color="#2C2C2A", pad=6)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8, color="#5F5E5A")
        if ylabel: ax.set_ylabel(ylabel, fontsize=8, color="#5F5E5A")
        ax.grid(True, linewidth=0.4, color="#D3D1C7", alpha=0.7)
 
    def _style_all_axes(self):
        self._style_ax(self.ax_reward,  "Episode reward",     "steps", "reward")
        self._style_ax(self.ax_q,       "Q-value estimates",  "steps", "Q mean")
        self._style_ax(self.ax_loss,    "Training losses",    "steps", "loss")
        self._style_ax(self.ax_action,  "Action distribution","action value", "count")
        self._style_ax(self.ax_td,      "TD error",           "steps", "|TD error|")
        self._style_ax(self.ax_render,  "Latest eval episode","","")
        self.ax_render.set_xticks([])
        self.ax_render.set_yticks([])
 
    def _draw_static_labels(self):
        # Solved threshold line on reward panel
        self.ax_reward.axhline(300, color=C_SOLVED, linewidth=1,
                               linestyle="--", alpha=0.6, label="solved (+300)")
        self.ax_reward.legend(fontsize=7, loc="upper left",
                              framealpha=0.7, edgecolor="#D3D1C7")
 
    # ── Data ingestion ─────────────────────────────────────────────────────
 
    def log_episode(self, step, reward):
        """Call at the end of every training episode."""
        self._recent_rewards.append(reward)
        self.reward_steps.append(step)
        self.rewards.append(reward)
        self.avg_rewards.append(np.mean(self._recent_rewards))
 
    def log_update(self, step, metrics, actions_batch):
        """Call after agent.update() with the returned metrics dict."""
        if step not in self.q_steps:
            self.q_steps.append(step)
            self.q1_history.append(metrics["q1_mean"])
            self.q2_history.append(metrics["q2_mean"])
            self.td_steps.append(step)
            self.td_errors.append(metrics["td_error"])
 
        if metrics["actor_loss"] is not None:
            self.loss_steps.append(step)
            self.critic1_losses.append(metrics["critic1_loss"])
            self.critic2_losses.append(metrics["critic2_loss"])
            self.actor_losses.append(metrics["actor_loss"])
 
        # Collect raw actions for histogram
        if actions_batch is not None:
            self.recent_actions.extend(actions_batch.tolist())
 
    def log_render_frames(self, frames):
        """Call with a list of RGB numpy arrays from an eval rollout."""
        self.render_frames = frames
 
    # ── Rendering ──────────────────────────────────────────────────────────
 
    def update(self, step, episode, buffer_size, updates_per_sec=0):
        """Redraw all panels. Call every N training steps."""
        self._draw_reward()
        self._draw_q_values()
        self._draw_losses()
        self._draw_action_dist()
        self._draw_td_error()
        self._draw_render()
        self._draw_stats(step, episode, buffer_size, updates_per_sec)
 
        self.fig.canvas.draw_idle()
        if not self.headless:
            self.fig.canvas.flush_events()
 
    def _smooth(self, data, window=20):
        """Simple moving average for noisy curves."""
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="valid")
 
    def _draw_reward(self):
        ax = self.ax_reward
        ax.cla()
        self._style_ax(ax, "Episode reward", "steps (k)", "reward")
        ax.axhline(300, color=C_SOLVED, linewidth=1,
                   linestyle="--", alpha=0.6, label="solved (+300)")
 
        if not self.rewards:
            return
 
        xs = np.array(self.reward_steps) / 1000   # convert to k-steps
        ys = np.array(self.rewards)
        avgs = np.array(self.avg_rewards)
 
        # Raw rewards — thin, low alpha
        ax.plot(xs, ys, color=C_REWARD, alpha=0.25, linewidth=0.6)
        # Rolling average — thick, prominent
        ax.plot(xs, avgs, color=C_AVG, linewidth=1.8, label="avg100")
 
        # Best reward marker
        best_idx = np.argmax(ys)
        ax.scatter([xs[best_idx]], [ys[best_idx]],
                   color=C_SOLVED, s=30, zorder=5, label=f"best: {ys[best_idx]:.0f}")
 
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7, edgecolor="#D3D1C7")
        ax.set_ylim(bottom=min(-150, ys.min() - 10))
 
    def _draw_q_values(self):
        ax = self.ax_q
        ax.cla()
        self._style_ax(ax, "Q-value estimates", "steps (k)", "Q mean")
 
        if not self.q1_history:
            return
 
        xs = np.array(self.q_steps) / 1000
 
        # Smooth to make trends visible
        win = max(1, len(self.q1_history) // 50)
        q1s = self._smooth(self.q1_history, win)
        q2s = self._smooth(self.q2_history, win)
        xs_s = xs[len(xs) - len(q1s):]
 
        ax.plot(xs_s, q1s, color=C_Q1, linewidth=1.4, label="Q1")
        ax.plot(xs_s, q2s, color=C_Q2, linewidth=1.4, label="Q2", linestyle="--")
 
        # Shade gap between Q1 and Q2 — divergence is a warning sign
        ax.fill_between(xs_s, q1s, q2s, alpha=0.12, color="#888780")
 
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7, edgecolor="#D3D1C7")
 
    def _draw_losses(self):
        ax = self.ax_loss
        ax.cla()
        self._style_ax(ax, "Training losses", "steps (k)", "loss")
 
        if not self.critic1_losses:
            return
 
        xs = np.array(self.loss_steps) / 1000
 
        win = max(1, len(self.critic1_losses) // 50)
        c1 = self._smooth(self.critic1_losses, win)
        c2 = self._smooth(self.critic2_losses, win)
        al = self._smooth([abs(v) for v in self.actor_losses], win)
        xs_s = xs[len(xs) - len(c1):]
 
        ax.plot(xs_s, c1, color=C_CRITIC,         linewidth=1.2, label="critic1")
        ax.plot(xs_s, c2, color=C_CRITIC, alpha=0.5, linewidth=1.0,
                linestyle=":", label="critic2")
        ax.plot(xs_s, al, color=C_ACTOR,  linewidth=1.2, label="|actor|")
 
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7, edgecolor="#D3D1C7")
        ax.set_yscale("log")   # log scale: losses can span orders of magnitude
 
    def _draw_action_dist(self):
        ax = self.ax_action
        ax.cla()
        self._style_ax(ax, "Action distribution\n(last 2000 actions)",
                       "value", "count")
 
        if not self.recent_actions:
            ax.text(0.5, 0.5, "collecting actions...",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888780")
            return
 
        arr = np.array(self.recent_actions)   # shape (N, action_dim)
        joint_names = ["Hip L", "Knee L", "Hip R", "Knee R"]
 
        for j in range(min(self.action_dim, 4)):
            ax.hist(arr[:, j], bins=40, range=(-1.05, 1.05),
                    alpha=0.45, color=JOINT_COLS[j],
                    label=joint_names[j], density=True)
 
        ax.axvline(0, color="#888780", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7,
                  ncol=2, edgecolor="#D3D1C7")
 
        # Saturation warning: if many actions are near ±1, policy may be stuck
        sat_frac = np.mean(np.abs(arr) > 0.95)
        if sat_frac > 0.3:
            ax.text(0.5, 0.94, f"saturation: {sat_frac*100:.0f}%",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=8, color=C_SOLVED,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FCEBEB",
                              edgecolor=C_SOLVED, alpha=0.8))
 
    def _draw_td_error(self):
        ax = self.ax_td
        ax.cla()
        self._style_ax(ax, "TD error  |r + γQ' − Q|", "steps (k)", "magnitude")
 
        if not self.td_errors:
            return
 
        xs = np.array(self.td_steps) / 1000
        ys = np.array(self.td_errors)
 
        win = max(1, len(ys) // 50)
        ys_smooth = self._smooth(ys, win)
        xs_s = xs[len(xs) - len(ys_smooth):]
 
        ax.fill_between(xs_s, 0, ys_smooth, alpha=0.2, color=C_TD)
        ax.plot(xs_s, ys_smooth, color=C_TD, linewidth=1.4)
 
        # Annotate current value
        ax.text(0.98, 0.96, f"now: {ys[-1]:.3f}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=8, color=C_TD)
 
    def _draw_render(self):
        ax = self.ax_render
        ax.cla()
        self._style_ax(ax, "Latest eval episode", "", "")
        ax.set_xticks([]); ax.set_yticks([])
 
        if not self.render_frames:
            ax.text(0.5, 0.5, "eval snapshot\nappears here",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888780")
            return
 
        # Build a horizontal strip of evenly-spaced frames
        frames = self.render_frames
        n_show = min(5, len(frames))
        indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
        strip = np.concatenate([frames[i] for i in indices], axis=1)
 
        ax.imshow(strip, aspect="auto")
        ax.text(0.5, -0.04,
                f"{len(frames)} frames  ·  {n_show} shown",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=7, color="#888780")
 
    def _draw_stats(self, step, episode, buffer_size, ups):
        avg100 = (f"{np.mean(list(self._recent_rewards)):.1f}"
                  if self._recent_rewards else "—")
        best   = (f"{max(self.rewards):.1f}"
                  if self.rewards else "—")
        text = (
            f"step: {step:>8,}   "
            f"episode: {episode:>5,}   "
            f"avg100: {avg100:>8}   "
            f"best: {best:>8}   "
            f"buffer: {buffer_size:>7,}   "
            f"upd/s: {ups:>5.1f}"
        )
        self.stats_text.set_text(text)
 
    def save_snapshot(self, step):
        path = os.path.join(self.save_dir, f"dashboard_step{step:07d}.png")
        self.fig.savefig(path, dpi=120, bbox_inches="tight",
                         facecolor=FIGURE_BG)
        return path
 
 