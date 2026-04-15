"""
live_render.py
==================
live render window for TD3 training.

Spawns a background thread that opens a pygame window and displays
environment frames in real time as training runs.  Training thread
is never blocked — frames are dropped if the renderer falls behind.

Usage (add these 4 lines to your training loop):
-------------------------------------------------
    from live_render import LiveRenderer

    renderer = LiveRenderer(width=600, height=400, title="BipedalWalker")
    renderer.start()

    # inside your step loop:
    frame = env.render()          # env must use render_mode="rgb_array"
    renderer.push(frame, stats={  # stats dict is optional
        "step": step,
        "episode": ep_num,
        "reward": episode_reward,
        "avg100": avg100,
        "airborne": airborne,     # hardcore only
    })

    # at the end:
    renderer.stop()

What you see in the window
--------------------------
  - Live environment frame (scales to window size)
  - HUD overlay: step, episode, reward, avg100
  - Airborne indicator bar (red when both feet off ground)
  - FPS counter (render fps, not training fps)
  - Pause/resume with SPACE, quit with Q or close button
"""

import queue
import threading
import time
from typing import Optional
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pygame

try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False
    print("[LiveRenderer] pygame not found — pip install pygame")


# ─────────────────────────────────────────────────────────
#  COLOURS  (all hardcoded — renderer doesn't use theme vars)
# ─────────────────────────────────────────────────────────

COL_BG       = (18,  18,  22)    # near-black background
COL_HUD_BG   = (18,  18,  22,  180)  # semi-transparent HUD strip
COL_TEXT     = (220, 218, 210)   # off-white
COL_MUTED    = (136, 135, 128)   # muted text
COL_TEAL     = ( 29, 158, 117)   # reward positive
COL_CORAL    = (216,  90,  48)   # reward negative / airborne
COL_PURPLE   = ( 83,  74, 183)   # accent
COL_AMBER    = (186, 117,  23)   # warning / pit
COL_GREEN    = ( 99, 153,  34)
COL_RED      = (226,  75,  74)


# ─────────────────────────────────────────────────────────
#  LIVE RENDERER
# ─────────────────────────────────────────────────────────

class LiveRenderer:
    """
    Threaded pygame window.  All public methods are thread-safe.

    Parameters
    ----------
    width, height : int
        Window dimensions in pixels.
    fps_cap : int
        Max render FPS.  Keeps the renderer from burning CPU.
    title : str
        Window title.
    scale : float
        Extra scale factor applied to the env frame before display.
        Use 1.0 to auto-fit to window.
    show_hud : bool
        Whether to draw the stats overlay.
    hardcore : bool
        Enables the airborne indicator bar.
    """

    def __init__(
        self,
        width: int    = 600,
        height: int   = 400,
        fps_cap: int  = 30,
        title: str    = "TD3 — Live Training",
        scale: float  = 1.0,
        show_hud: bool = True,
        hardcore: bool = False,
    ):
        if not _PYGAME_OK:
            raise RuntimeError("pygame is required: pip install pygame")

        self.width    = width
        self.height   = height
        self.fps_cap  = fps_cap
        self.title    = title
        self.scale    = scale
        self.show_hud = show_hud
        self.hardcore = hardcore

        # Thread-safe frame queue.  maxsize=4 means training drops frames
        # rather than blocking when the renderer is slow.
        self._queue   : queue.Queue = queue.Queue(maxsize=4)
        self._thread  : Optional[threading.Thread] = None
        self._stop_evt: threading.Event = threading.Event()
        self._paused  : threading.Event = threading.Event()   # set = paused

        # Latest stats — updated by push(), read by renderer thread
        self._stats: dict = {}
        self._stats_lock  = threading.Lock()

        # Renderer exposes these for external monitoring
        self.render_fps  = 0.0
        self.frames_recv = 0
        self.frames_drop = 0

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        """Start the background render thread.  Call once before training."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._render_loop, daemon=True, name="LiveRenderer")
        self._thread.start()
        # Give pygame a moment to initialise before training hammers the queue
        time.sleep(0.3)

    def stop(self):
        """Gracefully shut down.  Call after training finishes."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def push(self, frame: np.ndarray, stats: Optional[dict] = None):
        """
        Send a new frame to the renderer.  Never blocks — drops if full.

        Parameters
        ----------
        frame : np.ndarray
            RGB uint8 array, shape (H, W, 3).  This is what env.render()
            returns when render_mode="rgb_array".
        stats : dict, optional
            Keys recognised:  step, episode, reward, avg100,
                               airborne, pit_ahead, vel_x
        """
        if stats is not None:
            with self._stats_lock:
                self._stats.update(stats)

        try:
            self._queue.put_nowait(frame)
            self.frames_recv += 1
        except queue.Full:
            self.frames_drop += 1   

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal render loop (runs in background thread) ──────────────────

    def _render_loop(self):
        try:
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.title)
            clock  = pygame.time.Clock()

            font_lg = pygame.font.SysFont("monospace", 15, bold=True)
            font_sm = pygame.font.SysFont("monospace", 12)
            font_xs = pygame.font.SysFont("monospace", 10)

            last_surface: Optional[pygame.Surface] = None
            fps_timer = time.time()
            fps_frames = 0

            while not self._stop_evt.is_set():

                # ── Event handling ─────────────────────────────────────────────
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self._stop_evt.set()
                            break
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                self._stop_evt.set()
                            if event.key == pygame.K_SPACE:
                                if self._paused.is_set():
                                    self._paused.clear()
                                else:
                                    self._paused.set()
                except pygame.error:
                    pass

                if self._stop_evt.is_set():
                    break

                # ── Grab latest frame ──────────────────────────────────────────
                frame = None
                try:
                    while True:
                        frame = self._queue.get_nowait()
                except queue.Empty:
                    pass

                # ── Render frame ───────────────────────────────────────────────
                screen.fill(COL_BG)

                if frame is not None:
                    surface = self._frame_to_surface(frame)
                    last_surface = surface

                if last_surface is not None:
                    scaled = self._fit_surface(last_surface)
                    fx = (self.width  - scaled.get_width())  // 2
                    fy = (self.height - scaled.get_height()) // 2
                    screen.blit(scaled, (fx, fy))

                # ── HUD overlay ────────────────────────────────────────────────
                if self.show_hud:
                    with self._stats_lock:
                        stats = dict(self._stats)
                    self._draw_hud(screen, font_lg, font_sm, font_xs, stats)

                # ── Paused banner ──────────────────────────────────────────────
                if self._paused.is_set():
                    self._draw_banner(screen, font_lg, "PAUSED — press SPACE to resume")

                # ── FPS counter ────────────────────────────────────────────────
                fps_frames += 1
                if time.time() - fps_timer >= 1.0:
                    self.render_fps = fps_frames / (time.time() - fps_timer)
                    fps_frames = 0
                    fps_timer  = time.time()

                fps_txt = font_xs.render(
                    f"render {self.render_fps:.0f} fps  "
                    f"recv {self.frames_recv}  drop {self.frames_drop}",
                    True, COL_MUTED)
                screen.blit(fps_txt, (8, self.height - 18))

                pygame.display.flip()
                clock.tick(self.fps_cap)

        except Exception as e:
            print(f"[LiveRenderer] crashed: {e}")
        finally:
            try:
                pygame.quit()
            except:
                pass

    # ── HUD drawing ───────────────────────────────────────────────────────

    def _draw_hud(self, screen, font_lg, font_sm, font_xs, stats):
        """Draw a compact stats overlay in the top-left corner."""
        if not stats:
            return

        step    = stats.get("step",    0)
        episode = stats.get("episode", 0)
        reward  = stats.get("reward",  None)
        avg100  = stats.get("avg100",  None)
        airborne= stats.get("airborne",False)
        pit     = stats.get("pit_ahead",False)
        vel_x   = stats.get("vel_x",  None)

        lines : list[tuple[str, tuple[int, int, int]]] = [
            (f"step    {step:>9,}",         COL_TEXT),
            (f"episode {episode:>9,}",       COL_TEXT),
        ]
        if reward is not None:
            col = COL_TEAL if reward >= 0 else COL_CORAL
            lines.append((f"reward  {reward:>+9.1f}", col))
        if avg100 is not None:
            col = COL_GREEN if avg100 > 100 else COL_MUTED
            lines.append((f"avg100  {avg100:>+9.1f}", col))
        if vel_x is not None:
            lines.append((f"vel_x   {vel_x:>9.2f}", COL_MUTED))

        # Background pill
        pad = 8
        line_h = 18
        box_h  = pad*2 + len(lines) * line_h + (12 if self.hardcore else 0)
        box_w  = 190
        box_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box_surf.fill((18, 18, 22, 200))
        screen.blit(box_surf, (8, 8))

        for i, (text, col) in enumerate(lines):
            surf = font_sm.render(text, True, col)
            screen.blit(surf, (8 + pad, 8 + pad + i * line_h))

        # Hardcore extras
        if self.hardcore:
            y_bar = 8 + pad + len(lines) * line_h + 4

            # Airborne bar
            bar_w = box_w - pad*2
            pygame.draw.rect(screen, (40, 40, 50), (8+pad, y_bar, bar_w, 8), border_radius=4)
            if airborne:
                pygame.draw.rect(screen, COL_CORAL,
                                 (8+pad, y_bar, bar_w, 8), border_radius=4)
                label = font_xs.render("AIRBORNE", True, COL_CORAL)
            else:
                pygame.draw.rect(screen, COL_TEAL,
                                 (8+pad, y_bar, int(bar_w*0.6), 8), border_radius=4)
                label = font_xs.render("grounded", True, COL_MUTED)
            screen.blit(label, (8+pad, y_bar + 11))

            if pit:
                pit_surf = font_xs.render("PIT AHEAD", True, COL_AMBER)
                screen.blit(pit_surf, (8 + pad + 90, y_bar + 11))

    def _draw_banner(self, screen, font, text):
        surf = font.render(text, True, COL_AMBER)
        x = (self.width  - surf.get_width())  // 2
        y = (self.height - surf.get_height()) // 2
        bg = pygame.Surface((surf.get_width()+24, surf.get_height()+12), pygame.SRCALPHA)
        bg.fill((18, 18, 22, 220))
        screen.blit(bg, (x-12, y-6))
        screen.blit(surf, (x, y))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _frame_to_surface(self, frame: np.ndarray) -> pygame.Surface:
        """Convert an RGB numpy array to a pygame Surface."""
        # pygame expects (W, H, 3) with axis swapped
        return pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2)))

    def _fit_surface(self, surf: pygame.Surface) -> pygame.Surface:
        """Scale surface to fit window while preserving aspect ratio."""
        sw, sh = surf.get_size()
        if self.scale != 1.0:
            sw = int(sw * self.scale)
            sh = int(sh * self.scale)
        scale_w = self.width  / sw
        scale_h = self.height / sh
        s = min(scale_w, scale_h, 1.5)   # cap upscaling at 1.5×
        nw, nh = int(sw * s), int(sh * s)
        if nw != surf.get_width() or nh != surf.get_height():
            return pygame.transform.smoothscale(surf, (nw, nh))
        return surf
    
