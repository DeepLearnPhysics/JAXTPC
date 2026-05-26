"""
Render 3D track convergence GIF from saved optimization history.

Loads optimization_history.npz, re-runs forward passes to get track
positions, and renders track_3d_optimization.gif showing truth vs reco
track converging over optimization steps.

Run from project root:
    python3 closure_analysis_muon/render_optimization_gifs.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from PIL import Image

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments_trig,
    mask_outside_volume,
    build_muon_forward,
)

# =============================================================================
# GIF settings
# =============================================================================

FRAME_EVERY = 5        # every 5th step → 61 frames
GIF_FPS = 10           # playback speed (same total duration)
FINAL_PAUSE_MS = 500   # hold on last frame

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(OUT_DIR, 'optimization_history.npz')


# =============================================================================
# Helpers
# =============================================================================

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image via canvas buffer (no PNG encode)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    return Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).convert('RGB').copy()


def _info_text(step, n_steps, loss, params_phys):
    """Format title string."""
    eff_theta = np.degrees(np.arctan2(params_phys[3], params_phys[4]))
    eff_phi = np.degrees(np.arctan2(params_phys[5], params_phys[6]))
    return (
        f'Step {step:3d}/{n_steps}  |  Loss: {loss:.6f}\n'
        f'x={params_phys[0]:.1f}, y={params_phys[1]:.1f}, '
        f'z={params_phys[2]:.1f} mm  |  '
        f'\u03b8={eff_theta:.1f}\u00b0, \u03c6={eff_phi:.1f}\u00b0  |  '
        f'E={params_phys[7]:.1f} MeV'
    )


# =============================================================================
# 3D Track Renderer
# =============================================================================

class Track3DRenderer:
    """Reusable 3D figure — truth line fixed, update reco line per frame."""

    def __init__(self, truth_start, truth_end, truth_phys, view_lims):
        self.fig = plt.figure(figsize=(10, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Truth: start-to-end dashed line (static)
        self.ax.plot([truth_start[0], truth_end[0]],
                     [truth_start[1], truth_end[1]],
                     [truth_start[2], truth_end[2]],
                     color='green', ls='--', lw=2.5, alpha=0.8, label='Truth')
        self.ax.scatter(*truth_start, c='green', s=120, marker='o',
                        edgecolors='black', linewidths=1.0, zorder=5,
                        label='Truth start')
        self.ax.scatter(*truth_end, c='green', s=80, marker='s',
                        edgecolors='black', linewidths=1.0, zorder=5)

        # Reco: placeholder line (updated each frame)
        self._reco_line, = self.ax.plot([], [], [], color='red', ls='-',
                                         lw=2.5, alpha=0.8, label='Reco')
        self._reco_start = self.ax.scatter([0], [0], [0], c='red', s=120,
                            marker='o', edgecolors='black', linewidths=1.0,
                            zorder=5, label='Reco start')
        self._reco_end = self.ax.scatter([0], [0], [0], c='red', s=80,
                            marker='s', edgecolors='black', linewidths=1.0, zorder=5)

        self.ax.set_xlim(view_lims['x'])
        self.ax.set_ylim(view_lims['y'])
        self.ax.set_zlim(view_lims['z'])
        self.ax.set_xlabel('x (mm)', fontsize=15)
        self.ax.set_ylabel('y (mm)', fontsize=15)
        self.ax.set_zlabel('z (mm)', fontsize=15)
        self.ax.tick_params(labelsize=12)
        self.ax.view_init(elev=20, azim=-55)
        self.ax.legend(loc='upper left', fontsize=14)

        truth_theta = np.degrees(np.arctan2(truth_phys[3], truth_phys[4]))
        truth_phi = np.degrees(np.arctan2(truth_phys[5], truth_phys[6]))
        self._truth_subtitle = (
            f'Truth: x={truth_phys[0]:.0f}, y={truth_phys[1]:.0f}, '
            f'z={truth_phys[2]:.0f}, '
            f'\u03b8={truth_theta:.1f}\u00b0, \u03c6={truth_phi:.1f}\u00b0, '
            f'E={truth_phys[7]:.0f} MeV')

        self._title = self.fig.suptitle('', fontsize=15, y=0.98)
        self.fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=0.88)

    def render(self, reco_start, reco_end, step, n_steps, loss, params_phys):
        self._reco_line.set_data_3d([reco_start[0], reco_end[0]],
                                     [reco_start[1], reco_end[1]],
                                     [reco_start[2], reco_end[2]])
        self._reco_start._offsets3d = ([reco_start[0]], [reco_start[1]], [reco_start[2]])
        self._reco_end._offsets3d = ([reco_end[0]], [reco_end[1]], [reco_end[2]])

        self._title.set_text(
            _info_text(step, n_steps, loss, params_phys) + '\n' +
            self._truth_subtitle)
        return fig_to_image(self.fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RENDER 3D TRACK GIF")
    print("=" * 70)

    # --- Load history ---
    print(f"Loading {HISTORY_FILE}...")
    data = np.load(HISTORY_FILE)
    param_history = data['param_history_phys']   # (N+1, 8)
    loss_history = data['loss_history']           # (N+1,)
    truth_phys = data['truth_phys']               # (8,)
    n_segments = int(data['n_segments'])
    step_size_mm = float(data['step_size_mm'])
    n_steps = int(data['n_steps'])

    print(f"  {n_steps} optimization steps, {len(param_history)} param snapshots")

    # Select frame steps
    frame_steps = list(range(0, n_steps + 1, FRAME_EVERY))
    if frame_steps[-1] != n_steps:
        frame_steps.append(n_steps)
    n_frames = len(frame_steps)
    print(f"  Rendering {n_frames} frames")

    # --- Set up simulator (same approach as previous working version) ---
    print("\nSetting up simulator...", flush=True)
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=n_segments)
    forward = build_muon_forward(sim, n_segments, step_size_mm)

    @jax.jit
    def run_forward(phys_np):
        phys = jnp.array(phys_np, dtype=jnp.float32)
        pos, de = generate_muon_segments_trig(
            phys[7], jnp.array([phys[0], phys[1], phys[2]]),
            phys[3], phys[4], phys[5], phys[6],
            step_size_mm, n_segments, log_T, dedx,
        )
        de = mask_outside_volume(pos, de)
        sigs = forward(pos, de)
        return sigs, pos, de

    print("Compiling forward (single JIT)...", flush=True)
    t0 = time.time()
    truth_sigs, truth_pos, truth_de = run_forward(truth_phys)
    for s in truth_sigs:
        jax.block_until_ready(s)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Extract truth endpoints
    truth_pos_np = np.array(truth_pos)
    truth_de_np = np.array(truth_de)
    truth_active = truth_de_np > 0
    tp = truth_pos_np[truth_active]
    truth_start = tp[0]
    truth_end = tp[-1]

    # --- Collect all track endpoints via forward passes ---
    print(f"\nCollecting track endpoints ({n_frames} forward passes)...", flush=True)
    track_data = []  # (start, end, step, loss, phys)

    for i, step in enumerate(frame_steps):
        phys = param_history[step]
        _, reco_pos, reco_de = run_forward(phys)
        jax.block_until_ready(reco_de)
        reco_pos_np = np.array(reco_pos)
        reco_de_np = np.array(reco_de)
        reco_active = reco_de_np > 0

        if np.any(reco_active):
            rp = reco_pos_np[reco_active]
            rp_start, rp_end = rp[0].copy(), rp[-1].copy()
        else:
            rp_start = reco_pos_np[0].copy()
            rp_end = reco_pos_np[1].copy()

        track_data.append((rp_start, rp_end, step, loss_history[step], phys.copy()))

        if (i + 1) % 50 == 0 or i == 0 or i == n_frames - 1:
            print(f"  {i+1}/{n_frames} (step {step})", flush=True)

    # --- Compute view limits from ALL endpoints ---
    all_pts = [truth_start[None, :], truth_end[None, :]]
    for rp_start, rp_end, _, _, _ in track_data:
        all_pts.append(rp_start[None, :])
        all_pts.append(rp_end[None, :])
    all_pts = np.concatenate(all_pts)
    pad_3d = 200.0
    view_lims = {
        'x': (float(all_pts[:, 0].min() - pad_3d), float(all_pts[:, 0].max() + pad_3d)),
        'y': (float(all_pts[:, 1].min() - pad_3d), float(all_pts[:, 1].max() + pad_3d)),
        'z': (float(all_pts[:, 2].min() - pad_3d), float(all_pts[:, 2].max() + pad_3d)),
    }
    print(f"  View limits: x={view_lims['x']}, y={view_lims['y']}, z={view_lims['z']}")

    # --- Render frames ---
    print(f"\nRendering {n_frames} 3D track frames...", flush=True)
    renderer = Track3DRenderer(truth_start, truth_end, truth_phys, view_lims)
    frames = []

    for i, (rp_start, rp_end, step, loss, phys) in enumerate(track_data):
        t0 = time.time()
        frames.append(renderer.render(rp_start, rp_end, step, n_steps, loss, phys))
        elapsed = time.time() - t0
        if (i + 1) % 50 == 0 or i == 0 or i == n_frames - 1:
            print(f"  Frame {i+1}/{n_frames}: {elapsed:.2f}s", flush=True)

    # --- Save GIF ---
    duration_ms = int(1000 / GIF_FPS)
    durations = [duration_ms] * n_frames
    durations[-1] = FINAL_PAUSE_MS

    track_gif = os.path.join(OUT_DIR, 'track_3d_optimization.gif')
    print(f"Saving {track_gif}...", flush=True)
    frames[0].save(
        track_gif, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=False,
    )
    print(f"  Done ({os.path.getsize(track_gif) / 1e6:.1f} MB)")

    print("\nAll done!")


if __name__ == '__main__':
    main()
