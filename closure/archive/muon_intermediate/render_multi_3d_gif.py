"""
Render 3D GIF showing optimization tracks converging to truth.

Loads precomputed endpoints from individual optimization_X.npz files.

Usage:
    python3 closure_analysis_muon/render_multi_3d_gif.py A B D
    python3 closure_analysis_muon/render_multi_3d_gif.py A B C D
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from PIL import Image

# =============================================================================
# Settings
# =============================================================================

FRAME_EVERY = 2
GIF_FPS = 50
FINAL_PAUSE_MS = 500

COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Helpers
# =============================================================================

def fig_to_image(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    return Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).convert('RGB').copy()


# =============================================================================
# Renderer
# =============================================================================

class MultiTrack3DRenderer:
    def __init__(self, truth_start, truth_end, truth_phys, view_lims, labels):
        self.labels = labels
        self.fig = plt.figure(figsize=(10, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Arrow length as fraction of track length
        self._arrow_frac = 0.15

        # Truth track (static) with direction arrow
        truth_dir = truth_end - truth_start
        truth_len = np.linalg.norm(truth_dir)
        truth_arrow_len = truth_len * self._arrow_frac
        truth_unit = truth_dir / max(truth_len, 1e-6)
        arrow_base = truth_end - truth_unit * truth_arrow_len

        self.ax.plot([truth_start[0], arrow_base[0]],
                     [truth_start[1], arrow_base[1]],
                     [truth_start[2], arrow_base[2]],
                     color='black', ls='--', lw=2.5, alpha=0.8, label='Truth')
        self.ax.quiver(arrow_base[0], arrow_base[1], arrow_base[2],
                       truth_unit[0], truth_unit[1], truth_unit[2],
                       length=truth_arrow_len, color='black', alpha=0.8,
                       arrow_length_ratio=0.4, linewidth=2.5)
        self.ax.scatter(*truth_start, c='white', s=140, marker='o',
                        edgecolors='black', linewidths=2.0, zorder=5)

        # Reco tracks (updated each frame)
        self._lines = []
        self._starts = []
        self._arrows = []
        for i, label in enumerate(labels):
            color = COLORS[label]
            line, = self.ax.plot([], [], [], color=color, ls='-',
                                  lw=2.0, alpha=0.85, label=f'Init {i+1}')
            start = self.ax.scatter([0], [0], [0], c='white', s=120,
                        marker='o', edgecolors=color, linewidths=2.0, zorder=5)
            self._lines.append(line)
            self._starts.append(start)

        self.ax.set_xlim(view_lims['x'])
        self.ax.set_ylim(view_lims['y'])
        self.ax.set_zlim(view_lims['z'])
        self.ax.set_xlabel('x (mm)', fontsize=15)
        self.ax.set_ylabel('y (mm)', fontsize=15)
        self.ax.set_zlabel('z (mm)', fontsize=15)
        self.ax.tick_params(labelsize=12)
        self.ax.view_init(elev=20, azim=-55)
        self.ax.legend(loc='upper left', fontsize=12)

        truth_theta = np.degrees(np.arctan2(truth_phys[3], truth_phys[4]))
        truth_phi = np.degrees(np.arctan2(truth_phys[5], truth_phys[6]))
        self._truth_sub = (
            f'Truth: x={truth_phys[0]:.0f}, y={truth_phys[1]:.0f}, '
            f'z={truth_phys[2]:.0f}, '
            f'\u03b8={truth_theta:.1f}\u00b0, \u03c6={truth_phi:.1f}\u00b0, '
            f'E={truth_phys[7]:.0f} MeV')

        self._title = self.fig.suptitle('', fontsize=15, y=0.98)
        self.fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=0.88)

    def render(self, all_starts, all_ends, step, n_steps, losses):
        # Remove previous arrows
        for arrow in self._arrows:
            arrow.remove()
        self._arrows.clear()

        for i in range(len(self.labels)):
            s, e = all_starts[i], all_ends[i]
            d = e - s
            length = np.linalg.norm(d)
            unit = d / max(length, 1e-6)
            arrow_len = length * self._arrow_frac
            arrow_base = e - unit * arrow_len

            self._lines[i].set_data_3d(
                [s[0], arrow_base[0]], [s[1], arrow_base[1]], [s[2], arrow_base[2]])
            self._starts[i]._offsets3d = ([s[0]], [s[1]], [s[2]])

            color = COLORS[self.labels[i]]
            q = self.ax.quiver(arrow_base[0], arrow_base[1], arrow_base[2],
                               unit[0], unit[1], unit[2],
                               length=arrow_len, color=color, alpha=0.85,
                               arrow_length_ratio=0.4, linewidth=2.0)
            self._arrows.append(q)

        loss_str = '  '.join(
            f'Init {i+1}:{losses[i]:.2f}' for i in range(len(self.labels)))
        self._title.set_text(
            f'Step {step:3d}/{n_steps}  |  Loss: {loss_str}\n{self._truth_sub}')
        return fig_to_image(self.fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', nargs='+', help='Run labels to include (e.g. A B D)')
    args = parser.parse_args()

    labels = args.labels
    print(f"Rendering 3D GIF for runs: {labels}")

    # Load individual NPZ files
    runs = {}
    for label in labels:
        path = os.path.join(OUT_DIR, f'optimization_{label}.npz')
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run optimization first.")
            sys.exit(1)
        runs[label] = np.load(path, allow_pickle=True)

    # Use truth from first run (all should be the same)
    first = runs[labels[0]]
    truth_phys = first['truth_phys']
    truth_start = first['truth_start']
    truth_end = first['truth_end']
    n_steps = int(first['n_steps'])

    # Check all runs have same n_steps (use min if different)
    for label in labels:
        ns = int(runs[label]['n_steps'])
        if ns < n_steps:
            n_steps = ns
    print(f"  Using {n_steps} steps")

    loss_histories = [runs[l]['loss_history'] for l in labels]
    ep_starts = [runs[l]['endpoints_starts'] for l in labels]
    ep_ends = [runs[l]['endpoints_ends'] for l in labels]

    # Frame steps
    frame_steps = list(range(0, n_steps + 1, FRAME_EVERY))
    if frame_steps[-1] != n_steps:
        frame_steps.append(n_steps)
    n_frames = len(frame_steps)
    print(f"  {n_frames} frames (every {FRAME_EVERY} steps)")

    # Compute view limits from all endpoints
    all_pts = [truth_start[None, :], truth_end[None, :]]
    for ri in range(len(labels)):
        all_pts.append(ep_starts[ri])
        all_pts.append(ep_ends[ri])
    all_pts = np.concatenate(all_pts)
    pad_3d = 200.0
    view_lims = {
        'x': (float(all_pts[:, 0].min() - pad_3d), float(all_pts[:, 0].max() + pad_3d)),
        'y': (float(all_pts[:, 1].min() - pad_3d), float(all_pts[:, 1].max() + pad_3d)),
        'z': (float(all_pts[:, 2].min() - pad_3d), float(all_pts[:, 2].max() + pad_3d)),
    }
    print(f"  View limits: x={view_lims['x']}, y={view_lims['y']}, z={view_lims['z']}")

    # Render frames
    print(f"\nRendering {n_frames} frames...", flush=True)
    renderer = MultiTrack3DRenderer(truth_start, truth_end, truth_phys, view_lims, labels)
    frames = []

    t0 = time.time()
    for fi, step in enumerate(frame_steps):
        starts = [ep_starts[ri][step] for ri in range(len(labels))]
        ends = [ep_ends[ri][step] for ri in range(len(labels))]
        losses = [loss_histories[ri][step] for ri in range(len(labels))]
        frames.append(renderer.render(starts, ends, step, n_steps, losses))

        if (fi + 1) % 20 == 0 or fi == 0 or fi == n_frames - 1:
            print(f"  Frame {fi+1}/{n_frames} ({time.time()-t0:.1f}s)", flush=True)

    # Save GIF
    duration_ms = int(1000 / GIF_FPS)
    durations = [duration_ms] * n_frames
    durations[-1] = FINAL_PAUSE_MS

    gif_path = os.path.join(OUT_DIR, 'multi_track_3d_optimization.gif')
    print(f"Saving {gif_path}...", flush=True)
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=False,
    )
    print(f"  Done ({os.path.getsize(gif_path) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
