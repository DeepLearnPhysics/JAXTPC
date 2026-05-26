"""
Event display animation: truth / sim / diff for east and west sides.

Top row: truth wire-summed profiles (fixed)
Middle row: sim profiles (evolving)
Bottom row: diff (truth - sim)

One GIF per side (east, west). Each frame = one signal checkpoint.

Usage:
    python3 closure_analysis_full/sweeps/render_event_display.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, 'best_final.npz')
FPS = 10  # slower for event display — want to see detail
DPI = 120
PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']


def main():
    print("Loading data...")
    data = np.load(DATA_PATH, allow_pickle=True)

    active_planes = list(data['active_planes'])
    signal_steps = data['signal_steps']
    n_snapshots = int(data['n_signal_snapshots'])
    losses = data['losses']

    # Load truth and sim profiles
    truth_profiles = {}
    sim_profiles = {}
    for p in active_planes:
        truth_profiles[p] = data[f'truth_profile_{p}']
        sim_profiles[p] = [data[f'sig_profile_{p}_{j}'] for j in range(n_snapshots)]

    # Sides: east = planes 0,1,2; west = planes 3,4,5
    sides = {
        'east': [p for p in active_planes if p < 3],
        'west': [p for p in active_planes if p >= 3],
    }

    for side_name, planes in sides.items():
        if not planes:
            print(f"  No active planes for {side_name}, skipping")
            continue

        print(f"Rendering {side_name} event display ({len(planes)} planes, {n_snapshots} frames)...")
        n_planes = len(planes)
        fig, axes = plt.subplots(3, n_planes, figsize=(6 * n_planes, 12))
        if n_planes == 1:
            axes = axes[:, np.newaxis]

        # Compute fixed y-limits per plane
        ylims = {}
        for col, p in enumerate(planes):
            t_prof = truth_profiles[p]
            all_sim = np.array(sim_profiles[p])
            ymax = max(t_prof.max(), all_sim.max()) * 1.1
            # Wire range (zoom to active)
            nz = np.where(t_prof > 0)[0]
            if len(nz) > 0:
                wl, wh = max(0, nz[0] - 10), min(len(t_prof), nz[-1] + 11)
            else:
                wl, wh = 0, len(t_prof)
            ylims[p] = (wl, wh, ymax)

        # Compute fixed diff range
        diff_max = 0
        for p in planes:
            for j in range(n_snapshots):
                wl, wh, _ = ylims[p]
                diff = truth_profiles[p][wl:wh] - sim_profiles[p][j][wl:wh]
                diff_max = max(diff_max, np.abs(diff).max())
        diff_max *= 1.1

        lines_truth = {}
        lines_sim = {}
        lines_diff = {}

        for col, p in enumerate(planes):
            wl, wh, ymax = ylims[p]
            wires = np.arange(wl, wh)
            t_prof = truth_profiles[p][wl:wh]

            # Truth (top row) — static
            ax = axes[0, col]
            ax.plot(wires, t_prof, 'b-', lw=1.5)
            ax.set_ylim(0, ymax)
            ax.set_title(f'Truth {PLANE_NAMES[p]}', fontsize=12)
            ax.set_ylabel('|Signal|', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Sim (middle row) — animated
            ax = axes[1, col]
            line, = ax.plot(wires, t_prof * 0, 'r-', lw=1.5)
            ax.set_ylim(0, ymax)
            ax.set_title(f'Sim {PLANE_NAMES[p]}', fontsize=12)
            ax.set_ylabel('|Signal|', fontsize=10)
            ax.grid(True, alpha=0.3)
            lines_sim[p] = (line, wl, wh)

            # Diff (bottom row) — animated
            ax = axes[2, col]
            line, = ax.plot(wires, t_prof * 0, 'k-', lw=1.5)
            ax.set_ylim(-diff_max, diff_max)
            ax.axhline(0, color='gray', ls='-', lw=0.5)
            ax.set_title(f'Diff {PLANE_NAMES[p]}', fontsize=12)
            ax.set_xlabel('Wire', fontsize=10)
            ax.set_ylabel('Truth - Sim', fontsize=10)
            ax.grid(True, alpha=0.3)
            lines_diff[p] = (line, wl, wh)

        title = fig.suptitle('', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        def update(frame_idx):
            step = signal_steps[frame_idx]
            loss = losses[min(step, len(losses)-1)]
            title.set_text(f'{side_name.upper()} | Step {step} | Loss: {loss:.6f}')
            for p in planes:
                wl, wh = ylims[p][0], ylims[p][1]
                s_prof = sim_profiles[p][frame_idx][wl:wh]
                t_prof = truth_profiles[p][wl:wh]
                lines_sim[p][0].set_ydata(s_prof)
                lines_diff[p][0].set_ydata(t_prof - s_prof)

        anim = animation.FuncAnimation(fig, update, frames=n_snapshots,
                                        blit=False, interval=1000/FPS)
        out_path = os.path.join(OUT_DIR, f'event_display_{side_name}.gif')
        writer = animation.PillowWriter(fps=FPS)
        anim.save(out_path, writer=writer, dpi=DPI)
        plt.close(fig)
        print(f"  Saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
