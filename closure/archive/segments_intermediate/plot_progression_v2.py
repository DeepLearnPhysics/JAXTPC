"""
Corrected progression plot with dE waterfall as 6th panel.
Waterfall: x=step, y=dE (keV), color=count. Dead row at bottom, overflow at top.

Usage:
    python3 closure_analysis_full/sweeps/plot_progression_v2.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, 'best_final.npz')
PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']
DEATH_THRESH = 0.012  # MeV


def main():
    print("Loading data...")
    data = np.load(DATA_PATH, allow_pickle=True)

    losses = data['losses']
    active_planes = list(data['active_planes'])
    n_de_hist = int(data['n_de_hist'])
    de_history_steps = data['de_history_steps']
    n_signal_snapshots = int(data['n_signal_snapshots'])
    signal_steps = data['signal_steps']

    # Reconstruct dE history
    de_history = []
    for i in range(n_de_hist):
        alive_de = data[f'de_hist_{i}_alive']
        n_dead = int(data[f'de_hist_{i}_dead'])
        de_history.append((alive_de, n_dead))

    # Load profiles
    truth_profiles = {}
    sim_profiles = {}
    for p in active_planes:
        truth_profiles[p] = data[f'truth_profile_{p}']
        sim_profiles[p] = [data[f'sig_profile_{p}_{j}'] for j in range(n_signal_snapshots)]

    # Q ratios (recompute from losses metadata — approximate from printed)
    # Just use step markers for now

    # =====================================================================
    # Build waterfall data
    # =====================================================================
    all_alive_kev = np.concatenate([h[0] * 1000 for h in de_history])
    p99_kev = np.percentile(all_alive_kev, 99)
    n_bins = 80
    bin_edges_kev = np.linspace(DEATH_THRESH * 1000, p99_kev, n_bins + 1)
    bin_centers_kev = 0.5 * (bin_edges_kev[:-1] + bin_edges_kev[1:])

    # Waterfall: columns = steps, rows = dE bins + dead row + overflow row
    # Layout: row 0 = overflow (top), rows 1..n_bins = dE bins (high to low), row n_bins+1 = dead (bottom)
    n_steps_hist = len(de_history)
    waterfall = np.zeros((n_bins + 2, n_steps_hist))

    for j, (alive_de, n_dead) in enumerate(de_history):
        alive_kev = alive_de * 1000
        hist, _ = np.histogram(alive_kev, bins=bin_edges_kev)
        overflow = int(np.sum(alive_kev > p99_kev))
        waterfall[0, j] = overflow          # top row = overflow
        waterfall[1:n_bins+1, j] = hist[::-1]  # flip so high dE is at top
        waterfall[n_bins+1, j] = n_dead     # bottom row = dead

    # =====================================================================
    # Plot 3x2
    # =====================================================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    # (0,0) Loss
    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.0, alpha=0.7)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Loss', fontsize=15); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # (0,1) dE Waterfall
    ax = axes[0, 1]
    # y-axis: dE from death_thresh to p99, plus dead and overflow bands
    # Use pcolormesh for proper axis mapping
    step_arr = np.array(de_history_steps)

    # Build y-axis: dead band | dE bins | overflow band
    dead_band = 1  # fraction of dE range for dead band
    over_band = 1
    y_edges = np.concatenate([
        [0],  # dead bottom
        [DEATH_THRESH * 1000],  # dead top = dE start
        bin_edges_kev,  # dE bins
        [p99_kev + (p99_kev - DEATH_THRESH * 1000) * 0.05],  # overflow
    ])

    # For imshow: transpose so x=step, y=dE
    # waterfall shape is (n_bins+2, n_steps) — already correct for imshow with y=row
    # But we want y=0 at bottom (dead), y=max at top (overflow)
    # imshow origin='lower' handles this

    # imshow: x=step, y=dE. Data is (n_bins, n_steps), need to show with origin='lower'
    de_data = waterfall[1:n_bins+1, :]  # just dE bins, shape (n_bins, n_steps)
    im = ax.imshow(de_data, aspect='auto', origin='lower',
                   extent=[step_arr[0], step_arr[-1],
                           DEATH_THRESH * 1000, p99_kev],
                   cmap='hot', norm=mcolors.LogNorm(vmin=1, vmax=max(de_data.max(), 2)))

    # Dead count as a separate colored band at the bottom
    dead_counts = waterfall[n_bins+1, :]
    ax2 = ax.twinx()
    ax2.fill_between(step_arr, 0, dead_counts, alpha=0.4, color='cyan', label='Dead count')
    ax2.set_ylabel('Dead segments', fontsize=11, color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan', labelsize=10)
    ax2.set_ylim(0, max(dead_counts) * 1.3 if max(dead_counts) > 0 else 1)

    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('dE (keV)', fontsize=14)
    ax.set_title(f'Segment dE Distribution (99th %ile = {p99_kev:.0f} keV)', fontsize=14)
    cbar = fig.colorbar(im, ax=ax, label='Count', pad=0.12)
    ax.tick_params(labelsize=11)

    # (1,0) West U
    ax = axes[1, 0]
    p = 3
    if p in active_planes:
        t_prof = truth_profiles[p]
        r_prof = sim_profiles[p][-1]
        nz = np.where(t_prof > 0)[0]
        wl, wh = (max(0, nz[0]-10), min(len(t_prof), nz[-1]+11)) if len(nz) > 0 else (0, len(t_prof))
        w = np.arange(wl, wh)
        ax.plot(w, t_prof[wl:wh], 'b-', lw=1.5, label='Truth')
        ax.plot(w, r_prof[wl:wh], 'r--', lw=1.5, label='Recon')
    ax.set_title(f'Signal ({PLANE_NAMES[p]})', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # (1,1) West V
    ax = axes[1, 1]
    p = 4
    if p in active_planes:
        t_prof = truth_profiles[p]
        r_prof = sim_profiles[p][-1]
        nz = np.where(t_prof > 0)[0]
        wl, wh = (max(0, nz[0]-10), min(len(t_prof), nz[-1]+11)) if len(nz) > 0 else (0, len(t_prof))
        w = np.arange(wl, wh)
        ax.plot(w, t_prof[wl:wh], 'b-', lw=1.5, label='Truth')
        ax.plot(w, r_prof[wl:wh], 'r--', lw=1.5, label='Recon')
    ax.set_title(f'Signal ({PLANE_NAMES[p]})', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # (2,0) West Y
    ax = axes[2, 0]
    p = 5
    if p in active_planes:
        t_prof = truth_profiles[p]
        r_prof = sim_profiles[p][-1]
        nz = np.where(t_prof > 0)[0]
        wl, wh = (max(0, nz[0]-10), min(len(t_prof), nz[-1]+11)) if len(nz) > 0 else (0, len(t_prof))
        w = np.arange(wl, wh)
        ax.plot(w, t_prof[wl:wh], 'b-', lw=1.5, label='Truth')
        ax.plot(w, r_prof[wl:wh], 'r--', lw=1.5, label='Recon')
    ax.set_title(f'Signal ({PLANE_NAMES[p]})', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # (2,1) Overflow + dead summary
    ax = axes[2, 1]
    overflow_counts = waterfall[0, :]
    ax.plot(step_arr, dead_counts, 'c-', lw=2, label='Dead segments')
    ax.plot(step_arr, overflow_counts, 'r-', lw=2, label=f'Overflow (>{p99_kev:.0f} keV)')
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Dead & Overflow Segments', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    fig.suptitle(f'Best Final | 50k segs, s=1.0, lr=1.0, reloc 25/1000\n'
                 f'loss={losses[-1]:.6f}',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUT_DIR, 'best_final_progression_v2.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == '__main__':
    main()
