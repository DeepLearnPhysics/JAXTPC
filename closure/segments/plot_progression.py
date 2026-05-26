"""
Progression plot v3: loss, Q ratio, U/V/Y signals, dE waterfall with dead row.

Waterfall: x=step, y=dE bins with bottom row = dead count, properly labeled.

Usage:
    python3 closure_analysis_full/sweeps/plot_progression_v3.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, 'best_final.npz')
PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']
DEATH_THRESH = 0.012  # MeV
ALPHA_RECOMB = 0.93
W_ION = 23.6e-6


def main():
    print("Loading data...")
    data = np.load(DATA_PATH, allow_pickle=True)

    losses = data['losses']
    active_planes = list(data['active_planes'])
    n_de_hist = int(data['n_de_hist'])
    de_history_steps = data['de_history_steps']
    n_signal_snapshots = int(data['n_signal_snapshots'])
    signal_steps = data['signal_steps']
    truth_total_de = float(data['truth_total_de'])
    B_eff = float(data['B_eff'])
    dx_cm = float(data['dx_mm']) / 10.0

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

    # Compute Q ratio from checkpoint params
    def compute_Q(de_arr):
        return (dx_cm / B_eff) * np.log(np.maximum(ALPHA_RECOMB + B_eff * de_arr / dx_cm, 1.0))

    truth_de_all = data['truth_de']
    truth_total_Q = compute_Q(truth_de_all).sum()

    checkpoint_params = data['checkpoint_params']
    checkpoint_steps = data['checkpoint_steps']
    q_steps, q_vals = [], []
    for i in range(0, len(checkpoint_steps), 10):  # every 10th checkpoint
        p_np = checkpoint_params[i]
        alive = p_np[:, 3] > DEATH_THRESH
        sim_Q = compute_Q(p_np[alive, 3]).sum()
        q_steps.append(checkpoint_steps[i])
        q_vals.append(sim_Q / truth_total_Q)

    # =====================================================================
    # Waterfall data: rows = dE bins + 1 dead row at bottom
    # =====================================================================
    all_alive_kev = np.concatenate([h[0] * 1000 for h in de_history])
    p99_kev = np.percentile(all_alive_kev, 99)
    n_bins = 80
    bin_edges_kev = np.linspace(DEATH_THRESH * 1000, p99_kev, n_bins + 1)

    n_steps_hist = len(de_history)

    # Row 0 = dead (bottom), rows 1..n_bins = dE bins low to high, row n_bins+1 = overflow (top)
    waterfall = np.zeros((n_bins + 2, n_steps_hist))
    for j, (alive_de, n_dead) in enumerate(de_history):
        alive_kev = alive_de * 1000
        hist, _ = np.histogram(alive_kev, bins=bin_edges_kev)
        overflow = int(np.sum(alive_kev > p99_kev))
        waterfall[0, j] = n_dead
        waterfall[1:n_bins+1, j] = hist
        waterfall[n_bins+1, j] = overflow

    # =====================================================================
    # Plot 3x2
    # =====================================================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    # (0,0) Loss
    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.0, alpha=0.7)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Loss', fontsize=15); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # (0,1) Q ratio
    ax = axes[0, 1]
    ax.plot(q_steps, q_vals, 'b-', lw=2, label='Q ratio (sim/truth)')
    ax.axhline(1.0, color='green', ls='--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Q Ratio', fontsize=14)
    ax.set_title('Charge Conservation', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

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

    # (2,1) dE Waterfall
    ax = axes[2, 1]

    # Build y-axis tick positions and labels
    # Row 0 = dead, rows 1..n_bins = dE
    # y extent: 0 to n_bins+1
    step_arr = de_history_steps

    im = ax.imshow(waterfall, aspect='auto', origin='lower',
                   extent=[step_arr[0], step_arr[-1], 0, n_bins + 2],
                   cmap='inferno',
                   norm=mcolors.LogNorm(vmin=1, vmax=max(waterfall.max(), 2)),
                   interpolation='none')

    # Y-axis labels
    de_tick_positions = [0.5]  # dead center
    de_tick_labels = ['Dead']
    for kev_val in [50, 200, 500, 1000]:
        if kev_val <= p99_kev:
            bin_pos = 1 + (kev_val - DEATH_THRESH * 1000) / (p99_kev - DEATH_THRESH * 1000) * n_bins
            de_tick_positions.append(bin_pos)
            de_tick_labels.append(f'{kev_val}')
    de_tick_positions.append(n_bins + 1.5)  # overflow center
    de_tick_labels.append(f'>{p99_kev:.0f}')

    ax.set_yticks(de_tick_positions)
    ax.set_yticklabels(de_tick_labels)
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('dE (keV)', fontsize=14)
    ax.set_title(f'Segment dE Distribution Over Training', fontsize=14)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(im, ax=ax, label='Segment count')

    fig.suptitle(f'Best Final | 50k segs, s=1.0, lr=1.0, reloc 25/1000\n'
                 f'loss={losses[-1]:.6f}, Q_ratio={q_vals[-1]:.3f}',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUT_DIR, 'best_final_progression_v3.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == '__main__':
    main()
