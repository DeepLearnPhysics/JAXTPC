"""
Plot Q-diff percentile progression over training.

Shows 68th, 99th percentile and max of |Q_truth - Q_sim| per voxel over steps.
Also shows Q_ratio and dE_ratio trajectories for comparison.

Usage:
    python3 closure_analysis_full/sweeps/plot_q_progression.py --data best_run_out_50k_2000.npz --name out_50k
"""

import numpy as np
import matplotlib.pyplot as plt
import os, argparse

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

VOXEL_SIZE = 25.0
MIN_DIFF = 0.0001
ALPHA = 0.93
BETA = 0.212
LAR_DENSITY = 1.396


def de_to_Q(de, dx_cm, B_eff):
    return (dx_cm / B_eff) * np.log(ALPHA + B_eff * de / dx_cm)


def voxelize_Q(positions, energies, dx_cm, B_eff, voxel_size, origin):
    Q = de_to_Q(energies, dx_cm, B_eff)
    idx = ((positions - origin) / voxel_size).astype(np.int64)
    keys = idx[:, 0] * 1_000_000 + idx[:, 1] * 1_000 + idx[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    vox_Q = np.zeros(len(unique_keys))
    np.add.at(vox_Q, inverse, Q)
    return dict(zip(unique_keys.tolist(), vox_Q.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    data_path = os.path.join(OUT_DIR, args.data)
    print(f"Loading {data_path}...")
    data = np.load(data_path)

    truth_pos = data['truth_pos']
    truth_de = data['truth_de']
    checkpoint_params = data['checkpoint_params']
    checkpoint_steps = data['checkpoint_steps']
    checkpoint_losses = data['checkpoint_losses']
    n_truth = int(data['n_truth'])
    truth_total_de = float(data['truth_total_de'])

    dx_mm = float(data['dx_mm']) if 'dx_mm' in data else 0.5
    dx_cm = dx_mm / 10.0

    if 'B_eff' in data:
        B_eff = float(data['B_eff'])
    else:
        # Default for modified_box at 500 V/cm
        field_kVcm = 0.5
        B_eff = BETA / LAR_DENSITY / field_kVcm  # 0.3037

    n_frames = len(checkpoint_steps)
    print(f"  {n_frames} frames, dx={dx_mm}mm")

    all_pos_min = np.minimum(truth_pos.min(axis=0),
                              checkpoint_params[:, :, :3].reshape(-1, 3).min(axis=0))
    origin = all_pos_min - VOXEL_SIZE

    # Truth Q voxels (fixed)
    truth_vox = voxelize_Q(truth_pos, truth_de, dx_cm, B_eff, VOXEL_SIZE, origin)
    total_truth_Q = de_to_Q(truth_de, dx_cm, B_eff).sum()

    # Compute per-frame stats
    steps, p68, p99, maxdiff, mean_diff = [], [], [], [], []
    q_ratios, de_ratios, n_voxels = [], [], []

    for i in range(n_frames):
        params = checkpoint_params[i]
        alive = params[:, 3] > 0.001
        sim_de = params[alive, 3]

        # Total Q and dE
        total_sim_Q = de_to_Q(sim_de, dx_cm, B_eff).sum()
        total_sim_dE = params[:, 3].sum()

        sim_vox = voxelize_Q(params[alive, :3], sim_de, dx_cm, B_eff, VOXEL_SIZE, origin)

        all_keys = set(truth_vox.keys()) | set(sim_vox.keys())
        diffs = []
        for k in all_keys:
            t_q = truth_vox.get(k, 0)
            s_q = sim_vox.get(k, 0)
            d = t_q - s_q
            if abs(d) > MIN_DIFF:
                diffs.append(d)

        if not diffs:
            diffs = [0.0]
        da = np.abs(np.array(diffs))

        steps.append(checkpoint_steps[i])
        p68.append(np.percentile(da, 68))
        p99.append(np.percentile(da, 99))
        maxdiff.append(da.max())
        mean_diff.append(da.mean())
        q_ratios.append(total_sim_Q / total_truth_Q)
        de_ratios.append(total_sim_dE / truth_total_de)
        n_voxels.append(len(diffs))

    steps = np.array(steps)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Q-diff percentiles
    ax = axes[0, 0]
    ax.semilogy(steps, p99, 'r-', lw=2, label='99th %ile')
    ax.semilogy(steps, p68, 'b-', lw=2, label='68th %ile')
    ax.semilogy(steps, mean_diff, 'k--', lw=1.5, alpha=0.7, label='Mean')
    ax.semilogy(steps, maxdiff, 'gray', lw=1, alpha=0.4, label='Max')
    ax.set_xlabel('Step', fontsize=13)
    ax.set_ylabel('|Q diff| per voxel', fontsize=13)
    ax.set_title('Q-diff Percentiles Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # Top-right: Q ratio vs dE ratio
    ax = axes[0, 1]
    ax.plot(steps, q_ratios, 'b-', lw=2, label='Q ratio (sim/truth)')
    ax.plot(steps, de_ratios, 'r-', lw=2, label='dE ratio (sim/truth)')
    ax.axhline(1.0, color='green', ls='--', lw=1.5, alpha=0.5, label='Target=1.0')
    ax.set_xlabel('Step', fontsize=13)
    ax.set_ylabel('Ratio', fontsize=13)
    ax.set_title('Q Ratio vs dE Ratio', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # Bottom-left: N active voxels
    ax = axes[1, 0]
    ax.plot(steps, n_voxels, 'k-', lw=2)
    ax.set_xlabel('Step', fontsize=13)
    ax.set_ylabel('Active Q-diff Voxels', fontsize=13)
    ax.set_title('Voxel Count (non-zero Q diff)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # Bottom-right: Loss
    ax = axes[1, 1]
    ax.semilogy(steps, checkpoint_losses, 'b-', lw=2)
    ax.set_xlabel('Step', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Sobolev Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    fig.suptitle(f'{args.name} — Q-space Progression ({VOXEL_SIZE:.0f}mm voxels, dx={dx_mm}mm)',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'q_progression_{args.name}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == '__main__':
    main()
