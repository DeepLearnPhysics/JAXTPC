"""
3D annihilation plot using CHARGE (Q) difference instead of dE.

Renders for both the 20k mpvmpr run (s=1.5) and the 50k out.h5 run (s=1.0).
Pass --data to select which checkpoint file to use.

Usage:
    python3 closure/segments/render_3d.py --data closure_analysis_full/sweeps/best_run_20k_4000.npz --name 20k_s1.5
    python3 closure/segments/render_3d.py --data closure_analysis_full/sweeps/best_run_out_50k_2000.npz --name out_50k_s1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
import os, time, argparse

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

VOXEL_SIZE = 50.0          # mm — track-level resolution
W_ION = 23.6e-6           # MeV per electron-ion pair
WIRE_PITCH = 3.0           # mm
TICK_SPATIAL = 0.8         # mm (0.5μs × 1.6 mm/μs)
NOISE_RMS_CELL = 300.0     # electrons per wire×tick cell

# Floor: computed as 1% of 99th percentile Q-diff (set after frame computation)
MAX_POINT_SIZE = 120
MIN_POINT_SIZE = 8
POINT_ALPHA = 0.55
FPS = 20
FRAME_SKIP = 2  # use every 2nd checkpoint
DPI = 150

# Recombination constants (modified_box)
ALPHA = 0.93
BETA = 0.212
LAR_DENSITY = 1.396


def de_to_Q(de, dx_cm, B_eff):
    raw = (dx_cm / B_eff) * np.log(np.maximum(ALPHA + B_eff * de / dx_cm, 1.0))
    return raw


def voxelize_Q(positions, energies, dx_cm, B_eff, voxel_size, origin):
    """Voxelize positions, accumulate Q (not dE) per voxel."""
    Q = de_to_Q(energies, dx_cm, B_eff)
    idx = ((positions - origin) / voxel_size).astype(np.int64)
    keys = idx[:, 0] * 1_000_000 + idx[:, 1] * 1_000 + idx[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    vox_Q = np.zeros(len(unique_keys))
    np.add.at(vox_Q, inverse, Q)
    vox_idx = np.zeros((len(unique_keys), 3), dtype=np.int64)
    for i, k in enumerate(unique_keys):
        vox_idx[i] = [k // 1_000_000, (k % 1_000_000) // 1_000, k % 1_000]
    centers = origin + (vox_idx + 0.5) * voxel_size
    return dict(zip(unique_keys.tolist(),
                zip(vox_Q.tolist(), centers[:, 0].tolist(),
                    centers[:, 1].tolist(), centers[:, 2].tolist())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Checkpoint NPZ file')
    parser.add_argument('--name', required=True, help='Output name suffix')
    args = parser.parse_args()

    data_path = os.path.join(OUT_DIR, args.data)
    out_path = os.path.join(OUT_DIR, f'annihilation_3d_Q_{args.name}.gif')

    print(f"Loading {data_path}...")
    data = np.load(data_path)
    truth_pos = data['truth_pos']
    truth_de = data['truth_de']
    checkpoint_params = data['checkpoint_params']
    checkpoint_steps = data['checkpoint_steps']
    checkpoint_losses = data['checkpoint_losses']
    n_seg = int(data['n_seg'])

    # Get dx and B_eff
    if 'dx_mm' in data:
        dx_mm = float(data['dx_mm'])
    else:
        dx_mm = 0.5  # default for older runs
    dx_cm = dx_mm / 10.0

    if 'B_eff' in data:
        B_eff = float(data['B_eff'])
    else:
        field_kVcm = 0.5
        B_eff = BETA / LAR_DENSITY / field_kVcm

    n_frames = len(checkpoint_steps)
    print(f"  {n_frames} checkpoints, {n_seg} segments, dx={dx_mm}mm")

    all_pos_min = np.minimum(truth_pos.min(axis=0),
                              checkpoint_params[:, :, :3].reshape(-1, 3).min(axis=0))
    origin = all_pos_min - VOXEL_SIZE

    print("Voxelizing truth (Q-space)...")
    truth_voxels = voxelize_Q(truth_pos, truth_de, dx_cm, B_eff, VOXEL_SIZE, origin)
    total_truth_Q = sum(v[0] for v in truth_voxels.values())
    mean_vox_Q = total_truth_Q / len(truth_voxels)
    mean_vox_E = mean_vox_Q / W_ION
    print(f"  {len(truth_voxels)} truth voxels, total Q={total_truth_Q:.1f}")
    print(f"  Mean voxel Q={mean_vox_Q:.3f} MeV = {mean_vox_E:.0f} e⁻")

    # Pass 1: compute all diffs unfiltered to find 99th percentile
    print("Computing frame data (pass 1: unfiltered)...")
    t0 = time.time()
    frame_data_raw = []

    for i in range(n_frames):
        params = checkpoint_params[i]
        alive = params[:, 3] > 0.012
        sim_voxels = voxelize_Q(params[alive, :3], params[alive, 3],
                                 dx_cm, B_eff, VOXEL_SIZE, origin)
        all_keys = set(truth_voxels.keys()) | set(sim_voxels.keys())
        diffs = []
        for k in all_keys:
            t_q = truth_voxels.get(k, (0,0,0,0))[0]
            s_q = sim_voxels.get(k, (0,0,0,0))[0]
            diff_q = t_q - s_q
            if abs(diff_q) < 1e-12:  # skip truly zero
                continue
            diff_e = diff_q / W_ION
            c = truth_voxels[k] if k in truth_voxels else sim_voxels[k]
            diffs.append((c[1], c[2], c[3], diff_e))
        arr = np.array(diffs) if diffs else np.zeros((1, 4))
        frame_data_raw.append(arr)

    # Compute 99th percentile across all frames
    all_abs = np.concatenate([np.abs(fd[:, 3]) for fd in frame_data_raw if len(fd) > 1])
    p99 = np.percentile(all_abs, 99)
    floor_e = 0.15 * p99  # 15% of 99th percentile — ensures all plotted points have visible color
    global_max_diff = all_abs.max()
    print(f"  99th percentile: {p99:.0f} e⁻")
    print(f"  Floor (1% of p99): {floor_e:.0f} e⁻")
    print(f"  Max: {global_max_diff:.0f} e⁻")

    # Pass 2: filter by floor
    frame_data = []
    for arr in frame_data_raw:
        mask = np.abs(arr[:, 3]) >= floor_e
        filtered = arr[mask] if mask.any() else np.zeros((1, 4))
        frame_data.append(filtered)

    for i in range(0, n_frames, 50):
        print(f"  Frame {i}/{n_frames}: {len(frame_data[i])} voxels (after floor)")
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Rendering animation...")
    fig = plt.figure(figsize=(11, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Colorbar: full range up to 99th percentile
    vmax = p99
    print(f"  Linear colorbar: vmax={vmax:.0f} e⁻ (99th %ile), "
          f"floor={floor_e:.0f} e⁻ (5% of p99)")

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label(f'ΔQ (electrons)  |  floor={floor_e:.0f} e⁻, clip={vmax:.0f} e⁻',
                   color='white', fontsize=9)
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    ax.set_xlabel('X (mm)', color='white', fontsize=10)
    ax.set_ylabel('Y (mm)', color='white', fontsize=10)
    ax.set_zlabel('Z (mm)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=7)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray'); ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, alpha=0.2, color='gray')

    margin = 50
    ax.set_xlim(truth_pos[:, 0].min() - margin, truth_pos[:, 0].max() + margin)
    ax.set_ylim(truth_pos[:, 1].min() - margin, truth_pos[:, 1].max() + margin)
    ax.set_zlim(truth_pos[:, 2].min() - margin, truth_pos[:, 2].max() + margin)

    scatter = [None]
    title = ax.set_title('', color='white', fontsize=11, fontweight='bold', pad=20)

    def update(frame_idx):
        if scatter[0] is not None:
            scatter[0].remove()

        # Gentle rotation: 40° total over 4s (0.2° per frame at 50fps)
        azim = -60 + 40.0 * frame_idx / n_frames
        ax.view_init(elev=25, azim=azim)

        arr = frame_data[frame_idx]
        x, y, z, diff = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        # Log-based size
        abs_diff = np.abs(diff)
        ratio = abs_diff / floor_e
        sizes = np.clip(np.log10(ratio + 1) /
                        np.log10(vmax / floor_e + 1)
                        * MAX_POINT_SIZE, MIN_POINT_SIZE, MAX_POINT_SIZE)

        scatter[0] = ax.scatter(x, y, z, c=diff, cmap=cmap, norm=norm,
                                s=sizes, alpha=POINT_ALPHA,
                                edgecolors='none', depthshade=True)
        step = checkpoint_steps[frame_idx]
        loss = checkpoint_losses[frame_idx]
        max_e = np.abs(diff).max() if len(diff) > 0 else 0
        title.set_text(f'{args.name} | Step {step:4d} | Loss: {loss:.5f} | '
                       f'{len(arr)} voxels | max={max_e/1000:.0f}k e⁻')
        return scatter[0],

    render_indices = list(range(0, n_frames, FRAME_SKIP))
    if render_indices[-1] != n_frames - 1:
        render_indices.append(n_frames - 1)
    n_render = len(render_indices)
    print(f"  Creating {n_render} frames (skip={FRAME_SKIP})...")
    t0 = time.time()
    anim = animation.FuncAnimation(fig, update, frames=render_indices, blit=False, interval=1000/FPS)
    writer = animation.PillowWriter(fps=FPS)
    anim.save(out_path, writer=writer, dpi=DPI)
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"  Saved to {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    plt.close(fig)


if __name__ == '__main__':
    main()
