"""
3D annihilation plot: voxelized truth-sim difference over training.

Each frame shows voxels colored by (truth_dE - sim_dE):
  - Blue: truth has more charge (sim is missing)
  - Red: sim has more charge (sim excess)
  - Size proportional to |difference|

50fps, 8 seconds = 400 frames from checkpoints saved every 10 steps.

Usage:
    python3 closure_analysis_full/sweeps/render_3d_annihilation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os, sys, time

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_run_4000.npz')
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annihilation_3d.gif')

VOXEL_SIZE = 25.0  # mm (coarser for cleaner visualization)
MIN_DIFF = 0.005   # MeV, skip voxels with |diff| below this (no color for near-zero)
MAX_POINT_SIZE = 80
FPS = 50
DPI = 120


def voxelize(positions, energies, voxel_size, origin):
    """Hash positions into voxel keys, accumulate energy per voxel."""
    idx = ((positions - origin) / voxel_size).astype(np.int32)
    keys = idx[:, 0] * 1_000_000 + idx[:, 1] * 1_000 + idx[:, 2]
    voxels = {}
    for i, k in enumerate(keys):
        if k in voxels:
            voxels[k][0] += energies[i]
        else:
            center = origin + (idx[i] + 0.5) * voxel_size
            voxels[k] = [energies[i], center[0], center[1], center[2]]
    return voxels


def voxelize_vectorized(positions, energies, voxel_size, origin):
    """Vectorized voxelization using numpy."""
    idx = ((positions - origin) / voxel_size).astype(np.int64)
    keys = idx[:, 0] * 1_000_000 + idx[:, 1] * 1_000 + idx[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    # Sum energies per voxel
    vox_de = np.zeros(len(unique_keys))
    np.add.at(vox_de, inverse, energies)
    # Compute centers
    vox_centers = np.zeros((len(unique_keys), 3))
    for dim in range(3):
        np.add.at(vox_centers[:, dim], inverse, positions[:, dim] * energies)
    # Energy-weighted center (or just voxel center)
    vox_idx = np.zeros((len(unique_keys), 3), dtype=np.int64)
    for i, k in enumerate(unique_keys):
        vox_idx[i, 0] = k // 1_000_000
        vox_idx[i, 1] = (k % 1_000_000) // 1_000
        vox_idx[i, 2] = k % 1_000
    centers = origin + (vox_idx + 0.5) * voxel_size
    return dict(zip(unique_keys.tolist(), zip(vox_de.tolist(),
                centers[:, 0].tolist(), centers[:, 1].tolist(), centers[:, 2].tolist())))


def main():
    print("Loading data...")
    data = np.load(DATA_PATH)

    truth_pos = data['truth_pos']
    truth_de = data['truth_de']
    checkpoint_params = data['checkpoint_params']
    checkpoint_steps = data['checkpoint_steps']
    checkpoint_losses = data['checkpoint_losses']
    n_seg = int(data['n_seg'])
    truth_total_de = float(data['truth_total_de'])

    n_frames = len(checkpoint_steps)
    print(f"  {n_frames} checkpoints, {n_seg} segments")
    print(f"  Truth: {len(truth_de)} segments")

    # Compute voxel grid origin (min of all positions with margin)
    all_pos_min = np.minimum(truth_pos.min(axis=0),
                              checkpoint_params[:, :, :3].reshape(-1, 3).min(axis=0))
    origin = all_pos_min - VOXEL_SIZE

    # Pre-compute truth voxels (fixed for all frames)
    print("Voxelizing truth...")
    truth_voxels = voxelize_vectorized(truth_pos, truth_de, VOXEL_SIZE, origin)
    print(f"  {len(truth_voxels)} truth voxels")

    # Pre-compute all frame data
    print("Computing frame data...")
    t0 = time.time()

    frame_data = []
    global_max_diff = 0

    for i in range(n_frames):
        params = checkpoint_params[i]
        sim_pos = params[:, :3]
        sim_de = params[:, 3]

        # Only include alive segments
        alive = sim_de > 0.001
        sim_voxels = voxelize_vectorized(sim_pos[alive], sim_de[alive], VOXEL_SIZE, origin)

        # Compute difference: truth - sim
        all_keys = set(truth_voxels.keys()) | set(sim_voxels.keys())
        diffs = []
        for k in all_keys:
            t_de = truth_voxels.get(k, [0, 0, 0, 0])[0]
            s_de = sim_voxels.get(k, [0, 0, 0, 0])[0]
            diff = t_de - s_de
            if abs(diff) < MIN_DIFF:
                continue
            # Get center from whichever has the key
            if k in truth_voxels:
                _, cx, cy, cz = truth_voxels[k]
            else:
                _, cx, cy, cz = sim_voxels[k]
            diffs.append((cx, cy, cz, diff))

        if diffs:
            arr = np.array(diffs)
            frame_data.append(arr)
            global_max_diff = max(global_max_diff, np.abs(arr[:, 3]).max())
        else:
            frame_data.append(np.zeros((1, 4)))

        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames}: {len(diffs)} voxels, "
                  f"max|diff|={np.abs(arr[:, 3]).max():.4f}")

    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Global max |diff|: {global_max_diff:.4f} MeV")

    # Create figure
    print("Rendering animation...")
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Fixed view angle
    ax.view_init(elev=25, azim=-60)

    # Color normalization (fixed across all frames, symmetric around 0)
    vmax = global_max_diff * 0.8  # clip top 20% for better contrast
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r  # red=sim excess, blue=truth excess

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label='Truth - Sim dE (MeV)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    # Style
    ax.set_xlabel('X (mm)', color='white', fontsize=10)
    ax.set_ylabel('Y (mm)', color='white', fontsize=10)
    ax.set_zlabel('Z (mm)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, alpha=0.2, color='gray')

    # Set axis limits from truth extent
    margin = 50
    ax.set_xlim(truth_pos[:, 0].min() - margin, truth_pos[:, 0].max() + margin)
    ax.set_ylim(truth_pos[:, 1].min() - margin, truth_pos[:, 1].max() + margin)
    ax.set_zlim(truth_pos[:, 2].min() - margin, truth_pos[:, 2].max() + margin)

    scatter = [None]
    title = ax.set_title('', color='white', fontsize=12, fontweight='bold', pad=20)

    def update(frame_idx):
        if scatter[0] is not None:
            scatter[0].remove()

        arr = frame_data[frame_idx]
        x, y, z, diff = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        # Size proportional to |diff|
        sizes = np.clip(np.abs(diff) / vmax * MAX_POINT_SIZE, 2, MAX_POINT_SIZE)

        # Alpha proportional to |diff| — near-zero voxels are invisible
        alphas = np.clip(np.abs(diff) / vmax, 0.05, 0.85)

        # Build RGBA colors manually so zero → transparent
        rgba = cmap(norm(diff))
        rgba[:, 3] = alphas

        scatter[0] = ax.scatter(x, y, z, c=rgba,
                                s=sizes, edgecolors='none',
                                depthshade=True)

        step = checkpoint_steps[frame_idx]
        loss = checkpoint_losses[frame_idx]
        title.set_text(f'Step {step:4d}  |  Loss: {loss:.4f}  |  '
                       f'{len(arr)} voxels  |  '
                       f'max|diff|: {np.abs(diff).max():.3f} MeV')
        return scatter[0],

    print(f"  Creating {n_frames} frames at {FPS}fps...")
    t0 = time.time()

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    blit=False, interval=1000/FPS)

    writer = animation.PillowWriter(fps=FPS)
    anim.save(OUT_PATH, writer=writer, dpi=DPI)

    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"  Saved to {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
    plt.close(fig)


if __name__ == '__main__':
    main()
