"""
Precompute wire signals from saved checkpoint params.
Saves per-plane 2D signal arrays at selected steps for visualization.

Usage:
    python3 closure_analysis_full/sweeps/precompute_signals.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax, jax.numpy as jnp, numpy as np

from tools.geometry import generate_detector
from tools.config import SegmentData, DepositData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, 'best_final.npz')
DX_MM = 0.3
FRAME_EVERY = 5  # every 5th checkpoint = every 50 steps


def main():
    print("Loading checkpoint data...")
    data = np.load(DATA_PATH, allow_pickle=True)
    checkpoint_params = data['checkpoint_params']
    checkpoint_steps = data['checkpoint_steps']
    checkpoint_losses = data['checkpoint_losses']
    n_seg = int(data['n_seg'])
    active_planes = list(data['active_planes'])

    frame_indices = list(range(0, len(checkpoint_steps), FRAME_EVERY))
    if frame_indices[-1] != len(checkpoint_steps) - 1:
        frame_indices.append(len(checkpoint_steps) - 1)
    n_frames = len(frame_indices)
    print(f"  {len(checkpoint_steps)} checkpoints → {n_frames} frames")

    # Setup
    deposit_data = load_particle_step_data('out.h5', 2)
    n_truth = deposit_data.positions_mm.shape[0]
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    # Truth signals
    forced_deposit = DepositData(
        positions_mm=deposit_data.positions_mm, de=deposit_data.de,
        dx=np.full(n_truth, DX_MM, dtype=np.float32),
        valid_mask=deposit_data.valid_mask, theta=deposit_data.theta,
        phi=deposit_data.phi, track_ids=deposit_data.track_ids)

    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model='modified_box')
    print("Computing truth signals...")
    response_signals, _ = sim_truth(forced_deposit)

    truth_signals = {}
    for (side, plane), signal in response_signals.items():
        truth_signals[(side, plane)] = np.array(signal)

    # Optimizer forward
    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model='modified_box')
    fwd_opt = sim_opt.build_forward(dx_mm=DX_MM)

    # JIT warmup
    seg0 = SegmentData(
        positions_mm=jnp.array(checkpoint_params[0][:, :3]),
        de=jnp.array(checkpoint_params[0][:, 3]))
    _ = fwd_opt(seg0)
    for s in fwd_opt(seg0):
        jax.block_until_ready(s)
    print("JIT ready.\n")

    # Precompute signals at each frame
    print(f"Computing {n_frames} forward passes...")
    save_data = {
        'frame_steps': np.array([checkpoint_steps[ci] for ci in frame_indices]),
        'frame_losses': np.array([checkpoint_losses[ci] for ci in frame_indices]),
        'n_frames': n_frames,
        'active_planes': np.array(active_planes),
    }

    # Save truth signals
    for (side, plane), sig in truth_signals.items():
        save_data[f'truth_{side}_{plane}'] = sig

    t0 = time.time()
    for fi, ci in enumerate(frame_indices):
        params = checkpoint_params[ci]
        seg = SegmentData(
            positions_mm=jnp.array(params[:, :3]),
            de=jnp.array(params[:, 3]))
        recon = fwd_opt(seg)

        for pidx in active_planes:
            side, plane = pidx // 3, pidx % 3
            save_data[f'sim_{fi}_{side}_{plane}'] = np.array(recon[pidx])

        if fi % 5 == 0 or fi == n_frames - 1:
            print(f"  Frame {fi+1}/{n_frames} (step {checkpoint_steps[ci]}) — {time.time()-t0:.0f}s")

    save_path = os.path.join(OUT_DIR, 'precomputed_signals.npz')
    np.savez_compressed(save_path, **save_data)
    print(f"\nSaved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
