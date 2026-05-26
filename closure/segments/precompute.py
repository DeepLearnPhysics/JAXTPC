"""
Precompute wire signals from saved checkpoint params — sparse storage.
Threshold at half-deadband (400 electrons) for compact storage.
Reads from power-law checkpoint files.

Usage:
    python3 -m closure.segments.precompute --checkpoint best_final_v2.npz --edepsim out.h5
    python3 -m closure.segments.precompute --checkpoint best_final_v2.npz --edepsim out.h5 \
        --config config/cubic_wireplane_config.yaml --event 2 --outdir closure/segments
"""

import os, time, argparse

import jax, jax.numpy as jnp, numpy as np

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data

DX_MM = 0.3
DEADBAND_ENC = 800
SPARSE_ENC = DEADBAND_ENC // 2  # 400 electrons


def to_sparse(arr, threshold_adc):
    """Convert dense 2D array to sparse (indices, values) above threshold."""
    mask = np.abs(arr) >= threshold_adc
    indices = np.argwhere(mask).astype(np.int32)
    values = arr[mask].astype(np.float32)
    return indices, values


def main():
    parser = argparse.ArgumentParser(
        description='Precompute wire signals from checkpoint params (sparse storage)')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint .npz file (e.g. best_final_v2.npz)')
    parser.add_argument('--edepsim', required=True,
                        help='HDF5 file with particle step data')
    parser.add_argument('--config', default='config/cubic_wireplane_config.yaml',
                        help='Detector config YAML (default: config/cubic_wireplane_config.yaml)')
    parser.add_argument('--event', type=int, default=2,
                        help='Event index for truth deposits (default: 2)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: same as checkpoint dir)')
    parser.add_argument('--frame-every', type=int, default=1,
                        help='Use every Nth checkpoint (default: 1 = all)')
    args = parser.parse_args()

    out_dir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(args.checkpoint))

    print(f"Loading {args.checkpoint}...")
    data = np.load(args.checkpoint, allow_pickle=True)
    checkpoint_params = data['checkpoint_params']
    checkpoint_steps = data['checkpoint_steps']
    checkpoint_losses = data['checkpoint_losses']
    n_seg = int(data['n_seg'])
    active_planes = list(data['active_planes'])

    # Subsample frames
    frame_indices = list(range(0, len(checkpoint_steps), args.frame_every))
    if frame_indices[-1] != len(checkpoint_steps) - 1:
        frame_indices.append(len(checkpoint_steps) - 1)
    n_frames = len(frame_indices)
    print(f"  {len(checkpoint_steps)} checkpoints → {n_frames} frames")

    # Setup
    step_data = load_particle_step_data(args.edepsim, args.event)
    positions_mm = step_data['positions_mm']
    de = step_data['de']
    n_truth = positions_mm.shape[0]
    detector_config = generate_detector(args.config)
    electrons_per_adc = float(detector_config['electrons_per_adc'])
    sparse_threshold_adc = SPARSE_ENC / electrons_per_adc
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    print(f"  Sparse threshold: {SPARSE_ENC} e⁻ = {sparse_threshold_adc:.2f} ADC")

    # Truth signals — build proper DepositData via build_deposit_data
    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model='modified_box')

    deposits = build_deposit_data(
        positions_mm=positions_mm,
        de=de,
        dx=np.full(n_truth, DX_MM, dtype=np.float32),
        sim_config=sim_truth.config,
        theta=step_data['theta'],
        phi=step_data['phi'],
        track_ids=step_data['track_ids'],
    )

    print("Computing truth signals...")
    response_signals, _, _ = sim_truth.process_event(deposits)

    # Save truth as sparse
    save_data = {
        'n_frames': n_frames,
        'frame_steps': np.array([checkpoint_steps[ci] for ci in frame_indices]),
        'frame_losses': np.array([checkpoint_losses[ci] for ci in frame_indices]),
        'active_planes': np.array(active_planes),
        'sparse_threshold_adc': sparse_threshold_adc,
        'sparse_threshold_enc': SPARSE_ENC,
        'deadband_enc': DEADBAND_ENC,
    }

    truth_shapes = {}
    for (side, plane), signal in response_signals.items():
        pidx = side * 3 + plane
        arr = np.array(signal)
        truth_shapes[pidx] = arr.shape
        idx, val = to_sparse(arr, sparse_threshold_adc)
        save_data[f'truth_{side}_{plane}_indices'] = idx
        save_data[f'truth_{side}_{plane}_values'] = val
        save_data[f'truth_{side}_{plane}_shape'] = np.array(arr.shape)
        n_total = arr.shape[0] * arr.shape[1]
        print(f"  truth ({side},{plane}): {len(val):,}/{n_total:,} nonzero "
              f"({len(val)/n_total*100:.1f}%)")

    # Optimizer forward — inline wrapper replacing build_forward + SegmentData
    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model='modified_box')
    sim_params = sim_opt.default_sim_params

    def fwd(positions_mm, de):
        return sim_opt.forward_segments(sim_params, positions_mm, de, dx=DX_MM)

    # JIT warmup
    pos0 = jnp.array(checkpoint_params[0][:, :3])
    de0 = jnp.array(checkpoint_params[0][:, 3])
    _ = fwd(pos0, de0)
    for s in fwd(pos0, de0):
        jax.block_until_ready(s)
    print("JIT ready.\n")

    # Precompute signals at each frame — save sparse
    print(f"Computing {n_frames} forward passes (sparse storage)...")
    t0 = time.time()
    total_stored = 0

    for fi, ci in enumerate(frame_indices):
        params = checkpoint_params[ci]
        recon = fwd(jnp.array(params[:, :3]), jnp.array(params[:, 3]))

        for pidx in active_planes:
            side, plane = pidx // 3, pidx % 3
            arr = np.array(recon[pidx])
            idx, val = to_sparse(arr, sparse_threshold_adc)
            save_data[f'sim_{fi}_{side}_{plane}_indices'] = idx
            save_data[f'sim_{fi}_{side}_{plane}_values'] = val
            total_stored += idx.nbytes + val.nbytes

        if fi % 10 == 0 or fi == n_frames - 1:
            print(f"  Frame {fi+1}/{n_frames} (step {checkpoint_steps[ci]}) — "
                  f"{time.time()-t0:.0f}s, {total_stored/1e6:.0f} MB stored")

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'precomputed_signals_v2.npz')
    np.savez_compressed(save_path, **save_data)
    print(f"\nSaved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")
    print(f"  vs dense estimate: ~{n_frames * 6 * 1969 * 2701 * 4 / 1e9:.1f} GB")


if __name__ == '__main__':
    main()
