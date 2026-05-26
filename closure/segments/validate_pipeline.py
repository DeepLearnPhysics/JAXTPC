"""
Pipeline validation for full closure analysis.

Verifies forward pass consistency, tests subsampling strategies,
and measures representation error vs n_seg.

Forces dx=0.1mm for both truth and optimizer to eliminate dx mismatch.

Run from project root:
    python3 -m closure.segments.validate_pipeline --data mpvmpr_20.h5
    python3 -m closure.segments.validate_pipeline --data mpvmpr_20.h5 --n-seg 10000
"""

import argparse, time

import jax
import jax.numpy as jnp
import numpy as np

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data
from tools.losses import sobolev_loss_geomean_log1p, sobolev_loss_single, make_sobolev_weight

PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']
DX_MM = 0.1  # fixed dx for both truth and optimizer


# =========================================================================
# Subsampling strategies
# =========================================================================

def _de_to_Q(de, dx_cm, alpha, B_eff):
    """Convert dE to charge Q via recombination: Q = (dx/B) * ln(alpha + B*dE/dx)."""
    return (dx_cm / B_eff) * np.log(alpha + B_eff * de / dx_cm)


def _Q_to_de(Q, dx_cm, alpha, B_eff):
    """Invert recombination: dE = (dx/B) * [exp(B*Q/dx) - alpha]."""
    return (dx_cm / B_eff) * (np.exp(B_eff * Q / dx_cm) - alpha)


def _get_recomb_constants(sim):
    """Extract recombination constants from the simulator's default params."""
    rp = sim.default_sim_params.recomb_params
    density = float(rp.density)
    field_Vcm = float(rp.field_strength_Vcm)
    alpha = float(rp.alpha)
    beta = float(rp.beta) if hasattr(rp, 'beta') else float(rp.beta_90)
    field_kVcm = field_Vcm / 1000.0
    B_eff = beta / density / field_kVcm
    return alpha, B_eff


def subsample_random(truth_pos, truth_de, n_seg, alpha, B_eff, seed=42):
    """Random subsampling with Q-space scaling.

    Each optimizer segment represents e_scale truth segments.
    Scales in charge space: Q_opt = e_scale * Q_truth, then inverts to dE.
    """
    rng = np.random.RandomState(seed)
    n_truth = len(truth_de)
    e_scale = n_truth / n_seg
    replace = n_seg > n_truth
    indices = rng.choice(n_truth, size=n_seg, replace=replace)

    dx_cm = DX_MM / 10.0
    Q_truth = _de_to_Q(truth_de[indices], dx_cm, alpha, B_eff)
    Q_scaled = Q_truth * e_scale
    de_scaled = _Q_to_de(Q_scaled, dx_cm, alpha, B_eff)
    de_scaled = np.maximum(de_scaled, 1e-6)  # floor

    return truth_pos[indices].copy(), de_scaled


def subsample_voxelize(truth_pos, truth_de, n_seg, alpha, B_eff, seed=42):
    """Voxelize truth segments and merge into Q-weighted centroids.

    Binary searches for voxel size that gives ~n_seg occupied voxels.
    Each voxel becomes one optimizer segment with:
      - position = Q-weighted centroid of truth segments in voxel
      - dE = inverted from sum of Q in voxel (charge-preserving)
    """
    dx_cm = DX_MM / 10.0
    truth_Q = _de_to_Q(truth_de, dx_cm, alpha, B_eff)

    pos_min = truth_pos.min(axis=0)
    pos_max = truth_pos.max(axis=0)
    extent = pos_max - pos_min

    # Binary search for voxel size
    lo, hi = 0.1, 100.0  # mm
    for _ in range(50):
        mid = (lo + hi) / 2
        n_bins = np.maximum((extent / mid).astype(int), 1)
        bin_idx = ((truth_pos - pos_min) / mid).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        keys = bin_idx[:, 0] * (n_bins[1] * n_bins[2]) + bin_idx[:, 1] * n_bins[2] + bin_idx[:, 2]
        n_occupied = len(np.unique(keys))
        if n_occupied > n_seg:
            lo = mid
        else:
            hi = mid

    # Use the converged voxel size
    voxel_size = (lo + hi) / 2
    n_bins = np.maximum((extent / voxel_size).astype(int), 1)
    bin_idx = ((truth_pos - pos_min) / voxel_size).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    keys = bin_idx[:, 0] * (n_bins[1] * n_bins[2]) + bin_idx[:, 1] * n_bins[2] + bin_idx[:, 2]

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_voxels = len(unique_keys)

    # Compute Q-weighted centroid and total Q per voxel
    vox_pos = np.zeros((n_voxels, 3))
    vox_Q = np.zeros(n_voxels)
    for i in range(n_voxels):
        mask = inverse == i
        Q_in = truth_Q[mask]
        pos_in = truth_pos[mask]
        total_Q = np.sum(Q_in)
        vox_Q[i] = total_Q
        if total_Q > 0:
            vox_pos[i] = np.sum(pos_in * Q_in[:, None], axis=0) / total_Q
        else:
            vox_pos[i] = np.mean(pos_in, axis=0)

    # Convert total Q back to dE
    vox_de = _Q_to_de(vox_Q, dx_cm, alpha, B_eff)
    vox_de = np.maximum(vox_de, 1e-6)

    # If too many voxels, randomly subsample with Q-space scaling
    if n_voxels > n_seg:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_voxels, size=n_seg, replace=False)
        vox_pos = vox_pos[idx]
        q_scale = n_voxels / n_seg
        vox_de = _Q_to_de(vox_Q[idx] * q_scale, dx_cm, alpha, B_eff)
        vox_de = np.maximum(vox_de, 1e-6)
    # If too few voxels, split largest ones (Q-preserving)
    elif n_voxels < n_seg:
        rng = np.random.RandomState(seed)
        n_extra = n_seg - n_voxels
        probs = vox_Q / vox_Q.sum()
        extra_idx = rng.choice(n_voxels, size=n_extra, p=probs)
        # Split Q in half for donor and clone
        half_Q = vox_Q[extra_idx] * 0.5
        extra_pos = vox_pos[extra_idx] + rng.normal(0, voxel_size * 0.3, size=(n_extra, 3))
        extra_de = _Q_to_de(half_Q, dx_cm, alpha, B_eff)
        extra_de = np.maximum(extra_de, 1e-6)
        # Update originals to half Q
        vox_Q[extra_idx] *= 0.5
        vox_de = _Q_to_de(vox_Q, dx_cm, alpha, B_eff)
        vox_de = np.maximum(vox_de, 1e-6)
        vox_pos = np.vstack([vox_pos, extra_pos])
        vox_de = np.concatenate([vox_de, extra_de])

    print(f"    Voxelize: size={voxel_size:.1f}mm, {n_voxels} occupied -> {len(vox_de)} segments")
    return vox_pos, vox_de


# =========================================================================
# Validation
# =========================================================================

def validate(h5_path, config_path, event_idx=0, n_seg=10000, recomb_model='modified_box'):
    print("=" * 70)
    print("PIPELINE VALIDATION (forced dx=0.1mm)")
    print("=" * 70)

    # --- Load ---
    raw = load_particle_step_data(h5_path, event_idx, verbose=False)
    truth_pos = np.asarray(raw['positions_mm'])
    truth_de = np.asarray(raw['de'])
    n_truth = truth_pos.shape[0]
    print(f"Truth: {n_truth:,} segments, total dE={np.sum(truth_de):.2f} MeV")

    detector_config = generate_detector(config_path)

    # --- Truth simulation with forced dx=0.1mm ---
    print(f"\n[1] Truth simulation (forced dx={DX_MM}mm, recomb={recomb_model})...")
    sim_truth = DetectorSimulator(
        detector_config, use_bucketed=False, total_pad=200_000,
        response_chunk_size=50_000, include_noise=False,
        include_electronics=False, include_track_hits=False,
        recombination_model=recomb_model)

    deposits = build_deposit_data(
        raw['positions_mm'], raw['de'],
        dx=np.full(n_truth, DX_MM, dtype=np.float32),
        sim_config=sim_truth.config,
        theta=raw['theta'], phi=raw['phi'],
        track_ids=raw['track_ids'],
        t0_us=raw['t0_us'],
        interaction_ids=raw['interaction_ids'],
        root_track_ids=raw['root_track_ids'],
        pdg=raw['pdg'])

    t0 = time.time()
    response_signals, _, _ = sim_truth.process_event(deposits)
    print(f"    Done in {time.time()-t0:.1f}s")

    truth_dict, active_planes, sob_dict = {}, [], {}
    for (side, plane), signal in response_signals.items():
        pidx = side * 3 + plane
        signal = jnp.asarray(signal)
        truth_dict[pidx] = signal
        if jnp.any(signal != 0):
            active_planes.append(pidx)
            sob_dict[pidx] = make_sobolev_weight(*signal.shape, s=1.5)
    active_planes.sort()
    truth_signals = tuple(truth_dict.get(i, jnp.zeros((1, 1))) for i in range(6))
    spectral_weights = tuple(sob_dict.get(i, jnp.zeros((1, 1))) for i in range(6))
    planes_tuple = tuple(active_planes)

    for p in active_planes:
        sig = truth_signals[p]
        print(f"    {PLANE_NAMES[p]}: shape={sig.shape}, "
              f"sum|sig|={float(jnp.sum(jnp.abs(sig))):.0f}")

    # --- Sanity check: full segments through optimizer path ---
    print(f"\n[2] Sanity check: ALL {n_truth} segs through optimizer forward (dx={DX_MM}mm)...")
    sim_full = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_truth,
        recombination_model=recomb_model)
    sim_params = sim_full.default_sim_params

    def fwd_full(positions_mm, de):
        return sim_full.forward_segments(sim_params, positions_mm, de, dx=DX_MM)

    t0 = time.time()
    full_signals = fwd_full(jnp.array(truth_pos), jnp.array(truth_de))
    for s in full_signals:
        jax.block_until_ready(s)
    loss_full = float(sobolev_loss_geomean_log1p(
        full_signals, truth_signals, spectral_weights, planes=planes_tuple))
    print(f"    Loss (all segs, matched dx): {loss_full:.8f}")
    assert loss_full < 0.001, f"Forward pass mismatch! loss={loss_full}"
    print(f"    OK — forward passes are consistent")

    # --- Extract recombination constants for subsampling ---
    alpha, B_eff = _get_recomb_constants(sim_full)

    # --- Subsampling comparison ---
    print(f"\n[3] Subsampling strategies at n_seg={n_seg}...")
    strategies = {
        'random': subsample_random,
        'voxelize': subsample_voxelize,
    }

    for name, fn in strategies.items():
        print(f"\n  --- {name} ---")
        sub_pos, sub_de = fn(truth_pos, truth_de, n_seg, alpha, B_eff)
        total_de = np.sum(sub_de)
        print(f"    Total dE: {total_de:.2f} MeV (truth: {np.sum(truth_de):.2f})")
        print(f"    dE range: [{sub_de.min():.6f}, {sub_de.max():.6f}]")

        sim_sub = DetectorSimulator(
            detector_config, differentiable=True, n_segments=n_seg,
            recombination_model=recomb_model)
        sim_params_sub = sim_sub.default_sim_params

        def fwd_sub(positions_mm, de):
            return sim_sub.forward_segments(sim_params_sub, positions_mm, de, dx=DX_MM)

        t0 = time.time()
        sub_signals = fwd_sub(jnp.array(sub_pos), jnp.array(sub_de))
        for s in sub_signals:
            jax.block_until_ready(s)
        dt = time.time() - t0

        loss_sub = float(sobolev_loss_geomean_log1p(
            sub_signals, truth_signals, spectral_weights, planes=planes_tuple))
        print(f"    Loss at truth positions: {loss_sub:.4f} ({dt:.1f}s)")

        # Per-plane signal ratio
        for p in active_planes:
            t_sum = float(jnp.sum(jnp.abs(truth_signals[p])))
            s_sum = float(jnp.sum(jnp.abs(sub_signals[p])))
            print(f"      {PLANE_NAMES[p]}: ratio={s_sum/t_sum:.4f}")

    # --- Representation error vs n_seg ---
    print(f"\n[4] Representation error vs n_seg...")
    print(f"  {'n_seg':>8} {'Random':>12} {'Voxelize':>12}")
    print(f"  {'-'*35}")
    for ns in [1000, 2000, 5000, 10000, 20000]:
        losses = {}
        for name, fn in strategies.items():
            sub_pos, sub_de = fn(truth_pos, truth_de, ns, alpha, B_eff)
            sim_s = DetectorSimulator(
                detector_config, differentiable=True, n_segments=ns,
                recombination_model=recomb_model)
            sim_params_s = sim_s.default_sim_params

            def fwd_s(positions_mm, de):
                return sim_s.forward_segments(sim_params_s, positions_mm, de, dx=DX_MM)

            sig_s = fwd_s(jnp.array(sub_pos), jnp.array(sub_de))
            for s in sig_s:
                jax.block_until_ready(s)
            losses[name] = float(sobolev_loss_geomean_log1p(
                sig_s, truth_signals, spectral_weights, planes=planes_tuple))
        print(f"  {ns:>8} {losses['random']:>12.4f} {losses['voxelize']:>12.4f}")

    # --- Gradient check at n_seg ---
    print(f"\n[5] Gradient analysis at n_seg={n_seg} (voxelized init)...")
    vox_pos, vox_de = subsample_voxelize(truth_pos, truth_de, n_seg, alpha, B_eff)
    sim_grad = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=recomb_model)
    sim_params_grad = sim_grad.default_sim_params

    def loss_fn(params):
        pos = params[:, :3]
        de = params[:, 3]
        sigs = sim_grad.forward_segments(sim_params_grad, pos, de, dx=DX_MM)
        return sobolev_loss_geomean_log1p(sigs, truth_signals, spectral_weights,
                                          planes=planes_tuple)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    params = jnp.array(np.column_stack([vox_pos, vox_de]))

    t0 = time.time()
    loss_val, grads = grad_fn(params)
    jax.block_until_ready(grads)
    print(f"    JIT compiled in {time.time()-t0:.1f}s")
    print(f"    Loss: {float(loss_val):.4f}")

    pos_g = np.array(grads[:, :3])
    e_g = np.array(grads[:, 3])
    alive = np.array(params[:, 3]) > 0.001

    mean_pos_g = np.mean(np.abs(pos_g[alive]))
    mean_e_g = np.mean(np.abs(e_g[alive]))
    ratio = mean_e_g / mean_pos_g
    print(f"    mean |grad_pos|: {mean_pos_g:.8f}")
    print(f"    mean |grad_e|:   {mean_e_g:.8f}")
    print(f"    Ratio e/pos: {ratio:.2f}")
    print(f"    Suggested lr_e_mult: ~{1.0/ratio:.4f}")

    # Energy gradient sign
    neg_frac = np.mean(e_g[alive] < 0)
    print(f"    Energy grad: {neg_frac*100:.1f}% negative (want up), "
          f"{(1-neg_frac)*100:.1f}% positive (want down)")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate closure analysis pipeline')
    parser.add_argument('--data', required=True, help='Path to HDF5 event file')
    parser.add_argument('--config', default='config/cubic_wireplane_config.yaml',
                        help='Detector config YAML')
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--n-seg', type=int, default=10000)
    parser.add_argument('--recomb', type=str, default='modified_box',
                        choices=['modified_box', 'emb'])
    args = parser.parse_args()
    validate(args.data, args.config, args.event, args.n_seg, args.recomb)
