"""
Final best config run with power-law checkpoint spacing.
alpha=2: quadratic -- more checkpoints early where changes happen.

Usage:
    python3 closure/segments/run_final.py --data out.h5 --event 2
    python3 closure/segments/run_final.py --data out.h5 --event 2 --config-yaml config/sbnd_config.yaml
"""

import os, time, argparse

import jax, jax.numpy as jnp, numpy as np, optax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from closure.segments.run import DEFAULTS, PLANE_NAMES, build_loss_fn, relocate_segments

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOBOLEV_S = 1.0
DX_MM = 0.3
ALPHA_RECOMB = 0.93
W_ION = 23.6e-6


def power_law_steps(total_steps, n_checkpoints, alpha=2.0):
    """Generate non-uniform checkpoint steps with power-law spacing.

    step_k = total_steps * (k/N)^alpha
    alpha=1: uniform, alpha=2: quadratic (4x denser at start)
    """
    k = np.linspace(0, 1, n_checkpoints)
    steps = np.round(total_steps * k ** alpha).astype(int)
    steps = np.unique(np.clip(steps, 0, total_steps - 1))
    return steps


def main():
    parser = argparse.ArgumentParser(
        description='Final best config run with power-law checkpoint spacing')
    parser.add_argument('--data', default='out.h5',
                        help='Path to HDF5 event file (default: out.h5)')
    parser.add_argument('--config-yaml', default='config/cubic_wireplane_config.yaml',
                        help='Path to detector YAML config (default: config/cubic_wireplane_config.yaml)')
    parser.add_argument('--event', type=int, default=2,
                        help='Event index (default: 2)')
    args = parser.parse_args()

    n_seg = 50000
    total_steps = 2000
    n_checkpoints = 400
    alpha = 2.0

    # Best config
    lr, decay_rate, lr_e_mult = 1.0, 0.999, 0.01
    warmup, noise_lr = 100, 0.3
    reloc_every, max_reloc = 25, 1000
    track_jitter_mm = 50.0

    # Compute checkpoint schedule
    checkpoint_schedule = power_law_steps(total_steps, n_checkpoints, alpha)
    checkpoint_set = set(checkpoint_schedule.tolist())

    print(f"Final v2: {n_seg} segs, {total_steps} steps, alpha={alpha}")
    print(f"  {len(checkpoint_schedule)} checkpoints (power-law)")
    print(f"  First 10 steps: {checkpoint_schedule[:10]}")
    print(f"  Last 10 steps:  {checkpoint_schedule[-10:]}")
    print(f"  Steps 0-100:  {np.sum(checkpoint_schedule <= 100)} checkpoints")
    print(f"  Steps 100-500: {np.sum((checkpoint_schedule > 100) & (checkpoint_schedule <= 500))} checkpoints")
    print(f"  Steps 500+:   {np.sum(checkpoint_schedule > 500)} checkpoints")
    print(f"  lr={lr}, d={decay_rate}, e_mult={lr_e_mult}, reloc {reloc_every}/{max_reloc}\n")

    # Setup
    raw = load_particle_step_data(args.data, args.event)
    n_truth = raw['positions_mm'].shape[0]
    truth_total_de = float(np.sum(raw['de']))
    truth_pos = np.asarray(raw['positions_mm'])
    truth_de = np.asarray(raw['de'])
    truth_tids = np.asarray(raw['track_ids'])

    detector_config = generate_detector(args.config_yaml)
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model='modified_box')

    # Build DepositData via build_deposit_data (multi-volume aware)
    deposits = build_deposit_data(
        raw['positions_mm'], raw['de'],
        np.full(n_truth, DX_MM, dtype=np.float32),
        sim_truth.config,
        theta=raw['theta'], phi=raw['phi'], track_ids=raw['track_ids'],
        t0_us=raw['t0_us'], interaction_ids=raw.get('interaction_ids'),
        root_track_ids=raw.get('root_track_ids'), pdg=raw.get('pdg'))

    t0 = time.time()
    response_signals, _, _ = sim_truth.process_event(deposits)
    print(f"Truth sim: {time.time()-t0:.1f}s")

    truth_dict, active_planes, sob_dict = {}, [], {}
    for (side, plane), signal in response_signals.items():
        pidx = side * 3 + plane
        signal = jnp.asarray(signal)
        truth_dict[pidx] = signal
        if jnp.any(signal != 0):
            active_planes.append(pidx)
            sob_dict[pidx] = make_sobolev_weight(*signal.shape, s=SOBOLEV_S)
    active_planes.sort()
    truth_signals = tuple(truth_dict.get(i, jnp.zeros((1,1))) for i in range(6))
    spectral_weights = tuple(sob_dict.get(i, jnp.zeros((1,1))) for i in range(6))

    truth_profiles = {}
    for p in active_planes:
        truth_profiles[p] = np.sum(np.abs(np.array(truth_signals[p])), axis=1)

    # Recombination constants from sim params
    rp = sim_truth.default_sim_params.recomb_params
    dens = float(rp.density)
    alpha_r = float(rp.alpha)
    beta_r = float(rp.beta)
    field_kVcm = float(rp.field_strength_Vcm) / 1000.0
    B_eff = beta_r / dens / field_kVcm
    dx_cm = DX_MM / 10.0
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, dx_cm)

    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model='modified_box')
    sim_params = sim_opt.default_sim_params
    dx_val = DX_MM
    def fwd_opt(positions_mm, de):
        return sim_opt.forward_segments(sim_params, positions_mm, de, dx=dx_val)

    loss_fn = build_loss_fn(fwd_opt, truth_signals, spectral_weights, active_planes)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Per-track jitter init
    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy()
    init_de = truth_de[indices].copy() * e_scale
    init_tids = truth_tids[indices]
    for tid in np.unique(init_tids):
        init_pos[init_tids == tid] += rng.normal(0, track_jitter_mm, size=3)
    init_de *= rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    print("JIT warmup...")
    t0 = time.time()
    _ = grad_fn(init_params)
    print(f"Done in {time.time()-t0:.1f}s\n")

    def compute_Q(de_arr):
        return (dx_cm / B_eff) * np.log(np.maximum(ALPHA_RECOMB + B_eff * de_arr / dx_cm, 1.0))
    truth_total_Q = compute_Q(truth_de).sum()

    # Training
    schedule = optax.exponential_decay(init_value=lr, transition_steps=1, decay_rate=decay_rate)
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)
    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    # Storage
    actual_checkpoint_steps = []
    checkpoint_losses = []
    checkpoint_params_list = []
    de_history = []

    losses = []
    cumulative_relocs = 0
    t_start = time.time()

    for step in range(total_steps):
        (total_loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(lr_e_mult)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], 0.001))

        lr_cur = float(schedule(step))
        rng_key, nk = jax.random.split(rng_key)
        params = params.at[:, :3].add(lr_cur * noise_lr * jax.random.normal(nk, shape=(n_seg, 3)))

        if step >= warmup and step % reloc_every == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key, recomb_constants, max_reloc)
            cumulative_relocs += int(n_reloc)

        loss_val = float(total_loss)
        losses.append(loss_val)

        # Power-law checkpoints
        if step in checkpoint_set:
            actual_checkpoint_steps.append(step)
            checkpoint_losses.append(loss_val)
            checkpoint_params_list.append(np.array(params))

            # dE histogram
            p_np = np.array(params[:, 3])
            alive_de = p_np[p_np > DEFAULTS['death_thresh']]
            n_dead = int(np.sum(p_np <= DEFAULTS['death_thresh']))
            de_history.append((alive_de.copy(), n_dead))

        if step % 50 == 0 or step == total_steps - 1:
            p_np = np.array(params)
            alive = p_np[:, 3] > 0.012
            sim_Q = compute_Q(p_np[alive, 3]).sum()
            q_ratio = sim_Q / truth_total_Q
            n_dead = int((~alive).sum())
            n_cp_so_far = len(actual_checkpoint_steps)
            print(f"  Step {step:5d}: loss={loss_val:.6f}  Q={q_ratio:.3f}  "
                  f"dead={n_dead}  cp={n_cp_so_far}  ({time.time()-t_start:.0f}s)",
                  flush=True)

    print(f"\nDone in {time.time()-t_start:.0f}s")
    print(f"Saved {len(actual_checkpoint_steps)} checkpoints")

    # Stack checkpoint params
    checkpoint_params = np.stack(checkpoint_params_list, axis=0)

    # Save
    save_path = os.path.join(OUT_DIR, 'best_final_v2.npz')
    save_data = {
        'checkpoint_steps': np.array(actual_checkpoint_steps),
        'checkpoint_losses': np.array(checkpoint_losses),
        'checkpoint_params': checkpoint_params,
        'truth_pos': truth_pos, 'truth_de': truth_de,
        'n_truth': n_truth, 'n_seg': n_seg, 'total_steps': total_steps,
        'truth_total_de': truth_total_de, 'dx_mm': DX_MM,
        'B_eff': B_eff, 'alpha': alpha_r,
        'losses': np.array(losses),
        'active_planes': np.array(active_planes),
        'checkpoint_alpha': alpha,
        'de_history_steps': np.array(actual_checkpoint_steps),
        'n_de_hist': len(de_history),
    }
    for i, (alive_de, n_dead) in enumerate(de_history):
        save_data[f'de_hist_{i}_alive'] = alive_de
        save_data[f'de_hist_{i}_dead'] = n_dead
    for p in active_planes:
        save_data[f'truth_profile_{p}'] = truth_profiles[p]

    np.savez_compressed(save_path, **save_data)
    print(f"Saved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
