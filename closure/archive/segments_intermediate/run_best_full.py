"""
Run best config for 4000 steps, saving segment state every 10 steps.

Saves: positions, energies, loss, total_dE, dead count at each checkpoint.
Output: NPZ file for post-visualization.

Usage:
    python3 closure_analysis_full/sweeps/run_best_full.py mpvmpr_20.h5
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from tools.recombination import extract_recombination_params
from closure_analysis_full.full_closure import (
    DEFAULTS, PLANE_NAMES, build_loss_fn, relocate_segments,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'mpvmpr_20.h5'
    n_seg = 10000
    total_steps = 4000
    save_every = 10

    # Best config
    lr = 1.0
    decay_rate = 0.999
    lr_e_mult = 0.003
    warmup = 100
    noise_lr = 0.3
    l1 = 0.0
    reloc_every = 25
    max_reloc = 100
    recomb_model = 'modified_box'

    print(f"Best config run — {total_steps} steps, saving every {save_every}")
    print(f"lr={lr}, decay={decay_rate}, e_mult={lr_e_mult}, warmup={warmup}")
    print(f"noise_lr={noise_lr}, l1={l1}, reloc={reloc_every}/{max_reloc}")
    print(f"File: {h5_path}, n_seg={n_seg}\n")

    # --- Setup ---
    deposit_data = load_particle_step_data(h5_path, 0)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_total_de = float(np.sum(deposit_data.de))
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)
    print(f"Truth: {n_truth:,} segments, total dE={truth_total_de:.2f} MeV")

    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(
        detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False,
        include_electronics=False, include_track_hits=False,
        recombination_model=recomb_model)
    response_signals, _ = sim_truth(deposit_data)

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

    # Recomb constants
    _, dens, _, alpha_r, beta_r = extract_recombination_params(detector_config)
    field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
    B_eff = beta_r / dens / field_kVcm
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, 0.05)

    # Optimizer sim
    sim_opt = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=recomb_model)
    fwd_opt = sim_opt.build_forward()
    loss_fn = build_loss_fn(fwd_opt, truth_signals, spectral_weights, active_planes)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Init
    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy() + rng.normal(0, 100.0, size=(n_seg, 3))
    init_de = truth_de[indices].copy() * e_scale * rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    # JIT warmup
    print("JIT warmup...")
    t0 = time.time()
    _ = grad_fn(init_params)
    print(f"Done in {time.time()-t0:.1f}s\n")

    # --- Training loop with checkpointing ---
    schedule = optax.exponential_decay(
        init_value=lr, transition_steps=1, decay_rate=decay_rate)
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)

    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    # Storage
    n_checkpoints = total_steps // save_every + 1
    checkpoint_steps = []
    checkpoint_losses = []
    checkpoint_total_de = []
    checkpoint_dead = []
    checkpoint_relocs = []
    checkpoint_params = np.zeros((n_checkpoints, n_seg, 4), dtype=np.float32)

    cumulative_relocs = 0
    print_every = max(50, total_steps // 40)
    t_start = time.time()

    for step in range(total_steps):
        # Forward + backward
        (total_loss, _), grads = grad_fn(params)

        # Adam update
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(lr_e_mult)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], DEFAULTS['min_energy']))

        # Noise
        if noise_lr > 0:
            lr_cur = float(schedule(step))
            noise_scale = lr_cur * noise_lr
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                noise_scale * jax.random.normal(nk, shape=(n_seg, 3)))

        # Relocation
        if step >= warmup and step % reloc_every == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key, recomb_constants, max_reloc)
            cumulative_relocs += int(n_reloc)

        # Checkpoint
        if step % save_every == 0:
            idx = step // save_every
            checkpoint_steps.append(step)
            checkpoint_losses.append(float(total_loss))
            total_e = float(jnp.sum(params[:, 3]))
            checkpoint_total_de.append(total_e)
            n_dead = int(jnp.sum(params[:, 3] <= DEFAULTS['death_thresh']))
            checkpoint_dead.append(n_dead)
            checkpoint_relocs.append(cumulative_relocs)
            checkpoint_params[idx] = np.array(params)

        # Print
        if step % print_every == 0 or step == total_steps - 1:
            loss_val = float(total_loss)
            total_e = float(jnp.sum(params[:, 3]))
            n_dead = int(jnp.sum(params[:, 3] <= DEFAULTS['death_thresh']))
            elapsed = time.time() - t_start
            print(f"  Step {step:5d}: loss={loss_val:.6f}  "
                  f"total_dE={total_e:.2f}  dead={n_dead}  "
                  f"relocs={cumulative_relocs}  ({elapsed:.0f}s)")

    # Final checkpoint
    idx = total_steps // save_every
    if total_steps % save_every == 0:
        checkpoint_steps.append(total_steps)
        checkpoint_losses.append(float(total_loss))
        checkpoint_total_de.append(float(jnp.sum(params[:, 3])))
        checkpoint_dead.append(int(jnp.sum(params[:, 3] <= DEFAULTS['death_thresh'])))
        checkpoint_relocs.append(cumulative_relocs)
        checkpoint_params[idx] = np.array(params)

    total_time = time.time() - t_start
    final_de = float(np.sum(np.array(params)[:, 3]))
    de_ratio = final_de / truth_total_de

    print(f"\nDone in {total_time:.0f}s ({total_time/total_steps*1000:.0f} ms/step)")
    print(f"Final: loss={checkpoint_losses[-1]:.6f}, dE_ratio={de_ratio:.3f}, "
          f"dead={checkpoint_dead[-1]}, relocs={cumulative_relocs}")

    # Save
    save_path = os.path.join(OUT_DIR, 'best_run_4000.npz')
    np.savez_compressed(save_path,
        # Checkpoints (every 10 steps)
        checkpoint_steps=np.array(checkpoint_steps),
        checkpoint_losses=np.array(checkpoint_losses),
        checkpoint_total_de=np.array(checkpoint_total_de),
        checkpoint_dead=np.array(checkpoint_dead),
        checkpoint_relocs=np.array(checkpoint_relocs),
        checkpoint_params=checkpoint_params[:len(checkpoint_steps)],
        # Truth info
        truth_total_de=truth_total_de,
        truth_pos=truth_pos,
        truth_de=truth_de,
        n_truth=n_truth,
        # Init info
        init_indices=indices,
        e_scale=e_scale,
        # Config
        n_seg=n_seg,
        total_steps=total_steps,
        lr=lr, decay_rate=decay_rate, lr_e_mult=lr_e_mult,
        warmup=warmup, noise_lr=noise_lr, l1=l1,
        reloc_every=reloc_every, max_reloc=max_reloc,
    )
    print(f"Saved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
