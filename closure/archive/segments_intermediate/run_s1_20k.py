"""
Run 20k segments with Sobolev s=1.0 (instead of 1.5).
Saves checkpoints for 3D visualization comparison.

Usage:
    python3 closure_analysis_full/sweeps/run_s1_20k.py mpvmpr_20.h5
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax, jax.numpy as jnp, numpy as np, optax
import matplotlib.pyplot as plt

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from tools.recombination import extract_recombination_params
from closure_analysis_full.full_closure import DEFAULTS, PLANE_NAMES, relocate_segments

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOBOLEV_S = 1.0  # changed from 1.5


def main():
    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'mpvmpr_20.h5'
    n_seg = 20000
    total_steps = 4000
    save_every = 10

    lr, decay_rate, lr_e_mult = 1.0, 0.999, 0.003
    warmup, noise_lr, l1 = 100, 0.3, 0.0
    reloc_every, max_reloc = 25, 100
    recomb_model = 'modified_box'

    print(f"20k segments, Sobolev s={SOBOLEV_S} — {total_steps} steps")
    print(f"lr={lr}, decay={decay_rate}, e_mult={lr_e_mult}")
    print(f"File: {h5_path}\n")

    # Setup
    deposit_data = load_particle_step_data(h5_path, 0)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_total_de = float(np.sum(deposit_data.de))
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)

    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model=recomb_model)
    response_signals, _ = sim_truth(deposit_data)

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
    planes_tuple = tuple(active_planes)

    _, dens, _, alpha_r, beta_r = extract_recombination_params(detector_config)
    field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
    B_eff = beta_r / dens / field_kVcm
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, 0.05)

    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=recomb_model)
    fwd_opt = sim_opt.build_forward()

    def loss_fn(params):
        seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
        sigs = fwd_opt(seg)
        loss = sobolev_loss_geomean_log1p(sigs, truth_signals, spectral_weights, planes=planes_tuple)
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Init
    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy() + rng.normal(0, 100.0, size=(n_seg, 3))
    init_de = truth_de[indices].copy() * e_scale * rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    print("JIT warmup...")
    t0 = time.time()
    _ = grad_fn(init_params)
    print(f"Done in {time.time()-t0:.1f}s\n")

    # Training
    schedule = optax.exponential_decay(init_value=lr, transition_steps=1, decay_rate=decay_rate)
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)
    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    n_checkpoints = total_steps // save_every + 1
    checkpoint_steps = []
    checkpoint_losses = []
    checkpoint_params = np.zeros((n_checkpoints, n_seg, 4), dtype=np.float32)
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

        if step % save_every == 0:
            idx = step // save_every
            checkpoint_steps.append(step)
            checkpoint_losses.append(float(total_loss))
            checkpoint_params[idx] = np.array(params)

        if step % 200 == 0 or step == total_steps - 1:
            total_e = float(jnp.sum(params[:, 3]))
            n_dead = int(jnp.sum(params[:, 3] <= DEFAULTS['death_thresh']))
            print(f"  Step {step:5d}: loss={float(total_loss):.6f}  "
                  f"total_dE={total_e:.2f}  dead={n_dead}  "
                  f"relocs={cumulative_relocs}  ({time.time()-t_start:.0f}s)")

    total_time = time.time() - t_start
    final_params_np = np.array(params)
    final_de_total = float(np.sum(final_params_np[:, 3]))
    de_ratio = final_de_total / truth_total_de
    n_dead_final = int(np.sum(final_params_np[:, 3] <= DEFAULTS['death_thresh']))

    print(f"\nDone in {total_time:.0f}s ({total_time/total_steps*1000:.0f} ms/step)")
    print(f"Final: loss={checkpoint_losses[-1]:.6f}, dE_ratio={de_ratio:.3f}, "
          f"dead={n_dead_final}, relocs={cumulative_relocs}")

    # Save checkpoints
    save_path = os.path.join(OUT_DIR, f'best_run_20k_s{SOBOLEV_S}_4000.npz')
    np.savez_compressed(save_path,
        checkpoint_steps=np.array(checkpoint_steps),
        checkpoint_losses=np.array(checkpoint_losses),
        checkpoint_params=checkpoint_params[:len(checkpoint_steps)],
        truth_pos=truth_pos, truth_de=truth_de,
        n_truth=n_truth, n_seg=n_seg, total_steps=total_steps,
        truth_total_de=truth_total_de, sobolev_s=SOBOLEV_S,
    )
    print(f"Saved checkpoints to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    # Diagnostic plot
    seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
    recon_signals = fwd_opt(seg)

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    # Loss
    ax = axes[0, 0]
    ax.semilogy(checkpoint_losses, 'b-', lw=1.0, alpha=0.7)
    ax.set_xlabel('Checkpoint', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(f'Loss Convergence (Sobolev s={SOBOLEV_S})', fontsize=15)
    ax.grid(True, alpha=0.3)

    # West U
    ax = axes[0, 1]
    p = 3
    if p in active_planes:
        t_prof = np.sum(np.abs(np.array(truth_signals[p])), axis=1)
        r_prof = np.sum(np.abs(np.array(recon_signals[p])), axis=1)
        nz = np.where(t_prof > 0)[0]
        wl, wh = (max(0, nz[0]-10), min(len(t_prof), nz[-1]+11)) if len(nz) > 0 else (0, len(t_prof))
        w = np.arange(wl, wh)
        ax.plot(w, t_prof[wl:wh], 'b-', lw=1.5, label='Truth')
        ax.plot(w, r_prof[wl:wh], 'r--', lw=1.5, label='Recon')
    ax.set_title(f'Signal Comparison ({PLANE_NAMES[p]})', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    # West Y
    ax = axes[1, 0]
    p = 5
    if p in active_planes:
        t_prof = np.sum(np.abs(np.array(truth_signals[p])), axis=1)
        r_prof = np.sum(np.abs(np.array(recon_signals[p])), axis=1)
        nz = np.where(t_prof > 0)[0]
        wl, wh = (max(0, nz[0]-10), min(len(t_prof), nz[-1]+11)) if len(nz) > 0 else (0, len(t_prof))
        w = np.arange(wl, wh)
        ax.plot(w, t_prof[wl:wh], 'b-', lw=1.5, label='Truth')
        ax.plot(w, r_prof[wl:wh], 'r--', lw=1.5, label='Recon')
    ax.set_title(f'Signal Comparison ({PLANE_NAMES[p]})', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    # Energy trajectory
    ax = axes[1, 1]
    # Compute from checkpoints
    cp_energies = [float(checkpoint_params[i][:, 3].sum()) for i in range(0, len(checkpoint_steps), 10)]
    ax.plot(cp_energies, 'k-', lw=1.5)
    ax.axhline(truth_total_de, color='b', ls=':', lw=1.5, label=f'Truth={truth_total_de:.0f}')
    ax.set_title('Total Energy Trajectory', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[2, 0]
    alive_de = final_params_np[:, 3][final_params_np[:, 3] > DEFAULTS['death_thresh']] * 1000
    ax.hist(alive_de, bins=100, alpha=0.7, color='steelblue',
            label=f'Alive ({len(alive_de):,})')
    ax.axvline(DEFAULTS['death_thresh'] * 1000, color='r', ls='--', lw=1.5,
               label=f'Death thresh ({n_dead_final:,} dead)')
    ax.set_xlabel('dE (keV)', fontsize=14)
    ax.set_title('Final Segment Energy Distribution', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    axes[2, 1].axis('off')

    fig.suptitle(f'20k segments, Sobolev s={SOBOLEV_S}  |  {total_steps} steps\n'
                 f'loss={checkpoint_losses[-1]:.6f}, dE_ratio={de_ratio:.3f}, '
                 f'dead={n_dead_final}', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'run_20k_s{SOBOLEV_S}_4000steps.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == '__main__':
    main()
