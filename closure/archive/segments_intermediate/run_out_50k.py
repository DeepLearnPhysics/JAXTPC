"""
Run out.h5 event 2 with 50k segments, s=1.0, dx=0.3mm matched, best hyperparameters.

Usage:
    python3 closure_analysis_full/sweeps/run_out_50k.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax, jax.numpy as jnp, numpy as np, optax
import matplotlib.pyplot as plt

from tools.geometry import generate_detector
from tools.config import SegmentData, DepositData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from tools.recombination import extract_recombination_params
from closure_analysis_full.full_closure import DEFAULTS, PLANE_NAMES, build_loss_fn, relocate_segments

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOBOLEV_S = 1.0
DX_MM = 0.3


def main():
    h5_path = 'out.h5'
    event_idx = 2
    n_seg = 50000
    total_steps = 2000

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

    print(f"out.h5 event {event_idx}, 50k segments, s={SOBOLEV_S}, dx={DX_MM}mm")
    print(f"lr={lr}, decay={decay_rate}, e_mult={lr_e_mult}")
    print(f"warmup={warmup}, noise={noise_lr}, reloc={reloc_every}/{max_reloc}")
    print(f"steps={total_steps}\n")

    # Setup
    deposit_data = load_particle_step_data(h5_path, event_idx)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_total_de = float(np.sum(deposit_data.de))
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)
    print(f"Truth: {n_truth:,} segs, dE={truth_total_de:.1f} MeV, e_scale={n_truth/n_seg:.2f}")

    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    # Force dx in truth
    forced_deposit = DepositData(
        positions_mm=deposit_data.positions_mm,
        de=deposit_data.de,
        dx=np.full(n_truth, DX_MM, dtype=np.float32),
        valid_mask=deposit_data.valid_mask,
        theta=deposit_data.theta,
        phi=deposit_data.phi,
        track_ids=deposit_data.track_ids,
    )

    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model=recomb_model)
    t0 = time.time()
    response_signals, _ = sim_truth(forced_deposit)
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
    planes_tuple = tuple(active_planes)

    print(f"Active planes: {[PLANE_NAMES[p] for p in active_planes]}")
    for p in active_planes:
        print(f"  {PLANE_NAMES[p]}: {truth_signals[p].shape}, "
              f"sum|sig|={float(jnp.sum(jnp.abs(truth_signals[p]))):.0f}")

    # Recomb constants
    _, dens, _, alpha_r, beta_r = extract_recombination_params(detector_config)
    field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
    B_eff = beta_r / dens / field_kVcm
    dx_cm = DX_MM / 10.0
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, dx_cm)

    # Optimizer sim
    print(f"\nBuilding differentiable simulator (n_seg={n_seg:,})...")
    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=recomb_model)
    fwd_opt = sim_opt.build_forward(dx_mm=DX_MM)

    loss_fn_inner = build_loss_fn(fwd_opt, truth_signals, spectral_weights, active_planes)
    grad_fn = jax.value_and_grad(loss_fn_inner, has_aux=True)

    # Init
    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    replace = n_seg > n_truth
    indices = rng.choice(n_truth, size=n_seg, replace=replace)
    init_pos = truth_pos[indices].copy() + rng.normal(0, 100.0, size=(n_seg, 3))
    init_de = truth_de[indices].copy() * e_scale * rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))
    print(f"Init total dE={float(jnp.sum(init_params[:, 3])):.1f} MeV")

    # JIT
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

    losses = []
    total_energies = []
    dead_counts = []
    cumulative_relocs = 0
    print_every = max(50, total_steps // 30)
    t_start = time.time()

    for step in range(total_steps):
        (total_loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(lr_e_mult)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], 0.001))

        if noise_lr > 0:
            lr_cur = float(schedule(step))
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                lr_cur * noise_lr * jax.random.normal(nk, shape=(n_seg, 3)))

        if step >= warmup and step % reloc_every == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key, recomb_constants, max_reloc)
            cumulative_relocs += int(n_reloc)

        loss_val = float(total_loss)
        losses.append(loss_val)

        if step % print_every == 0 or step == total_steps - 1:
            n_dead = int(jnp.sum(params[:, 3] <= DEFAULTS['death_thresh']))
            total_e = float(jnp.sum(params[:, 3]))
            total_energies.append(total_e)
            dead_counts.append(n_dead)
            elapsed = time.time() - t_start
            print(f"  Step {step:5d}: loss={loss_val:.6f}  "
                  f"total_dE={total_e:.1f}  dead={n_dead}  "
                  f"relocs={cumulative_relocs}  ({elapsed:.0f}s)")
        else:
            total_energies.append(total_energies[-1] if total_energies else 0)
            dead_counts.append(dead_counts[-1] if dead_counts else 0)

    total_time = time.time() - t_start
    final_params = np.array(params)
    final_de_total = float(np.sum(final_params[:, 3]))
    de_ratio = final_de_total / truth_total_de
    n_dead_final = int(np.sum(final_params[:, 3] <= DEFAULTS['death_thresh']))

    print(f"\nDone in {total_time:.0f}s ({total_time/total_steps*1000:.0f} ms/step)")
    print(f"Final: loss={losses[-1]:.6f}, dE_ratio={de_ratio:.3f}, "
          f"dead={n_dead_final}, relocs={cumulative_relocs}")

    # Recon signals
    seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
    recon_signals = fwd_opt(seg)

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    # Loss
    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.0, alpha=0.7)
    ax.axvline(warmup, color='gray', ls='--', lw=1.0, alpha=0.5)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(f'Loss (s={SOBOLEV_S}, dx={DX_MM}mm)', fontsize=15)
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
    ax.set_title(f'Signal ({PLANE_NAMES[p]})', fontsize=15)
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
    ax.set_title(f'Signal ({PLANE_NAMES[p]})', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    # Energy trajectory
    ax = axes[1, 1]
    ax.plot(np.array(total_energies)/1000, 'k-', lw=1.5)
    ax.axhline(truth_total_de/1000, color='b', ls=':', lw=1.5,
               label=f'Truth={truth_total_de/1000:.1f} GeV')
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Total Energy (GeV)', fontsize=14)
    ax.set_title('Total Energy', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[2, 0]
    alive_de = final_params[:, 3][final_params[:, 3] > DEFAULTS['death_thresh']] * 1000
    ax.hist(alive_de, bins=100, alpha=0.7, color='steelblue',
            label=f'Alive ({len(alive_de):,})')
    ax.axvline(DEFAULTS['death_thresh'] * 1000, color='r', ls='--', lw=1.5,
               label=f'Dead ({n_dead_final:,})')
    ax.set_xlabel('dE (keV)', fontsize=14)
    ax.set_title('Segment Energy Distribution', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    axes[2, 1].axis('off')

    fig.suptitle(f'out.h5 event {event_idx} | 50k segs, s={SOBOLEV_S}, dx={DX_MM}mm | '
                 f'{total_steps} steps\n'
                 f'loss={losses[-1]:.6f}, dE_ratio={de_ratio:.3f}, dead={n_dead_final}',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'run_out_50k.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

    # Save
    np.savez_compressed(os.path.join(OUT_DIR, 'run_out_50k.npz'),
        losses=np.array(losses), total_energies=np.array(total_energies),
        dead_counts=np.array(dead_counts), final_params=final_params,
        truth_total_de=truth_total_de, de_ratio=de_ratio, n_seg=n_seg)
    print("Saved run_out_50k.npz")


if __name__ == '__main__':
    main()
