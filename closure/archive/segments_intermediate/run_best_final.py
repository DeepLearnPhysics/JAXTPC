"""
Final best config run with full diagnostics.
Saves checkpoints, computes wire signals at each checkpoint for event display animations.

Usage:
    python3 closure_analysis_full/sweeps/run_best_final.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax, jax.numpy as jnp, numpy as np, optax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
ALPHA_RECOMB = 0.93
W_ION = 23.6e-6


def main():
    n_seg = 50000
    total_steps = 2000
    save_every = 10  # checkpoints for 3D viz
    signal_save_every = 50  # wire signals for event display (heavier)

    # Best config
    lr, decay_rate, lr_e_mult = 1.0, 0.999, 0.01
    warmup, noise_lr = 100, 0.3
    reloc_every, max_reloc = 25, 1000
    track_jitter_mm = 50.0

    print(f"Final best run: {n_seg} segs, {total_steps} steps")
    print(f"lr={lr}, d={decay_rate}, e_mult={lr_e_mult}, reloc {reloc_every}/{max_reloc}")
    print(f"Track jitter {track_jitter_mm}mm, noise={noise_lr}\n")

    # Setup
    deposit_data = load_particle_step_data('out.h5', 2)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_total_de = float(np.sum(deposit_data.de))
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)
    truth_tids = np.asarray(deposit_data.track_ids)

    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    forced_deposit = DepositData(
        positions_mm=deposit_data.positions_mm, de=deposit_data.de,
        dx=np.full(n_truth, DX_MM, dtype=np.float32),
        valid_mask=deposit_data.valid_mask, theta=deposit_data.theta,
        phi=deposit_data.phi, track_ids=deposit_data.track_ids)

    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model='modified_box')
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

    # Save truth wire-summed profiles for event display
    truth_profiles = {}
    for p in active_planes:
        truth_profiles[p] = np.sum(np.abs(np.array(truth_signals[p])), axis=1)

    _, dens, _, alpha_r, beta_r = extract_recombination_params(detector_config)
    field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
    B_eff = beta_r / dens / field_kVcm
    dx_cm = DX_MM / 10.0
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, dx_cm)

    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model='modified_box')
    fwd_opt = sim_opt.build_forward(dx_mm=DX_MM)
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
    _ = grad_fn(init_params)
    print("Done.\n")

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
    checkpoint_steps, checkpoint_losses = [], []
    n_cp = total_steps // save_every + 1
    checkpoint_params = np.zeros((n_cp, n_seg, 4), dtype=np.float32)

    # dE histogram over steps (for waterfall plot)
    de_history_steps = []
    de_history = []  # list of (alive_des, n_dead) per recorded step

    # Wire signal profiles at signal_save_every
    signal_steps = []
    signal_profiles = {}  # {plane: list of profiles}
    for p in active_planes:
        signal_profiles[p] = []

    losses, q_ratios = [], []
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

        # Checkpoint (segment positions)
        if step % save_every == 0:
            idx = step // save_every
            checkpoint_steps.append(step)
            checkpoint_losses.append(loss_val)
            checkpoint_params[idx] = np.array(params)

        # dE histogram snapshot
        if step % save_every == 0:
            p_np = np.array(params[:, 3])
            alive_de = p_np[p_np > DEFAULTS['death_thresh']]
            n_dead = int(np.sum(p_np <= DEFAULTS['death_thresh']))
            de_history_steps.append(step)
            de_history.append((alive_de.copy(), n_dead))

        # Wire signal profiles (less frequent — forward pass is expensive)
        if step % signal_save_every == 0 or step == total_steps - 1:
            seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
            recon = fwd_opt(seg)
            signal_steps.append(step)
            for p in active_planes:
                prof = np.sum(np.abs(np.array(recon[p])), axis=1)
                signal_profiles[p].append(prof)

        if step % 100 == 0 or step == total_steps - 1:
            p_np = np.array(params)
            alive = p_np[:, 3] > 0.012
            sim_Q = compute_Q(p_np[alive, 3]).sum()
            q_ratio = sim_Q / truth_total_Q
            q_ratios.append((step, q_ratio))
            n_dead = int((~alive).sum())
            print(f"  Step {step:5d}: loss={loss_val:.6f}  Q={q_ratio:.3f}  "
                  f"dead={n_dead}  relocs={cumulative_relocs}  ({time.time()-t_start:.0f}s)")

    print(f"\nDone in {time.time()-t_start:.0f}s")

    # =====================================================================
    # Save everything
    # =====================================================================
    save_path = os.path.join(OUT_DIR, 'best_final.npz')
    save_data = {
        'checkpoint_steps': np.array(checkpoint_steps),
        'checkpoint_losses': np.array(checkpoint_losses),
        'checkpoint_params': checkpoint_params[:len(checkpoint_steps)],
        'truth_pos': truth_pos, 'truth_de': truth_de,
        'n_truth': n_truth, 'n_seg': n_seg, 'total_steps': total_steps,
        'truth_total_de': truth_total_de, 'dx_mm': DX_MM,
        'B_eff': B_eff, 'alpha': alpha_r,
        'de_history_steps': np.array(de_history_steps),
        'signal_steps': np.array(signal_steps),
        'losses': np.array(losses),
        'active_planes': np.array(active_planes),
    }
    # Save dE histograms
    for i, (alive_de, n_dead) in enumerate(de_history):
        save_data[f'de_hist_{i}_alive'] = alive_de
        save_data[f'de_hist_{i}_dead'] = n_dead
    save_data['n_de_hist'] = len(de_history)

    # Save signal profiles
    for p in active_planes:
        save_data[f'truth_profile_{p}'] = truth_profiles[p]
        for j, prof in enumerate(signal_profiles[p]):
            save_data[f'sig_profile_{p}_{j}'] = prof
    save_data['n_signal_snapshots'] = len(signal_steps)

    np.savez_compressed(save_path, **save_data)
    print(f"Saved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    # =====================================================================
    # dE waterfall plot
    # =====================================================================
    print("Creating dE waterfall plot...")

    # Compute 99th percentile across all steps
    all_alive = np.concatenate([h[0] for h in de_history])
    p99 = np.percentile(all_alive, 99) * 1000  # keV
    n_bins = 100
    bin_edges = np.linspace(DEFAULTS['death_thresh'] * 1000, p99, n_bins + 1)

    waterfall = np.zeros((len(de_history), n_bins + 2))  # +2 for dead and overflow
    for i, (alive_de, n_dead) in enumerate(de_history):
        alive_kev = alive_de * 1000
        hist, _ = np.histogram(alive_kev, bins=bin_edges)
        overflow = int(np.sum(alive_kev > p99))
        waterfall[i, 0] = n_dead
        waterfall[i, 1:-1] = hist
        waterfall[i, -1] = overflow

    fig, ax = plt.subplots(figsize=(14, 8))
    extent = [0, n_bins + 2, de_history_steps[-1], de_history_steps[0]]
    im = ax.imshow(waterfall, aspect='auto', extent=[0, n_bins+2, de_history_steps[-1], 0],
                   cmap='hot', norm=mcolors.LogNorm(vmin=1, vmax=waterfall.max()))

    # Labels
    ax.set_xlabel('dE bin', fontsize=14)
    ax.set_ylabel('Step', fontsize=14)
    ax.set_title(f'Segment dE Distribution Over Training (99th %ile={p99:.0f} keV)', fontsize=15)

    # Mark dead and overflow bins
    ax.axvline(0.5, color='cyan', ls='--', lw=1, alpha=0.5)
    ax.axvline(n_bins + 1.5, color='cyan', ls='--', lw=1, alpha=0.5)
    ax.text(0, -20, 'Dead', ha='center', fontsize=9, color='cyan')
    ax.text(n_bins + 2, -20, 'Over', ha='center', fontsize=9, color='cyan')

    cbar = fig.colorbar(im, ax=ax, label='Count')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'best_final_waterfall.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

    # =====================================================================
    # Progression plot (loss, Q, U, V, Y)
    # =====================================================================
    print("Creating progression plot...")
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.0, alpha=0.7)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Loss', fontsize=15); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    qs, qv = zip(*q_ratios)
    ax.plot(qs, qv, 'b-', lw=2)
    ax.axhline(1.0, color='green', ls='--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Q ratio', fontsize=14)
    ax.set_title('Charge Conservation', fontsize=15); ax.grid(True, alpha=0.3)

    # U, V, Y signal comparisons (final)
    for idx, (p, ax) in enumerate(zip([3, 4, 5], [axes[1,0], axes[1,1], axes[2,0]])):
        if p in active_planes:
            t_prof = truth_profiles[p]
            r_prof = signal_profiles[p][-1]
            nz = np.where(t_prof > 0)[0]
            wl, wh = (max(0, nz[0]-10), min(len(t_prof), nz[-1]+11)) if len(nz) > 0 else (0, len(t_prof))
            w = np.arange(wl, wh)
            ax.plot(w, t_prof[wl:wh], 'b-', lw=1.5, label='Truth')
            ax.plot(w, r_prof[wl:wh], 'r--', lw=1.5, label='Recon')
        ax.set_title(f'Signal ({PLANE_NAMES[p]})', fontsize=15)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    axes[2, 1].axis('off')
    fig.suptitle(f'Best Final | 50k segs, s=1.0, lr={lr}, reloc {reloc_every}/{max_reloc}\n'
                 f'loss={losses[-1]:.6f}, Q={qv[-1]:.3f}',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'best_final_progression.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == '__main__':
    main()
