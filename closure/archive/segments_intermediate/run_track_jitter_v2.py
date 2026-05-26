"""Per-track 50mm jitter, 50k segs, s=1.0, dx=0.3mm. More checkpoints, U/V/Y planes."""

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
W_ION = 23.6e-6
ALPHA_RECOMB = 0.93


def main():
    n_seg = 50000
    total_steps = 2000
    save_every = 5  # more checkpoints
    track_jitter_mm = 50.0
    e_jitter_frac = 0.8

    lr, decay_rate, lr_e_mult = 1.0, 0.999, 0.003
    warmup, noise_lr = 100, 0.3
    reloc_every, max_reloc = 25, 100

    print(f"Per-track jitter v2: {track_jitter_mm}mm, {n_seg} segs, {total_steps} steps")
    print(f"Checkpoints every {save_every} steps\n")

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
    response_signals, _ = sim_truth(forced_deposit)

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
    dx_cm = DX_MM / 10.0
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, dx_cm)

    # Per-track jitter init
    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy()
    init_de = truth_de[indices].copy() * e_scale
    init_tids = truth_tids[indices]

    unique_tids = np.unique(init_tids)
    for tid in unique_tids:
        disp = rng.normal(0, track_jitter_mm, size=3)
        mask = init_tids == tid
        init_pos[mask] += disp

    init_de *= rng.uniform(1.0 - e_jitter_frac, 1.0 + e_jitter_frac, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model='modified_box')
    fwd_opt = sim_opt.build_forward(dx_mm=DX_MM)

    def loss_fn(params):
        seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
        sigs = fwd_opt(seg)
        return sobolev_loss_geomean_log1p(sigs, truth_signals, spectral_weights,
                                           planes=planes_tuple), None

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    print("JIT warmup...")
    _ = grad_fn(init_params)
    print("Done.\n")

    schedule = optax.exponential_decay(init_value=lr, transition_steps=1, decay_rate=decay_rate)
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)
    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    n_checkpoints = total_steps // save_every + 1
    checkpoint_steps, checkpoint_losses = [], []
    checkpoint_params = np.zeros((n_checkpoints, n_seg, 4), dtype=np.float32)

    def compute_Q(de_arr):
        return (dx_cm / B_eff) * np.log(np.maximum(ALPHA_RECOMB + B_eff * de_arr / dx_cm, 1.0))

    truth_total_Q = compute_Q(truth_de).sum()
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

        if step % save_every == 0:
            idx = step // save_every
            checkpoint_steps.append(step)
            checkpoint_losses.append(loss_val)
            checkpoint_params[idx] = np.array(params)

        if step % 100 == 0 or step == total_steps - 1:
            p_np = np.array(params)
            alive = p_np[:, 3] > 0.012
            sim_Q = compute_Q(p_np[alive, 3]).sum()
            q_ratio = sim_Q / truth_total_Q
            q_ratios.append((step, q_ratio))
            print(f"  Step {step:5d}: loss={loss_val:.6f}  Q_ratio={q_ratio:.3f}  "
                  f"dead={int((~alive).sum())}  ({time.time()-t_start:.0f}s)")

    final_params = np.array(params)
    print(f"\nDone in {time.time()-t_start:.0f}s")

    # Save checkpoints
    save_path = os.path.join(OUT_DIR, 'track_jitter_v2_50k.npz')
    np.savez_compressed(save_path,
        checkpoint_steps=np.array(checkpoint_steps),
        checkpoint_losses=np.array(checkpoint_losses),
        checkpoint_params=checkpoint_params[:len(checkpoint_steps)],
        truth_pos=truth_pos, truth_de=truth_de,
        n_truth=n_truth, n_seg=n_seg, total_steps=total_steps,
        truth_total_de=truth_total_de, dx_mm=DX_MM, B_eff=B_eff, alpha=alpha_r,
        track_jitter_mm=track_jitter_mm,
    )
    print(f"Saved {save_path}")

    # Recon signals
    seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
    recon_signals = fwd_opt(seg)

    # Plot: 3x2 — loss, Q ratio, west_U, west_V, west_Y, energy hist
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    # Loss
    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.0, alpha=0.7)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(f'Loss (track jitter {track_jitter_mm}mm)', fontsize=15)
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # Q ratio
    ax = axes[0, 1]
    q_steps, q_vals = zip(*q_ratios)
    ax.plot(q_steps, q_vals, 'b-', lw=2, label='Q ratio (sim/truth)')
    ax.axhline(1.0, color='green', ls='--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Step', fontsize=14); ax.set_ylabel('Q Ratio', fontsize=14)
    ax.set_title('Charge Conservation', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # West U
    ax = axes[1, 0]
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
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # West V
    ax = axes[1, 1]
    p = 4
    if p in active_planes:
        t_prof = np.sum(np.abs(np.array(truth_signals[p])), axis=1)
        r_prof = np.sum(np.abs(np.array(recon_signals[p])), axis=1)
        nz = np.where(t_prof > 0)[0]
        wl, wh = (max(0, nz[0]-10), min(len(t_prof), nz[-1]+11)) if len(nz) > 0 else (0, len(t_prof))
        w = np.arange(wl, wh)
        ax.plot(w, t_prof[wl:wh], 'b-', lw=1.5, label='Truth')
        ax.plot(w, r_prof[wl:wh], 'r--', lw=1.5, label='Recon')
    ax.set_title(f'Signal ({PLANE_NAMES[p]})', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # West Y
    ax = axes[2, 0]
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
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    # Energy histogram
    ax = axes[2, 1]
    alive_de = final_params[:, 3][final_params[:, 3] > 0.012] * 1000
    n_dead = len(final_params) - len(alive_de)
    ax.hist(alive_de, bins=100, alpha=0.7, color='steelblue',
            label=f'Alive ({len(alive_de):,})')
    ax.axvline(0.012 * 1000, color='r', ls='--', lw=1.5, label=f'Dead ({n_dead:,})')
    ax.set_xlabel('dE (keV)', fontsize=14)
    ax.set_title('Segment Energy Distribution', fontsize=15)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=11)

    fig.suptitle(f'out.h5 ev2 | 50k segs, per-track {track_jitter_mm}mm jitter, s={SOBOLEV_S}\n'
                 f'loss={losses[-1]:.6f}, Q_ratio={q_vals[-1]:.3f}',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'track_jitter_v2_50k.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == '__main__':
    main()
