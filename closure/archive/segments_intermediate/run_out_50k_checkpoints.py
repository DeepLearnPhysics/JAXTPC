"""Run out.h5 event 2 with 50k segments, saving checkpoints for 3D viz."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax, jax.numpy as jnp, numpy as np, optax
from tools.geometry import generate_detector
from tools.config import SegmentData, DepositData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from tools.recombination import extract_recombination_params
from closure_analysis_full.full_closure import DEFAULTS, PLANE_NAMES, build_loss_fn, relocate_segments

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    n_seg = 50000
    total_steps = 2000
    save_every = 10
    DX_MM = 0.3

    deposit_data = load_particle_step_data('out.h5', 2)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)
    print(f"Truth: {n_truth:,} segs, e_scale={n_truth/n_seg:.2f}")

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
            sob_dict[pidx] = make_sobolev_weight(*signal.shape, s=1.0)
    active_planes.sort()
    truth_signals = tuple(truth_dict.get(i, jnp.zeros((1,1))) for i in range(6))
    spectral_weights = tuple(sob_dict.get(i, jnp.zeros((1,1))) for i in range(6))

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

    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy() + rng.normal(0, 100.0, size=(n_seg, 3))
    init_de = truth_de[indices].copy() * e_scale * rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    print("JIT warmup...")
    _ = grad_fn(init_params)
    print("Done.\n")

    schedule = optax.exponential_decay(init_value=1.0, transition_steps=1, decay_rate=0.999)
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)
    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    n_checkpoints = total_steps // save_every + 1
    checkpoint_steps, checkpoint_losses = [], []
    checkpoint_params = np.zeros((n_checkpoints, n_seg, 4), dtype=np.float32)
    cumulative_relocs = 0
    t_start = time.time()

    for step in range(total_steps):
        (total_loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(0.003)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], 0.001))

        lr_cur = float(schedule(step))
        rng_key, nk = jax.random.split(rng_key)
        params = params.at[:, :3].add(lr_cur * 0.3 * jax.random.normal(nk, shape=(n_seg, 3)))

        if step >= 100 and step % 25 == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key, recomb_constants, 100)
            cumulative_relocs += int(n_reloc)

        if step % save_every == 0:
            idx = step // save_every
            checkpoint_steps.append(step)
            checkpoint_losses.append(float(total_loss))
            checkpoint_params[idx] = np.array(params)

        if step % 200 == 0 or step == total_steps - 1:
            print(f"  Step {step:5d}: loss={float(total_loss):.6f}  "
                  f"relocs={cumulative_relocs}  ({time.time()-t_start:.0f}s)")

    save_path = os.path.join(OUT_DIR, 'best_run_out_50k_2000.npz')
    np.savez_compressed(save_path,
        checkpoint_steps=np.array(checkpoint_steps),
        checkpoint_losses=np.array(checkpoint_losses),
        checkpoint_params=checkpoint_params[:len(checkpoint_steps)],
        truth_pos=truth_pos, truth_de=truth_de,
        n_truth=n_truth, n_seg=n_seg, total_steps=total_steps,
        truth_total_de=float(np.sum(truth_de)),
        dx_mm=DX_MM, B_eff=B_eff, alpha=alpha_r,
    )
    print(f"\nSaved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

if __name__ == '__main__':
    main()
