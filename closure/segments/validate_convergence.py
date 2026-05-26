"""
Convergence validation: does the optimizer converge from practical initialization?

Tests at both dx=0.1mm and dx=0.5mm, with simple dE-scaled subsampling.
Runs a short optimization (200 steps) to verify loss decreases.

Run from project root:
    python3 -m closure.segments.validate_convergence --data mpvmpr_20.h5
"""

import argparse, time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight

PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']


def build_truth(raw, detector_config, dx_mm, recomb_model):
    """Generate truth signals with forced dx."""
    n_truth = raw['positions_mm'].shape[0]
    total_pad = max(200_000, n_truth + 1000)
    total_pad = ((total_pad + 9999) // 10000) * 10000
    sim = DetectorSimulator(
        detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False,
        include_electronics=False, include_track_hits=False,
        recombination_model=recomb_model)

    deposits = build_deposit_data(
        raw['positions_mm'], raw['de'],
        dx=np.full(n_truth, dx_mm, dtype=np.float32),
        sim_config=sim.config,
        theta=raw['theta'], phi=raw['phi'],
        track_ids=raw['track_ids'],
        t0_us=raw['t0_us'],
        interaction_ids=raw['interaction_ids'],
        root_track_ids=raw['root_track_ids'],
        pdg=raw['pdg'])

    response_signals, _, _ = sim.process_event(deposits)

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
    return truth_signals, spectral_weights, tuple(active_planes)


def init_params(truth_pos, truth_de, n_seg, jitter_mm=0.0, e_jitter=0.0, seed=42):
    """Simple practical initialization: random subsample + dE scaling."""
    rng = np.random.RandomState(seed)
    n_truth = len(truth_de)
    e_scale = n_truth / n_seg
    replace = n_seg > n_truth
    indices = rng.choice(n_truth, size=n_seg, replace=replace)
    pos = truth_pos[indices].copy()
    de = truth_de[indices].copy() * e_scale
    if jitter_mm > 0:
        pos += rng.normal(0, jitter_mm, size=(n_seg, 3))
    if e_jitter > 0:
        de *= rng.uniform(1.0 - e_jitter, 1.0 + e_jitter, size=n_seg)
    de = np.maximum(de, 1e-6)
    return jnp.array(np.column_stack([pos, de]))


def test_dx(dx_mm, raw, detector_config, n_seg, recomb_model, opt_steps):
    """Test convergence at a given dx."""
    truth_pos = np.asarray(raw['positions_mm'])
    truth_de = np.asarray(raw['de'])

    print(f"\n{'='*60}")
    print(f"  dx = {dx_mm}mm, n_seg = {n_seg}, recomb = {recomb_model}")
    print(f"{'='*60}")

    # Truth signals
    print("  Generating truth signals...", flush=True)
    t0 = time.time()
    truth_signals, spectral_weights, planes_tuple = build_truth(
        raw, detector_config, dx_mm, recomb_model)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Optimizer simulator
    sim_opt = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=recomb_model)
    sim_params = sim_opt.default_sim_params

    def loss_fn(params):
        pos = params[:, :3]
        de = params[:, 3]
        sigs = sim_opt.forward_segments(sim_params, pos, de, dx=dx_mm)
        return sobolev_loss_geomean_log1p(
            sigs, truth_signals, spectral_weights, planes=planes_tuple)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # --- Test A: at truth positions (no jitter) ---
    print("\n  [A] At truth positions (no jitter, dE-scaled):")
    params_truth = init_params(truth_pos, truth_de, n_seg, jitter_mm=0, e_jitter=0)
    t0 = time.time()
    loss_truth, grads_truth = grad_fn(params_truth)
    jax.block_until_ready(grads_truth)
    print(f"      JIT compiled in {time.time()-t0:.1f}s")
    print(f"      Loss: {float(loss_truth):.4f}")

    # Signal ratios
    sigs_t = sim_opt.forward_segments(
        sim_params, params_truth[:, :3], params_truth[:, 3], dx=dx_mm)
    for p in planes_tuple:
        t_sum = float(jnp.sum(jnp.abs(truth_signals[p])))
        s_sum = float(jnp.sum(jnp.abs(sigs_t[p])))
        print(f"      {PLANE_NAMES[p]}: signal ratio = {s_sum/t_sum:.4f}")

    # Gradient analysis
    alive = np.array(params_truth[:, 3]) > 0.001
    pos_g = np.abs(np.array(grads_truth[:, :3])[alive])
    e_g = np.array(grads_truth[:, 3])[alive]
    ratio = np.mean(np.abs(e_g)) / np.mean(pos_g)
    neg_frac = np.mean(e_g < 0)
    print(f"      Grad ratio e/pos: {ratio:.1f}")
    print(f"      Energy grad: {neg_frac*100:.1f}% want dE up, "
          f"{(1-neg_frac)*100:.1f}% want dE down")
    print(f"      Suggested lr_e_mult at lr=0.5: ~{0.5/ratio:.5f}")

    # --- Test B: at perturbed positions ---
    print("\n  [B] At perturbed positions (+100mm jitter, +/-80% energy):")
    params_pert = init_params(truth_pos, truth_de, n_seg,
                               jitter_mm=100.0, e_jitter=0.8)
    loss_pert, grads_pert = grad_fn(params_pert)
    print(f"      Loss: {float(loss_pert):.4f} ({float(loss_pert)/float(loss_truth):.1f}x vs truth pos)")

    alive_p = np.array(params_pert[:, 3]) > 0.001
    pos_gp = np.abs(np.array(grads_pert[:, :3])[alive_p])
    e_gp = np.array(grads_pert[:, 3])[alive_p]
    ratio_p = np.mean(np.abs(e_gp)) / np.mean(pos_gp)
    neg_frac_p = np.mean(e_gp < 0)
    print(f"      Grad ratio e/pos: {ratio_p:.1f}")
    print(f"      Energy grad: {neg_frac_p*100:.1f}% want dE up, "
          f"{(1-neg_frac_p)*100:.1f}% want dE down")

    # Gradient direction check
    displacement = np.array(params_truth[:, :3] - params_pert[:, :3])
    grad_pos_p = np.array(grads_pert[:, :3])
    dot = np.sum(-grad_pos_p * displacement, axis=1)
    norms = np.linalg.norm(grad_pos_p, axis=1) * np.linalg.norm(displacement, axis=1)
    valid = norms > 1e-10
    cos_angles = dot[valid] / norms[valid]
    print(f"      Pos grads toward truth: {np.mean(cos_angles > 0)*100:.1f}%")
    print(f"      Mean cos(angle): {np.mean(cos_angles):.3f}")

    # --- Test C: short optimization ---
    print(f"\n  [C] Short optimization ({opt_steps} steps):")
    lr = 0.5
    lr_e_mult = max(0.5 / ratio_p, 0.0001)  # auto from gradient ratio
    print(f"      lr={lr}, lr_e_mult={lr_e_mult:.5f} (auto from grad ratio)")

    schedule = optax.exponential_decay(
        init_value=lr, transition_steps=1, decay_rate=0.9995)
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)

    params = params_pert
    opt_state = optimizer.init(params)

    losses = []
    for step in range(opt_steps):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(lr_e_mult)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], 1e-6))

        loss_val = float(loss)
        losses.append(loss_val)
        if step % 50 == 0 or step == opt_steps - 1:
            total_de = float(jnp.sum(params[:, 3]))
            print(f"      Step {step:4d}: loss={loss_val:.4f}, total_dE={total_de:.2f}")

    losses = np.array(losses)
    converging = losses[-1] < losses[0]
    monotonic = np.all(np.diff(losses[:20]) < 0)  # first 20 steps decreasing?
    print(f"\n      Initial loss: {losses[0]:.4f}")
    print(f"      Final loss:   {losses[-1]:.4f}")
    print(f"      Reduction:    {losses[0]/losses[-1]:.1f}x")
    print(f"      Converging:   {'YES' if converging else 'NO'}")
    print(f"      First 20 steps monotonic: {'YES' if monotonic else 'NO'}")

    return {
        'dx_mm': dx_mm,
        'loss_truth_pos': float(loss_truth),
        'loss_perturbed': float(loss_pert),
        'loss_final': losses[-1],
        'grad_ratio': ratio_p,
        'lr_e_mult': lr_e_mult,
        'converging': converging,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate convergence')
    parser.add_argument('--data', required=True, help='Path to HDF5 event file')
    parser.add_argument('--config', default='config/cubic_wireplane_config.yaml',
                        help='Detector config YAML')
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--n-seg', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--recomb', type=str, default='modified_box',
                        choices=['modified_box', 'emb'])
    args = parser.parse_args()

    raw = load_particle_step_data(args.data, args.event, verbose=False)
    detector_config = generate_detector(args.config)

    results = []
    for dx_mm in [0.1, 0.5]:
        r = test_dx(dx_mm, raw, detector_config,
                    args.n_seg, args.recomb, args.steps)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'dx_mm':>6} {'Loss@truth':>12} {'Loss@pert':>12} {'Loss@final':>12} "
          f"{'Reduction':>10} {'grad e/pos':>10} {'lr_e_mult':>10}")
    print("-" * 75)
    for r in results:
        red = r['loss_perturbed'] / r['loss_final']
        print(f"{r['dx_mm']:>6.1f} {r['loss_truth_pos']:>12.4f} "
              f"{r['loss_perturbed']:>12.4f} {r['loss_final']:>12.4f} "
              f"{red:>10.1f}x {r['grad_ratio']:>10.1f} {r['lr_e_mult']:>10.5f}")


if __name__ == '__main__':
    main()
