"""Extended noise sweep: higher noise_lr + constant noise test."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax, jax.numpy as jnp, numpy as np, optax
from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight
from tools.recombination import extract_recombination_params
from closure_analysis_full.full_closure import (
    DEFAULTS, PLANE_NAMES, build_loss_fn, relocate_segments,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_with_noise(grad_fn, init_params, n_seg, total_steps, cfg,
                   constant_noise_mm=None):
    """Like run_training_loop but supports constant noise option."""
    schedule = optax.exponential_decay(
        init_value=cfg['lr'], transition_steps=1, decay_rate=cfg['decay_rate'])
    optimizer = optax.adam(schedule, b1=cfg['b1'], b2=cfg['b2'])

    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)
    losses = []
    total_energies = []
    dead_counts = []
    cumulative_relocs = 0
    max_reloc = cfg.get('max_reloc', 100)
    reloc_every = cfg.get('reloc_every', 25)

    for step in range(total_steps):
        (total_loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(cfg['lr_e_mult'])
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], cfg['min_energy']))

        # Noise: either LR-coupled or constant
        if constant_noise_mm is not None and constant_noise_mm > 0:
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                constant_noise_mm * jax.random.normal(nk, shape=(n_seg, 3)))
        elif cfg.get('noise_lr', 0) > 0:
            lr_cur = float(schedule(step))
            noise_scale = lr_cur * cfg['noise_lr']
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                noise_scale * jax.random.normal(nk, shape=(n_seg, 3)))

        # Relocation
        if step >= cfg['warmup'] and step % reloc_every == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key,
                cfg['recomb_constants'], max_reloc)
            cumulative_relocs += int(n_reloc)

        loss_val = float(total_loss)
        losses.append(loss_val)

        if step % 100 == 0 or step == total_steps - 1:
            n_dead = int(jnp.sum(params[:, 3] <= cfg['death_thresh']))
            total_e = float(jnp.sum(params[:, 3]))
            total_energies.append(total_e)
            dead_counts.append(n_dead)
            print(f"  Step {step:5d}: loss={loss_val:.6f}  "
                  f"total_dE={total_e:.2f}  dead={n_dead}  "
                  f"relocs={cumulative_relocs}")
        else:
            total_energies.append(total_energies[-1] if total_energies else 0)
            dead_counts.append(dead_counts[-1] if dead_counts else 0)

    return {
        'losses': np.array(losses),
        'total_energies': np.array(total_energies),
        'final_params': np.array(params),
        'dead_counts': np.array(dead_counts),
        'cumulative_relocs': cumulative_relocs,
    }


def main():
    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'mpvmpr_20.h5'
    n_seg = 10000
    total_steps = 1500

    deposit_data = load_particle_step_data(h5_path, 0)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_total_de = float(np.sum(deposit_data.de))
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model='modified_box')
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
    truth_signals = tuple(truth_dict.get(i, jnp.zeros((1,1))) for i in range(6))
    spectral_weights = tuple(sob_dict.get(i, jnp.zeros((1,1))) for i in range(6))

    _, dens, _, alpha_r, beta_r = extract_recombination_params(detector_config)
    field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
    B_eff = beta_r / dens / field_kVcm
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, 0.05)

    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg,
        recombination_model='modified_box')
    fwd_opt = sim_opt.build_forward()
    loss_fn = build_loss_fn(fwd_opt, truth_signals, spectral_weights, active_planes)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    rng = np.random.RandomState(42)
    truth_pos, truth_de = np.asarray(deposit_data.positions_mm), np.asarray(deposit_data.de)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy() + rng.normal(0, 100.0, size=(n_seg, 3))
    init_de = truth_de[indices].copy() * e_scale * rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    print("JIT warmup...")
    _ = grad_fn(init_params)
    print("Done.\n")

    base_cfg = dict(DEFAULTS)
    base_cfg.update({'lr': 1.0, 'decay_rate': 0.999, 'lr_e_mult': 0.003,
                     'warmup': 100, 'l1': 0.0, 'reloc_every': 25, 'max_reloc': 100,
                     'recomb_constants': recomb_constants})

    combos = [
        {'name': 'lr_coupled_0.5', 'noise_lr': 0.5, 'constant': None},
        {'name': 'lr_coupled_1.0', 'noise_lr': 1.0, 'constant': None},
        {'name': 'constant_0.1mm', 'noise_lr': 0.0, 'constant': 0.1},
    ]

    results = []
    for combo in combos:
        cfg = dict(base_cfg)
        cfg['noise_lr'] = combo['noise_lr']
        const = combo['constant']

        desc = f"noise_lr={combo['noise_lr']}" if const is None else f"constant={const}mm"
        print(f"[{combo['name']}] {desc}...")
        t0 = time.time()
        result = run_with_noise(grad_fn, init_params, n_seg, total_steps, cfg,
                                constant_noise_mm=const)
        dt = time.time() - t0
        final_de = float(np.sum(result['final_params'][:, 3]))
        de_ratio = final_de / truth_total_de
        final_loss = result['losses'][-1]
        print(f"  loss={final_loss:.6f}  dE_ratio={de_ratio:.3f}  "
              f"dead={result['dead_counts'][-1]}  {dt:.0f}s\n")
        results.append({'name': combo['name'], 'loss': final_loss,
                        'de_ratio': de_ratio, 'dead': result['dead_counts'][-1],
                        'losses': result['losses']})

    print(f"\n{'='*60}")
    print(f"EXTENDED NOISE SWEEP — 1500 steps")
    print(f"{'='*60}")
    print(f"Previous results for context:")
    print(f"  noise_lr=0.0 (lr-coupled): loss=0.027, dE_ratio=1.032")
    print(f"  noise_lr=0.1 (lr-coupled): loss=0.024, dE_ratio=1.024")
    print(f"  noise_lr=0.3 (lr-coupled): loss=0.022, dE_ratio=1.013")
    print(f"\nNew results:")
    print(f"{'Name':<25} {'Loss':>10} {'dE_ratio':>10} {'Dead':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<25} {r['loss']:>10.6f} {r['de_ratio']:>10.3f} {r['dead']:>6d}")

    save_path = os.path.join(OUT_DIR, 'sweep_noise2.npz')
    save_data = {}
    for r in results:
        save_data[f"{r['name']}_losses"] = r['losses']
        save_data[f"{r['name']}_loss"] = r['loss']
    np.savez(save_path, **save_data)
    print(f"\nSaved to {save_path}")

if __name__ == '__main__':
    main()
