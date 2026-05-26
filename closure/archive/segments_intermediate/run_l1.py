"""L1 sweep at best config with noise_lr=0.3."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax, jax.numpy as jnp, numpy as np
from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight
from tools.recombination import extract_recombination_params
from closure_analysis_full.full_closure import DEFAULTS, PLANE_NAMES, build_loss_fn, run_training_loop

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'mpvmpr_20.h5'
    n_seg = 10000

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

    results = []
    for l1 in [0.0, 0.00005, 0.0001]:
        cfg = dict(DEFAULTS)
        cfg.update({'lr': 1.0, 'decay_rate': 0.999, 'lr_e_mult': 0.003,
                    'warmup': 100, 'noise_lr': 0.3, 'l1': l1,
                    'reloc_every': 25, 'max_reloc': 100,
                    'recomb_constants': recomb_constants})

        print(f"[L1={l1}]...")
        t0 = time.time()
        result = run_training_loop(grad_fn, init_params, n_seg, 1500, 'full', cfg)
        dt = time.time() - t0
        final_de = float(np.sum(result['final_params'][:, 3]))
        de_ratio = final_de / truth_total_de
        print(f"  loss={result['losses'][-1]:.6f}  dE_ratio={de_ratio:.3f}  "
              f"dead={result['dead_counts'][-1]}  {dt:.0f}s\n")
        results.append({'l1': l1, 'loss': result['losses'][-1],
                        'de_ratio': de_ratio, 'dead': result['dead_counts'][-1],
                        'losses': result['losses']})

    print(f"\n{'='*50}")
    print(f"L1 SWEEP — noise_lr=0.3, 1500 steps")
    print(f"{'='*50}")
    print(f"{'L1':>10} {'Loss':>10} {'dE_ratio':>10} {'Dead':>6}")
    print("-" * 40)
    for r in results:
        print(f"{r['l1']:>10.5f} {r['loss']:>10.6f} {r['de_ratio']:>10.3f} {r['dead']:>6d}")

    save_path = os.path.join(OUT_DIR, 'sweep_l1.npz')
    save_data = {}
    for r in results:
        save_data[f"l1_{r['l1']}_losses"] = r['losses']
        save_data[f"l1_{r['l1']}_loss"] = r['loss']
        save_data[f"l1_{r['l1']}_de_ratio"] = r['de_ratio']
    np.savez(save_path, **save_data)
    print(f"\nSaved to {save_path}")

if __name__ == '__main__':
    main()
