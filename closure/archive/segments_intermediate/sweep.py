"""
Hyperparameter sweep for full event closure analysis.

Runs multiple hyperparameter combinations with shared truth signals
and JIT-compiled gradient function. Truth generation runs once;
optimizer runs per combo. Results saved to NPZ for later analysis.

Run from project root:
    python3 closure_analysis_full/sweep.py mpvmpr_20.h5 --n-seg 10000 --steps 1000
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight
from closure_analysis_full.full_closure import (
    DEFAULTS, PLANE_NAMES,
    build_loss_fn, run_training_loop,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# lr x decay sweep with relocation, no noise/L1
COMBOS = [
    {'name': 'lr10_d999',  'lr': 1.0, 'decay_rate': 0.999},
]


def main():
    parser = argparse.ArgumentParser(
        description='Full closure hyperparameter sweep')
    parser.add_argument('h5_path', help='Path to HDF5 event file')
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--n-seg', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--mode', choices=['baseline', 'noise', 'full'],
                        default='baseline')
    parser.add_argument('--recomb', type=str, default='modified_box',
                        choices=['modified_box', 'emb'])
    args = parser.parse_args()

    n_seg = args.n_seg
    total_steps = args.steps
    mode = args.mode

    print(f"Full Closure Sweep — {len(COMBOS)} combos, "
          f"{total_steps} steps, mode={mode}")
    print(f"File: {args.h5_path}, event={args.event}, n_seg={n_seg:,}")
    print(f"Recomb: {args.recomb}\n")

    # =================================================================
    # Shared setup (runs once)
    # =================================================================

    print("Loading event data...")
    deposit_data = load_particle_step_data(args.h5_path, args.event)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_total_de = float(np.sum(deposit_data.de))
    print(f"  {n_truth:,} segments, total dE={truth_total_de:.2f} MeV")

    print("Generating truth signals...")
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = max(200_000, n_truth + 1000)
    total_pad = ((total_pad + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(
        detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False,
        include_electronics=False, include_track_hits=False,
        recombination_model=args.recomb)

    t0 = time.time()
    response_signals, _ = sim_truth(deposit_data)
    print(f"  Truth sim done in {time.time() - t0:.1f}s")

    # Build 6-element tuples
    truth_signals_dict = {}
    active_planes = []
    sob_weights_dict = {}
    for (side, plane), signal in response_signals.items():
        plane_idx = side * 3 + plane
        signal = jnp.asarray(signal)
        truth_signals_dict[plane_idx] = signal
        if jnp.any(signal != 0):
            active_planes.append(plane_idx)
            sob_weights_dict[plane_idx] = make_sobolev_weight(*signal.shape, s=1.5)
    active_planes.sort()

    truth_signals = tuple(
        truth_signals_dict.get(i, jnp.zeros((1, 1))) for i in range(6))
    spectral_weights = tuple(
        sob_weights_dict.get(i, jnp.zeros((1, 1))) for i in range(6))
    print(f"  Active planes: {[PLANE_NAMES[p] for p in active_planes]}")

    # Differentiable optimizer simulator
    print(f"Building differentiable simulator (n_seg={n_seg:,})...")
    sim_opt = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=args.recomb)
    fwd_opt = sim_opt.build_forward()

    # Shared loss + grad
    loss_fn = build_loss_fn(fwd_opt, truth_signals, spectral_weights,
                            active_planes)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Shared init params (subsample + jitter + e_scale)
    print("Initializing optimizer segments...")
    rng = np.random.RandomState(42)
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)

    e_scale = n_truth / n_seg
    replace = n_seg > n_truth
    indices = rng.choice(n_truth, size=n_seg, replace=replace)
    init_pos = truth_pos[indices].copy()
    init_de = truth_de[indices].copy() * e_scale

    init_pos += rng.normal(0, DEFAULTS['pos_jitter_mm'], size=(n_seg, 3))
    init_de *= rng.uniform(
        1.0 - DEFAULTS['e_jitter_frac'],
        1.0 + DEFAULTS['e_jitter_frac'], size=n_seg)
    init_de = np.maximum(init_de, DEFAULTS['min_energy'])

    init_params = jnp.array(np.column_stack([init_pos, init_de]))
    init_total = float(jnp.sum(init_params[:, 3]))
    print(f"  e_scale={e_scale:.2f}, init total dE={init_total:.2f} MeV")

    # JIT warmup
    print("Warming up JIT...")
    t0 = time.time()
    _ = grad_fn(init_params)
    print(f"JIT warm-up done in {time.time() - t0:.1f}s\n")

    # =================================================================
    # Run combos
    # =================================================================

    all_results = {}

    # Compute recomb constants for relocation (needed for full mode)
    from tools.recombination import extract_recombination_params
    _, dens, _, alpha_r, beta_r = extract_recombination_params(detector_config)
    field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
    B_eff = beta_r / dens / field_kVcm
    dx_cm = 0.05  # 0.5mm
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, dx_cm)

    for combo in COMBOS:
        cfg = dict(DEFAULTS)
        cfg['lr'] = 0.5
        cfg['decay_rate'] = 0.9995
        cfg['lr_e_mult'] = 0.003
        cfg['noise_lr'] = 0.0
        cfg['l1'] = 0.0
        cfg['reloc_every'] = 25
        cfg['max_reloc'] = 100
        cfg['recomb_model'] = args.recomb
        cfg['recomb_constants'] = recomb_constants
        cfg.update(combo)

        print(f"[{combo['name']}] lr={cfg['lr']}, decay={cfg['decay_rate']}...")
        t0 = time.time()
        result = run_training_loop(
            grad_fn, init_params, n_seg, total_steps, mode, cfg)
        dt = time.time() - t0

        final_de = float(np.sum(result['final_params'][:, 3]))
        de_ratio = final_de / truth_total_de if truth_total_de > 0 else 0

        print(f"  loss={result['losses'][-1]:.6f}  "
              f"dE_ratio={de_ratio:.3f}  "
              f"dead={result['dead_counts'][-1]}  "
              f"{dt:.0f}s\n")

        all_results[combo['name']] = {
            'losses': result['losses'],
            'total_energies': result['total_energies'],
            'dead_counts': result['dead_counts'],
            'final_loss': result['losses'][-1],
            'de_ratio': de_ratio,
            'lr': cfg['lr'],
            'decay_rate': cfg['decay_rate'],
            'lr_e_mult': cfg['lr_e_mult'],
        }

    # =================================================================
    # Summary table
    # =================================================================

    print(f"\n{'=' * 70}")
    print(f"SUMMARY — {n_seg:,} segs, {total_steps} steps, mode={mode}, "
          f"lr_e_mult=0.003")
    print(f"{'=' * 70}")
    print(f"{'Name':<20} {'LR':>6} {'Decay':>8} {'Final Loss':>12} "
          f"{'dE_ratio':>10} {'Dead':>6}")
    print("-" * 65)
    for name in [c['name'] for c in COMBOS]:
        r = all_results[name]
        print(f"{name:<20} {r.get('lr', 0.5):>6.2f} {r.get('decay_rate', 0.9995):>8.4f} "
              f"{r['final_loss']:>12.6f} {r['de_ratio']:>10.3f} "
              f"{r['dead_counts'][-1]:>6d}")

    # =================================================================
    # Save results
    # =================================================================

    save_path = os.path.join(OUT_DIR, 'sweep_results.npz')
    save_data = {}
    for name, r in all_results.items():
        save_data[f'{name}_losses'] = r['losses']
        save_data[f'{name}_total_energies'] = r['total_energies']
        save_data[f'{name}_dead_counts'] = r['dead_counts']
        save_data[f'{name}_final_loss'] = r['final_loss']
        save_data[f'{name}_de_ratio'] = r['de_ratio']
        save_data[f'{name}_lr_e_mult'] = r['lr_e_mult']

    save_data['combo_names'] = np.array([c['name'] for c in COMBOS])
    save_data['n_seg'] = n_seg
    save_data['total_steps'] = total_steps
    save_data['mode'] = mode
    save_data['truth_total_de'] = truth_total_de

    np.savez(save_path, **save_data)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
