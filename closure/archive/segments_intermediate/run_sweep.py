"""
Standalone sweep runner. Does NOT modify source code.
Writes results to sweeps/ folder as NPZ + text summary.

Usage:
    python3 closure_analysis_full/sweeps/run_sweep.py mpvmpr_20.h5 --sweep lr_decay
    python3 closure_analysis_full/sweeps/run_sweep.py mpvmpr_20.h5 --sweep steps
    python3 closure_analysis_full/sweeps/run_sweep.py mpvmpr_20.h5 --sweep nseg
"""

import sys, os, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from tools.recombination import extract_recombination_params
from closure_analysis_full.full_closure import (
    DEFAULTS, PLANE_NAMES, build_loss_fn, run_training_loop,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# Sweep definitions
# =========================================================================

SWEEPS = {
    'lr_decay': {
        'description': 'LR x decay_rate sweep (lr_e_mult=0.003 fixed)',
        'combos': [
            {'name': 'lr03_d9995', 'lr': 0.3, 'decay_rate': 0.9995},
            {'name': 'lr05_d9995', 'lr': 0.5, 'decay_rate': 0.9995},
            {'name': 'lr07_d999',  'lr': 0.7, 'decay_rate': 0.999},
            {'name': 'lr10_d999',  'lr': 1.0, 'decay_rate': 0.999},
            {'name': 'lr07_d9995', 'lr': 0.7, 'decay_rate': 0.9995},
            {'name': 'lr10_d9995', 'lr': 1.0, 'decay_rate': 0.9995},
        ],
        'fixed': {'lr_e_mult': 0.003, 'n_seg': 10000, 'steps': 1500},
    },
    'lr_decay_fine': {
        'description': 'Fine lr x decay around best region',
        'combos': [
            {'name': 'lr05_d999',  'lr': 0.5, 'decay_rate': 0.999},
            {'name': 'lr05_d9998', 'lr': 0.5, 'decay_rate': 0.9998},
            {'name': 'lr07_d9993', 'lr': 0.7, 'decay_rate': 0.9993},
            {'name': 'lr07_d9997', 'lr': 0.7, 'decay_rate': 0.9997},
        ],
        'fixed': {'lr_e_mult': 0.003, 'n_seg': 10000, 'steps': 1500},
    },
    'steps': {
        'description': 'Step count sweep at best lr/decay',
        'combos': [
            {'name': 'steps_1000', 'steps_override': 1000},
            {'name': 'steps_1500', 'steps_override': 1500},
            {'name': 'steps_2000', 'steps_override': 2000},
            {'name': 'steps_3000', 'steps_override': 3000},
        ],
        'fixed': {'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.003, 'n_seg': 10000},
    },
    'warmup': {
        'description': 'Warmup length sweep',
        'combos': [
            {'name': 'warmup_100', 'warmup': 100},
            {'name': 'warmup_200', 'warmup': 200},
            {'name': 'warmup_300', 'warmup': 300},
            {'name': 'warmup_500', 'warmup': 500},
        ],
        'fixed': {'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.003, 'n_seg': 10000, 'steps': 1500},
    },
}


def setup_shared(h5_path, event_idx, n_seg, recomb_model):
    """Shared setup: load data, truth sim, optimizer sim, init params."""
    print("Loading event data...")
    deposit_data = load_particle_step_data(h5_path, event_idx)
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
        recombination_model=recomb_model)
    t0 = time.time()
    response_signals, _ = sim_truth(deposit_data)
    print(f"  Truth sim done in {time.time()-t0:.1f}s")

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
    print(f"  Active planes: {[PLANE_NAMES[p] for p in active_planes]}")

    # Recomb constants for relocation
    _, dens, _, alpha_r, beta_r = extract_recombination_params(detector_config)
    field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
    B_eff = beta_r / dens / field_kVcm
    dx_cm = 0.05  # 0.5mm
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, dx_cm)

    # Build optimizer sim
    print(f"Building differentiable simulator (n_seg={n_seg:,})...")
    sim_opt = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=recomb_model)
    fwd_opt = sim_opt.build_forward()

    loss_fn = build_loss_fn(fwd_opt, truth_signals, spectral_weights, active_planes)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Init params
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
    print(f"  e_scale={e_scale:.2f}, init total dE={float(jnp.sum(init_params[:, 3])):.2f} MeV")

    # JIT warmup
    print("Warming up JIT...")
    t0 = time.time()
    _ = grad_fn(init_params)
    print(f"JIT warm-up done in {time.time()-t0:.1f}s\n")

    return {
        'grad_fn': grad_fn,
        'init_params': init_params,
        'n_seg': n_seg,
        'truth_total_de': truth_total_de,
        'recomb_constants': recomb_constants,
        'n_truth': n_truth,
    }


def run_combo(shared, combo, fixed, mode='full'):
    """Run a single combo and return results."""
    cfg = dict(DEFAULTS)
    cfg['noise_lr'] = 0.0
    cfg['l1'] = 0.0
    cfg['reloc_every'] = 25
    cfg['max_reloc'] = 100
    cfg['recomb_constants'] = shared['recomb_constants']

    # Apply fixed settings, then combo overrides
    cfg.update(fixed)
    cfg.update(combo)

    n_seg = cfg.pop('n_seg', shared['n_seg'])
    total_steps = cfg.pop('steps_override', cfg.pop('steps', 1500))

    name = combo['name']
    print(f"[{name}] lr={cfg['lr']}, decay={cfg['decay_rate']}, "
          f"e_mult={cfg['lr_e_mult']}, warmup={cfg['warmup']}, "
          f"steps={total_steps}...")

    t0 = time.time()
    result = run_training_loop(
        shared['grad_fn'], shared['init_params'], shared['n_seg'],
        total_steps, mode, cfg)
    dt = time.time() - t0

    final_de = float(np.sum(result['final_params'][:, 3]))
    de_ratio = final_de / shared['truth_total_de']

    print(f"  loss={result['losses'][-1]:.6f}  dE_ratio={de_ratio:.3f}  "
          f"dead={result['dead_counts'][-1]}  {dt:.0f}s\n")

    return {
        'name': name,
        'losses': result['losses'],
        'total_energies': result['total_energies'],
        'dead_counts': result['dead_counts'],
        'final_loss': float(result['losses'][-1]),
        'de_ratio': de_ratio,
        'total_steps': total_steps,
        **{k: v for k, v in combo.items() if k != 'name'},
        **{k: v for k, v in fixed.items()},
        'time_s': dt,
    }


def main():
    parser = argparse.ArgumentParser(description='Systematic sweep runner')
    parser.add_argument('h5_path', help='Path to HDF5 event file')
    parser.add_argument('--sweep', required=True, choices=list(SWEEPS.keys()))
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--recomb', type=str, default='modified_box')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['baseline', 'full'])
    args = parser.parse_args()

    sweep_def = SWEEPS[args.sweep]
    combos = sweep_def['combos']
    fixed = sweep_def['fixed']
    n_seg = fixed.get('n_seg', 10000)

    print(f"{'='*70}")
    print(f"SWEEP: {args.sweep} — {sweep_def['description']}")
    print(f"  {len(combos)} combos, mode={args.mode}, recomb={args.recomb}")
    print(f"  Fixed: {fixed}")
    print(f"{'='*70}\n")

    shared = setup_shared(args.h5_path, args.event, n_seg, args.recomb)

    all_results = []
    for combo in combos:
        r = run_combo(shared, combo, fixed, mode=args.mode)
        all_results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — {args.sweep}")
    print(f"{'='*70}")

    # Build header from combo keys
    combo_keys = [k for k in combos[0].keys() if k != 'name']
    header = f"{'Name':<20}"
    for k in combo_keys:
        header += f" {k:>12}"
    header += f" {'Loss':>10} {'dE_ratio':>10} {'Dead':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = f"{r['name']:<20}"
        for k in combo_keys:
            v = r.get(k, '')
            if isinstance(v, float):
                line += f" {v:>12.4f}"
            else:
                line += f" {str(v):>12}"
        line += f" {r['final_loss']:>10.6f} {r['de_ratio']:>10.3f}"
        line += f" {r['dead_counts'][-1]:>6d} {r['time_s']:>5.0f}s"
        print(line)

    # Save
    save_path = os.path.join(OUT_DIR, f'sweep_{args.sweep}.npz')
    save_data = {'sweep_name': args.sweep, 'mode': args.mode}
    for r in all_results:
        prefix = r['name']
        save_data[f'{prefix}_losses'] = r['losses']
        save_data[f'{prefix}_total_energies'] = r['total_energies']
        save_data[f'{prefix}_dead_counts'] = r['dead_counts']
        save_data[f'{prefix}_final_loss'] = r['final_loss']
        save_data[f'{prefix}_de_ratio'] = r['de_ratio']
    save_data['combo_names'] = np.array([c['name'] for c in combos])
    np.savez(save_path, **save_data)
    print(f"\nSaved to {save_path}")

    # Also save text summary
    txt_path = os.path.join(OUT_DIR, f'sweep_{args.sweep}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Sweep: {args.sweep}\n")
        f.write(f"Description: {sweep_def['description']}\n")
        f.write(f"Mode: {args.mode}, Recomb: {args.recomb}\n")
        f.write(f"Fixed: {fixed}\n\n")
        for r in all_results:
            f.write(f"{r['name']}: loss={r['final_loss']:.6f}, "
                    f"dE_ratio={r['de_ratio']:.3f}, "
                    f"dead={r['dead_counts'][-1]}, "
                    f"steps={r['total_steps']}, time={r['time_s']:.0f}s\n")
    print(f"Saved to {txt_path}")


if __name__ == '__main__':
    main()
