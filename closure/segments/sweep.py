"""
Systematic hyperparameter study for Sobolev s=1.0.

Runs sweeps sequentially, appends results to a findings file.
Each sweep is a set of combos varying one parameter.

Usage:
    python3 closure/segments/sweep.py --data out.h5 --event 2 --sweep lr_emult
    python3 closure/segments/sweep.py --data out.h5 --event 2 --sweep lr_decay
    python3 closure/segments/sweep.py --data out.h5 --event 2 --sweep noise
    python3 closure/segments/sweep.py --data out.h5 --event 2 --sweep reloc
    python3 closure/segments/sweep.py --data out.h5 --event 2 --sweep warmup
    python3 closure/segments/sweep.py --data out.h5 --event 2 --sweep steps
    python3 closure/segments/sweep.py --data out.h5 --event 2 --sweep nseg
"""

import os, argparse, time, json

import jax, jax.numpy as jnp, numpy as np, optax
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from closure.segments.run import DEFAULTS, PLANE_NAMES, relocate_segments

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FINDINGS_FILE = os.path.join(OUT_DIR, 'FINDINGS_S1.md')
SOBOLEV_S = 1.0

# Default best config from previous sweeps (adapted for s=1.0)
BEST = {
    'lr': 1.0, 'decay_rate': 0.999, 'lr_e_mult': 0.003,
    'warmup': 100, 'noise_lr': 0.3, 'l1': 0.0,
    'reloc_every': 25, 'max_reloc': 100,
    'n_seg': 10000, 'steps': 1500,
}

SWEEPS = {
    'lr_emult': {
        'desc': 'lr_e_mult sweep (s=1.0)',
        'vary': 'lr_e_mult',
        'values': [0.001, 0.003, 0.005, 0.01],
    },
    'lr_decay': {
        'desc': 'LR x decay sweep (s=1.0)',
        'vary': ['lr', 'decay_rate'],
        'combos': [
            {'lr': 0.5, 'decay_rate': 0.9995},
            {'lr': 0.7, 'decay_rate': 0.999},
            {'lr': 1.0, 'decay_rate': 0.999},
            {'lr': 1.0, 'decay_rate': 0.9985},
            {'lr': 1.5, 'decay_rate': 0.998},
            {'lr': 2.0, 'decay_rate': 0.997},
        ],
    },
    'noise': {
        'desc': 'Noise sweep (s=1.0)',
        'vary': 'noise_lr',
        'values': [0.0, 0.1, 0.3, 0.5, 1.0],
    },
    'reloc': {
        'desc': 'Relocation config sweep (s=1.0)',
        'vary': ['reloc_every', 'max_reloc'],
        'combos': [
            {'reloc_every': 25, 'max_reloc': 50},
            {'reloc_every': 25, 'max_reloc': 100},
            {'reloc_every': 25, 'max_reloc': 200},
            {'reloc_every': 50, 'max_reloc': 100},
            {'reloc_every': 10, 'max_reloc': 50},
        ],
    },
    'warmup': {
        'desc': 'Warmup sweep (s=1.0)',
        'vary': 'warmup',
        'values': [50, 100, 200, 300],
    },
    'steps': {
        'desc': 'Step count sweep (s=1.0)',
        'vary': 'steps',
        'values': [500, 1000, 1500, 2000, 3000],
    },
    'nseg': {
        'desc': 'Segment count sweep (s=1.0)',
        'vary': 'n_seg',
        'values': [5000, 10000, 20000],
    },
}


def setup(h5_path, event_idx, n_seg, dx_mm, config_yaml, recomb_model='modified_box'):
    """Setup shared resources."""
    raw = load_particle_step_data(h5_path, event_idx)
    n_truth = raw['positions_mm'].shape[0]
    truth_total_de = float(np.sum(raw['de']))
    truth_pos = np.asarray(raw['positions_mm'])
    truth_de = np.asarray(raw['de'])
    print(f"Truth: {n_truth:,} segs, dE={truth_total_de:.1f} MeV, dx_mm={dx_mm}")

    detector_config = generate_detector(config_yaml)
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False, include_electronics=False,
        include_track_hits=False, recombination_model=recomb_model)

    # Build DepositData via build_deposit_data (multi-volume aware)
    deposits = build_deposit_data(
        raw['positions_mm'], raw['de'],
        np.full(n_truth, dx_mm, dtype=np.float32),
        sim_truth.config,
        theta=raw['theta'], phi=raw['phi'], track_ids=raw['track_ids'],
        t0_us=raw['t0_us'], interaction_ids=raw.get('interaction_ids'),
        root_track_ids=raw.get('root_track_ids'), pdg=raw.get('pdg'))

    response_signals, _, _ = sim_truth.process_event(deposits)

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

    # Recombination constants from sim params
    rp = sim_truth.default_sim_params.recomb_params
    dens = float(rp.density)
    alpha_r = float(rp.alpha)
    if recomb_model == 'emb':
        beta_r = float(rp.beta_90)
    else:
        beta_r = float(rp.beta)
    field_kVcm = float(rp.field_strength_Vcm) / 1000.0
    B_eff = beta_r / dens / field_kVcm
    dx_cm = dx_mm / 10.0
    recomb_constants = (DEFAULTS['death_thresh'], alpha_r, B_eff, dx_cm)

    return {
        'detector_config': detector_config,
        'truth_signals': truth_signals,
        'spectral_weights': spectral_weights,
        'planes_tuple': planes_tuple,
        'active_planes': active_planes,
        'recomb_constants': recomb_constants,
        'truth_pos': truth_pos,
        'truth_de': truth_de,
        'truth_total_de': truth_total_de,
        'n_truth': n_truth,
        'dx_mm': dx_mm,
        'recomb_model': recomb_model,
    }


def run_one(shared, cfg):
    """Run one configuration. Returns result dict."""
    n_seg = cfg['n_seg']
    total_steps = cfg['steps']

    sim_opt = DetectorSimulator(
        shared['detector_config'], differentiable=True, n_segments=n_seg,
        recombination_model=shared['recomb_model'])
    sim_params = sim_opt.default_sim_params
    dx_val = shared['dx_mm']

    def fwd_opt(positions_mm, de):
        return sim_opt.forward_segments(sim_params, positions_mm, de, dx=dx_val)

    def loss_fn(params):
        positions_mm = params[:, :3]
        de = params[:, 3]
        sigs = fwd_opt(positions_mm, de)
        loss = sobolev_loss_geomean_log1p(sigs, shared['truth_signals'],
                                           shared['spectral_weights'],
                                           planes=shared['planes_tuple'])
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Init
    rng = np.random.RandomState(42)
    n_truth = shared['n_truth']
    e_scale = n_truth / n_seg
    replace = n_seg > n_truth
    indices = rng.choice(n_truth, size=n_seg, replace=replace)
    init_pos = shared['truth_pos'][indices].copy() + rng.normal(0, 100.0, size=(n_seg, 3))
    init_de = shared['truth_de'][indices].copy() * e_scale * rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    # JIT warmup
    _ = grad_fn(init_params)

    schedule = optax.exponential_decay(
        init_value=cfg['lr'], transition_steps=1, decay_rate=cfg['decay_rate'])
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)
    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)
    cumulative_relocs = 0

    losses = []
    t0 = time.time()

    for step in range(total_steps):
        (total_loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(cfg['lr_e_mult'])
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], 0.001))

        if cfg['noise_lr'] > 0:
            lr_cur = float(schedule(step))
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                lr_cur * cfg['noise_lr'] * jax.random.normal(nk, shape=(n_seg, 3)))

        if cfg['l1'] > 0 and step >= cfg['warmup']:
            params = params.at[:, 3].add(-cfg['l1'])
            params = params.at[:, 3].set(jnp.maximum(params[:, 3], 0.001))

        if step >= cfg['warmup'] and step % cfg['reloc_every'] == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key,
                shared['recomb_constants'], cfg['max_reloc'])
            cumulative_relocs += int(n_reloc)

        losses.append(float(total_loss))

    dt = time.time() - t0
    final_params = np.array(params)
    final_de = float(np.sum(final_params[:, 3]))
    de_ratio = final_de / shared['truth_total_de']
    n_dead = int(np.sum(final_params[:, 3] <= DEFAULTS['death_thresh']))

    return {
        'loss': losses[-1],
        'de_ratio': de_ratio,
        'dead': n_dead,
        'relocs': cumulative_relocs,
        'time': dt,
        'ms_per_step': dt / total_steps * 1000,
    }


def append_findings(sweep_name, desc, h5_file, event_idx, dx_mm, results_table):
    """Append results to findings file."""
    with open(FINDINGS_FILE, 'a') as f:
        f.write(f"\n## {sweep_name}: {desc}\n")
        f.write(f"File: {h5_file}, event={event_idx}, dx={dx_mm}mm, s={SOBOLEV_S}\n\n")
        f.write(results_table)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description='Systematic hyperparameter study for Sobolev s=1.0')
    parser.add_argument('--data', required=True,
                        help='Path to HDF5 event file')
    parser.add_argument('--config-yaml', default='config/cubic_wireplane_config.yaml',
                        help='Path to detector YAML config (default: config/cubic_wireplane_config.yaml)')
    parser.add_argument('--event', type=int, default=2)
    parser.add_argument('--sweep', required=True, choices=list(SWEEPS.keys()))
    parser.add_argument('--dx', type=float, default=0.3,
                        help='dx_mm for both truth and optimizer (default: 0.3 for out.h5)')
    args = parser.parse_args()

    sweep_def = SWEEPS[args.sweep]
    print(f"{'='*60}")
    print(f"SWEEP: {args.sweep} -- {sweep_def['desc']}")
    print(f"File: {args.data}, event={args.event}, dx={args.dx}mm")
    print(f"{'='*60}\n")

    # Build combo list
    if 'combos' in sweep_def:
        combos = sweep_def['combos']
    else:
        vary_key = sweep_def['vary']
        combos = [{vary_key: v} for v in sweep_def['values']]

    # For nseg sweep, we need to rebuild sim for each n_seg
    # For steps sweep, same sim different step counts
    # Handle by rebuilding per-combo if n_seg changes

    all_results = []
    shared = None
    last_n_seg = None

    for i, combo in enumerate(combos):
        cfg = dict(BEST)
        cfg.update(combo)

        # Rebuild shared if n_seg changed or first run
        if shared is None or cfg['n_seg'] != last_n_seg:
            print(f"Setting up for n_seg={cfg['n_seg']}...")
            shared = setup(args.data, args.event, cfg['n_seg'], args.dx,
                           args.config_yaml)
            last_n_seg = cfg['n_seg']

        combo_name = ', '.join(f'{k}={v}' for k, v in combo.items())
        print(f"\n[{i+1}/{len(combos)}] {combo_name}...")

        t0 = time.time()
        try:
            result = run_one(shared, cfg)
            print(f"  loss={result['loss']:.6f}  dE_ratio={result['de_ratio']:.3f}  "
                  f"dead={result['dead']}  {result['time']:.0f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            result = {'loss': float('nan'), 'de_ratio': 0, 'dead': -1, 'relocs': 0,
                      'time': time.time()-t0, 'ms_per_step': 0}

        result['combo'] = combo
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY -- {args.sweep}")
    print(f"{'='*60}")

    header_keys = list(combos[0].keys())
    header = '| ' + ' | '.join(header_keys) + ' | Loss | dE_ratio | Dead | Time |\n'
    header += '|' + '|'.join(['---'] * (len(header_keys) + 4)) + '|\n'
    rows = ''
    for r in all_results:
        vals = ' | '.join(str(r['combo'].get(k, '')) for k in header_keys)
        rows += f"| {vals} | {r['loss']:.6f} | {r['de_ratio']:.3f} | {r['dead']} | {r['time']:.0f}s |\n"

    table = header + rows
    print(table)

    # Save
    append_findings(args.sweep, sweep_def['desc'], args.data, args.event, args.dx, table)
    print(f"Appended to {FINDINGS_FILE}")

    # Also save NPZ
    save_path = os.path.join(OUT_DIR, f'sweep_s1_{args.sweep}_{os.path.basename(args.data)}_ev{args.event}.npz')
    save_data = {}
    for i, r in enumerate(all_results):
        save_data[f'combo_{i}_loss'] = r['loss']
        save_data[f'combo_{i}_de_ratio'] = r['de_ratio']
        save_data[f'combo_{i}_dead'] = r['dead']
        save_data[f'combo_{i}_config'] = json.dumps(r['combo'])
    np.savez(save_path, **save_data)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
