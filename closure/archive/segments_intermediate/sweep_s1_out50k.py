"""
Sequential hyperparameter sweeps for s=1.0 on out.h5 event 2, 50k segments.
Runs: lr_e_mult → lr_decay → noise → relocation.

Usage:
    python3 closure_analysis_full/sweeps/sweep_s1_out50k.py
"""

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
FINDINGS = os.path.join(OUT_DIR, 'FINDINGS_S1.md')
SOBOLEV_S = 1.0
DX_MM = 0.3
ALPHA_RECOMB = 0.93
W_ION = 23.6e-6


def setup(n_seg=50000):
    deposit_data = load_particle_step_data('out.h5', 2)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)
    truth_tids = np.asarray(deposit_data.track_ids)
    truth_total_de = float(np.sum(truth_de))

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

    # Per-track 50mm jitter init
    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy()
    init_de = truth_de[indices].copy() * e_scale
    init_tids = truth_tids[indices]
    for tid in np.unique(init_tids):
        disp = rng.normal(0, 50.0, size=3)
        init_pos[init_tids == tid] += disp
    init_de *= rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    # JIT warmup
    _ = grad_fn(init_params)

    def compute_Q(de_arr):
        return (dx_cm / B_eff) * np.log(np.maximum(ALPHA_RECOMB + B_eff * de_arr / dx_cm, 1.0))
    truth_total_Q = compute_Q(truth_de).sum()

    return {
        'grad_fn': grad_fn, 'init_params': init_params, 'n_seg': n_seg,
        'truth_total_de': truth_total_de, 'truth_total_Q': truth_total_Q,
        'recomb_constants': recomb_constants, 'compute_Q': compute_Q,
        'B_eff': B_eff, 'dx_cm': dx_cm,
    }


def run_one(shared, cfg, total_steps=1000):
    n_seg = shared['n_seg']
    schedule = optax.exponential_decay(
        init_value=cfg['lr'], transition_steps=1, decay_rate=cfg['decay_rate'])
    optimizer = optax.adam(schedule, b1=0.9, b2=0.999)
    params = shared['init_params']
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)
    cumulative_relocs = 0
    losses = []
    t0 = time.time()

    for step in range(total_steps):
        (total_loss, _), grads = shared['grad_fn'](params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(cfg['lr_e_mult'])
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], 0.001))

        if cfg.get('noise_lr', 0) > 0:
            lr_cur = float(schedule(step))
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                lr_cur * cfg['noise_lr'] * jax.random.normal(nk, shape=(n_seg, 3)))

        if step >= cfg.get('warmup', 100) and step % cfg.get('reloc_every', 25) == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key,
                shared['recomb_constants'], cfg.get('max_reloc', 100))
            cumulative_relocs += int(n_reloc)

        losses.append(float(total_loss))

    dt = time.time() - t0
    p_np = np.array(params)
    alive = p_np[:, 3] > 0.012
    sim_Q = shared['compute_Q'](p_np[alive, 3]).sum()
    q_ratio = sim_Q / shared['truth_total_Q']
    de_ratio = p_np[:, 3].sum() / shared['truth_total_de']
    n_dead = int((~alive).sum())

    return {
        'loss': losses[-1], 'q_ratio': q_ratio, 'de_ratio': de_ratio,
        'dead': n_dead, 'relocs': cumulative_relocs, 'time': dt,
    }


def run_sweep(name, desc, combos, shared, total_steps=1000, base_cfg=None):
    if base_cfg is None:
        base_cfg = {'lr': 1.0, 'decay_rate': 0.999, 'lr_e_mult': 0.003,
                     'warmup': 100, 'noise_lr': 0.3, 'reloc_every': 25, 'max_reloc': 100}

    print(f"\n{'='*60}")
    print(f"SWEEP: {name} — {desc}")
    print(f"{'='*60}")

    results = []
    for combo in combos:
        cfg = dict(base_cfg)
        cfg.update(combo)
        combo_str = ', '.join(f'{k}={v}' for k, v in combo.items())
        print(f"\n  [{combo_str}]...", end=' ', flush=True)
        r = run_one(shared, cfg, total_steps)
        print(f"loss={r['loss']:.6f}  Q={r['q_ratio']:.3f}  dead={r['dead']}  {r['time']:.0f}s")
        r['combo'] = combo
        results.append(r)

    # Summary
    print(f"\n  {'Config':<30} {'Loss':>10} {'Q_ratio':>8} {'Dead':>6}")
    print(f"  {'-'*58}")
    for r in results:
        cs = ', '.join(f'{k}={v}' for k, v in r['combo'].items())
        print(f"  {cs:<30} {r['loss']:>10.6f} {r['q_ratio']:>8.3f} {r['dead']:>6}")

    # Append to findings
    with open(FINDINGS, 'a') as f:
        f.write(f"\n## {name}: {desc}\n")
        f.write(f"out.h5 ev2, 50k segs, s=1.0, dx=0.3mm, {total_steps} steps, track jitter 50mm\n\n")
        f.write(f"| Config | Loss | Q_ratio | Dead |\n|---|---|---|---|\n")
        for r in results:
            cs = ', '.join(f'{k}={v}' for k, v in r['combo'].items())
            f.write(f"| {cs} | {r['loss']:.6f} | {r['q_ratio']:.3f} | {r['dead']} |\n")
        f.write('\n')
    print(f"\n  Appended to {FINDINGS}")

    return results


def main():
    print("Setting up (50k segs, out.h5 event 2, s=1.0, dx=0.3mm)...")
    shared = setup(50000)
    print("Ready.\n")

    # Sweep 1: lr_e_mult
    results_emult = run_sweep(
        'lr_e_mult (s=1.0, out.h5)',
        'lr_e_mult sweep at 50k segments',
        [{'lr_e_mult': 0.001}, {'lr_e_mult': 0.003},
         {'lr_e_mult': 0.005}, {'lr_e_mult': 0.01}],
        shared, total_steps=1000)

    best_emult = min(results_emult, key=lambda r: r['loss'])['combo']['lr_e_mult']
    print(f"\n>>> Best lr_e_mult: {best_emult}")

    # Sweep 2: lr × decay
    base2 = {'lr': 1.0, 'decay_rate': 0.999, 'lr_e_mult': best_emult,
             'warmup': 100, 'noise_lr': 0.3, 'reloc_every': 25, 'max_reloc': 100}
    results_lr = run_sweep(
        'lr_decay (s=1.0, out.h5)',
        f'lr × decay sweep (lr_e_mult={best_emult})',
        [{'lr': 0.5, 'decay_rate': 0.9995},
         {'lr': 0.7, 'decay_rate': 0.999},
         {'lr': 1.0, 'decay_rate': 0.999},
         {'lr': 1.5, 'decay_rate': 0.998},
         {'lr': 2.0, 'decay_rate': 0.997}],
        shared, total_steps=1000, base_cfg=base2)

    best_lr_combo = min(results_lr, key=lambda r: r['loss'])['combo']
    print(f"\n>>> Best lr/decay: {best_lr_combo}")

    # Sweep 3: noise
    base3 = dict(base2)
    base3.update(best_lr_combo)
    results_noise = run_sweep(
        'noise (s=1.0, out.h5)',
        f'noise sweep (lr_e_mult={best_emult}, {best_lr_combo})',
        [{'noise_lr': 0.0}, {'noise_lr': 0.1}, {'noise_lr': 0.3},
         {'noise_lr': 0.5}, {'noise_lr': 1.0}],
        shared, total_steps=1000, base_cfg=base3)

    best_noise = min(results_noise, key=lambda r: r['loss'])['combo']['noise_lr']
    print(f"\n>>> Best noise_lr: {best_noise}")

    # Sweep 4: relocation
    base4 = dict(base3)
    base4['noise_lr'] = best_noise
    results_reloc = run_sweep(
        'relocation (s=1.0, out.h5)',
        f'relocation sweep (all best so far)',
        [{'reloc_every': 25, 'max_reloc': 50},
         {'reloc_every': 25, 'max_reloc': 100},
         {'reloc_every': 25, 'max_reloc': 200},
         {'reloc_every': 25, 'max_reloc': 500},
         {'reloc_every': 10, 'max_reloc': 100},
         {'reloc_every': 10, 'max_reloc': 200}],
        shared, total_steps=1000, base_cfg=base4)

    best_reloc = min(results_reloc, key=lambda r: r['loss'])['combo']
    print(f"\n>>> Best relocation: {best_reloc}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL BEST CONFIG (s=1.0, out.h5, 50k segs):")
    print(f"  lr_e_mult={best_emult}")
    print(f"  {best_lr_combo}")
    print(f"  noise_lr={best_noise}")
    print(f"  {best_reloc}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
