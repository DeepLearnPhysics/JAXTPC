"""Push max_reloc higher: 750, 1000."""

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
DX_MM = 0.3
ALPHA_RECOMB = 0.93
W_ION = 23.6e-6

def main():
    n_seg = 50000
    total_steps = 1000

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

    # Per-track 50mm jitter init
    rng = np.random.RandomState(42)
    e_scale = n_truth / n_seg
    indices = rng.choice(n_truth, size=n_seg, replace=False)
    init_pos = truth_pos[indices].copy()
    init_de = truth_de[indices].copy() * e_scale
    init_tids = truth_tids[indices]
    for tid in np.unique(init_tids):
        init_pos[init_tids == tid] += rng.normal(0, 50.0, size=3)
    init_de *= rng.uniform(0.2, 1.8, size=n_seg)
    init_de = np.maximum(init_de, 0.001)
    init_params = jnp.array(np.column_stack([init_pos, init_de]))

    _ = grad_fn(init_params)
    print("Ready.\n")

    def compute_Q(de_arr):
        return (dx_cm / B_eff) * np.log(np.maximum(ALPHA_RECOMB + B_eff * de_arr / dx_cm, 1.0))
    truth_total_Q = compute_Q(truth_de).sum()

    combos = [
        {'reloc_every': 25, 'max_reloc': 500},   # reference from previous sweep
        {'reloc_every': 25, 'max_reloc': 750},
        {'reloc_every': 25, 'max_reloc': 1000},
    ]

    results = []
    for combo in combos:
        cfg = {'lr': 1.0, 'decay_rate': 0.999, 'lr_e_mult': 0.01,
               'warmup': 100, 'noise_lr': 0.3}
        cfg.update(combo)

        schedule = optax.exponential_decay(init_value=cfg['lr'], transition_steps=1,
                                            decay_rate=cfg['decay_rate'])
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

            lr_cur = float(schedule(step))
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                lr_cur * cfg['noise_lr'] * jax.random.normal(nk, shape=(n_seg, 3)))

            if step >= cfg['warmup'] and step % cfg['reloc_every'] == 0:
                params, opt_state, rng_key, n_reloc = relocate_segments(
                    params, opt_state, rng_key, recomb_constants, cfg['max_reloc'])
                cumulative_relocs += int(n_reloc)

            losses.append(float(total_loss))

        dt = time.time() - t0
        p_np = np.array(params)
        alive = p_np[:, 3] > 0.012
        sim_Q = compute_Q(p_np[alive, 3]).sum()
        q_ratio = sim_Q / truth_total_Q
        n_dead = int((~alive).sum())

        cs = f"every={combo['reloc_every']}, max={combo['max_reloc']}"
        print(f"  [{cs}] loss={losses[-1]:.6f}  Q={q_ratio:.3f}  "
              f"dead={n_dead}  relocs={cumulative_relocs}  {dt:.0f}s")
        results.append({'combo': combo, 'loss': losses[-1], 'q_ratio': q_ratio,
                        'dead': n_dead, 'relocs': cumulative_relocs})

    print(f"\n{'Config':<30} {'Loss':>10} {'Q':>6} {'Dead':>6} {'Relocs':>8}")
    print("-" * 65)
    for r in results:
        cs = f"every={r['combo']['reloc_every']}, max={r['combo']['max_reloc']}"
        print(f"{cs:<30} {r['loss']:>10.6f} {r['q_ratio']:>6.3f} {r['dead']:>6} {r['relocs']:>8}")

    with open(FINDINGS, 'a') as f:
        f.write(f"\n## relocation_big: Push max_reloc higher\n")
        f.write(f"out.h5 ev2, 50k, s=1.0, lr=1.0, d=0.999, e_mult=0.01, noise=0.3, {total_steps} steps\n\n")
        f.write(f"| Config | Loss | Q_ratio | Dead | Relocs |\n|---|---|---|---|---|\n")
        for r in results:
            cs = f"every={r['combo']['reloc_every']}, max={r['combo']['max_reloc']}"
            f.write(f"| {cs} | {r['loss']:.6f} | {r['q_ratio']:.3f} | {r['dead']} | {r['relocs']} |\n")
        f.write('\n')
    print(f"\nAppended to {FINDINGS}")

if __name__ == '__main__':
    main()
