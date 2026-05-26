"""
Run 3 overcomplete combos (10/5) with new base config.
Each gets its own diagnostic plot saved with unique name.

Base: LR=0.7, exp d=0.9995, lr_e_mult=0.02, b1=0.95
MCMC: warmup=400, DT=0.005, linear noise NOISE_LR=0.05

Combos:
  A: L1=1e-4, split=80/20
  B: L1=2e-4, split=70/30
  C: L1=1e-4, split=50/50

Run:  python3 closure_analysis/run_overcomplete_combos.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit
from closure_analysis.sgld_closure import (
    TRUTH_BANK, INIT_OFFSET, EXTRA_INIT_BOUNDS, PLANES, K, N_PROJ,
    build_loss_fn, best_permutation, relocate_segments, reset_adam_moments,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_SEG = 10
N_TRUTH = 5
MIN_ENERGY = 0.001
STEPS = 2000

# Base config
LR = 0.5
DECAY_RATE = 0.999
LR_E_MULT = 0.02
B1 = 0.95
B2 = 0.999

# MCMC config
WARMUP = 400
DEATH_THRESH = 0.005
NOISE_LR = 0.3  # linear coupling
NOISE_COUPLING = 'linear'

# Plot sizes
LABEL_SIZE = 16
TITLE_SIZE = 17
TICK_SIZE = 13
LEGEND_SIZE = 12
SUPTITLE_SIZE = 16


COMBOS = [
    {'name': 'G_L5e4_split80', 'l1': 5e-4, 'split': 0.8},
    {'name': 'H_L1e3_split70', 'l1': 1e-3, 'split': 0.7},
    {'name': 'I_L5e4_split50', 'l1': 5e-4, 'split': 0.5},
]


def run_combo(combo, grad_fn, init_params, truth_params):
    name = combo['name']
    l1_rate = combo['l1']
    split_ratio = combo['split']

    schedule = optax.exponential_decay(init_value=LR, transition_steps=1, decay_rate=DECAY_RATE)
    optimizer = optax.adam(schedule, b1=B1, b2=B2)

    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    losses = []
    param_history = []
    dead_counts = []
    relocation_steps = []
    cumulative_relocs = 0

    for step in range(STEPS):
        (loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(LR_E_MULT)
        params = optax.apply_updates(params, updates)

        # Energy floor
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], MIN_ENERGY))

        # L1 drain after warmup
        if step >= WARMUP:
            params = params.at[:, 3].add(-l1_rate)
            params = params.at[:, 3].set(jnp.maximum(params[:, 3], MIN_ENERGY))

        # Noise (linear coupling)
        lr_cur = float(schedule(step))
        ns = lr_cur * NOISE_LR
        rng_key, nk = jax.random.split(rng_key)
        params = params.at[:, :3].add(ns * jax.random.normal(nk, shape=(N_SEG, 3)))

        # Relocation after warmup
        n_dead = int(jnp.sum(params[:, 3] <= DEATH_THRESH))
        if step >= WARMUP and n_dead > 0:
            # Inline relocation with custom split ratio
            energies = np.array(params[:, 3])
            dead_mask = energies <= DEATH_THRESH
            dead_indices = np.where(dead_mask)[0]
            alive_mask = ~dead_mask
            alive_indices = np.where(alive_mask)[0]

            if len(dead_indices) > 0 and len(alive_indices) > 0:
                alive_energies = np.array(params[alive_indices, 3], dtype=np.float64)
                n_reloc = 0
                for d_idx in dead_indices:
                    if alive_energies.sum() <= 0:
                        break
                    probs = alive_energies / alive_energies.sum()
                    rng_key, sk = jax.random.split(rng_key)
                    donor_local = int(jax.random.choice(sk, len(alive_indices), p=jnp.array(probs)))
                    donor_idx = alive_indices[donor_local]
                    donor_energy = float(params[donor_idx, 3])

                    rng_key, ok = jax.random.split(rng_key)
                    offset = jax.random.normal(ok, shape=(3,)) * 3.0
                    new_pos = params[donor_idx, :3] + offset

                    clone_energy = donor_energy * (1.0 - split_ratio)
                    donor_new_energy = donor_energy * split_ratio

                    params = params.at[d_idx, :3].set(new_pos)
                    params = params.at[d_idx, 3].set(clone_energy)
                    params = params.at[donor_idx, 3].set(donor_new_energy)
                    alive_energies[donor_local] = donor_new_energy

                    opt_state = reset_adam_moments(opt_state, d_idx)
                    n_reloc += 1

                if n_reloc > 0:
                    cumulative_relocs += n_reloc
                    relocation_steps.append((step, n_reloc))

        losses.append(float(loss))
        param_history.append(np.array(params))
        dead_counts.append(n_dead)

    # Evaluate
    final = np.array(params)
    param_history = np.array(param_history)
    assignment, errors, matched_idx = best_permutation(final, truth_params)
    matched_errors = errors[matched_idx]
    extra_idx = [i for i in range(N_SEG) if i not in matched_idx]

    mean_pos = np.mean(np.sqrt(np.sum(matched_errors[:, :3] ** 2, axis=1)))
    max_pos = np.max(np.sqrt(np.sum(matched_errors[:, :3] ** 2, axis=1)))
    max_de = np.max(np.abs(matched_errors[:, 3])) * 1000
    extra_energies = [float(final[i, 3]) for i in extra_idx]
    n_extra_dead = sum(1 for e in extra_energies if e <= DEATH_THRESH)

    return {
        'name': name, 'l1': l1_rate, 'split': split_ratio,
        'mean_pos': mean_pos, 'max_pos': max_pos, 'max_de': max_de,
        'loss': losses[-1], 'losses': losses,
        'param_history': param_history,
        'assignment': assignment, 'matched_idx': matched_idx,
        'extra_idx': extra_idx, 'extra_energies': extra_energies,
        'n_extra_dead': n_extra_dead, 'dead_counts': dead_counts,
        'relocation_steps': relocation_steps, 'cumulative_relocs': cumulative_relocs,
    }


def plot_combo(r, truth_params):
    n_seg = N_SEG
    seg_colors = plt.cm.tab10(np.linspace(0, 1, max(n_seg, 3)))
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    losses = r['losses']
    param_history = r['param_history']
    assignment = r['assignment']
    matched_idx = r['matched_idx']
    extra_idx = r['extra_idx']
    relocation_steps = r['relocation_steps']

    # --- Top-left: loss curve ---
    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.2, alpha=0.7, label='SW loss')
    ax.axvline(WARMUP, color='gray', ls='--', lw=1.0, alpha=0.5, label=f'Warmup={WARMUP}')
    for rs, nr in relocation_steps:
        ax.axvline(rs, color='red', ls=':', lw=0.5, alpha=0.3)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Loss', fontsize=LABEL_SIZE)
    ax.set_title('Loss Convergence (red = relocation)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Top-right: position error per matched segment ---
    ax = axes[0, 1]
    for s in matched_idx:
        pos_err = np.sqrt(np.sum(
            (param_history[:, s, :3] - assignment[s, :3]) ** 2, axis=1))
        ax.plot(pos_err, color=seg_colors[s], lw=1.2, label=f'Seg {s}')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('||pos error|| (mm)', fontsize=LABEL_SIZE)
    ax.set_title(f'Position Error (matched {N_TRUTH} segments)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE - 1, ncol=2)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: total energy + per-segment energy ---
    ax = axes[1, 0]
    total_e = param_history[:, :, 3].sum(axis=1) * 1000
    truth_total_e = sum(assignment[s, 3] for s in matched_idx) * 1000
    ax.plot(total_e, 'k-', lw=2.5, alpha=0.8, label='Total energy')
    ax.axhline(truth_total_e, color='k', ls=':', lw=1.5, alpha=0.5,
               label=f'Truth total={truth_total_e:.0f} keV')
    for s in range(n_seg):
        is_extra = s in extra_idx
        ls = '--' if is_extra else '-'
        ax.plot(param_history[:, s, 3] * 1000, color=seg_colors[s],
                lw=0.8, ls=ls, alpha=0.4)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Energy (keV)', fontsize=LABEL_SIZE)
    ax.set_title('Energy (total=black, segments=thin)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: per-segment energy trajectories ---
    ax = axes[1, 1]
    for s in range(n_seg):
        is_extra = s in extra_idx
        ls = '--' if is_extra else '-'
        alpha_val = 0.5 if is_extra else 1.0
        lbl = f'Seg {s}' + (' (extra)' if is_extra else '')
        ax.plot(param_history[:, s, 3] * 1000, color=seg_colors[s],
                lw=1.2, ls=ls, alpha=alpha_val, label=lbl)
        if not is_extra:
            ax.axhline(assignment[s, 3] * 1000, color=seg_colors[s],
                       ls=':', lw=0.8, alpha=0.4)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('dE (keV)', fontsize=LABEL_SIZE)
    ax.set_title('Per-Segment Energy (solid=matched, dashed=extra, dotted=truth)',
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE - 2, ncol=2)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    split_pct = f"{int(r['split']*100)}/{int((1-r['split'])*100)}"
    fig.suptitle(
        f"10/5 Overcomplete — {r['name']}  |  L1={r['l1']}, split={split_pct}, "
        f"noise={NOISE_LR}lin, WU={WARMUP}\n"
        f"pos={r['mean_pos']:.2f}mm, max={r['max_pos']:.2f}mm, dE={r['max_de']:.1f}keV, "
        f"relocs={r['cumulative_relocs']}, extras_dead={r['n_extra_dead']}/5",
        fontsize=SUPTITLE_SIZE, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'overcomplete_{r["name"]}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    print(f"Overcomplete 10/5 sweep — 3 combos, {STEPS} steps each")
    print(f"Base: LR={LR}, d={DECAY_RATE}, e_mult={LR_E_MULT}, b1={B1}")
    print(f"MCMC: WU={WARMUP}, DT={DEATH_THRESH}, noise={NOISE_LR} linear\n")

    # Build simulators
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim_truth = DetectorSimulator(detector_config, differentiable=True, n_segments=N_TRUTH)
    fwd_truth = sim_truth.build_forward()
    sim_opt = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEG)
    fwd_opt = sim_opt.build_forward()

    truth_params = TRUTH_BANK[:N_TRUTH]
    truth_seg = SegmentData(positions_mm=jnp.array(truth_params[:, :3]),
                            de=jnp.array(truth_params[:, 3]))
    target_signals = fwd_truth(truth_seg)
    key = jax.random.PRNGKey(42)
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    loss_fn = build_loss_fn(fwd_opt, target_clouds, key)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Init params
    rng_init = np.random.RandomState(42)
    init_p = np.zeros((N_SEG, 4))
    init_p[:N_TRUTH] = truth_params + INIT_OFFSET
    median_e = float(np.median(truth_params[:, 3]))
    for i in range(N_TRUTH, N_SEG):
        init_p[i, 0] = rng_init.uniform(*EXTRA_INIT_BOUNDS['x'])
        init_p[i, 1] = rng_init.uniform(*EXTRA_INIT_BOUNDS['y'])
        init_p[i, 2] = rng_init.uniform(*EXTRA_INIT_BOUNDS['z'])
        init_p[i, 3] = median_e
    init_params = jnp.array(init_p)

    print("Warming up JIT...")
    _ = grad_fn(init_params)
    print("Ready.\n")

    # Run combos
    results = []
    for combo in COMBOS:
        split_pct = f"{int(combo['split']*100)}/{int((1-combo['split'])*100)}"
        print(f"[{combo['name']}] L1={combo['l1']}, split={split_pct}...")
        t0 = time.time()
        r = run_combo(combo, grad_fn, init_params, truth_params)
        dt = time.time() - t0
        ee = ', '.join(f"{e:.4f}" for e in r['extra_energies'])
        print(f"  pos={r['mean_pos']:.2f}mm  max={r['max_pos']:.2f}mm  "
              f"dE={r['max_de']:.1f}keV  loss={r['loss']:.6f}  "
              f"relocs={r['cumulative_relocs']}  extras_dead={r['n_extra_dead']}/5  "
              f"extras=[{ee}]  {dt:.0f}s")
        plot_combo(r, truth_params)
        results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — 10/5 overcomplete, {STEPS} steps")
    print(f"{'='*80}")
    print(f"{'Name':<20} {'Pos':>6} {'Max':>6} {'dE':>6} {'Loss':>9} {'Relocs':>6} {'Dead':>5}")
    for r in results:
        print(f"{r['name']:<20} {r['mean_pos']:6.2f} {r['max_pos']:6.2f} "
              f"{r['max_de']:6.1f} {r['loss']:9.6f} {r['cumulative_relocs']:6d} "
              f"{r['n_extra_dead']:3d}/5")


if __name__ == '__main__':
    main()
