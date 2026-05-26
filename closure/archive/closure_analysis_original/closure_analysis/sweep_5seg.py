"""
Sweep script for 5-seg = 5-truth Adam+noise optimization.

Finds optimal schedule, LR, and noise configuration before introducing
L1/relocation for the overcomplete case.

Run from project root:
    python3 closure_analysis/sweep_5seg.py [--steps 1000]
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
    TRUTH_BANK, INIT_OFFSET, PLANES, K, N_PROJ,
    build_loss_fn, best_permutation,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_SEG = 5
N_TRUTH = 5
MIN_ENERGY = 0.001


def run_config(config, grad_fn, schedule_fn, total_steps):
    """Run a single config, return results dict."""
    name = config['name']
    lr = config['lr']
    lr_e_mult = config.get('lr_e_mult', 0.03)
    noise_lr = config.get('noise_lr', 0.0)
    noise_coupling = config.get('noise_coupling', 'linear')
    b1 = config.get('b1', 0.9)
    b2 = config.get('b2', 0.999)

    # Build schedule
    sched_type = config['schedule']
    if sched_type == 'cosine':
        alpha = config.get('cosine_alpha', 0.01)
        schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=total_steps, alpha=alpha)
    elif sched_type == 'exp':
        decay_rate = config['decay_rate']
        schedule = optax.exponential_decay(
            init_value=lr, transition_steps=1, decay_rate=decay_rate)
    elif sched_type == 'constant':
        schedule = optax.constant_schedule(lr)
    elif sched_type == 'warmup_exp':
        warmup_steps = config.get('warmup_steps', 100)
        decay_rate = config['decay_rate']
        warmup = optax.linear_schedule(
            init_value=lr * 0.01, end_value=lr,
            transition_steps=warmup_steps)
        decay = optax.exponential_decay(
            init_value=lr, transition_steps=1, decay_rate=decay_rate)
        schedule = optax.join_schedules(
            [warmup, decay], boundaries=[warmup_steps])
    else:
        raise ValueError(f"Unknown schedule: {sched_type}")

    optimizer = optax.adam(schedule, b1=b1, b2=b2)

    # Init params: truth + offset
    truth_params = TRUTH_BANK[:N_TRUTH]
    init_params = jnp.array(truth_params + INIT_OFFSET)

    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    losses = []
    param_history = []

    for step in range(total_steps):
        (loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(lr_e_mult)
        params = optax.apply_updates(params, updates)

        # Energy floor
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], MIN_ENERGY))

        # Noise on positions
        if noise_lr > 0:
            lr_current = float(schedule(step))
            if noise_coupling == 'quadratic':
                noise_scale = (lr_current ** 2 / lr) * noise_lr
            else:  # linear
                noise_scale = lr_current * noise_lr
            rng_key, noise_key = jax.random.split(rng_key)
            noise_vec = jax.random.normal(noise_key, shape=(N_SEG, 3))
            params = params.at[:, :3].add(noise_scale * noise_vec)

        losses.append(float(loss))
        param_history.append(np.array(params))

    # Evaluate
    final = np.array(params)
    param_history = np.array(param_history)
    assignment, errors, matched_idx = best_permutation(final, truth_params)
    matched_errors = errors[matched_idx]

    mean_pos = np.mean(np.sqrt(np.sum(matched_errors[:, :3] ** 2, axis=1)))
    max_pos = np.max(np.sqrt(np.sum(matched_errors[:, :3] ** 2, axis=1)))
    max_de = np.max(np.abs(matched_errors[:, 3])) * 1000
    mean_de = np.mean(np.abs(matched_errors[:, 3])) * 1000
    final_loss = losses[-1]

    # Find step where mean_pos first < 3mm (convergence speed)
    step_3mm = total_steps  # default: never
    for s in range(0, total_steps, 10):
        ph = param_history[s]
        _, errs, midx = best_permutation(ph, truth_params)
        mp = np.mean(np.sqrt(np.sum(errs[midx, :3] ** 2, axis=1)))
        if mp < 3.0:
            step_3mm = s
            break

    return {
        'name': name,
        'loss': final_loss,
        'mean_pos': mean_pos,
        'max_pos': max_pos,
        'mean_de': mean_de,
        'max_de': max_de,
        'step_3mm': step_3mm,
        'losses': losses,
        'param_history': param_history,
        'assignment': assignment,
        'matched_idx': matched_idx,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    total_steps = args.steps

    # =========================================================================
    # Config grid — focused on the most informative comparisons
    # =========================================================================
    configs = [
        # --- Group 1: Schedule comparison at LR=0.3, no noise ---
        {'name': 'cosine_0.3',       'schedule': 'cosine', 'lr': 0.3, 'lr_e_mult': 0.03, 'noise_lr': 0},
        {'name': 'exp999_0.3',       'schedule': 'exp', 'lr': 0.3, 'decay_rate': 0.999, 'lr_e_mult': 0.03, 'noise_lr': 0},
        {'name': 'exp9995_0.3',      'schedule': 'exp', 'lr': 0.3, 'decay_rate': 0.9995, 'lr_e_mult': 0.03, 'noise_lr': 0},
        {'name': 'const_0.3',        'schedule': 'constant', 'lr': 0.3, 'lr_e_mult': 0.03, 'noise_lr': 0},

        # --- Group 2: LR sweep with exp d=0.9995, no noise ---
        {'name': 'exp9995_0.5',      'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.01, 'noise_lr': 0},
        {'name': 'exp9995_0.7',      'schedule': 'exp', 'lr': 0.7, 'decay_rate': 0.9995, 'lr_e_mult': 0.01, 'noise_lr': 0},
        {'name': 'exp9995_1.0',      'schedule': 'exp', 'lr': 1.0, 'decay_rate': 0.9995, 'lr_e_mult': 0.01, 'noise_lr': 0},

        # --- Group 3: Decay rate sweep at LR=0.5, no noise ---
        {'name': 'exp999_0.5',       'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.999, 'lr_e_mult': 0.01, 'noise_lr': 0},
        {'name': 'exp9998_0.5',      'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9998, 'lr_e_mult': 0.01, 'noise_lr': 0},

        # --- Group 4: Noise sweep on best no-noise config ---
        {'name': 'exp9995_0.5_n2q',  'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.01,
         'noise_lr': 2, 'noise_coupling': 'quadratic'},
        {'name': 'exp9995_0.5_n5q',  'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.01,
         'noise_lr': 5, 'noise_coupling': 'quadratic'},
        {'name': 'exp9995_0.5_n5l',  'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.01,
         'noise_lr': 5, 'noise_coupling': 'linear'},
        {'name': 'exp9995_0.5_n8q',  'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.01,
         'noise_lr': 8, 'noise_coupling': 'quadratic'},

        # --- Group 5: LR_energy_mult comparison ---
        {'name': 'exp9995_0.5_e03',  'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.03, 'noise_lr': 0},
        {'name': 'exp9995_0.5_e10',  'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.10, 'noise_lr': 0},

        # --- Group 6: b1 comparison ---
        {'name': 'exp9995_0.5_b09',  'schedule': 'exp', 'lr': 0.5, 'decay_rate': 0.9995, 'lr_e_mult': 0.01, 'noise_lr': 0, 'b1': 0.9},

        # --- Group 7: warmup + exp ---
        {'name': 'warmexp_0.5',      'schedule': 'warmup_exp', 'lr': 0.5, 'decay_rate': 0.9995,
         'warmup_steps': 50, 'lr_e_mult': 0.01, 'noise_lr': 0},
    ]

    print(f"Running {len(configs)} configs, {total_steps} steps each, 5-seg = 5-truth")
    print(f"Init offset: {INIT_OFFSET[:3]} mm, dE offset: {INIT_OFFSET[3]} MeV")
    print(f"Init distance: {np.linalg.norm(INIT_OFFSET[:3]):.1f} mm")
    print()

    # --- Build simulator (shared across all configs) ---
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEG)
    fwd = sim.build_forward()

    # Target signals
    truth_params = TRUTH_BANK[:N_TRUTH]
    truth_seg = SegmentData(
        positions_mm=jnp.array(truth_params[:, :3]),
        de=jnp.array(truth_params[:, 3]),
    )
    target_signals = fwd(truth_seg)
    key = jax.random.PRNGKey(42)
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    loss_fn = build_loss_fn(fwd, target_clouds, key)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # JIT warmup
    print("Warming up JIT...")
    init_params = jnp.array(truth_params + INIT_OFFSET)
    _ = grad_fn(init_params)
    print("JIT ready.\n")

    # --- Run all configs ---
    results = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        print(f"[{i+1}/{len(configs)}] {cfg['name']}...", end=' ', flush=True)
        res = run_config(cfg, grad_fn, None, total_steps)
        elapsed = time.time() - t0
        print(f"pos={res['mean_pos']:.2f}mm, loss={res['loss']:.6f}, "
              f"step<3mm={res['step_3mm']}, {elapsed:.1f}s")
        results.append(res)

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'=' * 100}")
    print(f"RESULTS SUMMARY — 5-seg, 5-truth, {total_steps} steps")
    print(f"{'=' * 100}")
    print(f"{'#':>2s}  {'Name':<24s}  {'Loss':>8s}  {'Pos(mm)':>8s}  {'MaxPos':>8s}  "
          f"{'dE(keV)':>8s}  {'MaxdE':>8s}  {'Step<3mm':>8s}")
    print(f"{'--':>2s}  {'----':<24s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  "
          f"{'--------':>8s}  {'--------':>8s}  {'--------':>8s}")

    for i, r in enumerate(results):
        s3 = f"{r['step_3mm']}" if r['step_3mm'] < total_steps else ">1000"
        print(f"{i+1:2d}  {r['name']:<24s}  {r['loss']:8.6f}  {r['mean_pos']:8.2f}  "
              f"{r['max_pos']:8.2f}  {r['mean_de']:8.1f}  {r['max_de']:8.1f}  {s3:>8s}")

    # Sort by mean_pos
    sorted_results = sorted(results, key=lambda r: r['mean_pos'])
    print(f"\nTop 5 by position:")
    for i, r in enumerate(sorted_results[:5]):
        s3 = f"{r['step_3mm']}" if r['step_3mm'] < total_steps else ">1000"
        print(f"  {i+1}. {r['name']:<24s}  pos={r['mean_pos']:.2f}mm  "
              f"dE={r['max_de']:.1f}keV  step<3mm={s3}")

    # Sort by step_3mm (convergence speed)
    fast_results = [r for r in results if r['step_3mm'] < total_steps]
    if fast_results:
        fast_results.sort(key=lambda r: r['step_3mm'])
        print(f"\nFastest to <3mm:")
        for i, r in enumerate(fast_results[:5]):
            print(f"  {i+1}. {r['name']:<24s}  step<3mm={r['step_3mm']}  "
                  f"final_pos={r['mean_pos']:.2f}mm")

    # =========================================================================
    # Plot: loss curves for all configs
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    LABEL_SIZE = 13
    TITLE_SIZE = 14
    TICK_SIZE = 11
    LEGEND_SIZE = 9

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    # Top-left: all loss curves
    ax = axes[0, 0]
    for i, r in enumerate(results):
        ax.semilogy(r['losses'], color=colors[i], lw=1.0, alpha=0.7, label=r['name'])
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Loss', fontsize=LABEL_SIZE)
    ax.set_title('Loss Convergence', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE - 2, ncol=2, loc='upper right')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # Top-right: mean position error over time (top 6 configs)
    ax = axes[0, 1]
    top6 = sorted_results[:6]
    for r in top6:
        # Compute position error trajectory
        truth = TRUTH_BANK[:N_TRUTH]
        pos_errs = []
        for s in range(0, total_steps, max(1, total_steps // 200)):
            ph = r['param_history'][s]
            _, errs, midx = best_permutation(ph, truth)
            mp = np.mean(np.sqrt(np.sum(errs[midx, :3] ** 2, axis=1)))
            pos_errs.append((s, mp))
        steps_arr, errs_arr = zip(*pos_errs)
        ax.plot(steps_arr, errs_arr, lw=1.5, label=r['name'])
    ax.axhline(3.0, color='k', ls='--', lw=0.8, alpha=0.5, label='3mm target')
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Mean pos error (mm)', fontsize=LABEL_SIZE)
    ax.set_title('Position Convergence (top 6)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # Bottom-left: bar chart of final mean pos
    ax = axes[1, 0]
    names = [r['name'] for r in sorted_results]
    pos_vals = [r['mean_pos'] for r in sorted_results]
    bar_colors = ['green' if p < 3 else 'orange' if p < 5 else 'red' for p in pos_vals]
    ax.barh(range(len(names)), pos_vals, color=bar_colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=TICK_SIZE - 2)
    ax.set_xlabel('Mean pos error (mm)', fontsize=LABEL_SIZE)
    ax.set_title('Final Position Error (sorted)', fontsize=TITLE_SIZE)
    ax.axvline(3.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Bottom-right: bar chart of convergence speed (step to <3mm)
    ax = axes[1, 1]
    if fast_results:
        fast_names = [r['name'] for r in fast_results]
        fast_steps = [r['step_3mm'] for r in fast_results]
        ax.barh(range(len(fast_names)), fast_steps, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(fast_names)))
        ax.set_yticklabels(fast_names, fontsize=TICK_SIZE - 2)
        ax.set_xlabel('Steps to reach <3mm', fontsize=LABEL_SIZE)
        ax.set_title('Convergence Speed', fontsize=TITLE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'No config reached <3mm', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_title('Convergence Speed', fontsize=TITLE_SIZE)

    fig.suptitle(f'5-Seg Sweep — {total_steps} steps, {len(configs)} configs',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'sweep_5seg_{total_steps}steps.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved {fname}")


if __name__ == '__main__':
    main()
