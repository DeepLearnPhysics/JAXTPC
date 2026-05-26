"""
Closure test with Adam optimization using Huber SW + mass matching loss.

Uses uniform-direction Huber sliced Wasserstein (normalized quantiles for
spatial term, mass ratio penalty for energy) summed over 3 east-side planes.

Run from project root:
    python3 closure_analysis/huber_closure.py N
where N is the number of segments (default 5).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import time

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from closure_analysis.optimization_closure import (
    TRUTH_BANK, INIT_OFFSET, best_permutation,
)

PLANES = [0, 1, 2]
K = 10000
N_PROJ = 200
N_GRID = 500
DELTA = 0.01
LAMBDA_MASS = 1e-4

LR = 0.3
B1 = 0.95
DE_SCALE = 1.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Huber SW + mass matching kernel ─────────────────────────────────────────

def make_huber_sw_mass(n_proj, delta, n_grid, lambda_mass):
    """Huber uniform SW with normalized quantiles + mass matching.

    spatial = mean Huber(quantile_a - quantile_b)  over projections
    mass    = (sum(wts_a)/sum(wts_b) - 1)^2
    loss    = spatial + lambda_mass * mass
    """
    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    grid = jnp.linspace(1e-6, 1.0 - 1e-6, n_grid)

    @jax.jit
    def sw(pts_a, wts_a, pts_b, wts_b):
        total_a = jnp.sum(wts_a)
        total_b = jnp.sum(wts_b)
        wts_a_norm = wts_a / total_a
        wts_b_norm = wts_b / total_b

        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(proj_a_i, proj_b_i):
            sort_a = jnp.argsort(proj_a_i)
            sort_b = jnp.argsort(proj_b_i)
            cdf_a = jnp.cumsum(wts_a_norm[sort_a])
            cdf_b = jnp.cumsum(wts_b_norm[sort_b])
            quant_a = jnp.interp(grid, cdf_a, proj_a_i[sort_a])
            quant_b = jnp.interp(grid, cdf_b, proj_b_i[sort_b])
            diff = quant_a - quant_b
            abs_diff = jnp.abs(diff)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.mean(huber)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        spatial = jnp.mean(costs)

        mass_ratio = total_a / total_b
        mass_term = (mass_ratio - 1.0) ** 2

        return spatial + lambda_mass * mass_term

    return sw


# ── Loss builder ─────────────────────────────────────────────────────────────

def build_huber_loss(forward, target_clouds, kernel):
    """Build Huber SW loss: params (N,4) -> scalar, summed over 3 planes."""
    def loss_fn(params):
        if params.ndim == 1:
            seg = SegmentData(positions_mm=params[:3][None], de=params[3:4])
        else:
            seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
        sigs = forward(seg)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            loss = loss + kernel(pts, w, target_clouds[p][0], target_clouds[p][1])
        return loss
    return loss_fn


# ── Adam optimizer ───────────────────────────────────────────────────────────

def run_adam(loss_fn, init_params, lr, n_steps, b1=0.9, de_scale=1.0,
             print_every=20):
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_steps, alpha=0.01)
    optimizer = optax.adam(schedule, b1=b1)
    params = init_params
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(loss_fn)

    losses = []
    param_history = []

    for step in range(n_steps):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        if updates.ndim == 1:
            updates = updates.at[3].multiply(de_scale)
        else:
            updates = updates.at[:, 3].multiply(de_scale)
        params = optax.apply_updates(params, updates)

        loss_val = float(loss)
        losses.append(loss_val)
        param_history.append(np.array(params))

        if step % print_every == 0 or step == n_steps - 1:
            print(f"  Step {step:4d}: loss = {loss_val:.8e}")

    return np.array(losses), np.array(param_history), params


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_closure(n_seg, losses, param_history, assignment, errors):
    seg_colors = plt.cm.tab10(np.linspace(0, 1, max(n_seg, 3)))

    if n_seg <= 3:
        _plot_per_component(n_seg, losses, param_history, assignment, seg_colors)
    else:
        _plot_total_error(n_seg, losses, param_history, assignment, seg_colors)


def _plot_per_component(n_seg, losses, param_history, assignment, seg_colors):
    pos_labels = ['x', 'y', 'z']
    pos_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    seg_de_colors = ['#d62728', '#9467bd', '#8c564b']

    if n_seg == 1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].semilogy(losses, 'b-', lw=1.2)
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('Huber SW Loss')
        axes[0].set_title('Loss Convergence'); axes[0].grid(True, alpha=0.3)

        truth = assignment[0]
        for i, (lbl, clr) in enumerate(zip(pos_labels, pos_colors)):
            axes[1].plot(param_history[:, 0, i] - truth[i], color=clr,
                         lw=1.2, label=lbl)
        axes[1].axhline(0, color='k', ls='--', lw=0.8)
        axes[1].set_xlabel('Step'); axes[1].set_ylabel('Error (mm)')
        axes[1].set_title('Position Errors')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot((param_history[:, 0, 3] - truth[3]) * 1000,
                     color='#d62728', lw=1.2)
        axes[2].axhline(0, color='k', ls='--', lw=0.8)
        axes[2].set_xlabel('Step'); axes[2].set_ylabel('Error (keV)')
        axes[2].set_title('Energy Error'); axes[2].grid(True, alpha=0.3)

    elif n_seg == 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].semilogy(losses, 'b-', lw=1.2)
        axes[0, 0].set_xlabel('Step'); axes[0, 0].set_ylabel('Huber SW Loss')
        axes[0, 0].set_title('Loss Convergence'); axes[0, 0].grid(True, alpha=0.3)

        for s in range(2):
            axes[0, 1].plot(param_history[:, s, 3] * 1000,
                            color=seg_de_colors[s], lw=1.2, label=f'Seg {s}')
            axes[0, 1].axhline(assignment[s, 3] * 1000,
                               color=seg_de_colors[s], ls=':', lw=1.0, alpha=0.6)
        axes[0, 1].set_xlabel('Step'); axes[0, 1].set_ylabel('dE (keV)')
        axes[0, 1].set_title('Energy Trajectories (dotted = truth)')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        for s in range(2):
            ax = axes[1, s]
            for i, (lbl, clr) in enumerate(zip(pos_labels, pos_colors)):
                ax.plot(param_history[:, s, i], color=clr, lw=1.2, label=lbl)
                ax.axhline(assignment[s, i], color=clr, ls=':', lw=1.0, alpha=0.6)
            ax.set_xlabel('Step'); ax.set_ylabel('Position (mm)')
            ax.set_title(f'Segment {s} Positions (dotted = truth)')
            ax.legend(); ax.grid(True, alpha=0.3)

    else:  # n_seg == 3
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax_loss = fig.add_subplot(gs[0, 0])
        ax_loss.semilogy(losses, 'b-', lw=1.2)
        ax_loss.set_xlabel('Step'); ax_loss.set_ylabel('Huber SW Loss')
        ax_loss.set_title('Loss Convergence'); ax_loss.grid(True, alpha=0.3)

        ax_de = fig.add_subplot(gs[0, 1:])
        for s in range(3):
            ax_de.plot(param_history[:, s, 3] * 1000,
                       color=seg_de_colors[s], lw=1.2, label=f'Seg {s}')
            ax_de.axhline(assignment[s, 3] * 1000, color=seg_de_colors[s],
                          ls=':', lw=1.0, alpha=0.6)
        ax_de.set_xlabel('Step'); ax_de.set_ylabel('dE (keV)')
        ax_de.set_title('Energy Trajectories (dotted = truth)')
        ax_de.legend(); ax_de.grid(True, alpha=0.3)

        for s in range(3):
            ax = fig.add_subplot(gs[1, s])
            for i, (lbl, clr) in enumerate(zip(pos_labels, pos_colors)):
                ax.plot(param_history[:, s, i], color=clr, lw=1.2, label=lbl)
                ax.axhline(assignment[s, i], color=clr, ls=':', lw=1.0, alpha=0.6)
            ax.set_xlabel('Step'); ax.set_ylabel('Position (mm)')
            ax.set_title(f'Segment {s} Positions (dotted = truth)')
            ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'{n_seg}-Segment Huber SW Closure (Adam, lr={LR}, b1={B1}, '
        f'K={K}, n_proj={N_PROJ})',
        fontsize=13, fontweight='bold')
    if n_seg <= 2:
        fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'huber_closure_{n_seg}seg.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def _plot_total_error(n_seg, losses, param_history, assignment, seg_colors):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].semilogy(losses, 'b-', lw=1.2)
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Huber SW Loss')
    axes[0].set_title('Loss Convergence'); axes[0].grid(True, alpha=0.3)

    for s in range(n_seg):
        pos_err = np.sqrt(np.sum(
            (param_history[:, s, :3] - assignment[s, :3]) ** 2, axis=1))
        axes[1].plot(pos_err, color=seg_colors[s], lw=1.2, label=f'Seg {s}')
    axes[1].axhline(0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('||pos error|| (mm)')
    axes[1].set_title('Position Error per Segment')
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3)

    for s in range(n_seg):
        axes[2].plot(param_history[:, s, 3] * 1000, color=seg_colors[s],
                     lw=1.2, label=f'Seg {s}')
        axes[2].axhline(assignment[s, 3] * 1000, color=seg_colors[s],
                         ls=':', lw=1.0, alpha=0.5)
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('dE (keV)')
    axes[2].set_title('Energy Trajectories (dotted = truth)')
    axes[2].legend(fontsize=7, ncol=2); axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f'{n_seg}-Segment Huber SW Closure (Adam, lr={LR}, b1={B1}, '
        f'K={K}, n_proj={N_PROJ})',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'huber_closure_{n_seg}seg.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# ── Main closure ─────────────────────────────────────────────────────────────

def run_closure(n_seg):
    assert n_seg <= len(TRUTH_BANK), f"Only {len(TRUTH_BANK)} truth segments defined"

    n_steps = 300 + 100 * (n_seg - 1)
    print_every = max(20, n_steps // 15)

    print(f"\n{'=' * 70}")
    print(f"{n_seg}-SEGMENT HUBER SW CLOSURE TEST")
    print(f"{'=' * 70}")
    print(f"Hyperparams: lr={LR}, b1={B1}, de_scale={DE_SCALE}, steps={n_steps}")
    print(f"Loss: Huber SW + mass matching (K={K}, n_proj={N_PROJ}, "
          f"n_grid={N_GRID}, delta={DELTA}, lambda_mass={LAMBDA_MASS})")

    truth_params = TRUTH_BANK[:n_seg]
    init_params = jnp.array(truth_params + INIT_OFFSET)

    for i in range(n_seg):
        print(f"  Seg {i}: truth={truth_params[i]}  init={np.array(init_params[i])}")

    print("Building simulator...")
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg)
    forward = sim.build_forward()

    # Generate target signals and pointclouds
    truth_seg = SegmentData(
        positions_mm=jnp.array(truth_params[:, :3]),
        de=jnp.array(truth_params[:, 3]),
    )
    target_signals = forward(truth_seg)

    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    kernel = make_huber_sw_mass(N_PROJ, DELTA, N_GRID, LAMBDA_MASS)
    loss_fn = build_huber_loss(forward, target_clouds, kernel)

    if n_seg == 1:
        init_flat = init_params[0]  # (4,)
    else:
        init_flat = init_params  # (N, 4)

    # Warm up JIT
    print("Warming up JIT...")
    t0 = time.time()
    _ = loss_fn(init_flat)
    jax.block_until_ready(_)
    t1 = time.time()
    _ = jax.grad(loss_fn)(init_flat)
    jax.block_until_ready(_)
    t2 = time.time()
    print(f"JIT warm-up done: forward={t1-t0:.1f}s, grad={t2-t1:.1f}s")

    # Optimize
    print(f"\nOptimizing ({n_steps} steps)...")
    t0 = time.time()
    losses, param_history, final_params = run_adam(
        loss_fn, init_flat, lr=LR, n_steps=n_steps, b1=B1, de_scale=DE_SCALE,
        print_every=print_every,
    )
    opt_time = time.time() - t0
    print(f"Optimization done in {opt_time:.1f}s ({opt_time/n_steps*1000:.0f} ms/step)")

    final = np.array(final_params)
    if n_seg == 1:
        final_2d = final[None, :]
        param_history_3d = param_history[:, None, :]
    else:
        final_2d = final
        param_history_3d = param_history

    # Match segments to truth
    assignment, errors = best_permutation(final_2d, truth_params)

    print(f"\nResults:")
    for i in range(n_seg):
        e = errors[i]
        print(f"  Seg {i}: x={e[0]:+.3f} mm, y={e[1]:+.3f} mm, "
              f"z={e[2]:+.3f} mm, dE={e[3]*1000:+.1f} keV")
    max_pos = np.max(np.abs(errors[:, :3]))
    max_de = np.max(np.abs(errors[:, 3])) * 1000
    mean_pos = np.mean(np.sqrt(np.sum(errors[:, :3] ** 2, axis=1)))
    print(f"  Max position error: {max_pos:.3f} mm")
    print(f"  Mean position error: {mean_pos:.3f} mm")
    print(f"  Max dE error:       {max_de:.1f} keV")

    _plot_closure(n_seg, losses, param_history_3d, assignment, errors)

    return errors


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_closure(n)
