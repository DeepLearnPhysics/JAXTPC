"""
Closure test with Adam optimization for N-segment reconstruction.

Uses Sliced Wasserstein loss summed over 3 east-side planes (U, V, Y).

Run from project root:
    python3 closure_analysis/optimization_closure.py N
where N is the number of segments (default 1).
"""

import sys, os
from itertools import permutations
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

PLANES = [0, 1, 2]  # East-side U, V, Y
K = 1000
N_PROJ = 200
DE_SCALE = 1.0  # dE update multiplier (Adam already normalizes per-param)
LR = 0.3
B1 = 0.95  # higher momentum smooths noisy SW gradients (default 0.9)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Bank of truth segments: (x, y, z, dE) in mm / MeV
# Well-separated positions across the detector volume
TRUTH_BANK = np.array([
    [-100.0,   50.0, 100.0, 1.0],
    [-150.0,  -30.0, 200.0, 1.0],
    [ -50.0,  -80.0,  50.0, 0.8],
    [-120.0,  100.0, 280.0, 1.2],
    [ -80.0, -120.0, 150.0, 0.6],
    [-170.0,   20.0,  30.0, 0.9],
    [ -30.0,   80.0, 250.0, 1.1],
    [-140.0,  -90.0, 180.0, 0.7],
    [ -60.0,  130.0, 320.0, 1.3],
    [-180.0,  -50.0, 120.0, 0.5],
    [ -90.0,   70.0, 350.0, 1.0],
    [-160.0, -100.0,  80.0, 0.8],
    [ -40.0,  -20.0, 270.0, 1.4],
    [-110.0,  110.0, 170.0, 0.9],
    [ -70.0, -140.0, 230.0, 0.7],
    [-130.0,   40.0, 310.0, 1.1],
])

# Offsets applied to truth to get initial guess: +30 x, -30 y, +30 z, +0.5 dE
INIT_OFFSET = np.array([30.0, -30.0, 30.0, 0.5])


# =============================================================================
# Helpers
# =============================================================================

def build_sw_loss(forward, target_clouds, key):
    """Build SW loss: params (N,4) -> scalar, summed over 3 planes."""
    def loss_fn(params):
        if params.ndim == 1:
            seg = SegmentData(positions_mm=params[:3][None], de=params[3:4])
        else:
            seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
        sigs = forward(seg)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            loss = loss + sliced_wasserstein_loss_jit(
                pts, w, target_clouds[p][0], target_clouds[p][1],
                key, n_projections=N_PROJ,
            )
        return loss
    return loss_fn


def run_adam(loss_fn, init_params, lr, n_steps, b1=0.9, de_scale=1.0, print_every=20):
    """Run Adam optimization, return loss history and param history."""
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps, alpha=0.01)
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


def best_permutation(final, truth_params):
    """Find the permutation of truth rows that best matches final (min total L1)."""
    n = len(truth_params)
    best_err = np.inf
    best_perm = None
    for perm in permutations(range(n)):
        reordered = truth_params[list(perm)]
        err = np.abs(final - reordered).sum()
        if err < best_err:
            best_err = err
            best_perm = perm
    assignment = truth_params[list(best_perm)]
    return assignment, final - assignment


# =============================================================================
# General N-segment closure
# =============================================================================

def run_closure(n_seg):
    """Run closure test for n_seg segments."""
    assert n_seg <= len(TRUTH_BANK), f"Only {len(TRUTH_BANK)} truth segments defined"

    n_steps = 300 + 78 * (n_seg - 1)  # ~300 for 1, ~600 for 5, ~1000 for 10
    print_every = max(20, n_steps // 15)

    print(f"\n{'=' * 70}")
    print(f"{n_seg}-SEGMENT CLOSURE TEST")
    print(f"{'=' * 70}")
    print(f"Hyperparams: lr={LR}, b1={B1}, de_scale={DE_SCALE}, steps={n_steps}")

    truth_params = TRUTH_BANK[:n_seg]  # (n_seg, 4)
    init_params = jnp.array(truth_params + INIT_OFFSET)

    for i in range(n_seg):
        print(f"  Seg {i}: truth={truth_params[i]}  init={np.array(init_params[i])}")

    # Build simulator and forward
    print("Building simulator...")
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg)
    forward = sim.build_forward()

    # Generate target
    if n_seg == 1:
        truth_seg = SegmentData(
            positions_mm=jnp.array(truth_params[:, :3]),
            de=jnp.array(truth_params[:, 3]),
        )
        init_flat = init_params[0]  # (4,) for 1-seg
    else:
        truth_seg = SegmentData(
            positions_mm=jnp.array(truth_params[:, :3]),
            de=jnp.array(truth_params[:, 3]),
        )
        init_flat = init_params  # (N, 4)

    target_signals = forward(truth_seg)
    key = jax.random.PRNGKey(42)

    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    loss_fn = build_sw_loss(forward, target_clouds, key)

    # Warm up JIT
    print("Warming up JIT...")
    _ = loss_fn(init_flat)
    _ = jax.grad(loss_fn)(init_flat)
    print("JIT warm-up done.")

    # Optimize
    losses, param_history, final_params = run_adam(
        loss_fn, init_flat, lr=LR, n_steps=n_steps, b1=B1, de_scale=DE_SCALE,
        print_every=print_every,
    )

    final = np.array(final_params)
    if n_seg == 1:
        final_2d = final[None, :]          # (1, 4)
        param_history_3d = param_history[:, None, :]  # (steps, 1, 4)
    else:
        final_2d = final                    # (N, 4)
        param_history_3d = param_history    # (steps, N, 4)

    # Match segments to truth
    assignment, errors = best_permutation(final_2d, truth_params)

    print(f"\nResults:")
    for i in range(n_seg):
        e = errors[i]
        print(f"  Seg {i}: x={e[0]:+.3f} mm, y={e[1]:+.3f} mm, "
              f"z={e[2]:+.3f} mm, dE={e[3]*1000:+.1f} keV")
    max_pos = np.max(np.abs(errors[:, :3]))
    max_de = np.max(np.abs(errors[:, 3])) * 1000
    print(f"  Max position error: {max_pos:.3f} mm")
    print(f"  Max dE error:       {max_de:.1f} keV")

    # ── Plotting ──
    _plot_closure(n_seg, losses, param_history_3d, assignment, errors)

    return errors


# =============================================================================
# Plotting
# =============================================================================

def _plot_closure(n_seg, losses, param_history, assignment, errors):
    """Plot closure results. Per-component for N<=3, total error for N>3."""
    pos_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    seg_colors = plt.cm.tab10(np.linspace(0, 1, max(n_seg, 3)))

    if n_seg <= 3:
        _plot_per_component(n_seg, losses, param_history, assignment, seg_colors)
    else:
        _plot_total_error(n_seg, losses, param_history, assignment, seg_colors)


def _plot_per_component(n_seg, losses, param_history, assignment, seg_colors):
    """1-seg: 1x3, 2-seg: 2x2, 3-seg: 2x3."""
    pos_labels = ['x', 'y', 'z']
    pos_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    seg_de_colors = ['#d62728', '#9467bd', '#8c564b']

    if n_seg == 1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].semilogy(losses, 'b-', lw=1.2)
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('SW Loss')
        axes[0].set_title('Loss Convergence'); axes[0].grid(True, alpha=0.3)

        truth = assignment[0]
        for i, (lbl, clr) in enumerate(zip(pos_labels, pos_colors)):
            axes[1].plot(param_history[:, 0, i] - truth[i], color=clr, lw=1.2, label=lbl)
        axes[1].axhline(0, color='k', ls='--', lw=0.8)
        axes[1].set_xlabel('Step'); axes[1].set_ylabel('Error (mm)')
        axes[1].set_title('Position Errors'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot((param_history[:, 0, 3] - truth[3]) * 1000, color='#d62728', lw=1.2)
        axes[2].axhline(0, color='k', ls='--', lw=0.8)
        axes[2].set_xlabel('Step'); axes[2].set_ylabel('Error (keV)')
        axes[2].set_title('Energy Error'); axes[2].grid(True, alpha=0.3)

    elif n_seg == 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].semilogy(losses, 'b-', lw=1.2)
        axes[0, 0].set_xlabel('Step'); axes[0, 0].set_ylabel('SW Loss')
        axes[0, 0].set_title('Loss Convergence'); axes[0, 0].grid(True, alpha=0.3)

        for s in range(2):
            axes[0, 1].plot(param_history[:, s, 3] * 1000, color=seg_de_colors[s],
                           lw=1.2, label=f'Seg {s}')
            axes[0, 1].axhline(assignment[s, 3] * 1000, color=seg_de_colors[s],
                              ls=':', lw=1.0, alpha=0.6)
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
        ax_loss.set_xlabel('Step'); ax_loss.set_ylabel('SW Loss')
        ax_loss.set_title('Loss Convergence'); ax_loss.grid(True, alpha=0.3)

        ax_de = fig.add_subplot(gs[0, 1:])
        for s in range(3):
            ax_de.plot(param_history[:, s, 3] * 1000, color=seg_de_colors[s],
                      lw=1.2, label=f'Seg {s}')
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

    fig.suptitle(f'{n_seg}-Segment Closure Test (Adam, lr={LR}, b1={B1})',
                 fontsize=14, fontweight='bold')
    if n_seg <= 2:
        fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'closure_{n_seg}seg.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def _plot_total_error(n_seg, losses, param_history, assignment, seg_colors):
    """For N>3: loss, total position error per segment, dE per segment."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].semilogy(losses, 'b-', lw=1.2)
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('SW Loss')
    axes[0].set_title('Loss Convergence'); axes[0].grid(True, alpha=0.3)

    # Total position error per segment: ||pos - truth||
    for s in range(n_seg):
        pos_err = np.sqrt(np.sum((param_history[:, s, :3] - assignment[s, :3]) ** 2, axis=1))
        axes[1].plot(pos_err, color=seg_colors[s], lw=1.2, label=f'Seg {s}')
    axes[1].axhline(0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('||pos error|| (mm)')
    axes[1].set_title('Position Error per Segment')
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3)

    # dE per segment
    for s in range(n_seg):
        axes[2].plot(param_history[:, s, 3] * 1000, color=seg_colors[s],
                    lw=1.2, label=f'Seg {s}')
        axes[2].axhline(assignment[s, 3] * 1000, color=seg_colors[s],
                       ls=':', lw=1.0, alpha=0.5)
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('dE (keV)')
    axes[2].set_title('Energy Trajectories (dotted = truth)')
    axes[2].legend(fontsize=7, ncol=2); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'{n_seg}-Segment Closure Test (Adam, lr={LR}, b1={B1})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'closure_{n_seg}seg.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_closure(n)
