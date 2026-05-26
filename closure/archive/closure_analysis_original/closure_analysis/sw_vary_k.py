"""
Study how SW loss and gradients change with K (number of top-k points).

Uses K=3000 as the high-K reference (~all active pixels), then compares
lower K values. Sweeps x, y, z independently.

Generates 3 figures: sw_vary_k_x.png, sw_vary_k_y.png, sw_vary_k_z.png

Run from project root:
    python3 closure_analysis/sw_vary_k.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit

PLANES = [0, 1, 2]
N_PROJ = 200

# K values: well above active pixel count (~3180) down to sparse
K_VALUES = [10000, 5000, 3000, 1000, 500, 200]
K_REF = K_VALUES[0]

print("Loading detector config...")
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
sim = DetectorSimulator(detector_config, differentiable=True, n_segments=1)
forward = sim.build_forward()

truth_pos = np.array([-100.0, 50.0, 100.0])
truth_seg = SegmentData(positions_mm=jnp.array([truth_pos]), de=jnp.array([1.0]))
target_signals = forward(truth_seg)
key = jax.random.PRNGKey(42)

# Check active pixel count
n_active = int(jnp.sum(jnp.abs(target_signals[2]) > 0))
print(f"Active pixels (Y plane): {n_active}")

# Pre-compute targets for each (plane, K)
target_cache = {}
for K in K_VALUES:
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_cache[(p, K)] = (pts, w)

out_dir = os.path.dirname(os.path.abspath(__file__))

sweep_configs = {
    'x': {'idx': 0, 'range': (-195, -5),  'n': 60},
    'y': {'idx': 1, 'range': (-150, 250), 'n': 60},
    'z': {'idx': 2, 'range': (-100, 300), 'n': 60},
}


def make_sw_loss(axis_idx, K):
    """Build SW loss varying one coordinate with a specific K."""
    def sw_loss(val):
        pos = jnp.array([truth_pos]).at[0, axis_idx].set(val)
        seg = SegmentData(positions_mm=pos, de=jnp.array([1.0]))
        sigs = forward(seg)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            tgt_pts, tgt_w = target_cache[(p, K)]
            loss = loss + sliced_wasserstein_loss_jit(
                pts, w, tgt_pts, tgt_w, key, n_projections=N_PROJ
            )
        return loss
    return sw_loss


# ── Warm up JIT ──────────────────────────────────────────────────────────────
print("Warming up JIT...")
for K in K_VALUES:
    fn = make_sw_loss(0, K)
    _ = fn(truth_pos[0] - 30.0)
    _ = jax.grad(fn)(truth_pos[0] - 30.0)
print("Ready.\n")

# ── Compute for each axis ────────────────────────────────────────────────────

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(K_VALUES)))

for axis_name, cfg in sweep_configs.items():
    axis_idx = cfg['idx']
    lo, hi = cfg['range']
    n = cfg['n']
    truth_val = truth_pos[axis_idx]
    x_vals = np.linspace(lo, hi, n)

    print(f"--- Axis {axis_name} (truth={truth_val}) ---")

    all_losses = {}
    all_ad_grads = {}

    for ki, K in enumerate(K_VALUES):
        sw_fn = make_sw_loss(axis_idx, K)
        sw_grad_fn = jax.grad(sw_fn)

        print(f"  K={K}: computing losses...")
        losses = np.array([float(sw_fn(float(v))) for v in x_vals])
        print(f"  K={K}: computing gradients...")
        grads = np.array([float(sw_grad_fn(float(v))) for v in x_vals])

        all_losses[K] = losses
        all_ad_grads[K] = grads

    # ── Figure: 2x2 (loss, gradient, loss diff from ref, gradient diff) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: loss landscape for each K
    for ki, K in enumerate(K_VALUES):
        label = f'K={K}' + (' (ref)' if K == K_REF else '')
        lw = 2.0 if K == K_REF else 1.2
        axes[0, 0].plot(x_vals, all_losses[K] * 1e6, color=colors[ki], lw=lw, label=label)
    axes[0, 0].axvline(truth_val, color='green', ls='--', lw=2, alpha=0.7)
    axes[0, 0].set_xlabel(f'{axis_name} (mm)')
    axes[0, 0].set_ylabel('SW Loss (x10^-6)')
    axes[0, 0].set_title('Loss Landscape')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: gradient for each K
    for ki, K in enumerate(K_VALUES):
        label = f'K={K}' + (' (ref)' if K == K_REF else '')
        lw = 2.0 if K == K_REF else 1.2
        axes[0, 1].plot(x_vals, all_ad_grads[K], color=colors[ki], lw=lw, label=label)
    axes[0, 1].axhline(0, color='k', ls='-', lw=0.5)
    axes[0, 1].axvline(truth_val, color='green', ls='--', lw=2, alpha=0.7)
    axes[0, 1].set_xlabel(f'{axis_name} (mm)')
    axes[0, 1].set_ylabel(f'dSW/d{axis_name}')
    axes[0, 1].set_title('Gradient (Autodiff)')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: loss difference relative to K_REF
    ref_losses = all_losses[K_REF]
    for ki, K in enumerate(K_VALUES):
        if K == K_REF:
            continue
        diff = (all_losses[K] - ref_losses) * 1e6
        axes[1, 0].plot(x_vals, diff, color=colors[ki], lw=1.2, label=f'K={K} - K={K_REF}')
    axes[1, 0].axhline(0, color='k', ls='-', lw=0.5)
    axes[1, 0].axvline(truth_val, color='green', ls='--', lw=2, alpha=0.7)
    axes[1, 0].set_xlabel(f'{axis_name} (mm)')
    axes[1, 0].set_ylabel('Loss Diff (x10^-6)')
    axes[1, 0].set_title(f'Loss Difference from K={K_REF}')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: gradient difference relative to K_REF
    ref_grads = all_ad_grads[K_REF]
    for ki, K in enumerate(K_VALUES):
        if K == K_REF:
            continue
        diff = all_ad_grads[K] - ref_grads
        axes[1, 1].plot(x_vals, diff, color=colors[ki], lw=1.2, label=f'K={K} - K={K_REF}')
    axes[1, 1].axhline(0, color='k', ls='-', lw=0.5)
    axes[1, 1].axvline(truth_val, color='green', ls='--', lw=2, alpha=0.7)
    axes[1, 1].set_xlabel(f'{axis_name} (mm)')
    axes[1, 1].set_ylabel(f'Gradient Diff')
    axes[1, 1].set_title(f'Gradient Difference from K={K_REF}')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'SW Loss vs K — varying {axis_name} (U+V+Y planes)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fname = f'sw_vary_k_{axis_name}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

print("\nDone!")
