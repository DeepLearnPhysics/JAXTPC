"""
Plot loss and gradient surfaces for MSE and SW losses separately,
varying x, y, z, and dE independently. Loss is summed across all active planes.

Generates 8 figures:
  - mse_surface_{x,y,z,dE}.png
  - sw_surface_{x,y,z,dE}.png

Run from project root:
    python3 closure_analysis/loss_gradient_surfaces.py
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

# East side planes: U=0, V=1, Y=2
PLANES = [0, 1, 2]
K = 1000
N_PROJ = 200

print("Loading detector config...")
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
sim = DetectorSimulator(detector_config, differentiable=True, n_segments=1)
forward = sim.build_forward()

truth_pos = np.array([-100.0, 50.0, 100.0])
truth_de = 1.0
truth_seg = SegmentData(positions_mm=jnp.array([truth_pos]), de=jnp.array([truth_de]))
target_signals = forward(truth_seg)
key = jax.random.PRNGKey(42)

# Pre-compute targets for each plane
target_planes = {}
for p in PLANES:
    sig = target_signals[p]
    pts, w = signal_to_pointcloud(sig, K)
    target_planes[p] = (sig, pts, w)

out_dir = os.path.dirname(os.path.abspath(__file__))

# ── Sweep ranges for each axis ───────────────────────────────────────────────
sweep_configs = {
    'x':  {'idx': 0, 'range': (-195, -5),  'n': 80, 'truth': None, 'unit': 'mm'},
    'y':  {'idx': 1, 'range': (-150, 250), 'n': 80, 'truth': None, 'unit': 'mm'},
    'z':  {'idx': 2, 'range': (-100, 300), 'n': 80, 'truth': None, 'unit': 'mm'},
    'dE': {'idx': -1, 'range': (0.1, 3.0),  'n': 80, 'truth': 1.0,  'unit': 'MeV'},
}

FD_EPS = 0.05  # for position axes (mm)
FD_EPS_DE = 0.005  # for dE (MeV)


def make_loss_fns(axis_name, axis_idx):
    """Build MSE and SW loss functions that vary one parameter, summed over all planes."""

    if axis_name == 'dE':
        def mse_loss(val):
            seg = SegmentData(positions_mm=jnp.array([truth_pos]), de=jnp.array([val]))
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                loss = loss + jnp.mean((sigs[p] - target_planes[p][0]) ** 2)
            return loss

        def sw_loss(val):
            seg = SegmentData(positions_mm=jnp.array([truth_pos]), de=jnp.array([val]))
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                pts, w = signal_to_pointcloud(sigs[p], K)
                loss = loss + sliced_wasserstein_loss_jit(
                    pts, w, target_planes[p][1], target_planes[p][2],
                    key, n_projections=N_PROJ
                )
            return loss
    else:
        def mse_loss(val):
            pos = jnp.array([truth_pos]).at[0, axis_idx].set(val)
            seg = SegmentData(positions_mm=pos, de=jnp.array([truth_de]))
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                loss = loss + jnp.mean((sigs[p] - target_planes[p][0]) ** 2)
            return loss

        def sw_loss(val):
            pos = jnp.array([truth_pos]).at[0, axis_idx].set(val)
            seg = SegmentData(positions_mm=pos, de=jnp.array([truth_de]))
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                pts, w = signal_to_pointcloud(sigs[p], K)
                loss = loss + sliced_wasserstein_loss_jit(
                    pts, w, target_planes[p][1], target_planes[p][2],
                    key, n_projections=N_PROJ
                )
            return loss

    return mse_loss, sw_loss


# ── Warm up JIT ──────────────────────────────────────────────────────────────
print("Warming up JIT...")
for axis_name, cfg in sweep_configs.items():
    mse_fn, sw_fn = make_loss_fns(axis_name, cfg['idx'])
    test_val = cfg['truth'] if cfg['truth'] is not None else truth_pos[cfg['idx']]
    _ = mse_fn(test_val)
    _ = jax.grad(mse_fn)(test_val)
    _ = sw_fn(test_val)
    _ = jax.grad(sw_fn)(test_val)
print("Ready.\n")

# ── Generate plots for each axis ─────────────────────────────────────────────

for axis_name, cfg in sweep_configs.items():
    axis_idx = cfg['idx']
    lo, hi = cfg['range']
    n = cfg['n']
    unit = cfg['unit']
    truth_val = cfg['truth'] if cfg['truth'] is not None else truth_pos[axis_idx]
    eps = FD_EPS_DE if axis_name == 'dE' else FD_EPS

    mse_fn, sw_fn = make_loss_fns(axis_name, axis_idx)
    mse_grad_fn = jax.grad(mse_fn)
    sw_grad_fn = jax.grad(sw_fn)

    x_vals = np.linspace(lo, hi, n)
    truth_label = f'Truth {axis_name}={truth_val:.1f}'

    print(f"--- {axis_name} (truth={truth_val}) ---")

    print(f"  Computing MSE losses...")
    mse_losses = np.array([float(mse_fn(float(v))) for v in x_vals])
    print(f"  Computing MSE autodiff gradients...")
    mse_ad_grads = np.array([float(mse_grad_fn(float(v))) for v in x_vals])
    print(f"  Computing MSE finite-diff gradients...")
    mse_fd_grads = np.array([
        (float(mse_fn(float(v + eps))) - float(mse_fn(float(v - eps)))) / (2 * eps)
        for v in x_vals
    ])

    print(f"  Computing SW losses...")
    sw_losses = np.array([float(sw_fn(float(v))) for v in x_vals])
    print(f"  Computing SW autodiff gradients...")
    sw_ad_grads = np.array([float(sw_grad_fn(float(v))) for v in x_vals])
    print(f"  Computing SW finite-diff gradients...")
    sw_fd_grads = np.array([
        (float(sw_fn(float(v + eps))) - float(sw_fn(float(v - eps)))) / (2 * eps)
        for v in x_vals
    ])

    # ── MSE figure (loss + gradient, 1x2) ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_vals, mse_losses, 'b-', lw=1.5)
    axes[0].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
    axes[0].set_xlabel(f'{axis_name} ({unit})')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss Landscape')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_vals, mse_ad_grads, 'r-', lw=1.5, label='Autodiff')
    axes[1].plot(x_vals, mse_fd_grads, 'b--', lw=1.5, alpha=0.7, label=f'FD (eps={eps})')
    axes[1].axhline(0, color='k', ls='-', lw=0.5)
    axes[1].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
    axes[1].set_xlabel(f'{axis_name} ({unit})')
    axes[1].set_ylabel(f'dMSE/d{axis_name}')
    axes[1].set_title('Gradient (Autodiff vs FD)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'MSE — varying {axis_name} (summed over U+V+Y planes)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fname = f'mse_surface_{axis_name}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

    # ── SW figure (loss + gradient, 1x2) ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_vals, sw_losses, 'b-', lw=1.5)
    axes[0].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
    axes[0].set_xlabel(f'{axis_name} ({unit})')
    axes[0].set_ylabel('SW Loss')
    axes[0].set_title('Loss Landscape')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_vals, sw_ad_grads, 'r-', lw=1.5, label='Autodiff')
    axes[1].plot(x_vals, sw_fd_grads, 'b--', lw=1.5, alpha=0.7, label=f'FD (eps={eps})')
    axes[1].axhline(0, color='k', ls='-', lw=0.5)
    axes[1].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
    axes[1].set_xlabel(f'{axis_name} ({unit})')
    axes[1].set_ylabel(f'dSW/d{axis_name}')
    axes[1].set_title('Gradient (Autodiff vs FD)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Sliced Wasserstein — varying {axis_name} (summed over U+V+Y planes)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fname = f'sw_surface_{axis_name}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

print("\nDone!")
