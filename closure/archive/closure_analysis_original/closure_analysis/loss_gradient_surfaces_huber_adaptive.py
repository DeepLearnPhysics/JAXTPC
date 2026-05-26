"""
Plot loss and gradient surfaces for position-space Huber sliced distance.

Position-space Wasserstein integral:
    W = ∫ H(x - T(x)) × f_a(x) dx

where:
  - F_a = interp(pos_grid, pos_sorted, CDF) — CDF in fp → strong weight gradient
  - T(x) = Q_b(F_a(x)) = interp(F_a, CDF_b, pos_b) — F_a as query x
  - f_a × dx = diff(F_a) — mass per grid bin, gradient through diff(interp fp)
  - H(x - T(x)) — Huber spatial cost

Weight gradients flow through two chained interps: fp → F_a → query x → T.
No normalization, no mass matching term, no stop_gradient needed.

Generates 4 figures: huber_posspace_surface_{x,y,z,dE}.png

Run from project root:
    python3 closure_analysis/loss_gradient_surfaces_huber_adaptive.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud

PLANES = [0, 1, 2]
K = 25000
N_PROJ = 500
N_GRID = 2000
DELTA = 0.01  # Position-space Huber threshold


def make_huber_posspace(n_proj, delta, n_grid):
    """Position-space Huber sliced Wasserstein.

    Uses two chained interps:
      1. CDF interp: interp(pos_grid, pos_sorted, CDF) — CDF in fp (strong weight grad)
      2. Transport interp: interp(F_a, CDF_b, pos_b) — F_a as query x (carries weight grad)

    Integration weights from diff(F_a) = mass of a per grid bin.
    """
    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    pos_grid = jnp.linspace(0.0, 1.0, n_grid)

    @jax.jit
    def sw(pts_a, wts_a, pts_b, wts_b):
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(proj_a_i, proj_b_i):
            sort_a = jnp.argsort(proj_a_i)
            sort_b = jnp.argsort(proj_b_i)

            pos_a_sorted = proj_a_i[sort_a]
            pos_b_sorted = proj_b_i[sort_b]

            cdf_a = jnp.cumsum(wts_a[sort_a])
            cdf_b = jnp.cumsum(wts_b[sort_b])

            # Step 1: CDF of a at position grid (weights in fp → STRONG gradient)
            F_a = jnp.interp(pos_grid, pos_a_sorted, cdf_a)

            # Step 2: Transport map T(x) = Q_b(F_a(x))
            # F_a carries weight gradient from step 1 → flows through query x
            T = jnp.interp(F_a, cdf_b, pos_b_sorted)

            # Step 3: Spatial Huber cost
            diff_val = pos_grid - T
            abs_diff = jnp.abs(diff_val)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_val ** 2,
                              delta * (abs_diff - 0.5 * delta))

            # Step 4: Integration weight = mass of a per grid bin (via diff)
            mass_a = jnp.diff(F_a)  # (n_grid-1,)

            return jnp.sum(huber[:-1] * mass_a)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    return sw


# ── Setup ────────────────────────────────────────────────────────────────────

print("Loading detector config...")
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
sim = DetectorSimulator(detector_config, differentiable=True, n_segments=1)
forward = sim.build_forward()

truth_pos = np.array([-100.0, 50.0, 100.0])
truth_de = 1.0
truth_seg = SegmentData(positions_mm=jnp.array([truth_pos]), de=jnp.array([truth_de]))
target_signals = forward(truth_seg)

target_planes = {}
for p in PLANES:
    pts, w = signal_to_pointcloud(target_signals[p], K)
    target_planes[p] = (pts, w)

out_dir = os.path.dirname(os.path.abspath(__file__))
kernel = make_huber_posspace(N_PROJ, DELTA, N_GRID)

sweep_configs = {
    'x':  {'idx': 0, 'range': (-195, -5),  'n': 80, 'truth': None, 'unit': 'mm'},
    'y':  {'idx': 1, 'range': (-150, 250), 'n': 80, 'truth': None, 'unit': 'mm'},
    'z':  {'idx': 2, 'range': (-100, 300), 'n': 80, 'truth': None, 'unit': 'mm'},
    'dE': {'idx': -1, 'range': (0.1, 3.0),  'n': 80, 'truth': 1.0,  'unit': 'MeV'},
}

FD_EPS = 0.05
FD_EPS_DE = 0.005


def make_loss_fn(axis_name, axis_idx):
    if axis_name == 'dE':
        def loss_fn(val):
            seg = SegmentData(positions_mm=jnp.array([truth_pos]), de=jnp.array([val]))
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                pts, w = signal_to_pointcloud(sigs[p], K)
                loss = loss + kernel(pts, w, target_planes[p][0], target_planes[p][1])
            return loss
    else:
        def loss_fn(val):
            pos = jnp.array([truth_pos]).at[0, axis_idx].set(val)
            seg = SegmentData(positions_mm=pos, de=jnp.array([truth_de]))
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                pts, w = signal_to_pointcloud(sigs[p], K)
                loss = loss + kernel(pts, w, target_planes[p][0], target_planes[p][1])
            return loss
    return loss_fn


# ── Warm up JIT ──────────────────────────────────────────────────────────────

print(f"Params: K={K}, n_proj={N_PROJ}, n_grid={N_GRID}, delta={DELTA}")
print("Warming up JIT...")
for axis_name, cfg in sweep_configs.items():
    fn = make_loss_fn(axis_name, cfg['idx'])
    test_val = cfg['truth'] if cfg['truth'] is not None else truth_pos[cfg['idx']]
    t0 = time.time()
    _ = fn(test_val)
    _ = jax.grad(fn)(test_val)
    print(f"  {axis_name}: {time.time()-t0:.1f}s")
print("Ready.\n")


# ── Generate plots ───────────────────────────────────────────────────────────

for axis_name, cfg in sweep_configs.items():
    axis_idx = cfg['idx']
    lo, hi = cfg['range']
    n = cfg['n']
    unit = cfg['unit']
    truth_val = cfg['truth'] if cfg['truth'] is not None else truth_pos[axis_idx]
    eps = FD_EPS_DE if axis_name == 'dE' else FD_EPS
    x_vals = np.linspace(lo, hi, n)

    loss_fn = make_loss_fn(axis_name, axis_idx)
    grad_fn = jax.grad(loss_fn)

    t0 = time.time()
    print(f"--- {axis_name} (truth={truth_val}) ---")

    print(f"  Computing losses...")
    losses = np.array([float(loss_fn(float(v))) for v in x_vals])
    print(f"  Computing autodiff gradients...")
    ad_grads = np.array([float(grad_fn(float(v))) for v in x_vals])
    print(f"  Computing finite-diff gradients...")
    fd_grads = np.array([
        (float(loss_fn(float(v + eps))) - float(loss_fn(float(v - eps)))) / (2 * eps)
        for v in x_vals
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_vals, losses, '-', color='#27ae60', lw=1.5)
    axes[0].axvline(truth_val, color='green', ls='--', lw=2,
                    label=f'Truth {axis_name}={truth_val:.1f}')
    axes[0].set_xlabel(f'{axis_name} ({unit})')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Landscape')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_vals, ad_grads, '-', color='red', lw=1.5, label='Autodiff')
    axes[1].plot(x_vals, fd_grads, '--', color='blue', lw=1.5, alpha=0.7,
                 label=f'FD (eps={eps})')
    axes[1].axhline(0, color='k', ls='-', lw=0.5)
    axes[1].axvline(truth_val, color='green', ls='--', lw=2,
                    label=f'Truth {axis_name}={truth_val:.1f}')
    axes[1].set_xlabel(f'{axis_name} ({unit})')
    axes[1].set_ylabel(f'd(Loss)/d{axis_name}')
    axes[1].set_title('Gradient (Autodiff vs FD)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f'Position-Space Huber Sliced \u2014 varying {axis_name}  |  K={K}, '
        f'n_proj={N_PROJ}, n_grid={N_GRID}, \u03b4={DELTA}',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fname = f'huber_posspace_surface_{axis_name}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  {time.time()-t0:.0f}s \u2192 {fname}")

print("\nDone!")
