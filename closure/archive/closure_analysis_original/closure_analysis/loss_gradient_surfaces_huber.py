"""
Plot loss and gradient surfaces for Huber uniform SW + mass matching term,
varying x, y, z, and dE independently. Loss is summed across all active planes.

Approach:
  - Spatial term: normalized quantile Huber SW (proven autodiff for positions)
  - Mass term:    (sum(wts_a)/sum(wts_b) - 1)^2  (dE sensitivity)
  - Combined:     spatial + lambda_mass * mass_term

Generates 4 figures: huber_surface_{x,y,z,dE}.png

Run from project root:
    python3 closure_analysis/loss_gradient_surfaces_huber.py
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
DELTA = 0.01
LAMBDA_MASS = 1e-4   # mass matching coefficient


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
        # Normalize weights for quantile computation
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

        # Mass matching: penalize total weight mismatch
        mass_ratio = total_a / total_b
        mass_term = (mass_ratio - 1.0) ** 2

        return spatial + lambda_mass * mass_term

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
kernel = make_huber_sw_mass(N_PROJ, DELTA, N_GRID, LAMBDA_MASS)

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

print(f"Params: K={K}, n_proj={N_PROJ}, n_grid={N_GRID}, delta={DELTA}, "
      f"lambda_mass={LAMBDA_MASS}")
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
        f'Huber SW + mass — varying {axis_name}  |  K={K}, n_proj={N_PROJ}, '
        f'n_grid={N_GRID}, λ_m={LAMBDA_MASS}',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fname = f'huber_surface_{axis_name}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  {time.time()-t0:.0f}s → {fname}")

print("\nDone!")
