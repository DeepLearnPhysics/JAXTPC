"""
Plot loss and gradient surfaces for dense Huber SW (full image, no top-K).

Every pixel in the (W, T) signal is treated as a point. For each projection
direction, all W*T points are sorted and used for CDF/quantile/Huber.

This tests how slow the full-image approach is vs the sparse top-K approach.

Generates 4 figures: huber_dense_surface_{x,y,z,dE}.png

Run from project root:
    python3 closure_analysis/loss_gradient_surfaces_huber_dense.py
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

PLANES = [0, 1, 2]
N_PROJ = 10
N_GRID = 2000
DELTA = 0.01
LAMBDA_MASS = 1e-4


def make_huber_sw_dense(signal_shape, n_proj, delta, n_grid, lambda_mass,
                        mm_per_wire=3.0, mm_per_tick=0.8):
    """Huber SW on full (W, T) signals — no top-K, every pixel is a point.

    Pre-computes pixel positions (fixed grid). At runtime: project, sort,
    CDF, quantile, Huber. Uses lax.map over projections to control memory.
    """
    W, T = signal_shape
    n_pixels = W * T
    max_extent = max(W * mm_per_wire, T * mm_per_tick)

    # Pre-compute all pixel positions: (W*T, 2) normalized
    wire_idx = jnp.arange(W, dtype=jnp.float32)
    time_idx = jnp.arange(T, dtype=jnp.float32)
    # (W, T) meshgrid → flatten to (W*T,)
    wire_flat = jnp.repeat(wire_idx, T) * mm_per_wire / max_extent
    time_flat = jnp.tile(time_idx, W) * mm_per_tick / max_extent
    positions = jnp.stack([wire_flat, time_flat], axis=-1)  # (W*T, 2)

    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # (n_proj, 2)
    grid = jnp.linspace(1e-6, 1.0 - 1e-6, n_grid)

    # Pre-compute projections and sort orders (static — don't depend on signal)
    # proj_all: (n_proj, W*T)
    proj_all = positions @ directions.T  # (W*T, n_proj) → transpose below

    print(f"    Dense kernel: {W}x{T} = {n_pixels:,} pixels, "
          f"n_proj={n_proj}, positions={positions.shape}")

    # Pre-compute sort permutations for each direction (expensive but one-time)
    print(f"    Pre-computing {n_proj} sort permutations of {n_pixels:,} elements...")
    t0 = time.time()
    # Sort each direction's projections
    sort_perms = jnp.argsort(proj_all, axis=0)  # (W*T, n_proj) - sort along pixel axis
    sorted_positions = jnp.take_along_axis(proj_all, sort_perms, axis=0)  # (W*T, n_proj)
    print(f"    Sort pre-computation done in {time.time()-t0:.1f}s")

    @jax.jit
    def sw(signal_a, signal_b):
        abs_a = jnp.abs(signal_a).ravel()  # (W*T,)
        abs_b = jnp.abs(signal_b).ravel()
        total_a = jnp.sum(abs_a)
        total_b = jnp.sum(abs_b)

        def compute_one(d_idx):
            perm = sort_perms[:, d_idx]        # (W*T,)
            sorted_pos = sorted_positions[:, d_idx]  # (W*T,)

            sorted_wts_a = abs_a[perm]
            sorted_wts_b = abs_b[perm]

            cdf_a = jnp.cumsum(sorted_wts_a) / (total_a + 1e-10)
            cdf_b = jnp.cumsum(sorted_wts_b) / (total_b + 1e-10)

            quant_a = jnp.interp(grid, cdf_a, sorted_pos)
            quant_b = jnp.interp(grid, cdf_b, sorted_pos)

            diff = quant_a - quant_b
            abs_diff = jnp.abs(diff)
            return jnp.mean(jnp.where(abs_diff <= delta,
                                       0.5 * diff ** 2,
                                       delta * (abs_diff - 0.5 * delta)))

        costs = jax.vmap(compute_one)(jnp.arange(n_proj))
        spatial = jnp.mean(costs)

        mass_ratio = total_a / (total_b + 1e-10)
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

# Check signal shapes per plane
for p in PLANES:
    print(f"  Plane {p}: signal shape = {target_signals[p].shape}")

# Build dense kernels per plane (shapes may differ)
kernels = {}
for p in PLANES:
    shape = target_signals[p].shape
    print(f"  Building dense kernel for plane {p} ({shape})...")
    kernels[p] = make_huber_sw_dense(shape, N_PROJ, DELTA, N_GRID, LAMBDA_MASS)

out_dir = os.path.dirname(os.path.abspath(__file__))

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
                loss = loss + kernels[p](sigs[p], target_signals[p])
            return loss
    else:
        def loss_fn(val):
            pos = jnp.array([truth_pos]).at[0, axis_idx].set(val)
            seg = SegmentData(positions_mm=pos, de=jnp.array([truth_de]))
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                loss = loss + kernels[p](sigs[p], target_signals[p])
            return loss
    return loss_fn


# ── Warm up JIT + benchmark ─────────────────────────────────────────────────

print(f"\nParams: n_proj={N_PROJ}, n_grid={N_GRID}, delta={DELTA}, "
      f"lambda_mass={LAMBDA_MASS}")
print("Warming up JIT (this may take a while for dense kernels)...")

for axis_name, cfg in sweep_configs.items():
    fn = make_loss_fn(axis_name, cfg['idx'])
    test_val = cfg['truth'] if cfg['truth'] is not None else truth_pos[cfg['idx']]
    t0 = time.time()
    loss_val = fn(test_val)
    jax.block_until_ready(loss_val)
    t1 = time.time()
    grad_val = jax.grad(fn)(test_val)
    jax.block_until_ready(grad_val)
    t2 = time.time()
    print(f"  {axis_name}: JIT forward={t1-t0:.1f}s, JIT grad={t2-t1:.1f}s, "
          f"loss={float(loss_val):.6e}")

# Time a single value_and_grad call after warmup
fn_x = make_loss_fn('x', 0)
vg_fn = jax.value_and_grad(fn_x)
_ = vg_fn(-100.0)  # ensure warm
times = []
for _ in range(5):
    t0 = time.time()
    loss, grad = vg_fn(-90.0)
    jax.block_until_ready(grad)
    times.append(time.time() - t0)
med_ms = np.median(times) * 1000
print(f"\n  Single value_and_grad: {med_ms:.0f} ms (median of 5)")
est_total = med_ms * 80 * 4 * 4 / 1000 / 60  # 80 pts * 4 axes * 4 evals (loss+grad+2fd)
print(f"  Estimated total for all surface plots: {est_total:.1f} min")
print()


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
        f'Dense Huber SW — varying {axis_name}  |  n_proj={N_PROJ}, '
        f'n_grid={N_GRID}, δ={DELTA}',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fname = f'huber_dense_surface_{axis_name}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  {time.time()-t0:.0f}s → {fname}")

print("\nDone!")
