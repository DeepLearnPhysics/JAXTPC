"""
Benchmark Huber uniform SW timing as a function of K, n_grid, n_proj.

Varies one parameter at a time while holding the other two at defaults.
Times both forward and backward (value_and_grad) passes on the full
closure pipeline (sim → pointcloud → Huber SW summed over 3 planes).

Uses n_seg=5 segments to match the proven closure baseline.

Run: python3 closure_analysis/benchmark_huber_scaling.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import time

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from closure_analysis.optimization_closure import TRUTH_BANK, INIT_OFFSET

PLANES = [0, 1, 2]
DELTA = 0.01

# Defaults
K_DEFAULT = 10000
GRID_DEFAULT = 500
PROJ_DEFAULT = 200


def make_huber_sw(n_proj, delta, n_grid):
    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    grid = jnp.linspace(1e-6, 1.0 - 1e-6, n_grid)

    @jax.jit
    def sw(pts_a, wts_a, pts_b, wts_b):
        wts_a = wts_a / jnp.sum(wts_a)
        wts_b = wts_b / jnp.sum(wts_b)
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T

        def w1d(proj_a_i, proj_b_i):
            sort_a = jnp.argsort(proj_a_i)
            sort_b = jnp.argsort(proj_b_i)
            cdf_a = jnp.cumsum(wts_a[sort_a])
            cdf_b = jnp.cumsum(wts_b[sort_b])
            quant_a = jnp.interp(grid, cdf_a, proj_a_i[sort_a])
            quant_b = jnp.interp(grid, cdf_b, proj_b_i[sort_b])
            diff = quant_a - quant_b
            abs_diff = jnp.abs(diff)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.mean(huber)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)
    return sw


def build_closure_loss(forward, target_clouds, k, n_proj, n_grid):
    """Build full closure loss: sim → pointcloud → Huber SW over 3 planes."""
    kernel = make_huber_sw(n_proj, DELTA, n_grid)

    def loss_fn(params):
        seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
        sigs = forward(seg)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], k)
            tp, tw = target_clouds[p]
            loss = loss + kernel(pts, w, tp, tw)
        return loss
    return loss_fn


def time_vg(vg_fn, params, n_trials=20):
    """Time value_and_grad over n_trials, return median ms."""
    times = []
    for _ in range(n_trials):
        t0 = time.time()
        loss, grads = vg_fn(params)
        jax.block_until_ready(grads)
        times.append(time.time() - t0)
    times = np.array(times) * 1000
    return np.median(times), np.std(times)


def main():
    n_seg = 5
    truth_params = TRUTH_BANK[:n_seg]
    init_params = jnp.array(truth_params + INIT_OFFSET)

    # Check signal shape to know max K
    print("Building simulator...")
    config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(config, differentiable=True, n_segments=n_seg)
    forward = sim.build_forward()

    truth_seg = SegmentData(
        positions_mm=jnp.array(truth_params[:, :3]),
        de=jnp.array(truth_params[:, 3]),
    )
    target_signals = forward(truth_seg)
    sig_shape = target_signals[0].shape
    max_pixels = sig_shape[0] * sig_shape[1]
    print(f"Signal shape: {sig_shape} = {max_pixels} pixels per plane")

    # ── Sweep K ──────────────────────────────────────────────────────────
    k_values = [1000, 5000, 10000, 25000, 50000, 100000]
    k_values = [k for k in k_values if k <= max_pixels]

    print(f"\n{'='*70}")
    print(f"SWEEP K  (n_proj={PROJ_DEFAULT}, n_grid={GRID_DEFAULT})")
    print(f"{'='*70}")
    print(f"  {'K':>8s}  {'JIT (s)':>8s}  {'vg median (ms)':>15s}  {'std (ms)':>10s}")

    for k in k_values:
        # Build target clouds at this K
        target_clouds = {}
        for p in PLANES:
            pts, w = signal_to_pointcloud(target_signals[p], k)
            target_clouds[p] = (pts, w)

        loss_fn = build_closure_loss(forward, target_clouds, k,
                                     PROJ_DEFAULT, GRID_DEFAULT)
        vg_fn = jax.value_and_grad(loss_fn)

        # JIT warmup
        t0 = time.time()
        loss, grads = vg_fn(init_params)
        jax.block_until_ready(grads)
        jit_time = time.time() - t0

        med, std = time_vg(vg_fn, init_params)
        print(f"  {k:>8d}  {jit_time:>8.1f}  {med:>15.1f}  {std:>10.1f}")

    # ── Sweep n_grid ─────────────────────────────────────────────────────
    grid_values = [50, 100, 250, 500, 1000, 2000, 5000]

    # Build target clouds once at K_DEFAULT
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K_DEFAULT)
        target_clouds[p] = (pts, w)

    print(f"\n{'='*70}")
    print(f"SWEEP n_grid  (K={K_DEFAULT}, n_proj={PROJ_DEFAULT})")
    print(f"{'='*70}")
    print(f"  {'n_grid':>8s}  {'JIT (s)':>8s}  {'vg median (ms)':>15s}  {'std (ms)':>10s}")

    for ng in grid_values:
        loss_fn = build_closure_loss(forward, target_clouds, K_DEFAULT,
                                     PROJ_DEFAULT, ng)
        vg_fn = jax.value_and_grad(loss_fn)

        t0 = time.time()
        loss, grads = vg_fn(init_params)
        jax.block_until_ready(grads)
        jit_time = time.time() - t0

        med, std = time_vg(vg_fn, init_params)
        print(f"  {ng:>8d}  {jit_time:>8.1f}  {med:>15.1f}  {std:>10.1f}")

    # ── Sweep n_proj ─────────────────────────────────────────────────────
    proj_values = [2, 10, 25, 50, 100, 200, 500, 1000]

    print(f"\n{'='*70}")
    print(f"SWEEP n_proj  (K={K_DEFAULT}, n_grid={GRID_DEFAULT})")
    print(f"{'='*70}")
    print(f"  {'n_proj':>8s}  {'JIT (s)':>8s}  {'vg median (ms)':>15s}  {'std (ms)':>10s}")

    for np_ in proj_values:
        loss_fn = build_closure_loss(forward, target_clouds, K_DEFAULT,
                                     np_, GRID_DEFAULT)
        vg_fn = jax.value_and_grad(loss_fn)

        t0 = time.time()
        loss, grads = vg_fn(init_params)
        jax.block_until_ready(grads)
        jit_time = time.time() - t0

        med, std = time_vg(vg_fn, init_params)
        print(f"  {np_:>8d}  {jit_time:>8.1f}  {med:>15.1f}  {std:>10.1f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
