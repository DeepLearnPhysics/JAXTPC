"""
Benchmark: adaptive grid vs diff_q vs normalized interp Huber SW timing.

Sweeps K, n_proj, n_grid independently. Times value_and_grad through
the full closure pipeline (sim → pointcloud → Huber SW summed over 3 planes).

Uses n_seg=5 segments.

Run: python3 closure_analysis/benchmark_diffq_vs_interp.py
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


# ── Adaptive grid Huber (interp speed, correct weight grads) ────────────────

def make_huber_adaptive(n_proj, delta, n_grid):
    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    base_grid = jnp.linspace(1e-6, 1.0 - 1e-6, n_grid)

    @jax.jit
    def sw(pts_a, wts_a, pts_b, wts_b):
        sum_a = jnp.sum(wts_a)
        sum_b = jnp.sum(wts_b)
        max_mass = jnp.maximum(sum_a, sum_b)
        grid = base_grid * max_mass

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

        spatial = jnp.mean(jax.vmap(w1d)(proj_a.T, proj_b.T))
        return spatial * max_mass
    return sw


# ── Normalized interp Huber (old baseline) ──────────────────────────────────

def make_huber_interp(n_proj, delta, n_grid):
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


# ── diff_q-based Huber (OTT-style) ─────────────────────────────────────────

def make_huber_diffq(n_proj, delta, n_pts):
    angles = jnp.linspace(0, jnp.pi, n_proj, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    @jax.jit
    def sw(pts_a, wts_a, pts_b, wts_b):
        proj_a = pts_a @ directions.T
        proj_b = pts_b @ directions.T
        n = pts_a.shape[0]

        def w1d(pa, pb):
            i_a = jnp.argsort(pa)
            i_b = jnp.argsort(pb)
            sorted_a = pa[i_a]
            sorted_b = pb[i_b]

            all_values = jnp.concatenate([sorted_a, sorted_b])
            all_sorter = jnp.argsort(all_values)
            all_sorted = all_values[all_sorter]

            a_pdf = jnp.concatenate([wts_a[i_a], jnp.zeros(n)])
            a_pdf = a_pdf[all_sorter]
            b_pdf = jnp.concatenate([jnp.zeros(n), wts_b[i_b]])
            b_pdf = b_pdf[all_sorter]

            a_cdf = jnp.cumsum(a_pdf)
            b_cdf = jnp.cumsum(b_pdf)

            all_cdfs = jnp.concatenate([a_cdf, b_cdf])
            quantile_levels = jnp.sort(all_cdfs)

            i_a_inv = jnp.searchsorted(a_cdf, quantile_levels)
            i_b_inv = jnp.searchsorted(b_cdf, quantile_levels)
            qa = all_sorted[i_a_inv]
            qb = all_sorted[i_b_inv]

            diff_q = jnp.diff(quantile_levels)
            diff_pos = qa[1:] - qb[1:]
            abs_diff = jnp.abs(diff_pos)
            huber = jnp.where(abs_diff <= delta,
                              0.5 * diff_pos ** 2,
                              delta * (abs_diff - 0.5 * delta))
            return jnp.sum(huber * diff_q)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)
    return sw


# ── Benchmark harness ────────────────────────────────────────────────────────

def build_loss(forward, target_clouds, kernel, k):
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
    print(f"Signal shape: {sig_shape} = {max_pixels} pixels per plane\n")

    methods_label = ["adaptive", "interp", "diff_q"]

    def run_sweep(sweep_name, sweep_values, make_k, make_nproj, make_ngrid):
        print(f"\n{'='*80}")
        print(f"SWEEP {sweep_name}")
        print(f"{'='*80}")
        print(f"  {sweep_name:>8s}  {'Method':>10s}  {'JIT (s)':>8s}  "
              f"{'vg med (ms)':>12s}  {'std (ms)':>10s}")

        for val in sweep_values:
            k = make_k(val)
            np_ = make_nproj(val)
            ng = make_ngrid(val)

            if k > max_pixels:
                continue

            tc = {}
            for p in PLANES:
                pts, w = signal_to_pointcloud(target_signals[p], k)
                tc[p] = (pts, w)

            makers = [
                ("adaptive", lambda: make_huber_adaptive(np_, DELTA, ng)),
                ("interp", lambda: make_huber_interp(np_, DELTA, ng)),
                ("diff_q", lambda: make_huber_diffq(np_, DELTA, k)),
            ]

            for method_name, make_kernel in makers:
                kernel = make_kernel()
                loss_fn = build_loss(forward, tc, kernel, k)
                vg_fn = jax.value_and_grad(loss_fn)

                t0 = time.time()
                loss, grads = vg_fn(init_params)
                jax.block_until_ready(grads)
                jit_time = time.time() - t0

                med, std = time_vg(vg_fn, init_params)
                print(f"  {val:>8}  {method_name:>10s}  {jit_time:>8.1f}  "
                      f"{med:>12.1f}  {std:>10.1f}")

    # ── Sweep K ──────────────────────────────────────────────────────────
    k_values = [1000, 5000, 10000, 25000, 50000, 100000]
    run_sweep("K",
              k_values,
              make_k=lambda v: v,
              make_nproj=lambda v: PROJ_DEFAULT,
              make_ngrid=lambda v: GRID_DEFAULT)

    # ── Sweep n_proj ─────────────────────────────────────────────────────
    proj_values = [10, 25, 50, 100, 200, 500]
    run_sweep("n_proj",
              proj_values,
              make_k=lambda v: K_DEFAULT,
              make_nproj=lambda v: v,
              make_ngrid=lambda v: GRID_DEFAULT)

    # ── Sweep n_grid ─────────────────────────────────────────────────────
    grid_values = [50, 100, 250, 500, 1000, 2000, 5000]
    run_sweep("n_grid",
              grid_values,
              make_k=lambda v: K_DEFAULT,
              make_nproj=lambda v: PROJ_DEFAULT,
              make_ngrid=lambda v: v)

    print("\nNote: diff_q does NOT use n_grid (resolution = 4*K from CDF union).")
    print("Note: adaptive uses n_grid for its base grid (same as interp).")
    print("\nDone!")


if __name__ == '__main__':
    main()
