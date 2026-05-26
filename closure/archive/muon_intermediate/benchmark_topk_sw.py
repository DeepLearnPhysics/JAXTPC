"""
Benchmark top-k extraction + SW loss computation as a function of K.

Measures only the pointcloud extraction and SW distance (not the simulation).
Averages over multiple trials, skipping the first (warmup).

Run from project root:
    python3 closure_analysis_muon/benchmark_topk_sw.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments,
    build_muon_forward,
)

N_SEGMENTS = 2800
STEP_SIZE_MM = 0.5
N_PROJ = 200
N_TRIALS = 20  # total calls per K (first is warmup, rest averaged)
PLANES = [0, 1, 2]

TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 2
TRUTH_ENERGY = 200.0

# Perturbed signal for a nonzero SW distance
PERT_THETA = TRUTH_THETA + 0.15

K_VALUES = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60, flush=True)
    print("BENCHMARK: top-k + SW vs K", flush=True)
    print(f"N_PROJ={N_PROJ}, N_TRIALS={N_TRIALS} (first skipped)", flush=True)
    print("=" * 60, flush=True)

    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    # Generate truth and perturbed signals (just once, reuse arrays)
    print("Generating signals...", flush=True)
    pos_fixed = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], dtype=jnp.float32)

    truth_pos, truth_de = generate_muon_segments(
        jnp.float32(TRUTH_ENERGY), pos_fixed,
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    )
    truth_sigs = forward(truth_pos, truth_de)

    pert_pos, pert_de = generate_muon_segments(
        jnp.float32(TRUTH_ENERGY), pos_fixed,
        jnp.float32(PERT_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    )
    pert_sigs = forward(pert_pos, pert_de)

    for s in truth_sigs:
        jax.block_until_ready(s)
    for s in pert_sigs:
        jax.block_until_ready(s)
    print("  Signals ready.", flush=True)

    ot_key = jax.random.PRNGKey(42)

    # For each K, build a jitted function that does topk + SW for all 3 planes
    results = []

    for K in K_VALUES:
        print(f"\nK = {K:,}", flush=True)

        # Pre-extract truth pointclouds for this K (not timed — these are fixed)
        target_clouds = {}
        for p in PLANES:
            pts, w = signal_to_pointcloud(truth_sigs[p], K)
            target_clouds[p] = (pts, w)

        # JIT the topk + SW for the perturbed signal
        def topk_sw(pert_sig_0, pert_sig_1, pert_sig_2):
            loss = 0.0
            for p, sig in enumerate([pert_sig_0, pert_sig_1, pert_sig_2]):
                pts, w = signal_to_pointcloud(sig, K)
                loss = loss + sliced_wasserstein_loss_jit(
                    pts, w, target_clouds[p][0], target_clouds[p][1],
                    ot_key, n_projections=N_PROJ,
                )
            return loss

        topk_sw_jit = jax.jit(topk_sw)

        # Warmup (compile)
        t0 = time.time()
        loss = topk_sw_jit(pert_sigs[0], pert_sigs[1], pert_sigs[2])
        jax.block_until_ready(loss)
        compile_time = time.time() - t0
        print(f"  compile: {compile_time:.3f}s, loss={float(loss):.2f}", flush=True)

        # Timed runs
        times = []
        for trial in range(N_TRIALS):
            t0 = time.time()
            loss = topk_sw_jit(pert_sigs[0], pert_sigs[1], pert_sigs[2])
            jax.block_until_ready(loss)
            times.append(time.time() - t0)

        times = np.array(times)
        mean_ms = times.mean() * 1000
        std_ms = times.std() * 1000
        print(f"  runtime: {mean_ms:.1f} ± {std_ms:.1f} ms  "
              f"(min={times.min()*1000:.1f}, max={times.max()*1000:.1f})", flush=True)

        # Also time topk + SW + backward (value_and_grad)
        topk_sw_vg = jax.jit(jax.value_and_grad(topk_sw, argnums=(0, 1, 2)))

        # Warmup backward
        t0 = time.time()
        loss_g, grads_g = topk_sw_vg(pert_sigs[0], pert_sigs[1], pert_sigs[2])
        jax.block_until_ready(loss_g)
        for g in grads_g:
            jax.block_until_ready(g)
        bwd_compile = time.time() - t0
        print(f"  bwd compile: {bwd_compile:.3f}s", flush=True)

        bwd_times = []
        for trial in range(N_TRIALS):
            t0 = time.time()
            loss_g, grads_g = topk_sw_vg(pert_sigs[0], pert_sigs[1], pert_sigs[2])
            jax.block_until_ready(loss_g)
            for g in grads_g:
                jax.block_until_ready(g)
            bwd_times.append(time.time() - t0)

        bwd_times = np.array(bwd_times)
        bwd_mean_ms = bwd_times.mean() * 1000
        bwd_std_ms = bwd_times.std() * 1000
        print(f"  fwd+bwd: {bwd_mean_ms:.1f} ± {bwd_std_ms:.1f} ms", flush=True)

        results.append({
            'K': K,
            'fwd_mean_ms': mean_ms,
            'fwd_std_ms': std_ms,
            'bwd_mean_ms': bwd_mean_ms,
            'bwd_std_ms': bwd_std_ms,
            'compile_s': compile_time,
            'bwd_compile_s': bwd_compile,
            'loss': float(loss),
        })

    # =====================================================================
    # Plot
    # =====================================================================
    print("\nGenerating plot...", flush=True)

    Ks = [r['K'] for r in results]
    fwd_means = [r['fwd_mean_ms'] for r in results]
    fwd_stds = [r['fwd_std_ms'] for r in results]
    bwd_means = [r['bwd_mean_ms'] for r in results]
    bwd_stds = [r['bwd_std_ms'] for r in results]
    compiles = [r['compile_s'] for r in results]
    bwd_compiles = [r['bwd_compile_s'] for r in results]
    losses = [r['loss'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Runtime vs K
    axes[0].errorbar(Ks, fwd_means, yerr=fwd_stds, fmt='bo-', lw=1.5,
                      capsize=3, label='Forward only')
    axes[0].errorbar(Ks, bwd_means, yerr=bwd_stds, fmt='rs-', lw=1.5,
                      capsize=3, label='Forward + Backward')
    axes[0].set_xlabel('K (top-k points)')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Runtime: top-k + SW (3 planes)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Compile time vs K
    axes[1].plot(Ks, compiles, 'bo-', lw=1.5, label='Forward compile')
    axes[1].plot(Ks, bwd_compiles, 'rs-', lw=1.5, label='Fwd+Bwd compile')
    axes[1].set_xlabel('K (top-k points)')
    axes[1].set_ylabel('Compile time (s)')
    axes[1].set_title('JIT Compilation Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    # Loss vs K
    axes[2].plot(Ks, losses, 'go-', lw=1.5)
    axes[2].set_xlabel('K (top-k points)')
    axes[2].set_ylabel('SW Loss')
    axes[2].set_title('SW Loss Value vs K')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')

    fig.suptitle(
        f'Top-K + Sliced Wasserstein Scaling (N_PROJ={N_PROJ}, 3 east planes, {N_TRIALS} trials)',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'benchmark_topk_sw.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)

    # Print summary table
    print(f"\n{'K':>8} | {'Fwd (ms)':>10} | {'Fwd+Bwd (ms)':>14} | "
          f"{'Compile (s)':>12} | {'Bwd Compile':>12} | {'Loss':>8}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"{r['K']:>8,} | {r['fwd_mean_ms']:>8.1f}±{r['fwd_std_ms']:<4.1f}| "
              f"{r['bwd_mean_ms']:>10.1f}±{r['bwd_std_ms']:<6.1f}| "
              f"{r['compile_s']:>11.2f}s | {r['bwd_compile_s']:>10.2f}s | "
              f"{r['loss']:>8.2f}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
