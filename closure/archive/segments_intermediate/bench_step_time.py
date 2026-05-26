"""
Benchmark: optimization step time vs number of segments.

For each n_seg, JIT-compiles the grad function, runs 5 warmup steps,
then times 20 steps and reports mean ± std.

Usage:
    python3 closure_analysis_full/sweeps/bench_step_time.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from closure_analysis_full.full_closure import DEFAULTS, build_loss_fn

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOBOLEV_S = 1.0
WARMUP_STEPS = 5
BENCH_STEPS = 20

NSEG_VALUES = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000]


def main():
    print("Step time benchmark: n_seg vs time per step")
    print(f"Warmup: {WARMUP_STEPS} steps, Bench: {BENCH_STEPS} steps\n")

    # Use a minimal truth signal — we just need shapes, not accuracy
    # Generate from mpvmpr_20.h5 event 0 (already cached/fast)
    from tools.loader import load_particle_step_data
    deposit_data = load_particle_step_data('mpvmpr_20.h5', 0, verbose=False)
    n_truth = deposit_data.positions_mm.shape[0]
    truth_pos = np.asarray(deposit_data.positions_mm)
    truth_de = np.asarray(deposit_data.de)

    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    total_pad = ((max(200_000, n_truth + 1000) + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(
        detector_config, use_bucketed=False, total_pad=total_pad,
        response_chunk_size=50_000, include_noise=False,
        include_electronics=False, include_track_hits=False,
        recombination_model='modified_box')
    response_signals, _ = sim_truth(deposit_data)

    truth_dict, active_planes, sob_dict = {}, [], {}
    for (side, plane), signal in response_signals.items():
        pidx = side * 3 + plane
        signal = jnp.asarray(signal)
        truth_dict[pidx] = signal
        if jnp.any(signal != 0):
            active_planes.append(pidx)
            sob_dict[pidx] = make_sobolev_weight(*signal.shape, s=SOBOLEV_S)
    active_planes.sort()
    truth_signals = tuple(truth_dict.get(i, jnp.zeros((1, 1))) for i in range(6))
    spectral_weights = tuple(sob_dict.get(i, jnp.zeros((1, 1))) for i in range(6))

    print("Truth signals ready.\n")

    results = []

    for n_seg in NSEG_VALUES:
        print(f"n_seg={n_seg:,}...", end=' ', flush=True)

        try:
            # Build sim
            sim_opt = DetectorSimulator(
                detector_config, differentiable=True, n_segments=n_seg,
                recombination_model='modified_box')
            fwd_opt = sim_opt.build_forward()
            loss_fn = build_loss_fn(fwd_opt, truth_signals, spectral_weights, active_planes)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            # Random init
            rng = np.random.RandomState(42)
            e_scale = n_truth / n_seg
            replace = n_seg > n_truth
            indices = rng.choice(n_truth, size=n_seg, replace=replace)
            init_pos = truth_pos[indices].copy() + rng.normal(0, 100.0, size=(n_seg, 3))
            init_de = truth_de[indices].copy() * e_scale * rng.uniform(0.2, 1.8, size=n_seg)
            init_de = np.maximum(init_de, 0.001)
            params = jnp.array(np.column_stack([init_pos, init_de]))

            # JIT compile
            t_compile = time.time()
            (loss, _), grads = grad_fn(params)
            jax.block_until_ready(grads)
            compile_time = time.time() - t_compile

            # Optimizer
            optimizer = optax.adam(0.5)
            opt_state = optimizer.init(params)

            # Warmup steps (ignore timing)
            for _ in range(WARMUP_STEPS):
                (loss, _), grads = grad_fn(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                jax.block_until_ready(params)

            # Benchmark steps
            step_times = []
            for _ in range(BENCH_STEPS):
                t0 = time.time()
                (loss, _), grads = grad_fn(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                jax.block_until_ready(params)
                step_times.append(time.time() - t0)

            step_times = np.array(step_times)
            mean_ms = step_times.mean() * 1000
            std_ms = step_times.std() * 1000

            print(f"compile={compile_time:.1f}s, step={mean_ms:.1f} ± {std_ms:.1f} ms")

            results.append({
                'n_seg': n_seg,
                'mean_ms': mean_ms,
                'std_ms': std_ms,
                'compile_s': compile_time,
                'step_times': step_times,
            })

        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'n_seg': n_seg,
                'mean_ms': float('nan'),
                'std_ms': float('nan'),
                'compile_s': float('nan'),
            })

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'n_seg':>10} {'Step (ms)':>12} {'± (ms)':>10} {'Compile (s)':>12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['n_seg']:>10,} {r['mean_ms']:>12.1f} {r['std_ms']:>10.1f} "
              f"{r['compile_s']:>12.1f}")

    # Plot
    valid = [r for r in results if not np.isnan(r['mean_ms'])]
    n_segs = [r['n_seg'] for r in valid]
    means = [r['mean_ms'] for r in valid]
    stds = [r['std_ms'] for r in valid]
    compiles = [r['compile_s'] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: step time vs n_seg
    ax1.errorbar(n_segs, means, yerr=stds, fmt='o-', color='#2196F3',
                 capsize=4, capthick=1.5, markersize=7, linewidth=2,
                 label='Mean ± 1σ')
    ax1.set_xlabel('Number of Segments', fontsize=13)
    ax1.set_ylabel('Time per Step (ms)', fontsize=13)
    ax1.set_title('Optimization Step Time vs Segment Count', fontsize=14, fontweight='bold')
    ax1.tick_params(labelsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Add throughput annotation
    for i, r in enumerate(valid):
        throughput = r['n_seg'] / r['mean_ms'] * 1000  # segs/sec
        if i % 2 == 0 or i == len(valid) - 1:
            ax1.annotate(f'{throughput/1e6:.1f}M seg/s',
                        (r['n_seg'], r['mean_ms']),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=8, ha='center', color='gray')

    # Right: compile time vs n_seg
    ax2.plot(n_segs, compiles, 's-', color='#FF5722', markersize=7, linewidth=2)
    ax2.set_xlabel('Number of Segments', fontsize=13)
    ax2.set_ylabel('JIT Compile Time (s)', fontsize=13)
    ax2.set_title('JIT Compilation Time vs Segment Count', fontsize=14, fontweight='bold')
    ax2.tick_params(labelsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'bench_step_time.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {fname}")

    # Save data
    save_path = os.path.join(OUT_DIR, 'bench_step_time.npz')
    np.savez(save_path,
             n_segs=np.array(n_segs),
             means_ms=np.array(means),
             stds_ms=np.array(stds),
             compiles_s=np.array(compiles))
    print(f"Saved {save_path}")


if __name__ == '__main__':
    main()
