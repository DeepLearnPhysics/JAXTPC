"""
Focused sweep: lr_e_mult then noise, at LR=0.7 / exp d=0.9995.
Run:  python3 closure_analysis/sweep_5seg_v2.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit

from closure_analysis.sgld_closure import (
    TRUTH_BANK, INIT_OFFSET, PLANES, K, N_PROJ,
    build_loss_fn, best_permutation,
)

N_SEG = 5
N_TRUTH = 5
MIN_ENERGY = 0.001
STEPS = 1000


def run_one(grad_fn, lr=0.7, decay_rate=0.9995, lr_e_mult=0.01,
            noise_lr=0.0, noise_coupling='quadratic', b1=0.95):
    schedule = optax.exponential_decay(init_value=lr, transition_steps=1, decay_rate=decay_rate)
    optimizer = optax.adam(schedule, b1=b1, b2=0.999)

    truth_params = TRUTH_BANK[:N_TRUTH]
    params = jnp.array(truth_params + INIT_OFFSET)
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    losses = []
    for step in range(STEPS):
        (loss, _), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(lr_e_mult)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], MIN_ENERGY))

        if noise_lr > 0:
            lr_cur = float(schedule(step))
            if noise_coupling == 'quadratic':
                ns = (lr_cur ** 2 / lr) * noise_lr
            else:
                ns = lr_cur * noise_lr
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(ns * jax.random.normal(nk, shape=(N_SEG, 3)))

        losses.append(float(loss))

    final = np.array(params)
    _, errors, midx = best_permutation(final, truth_params)
    me = errors[midx]
    mean_pos = np.mean(np.sqrt(np.sum(me[:, :3]**2, axis=1)))
    max_pos = np.max(np.sqrt(np.sum(me[:, :3]**2, axis=1)))
    max_de = np.max(np.abs(me[:, 3])) * 1000
    return {'loss': losses[-1], 'mean_pos': mean_pos, 'max_pos': max_pos, 'max_de': max_de}


def main():
    # Setup
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEG)
    fwd = sim.build_forward()

    truth_params = TRUTH_BANK[:N_TRUTH]
    truth_seg = SegmentData(positions_mm=jnp.array(truth_params[:, :3]), de=jnp.array(truth_params[:, 3]))
    target_signals = fwd(truth_seg)
    key = jax.random.PRNGKey(42)
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    loss_fn = build_loss_fn(fwd, target_clouds, key)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    print("Warming up JIT...")
    _ = grad_fn(jnp.array(truth_params + INIT_OFFSET))
    print("Ready.\n")

    # =============================================
    # Round 1: lr_e_mult sweep
    # =============================================
    print("=" * 60)
    print("ROUND 1: lr_e_mult sweep (LR=0.7, d=0.9995, no noise)")
    print("=" * 60)

    e_mults = [0.003, 0.01, 0.03]
    best_em = None
    best_em_pos = 999

    for em in e_mults:
        t0 = time.time()
        r = run_one(grad_fn, lr_e_mult=em)
        dt = time.time() - t0
        tag = " <-- BEST" if r['mean_pos'] < best_em_pos else ""
        print(f"  lr_e_mult={em:<6}  pos={r['mean_pos']:.3f}mm  max={r['max_pos']:.3f}mm  "
              f"dE={r['max_de']:.1f}keV  loss={r['loss']:.6f}  {dt:.0f}s{tag}")
        if r['mean_pos'] < best_em_pos:
            best_em_pos = r['mean_pos']
            best_em = em

    print(f"\n  Winner: lr_e_mult={best_em}\n")

    # =============================================
    # Round 2: noise sweep at best lr_e_mult
    # =============================================
    print("=" * 60)
    print(f"ROUND 2: noise sweep (LR=0.7, d=0.9995, lr_e_mult={best_em})")
    print("=" * 60)

    noise_configs = [
        ('no_noise',  0.0, 'quadratic'),
        ('n1_quad',   1.0, 'quadratic'),
        ('n2_quad',   2.0, 'quadratic'),
        ('n5_quad',   5.0, 'quadratic'),
        ('n2_lin',    2.0, 'linear'),
        ('n5_lin',    5.0, 'linear'),
    ]

    best_noise = None
    best_noise_pos = 999

    for label, nlr, nc in noise_configs:
        t0 = time.time()
        r = run_one(grad_fn, lr_e_mult=best_em, noise_lr=nlr, noise_coupling=nc)
        dt = time.time() - t0
        tag = " <-- BEST" if r['mean_pos'] < best_noise_pos else ""
        print(f"  {label:<10}  pos={r['mean_pos']:.3f}mm  max={r['max_pos']:.3f}mm  "
              f"dE={r['max_de']:.1f}keV  loss={r['loss']:.6f}  {dt:.0f}s{tag}")
        if r['mean_pos'] < best_noise_pos:
            best_noise_pos = r['mean_pos']
            best_noise = (nlr, nc)

    print(f"\n  Winner: noise_lr={best_noise[0]}, coupling={best_noise[1]}")
    print(f"\n{'=' * 60}")
    print(f"BEST CONFIG: LR=0.7, d=0.9995, lr_e_mult={best_em}, "
          f"noise_lr={best_noise[0]}, coupling={best_noise[1]}")
    print(f"  -> {best_noise_pos:.3f}mm in {STEPS} steps")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
