"""
Overcomplete test: does energy learning rate control the separation?

Tests LR_ENERGY_MULT = 0.1 vs 0.5 vs 1.0, no L1, 1200 steps.
3 optimizer segments, 2 truths.

Run: python3 closure_analysis/test_l1_drain.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit
from closure_analysis.optimization_closure import TRUTH_BANK, INIT_OFFSET

K = 10000
N_PROJ = 200
PLANES = [0, 1, 2]


def main():
    n_truth = 2
    n_seg = 3
    truth_params = TRUTH_BANK[:n_truth]

    print("Building simulator...")
    config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(config, differentiable=True, n_segments=n_seg)
    forward = sim.build_forward()

    truth_padded = np.zeros((n_seg, 4))
    truth_padded[:n_truth] = truth_params
    truth_seg = SegmentData(
        positions_mm=jnp.array(truth_padded[:, :3]),
        de=jnp.array(truth_padded[:, 3]),
    )
    target_signals = forward(truth_seg)
    key = jax.random.PRNGKey(42)
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    def sw_loss(params):
        seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
        sigs = forward(seg)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            loss = loss + sliced_wasserstein_loss_jit(
                pts, w, target_clouds[p][0], target_clouds[p][1],
                key, n_projections=N_PROJ)
        return loss

    vg_fn = jax.value_and_grad(sw_loss)

    print("Warming up JIT...")
    init_params = jnp.array([
        truth_params[0] + INIT_OFFSET,
        truth_params[1] + INIT_OFFSET,
        [-200.0, 0.0, 300.0, 1.5],
    ])
    _ = vg_fn(init_params)
    print("Done.")
    print(f"\nTruth energies: [{truth_params[0,3]}, {truth_params[1,3]}]")

    n_steps = 1200
    for lr_e_mult in [0.1, 0.5, 1.0]:
        schedule = optax.cosine_decay_schedule(
            init_value=0.3, decay_steps=n_steps, alpha=0.01)
        optimizer = optax.adam(schedule, b1=0.95)
        params = init_params.copy()
        opt_state = optimizer.init(params)

        print(f"\n--- LR_ENERGY_MULT = {lr_e_mult} ({n_steps} steps, no L1) ---")
        print(f"  {'Step':>5s}  {'Loss':>10s}  {'E0':>7s}  {'E1':>7s}  {'E2(X)':>7s}  {'E0-E2':>7s}")

        t0 = time.time()
        print_every = max(1, n_steps // 15)
        for step in range(n_steps):
            loss, grads = vg_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            updates = updates.at[:, 3].multiply(lr_e_mult)
            params = optax.apply_updates(params, updates)

            if step % print_every == 0 or step == n_steps - 1:
                e = [float(params[i, 3]) for i in range(n_seg)]
                sep = e[0] - e[2]
                print(f"  {step:>5d}  {float(loss):>10.4f}  {e[0]:>7.3f}  {e[1]:>7.3f}  {e[2]:>7.3f}  {sep:>+7.3f}")

        final = np.array(params)
        print(f"  ({time.time()-t0:.1f}s)")
        print(f"  Final pos seg0: [{final[0,0]:+.1f}, {final[0,1]:+.1f}, {final[0,2]:+.1f}]  "
              f"(truth: {truth_params[0,:3]})")
        print(f"  Final pos seg2: [{final[2,0]:+.1f}, {final[2,1]:+.1f}, {final[2,2]:+.1f}]  "
              f"(EXTRA)")


if __name__ == '__main__':
    main()
