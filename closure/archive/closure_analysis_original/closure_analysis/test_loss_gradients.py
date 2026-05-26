"""
Gradient surface test: compare 4 loss variants on the same TPC closure problem.

Builds one simulation, extracts pointclouds, then evaluates loss + gradients
for each loss variant at the init position. Reports gradient magnitudes
per segment per dimension (x, y, z, dE).

Loss variants:
  1. OTT SW   — random projections, W2 cost (production baseline)
  2. Uniform W1 — fixed directions, |quant_a - quant_b|
  3. Uniform W2 — fixed directions, (quant_a - quant_b)^2
  4. Uniform Huber — fixed directions, Huber(quant_a - quant_b)

Run: python3 closure_analysis/test_loss_gradients.py
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
from ott_test.ot_losses import sliced_wasserstein_loss_jit
from closure_analysis.optimization_closure import TRUTH_BANK, INIT_OFFSET

K = 10000
N_PROJ = 200
PLANES = [0, 1, 2]


# =============================================================================
# Uniform SW factories (W1, W2, Huber)
# =============================================================================

def make_uniform_sw_w1(n_proj, n_grid=500):
    """Uniform sliced Wasserstein with L1 (absolute) cost."""
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
            return jnp.mean(jnp.abs(quant_a - quant_b))

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    return sw


def make_uniform_sw_w2(n_proj, n_grid=500):
    """Uniform sliced Wasserstein with L2 (squared) cost."""
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
            return jnp.mean((quant_a - quant_b) ** 2)

        costs = jax.vmap(w1d)(proj_a.T, proj_b.T)
        return jnp.mean(costs)

    return sw


def make_uniform_sw_huber(n_proj, delta=0.01, n_grid=500):
    """Uniform sliced Wasserstein with Huber cost.

    Huber(x, delta) = 0.5*x^2           if |x| <= delta
                      delta*(|x| - 0.5*delta)  otherwise

    Interpolates between W2 (near zero) and W1 (far from zero).
    delta is in the same units as the normalized point coordinates.
    """
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


# =============================================================================
# Main test
# =============================================================================

def main():
    n_seg = 5
    truth_params = TRUTH_BANK[:n_seg]

    print("Building simulator...")
    config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(config, differentiable=True, n_segments=n_seg)
    forward = sim.build_forward()

    # Generate target
    truth_seg = SegmentData(
        positions_mm=jnp.array(truth_params[:, :3]),
        de=jnp.array(truth_params[:, 3]),
    )
    target_signals = forward(truth_seg)
    key = jax.random.PRNGKey(42)
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    # Init params at offset from truth
    init_params = jnp.array(truth_params + INIT_OFFSET)

    # Build loss functions — all take params (n_seg, 4) and return scalar
    sw_w1 = make_uniform_sw_w1(N_PROJ)
    sw_w2 = make_uniform_sw_w2(N_PROJ)
    sw_huber = make_uniform_sw_huber(N_PROJ, delta=0.01)

    def make_closure_loss(plane_loss_fn, needs_key=False):
        """Wrap a per-plane loss into a full closure loss over 3 planes."""
        def loss_fn(params):
            seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
            sigs = forward(seg)
            loss = 0.0
            for p in PLANES:
                pts, w = signal_to_pointcloud(sigs[p], K)
                tp, tw = target_clouds[p]
                if needs_key:
                    loss = loss + plane_loss_fn(pts, w, tp, tw, key,
                                                n_projections=N_PROJ)
                else:
                    loss = loss + plane_loss_fn(pts, w, tp, tw)
            return loss
        return loss_fn

    losses = {
        'OTT SW (W2, random)': make_closure_loss(sliced_wasserstein_loss_jit,
                                                   needs_key=True),
        'Uniform W1 (abs)':    make_closure_loss(sw_w1),
        'Uniform W2 (sq)':     make_closure_loss(sw_w2),
        'Uniform Huber':       make_closure_loss(sw_huber),
    }

    # JIT warmup all grad functions
    print("Warming up JIT for all 4 loss variants...")
    grad_fns = {}
    for name, loss_fn in losses.items():
        vg = jax.value_and_grad(loss_fn)
        t0 = time.time()
        _ = vg(init_params)
        jax.block_until_ready(_[1])
        print(f"  {name}: JIT compiled in {time.time()-t0:.1f}s")
        grad_fns[name] = vg

    # Evaluate at init
    print(f"\n{'='*80}")
    print(f"GRADIENT SURFACE TEST — {n_seg} segments at init offset")
    print(f"{'='*80}")
    print(f"K={K}, N_PROJ={N_PROJ}, INIT_OFFSET={INIT_OFFSET}")
    print(f"\nTruth params:")
    for i in range(n_seg):
        print(f"  Seg {i}: x={truth_params[i,0]:+.0f}, y={truth_params[i,1]:+.0f}, "
              f"z={truth_params[i,2]:+.0f}, dE={truth_params[i,3]:.1f} MeV")

    for name, vg in grad_fns.items():
        loss, grads = vg(init_params)
        jax.block_until_ready(grads)
        grads_np = np.array(grads)

        print(f"\n--- {name} ---")
        print(f"  Loss = {float(loss):.6e}")
        print(f"  {'Seg':>5s}  {'dL/dx':>12s}  {'dL/dy':>12s}  {'dL/dz':>12s}  "
              f"{'dL/dE':>12s}  {'||grad_pos||':>12s}")
        for i in range(n_seg):
            gx, gy, gz, ge = grads_np[i]
            pos_norm = np.sqrt(gx**2 + gy**2 + gz**2)
            print(f"  {i:>5d}  {gx:>+12.4e}  {gy:>+12.4e}  {gz:>+12.4e}  "
                  f"{ge:>+12.4e}  {pos_norm:>12.4e}")

        # Summary stats
        pos_grads = grads_np[:, :3]
        e_grads = grads_np[:, 3]
        print(f"  Position grad: mean |g|={np.mean(np.abs(pos_grads)):.4e}, "
              f"max={np.max(np.abs(pos_grads)):.4e}, "
              f"min={np.min(np.abs(pos_grads)):.4e}")
        print(f"  Energy grad:   mean |g|={np.mean(np.abs(e_grads)):.4e}, "
              f"max={np.max(np.abs(e_grads)):.4e}, "
              f"min={np.min(np.abs(e_grads)):.4e}")
        print(f"  Ratio |dL/dE| / |dL/dpos|: "
              f"{np.mean(np.abs(e_grads)) / (np.mean(np.abs(pos_grads)) + 1e-10):.2f}")

    # Also test at near-converged state (truth positions, init energy)
    print(f"\n{'='*80}")
    print(f"GRADIENT AT NEAR-CONVERGED STATE (truth positions, init energy)")
    print(f"{'='*80}")

    near_params = jnp.array(truth_params.copy())
    # Keep truth positions but add energy offset
    near_params = near_params.at[:, 3].add(INIT_OFFSET[3])

    for name, vg in grad_fns.items():
        loss, grads = vg(near_params)
        jax.block_until_ready(grads)
        grads_np = np.array(grads)

        print(f"\n--- {name} ---")
        print(f"  Loss = {float(loss):.6e}")
        print(f"  {'Seg':>5s}  {'dL/dx':>12s}  {'dL/dy':>12s}  {'dL/dz':>12s}  "
              f"{'dL/dE':>12s}  {'||grad_pos||':>12s}")
        for i in range(n_seg):
            gx, gy, gz, ge = grads_np[i]
            pos_norm = np.sqrt(gx**2 + gy**2 + gz**2)
            print(f"  {i:>5d}  {gx:>+12.4e}  {gy:>+12.4e}  {gz:>+12.4e}  "
                  f"{ge:>+12.4e}  {pos_norm:>12.4e}")

        e_grads = grads_np[:, 3]
        print(f"  Energy grad: mean |g|={np.mean(np.abs(e_grads)):.4e}, "
              f"max={np.max(np.abs(e_grads)):.4e}")


if __name__ == '__main__':
    main()
