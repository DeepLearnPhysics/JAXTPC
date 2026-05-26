"""
Diagnostic script for MCMC closure optimizer.

Checks magnitudes of signals, gradients, L1 drain, noise, and top-K
membership at the 0.5mm segment energy scale (0.05-0.20 MeV) with K=10000.

Run from project root:
    python3 closure_analysis/diagnostic_mcmc.py
"""

import sys, os
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
    LR_POSITION, LR_ENERGY_MULT, LR_L1, MIN_ENERGY,
    B1, B2, NOISE_LR, TOTAL_STEPS,
    build_loss_fn,
)


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    N_SEG = 5
    config = generate_detector('config/cubic_wireplane_config.yaml')

    truth_params = TRUTH_BANK[:N_SEG]
    init_params = jnp.array(truth_params + INIT_OFFSET)

    print("Truth energies (MeV):", truth_params[:, 3])
    print("Init  energies (MeV):", np.array(init_params[:, 3]))

    # =========================================================================
    section("1. SIGNAL MAGNITUDE — single segment at 0.10 MeV")
    # =========================================================================

    sim1 = DetectorSimulator(config, differentiable=True, n_segments=1)
    fwd1 = sim1.build_forward()

    seg1 = SegmentData(
        positions_mm=jnp.array([[-100.0, 50.0, 100.0]]),
        de=jnp.array([0.10]),
    )
    sigs1 = fwd1(seg1)
    jax.block_until_ready(sigs1)

    for p in PLANES:
        s = sigs1[p]
        abs_s = jnp.abs(s)
        nonzero = int((abs_s > 1e-10).sum())
        print(f"  Plane {p}: shape={s.shape}, max={float(abs_s.max()):.6f}, "
              f"mean(nonzero)={float(abs_s.sum()) / max(nonzero, 1):.6f}, "
              f"nonzero_bins={nonzero}, total_bins={s.size}")

    # =========================================================================
    section("2. TOP-K MEMBERSHIP — 5 segments combined vs individual")
    # =========================================================================

    sim5 = DetectorSimulator(config, differentiable=True, n_segments=N_SEG)
    fwd5 = sim5.build_forward()

    truth_seg = SegmentData(
        positions_mm=jnp.array(truth_params[:, :3]),
        de=jnp.array(truth_params[:, 3]),
    )
    combined_sigs = fwd5(truth_seg)
    jax.block_until_ready(combined_sigs)

    # Get top-K indices from combined signal
    for p in PLANES:
        comb = combined_sigs[p]
        flat_comb = jnp.abs(comb.ravel())
        topk_vals, topk_idx = jax.lax.top_k(flat_comb, K)
        topk_set = set(np.array(topk_idx))

        min_topk_val = float(topk_vals[-1])
        print(f"\n  Plane {p}: top-K threshold = {min_topk_val:.8f}")

        # Check each segment's individual signal
        for s in range(N_SEG):
            seg_s = SegmentData(
                positions_mm=jnp.array(truth_params[s:s+1, :3]),
                de=jnp.array(truth_params[s:s+1, 3]),
            )
            sig_s = fwd1(seg_s)
            flat_s = jnp.abs(sig_s[p].ravel())

            # How many of this segment's nonzero bins are in the top-K?
            nonzero_mask = flat_s > 1e-10
            nonzero_idx = np.where(np.array(nonzero_mask))[0]
            in_topk = sum(1 for idx in nonzero_idx if idx in topk_set)

            # What fraction of this segment's signal energy is in top-K?
            seg_vals_in_topk = float(flat_s[np.array(list(topk_set & set(nonzero_idx)))].sum()) \
                if len(topk_set & set(nonzero_idx)) > 0 else 0.0
            seg_total = float(flat_s.sum())
            frac = seg_vals_in_topk / seg_total if seg_total > 0 else 0.0

            print(f"    Seg {s} (dE={truth_params[s, 3]:.2f}): "
                  f"nonzero={len(nonzero_idx)}, in_topK={in_topk}/{len(nonzero_idx)}, "
                  f"signal_frac_in_topK={frac:.3f}")

    # =========================================================================
    section("3. GRADIENT MAGNITUDES at init")
    # =========================================================================

    # Build loss for 5 segments
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(combined_sigs[p], K)
        target_clouds[p] = (pts, w)

    key = jax.random.PRNGKey(42)
    loss_fn = build_loss_fn(fwd5, target_clouds, key)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    print("\n  Warming up JIT...")
    (loss0, _), grads0 = grad_fn(init_params)
    print(f"  Initial loss: {float(loss0):.6f}")

    print(f"\n  {'Seg':>3s}  {'|grad_pos|':>10s}  {'|grad_x|':>10s}  {'|grad_y|':>10s}  "
          f"{'|grad_z|':>10s}  {'grad_dE':>10s}")
    print(f"  {'---':>3s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}  "
          f"{'----------':>10s}  {'----------':>10s}")
    for s in range(N_SEG):
        gp = np.array(grads0[s, :3])
        ge = float(grads0[s, 3])
        print(f"  {s:3d}  {np.linalg.norm(gp):10.4f}  {gp[0]:10.4f}  {gp[1]:10.4f}  "
              f"{gp[2]:10.4f}  {ge:10.4f}")

    # =========================================================================
    section("4. ADAM UPDATE vs DECOUPLED L1")
    # =========================================================================

    schedule = optax.cosine_decay_schedule(
        init_value=LR_POSITION, decay_steps=TOTAL_STEPS, alpha=0.01)
    optimizer = optax.adam(schedule, b1=B1, b2=B2)
    opt_state = optimizer.init(init_params)

    # One Adam step
    updates, _ = optimizer.update(grads0, opt_state, init_params)
    updates_scaled = updates.at[:, 3].multiply(LR_ENERGY_MULT)

    print(f"\n  After 1 Adam step:")
    print(f"  {'Seg':>3s}  {'Adam_pos (mm)':>14s}  {'Adam_dE (MeV)':>14s}  "
          f"{'L1_dE (MeV)':>12s}  {'Ratio (Adam/L1)':>16s}")
    print(f"  {'---':>3s}  {'--------------':>14s}  {'--------------':>14s}  "
          f"{'------------':>12s}  {'----------------':>16s}")
    for s in range(N_SEG):
        adam_pos = float(jnp.linalg.norm(updates_scaled[s, :3]))
        adam_de = float(updates_scaled[s, 3])
        ratio = abs(adam_de) / LR_L1 if LR_L1 > 0 else float('inf')
        print(f"  {s:3d}  {adam_pos:14.6f}  {adam_de:14.6f}  {-LR_L1:12.6f}  {ratio:16.2f}")

    # =========================================================================
    section("5. NOISE MAGNITUDE at different steps")
    # =========================================================================

    for step in [0, 500, 1500, 2500, 2999]:
        lr = float(schedule(step))
        noise_mag = lr * NOISE_LR
        print(f"  Step {step:4d}: lr={lr:.6f}, noise_std={noise_mag:.4f} mm "
              f"(~{noise_mag / 3.0:.2f} wire pitches)")

    # =========================================================================
    section("6. L1 DRAIN TIMELINE")
    # =========================================================================

    print(f"\n  LR_L1 = {LR_L1} MeV/step")
    print(f"  MIN_ENERGY = {MIN_ENERGY} MeV")
    print(f"  DEATH_THRESH = {MIN_ENERGY * 10:.3f} MeV")
    for e0 in [0.05, 0.10, 0.15, 0.20]:
        # Steps to reach death threshold (pure drain, no reconstruction gradient)
        steps_to_death = max(0, (e0 - MIN_ENERGY * 10) / LR_L1)
        print(f"  dE={e0:.2f} MeV -> death in {steps_to_death:.0f} pure-drain steps "
              f"({steps_to_death / 100:.1f} relocation intervals)")

    # =========================================================================
    section("7. SHORT OPTIMIZATION RUN (50 steps) — all modes")
    # =========================================================================

    for mode_label, apply_l1, apply_noise in [
        ("Adam only", False, False),
        ("Adam + noise", False, True),
        ("Adam + noise + L1", True, True),
    ]:
        print(f"\n  --- {mode_label} ---")
        params = init_params.copy()
        opt_state = optimizer.init(params)
        rng_key = jax.random.PRNGKey(99)
        grad_ema = jnp.zeros(N_SEG)

        for step in range(50):
            (loss, _), grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            updates = updates.at[:, 3].multiply(LR_ENERGY_MULT)
            params = optax.apply_updates(params, updates)

            if apply_l1:
                params = params.at[:, 3].add(-LR_L1)
                params = params.at[:, 3].set(jnp.maximum(params[:, 3], MIN_ENERGY))

            if apply_noise:
                lr_cur = float(schedule(step))
                rng_key, nk = jax.random.split(rng_key)
                nv = jax.random.normal(nk, shape=(N_SEG, 3))
                params = params.at[:, :3].add(lr_cur * NOISE_LR * nv)

            grad_pos_norm = jnp.linalg.norm(grads[:, :3], axis=-1)
            grad_ema = 0.95 * grad_ema + 0.05 * grad_pos_norm

            if step % 10 == 0 or step == 49:
                e_str = ", ".join(f"{float(params[i, 3]):.4f}" for i in range(N_SEG))
                g_str = ", ".join(f"{float(grad_ema[i]):.2f}" for i in range(N_SEG))
                print(f"    Step {step:3d}: loss={float(loss):.6f}  E=[{e_str}]")
                if step % 20 == 0:
                    g_str_sci = ", ".join(f"{float(grad_ema[i]):.2e}" for i in range(N_SEG))
                    print(f"              grad_ema=[{g_str_sci}]")

    print(f"\n{'=' * 70}")
    print("  Diagnostics complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
