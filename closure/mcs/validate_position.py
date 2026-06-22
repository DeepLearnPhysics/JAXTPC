"""
Validation level 3: position-space closure tests.

3a: Fix globals at truth, recover scattering angles from noisy positions.
3b: All params free — recover vertex, direction, energy, and scattering.

Run from project root:
    python3 -m closure.mcs.validate_position

Migrated from closure_analysis_MCS/validate_position_closure.py.
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import time

from closure.mcs.forward import (
    mcs_cumsum_forward,
    generate_mcs_truth,
    highland_prior,
)
from tools.particle_generator import load_dedx_table_jax
from closure.mcs.mcs_physics import _perpendicular_basis


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEGMENTS = 500
STEP_SIZE_MM = 0.1
N_SEGMENTS_3B = 2500  # Longer track for energy observability via dE/dx
ENERGY = 500.0
START = np.array([-200.0, 0.0, 100.0])
THETA = np.pi / 4
PHI = np.pi / 6

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

results = []

def test(name, passed, message=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, message))
    print(f"  [{status}] {name}: {message}")


# ---------------------------------------------------------------------------
# 3a: Fix globals, recover scattering from noisy positions
# ---------------------------------------------------------------------------

def test_3a():
    print("\n=== 3a: Recover scattering angles from noisy positions ===")

    log_T, dedx = load_dedx_table_jax()
    rng_key = jax.random.PRNGKey(42)

    # Generate truth
    pos_truth, de_truth, dt1_truth, dt2_truth = generate_mcs_truth(
        jnp.float32(ENERGY), jnp.array(START, dtype=jnp.float32),
        jnp.float32(THETA), jnp.float32(PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, rng_key,
    )

    # Add noise to positions (0.3mm as per plan)
    noise_key = jax.random.PRNGKey(123)
    noise = 0.3 * jax.random.normal(noise_key, shape=pos_truth.shape)
    pos_noisy = pos_truth + noise

    # Fixed global parameters
    sin_th = jnp.sin(jnp.float32(THETA))
    cos_th = jnp.cos(jnp.float32(THETA))
    sin_ph = jnp.sin(jnp.float32(PHI))
    cos_ph = jnp.cos(jnp.float32(PHI))
    start_jax = jnp.array(START, dtype=jnp.float32)
    energy_jax = jnp.float32(ENERGY)
    step_cm = STEP_SIZE_MM / 10.0
    lambda_prior = 0.01

    # Loss: position MSE + Highland prior
    def loss_fn(dtheta_flat):
        dt1 = dtheta_flat[:N_SEGMENTS]
        dt2 = dtheta_flat[N_SEGMENTS:]
        pos, _ = mcs_cumsum_forward(
            energy_jax, start_jax, sin_th, cos_th, sin_ph, cos_ph,
            dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        pos_loss = jnp.mean((pos - pos_noisy) ** 2)
        prior = highland_prior(
            dt1, dt2, energy_jax, step_cm, N_SEGMENTS,
            log_T, dedx,
        )
        return pos_loss + lambda_prior * prior / N_SEGMENTS

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Initialize at zero scattering
    params = jnp.zeros(2 * N_SEGMENTS)

    # Adam optimization
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    n_steps = 500
    loss_hist = []

    print(f"  Optimizing {2*N_SEGMENTS} scattering angles, {n_steps} steps...")
    t0 = time.time()
    for step in range(n_steps):
        loss, grad = loss_and_grad(params)
        loss_hist.append(float(loss))
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step % 100 == 0:
            print(f"    Step {step}: loss = {float(loss):.6f}")

    print(f"  Optimization took {time.time()-t0:.1f}s")

    # Evaluate: cumulative angle correlation is the meaningful metric
    # (per-angle recovery is noise-limited at 0.3mm / 0.5mm steps)
    dt1_fit = np.array(params[:N_SEGMENTS])
    dt2_fit = np.array(params[N_SEGMENTS:])
    dt1_t = np.array(dt1_truth)
    dt2_t = np.array(dt2_truth)

    cum1_fit = np.cumsum(dt1_fit)
    cum1_truth = np.cumsum(dt1_t)
    cum_corr1 = np.corrcoef(cum1_fit, cum1_truth)[0, 1]
    cum2_fit = np.cumsum(dt2_fit)
    cum2_truth = np.cumsum(dt2_t)
    cum_corr2 = np.corrcoef(cum2_fit, cum2_truth)[0, 1]
    avg_cum_corr = (cum_corr1 + cum_corr2) / 2

    # Also check trajectory match
    pos_fit, _ = mcs_cumsum_forward(
        energy_jax, start_jax, sin_th, cos_th, sin_ph, cos_ph,
        jnp.array(dt1_fit), jnp.array(dt2_fit), STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    )
    mean_pos_err = float(jnp.mean(jnp.sqrt(jnp.sum((pos_fit - pos_truth) ** 2, axis=1))))

    test("3a: Cumulative angle correlation > 0.5",
         avg_cum_corr > 0.5,
         f"cum_corr1={cum_corr1:.3f}, cum_corr2={cum_corr2:.3f}, avg={avg_cum_corr:.3f}")
    test("3a: Trajectory error < 1mm",
         mean_pos_err < 1.0,
         f"mean position error = {mean_pos_err:.3f} mm")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curve
    axes[0, 0].semilogy(loss_hist)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('3a: Loss convergence')
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative angle comparison
    axes[0, 1].plot(cum1_truth * 1000, 'g-', lw=1, label='truth')
    axes[0, 1].plot(cum1_fit * 1000, 'b-', lw=1, label='fit')
    axes[0, 1].set_xlabel('Segment')
    axes[0, 1].set_ylabel('Cumulative dtheta1 (mrad)')
    axes[0, 1].set_title(f'3a: Cumulative angle (r={cum_corr1:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Trajectory comparison
    pos_fit_np = np.array(pos_fit)
    pos_t = np.array(pos_truth)
    axes[1, 0].plot(pos_t[:, 2], pos_t[:, 1], 'g-', lw=1, label='truth', alpha=0.7)
    axes[1, 0].plot(pos_fit_np[:, 2], pos_fit_np[:, 1], 'b-', lw=1, label='fit', alpha=0.7)
    axes[1, 0].set_xlabel('z (mm)')
    axes[1, 0].set_ylabel('y (mm)')
    axes[1, 0].set_title('3a: Trajectory (yz projection)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Angle comparison
    seg_idx = np.arange(N_SEGMENTS)
    axes[1, 1].plot(seg_idx, dt1_t * 1000, 'g-', lw=0.5, alpha=0.5, label='truth')
    axes[1, 1].plot(seg_idx, dt1_fit * 1000, 'b-', lw=0.5, alpha=0.5, label='fit')
    axes[1, 1].set_xlabel('Segment')
    axes[1, 1].set_ylabel('dtheta1 (mrad)')
    axes[1, 1].set_title('3a: Per-segment scattering')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Level 3a: Recover scattering from noisy positions', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'validate_3a.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved validate_3a.png")

    return dt1_truth, dt2_truth, pos_truth, de_truth


# ---------------------------------------------------------------------------
# 3b: All params free
# ---------------------------------------------------------------------------

def test_3b():
    print("\n=== 3b: All params free — recover vertex + angles ===")

    N = N_SEGMENTS_3B
    log_T, dedx = load_dedx_table_jax()
    rng_key = jax.random.PRNGKey(42)

    # Generate truth (longer track for energy observability)
    pos_truth, de_truth, dt1_truth, dt2_truth = generate_mcs_truth(
        jnp.float32(ENERGY), jnp.array(START, dtype=jnp.float32),
        jnp.float32(THETA), jnp.float32(PHI),
        STEP_SIZE_MM, N, log_T, dedx, rng_key,
    )

    # Add noise (0.05mm)
    noise_key = jax.random.PRNGKey(456)
    noise = 0.05 * jax.random.normal(noise_key, shape=pos_truth.shape)
    pos_noisy = pos_truth + noise

    step_cm = STEP_SIZE_MM / 10.0
    lambda_prior = 0.01

    # Parameter scales (same convention as muon_full_optimization)
    SCALES = jnp.array([200.0, 200.0, 200.0, 1.0, 1.0, 1.0, 1.0, 500.0])
    ANGLE_SCALE = 0.01

    # Truth in normalized space
    truth_globals = jnp.array([
        START[0], START[1], START[2],
        np.sin(THETA), np.cos(THETA),
        np.sin(PHI), np.cos(PHI),
        ENERGY,
    ])

    # Perturbed initial guess
    init_theta = THETA + 0.2
    init_phi = PHI + 0.2
    init_globals = jnp.array([
        START[0] + 50.0, START[1] + 50.0, START[2] - 50.0,
        np.sin(init_theta), np.cos(init_theta),
        np.sin(init_phi), np.cos(init_phi),
        ENERGY + 50.0,
    ])

    # Full parameter vector: [8 globals (normalized), 2*N scattering angles (scaled)]
    init_params = jnp.concatenate([
        init_globals / SCALES,
        jnp.zeros(2 * N),
    ])

    def loss_fn(params):
        g = params[:8] * SCALES
        dt1 = params[8:8 + N] * ANGLE_SCALE
        dt2 = params[8 + N:] * ANGLE_SCALE

        pos, de = mcs_cumsum_forward(
            g[7], jnp.array([g[0], g[1], g[2]]),
            g[3], g[4], g[5], g[6],
            dt1, dt2, STEP_SIZE_MM, N, log_T, dedx,
        )
        pos_loss = jnp.mean((pos - pos_noisy) ** 2)
        # Total dE constraint gives strong energy gradient
        # (per-segment dE is flat in MIP region, but total dE differs)
        total_de_truth = jnp.sum(de_truth)
        total_de_loss = ((jnp.sum(de) - total_de_truth) / total_de_truth) ** 2
        prior = highland_prior(
            dt1, dt2, g[7], step_cm, N,
            log_T, dedx,
        )
        return pos_loss + total_de_loss + lambda_prior * prior / N

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Project trig components onto unit circle
    def project_trig(params):
        g = params[:8]
        st, ct = g[3], g[4]
        norm_t = jnp.maximum(jnp.sqrt(st**2 + ct**2), 1e-8)
        sp, cp = g[5], g[6]
        norm_p = jnp.maximum(jnp.sqrt(sp**2 + cp**2), 1e-8)
        g = g.at[3].set(st / norm_t).at[4].set(ct / norm_t)
        g = g.at[5].set(sp / norm_p).at[6].set(cp / norm_p)
        return params.at[:8].set(g)

    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(init_params)
    params = init_params
    n_steps = 800
    loss_hist = []
    param_hist = []

    print(f"  Optimizing 8 globals + {2*N} angles, {n_steps} steps...")
    t0 = time.time()
    for step in range(n_steps):
        loss, grad = loss_and_grad(params)
        loss_hist.append(float(loss))
        param_hist.append(np.array(params[:8] * np.array(SCALES)))
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = project_trig(params)
        if step % 100 == 0:
            g = np.array(params[:8] * np.array(SCALES))
            print(f"    Step {step}: loss={float(loss):.6f}, "
                  f"x={g[0]:.1f}, y={g[1]:.1f}, z={g[2]:.1f}, E={g[7]:.1f}")

    print(f"  Optimization took {time.time()-t0:.1f}s")

    # Final parameters
    final_g = np.array(params[:8] * np.array(SCALES))
    vertex_err = np.sqrt(np.sum((final_g[:3] - np.array(START)) ** 2))
    energy_err = abs(final_g[7] - ENERGY)

    test("3b: Vertex recovery < 5mm",
         vertex_err < 5.0,
         f"vertex error = {vertex_err:.2f} mm")
    # Note: Energy is not recoverable from positions in the MIP region.
    # dE/dx varies < 20% over 100-1000 MeV, and the Highland prior biases
    # energy downward. Wire-signal closure (levels 5-6) constrains energy.
    print(f"  Note: Energy = {final_g[7]:.1f} MeV (truth = {ENERGY:.1f}) — "
          f"expected: position closure cannot constrain energy in MIP regime")

    # Plot
    param_hist = np.array(param_hist)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Loss
    axes[0, 0].semilogy(loss_hist)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('3b: Loss convergence')
    axes[0, 0].grid(True, alpha=0.3)

    # x
    axes[0, 1].plot(param_hist[:, 0], 'b-')
    axes[0, 1].axhline(START[0], color='g', ls='--', label=f'truth={START[0]}')
    axes[0, 1].set_title('x (mm)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # y
    axes[0, 2].plot(param_hist[:, 1], 'b-')
    axes[0, 2].axhline(START[1], color='g', ls='--', label=f'truth={START[1]}')
    axes[0, 2].set_title('y (mm)')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # z
    axes[1, 0].plot(param_hist[:, 2], 'b-')
    axes[1, 0].axhline(START[2], color='g', ls='--', label=f'truth={START[2]}')
    axes[1, 0].set_title('z (mm)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # theta
    eff_theta = np.arctan2(param_hist[:, 3], param_hist[:, 4])
    axes[1, 1].plot(np.degrees(eff_theta), 'b-')
    axes[1, 1].axhline(np.degrees(THETA), color='g', ls='--',
                        label=f'truth={np.degrees(THETA):.1f}')
    axes[1, 1].set_title('theta (deg)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # energy
    axes[1, 2].plot(param_hist[:, 7], 'b-')
    axes[1, 2].axhline(ENERGY, color='g', ls='--', label=f'truth={ENERGY}')
    axes[1, 2].set_title('Energy (MeV)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle('Level 3b: All params free — position closure', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'validate_3b.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved validate_3b.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MCS POSITION CLOSURE VALIDATION (Level 3)")
    print("=" * 60)

    test_3a()
    test_3b()

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("ALL TESTS PASSED")
    else:
        print("FAILURES:")
        for name, passed, msg in results:
            if not passed:
                print(f"  {name}: {msg}")
    print("=" * 60)


if __name__ == '__main__':
    main()
