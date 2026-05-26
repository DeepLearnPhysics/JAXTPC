"""
Run muon full-parameter optimization and save history to NPZ.

Matches muon_full_optimization.py exactly, just saves param + loss history
instead of plotting, so GIF rendering can be done separately.

Run from project root:
    python3 closure_analysis_muon/run_optimization_save.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean, make_sobolev_weight

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments_trig,
    mask_outside_volume,
    build_muon_forward,
)

# =============================================================================
# Configuration (identical to muon_full_optimization.py)
# =============================================================================

N_SEGMENTS = 4000
STEP_SIZE_MM = 0.5

TRUTH_X = -200.0
TRUTH_Y = 0.0
TRUTH_Z = 100.0
TRUTH_THETA = np.pi / 4
TRUTH_PHI = np.pi / 6
TRUTH_ENERGY = 500.0

TRUTH_SIN_THETA = np.sin(TRUTH_THETA)
TRUTH_COS_THETA = np.cos(TRUTH_THETA)
TRUTH_SIN_PHI = np.sin(TRUTH_PHI)
TRUTH_COS_PHI = np.cos(TRUTH_PHI)

TRUTH_PHYS = np.array([
    TRUTH_X, TRUTH_Y, TRUTH_Z,
    TRUTH_SIN_THETA, TRUTH_COS_THETA,
    TRUTH_SIN_PHI, TRUTH_COS_PHI,
    TRUTH_ENERGY,
])

SCALES = np.array([200.0, 200.0, 200.0, 1.0, 1.0, 1.0, 1.0, 500.0])

THETA_PERTURB = 0.5
PHI_PERTURB = 0.5
INIT_THETA = TRUTH_THETA + THETA_PERTURB
INIT_PHI = TRUTH_PHI + PHI_PERTURB

INIT_PHYS = np.array([
    TRUTH_X + 100.0,
    TRUTH_Y + 100.0,
    TRUTH_Z - 100.0,
    np.sin(INIT_THETA),
    np.cos(INIT_THETA),
    np.sin(INIT_PHI),
    np.cos(INIT_PHI),
    TRUTH_ENERGY + 100.0,
])

TRUTH_NORM = TRUTH_PHYS / SCALES
INIT_NORM = INIT_PHYS / SCALES

N_STEPS = 300
LR = 0.015

SCALES_JAX = jnp.array(SCALES, dtype=jnp.float32)
PARAM_NAMES = ['x', 'y', 'z', 'sin_th', 'cos_th', 'sin_ph', 'cos_ph', 'energy']
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Helpers (identical to muon_full_optimization.py)
# =============================================================================

def to_physical(norm_params):
    return norm_params * SCALES_JAX


def project_unit_circle(norm_params):
    st, ct = norm_params[3], norm_params[4]
    norm_t = jnp.maximum(jnp.sqrt(st**2 + ct**2), 1e-8)
    sp, cp = norm_params[5], norm_params[6]
    norm_p = jnp.maximum(jnp.sqrt(sp**2 + cp**2), 1e-8)
    return norm_params.at[3].set(st / norm_t).at[4].set(ct / norm_t) \
                       .at[5].set(sp / norm_p).at[6].set(cp / norm_p)


def _make_sim_forward(forward, log_T, dedx):
    """Build sim forward closure."""
    def sim_forward(phys):
        pos, de = generate_muon_segments_trig(
            phys[7], jnp.array([phys[0], phys[1], phys[2]]),
            phys[3], phys[4], phys[5], phys[6],
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de)
        return forward(pos, de)
    return sim_forward


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("MUON OPTIMIZATION — SAVE HISTORY")
    print("=" * 70)
    print(f"Truth: x={TRUTH_X}, y={TRUTH_Y}, z={TRUTH_Z}")
    print(f"       theta={np.degrees(TRUTH_THETA):.1f} deg, "
          f"phi={np.degrees(TRUTH_PHI):.1f} deg, E={TRUTH_ENERGY} MeV")
    print(f"Init:  x={INIT_PHYS[0]:.0f}, y={INIT_PHYS[1]:.0f}, z={INIT_PHYS[2]:.0f}")
    print(f"       theta={np.degrees(INIT_THETA):.1f} deg, "
          f"phi={np.degrees(INIT_PHI):.1f} deg, E={INIT_PHYS[7]:.0f} MeV")
    print(f"N_SEGMENTS={N_SEGMENTS}, N_STEPS={N_STEPS}, LR={LR}")
    print()

    # --- Load resources ---
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True,
                            n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    sim_forward = _make_sim_forward(forward, log_T, dedx)

    # --- Generate truth signals ---
    print("Compiling forward...", flush=True)
    t0 = time.time()
    truth_signals = jax.jit(sim_forward)(
        jnp.array(TRUTH_PHYS, dtype=jnp.float32))
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"  Compiled ({time.time()-t0:.1f}s)", flush=True)

    # --- Precompute Sobolev spectral weights (s=1.5) ---
    print("\nPrecomputing Sobolev spectral weights (s=1.5)...", flush=True)
    spec_weights_tuple = tuple(
        make_sobolev_weight(*truth_signals[p].shape, s=1.5) for p in range(6)
    )

    # --- Loss function (takes NORMALIZED params) ---
    def loss_fn(norm_params):
        sigs = sim_forward(to_physical(norm_params))
        return sobolev_loss_geomean(sigs, truth_signals, spec_weights_tuple)

    # --- Compile loss + grad ---
    print("Compiling loss + gradient...", flush=True)
    t0 = time.time()
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    init_n = jnp.array(INIT_NORM, dtype=jnp.float32)
    init_loss, init_grad = loss_and_grad(init_n)
    jax.block_until_ready(init_grad)
    print(f"  Compiled ({time.time()-t0:.1f}s)", flush=True)
    print(f"  Initial loss: {float(init_loss):.6f}")

    # --- Optimizer ---
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(init_n)

    # --- Optimization loop ---
    print(f"\nRunning {N_STEPS} optimization steps (LR={LR})...", flush=True)
    params_n = init_n

    # History: N_STEPS+1 rows (row 0 = initial)
    param_history_phys = np.empty((N_STEPS + 1, 8))
    loss_history = np.empty(N_STEPS + 1)

    param_history_phys[0] = np.array(to_physical(params_n))
    loss_history[0] = float(init_loss)

    t_start = time.time()
    for step in range(N_STEPS):
        loss, grad = loss_and_grad(params_n)

        updates, opt_state = optimizer.update(grad, opt_state, params_n)
        params_n = optax.apply_updates(params_n, updates)
        params_n = project_unit_circle(params_n)

        step_num = step + 1
        p = np.array(to_physical(params_n))
        param_history_phys[step_num] = p
        loss_history[step_num] = float(loss)

        if step_num % 20 == 0 or step_num == N_STEPS:
            eff_theta = float(jnp.arctan2(params_n[3], params_n[4]))
            eff_phi = float(jnp.arctan2(params_n[5], params_n[6]))
            elapsed = time.time() - t_start
            print(f"  Step {step_num:3d}: loss={float(loss):.6f}, "
                  f"x={p[0]:7.1f}, y={p[1]:7.1f}, z={p[2]:7.1f}, "
                  f"th={np.degrees(eff_theta):5.1f} deg, "
                  f"ph={np.degrees(eff_phi):5.1f} deg, "
                  f"E={p[7]:6.1f} MeV  ({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t_start
    print(f"\nOptimization complete in {total_time:.1f}s "
          f"({total_time/N_STEPS:.2f}s/step)")

    # --- Final results ---
    final_phys = param_history_phys[-1]
    eff_theta_final = np.arctan2(final_phys[3], final_phys[4])
    eff_phi_final = np.arctan2(final_phys[5], final_phys[6])

    print(f"\n{'Parameter':<12} {'Truth':>10} {'Initial':>10} "
          f"{'Final':>10} {'Delta':>10}")
    print("-" * 55)
    print(f"{'x (mm)':<12} {TRUTH_X:>10.1f} {INIT_PHYS[0]:>10.1f} "
          f"{final_phys[0]:>10.1f} {final_phys[0]-TRUTH_X:>10.2f}")
    print(f"{'y (mm)':<12} {TRUTH_Y:>10.1f} {INIT_PHYS[1]:>10.1f} "
          f"{final_phys[1]:>10.1f} {final_phys[1]-TRUTH_Y:>10.2f}")
    print(f"{'z (mm)':<12} {TRUTH_Z:>10.1f} {INIT_PHYS[2]:>10.1f} "
          f"{final_phys[2]:>10.1f} {final_phys[2]-TRUTH_Z:>10.2f}")
    print(f"{'theta (deg)':<12} {np.degrees(TRUTH_THETA):>10.1f} "
          f"{np.degrees(INIT_THETA):>10.1f} "
          f"{np.degrees(eff_theta_final):>10.1f} "
          f"{np.degrees(eff_theta_final-TRUTH_THETA):>10.2f}")
    print(f"{'phi (deg)':<12} {np.degrees(TRUTH_PHI):>10.1f} "
          f"{np.degrees(INIT_PHI):>10.1f} "
          f"{np.degrees(eff_phi_final):>10.1f} "
          f"{np.degrees(eff_phi_final-TRUTH_PHI):>10.2f}")
    print(f"{'E (MeV)':<12} {TRUTH_ENERGY:>10.1f} {INIT_PHYS[7]:>10.1f} "
          f"{final_phys[7]:>10.1f} "
          f"{final_phys[7]-TRUTH_ENERGY:>10.2f}")

    # --- Save history ---
    out_path = os.path.join(OUT_DIR, 'optimization_history.npz')
    np.savez(
        out_path,
        param_history_phys=param_history_phys,
        loss_history=loss_history,
        truth_phys=TRUTH_PHYS,
        init_phys=INIT_PHYS,
        scales=SCALES,
        n_segments=N_SEGMENTS,
        step_size_mm=STEP_SIZE_MM,
        n_steps=N_STEPS,
        lr=LR,
    )
    print(f"\nSaved {out_path} ({os.path.getsize(out_path) / 1e3:.1f} KB)")
    print("Done!")


if __name__ == '__main__':
    main()
