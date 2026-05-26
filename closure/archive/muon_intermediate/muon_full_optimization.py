"""Full muon parameter optimization using both TPC sides.

Optimizes all 8 free variables (x, y, z, sin_theta, cos_theta, sin_phi,
cos_phi, energy) using Sobolev H^-1 geomean loss over all 6 planes
(east U/V/Y + west U/V/Y).

Parameters are normalized so all live in O(1) space for balanced gradients:
  normalized = physical / SCALES
The optimizer works in normalized space; the loss denormalizes before
calling the simulation.

Trig parameterization with unit circle projection after each Adam step.

Run from project root:
    python3 closure_analysis_muon/muon_full_optimization.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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
# Configuration
# =============================================================================

N_SEGMENTS = 4000
STEP_SIZE_MM = 0.5

# Truth parameters — track starts in east side, crosses to west.
# phi=pi/6 gives positive x-component: sin(pi/4)*cos(pi/6) = 0.61
# so the track travels from x=-200 toward positive x, crossing x=0.
TRUTH_X = -200.0          # mm, east side near center
TRUTH_Y = 0.0             # mm
TRUTH_Z = 100.0           # mm
TRUTH_THETA = np.pi / 4   # 45 deg polar
TRUTH_PHI = np.pi / 6     # 30 deg azimuthal
TRUTH_ENERGY = 500.0      # MeV

# Trig truth values
TRUTH_SIN_THETA = np.sin(TRUTH_THETA)
TRUTH_COS_THETA = np.cos(TRUTH_THETA)
TRUTH_SIN_PHI = np.sin(TRUTH_PHI)
TRUTH_COS_PHI = np.cos(TRUTH_PHI)

# Physical-space truth: [x, y, z, sin_th, cos_th, sin_ph, cos_ph, energy]
TRUTH_PHYS = np.array([
    TRUTH_X, TRUTH_Y, TRUTH_Z,
    TRUTH_SIN_THETA, TRUTH_COS_THETA,
    TRUTH_SIN_PHI, TRUTH_COS_PHI,
    TRUTH_ENERGY,
])

# Normalization scales — makes all params O(1) in optimizer space.
SCALES = np.array([200.0, 200.0, 200.0, 1.0, 1.0, 1.0, 1.0, 500.0])

# Initial guess: perturb all parameters from truth
THETA_PERTURB = 0.5   # rad (~29 deg)
PHI_PERTURB = 0.5     # rad (~29 deg)
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

# Optimization
N_STEPS = 300
LR = 0.015

PARAM_NAMES = ['x', 'y', 'z', 'sin_th', 'cos_th', 'sin_ph', 'cos_ph', 'energy']
PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Helpers
# =============================================================================

SCALES_JAX = jnp.array(SCALES, dtype=jnp.float32)


def to_physical(norm_params):
    """Normalized -> physical space."""
    return norm_params * SCALES_JAX


def project_unit_circle(norm_params):
    """Project sin/cos pairs onto unit circle."""
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
    print("FULL MUON PARAMETER OPTIMIZATION (BOTH SIDES, SOBOLEV GEOMEAN)")
    print("=" * 70)
    print(f"Truth: x={TRUTH_X}, y={TRUTH_Y}, z={TRUTH_Z}")
    print(f"       theta={np.degrees(TRUTH_THETA):.1f} deg, "
          f"phi={np.degrees(TRUTH_PHI):.1f} deg, E={TRUTH_ENERGY} MeV")
    print(f"Init:  x={INIT_PHYS[0]:.0f}, y={INIT_PHYS[1]:.0f}, "
          f"z={INIT_PHYS[2]:.0f}")
    print(f"       theta={np.degrees(INIT_THETA):.1f} deg, "
          f"phi={np.degrees(INIT_PHI):.1f} deg, E={INIT_PHYS[7]:.0f} MeV")
    print(f"Scales: {SCALES}")
    print(f"N_SEGMENTS={N_SEGMENTS}, N_STEPS={N_STEPS}, LR={LR}")
    print(f"Loss: Sobolev H^-1 Geomean s=1.5")
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

    for i, name in enumerate(PLANE_NAMES):
        sig = truth_signals[i]
        print(f"  {name}: shape={sig.shape}, "
              f"max={float(jnp.max(jnp.abs(sig))):.4f}, "
              f"sum_abs={float(jnp.sum(jnp.abs(sig))):.2f}")

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
    grad_strs = [f"{PARAM_NAMES[i]}={float(init_grad[i]):.4f}"
                 for i in range(8)]
    print(f"  Normalized grads: {', '.join(grad_strs)}")

    # --- Optimizer ---
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(init_n)

    # --- Optimization loop ---
    print(f"\nRunning {N_STEPS} optimization steps (LR={LR})...", flush=True)
    params_n = init_n

    # History (store normalized, convert to physical at end)
    loss_history = np.empty(N_STEPS)
    norm_history = np.empty((N_STEPS, 8))
    norm_theta_history = np.empty(N_STEPS)
    norm_phi_history = np.empty(N_STEPS)

    t_start = time.time()
    for step in range(N_STEPS):
        loss, grad = loss_and_grad(params_n)

        loss_history[step] = float(loss)
        norm_history[step] = np.array(params_n)

        updates, opt_state = optimizer.update(grad, opt_state, params_n)
        params_n = optax.apply_updates(params_n, updates)

        norm_theta_history[step] = float(
            jnp.sqrt(params_n[3]**2 + params_n[4]**2))
        norm_phi_history[step] = float(
            jnp.sqrt(params_n[5]**2 + params_n[6]**2))

        params_n = project_unit_circle(params_n)

        if step % 20 == 0 or step == N_STEPS - 1:
            p = np.array(to_physical(params_n))
            eff_theta = float(jnp.arctan2(params_n[3], params_n[4]))
            eff_phi = float(jnp.arctan2(params_n[5], params_n[6]))
            elapsed = time.time() - t_start
            print(f"  Step {step:3d}: loss={float(loss):.6f}, "
                  f"x={p[0]:7.1f}, y={p[1]:7.1f}, z={p[2]:7.1f}, "
                  f"th={np.degrees(eff_theta):5.1f} deg, "
                  f"ph={np.degrees(eff_phi):5.1f} deg, "
                  f"E={p[7]:6.1f} MeV  "
                  f"({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t_start
    print(f"\nOptimization complete in {total_time:.1f}s "
          f"({total_time/N_STEPS:.2f}s/step)")

    # --- Final results ---
    param_history = norm_history * SCALES[np.newaxis, :]
    final_phys = param_history[-1]
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

    # --- Plot ---
    _plot_results(loss_history, param_history,
                  norm_theta_history, norm_phi_history)
    print("\nDone!")


# =============================================================================
# Plotting
# =============================================================================

def _plot_convergence(ax, steps, values, truth, init, ylabel, title):
    """Plot a single parameter convergence subplot."""
    ax.plot(steps, values, 'b-', lw=1.5, label='optimized')
    ax.axhline(truth, color='green', ls='--', lw=2, label=f'truth={truth:.4g}')
    ax.axhline(init, color='red', ls=':', lw=1.5, label=f'init={init:.4g}')
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_results(loss_history, param_history,
                  norm_theta_history, norm_phi_history):
    """Generate 3x3 summary plot."""
    steps = np.arange(len(loss_history))
    eff_theta = np.arctan2(param_history[:, 3], param_history[:, 4])
    eff_phi = np.arctan2(param_history[:, 5], param_history[:, 6])

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # Loss convergence
    ax = axes[0, 0]
    ax.semilogy(steps, loss_history, 'b-', lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Sobolev Geomean Loss (log)')
    ax.set_title('Loss Convergence')
    ax.grid(True, alpha=0.3)

    # Parameter convergences
    specs = [
        (axes[0, 1], param_history[:, 0], TRUTH_X, INIT_PHYS[0],
         'x (mm)', 'x Convergence'),
        (axes[0, 2], param_history[:, 1], TRUTH_Y, INIT_PHYS[1],
         'y (mm)', 'y Convergence'),
        (axes[1, 0], param_history[:, 2], TRUTH_Z, INIT_PHYS[2],
         'z (mm)', 'z Convergence'),
        (axes[1, 1], np.degrees(eff_theta), np.degrees(TRUTH_THETA),
         np.degrees(INIT_THETA), 'theta (deg)', 'theta Convergence'),
        (axes[1, 2], np.degrees(eff_phi), np.degrees(TRUTH_PHI),
         np.degrees(INIT_PHI), 'phi (deg)', 'phi Convergence'),
        (axes[2, 0], param_history[:, 7], TRUTH_ENERGY, INIT_PHYS[7],
         'Energy (MeV)', 'Energy Convergence'),
    ]
    for ax, values, truth, init, ylabel, title in specs:
        _plot_convergence(ax, steps, values, truth, init, ylabel, title)

    # theta-phi trajectory
    ax = axes[2, 1]
    ax.plot(np.degrees(eff_phi), np.degrees(eff_theta), 'b-', lw=1, alpha=0.7)
    ax.plot(np.degrees(eff_phi[0]), np.degrees(eff_theta[0]),
            'ro', ms=10, label='start')
    ax.plot(np.degrees(eff_phi[-1]), np.degrees(eff_theta[-1]),
            'bs', ms=10, label='final')
    ax.plot(np.degrees(TRUTH_PHI), np.degrees(TRUTH_THETA),
            'g*', ms=15, label='truth')
    ax.set_xlabel('phi (deg)')
    ax.set_ylabel('theta (deg)')
    ax.set_title('theta-phi Trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # sin/cos norms
    ax = axes[2, 2]
    ax.plot(steps, norm_theta_history, 'b-', lw=1.5,
            label='||(sin_th, cos_th)||')
    ax.plot(steps, norm_phi_history, 'r-', lw=1.5,
            label='||(sin_ph, cos_ph)||')
    ax.axhline(1.0, color='green', ls='--', lw=2, label='target=1.0')
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm (pre-projection)')
    ax.set_title('sin/cos Pair Norms')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Full Muon Optimization: 6 planes, Sobolev Geomean s=1.5, '
        f'{N_STEPS} steps, Adam LR={LR}\n'
        f'Truth: x={TRUTH_X}, y={TRUTH_Y}, z={TRUTH_Z}, '
        f'theta={np.degrees(TRUTH_THETA):.1f} deg, '
        f'phi={np.degrees(TRUTH_PHI):.1f} deg, E={TRUTH_ENERGY} MeV',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'optimization_full_muon.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


if __name__ == '__main__':
    main()
