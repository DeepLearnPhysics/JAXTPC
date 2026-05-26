"""
MCS wire-signal closure optimization (Levels 5-6).

Generates a truth MCS muon track, simulates wire signals, then optimizes
to recover vertex, direction, energy, and per-segment scattering angles.

Modes:
  --mode globals-only    (Level 5):  fix scattering at truth, optimize 8 globals
  --mode scattering-only (Level 5d): fix globals at truth, optimize 2N angles
  --mode full            (Level 6):  optimize all 8+2N params
  --mode staged          (Level 6b): 200 steps globals-only, then all

Run from project root:
    python3 -m closure.mcs.run --mode globals-only --steps 300
    python3 -m closure.mcs.run --mode full --steps 500

Migrated from closure_analysis_MCS/mcs_closure.py.
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import time
import os

from closure.mcs.forward import (
    mcs_cumsum_forward,
    generate_mcs_truth,
    highland_prior,
    build_mcs_forward,
    mask_outside_volume,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight


# =============================================================================
# Configuration
# =============================================================================

N_SEGMENTS = 2000
STEP_SIZE_MM = 0.5
STEP_SIZE_CM = STEP_SIZE_MM / 10.0

# Truth parameters
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

TRUTH_GLOBALS = np.array([
    TRUTH_X, TRUTH_Y, TRUTH_Z,
    TRUTH_SIN_THETA, TRUTH_COS_THETA,
    TRUTH_SIN_PHI, TRUTH_COS_PHI,
    TRUTH_ENERGY,
])

# Normalization
SCALES = np.array([200.0, 200.0, 200.0, 1.0, 1.0, 1.0, 1.0, 500.0])
ANGLE_SCALE = 0.01

# Default optimization params
DEFAULT_LR = 0.015
DEFAULT_LR_PHASE2 = 0.003   # Lower for near-converged globals in stage 2
DEFAULT_LR_SCAT = 0.01      # Higher for scattering angles
LAMBDA_PRIOR = 0.001

PARAM_NAMES = ['x', 'y', 'z', 'sin_th', 'cos_th', 'sin_ph', 'cos_ph', 'energy']
PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Helpers
# =============================================================================

SCALES_JAX = jnp.array(SCALES, dtype=jnp.float32)


def project_unit_circle(g):
    """Project sin/cos pairs in globals onto unit circle."""
    st, ct = g[3], g[4]
    norm_t = jnp.maximum(jnp.sqrt(st**2 + ct**2), 1e-8)
    sp, cp = g[5], g[6]
    norm_p = jnp.maximum(jnp.sqrt(sp**2 + cp**2), 1e-8)
    return g.at[3].set(st / norm_t).at[4].set(ct / norm_t) \
            .at[5].set(sp / norm_p).at[6].set(cp / norm_p)


# =============================================================================
# Build loss functions for each mode
# =============================================================================

def build_globals_only_loss(forward, log_T, dedx, truth_signals, spec_weights,
                            dt1_truth, dt2_truth, half_ext):
    """Loss over 8 global params with scattering fixed at truth."""

    def loss_fn(norm_globals):
        g = norm_globals * SCALES_JAX
        pos, de = mcs_cumsum_forward(
            g[7], jnp.array([g[0], g[1], g[2]]),
            g[3], g[4], g[5], g[6],
            dt1_truth, dt2_truth, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de, half_ext)
        sigs = forward(pos, de)
        return sobolev_loss_geomean_log1p(sigs, truth_signals, spec_weights)

    return loss_fn


def build_scattering_only_loss(forward, log_T, dedx, truth_signals, spec_weights, half_ext):
    """Loss over 2N scattering angles with globals fixed at truth."""
    truth_g = jnp.array(TRUTH_GLOBALS, dtype=jnp.float32)

    def loss_fn(angles_norm):
        dt1 = angles_norm[:N_SEGMENTS] * ANGLE_SCALE
        dt2 = angles_norm[N_SEGMENTS:] * ANGLE_SCALE
        pos, de = mcs_cumsum_forward(
            truth_g[7], jnp.array([truth_g[0], truth_g[1], truth_g[2]]),
            truth_g[3], truth_g[4], truth_g[5], truth_g[6],
            dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de, half_ext)
        sigs = forward(pos, de)
        wire_loss = sobolev_loss_geomean_log1p(sigs, truth_signals, spec_weights)
        prior = highland_prior(dt1, dt2, truth_g[7], STEP_SIZE_CM, N_SEGMENTS,
                               log_T, dedx)
        return wire_loss + LAMBDA_PRIOR * prior / N_SEGMENTS

    return loss_fn


def build_angles_energy_loss(forward, log_T, dedx, truth_signals, spec_weights, half_ext):
    """Loss over energy + 2N scattering angles, with vertex/direction fixed at truth.

    params layout: [energy_norm, dtheta1[0:N], dtheta2[0:N]]
    """
    truth_start = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], dtype=jnp.float32)
    truth_sth = jnp.float32(TRUTH_SIN_THETA)
    truth_cth = jnp.float32(TRUTH_COS_THETA)
    truth_sph = jnp.float32(TRUTH_SIN_PHI)
    truth_cph = jnp.float32(TRUTH_COS_PHI)
    energy_scale = jnp.float32(SCALES[7])

    def loss_fn(params):
        energy = params[0] * energy_scale
        dt1 = params[1:1 + N_SEGMENTS] * ANGLE_SCALE
        dt2 = params[1 + N_SEGMENTS:] * ANGLE_SCALE
        pos, de = mcs_cumsum_forward(
            energy, truth_start,
            truth_sth, truth_cth, truth_sph, truth_cph,
            dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de, half_ext)
        sigs = forward(pos, de)
        wire_loss = sobolev_loss_geomean_log1p(sigs, truth_signals, spec_weights)
        prior = highland_prior(dt1, dt2, energy, STEP_SIZE_CM, N_SEGMENTS,
                               log_T, dedx)
        return wire_loss + LAMBDA_PRIOR * prior / N_SEGMENTS

    return loss_fn


def build_full_loss(forward, log_T, dedx, truth_signals, spec_weights, half_ext):
    """Loss over 8 globals + 2N scattering angles."""

    def loss_fn(params):
        g = params[:8] * SCALES_JAX
        dt1 = params[8:8 + N_SEGMENTS] * ANGLE_SCALE
        dt2 = params[8 + N_SEGMENTS:] * ANGLE_SCALE
        pos, de = mcs_cumsum_forward(
            g[7], jnp.array([g[0], g[1], g[2]]),
            g[3], g[4], g[5], g[6],
            dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de, half_ext)
        sigs = forward(pos, de)
        wire_loss = sobolev_loss_geomean_log1p(sigs, truth_signals, spec_weights)
        prior = highland_prior(dt1, dt2, g[7], STEP_SIZE_CM, N_SEGMENTS,
                               log_T, dedx)
        return wire_loss + LAMBDA_PRIOR * prior / N_SEGMENTS

    return loss_fn


# =============================================================================
# Optimization loop
# =============================================================================

def run_optimization(loss_fn, init_params, n_steps, lr, mode,
                     project_fn=None, lr_scat=None, has_globals=True):
    """Generic Adam optimization loop.

    Parameters
    ----------
    loss_fn : callable — loss(params) -> scalar
    init_params : jnp.ndarray — initial parameter vector
    n_steps : int — number of Adam steps
    lr : float — learning rate
    mode : str — for printing
    project_fn : callable or None — called on params after each step
    lr_scat : float or None — if set, scale gradients for indices >= 8
    has_globals : bool — whether first 8 params are globals (for display)

    Returns
    -------
    params : final parameters
    loss_hist : list of float
    param_hist : list of np.ndarray (first 8 elements only)
    opt_state : final optimizer state
    """
    print(f"\nCompiling loss + gradient ({mode})...", flush=True)
    t0 = time.time()
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    init_loss, init_grad = loss_and_grad(init_params)
    jax.block_until_ready(init_grad)
    print(f"  Compiled ({time.time()-t0:.1f}s), initial loss = {float(init_loss):.6f}")

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_params)
    params = init_params

    loss_hist = []
    param_hist = []

    print(f"Running {n_steps} optimization steps (LR={lr})...", flush=True)
    t_start = time.time()

    for step in range(n_steps):
        loss, grad = loss_and_grad(params)
        loss_hist.append(float(loss))

        # Store globals for history
        if has_globals:
            param_hist.append(np.array(params[:8]))

        updates, opt_state = optimizer.update(grad, opt_state, params)

        # Scale scattering angle updates if separate LR
        if lr_scat is not None and params.shape[0] > 8:
            scale_ratio = lr_scat / lr
            updates = updates.at[8:].multiply(scale_ratio)

        params = optax.apply_updates(params, updates)

        if project_fn is not None:
            params = project_fn(params)

        if step % 50 == 0 or step == n_steps - 1:
            elapsed = time.time() - t_start
            if has_globals:
                g = np.array(params[:8] * SCALES)
                eff_theta = np.degrees(np.arctan2(g[3], g[4]))
                eff_phi = np.degrees(np.arctan2(g[5], g[6]))
                print(f"  Step {step:4d}: loss={float(loss):.6f}, "
                      f"x={g[0]:7.1f}, y={g[1]:7.1f}, z={g[2]:7.1f}, "
                      f"th={eff_theta:5.1f}, ph={eff_phi:5.1f}, "
                      f"E={g[7]:6.1f}  ({elapsed:.1f}s)", flush=True)
            else:
                # Scattering-only: show angle RMS
                dt = np.array(params[:params.shape[0]//2]) * ANGLE_SCALE
                rms = np.sqrt(np.mean(dt**2)) * 1000
                print(f"  Step {step:4d}: loss={float(loss):.6f}, "
                      f"angle RMS={rms:.3f} mrad  ({elapsed:.1f}s)",
                      flush=True)

    total_time = time.time() - t_start
    print(f"Optimization complete in {total_time:.1f}s ({total_time/n_steps:.2f}s/step)")

    return params, loss_hist, param_hist, opt_state


# =============================================================================
# Plotting
# =============================================================================

def plot_results(loss_hist, param_hist, mode, truth_globals_norm, init_globals_norm,
                 dt1_truth=None, dt1_fit=None, dt2_truth=None, dt2_fit=None):
    """Generate summary plots."""
    steps = np.arange(len(loss_hist))
    has_globals = len(param_hist) > 0
    has_angles = dt1_fit is not None

    n_rows = 2 if has_globals else 1
    n_cols = 3 if has_angles else 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Loss
    axes[0, 0].semilogy(steps, loss_hist, 'b-', lw=1.5)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss (log)')
    axes[0, 0].set_title('Loss Convergence')
    axes[0, 0].grid(True, alpha=0.3)

    if has_globals:
        ph = np.array(param_hist) * SCALES[np.newaxis, :]
        truth_phys = truth_globals_norm * SCALES
        init_phys = init_globals_norm * SCALES

        # Vertex convergence
        for i, (name, idx) in enumerate([('x', 0), ('y', 1), ('z', 2)]):
            if i < 2:
                ax = axes[0, i + 1]
            else:
                ax = axes[1, 0]
            ax.plot(ph[:, idx], 'b-', lw=1.5)
            ax.axhline(truth_phys[idx], color='g', ls='--', lw=2, label=f'truth')
            ax.axhline(init_phys[idx], color='r', ls=':', lw=1.5, label=f'init')
            ax.set_xlabel('Step')
            ax.set_ylabel(f'{name} (mm)')
            ax.set_title(f'{name} Convergence')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Theta
        eff_theta = np.degrees(np.arctan2(ph[:, 3], ph[:, 4]))
        ax = axes[1, 1]
        ax.plot(eff_theta, 'b-', lw=1.5)
        ax.axhline(np.degrees(TRUTH_THETA), color='g', ls='--', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('theta (deg)')
        ax.set_title('theta Convergence')
        ax.grid(True, alpha=0.3)

        # Energy
        ax = axes[1, 2]
        ax.plot(ph[:, 7], 'b-', lw=1.5)
        ax.axhline(TRUTH_ENERGY, color='g', ls='--', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy (MeV)')
        ax.set_title('Energy Convergence')
        ax.grid(True, alpha=0.3)

    elif has_angles:
        # Scattering angle comparison
        dt1_t = np.array(dt1_truth)
        dt1_f = np.array(dt1_fit)
        corr = np.corrcoef(dt1_f, dt1_t)[0, 1]

        ax = axes[0, 1]
        seg_idx = np.arange(len(dt1_t))
        ax.plot(seg_idx, dt1_t * 1000, 'g-', lw=0.3, alpha=0.5, label='truth')
        ax.plot(seg_idx, dt1_f * 1000, 'b-', lw=0.3, alpha=0.5, label='fit')
        ax.set_xlabel('Segment')
        ax.set_ylabel('dtheta1 (mrad)')
        ax.set_title(f'Scattering angles (corr={corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # RMS comparison
        window = 50
        rms_truth = np.sqrt(np.convolve(dt1_t**2, np.ones(window)/window, mode='valid'))
        rms_fit = np.sqrt(np.convolve(dt1_f**2, np.ones(window)/window, mode='valid'))
        ax = axes[0, 2]
        ax.plot(rms_truth * 1000, 'g-', lw=1, label='truth RMS')
        ax.plot(rms_fit * 1000, 'b-', lw=1, label='fit RMS')
        ax.set_xlabel('Segment')
        ax.set_ylabel('RMS dtheta (mrad)')
        ax.set_title('Running RMS scattering')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'MCS Closure: {mode}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'mcs_closure_{mode.replace("-", "_")}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Main
# =============================================================================

def main():
    global LAMBDA_PRIOR

    parser = argparse.ArgumentParser(description='MCS closure optimization')
    parser.add_argument('--mode', choices=['globals-only', 'scattering-only',
                                          'angles-energy', 'full', 'staged'],
                        default='staged')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--stage1-steps', type=int, default=200,
                        help='Globals-only steps in staged mode')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--lr-phase2', type=float, default=DEFAULT_LR_PHASE2,
                        help='Globals LR in stage 2 (lower to protect converged globals)')
    parser.add_argument('--lr-scat', type=float, default=DEFAULT_LR_SCAT)
    parser.add_argument('--lambda-prior', type=float, default=LAMBDA_PRIOR)
    args = parser.parse_args()

    LAMBDA_PRIOR = args.lambda_prior

    print("=" * 70)
    print(f"MCS CLOSURE OPTIMIZATION — mode={args.mode}")
    print("=" * 70)
    print(f"Truth: x={TRUTH_X}, y={TRUTH_Y}, z={TRUTH_Z}")
    print(f"       theta={np.degrees(TRUTH_THETA):.1f} deg, "
          f"phi={np.degrees(TRUTH_PHI):.1f} deg, E={TRUTH_ENERGY} MeV")
    print(f"N_SEGMENTS={N_SEGMENTS}, STEP_SIZE={STEP_SIZE_MM}mm, "
          f"steps={args.steps}, lr={args.lr}")
    print()

    # --- Load resources ---
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(detector_config)
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    # --- Generate truth ---
    print("Generating truth MCS track...", flush=True)
    rng_key = jax.random.PRNGKey(42)
    pos_truth, de_truth, dt1_truth, dt2_truth = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], dtype=jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, rng_key,
    )
    de_truth_masked = mask_outside_volume(pos_truth, de_truth, half_ext)

    # --- Truth wire signals ---
    print("Computing truth wire signals...", flush=True)
    t0 = time.time()
    truth_signals = jax.jit(forward)(pos_truth, de_truth_masked)
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"  Compiled ({time.time()-t0:.1f}s)")

    for i, name in enumerate(PLANE_NAMES):
        sig = truth_signals[i]
        print(f"  {name}: shape={sig.shape}, "
              f"max={float(jnp.max(jnp.abs(sig))):.4f}, "
              f"sum_abs={float(jnp.sum(jnp.abs(sig))):.2f}")

    # --- Sobolev weights ---
    print("Precomputing Sobolev spectral weights (s=1.5)...", flush=True)
    spec_weights = tuple(
        make_sobolev_weight(*truth_signals[p].shape, s=1.5) for p in range(6)
    )

    # --- Initial guess for globals ---
    init_theta = TRUTH_THETA + 0.5
    init_phi = TRUTH_PHI + 0.5
    init_globals = np.array([
        TRUTH_X + 100.0, TRUTH_Y + 100.0, TRUTH_Z - 100.0,
        np.sin(init_theta), np.cos(init_theta),
        np.sin(init_phi), np.cos(init_phi),
        TRUTH_ENERGY + 100.0,
    ])
    truth_globals_norm = TRUTH_GLOBALS / SCALES
    init_globals_norm = init_globals / SCALES

    # === Mode dispatch ===

    if args.mode == 'globals-only':
        loss_fn = build_globals_only_loss(
            forward, log_T, dedx, truth_signals, spec_weights,
            dt1_truth, dt2_truth, half_ext,
        )

        def project(p):
            return project_unit_circle(p)

        params, loss_hist, param_hist, _ = run_optimization(
            loss_fn, jnp.array(init_globals_norm, dtype=jnp.float32),
            args.steps, args.lr, args.mode, project_fn=project,
        )
        _print_globals_summary(params * SCALES, init_globals)
        plot_results(loss_hist, param_hist, args.mode,
                     truth_globals_norm, init_globals_norm)

    elif args.mode == 'scattering-only':
        loss_fn = build_scattering_only_loss(
            forward, log_T, dedx, truth_signals, spec_weights, half_ext,
        )
        init_angles = jnp.zeros(2 * N_SEGMENTS)

        params, loss_hist, _, _ = run_optimization(
            loss_fn, init_angles, args.steps, args.lr_scat,
            args.mode, has_globals=False,
        )
        dt1_fit = np.array(params[:N_SEGMENTS]) * ANGLE_SCALE
        dt2_fit = np.array(params[N_SEGMENTS:]) * ANGLE_SCALE
        _print_scattering_summary(dt1_fit, dt2_fit, dt1_truth, dt2_truth)
        plot_results(loss_hist, [], args.mode,
                     truth_globals_norm, init_globals_norm,
                     dt1_truth, dt1_fit, dt2_truth, dt2_fit)

    elif args.mode == 'angles-energy':
        loss_fn = build_angles_energy_loss(
            forward, log_T, dedx, truth_signals, spec_weights, half_ext,
        )
        # params: [energy_norm, dtheta1[0:N], dtheta2[0:N]]
        init_energy_norm = jnp.float32((TRUTH_ENERGY + 100.0) / SCALES[7])
        init_params = jnp.concatenate([
            jnp.array([init_energy_norm]),
            jnp.zeros(2 * N_SEGMENTS),
        ])

        params, loss_hist, _, _ = run_optimization(
            loss_fn, init_params, args.steps, args.lr_scat,
            args.mode, has_globals=False,
        )
        final_energy = float(params[0]) * SCALES[7]
        print(f"\nEnergy: truth={TRUTH_ENERGY:.1f}, "
              f"init={TRUTH_ENERGY + 100.0:.1f}, "
              f"final={final_energy:.1f}, "
              f"delta={final_energy - TRUTH_ENERGY:.2f} MeV")
        dt1_fit = np.array(params[1:1 + N_SEGMENTS]) * ANGLE_SCALE
        dt2_fit = np.array(params[1 + N_SEGMENTS:]) * ANGLE_SCALE
        _print_scattering_summary(dt1_fit, dt2_fit, dt1_truth, dt2_truth)
        plot_results(loss_hist, [], args.mode,
                     truth_globals_norm, init_globals_norm,
                     dt1_truth, dt1_fit, dt2_truth, dt2_fit)

    elif args.mode == 'full':
        loss_fn = build_full_loss(
            forward, log_T, dedx, truth_signals, spec_weights, half_ext,
        )
        init_params = jnp.concatenate([
            jnp.array(init_globals_norm, dtype=jnp.float32),
            jnp.zeros(2 * N_SEGMENTS),
        ])

        def project_full(p):
            g = project_unit_circle(p[:8])
            return p.at[:8].set(g)

        params, loss_hist, param_hist, _ = run_optimization(
            loss_fn, init_params, args.steps, args.lr,
            args.mode, project_fn=project_full, lr_scat=args.lr_scat,
        )
        _print_globals_summary(np.array(params[:8]) * SCALES, init_globals)
        dt1_fit = np.array(params[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        dt2_fit = np.array(params[8 + N_SEGMENTS:]) * ANGLE_SCALE
        _print_scattering_summary(dt1_fit, dt2_fit, dt1_truth, dt2_truth)
        plot_results(loss_hist, param_hist, args.mode,
                     truth_globals_norm, init_globals_norm,
                     dt1_truth, dt1_fit, dt2_truth, dt2_fit)

    elif args.mode == 'staged':
        # Stage 1: globals only
        print("\n--- Stage 1: globals-only (200 steps) ---")
        loss_fn_g = build_globals_only_loss(
            forward, log_T, dedx, truth_signals, spec_weights,
            dt1_truth, dt2_truth, half_ext,
        )
        stage1_steps = min(args.stage1_steps, args.steps)

        def project_g(p):
            return project_unit_circle(p)

        params_g, loss_hist_1, param_hist_1, _ = run_optimization(
            loss_fn_g, jnp.array(init_globals_norm, dtype=jnp.float32),
            stage1_steps, args.lr, 'staged-phase1', project_fn=project_g,
        )

        # Stage 2: all params
        print("\n--- Stage 2: full (remaining steps) ---")
        remaining = args.steps - stage1_steps
        if remaining > 0:
            loss_fn_full = build_full_loss(
                forward, log_T, dedx, truth_signals, spec_weights, half_ext,
            )
            init_full = jnp.concatenate([
                params_g,
                jnp.zeros(2 * N_SEGMENTS),
            ])

            def project_full(p):
                g = project_unit_circle(p[:8])
                return p.at[:8].set(g)

            params, loss_hist_2, param_hist_2, _ = run_optimization(
                loss_fn_full, init_full, remaining, args.lr_phase2,
                'staged-phase2', project_fn=project_full, lr_scat=args.lr_scat,
            )
            loss_hist = loss_hist_1 + loss_hist_2
            param_hist = param_hist_1 + param_hist_2
        else:
            params = jnp.concatenate([params_g, jnp.zeros(2 * N_SEGMENTS)])
            loss_hist = loss_hist_1
            param_hist = param_hist_1

        _print_globals_summary(np.array(params[:8]) * SCALES, init_globals)
        dt1_fit = np.array(params[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        dt2_fit = np.array(params[8 + N_SEGMENTS:]) * ANGLE_SCALE
        _print_scattering_summary(dt1_fit, dt2_fit, dt1_truth, dt2_truth)
        plot_results(loss_hist, param_hist, args.mode,
                     truth_globals_norm, init_globals_norm,
                     dt1_truth, dt1_fit, dt2_truth, dt2_fit)

    print("\nDone!")


# =============================================================================
# Summary printers
# =============================================================================

def _print_globals_summary(final_phys, init_phys):
    """Print parameter convergence table."""
    eff_theta = np.arctan2(final_phys[3], final_phys[4])
    eff_phi = np.arctan2(final_phys[5], final_phys[6])

    print(f"\n{'Parameter':<12} {'Truth':>10} {'Initial':>10} {'Final':>10} {'Delta':>10}")
    print("-" * 55)
    print(f"{'x (mm)':<12} {TRUTH_X:>10.1f} {init_phys[0]:>10.1f} "
          f"{final_phys[0]:>10.1f} {final_phys[0]-TRUTH_X:>10.2f}")
    print(f"{'y (mm)':<12} {TRUTH_Y:>10.1f} {init_phys[1]:>10.1f} "
          f"{final_phys[1]:>10.1f} {final_phys[1]-TRUTH_Y:>10.2f}")
    print(f"{'z (mm)':<12} {TRUTH_Z:>10.1f} {init_phys[2]:>10.1f} "
          f"{final_phys[2]:>10.1f} {final_phys[2]-TRUTH_Z:>10.2f}")
    print(f"{'theta (deg)':<12} {np.degrees(TRUTH_THETA):>10.1f} "
          f"{np.degrees(np.arctan2(init_phys[3], init_phys[4])):>10.1f} "
          f"{np.degrees(eff_theta):>10.1f} "
          f"{np.degrees(eff_theta - TRUTH_THETA):>10.2f}")
    print(f"{'phi (deg)':<12} {np.degrees(TRUTH_PHI):>10.1f} "
          f"{np.degrees(np.arctan2(init_phys[5], init_phys[6])):>10.1f} "
          f"{np.degrees(eff_phi):>10.1f} "
          f"{np.degrees(eff_phi - TRUTH_PHI):>10.2f}")
    print(f"{'E (MeV)':<12} {TRUTH_ENERGY:>10.1f} {init_phys[7]:>10.1f} "
          f"{final_phys[7]:>10.1f} {final_phys[7]-TRUTH_ENERGY:>10.2f}")


def _print_scattering_summary(dt1_fit, dt2_fit, dt1_truth, dt2_truth):
    """Print scattering angle recovery statistics.

    Uses windowed cumulative deflection as the primary metric: bin
    per-segment angles into wire-pitch-scale windows and compare the
    cumulative deflection per window.  This matches the resolution
    that wire signals can actually resolve (~3mm wire pitch / step_size).
    """
    dt1_t = np.array(dt1_truth)
    dt2_t = np.array(dt2_truth)
    N = len(dt1_t)

    # Per-segment stats (for reference)
    rms_truth = np.sqrt(np.mean(dt1_t**2 + dt2_t**2))
    rms_fit = np.sqrt(np.mean(dt1_fit**2 + dt2_fit**2))

    print(f"\nScattering recovery:")
    print(f"  Per-segment RMS: truth={rms_truth*1000:.3f} mrad, "
          f"fit={rms_fit*1000:.3f} mrad")

    # Windowed cumulative deflection — the observable metric
    # Wire pitch ~3mm; window = pitch / step_size segments
    wire_pitch_mm = 3.0
    window = max(1, int(round(wire_pitch_mm / STEP_SIZE_MM)))
    n_windows = N // window

    if n_windows > 1:
        dt1_t_win = dt1_t[:n_windows * window].reshape(n_windows, window).sum(axis=1)
        dt2_t_win = dt2_t[:n_windows * window].reshape(n_windows, window).sum(axis=1)
        dt1_f_win = dt1_fit[:n_windows * window].reshape(n_windows, window).sum(axis=1)
        dt2_f_win = dt2_fit[:n_windows * window].reshape(n_windows, window).sum(axis=1)

        rms_win_truth = np.sqrt(np.mean(dt1_t_win**2 + dt2_t_win**2))
        rms_win_fit = np.sqrt(np.mean(dt1_f_win**2 + dt2_f_win**2))
        rms_win_resid = np.sqrt(np.mean((dt1_f_win - dt1_t_win)**2 +
                                         (dt2_f_win - dt2_t_win)**2))

        # Correlation on windowed deflections
        corr_win1 = np.corrcoef(dt1_f_win, dt1_t_win)[0, 1]
        corr_win2 = np.corrcoef(dt2_f_win, dt2_t_win)[0, 1]

        print(f"  Windowed deflection ({window} segs = {window*STEP_SIZE_MM:.1f}mm, "
              f"{n_windows} windows):")
        print(f"    RMS: truth={rms_win_truth*1000:.3f} mrad, "
              f"fit={rms_win_fit*1000:.3f} mrad, "
              f"residual={rms_win_resid*1000:.3f} mrad")
        print(f"    Correlation: plane1={corr_win1:.4f}, plane2={corr_win2:.4f}")

    # Cumulative angle agreement
    cum1_t = np.cumsum(dt1_t)
    cum2_t = np.cumsum(dt2_t)
    cum1_f = np.cumsum(dt1_fit)
    cum2_f = np.cumsum(dt2_fit)
    cum_corr1 = np.corrcoef(cum1_f, cum1_t)[0, 1]
    cum_corr2 = np.corrcoef(cum2_f, cum2_t)[0, 1]
    total_truth = np.sqrt(cum1_t[-1]**2 + cum2_t[-1]**2)
    total_fit = np.sqrt(cum1_f[-1]**2 + cum2_f[-1]**2)

    print(f"  Cumulative angle: truth={total_truth*1000:.1f} mrad, "
          f"fit={total_fit*1000:.1f} mrad")
    print(f"    Correlation: plane1={cum_corr1:.4f}, plane2={cum_corr2:.4f}")


if __name__ == '__main__':
    main()
