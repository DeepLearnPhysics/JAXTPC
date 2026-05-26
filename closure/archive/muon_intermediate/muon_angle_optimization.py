"""
Compare angle optimization: direct (theta, phi) vs trig (sin/cos) parameterization.

Both optimize only the two angles (theta, phi) to match a truth muon signal,
with all other parameters (x, y, z, energy) fixed at truth.

1. Direct: optimize theta, phi with Adam
2. Trig:   optimize (sin_theta, cos_theta, sin_phi, cos_phi) with Adam,
           reconstruct angles via atan2 for the simulation

Both use SW loss on east-side planes.

Run from project root:
    python3 closure_analysis_muon/muon_angle_optimization.py
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
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments,
    generate_muon_segments_trig,
    build_muon_forward,
)

# =============================================================================
# Configuration
# =============================================================================

N_SEGMENTS = 2800
STEP_SIZE_MM = 0.5

TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA = np.pi / 4      # 0.785 rad
TRUTH_PHI = np.pi / 2        # 1.571 rad
TRUTH_ENERGY = 200.0

# Starting guess: offset from truth
INIT_THETA = TRUTH_THETA + 0.3   # ~62 deg vs 45 deg
INIT_PHI = TRUTH_PHI - 0.4       # ~67 deg vs 90 deg

PLANES = [0, 1, 2]
K = 10000
N_PROJ = 200
N_STEPS = 200
LR = 1e-2

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60, flush=True)
    print("ANGLE OPTIMIZATION: DIRECT vs TRIG", flush=True)
    print("=" * 60, flush=True)
    print(f"Truth: theta={TRUTH_THETA:.4f}, phi={TRUTH_PHI:.4f}", flush=True)
    print(f"Init:  theta={INIT_THETA:.4f}, phi={INIT_PHI:.4f}", flush=True)
    print(f"Steps: {N_STEPS}, LR: {LR}", flush=True)

    # --- Setup ---
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    ot_key = jax.random.PRNGKey(42)

    # Fixed params
    pos_fixed = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], dtype=jnp.float32)
    energy_fixed = jnp.float32(TRUTH_ENERGY)

    # --- Generate truth ---
    print("\nGenerating truth signal...", flush=True)
    truth_pos, truth_de = generate_muon_segments(
        energy_fixed, pos_fixed, jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    )
    truth_sigs = forward(truth_pos, truth_de)
    for s in truth_sigs:
        jax.block_until_ready(s)

    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(truth_sigs[p], K)
        target_clouds[p] = (pts, w)

    # =================================================================
    # Method 1: Direct angle parameterization
    # =================================================================
    print("\n--- Compiling DIRECT loss ---", flush=True)

    def direct_loss(params):
        theta, phi = params['theta'], params['phi']
        pos, de = generate_muon_segments(
            energy_fixed, pos_fixed, theta, phi,
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        sigs = forward(pos, de)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            loss = loss + sliced_wasserstein_loss_jit(
                pts, w, target_clouds[p][0], target_clouds[p][1],
                ot_key, n_projections=N_PROJ,
            )
        return loss

    direct_vg = jax.jit(jax.value_and_grad(direct_loss))

    # Warmup compile
    t0 = time.time()
    init_direct = {'theta': jnp.float32(INIT_THETA), 'phi': jnp.float32(INIT_PHI)}
    _l, _g = direct_vg(init_direct)
    jax.block_until_ready(_l)
    print(f"  Compiled ({time.time()-t0:.1f}s)", flush=True)
    print(f"  Init loss: {float(_l):.2f}", flush=True)

    # Run optimization
    print("  Running Adam...", flush=True)
    optimizer = optax.adam(LR)
    params = init_direct
    opt_state = optimizer.init(params)

    direct_history = {'loss': [], 'theta': [], 'phi': []}
    t0 = time.time()
    for step in range(N_STEPS):
        loss, grads = direct_vg(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        direct_history['loss'].append(float(loss))
        direct_history['theta'].append(float(params['theta']))
        direct_history['phi'].append(float(params['phi']))

        if step % 50 == 0 or step == N_STEPS - 1:
            print(f"    step {step:3d}: loss={float(loss):.2f}, "
                  f"theta={float(params['theta']):.4f}, phi={float(params['phi']):.4f}",
                  flush=True)

    direct_time = time.time() - t0
    print(f"  Done ({direct_time:.1f}s)", flush=True)

    # =================================================================
    # Method 2: Trig parameterization
    # =================================================================
    print("\n--- Compiling TRIG loss ---", flush=True)

    def trig_loss(params):
        sin_theta = params['sin_theta']
        cos_theta = params['cos_theta']
        sin_phi = params['sin_phi']
        cos_phi = params['cos_phi']
        pos, de = generate_muon_segments_trig(
            energy_fixed, pos_fixed,
            sin_theta, cos_theta, sin_phi, cos_phi,
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        sigs = forward(pos, de)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            loss = loss + sliced_wasserstein_loss_jit(
                pts, w, target_clouds[p][0], target_clouds[p][1],
                ot_key, n_projections=N_PROJ,
            )
        return loss

    trig_vg = jax.jit(jax.value_and_grad(trig_loss))

    # Init from same angles
    init_trig = {
        'sin_theta': jnp.float32(jnp.sin(INIT_THETA)),
        'cos_theta': jnp.float32(jnp.cos(INIT_THETA)),
        'sin_phi':   jnp.float32(jnp.sin(INIT_PHI)),
        'cos_phi':   jnp.float32(jnp.cos(INIT_PHI)),
    }

    t0 = time.time()
    _l, _g = trig_vg(init_trig)
    jax.block_until_ready(_l)
    print(f"  Compiled ({time.time()-t0:.1f}s)", flush=True)
    print(f"  Init loss: {float(_l):.2f}", flush=True)

    # Run optimization
    print("  Running Adam...", flush=True)
    optimizer_trig = optax.adam(LR)
    params_trig = init_trig
    opt_state_trig = optimizer_trig.init(params_trig)

    trig_history = {'loss': [], 'theta': [], 'phi': [],
                    'sin_theta': [], 'cos_theta': [], 'sin_phi': [], 'cos_phi': []}
    t0 = time.time()
    for step in range(N_STEPS):
        loss, grads = trig_vg(params_trig)
        updates, opt_state_trig = optimizer_trig.update(grads, opt_state_trig, params_trig)
        params_trig = optax.apply_updates(params_trig, updates)

        # Project back onto unit circle after each step
        norm_t = jnp.sqrt(params_trig['sin_theta']**2 + params_trig['cos_theta']**2)
        norm_p = jnp.sqrt(params_trig['sin_phi']**2 + params_trig['cos_phi']**2)
        params_trig = {
            'sin_theta': params_trig['sin_theta'] / norm_t,
            'cos_theta': params_trig['cos_theta'] / norm_t,
            'sin_phi':   params_trig['sin_phi'] / norm_p,
            'cos_phi':   params_trig['cos_phi'] / norm_p,
        }

        # Recover effective angles
        eff_theta = float(jnp.arctan2(params_trig['sin_theta'], params_trig['cos_theta']))
        eff_phi = float(jnp.arctan2(params_trig['sin_phi'], params_trig['cos_phi']))

        trig_history['loss'].append(float(loss))
        trig_history['theta'].append(eff_theta)
        trig_history['phi'].append(eff_phi)
        trig_history['sin_theta'].append(float(params_trig['sin_theta']))
        trig_history['cos_theta'].append(float(params_trig['cos_theta']))
        trig_history['sin_phi'].append(float(params_trig['sin_phi']))
        trig_history['cos_phi'].append(float(params_trig['cos_phi']))

        if step % 50 == 0 or step == N_STEPS - 1:
            norm_t = float(jnp.sqrt(params_trig['sin_theta']**2 + params_trig['cos_theta']**2))
            norm_p = float(jnp.sqrt(params_trig['sin_phi']**2 + params_trig['cos_phi']**2))
            print(f"    step {step:3d}: loss={float(loss):.2f}, "
                  f"theta_eff={eff_theta:.4f}, phi_eff={eff_phi:.4f}, "
                  f"|t|={norm_t:.3f}, |p|={norm_p:.3f}",
                  flush=True)

    trig_time = time.time() - t0
    print(f"  Done ({trig_time:.1f}s)", flush=True)

    # =================================================================
    # Plot comparison
    # =================================================================
    print("\nGenerating plots...", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    steps = np.arange(N_STEPS)

    # Row 0: Loss convergence
    axes[0, 0].plot(steps, direct_history['loss'], 'b-', lw=1.5, label='Direct (θ, φ)')
    axes[0, 0].plot(steps, trig_history['loss'], 'r-', lw=1.5, label='Trig (sin/cos)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('SW Loss')
    axes[0, 0].set_title('Loss Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Row 0: Theta convergence
    axes[0, 1].plot(steps, direct_history['theta'], 'b-', lw=1.5, label='Direct θ')
    axes[0, 1].plot(steps, trig_history['theta'], 'r-', lw=1.5, label='Trig θ_eff')
    axes[0, 1].axhline(TRUTH_THETA, color='green', ls='--', lw=2, label=f'Truth θ={TRUTH_THETA:.3f}')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Theta (rad)')
    axes[0, 1].set_title('Theta Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Row 0: Phi convergence
    axes[0, 2].plot(steps, direct_history['phi'], 'b-', lw=1.5, label='Direct φ')
    axes[0, 2].plot(steps, trig_history['phi'], 'r-', lw=1.5, label='Trig φ_eff')
    axes[0, 2].axhline(TRUTH_PHI, color='green', ls='--', lw=2, label=f'Truth φ={TRUTH_PHI:.3f}')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Phi (rad)')
    axes[0, 2].set_title('Phi Convergence')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 1: Trig components
    axes[1, 0].plot(steps, trig_history['sin_theta'], '-', lw=1.5, label='sin_θ')
    axes[1, 0].plot(steps, trig_history['cos_theta'], '-', lw=1.5, label='cos_θ')
    axes[1, 0].axhline(np.sin(TRUTH_THETA), color='C0', ls='--', alpha=0.5)
    axes[1, 0].axhline(np.cos(TRUTH_THETA), color='C1', ls='--', alpha=0.5)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Trig: θ components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, trig_history['sin_phi'], '-', lw=1.5, label='sin_φ')
    axes[1, 1].plot(steps, trig_history['cos_phi'], '-', lw=1.5, label='cos_φ')
    axes[1, 1].axhline(np.sin(TRUTH_PHI), color='C0', ls='--', alpha=0.5)
    axes[1, 1].axhline(np.cos(TRUTH_PHI), color='C1', ls='--', alpha=0.5)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Trig: φ components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Row 1: Trajectory in theta-phi space
    axes[1, 2].plot(direct_history['theta'], direct_history['phi'], 'b.-',
                     ms=2, lw=0.5, alpha=0.7, label='Direct')
    axes[1, 2].plot(trig_history['theta'], trig_history['phi'], 'r.-',
                     ms=2, lw=0.5, alpha=0.7, label='Trig')
    axes[1, 2].plot(TRUTH_THETA, TRUTH_PHI, 'g*', ms=15, label='Truth')
    axes[1, 2].plot(INIT_THETA, INIT_PHI, 'kx', ms=10, mew=2, label='Init')
    axes[1, 2].set_xlabel('Theta (rad)')
    axes[1, 2].set_ylabel('Phi (rad)')
    axes[1, 2].set_title('Trajectory in (θ, φ) space')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(
        f'Angle Optimization: Direct vs Trig Parameterization\n'
        f'Init: θ={INIT_THETA:.3f}, φ={INIT_PHI:.3f} → '
        f'Truth: θ={TRUTH_THETA:.3f}, φ={TRUTH_PHI:.3f} | '
        f'Adam LR={LR}, {N_STEPS} steps',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'optimization_angle_comparison.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)

    # Print final summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Direct: final loss={direct_history['loss'][-1]:.2f}, "
          f"theta={direct_history['theta'][-1]:.4f}, phi={direct_history['phi'][-1]:.4f} "
          f"({direct_time:.1f}s)", flush=True)
    print(f"  Trig:   final loss={trig_history['loss'][-1]:.2f}, "
          f"theta={trig_history['theta'][-1]:.4f}, phi={trig_history['phi'][-1]:.4f} "
          f"({trig_time:.1f}s)", flush=True)
    print(f"  Truth:  theta={TRUTH_THETA:.4f}, phi={TRUTH_PHI:.4f}", flush=True)


if __name__ == '__main__':
    main()
