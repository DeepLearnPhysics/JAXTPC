"""
Loss and gradient surfaces using trig parameterization (sin/cos).

Instead of optimizing (theta, phi), use (sin_theta, cos_theta, sin_phi, cos_phi)
as free variables. The direction vector is built from these directly and
normalized to unit length:

    dir_unnorm = [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
    dir = dir_unnorm / |dir_unnorm|

This avoids the angle parameterization and may smooth out features
like the V-wire parallel singularity at theta=0.524.

Each trig component is swept independently while the others stay at truth.

Supports three loss types: MSE, SW (Sliced Wasserstein), and Pyramid Blur.

Run from project root:
    python3 closure_analysis_muon/muon_gradient_surfaces_trig.py --loss mse
    python3 closure_analysis_muon/muon_gradient_surfaces_trig.py --loss sw
    python3 closure_analysis_muon/muon_gradient_surfaces_trig.py --loss pyramid
    python3 closure_analysis_muon/muon_gradient_surfaces_trig.py --loss all
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit
from tools.losses import pyramid_blur_loss, DEFAULT_PYRAMID_SPEC
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments_trig,
    mask_outside_volume,
    build_muon_forward,
)

# =============================================================================
# Configuration
# =============================================================================

N_SEGMENTS = 2800
STEP_SIZE_MM = 0.5

TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 2
TRUTH_ENERGY = 200.0

# Trig truth values
TRUTH_SIN_THETA = np.sin(TRUTH_THETA)  # 0.7071
TRUTH_COS_THETA = np.cos(TRUTH_THETA)  # 0.7071
TRUTH_SIN_PHI = np.sin(TRUTH_PHI)      # 1.0
TRUTH_COS_PHI = np.cos(TRUTH_PHI)      # 0.0

PLANES = [0, 1, 2]
K = 10000
N_PROJ = 200
N_SWEEP = 40

FD_EPS_TRIG = 0.01
FD_EPS_POS = 0.5
FD_EPS_ENERGY = 0.5

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 8-parameter order: x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy
PARAM_ORDER = ['x', 'y', 'z', 'sin_theta', 'cos_theta', 'sin_phi', 'cos_phi', 'energy']
PARAM_IDX = {n: i for i, n in enumerate(PARAM_ORDER)}

SWEEP_CONFIGS = {
    'sin_theta': {'range': (0.2, 1.0),    'truth': TRUTH_SIN_THETA, 'unit': '',    'eps': FD_EPS_TRIG},
    'cos_theta': {'range': (0.2, 1.0),    'truth': TRUTH_COS_THETA, 'unit': '',    'eps': FD_EPS_TRIG},
    'sin_phi':   {'range': (0.3, 1.0),    'truth': TRUTH_SIN_PHI,   'unit': '',    'eps': FD_EPS_TRIG},
    'cos_phi':   {'range': (-0.7, 0.7),   'truth': TRUTH_COS_PHI,   'unit': '',    'eps': FD_EPS_TRIG},
}


# =============================================================================
# Main
# =============================================================================

def main(loss_types=None):
    if loss_types is None:
        loss_types = ['mse', 'sw', 'pyramid']

    print("=" * 60, flush=True)
    print("TRIG PARAMETERIZATION GRADIENT SURFACES", flush=True)
    print(f"Loss types: {', '.join(loss_types)}", flush=True)
    print("=" * 60, flush=True)
    print(f"Truth trig: sin_θ={TRUTH_SIN_THETA:.4f}, cos_θ={TRUTH_COS_THETA:.4f}, "
          f"sin_φ={TRUTH_SIN_PHI:.4f}, cos_φ={TRUTH_COS_PHI:.4f}", flush=True)

    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    # Truth args: (x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy)
    truth_args = (
        jnp.float32(TRUTH_X), jnp.float32(TRUTH_Y), jnp.float32(TRUTH_Z),
        jnp.float32(TRUTH_SIN_THETA), jnp.float32(TRUTH_COS_THETA),
        jnp.float32(TRUTH_SIN_PHI), jnp.float32(TRUTH_COS_PHI),
        jnp.float32(TRUTH_ENERGY),
    )

    def sim_forward(x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy):
        pos, de = generate_muon_segments_trig(
            energy, jnp.array([x, y, z]),
            sin_theta, cos_theta, sin_phi, cos_phi,
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de)
        return forward(pos, de)

    # Compile sim
    print("Compiling sim_forward...", flush=True)
    t0 = time.time()
    sim_jit = jax.jit(sim_forward)
    truth_signals = sim_jit(*truth_args)
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"  sim compiled ({time.time()-t0:.1f}s)", flush=True)

    ot_key = jax.random.PRNGKey(42)
    target_sigs = {p: truth_signals[p] for p in PLANES}

    ALL = tuple(range(8))

    # Track compiled losses for comparison plots
    compiled = {}

    # --- MSE ---
    if 'mse' in loss_types:
        def mse_loss(x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy):
            sigs = sim_forward(x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy)
            loss = 0.0
            for p in PLANES:
                loss = loss + jnp.mean((sigs[p] - target_sigs[p]) ** 2)
            return loss

        print("\nCompiling MSE loss...", flush=True)
        t0 = time.time()
        mse_jit = jax.jit(mse_loss)
        _ = mse_jit(*truth_args)
        print(f"  MSE fwd ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        mse_vg = jax.jit(jax.value_and_grad(mse_loss, argnums=ALL))
        _ = mse_vg(*truth_args)
        print(f"  MSE fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

        _run_sweeps('MSE', mse_jit, mse_vg, truth_args)
        compiled['mse'] = (mse_jit, mse_vg)

    # --- SW ---
    if 'sw' in loss_types:
        target_clouds = {}
        for p in PLANES:
            pts, w = signal_to_pointcloud(truth_signals[p], K)
            target_clouds[p] = (pts, w)

        def sw_loss(x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy):
            sigs = sim_forward(x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy)
            loss = 0.0
            for p in PLANES:
                pts, w = signal_to_pointcloud(sigs[p], K)
                loss = loss + sliced_wasserstein_loss_jit(
                    pts, w, target_clouds[p][0], target_clouds[p][1],
                    ot_key, n_projections=N_PROJ,
                )
            return loss

        print("\nCompiling SW loss...", flush=True)
        t0 = time.time()
        sw_jit = jax.jit(sw_loss)
        _ = sw_jit(*truth_args)
        print(f"  SW fwd ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        sw_vg = jax.jit(jax.value_and_grad(sw_loss, argnums=ALL))
        _ = sw_vg(*truth_args)
        print(f"  SW fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

        _run_sweeps('SW', sw_jit, sw_vg, truth_args)
        compiled['sw'] = (sw_jit, sw_vg)

    # --- Pyramid Blur ---
    if 'pyramid' in loss_types:
        planes_tuple = tuple(PLANES)

        def pyramid_loss(x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy):
            sigs = sim_forward(x, y, z, sin_theta, cos_theta, sin_phi, cos_phi, energy)
            return pyramid_blur_loss(
                sigs, truth_signals, DEFAULT_PYRAMID_SPEC,
                planes=planes_tuple,
            )

        print("\nCompiling Pyramid Blur loss...", flush=True)
        t0 = time.time()
        pyr_jit = jax.jit(pyramid_loss)
        _ = pyr_jit(*truth_args)
        print(f"  Pyramid fwd ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        pyr_vg = jax.jit(jax.value_and_grad(pyramid_loss, argnums=ALL))
        _ = pyr_vg(*truth_args)
        print(f"  Pyramid fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

        _run_sweeps('Pyramid', pyr_jit, pyr_vg, truth_args)

    # --- Side-by-side comparison (needs both sw and mse) ---
    if 'sw' in compiled and 'mse' in compiled:
        print("\nGenerating comparison plots...", flush=True)
        sw_jit, sw_vg = compiled['sw']
        mse_jit, mse_vg = compiled['mse']
        _comparison_plots(sw_jit, sw_vg, truth_args, mse_jit, mse_vg)

    print("\nDone!", flush=True)


def _run_sweeps(loss_name, loss_jit, loss_vg, truth_args):
    """Run trig parameter sweeps."""
    prefix = {'MSE': 'mse', 'SW': 'sw', 'Pyramid': 'pyramid'}[loss_name]

    for pname, cfg in SWEEP_CONFIGS.items():
        lo, hi = cfg['range']
        truth_val = cfg['truth']
        unit = cfg['unit']
        eps = cfg['eps']
        pidx = PARAM_IDX[pname]

        x_vals = np.linspace(lo, hi, N_SWEEP)
        print(f"\n  {loss_name} — {pname} [{lo}, {hi}]", flush=True)

        def _args(val):
            a = list(truth_args)
            a[pidx] = jnp.float32(val)
            return tuple(a)

        t0 = time.time()
        losses = np.empty(N_SWEEP)
        ad_grads = np.empty(N_SWEEP)
        for i, v in enumerate(x_vals):
            loss, grads = loss_vg(*_args(v))
            losses[i] = float(loss)
            ad_grads[i] = float(grads[pidx])

        fd_grads = np.array([
            (float(loss_jit(*_args(v + eps)))
             - float(loss_jit(*_args(v - eps)))) / (2 * eps)
            for v in x_vals
        ])
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)

        # Plot
        truth_label = f'Truth={truth_val:.4f}'
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(x_vals, losses, 'b-', lw=1.5)
        axes[0].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
        axes[0].set_xlabel(pname)
        axes[0].set_ylabel(f'{loss_name} Loss')
        axes[0].set_title('Loss Landscape')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_vals, ad_grads, 'r-', lw=1.5, label='Autodiff')
        axes[1].plot(x_vals, fd_grads, 'b--', lw=1.5, alpha=0.7, label=f'FD (eps={eps})')
        axes[1].axhline(0, color='k', ls='-', lw=0.5)
        axes[1].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
        axes[1].set_xlabel(pname)
        axes[1].set_ylabel(f'd{loss_name}/d{pname}')
        axes[1].set_title('Gradient (Autodiff vs FD)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f'{loss_name} — varying {pname} (trig param, normalized direction)',
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f'{prefix}_trig_surface_{pname}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}", flush=True)


def _comparison_plots(sw_jit, sw_vg, truth_args, mse_jit, mse_vg):
    """Side-by-side: angle param vs trig param for theta direction."""

    # --- Sweep sin_theta while cos_theta fixed (off-circle) ---
    sin_vals = np.linspace(0.2, 1.0, N_SWEEP)

    sw_losses_sin = np.empty(N_SWEEP)
    mse_losses_sin = np.empty(N_SWEEP)
    # Effective theta for each sin_theta value (with cos_theta fixed at truth)
    effective_theta_sin = np.arctan2(sin_vals, TRUTH_COS_THETA)

    for i, s in enumerate(sin_vals):
        a = list(truth_args)
        a[PARAM_IDX['sin_theta']] = jnp.float32(s)
        sw_losses_sin[i] = float(sw_jit(*tuple(a)))
        mse_losses_sin[i] = float(mse_jit(*tuple(a)))

    # --- Sweep cos_theta while sin_theta fixed (off-circle) ---
    cos_vals = np.linspace(0.2, 1.0, N_SWEEP)

    sw_losses_cos = np.empty(N_SWEEP)
    mse_losses_cos = np.empty(N_SWEEP)
    effective_theta_cos = np.arctan2(TRUTH_SIN_THETA, cos_vals)

    for i, c in enumerate(cos_vals):
        a = list(truth_args)
        a[PARAM_IDX['cos_theta']] = jnp.float32(c)
        sw_losses_cos[i] = float(sw_jit(*tuple(a)))
        mse_losses_cos[i] = float(mse_jit(*tuple(a)))

    # V-parallel condition: -0.866*sin_theta + 0.5*cos_theta = 0
    # For sin_theta sweep: sin_theta_parallel = 0.5*cos_theta_truth/0.866
    sin_parallel = 0.5 * TRUTH_COS_THETA / 0.866
    # For cos_theta sweep: cos_theta_parallel = 0.866*sin_theta_truth/0.5
    cos_parallel = 0.866 * TRUTH_SIN_THETA / 0.5

    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: SW vs sin_theta
    axes[0, 0].plot(sin_vals, sw_losses_sin, 'b-', lw=2)
    axes[0, 0].axvline(TRUTH_SIN_THETA, color='green', ls='--', lw=2,
                        label=f'Truth sin_θ={TRUTH_SIN_THETA:.4f}')
    if 0.2 <= sin_parallel <= 1.0:
        axes[0, 0].axvline(sin_parallel, color='red', ls=':', lw=2,
                            label=f'V-parallel sin_θ={sin_parallel:.4f}')
    axes[0, 0].set_xlabel('sin_theta')
    axes[0, 0].set_ylabel('SW Loss')
    axes[0, 0].set_title('SW vs sin_theta (cos_theta fixed)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: SW vs cos_theta
    axes[0, 1].plot(cos_vals, sw_losses_cos, 'b-', lw=2)
    axes[0, 1].axvline(TRUTH_COS_THETA, color='green', ls='--', lw=2,
                        label=f'Truth cos_θ={TRUTH_COS_THETA:.4f}')
    if 0.2 <= cos_parallel <= 1.0:
        axes[0, 1].axvline(cos_parallel, color='red', ls=':', lw=2,
                            label=f'V-parallel cos_θ={cos_parallel:.4f}')
    axes[0, 1].set_xlabel('cos_theta')
    axes[0, 1].set_ylabel('SW Loss')
    axes[0, 1].set_title('SW vs cos_theta (sin_theta fixed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Effective theta for sin_theta sweep
    ax2 = axes[1, 0]
    ax2.plot(sin_vals, np.degrees(effective_theta_sin), 'k-', lw=1.5)
    ax2.axhline(np.degrees(0.524), color='red', ls=':', lw=1.5, label='V-parallel (30°)')
    ax2.axhline(np.degrees(TRUTH_THETA), color='green', ls='--', lw=1.5, label='Truth (45°)')
    ax2.set_xlabel('sin_theta')
    ax2.set_ylabel('Effective theta (deg)')
    ax2.set_title('Effective angle vs sin_theta')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom-right: Effective theta for cos_theta sweep
    ax3 = axes[1, 1]
    ax3.plot(cos_vals, np.degrees(effective_theta_cos), 'k-', lw=1.5)
    ax3.axhline(np.degrees(0.524), color='red', ls=':', lw=1.5, label='V-parallel (30°)')
    ax3.axhline(np.degrees(TRUTH_THETA), color='green', ls='--', lw=1.5, label='Truth (45°)')
    ax3.set_xlabel('cos_theta')
    ax3.set_ylabel('Effective theta (deg)')
    ax3.set_title('Effective angle vs cos_theta')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        'Trig Parameterization: V-parallel at -0.866·sin_θ + 0.5·cos_θ = 0\n'
        f'sin sweep: V-parallel at sin_θ={sin_parallel:.3f} | '
        f'cos sweep: V-parallel at cos_θ={cos_parallel:.3f} (off-range)',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'sw_trig_vs_angle_comparison.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trig parameterization gradient surfaces')
    parser.add_argument(
        '--loss', type=str, default='all',
        choices=['mse', 'sw', 'pyramid', 'all'],
        help='Which loss to run (default: all)',
    )
    args = parser.parse_args()
    if args.loss == 'all':
        main(['mse', 'sw', 'pyramid'])
    else:
        main([args.loss])
