"""
Loss and gradient surface plots for differentiable muon simulation.

For each of 6 muon parameters (x, y, z, theta, phi, energy), sweeps the
parameter while holding the others fixed at truth and plots:
  - Loss landscape
  - Gradient comparison (autodiff vs finite-difference)

Supports three loss types: MSE, SW (Sliced Wasserstein), and Blur MSE.

Run from project root:
    python3 closure_analysis_muon/muon_gradient_surfaces.py --loss mse
    python3 closure_analysis_muon/muon_gradient_surfaces.py --loss sw
    python3 closure_analysis_muon/muon_gradient_surfaces.py --loss blur
    python3 closure_analysis_muon/muon_gradient_surfaces.py --loss all
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
from tools.losses import (
    blur_mse_loss, make_spectral_weight, DEFAULT_BLUR_SIGMAS,
    sobolev_loss, sobolev_loss_geomean, sobolev_loss_geomean_log1p,
    make_sobolev_weight,
)

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments,
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

PLANES = [0, 1, 2]
K = 10000
N_PROJ = 200
N_SWEEP = 120

FD_EPS_POS = 0.5
FD_EPS_ANGLE = 0.01
FD_EPS_ENERGY = 0.5

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PARAM_ORDER = ['x', 'y', 'z', 'theta', 'phi', 'energy']
PARAM_IDX = {n: i for i, n in enumerate(PARAM_ORDER)}

SWEEP_CONFIGS = {
    'energy': {'range': (100.0, 280.0), 'truth': TRUTH_ENERGY, 'unit': 'MeV', 'eps': FD_EPS_ENERGY},
    'x':      {'range': (-700, -300),   'truth': TRUTH_X,      'unit': 'mm',  'eps': FD_EPS_POS},
    'y':      {'range': (-400, 400),    'truth': TRUTH_Y,      'unit': 'mm',  'eps': FD_EPS_POS},
    'z':      {'range': (-200, 400),    'truth': TRUTH_Z,      'unit': 'mm',  'eps': FD_EPS_POS},
    'theta':  {'range': (0.3, 1.2),     'truth': TRUTH_THETA,  'unit': 'rad', 'eps': FD_EPS_ANGLE},
    'phi':    {'range': (0.8, 2.4),     'truth': TRUTH_PHI,    'unit': 'rad', 'eps': FD_EPS_ANGLE},
}


# =============================================================================
# Main
# =============================================================================

def main(loss_types=None):
    if loss_types is None:
        loss_types = ['mse', 'sw', 'blur']

    print("=" * 60, flush=True)
    print("MUON GRADIENT SURFACE ANALYSIS", flush=True)
    print(f"Loss types: {', '.join(loss_types)}", flush=True)
    print("=" * 60, flush=True)

    # --- Setup ---
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    # --- Truth signals ---
    truth_args = (
        jnp.float32(TRUTH_X), jnp.float32(TRUTH_Y), jnp.float32(TRUTH_Z),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI), jnp.float32(TRUTH_ENERGY),
    )

    def sim_forward(x, y, z, theta, phi, energy):
        pos, de = generate_muon_segments(
            energy, jnp.array([x, y, z]), theta, phi,
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        return forward(pos, de)

    print("Compiling sim_forward...", flush=True)
    t0 = time.time()
    sim_jit = jax.jit(sim_forward)
    truth_signals = sim_jit(*truth_args)
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"  compiled ({time.time()-t0:.1f}s)", flush=True)

    ot_key = jax.random.PRNGKey(42)
    target_sigs = {p: truth_signals[p] for p in PLANES}

    ALL = (0, 1, 2, 3, 4, 5)

    # --- MSE ---
    if 'mse' in loss_types:
        def mse_loss(x, y, z, theta, phi, energy):
            sigs = sim_forward(x, y, z, theta, phi, energy)
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

    # --- SW ---
    if 'sw' in loss_types:
        target_clouds = {}
        for p in PLANES:
            pts, w = signal_to_pointcloud(truth_signals[p], K)
            target_clouds[p] = (pts, w)

        def sw_loss(x, y, z, theta, phi, energy):
            sigs = sim_forward(x, y, z, theta, phi, energy)
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

    # --- Blur MSE (spectral) ---
    if 'blur' in loss_types:
        planes_tuple = tuple(PLANES)

        # Precompute spectral weights for each plane shape
        print("\nPrecomputing spectral weights...", flush=True)
        spec_weights = {}
        for p in PLANES:
            H, W = truth_signals[p].shape
            spec_weights[p] = make_spectral_weight(H, W, DEFAULT_BLUR_SIGMAS)
        # Build full 6-element tuple (only PLANES entries used)
        sw_tuple = tuple(
            spec_weights[i] if i in spec_weights
            else jnp.zeros((1, 1))
            for i in range(6)
        )

        def blur_loss(x, y, z, theta, phi, energy):
            sigs = sim_forward(x, y, z, theta, phi, energy)
            return blur_mse_loss(sigs, truth_signals, sw_tuple, planes=planes_tuple)

        print("Compiling Blur MSE loss...", flush=True)
        t0 = time.time()
        blur_jit = jax.jit(blur_loss)
        _ = blur_jit(*truth_args)
        print(f"  Blur fwd ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        blur_vg = jax.jit(jax.value_and_grad(blur_loss, argnums=ALL))
        _ = blur_vg(*truth_args)
        print(f"  Blur fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

        _run_sweeps('Blur', blur_jit, blur_vg, truth_args)

    # --- Sobolev: shared spectral weights for sum and geomean (s=2.0) ---
    if 'sobolev_sum' in loss_types or 'sobolev_geomean' in loss_types:
        planes_tuple = tuple(PLANES)

        print("\nPrecomputing Sobolev spectral weights (s=2.0)...", flush=True)
        sob_weights = {}
        for p in PLANES:
            H, W = truth_signals[p].shape
            sob_weights[p] = make_sobolev_weight(H, W, s=2.0)
        sob_tuple = tuple(
            sob_weights[i] if i in sob_weights
            else jnp.zeros((1, 1))
            for i in range(6)
        )

    # --- Sobolev Sum ---
    if 'sobolev_sum' in loss_types:
        def sob_sum_loss(x, y, z, theta, phi, energy):
            sigs = sim_forward(x, y, z, theta, phi, energy)
            return sobolev_loss(sigs, truth_signals, sob_tuple, planes=planes_tuple)

        print("\nCompiling Sobolev Sum loss...", flush=True)
        t0 = time.time()
        sob_sum_jit = jax.jit(sob_sum_loss)
        _ = sob_sum_jit(*truth_args)
        print(f"  Sobolev Sum fwd ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        sob_sum_vg = jax.jit(jax.value_and_grad(sob_sum_loss, argnums=ALL))
        _ = sob_sum_vg(*truth_args)
        print(f"  Sobolev Sum fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

        _run_sweeps('Sobolev_Sum', sob_sum_jit, sob_sum_vg, truth_args)

    # --- Sobolev Geomean ---
    if 'sobolev_geomean' in loss_types:
        def sob_geo_loss(x, y, z, theta, phi, energy):
            sigs = sim_forward(x, y, z, theta, phi, energy)
            return sobolev_loss_geomean(sigs, truth_signals, sob_tuple, planes=planes_tuple)

        print("\nCompiling Sobolev Geomean loss...", flush=True)
        t0 = time.time()
        sob_geo_jit = jax.jit(sob_geo_loss)
        _ = sob_geo_jit(*truth_args)
        print(f"  Sobolev Geomean fwd ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        sob_geo_vg = jax.jit(jax.value_and_grad(sob_geo_loss, argnums=ALL))
        _ = sob_geo_vg(*truth_args)
        print(f"  Sobolev Geomean fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

        _run_sweeps('Sobolev_Geomean', sob_geo_jit, sob_geo_vg, truth_args)

    # --- Sobolev Geomean Log1p (s=1.5, matching optimization settings) ---
    if 'sobolev_geomean_log1p' in loss_types:
        planes_tuple = tuple(PLANES)

        print("\nPrecomputing Sobolev spectral weights (s=1.5)...", flush=True)
        sob15_weights = {}
        for p in PLANES:
            H, W = truth_signals[p].shape
            sob15_weights[p] = make_sobolev_weight(H, W, s=1.5)
        sob15_tuple = tuple(
            sob15_weights[i] if i in sob15_weights
            else jnp.zeros((1, 1))
            for i in range(6)
        )

        def sob_log1p_loss(x, y, z, theta, phi, energy):
            sigs = sim_forward(x, y, z, theta, phi, energy)
            return sobolev_loss_geomean_log1p(sigs, truth_signals, sob15_tuple,
                                               planes=planes_tuple)

        print("Compiling Sobolev Geomean Log1p loss...", flush=True)
        t0 = time.time()
        sob_log1p_jit = jax.jit(sob_log1p_loss)
        _ = sob_log1p_jit(*truth_args)
        print(f"  Sobolev Geomean Log1p fwd ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        sob_log1p_vg = jax.jit(jax.value_and_grad(sob_log1p_loss, argnums=ALL))
        _ = sob_log1p_vg(*truth_args)
        print(f"  Sobolev Geomean Log1p fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

        _run_sweeps('Sobolev_Geomean_Log1p', sob_log1p_jit, sob_log1p_vg, truth_args)

    print("\nDone!", flush=True)


def _run_sweeps(loss_name, loss_jit, loss_vg, truth_args):
    """Run parameter sweeps and save plots for one loss type."""

    prefix = {
        'MSE': 'mse', 'SW': 'sw', 'Blur': 'blur',
        'Sobolev_Sum': 'sobolev_sum_s2.0_smooth0.2',
        'Sobolev_Geomean': 'sobolev_geomean_s2.0_smooth0.2',
        'Sobolev_Geomean_Log1p': 'sobolev_geomean_log1p_s1.5_smooth0.2',
    }[loss_name]

    for pname, cfg in SWEEP_CONFIGS.items():
        lo, hi = cfg['range']
        truth_val = cfg['truth']
        unit = cfg['unit']
        eps = cfg['eps']
        pidx = PARAM_IDX[pname]

        x_vals = np.linspace(lo, hi, N_SWEEP)
        print(f"\n  {loss_name} — {pname} [{lo}, {hi}] {unit}", flush=True)

        def _args(val):
            a = list(truth_args)
            a[pidx] = jnp.float32(val)
            return tuple(a)

        # Loss + autodiff gradient
        t0 = time.time()
        losses = np.empty(N_SWEEP)
        ad_grads = np.empty(N_SWEEP)
        for i, v in enumerate(x_vals):
            loss, grads = loss_vg(*_args(v))
            losses[i] = float(loss)
            ad_grads[i] = float(grads[pidx])

        # Finite-diff gradient
        fd_grads = np.array([
            (float(loss_jit(*_args(v + eps)))
             - float(loss_jit(*_args(v - eps)))) / (2 * eps)
            for v in x_vals
        ])
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)

        # --- Plot ---
        truth_label = f'Truth {pname}={truth_val:.2f}'
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(x_vals, losses, 'b-', lw=1.5)
        axes[0].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
        axes[0].set_xlabel(f'{pname} ({unit})')
        axes[0].set_ylabel(f'{loss_name} Loss')
        axes[0].set_title('Loss Landscape')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_vals, ad_grads, 'r-', lw=1.5, label='Autodiff')
        axes[1].plot(x_vals, fd_grads, 'b--', lw=1.5, alpha=0.7, label=f'FD (eps={eps})')
        axes[1].axhline(0, color='k', ls='-', lw=0.5)
        axes[1].axvline(truth_val, color='green', ls='--', lw=2, label=truth_label)
        axes[1].set_xlabel(f'{pname} ({unit})')
        axes[1].set_ylabel(f'd{loss_name}/d{pname}')
        axes[1].set_title('Gradient (Autodiff vs FD)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f'{loss_name} — varying {pname} (muon, {N_SEGMENTS} seg, summed U+V+Y)',
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f'{prefix}_surface_{pname}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Muon gradient surface analysis')
    parser.add_argument(
        '--loss', type=str, default='all',
        choices=['mse', 'sw', 'blur', 'sobolev_sum', 'sobolev_geomean',
                 'sobolev_geomean_log1p', 'all'],
        help='Which loss to run (default: all)',
    )
    args = parser.parse_args()
    if args.loss == 'all':
        main(['mse', 'sw', 'blur', 'sobolev_sum', 'sobolev_geomean',
              'sobolev_geomean_log1p'])
    else:
        main([args.loss])
