"""Per-plane Sobolev gradient surfaces.

Shows loss and gradient for each plane (U, V, Y) individually plus the sum.
Sim is compiled once; per-plane losses are thin wrappers.

Run from project root:
    python3 closure_analysis_muon/muon_gradient_surfaces_perplane.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss, make_sobolev_weight

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
PLANE_NAMES = {0: 'U', 1: 'V', 2: 'Y'}
PLANE_COLORS = {0: 'C0', 1: 'C1', 2: 'C2'}
N_SWEEP = 40
FD_EPS_POS = 0.5
FD_EPS_ANGLE = 0.01
FD_EPS_ENERGY = 0.5

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PARAM_ORDER = ['x', 'y', 'z', 'theta', 'phi', 'energy']
PARAM_IDX = {n: i for i, n in enumerate(PARAM_ORDER)}

SWEEP_CONFIGS = {
    'x':      {'range': (-700, -300),   'truth': TRUTH_X,      'unit': 'mm',  'eps': FD_EPS_POS},
    'y':      {'range': (-400, 400),    'truth': TRUTH_Y,      'unit': 'mm',  'eps': FD_EPS_POS},
    'z':      {'range': (-200, 400),    'truth': TRUTH_Z,      'unit': 'mm',  'eps': FD_EPS_POS},
    'theta':  {'range': (0.3, 1.2),     'truth': TRUTH_THETA,  'unit': 'rad', 'eps': FD_EPS_ANGLE},
    'phi':    {'range': (0.8, 2.4),     'truth': TRUTH_PHI,    'unit': 'rad', 'eps': FD_EPS_ANGLE},
    'energy': {'range': (100.0, 280.0), 'truth': TRUTH_ENERGY, 'unit': 'MeV', 'eps': FD_EPS_ENERGY},
}

ALL = (0, 1, 2, 3, 4, 5)


def main():
    print("=" * 60, flush=True)
    print("PER-PLANE SOBOLEV GRADIENT SURFACES", flush=True)
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
    print(f"  sim compiled ({time.time()-t0:.1f}s)", flush=True)

    # --- Sobolev weights ---
    print("\nPrecomputing Sobolev weights...", flush=True)
    sob_weights = {}
    for p in PLANES:
        H, W = truth_signals[p].shape
        sob_weights[p] = make_sobolev_weight(H, W)
    sob_tuple = tuple(
        sob_weights[i] if i in sob_weights else jnp.zeros((1, 1))
        for i in range(6)
    )

    # --- Per-plane loss functions (thin wrappers over same sim) ---
    def make_loss(plane_tuple):
        def loss_fn(x, y, z, theta, phi, energy):
            sigs = sim_forward(x, y, z, theta, phi, energy)
            return sobolev_loss(sigs, truth_signals, sob_tuple, planes=plane_tuple)
        return loss_fn

    # Compile per-plane and sum
    print("Compiling per-plane losses...", flush=True)
    loss_fns = {}   # name -> (jit_loss, jit_vg)
    for p in PLANES:
        name = PLANE_NAMES[p]
        fn = make_loss((p,))
        t0 = time.time()
        fn_jit = jax.jit(fn)
        vg_jit = jax.jit(jax.value_and_grad(fn, argnums=ALL))
        _ = fn_jit(*truth_args)
        _ = vg_jit(*truth_args)
        loss_fns[name] = (fn_jit, vg_jit)
        print(f"  {name} compiled ({time.time()-t0:.1f}s)", flush=True)

    # Sum
    fn_sum = make_loss(tuple(PLANES))
    t0 = time.time()
    fn_sum_jit = jax.jit(fn_sum)
    vg_sum_jit = jax.jit(jax.value_and_grad(fn_sum, argnums=ALL))
    _ = fn_sum_jit(*truth_args)
    _ = vg_sum_jit(*truth_args)
    loss_fns['Sum'] = (fn_sum_jit, vg_sum_jit)
    print(f"  Sum compiled ({time.time()-t0:.1f}s)", flush=True)

    # --- Sweeps ---
    ordered = ['U', 'V', 'Y', 'Sum']
    colors = {'U': 'C0', 'V': 'C1', 'Y': 'C2', 'Sum': 'k'}
    lws = {'U': 1.5, 'V': 1.5, 'Y': 1.5, 'Sum': 2.5}

    for pname, cfg in SWEEP_CONFIGS.items():
        lo, hi = cfg['range']
        truth_val = cfg['truth']
        unit = cfg['unit']
        eps = cfg['eps']
        pidx = PARAM_IDX[pname]

        x_vals = np.linspace(lo, hi, N_SWEEP)
        print(f"\n  {pname} [{lo}, {hi}] {unit}", flush=True)

        def _args(val):
            a = list(truth_args)
            a[pidx] = jnp.float32(val)
            return tuple(a)

        # Collect per-plane data
        t0 = time.time()
        data = {}
        for name in ordered:
            fn_jit, vg_jit = loss_fns[name]
            losses = np.empty(N_SWEEP)
            ad_grads = np.empty(N_SWEEP)
            for i, v in enumerate(x_vals):
                loss, grads = vg_jit(*_args(v))
                losses[i] = float(loss)
                ad_grads[i] = float(grads[pidx])
            data[name] = (losses, ad_grads)
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)

        # --- Plot: 2 panels ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Left: loss per plane
        for name in ordered:
            losses, _ = data[name]
            axes[0].plot(x_vals, losses, color=colors[name], lw=lws[name],
                         ls='-' if name != 'Sum' else '-',
                         alpha=0.8 if name != 'Sum' else 1.0,
                         label=name)
        axes[0].axvline(truth_val, color='green', ls='--', lw=2,
                         label=f'Truth={truth_val:.2f}')
        axes[0].set_xlabel(f'{pname} ({unit})')
        axes[0].set_ylabel('Sobolev Loss')
        axes[0].set_title('Loss per Plane')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: gradient per plane
        for name in ordered:
            _, ad_grads = data[name]
            axes[1].plot(x_vals, ad_grads, color=colors[name], lw=lws[name],
                         alpha=0.8 if name != 'Sum' else 1.0,
                         label=name)
        axes[1].axhline(0, color='k', ls='-', lw=0.5)
        axes[1].axvline(truth_val, color='green', ls='--', lw=2,
                         label=f'Truth={truth_val:.2f}')
        axes[1].set_xlabel(f'{pname} ({unit})')
        axes[1].set_ylabel(f'dLoss/d{pname}')
        axes[1].set_title('Gradient per Plane')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f'Sobolev per-plane — varying {pname} (U, V, Y + Sum)',
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f'sobolev_perplane_{pname}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
