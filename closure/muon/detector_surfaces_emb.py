"""
Loss and gradient surfaces for detector physics parameters with EMB recombination.

Same as detector_surfaces.py but passes real track angles (theta, phi)
per segment so the EMB angular-dependent recombination is fully exercised.
Includes R_anisotropy in the sweep parameters.

Run from project root:
    python3 closure/muon/detector_surfaces_emb.py
"""

import os, time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.particle_generator import (
    load_dedx_table_jax, generate_muon_segments_trig, mask_outside_volume,
)
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight

# =============================================================================
# Configuration
# =============================================================================

N_SEGMENTS = 4000
STEP_SIZE_MM = 0.5
HALF_EXT = (2160.0, 2160.0, 2160.0)
N_SWEEP = 40
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Truth muon parameters
TRUTH_X, TRUTH_Y, TRUTH_Z = -200.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 6
TRUTH_ENERGY = 500.0

# Detector parameter sweep configs
DETECTOR_PARAMS = {
    'diffusion_trans': {
        'range': (4.0e-6, 24.0e-6),
        'unit': 'cm2/us',
        'fd_eps': 1.0e-7,
        'getter': lambda p: p.diffusion_trans_cm2_us,
        'setter': lambda p, v: p._replace(diffusion_trans_cm2_us=v),
    },
    'diffusion_long': {
        'range': (2.0e-6, 14.0e-6),
        'unit': 'cm2/us',
        'fd_eps': 5.0e-8,
        'getter': lambda p: p.diffusion_long_cm2_us,
        'setter': lambda p, v: p._replace(diffusion_long_cm2_us=v),
    },
    'velocity': {
        'range': (0.10, 0.22),
        'unit': 'cm/us',
        'fd_eps': 0.001,
        'getter': lambda p: p.velocity_cm_us,
        'setter': lambda p, v: p._replace(velocity_cm_us=v),
    },
    'lifetime': {
        'range': (5000.0, 20000.0),
        'unit': 'us',
        'fd_eps': 10.0,
        'getter': lambda p: p.lifetime_us,
        'setter': lambda p, v: p._replace(lifetime_us=v),
    },
    'recomb_alpha': {
        'range': (0.5, 1.5),
        'unit': '',
        'fd_eps': 0.005,
        'getter': lambda p: p.recomb_params.alpha,
        'setter': lambda p, v: p._replace(
            recomb_params=p.recomb_params._replace(alpha=v)),
    },
    'recomb_beta_90': {
        'range': (0.1, 0.5),
        'unit': '',
        'fd_eps': 0.002,
        'getter': lambda p: p.recomb_params.beta_90,
        'setter': lambda p, v: p._replace(
            recomb_params=p.recomb_params._replace(beta_90=v)),
    },
    'recomb_R': {
        'range': (0.5, 2.5),
        'unit': '',
        'fd_eps': 0.005,
        'getter': lambda p: p.recomb_params.R,
        'setter': lambda p, v: p._replace(
            recomb_params=p.recomb_params._replace(R=v)),
    },
}


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("DETECTOR PARAMETER GRADIENT SURFACES (EMB + angles)")
    print("=" * 60)

    # --- Setup ---
    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(
        det, differentiable=True, n_segments=N_SEGMENTS,
        include_noise=False, include_electronics=False,
        include_track_hits=False,
    )
    truth_params = sim.default_sim_params

    # --- Generate fixed muon track ---
    pos, de = generate_muon_segments_trig(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], dtype=jnp.float32),
        jnp.float32(jnp.sin(TRUTH_THETA)), jnp.float32(jnp.cos(TRUTH_THETA)),
        jnp.float32(jnp.sin(TRUTH_PHI)), jnp.float32(jnp.cos(TRUTH_PHI)),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    )
    de = mask_outside_volume(pos, de, HALF_EXT)
    n_active = int((de > 0).sum())
    print(f"Muon: {N_SEGMENTS} segments, {n_active} active")
    print(f"Track angles: theta={TRUTH_THETA:.4f} rad, phi={TRUTH_PHI:.4f} rad\n")

    # --- Truth signals ---
    print("Computing truth signals...", flush=True)
    t0 = time.time()
    truth_sigs = jax.jit(
        lambda: sim.forward_segments(truth_params, pos, de, STEP_SIZE_MM)
    )()
    for s in truth_sigs:
        jax.block_until_ready(s)
    print(f"  Done ({time.time() - t0:.1f}s)\n")

    # --- Spectral weights ---
    spec_w = tuple(
        make_sobolev_weight(*truth_sigs[p].shape, s=1.5) for p in range(6)
    )

    # --- Loss function (parameterized by single scalar) ---
    def make_loss_fn(param_name):
        cfg = DETECTOR_PARAMS[param_name]
        setter = cfg['setter']

        def loss_fn(val):
            p = setter(truth_params, val)
            sigs = sim.forward_segments(p, pos, de, STEP_SIZE_MM)
            return sobolev_loss_geomean_log1p(sigs, truth_sigs, spec_w)

        return loss_fn

    # --- Sweep each parameter ---
    for param_name, cfg in DETECTOR_PARAMS.items():
        lo, hi = cfg['range']
        unit = cfg['unit']
        fd_eps = cfg['fd_eps']
        truth_val = float(cfg['getter'](truth_params))

        print(f"Sweeping {param_name} [{lo:.4g}, {hi:.4g}] (truth={truth_val:.4g})")

        loss_fn = make_loss_fn(param_name)

        # Compile
        t0 = time.time()
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        loss_jit = jax.jit(loss_fn)
        _ = loss_and_grad(jnp.float32(truth_val))
        print(f"  Compiled ({time.time() - t0:.1f}s)")

        # Sweep
        x_vals = np.linspace(lo, hi, N_SWEEP)
        losses = np.empty(N_SWEEP)
        ad_grads = np.empty(N_SWEEP)
        fd_grads = np.empty(N_SWEEP)

        t0 = time.time()
        for i, v in enumerate(x_vals):
            val = jnp.float32(v)
            loss, grad = loss_and_grad(val)
            losses[i] = float(loss)
            ad_grads[i] = float(grad)

            fd = (float(loss_jit(jnp.float32(v + fd_eps)))
                  - float(loss_jit(jnp.float32(v - fd_eps)))) / (2 * fd_eps)
            fd_grads[i] = fd

        print(f"  Sweep done ({time.time() - t0:.1f}s)")

        # --- Plot ---
        unit_str = f' ({unit})' if unit else ''
        truth_label = f'Truth = {truth_val:.4g}'
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].plot(x_vals, losses, 'b-', lw=1.5)
        axes[0].axvline(truth_val, color='green', ls='--', lw=2,
                        label=truth_label)
        axes[0].set_xlabel(f'{param_name}{unit_str}', fontsize=12)
        axes[0].set_ylabel('Sobolev Geomean Log1p Loss', fontsize=12)
        axes[0].set_title('Loss Landscape', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_vals, ad_grads, 'r-', lw=1.5, label='Autodiff')
        axes[1].plot(x_vals, fd_grads, 'b--', lw=1.5, alpha=0.7,
                     label=f'FD (eps={fd_eps:.2g})')
        axes[1].axhline(0, color='k', ls='-', lw=0.5)
        axes[1].axvline(truth_val, color='green', ls='--', lw=2,
                        label=truth_label)
        axes[1].set_xlabel(f'{param_name}{unit_str}', fontsize=12)
        axes[1].set_ylabel(f'dLoss/d({param_name})', fontsize=12)
        axes[1].set_title('Gradient (Autodiff vs FD)', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f'Detector param: {param_name} — muon {N_SEGMENTS} seg, '
            f'EMB recomb, Sobolev geomean log1p',
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f'detector_emb_surface_{param_name}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}\n")

    print("Done!")


if __name__ == '__main__':
    main()
