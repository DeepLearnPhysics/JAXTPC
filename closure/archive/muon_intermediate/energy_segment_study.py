"""
Energy gradient surface vs segment size.

Tests whether the seesaw in energy autodiff gradients is caused by
coarse segment discretization in the lax.scan dE/dx chain.

Run from project root:
    python3 closure_analysis_muon/energy_segment_study.py
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
from tools.losses import blur_mse_loss, make_spectral_weight, DEFAULT_BLUR_SIGMAS

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments,
    build_muon_forward,
)

# ── Config ──
TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 2
TRUTH_ENERGY = 200.0
PLANES = [0, 1, 2]
N_SWEEP = 40
ENERGY_RANGE = (100.0, 280.0)
FD_EPS = 0.5

# Segment configs: (step_size_mm, n_segments)
# 200 MeV muon range ~950mm, so need enough segments to cover it
SEGMENT_CONFIGS = [
    (1.0,  1400),   # coarse
    (0.5,  2800),   # current default
    (0.25, 5600),   # fine
    (0.1,  14000),  # very fine
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_energy_sweep(step_size, n_seg, log_T, dedx):
    """Run energy sweep for one segment configuration."""
    print(f"\n{'='*50}")
    print(f"Step size: {step_size}mm, N segments: {n_seg}")
    print(f"{'='*50}")

    # Build simulator
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=n_seg)
    forward = build_muon_forward(sim, n_seg, step_size)

    def sim_forward(x, y, z, theta, phi, energy):
        pos, de = generate_muon_segments(
            energy, jnp.array([x, y, z]), theta, phi,
            step_size, n_seg, log_T, dedx,
        )
        return forward(pos, de)

    truth_args = (
        jnp.float32(TRUTH_X), jnp.float32(TRUTH_Y), jnp.float32(TRUTH_Z),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI), jnp.float32(TRUTH_ENERGY),
    )

    # Compile sim
    print("Compiling sim...", flush=True)
    t0 = time.time()
    sim_jit = jax.jit(sim_forward)
    truth_signals = sim_jit(*truth_args)
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"  sim compiled ({time.time()-t0:.1f}s)", flush=True)

    # Precompute spectral weights
    planes_tuple = tuple(PLANES)
    sw_tuple = tuple(
        make_spectral_weight(*truth_signals[i].shape, DEFAULT_BLUR_SIGMAS)
        if i in PLANES else jnp.zeros((1, 1))
        for i in range(6)
    )

    def blur_loss(x, y, z, theta, phi, energy):
        sigs = sim_forward(x, y, z, theta, phi, energy)
        return blur_mse_loss(sigs, truth_signals, sw_tuple, planes=planes_tuple)

    # Compile loss + grad
    print("Compiling loss+grad...", flush=True)
    ALL = (0, 1, 2, 3, 4, 5)
    blur_jit = jax.jit(blur_loss)
    blur_vg = jax.jit(jax.value_and_grad(blur_loss, argnums=ALL))

    t0 = time.time()
    _ = blur_jit(*truth_args)
    print(f"  fwd ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    _ = blur_vg(*truth_args)
    print(f"  fwd+bwd ({time.time()-t0:.1f}s)", flush=True)

    # Energy sweep
    e_vals = np.linspace(*ENERGY_RANGE, N_SWEEP)
    losses = np.empty(N_SWEEP)
    ad_grads = np.empty(N_SWEEP)

    print("Sweeping energy...", flush=True)
    t0 = time.time()
    for i, e in enumerate(e_vals):
        a = list(truth_args)
        a[5] = jnp.float32(e)
        a = tuple(a)
        loss, grads = blur_vg(*a)
        losses[i] = float(loss)
        ad_grads[i] = float(grads[5])  # energy is index 5

    fd_grads = np.array([
        (float(blur_jit(*_args(truth_args, e + FD_EPS)))
         - float(blur_jit(*_args(truth_args, e - FD_EPS)))) / (2 * FD_EPS)
        for e in e_vals
    ])
    print(f"  done ({time.time()-t0:.1f}s)", flush=True)

    return e_vals, losses, ad_grads, fd_grads


def _args(truth_args, energy):
    a = list(truth_args)
    a[5] = jnp.float32(energy)
    return tuple(a)


def main():
    log_T, dedx = load_dedx_table_jax()

    results = {}
    for step_size, n_seg in SEGMENT_CONFIGS:
        label = f"{step_size}mm ({n_seg} seg)"
        e_vals, losses, ad_grads, fd_grads = run_energy_sweep(
            step_size, n_seg, log_T, dedx,
        )
        results[label] = (e_vals, losses, ad_grads, fd_grads)

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['C0', 'C1', 'C2', 'C3']

    # Loss landscapes
    ax = axes[0, 0]
    for (label, (e, l, _, _)), c in zip(results.items(), colors):
        ax.plot(e, l, color=c, lw=1.5, label=label)
    ax.axvline(TRUTH_ENERGY, color='green', ls='--', lw=2)
    ax.set_xlabel('energy (MeV)')
    ax.set_ylabel('Blur Loss')
    ax.set_title('Loss Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Autodiff gradients
    ax = axes[0, 1]
    for (label, (e, _, ag, _)), c in zip(results.items(), colors):
        ax.plot(e, ag, color=c, lw=1.5, label=label)
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.axvline(TRUTH_ENERGY, color='green', ls='--', lw=2)
    ax.set_xlabel('energy (MeV)')
    ax.set_ylabel('dBlur/denergy (autodiff)')
    ax.set_title('Autodiff Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FD gradients
    ax = axes[1, 0]
    for (label, (e, _, _, fg)), c in zip(results.items(), colors):
        ax.plot(e, fg, color=c, lw=1.5, label=label)
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.axvline(TRUTH_ENERGY, color='green', ls='--', lw=2)
    ax.set_xlabel('energy (MeV)')
    ax.set_ylabel('dBlur/denergy (FD)')
    ax.set_title('Finite-Difference Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Autodiff vs FD overlay for finest
    ax = axes[1, 1]
    finest_label = list(results.keys())[-1]
    e, _, ag, fg = results[finest_label]
    ax.plot(e, ag, 'r-', lw=1.5, label='Autodiff')
    ax.plot(e, fg, 'b--', lw=1.5, alpha=0.7, label='FD')
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.axvline(TRUTH_ENERGY, color='green', ls='--', lw=2)
    ax.set_xlabel('energy (MeV)')
    ax.set_ylabel('dBlur/denergy')
    ax.set_title(f'Autodiff vs FD — {finest_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Energy Gradient vs Segment Size (muon, summed U+V+Y)',
        fontsize=14, fontweight='bold',
    )
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'energy_segment_study.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved {fname}")


if __name__ == '__main__':
    main()
