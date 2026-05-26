"""Lossslide 4: Real muon event — truth vs displaced sim, MSE vs Sobolev view.

Generates a truth muon and a displaced sim muon through the full detector
simulation, then shows the raw difference (what MSE sees) vs the Sobolev-
convolved difference (what the Sobolev loss sees) for one wire plane.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
import time

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments,
    build_muon_forward,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SEGMENTS = 2800
STEP_SIZE_MM = 0.5
PLANE = 2  # Y-plane for clearest display

# Truth muon
TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 2
TRUTH_ENERGY = 200.0

# Displaced muon (shift position and angle)
SIM_X, SIM_Y, SIM_Z = -400.0, 80.0, 0.0
SIM_THETA, SIM_PHI = TRUTH_THETA + 0.15, TRUTH_PHI - 0.1
SIM_ENERGY = 220.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def convolve_sobolev_2d(diff_2d, s, max_pad):
    """Convolve 2D difference with Sobolev kernel K_hat = 1/(|f|^2+eps)^{s/2}."""
    H, W = diff_2d.shape
    H_pad, W_pad = H + 2 * max_pad, W + 2 * max_pad
    fy = np.fft.fftfreq(H_pad)
    fx = np.fft.fftfreq(W_pad)
    freq_sq = fy[:, None]**2 + fx[None, :]**2
    eps = 1.0 / (np.pi**2 * max_pad**2)
    K_hat = 1.0 / (freq_sq + eps)**(s / 2.0)
    diff_pad = np.pad(diff_2d, max_pad)
    convolved_pad = np.fft.ifft2(np.fft.fft2(diff_pad) * K_hat).real
    return convolved_pad[max_pad:max_pad + H, max_pad:max_pad + W]


def main():
    print("Setting up detector...", flush=True)
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    def make_signals(x, y, z, theta, phi, energy):
        pos, de = generate_muon_segments(
            energy, jnp.array([x, y, z]), theta, phi,
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        return forward(pos, de)

    make_signals_jit = jax.jit(make_signals)

    # Generate truth
    print("Generating truth signals...", flush=True)
    t0 = time.time()
    truth_sigs = make_signals_jit(
        jnp.float32(TRUTH_X), jnp.float32(TRUTH_Y), jnp.float32(TRUTH_Z),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI), jnp.float32(TRUTH_ENERGY))
    for s in truth_sigs:
        jax.block_until_ready(s)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # Generate displaced sim
    print("Generating displaced sim signals...", flush=True)
    t0 = time.time()
    sim_sigs = make_signals_jit(
        jnp.float32(SIM_X), jnp.float32(SIM_Y), jnp.float32(SIM_Z),
        jnp.float32(SIM_THETA), jnp.float32(SIM_PHI), jnp.float32(SIM_ENERGY))
    for s in sim_sigs:
        jax.block_until_ready(s)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    truth_plane = np.array(truth_sigs[PLANE])
    sim_plane = np.array(sim_sigs[PLANE])
    diff = sim_plane - truth_plane

    # Crop to region of interest (nonzero region with margin)
    nonzero_mask = (np.abs(truth_plane) + np.abs(sim_plane)) > 0
    rows = np.any(nonzero_mask, axis=1)
    cols = np.any(nonzero_mask, axis=0)
    r_lo, r_hi = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c_lo, c_hi = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    margin = 30
    r_lo = max(0, r_lo - margin)
    r_hi = min(truth_plane.shape[0], r_hi + margin)
    c_lo = max(0, c_lo - margin)
    c_hi = min(truth_plane.shape[1], c_hi + margin)

    truth_crop = truth_plane[r_lo:r_hi, c_lo:c_hi]
    sim_crop = sim_plane[r_lo:r_hi, c_lo:c_hi]
    diff_crop = diff[r_lo:r_hi, c_lo:c_hi]

    # Convolve full arrays then crop (avoids edge artifacts)
    max_pad = 256
    conv_15_full = convolve_sobolev_2d(diff, s=1.5, max_pad=max_pad)
    conv_20_full = convolve_sobolev_2d(diff, s=2.0, max_pad=max_pad)
    conv_15_crop = conv_15_full[r_lo:r_hi, c_lo:c_hi]
    conv_20_crop = conv_20_full[r_lo:r_hi, c_lo:c_hi]

    print(f"Plane shape: {truth_plane.shape}, crop: [{r_lo}:{r_hi}, {c_lo}:{c_hi}]")

    # -----------------------------------------------------------------------
    # Figure: 2 rows x 3 cols
    # Row 1: Truth | Sim | Difference
    # Row 2: K_{1.5} * Diff | K_{2} * Diff | (empty or annotation)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    sig_vmax = max(np.max(np.abs(truth_crop)), np.max(np.abs(sim_crop)))

    # Row 1: signals and raw difference
    im0 = axes[0, 0].imshow(truth_crop.T, cmap='inferno', origin='lower',
                             aspect='auto', vmin=0, vmax=sig_vmax)
    axes[0, 0].set_title('Truth')

    im1 = axes[0, 1].imshow(sim_crop.T, cmap='inferno', origin='lower',
                             aspect='auto', vmin=0, vmax=sig_vmax)
    axes[0, 1].set_title('Sim (displaced)')

    vmax_d = np.max(np.abs(diff_crop))
    norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
    im2 = axes[0, 2].imshow(diff_crop.T, cmap='RdBu_r', origin='lower',
                             aspect='auto', norm=norm_d)
    axes[0, 2].set_title('Difference  (MSE input)')
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.75, pad=0.02)

    # Row 2: convolved differences
    vmax_15 = np.max(np.abs(conv_15_crop))
    norm_15 = TwoSlopeNorm(vmin=-vmax_15, vcenter=0, vmax=vmax_15)
    im3 = axes[1, 0].imshow(conv_15_crop.T, cmap='RdBu_r', origin='lower',
                             aspect='auto', norm=norm_15)
    axes[1, 0].set_title(r'$K_{1.5} \ast$ Diff  (Sobolev $s\!=\!1.5$ input)')
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.75, pad=0.02)

    vmax_20 = np.max(np.abs(conv_20_crop))
    norm_20 = TwoSlopeNorm(vmin=-vmax_20, vcenter=0, vmax=vmax_20)
    im4 = axes[1, 1].imshow(conv_20_crop.T, cmap='RdBu_r', origin='lower',
                             aspect='auto', norm=norm_20)
    axes[1, 1].set_title(r'$K_{2} \ast$ Diff  (Sobolev $s\!=\!2$ input)')
    fig.colorbar(im4, ax=axes[1, 1], shrink=0.75, pad=0.02)

    # Bottom-right: annotation panel
    axes[1, 2].axis('off')
    info = (
        "Y-plane event display\n\n"
        f"Truth: ({TRUTH_X:.0f}, {TRUTH_Y:.0f}, {TRUTH_Z:.0f}) mm\n"
        f"  $\\theta$={np.degrees(TRUTH_THETA):.0f}°, "
        f"$\\phi$={np.degrees(TRUTH_PHI):.0f}°, "
        f"E={TRUTH_ENERGY:.0f} MeV\n\n"
        f"Sim:   ({SIM_X:.0f}, {SIM_Y:.0f}, {SIM_Z:.0f}) mm\n"
        f"  $\\theta$={np.degrees(SIM_THETA):.0f}°, "
        f"$\\phi$={np.degrees(SIM_PHI):.0f}°, "
        f"E={SIM_ENERGY:.0f} MeV\n\n"
        "The Sobolev kernel smears\n"
        "the difference signal across\n"
        "the gap between truth and sim,\n"
        "providing gradient information\n"
        "even where features don't overlap."
    )
    axes[1, 2].text(0.05, 0.95, info, transform=axes[1, 2].transAxes,
                    fontsize=14, va='top', ha='left', family='serif',
                    bbox=dict(boxstyle='round,pad=0.5', fc='#f7f7f7', ec='0.7'))

    for ax in axes.flat:
        if ax.images:
            ax.set_xlabel('Wire')
            ax.set_ylabel('Time Tick')

    fig.suptitle('Muon Event: What MSE vs Sobolev Loss "See"',
                 fontsize=20, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'lossslide4_event_convolution.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
