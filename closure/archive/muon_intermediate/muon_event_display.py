"""
Muon event display and parameter variation visualization.

Generates:
  1. muon_event_display.png         - 6-panel wire signal display (all planes)
  2. muon_variation_{param}.png     - truth vs perturbed signal for each parameter
  3. muon_track_3d.png              - 3D scatter of the generated muon segments

Run from project root:
    python3 closure_analysis_muon/muon_event_display.py

Design choices
--------------
- n_segments = 500, step_size = 3 mm (0.3 cm)
- Default muon: 200 MeV, start (-500, 0, 100) mm,
  theta = pi/4, phi = pi/2  (diagonal in y-z plane, deep in east side)
- Response path only: no noise, no electronics, no E-field distortions
- Visualization uses the tools.visualization module for consistent style
- Parameter perturbations chosen to give visible but small signal changes
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
from tools.visualization import visualize_wire_signals

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments,
    build_muon_forward,
)

# =============================================================================
# Configuration
# =============================================================================

N_SEGMENTS = 2800
STEP_SIZE_MM = 0.5    # 0.05 cm per step

# Truth muon parameters
TRUTH_X = -500.0      # mm (deep in east side, track stays east for all angle sweeps)
TRUTH_Y = 0.0         # mm
TRUTH_Z = 100.0       # mm
TRUTH_THETA = np.pi / 4   # 45 deg from z-axis
TRUTH_PHI = np.pi / 2     # in y-z plane (no x component)
TRUTH_ENERGY = 200.0       # MeV

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Perturbation sizes for parameter variation plots
PERTURBATIONS = {
    'x':      10.0,     # mm
    'y':      10.0,     # mm
    'z':      10.0,     # mm
    'theta':  0.1,      # rad (~5.7 deg)
    'phi':    0.1,      # rad
    'energy': 20.0,     # MeV
}


# =============================================================================
# Helpers
# =============================================================================

def make_muon_params(x=TRUTH_X, y=TRUTH_Y, z=TRUTH_Z,
                     theta=TRUTH_THETA, phi=TRUTH_PHI, energy=TRUTH_ENERGY):
    """Pack muon parameters as JAX scalars/arrays."""
    return (
        jnp.float32(energy),
        jnp.array([x, y, z], dtype=jnp.float32),
        jnp.float32(theta),
        jnp.float32(phi),
    )


def simulate_muon(energy, pos, theta, phi, forward, log_T, dedx):
    """Generate segments and run detector simulation."""
    positions, des = generate_muon_segments(
        energy, pos, theta, phi,
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    )
    signals = forward(positions, des)
    return signals, positions, des


def signals_to_dict(signals):
    """Convert 6-tuple to (side, plane) dict for visualization."""
    d = {}
    for side in range(2):
        for plane in range(3):
            d[(side, plane)] = signals[side * 3 + plane]
    return d


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("MUON EVENT DISPLAY AND PARAMETER VARIATIONS")
    print("=" * 60)
    print(f"n_segments={N_SEGMENTS}, step_size={STEP_SIZE_MM} mm")
    print(f"Truth: E={TRUTH_ENERGY} MeV, pos=({TRUTH_X},{TRUTH_Y},{TRUTH_Z}) mm")
    print(f"       theta={TRUTH_THETA:.4f} rad, phi={TRUTH_PHI:.4f} rad")

    # --- Load PDG table and detector ---
    print("\nLoading PDG dE/dx table...")
    log_T, dedx = load_dedx_table_jax()

    print("Loading detector config...")
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')

    print("Creating differentiable simulator...")
    sim = DetectorSimulator(
        detector_config,
        differentiable=True,
        n_segments=N_SEGMENTS,
    )

    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    # --- Generate truth muon ---
    print("Generating truth muon track...")
    energy, pos, theta, phi = make_muon_params()
    t0 = time.time()
    truth_signals, truth_pos, truth_de = simulate_muon(
        energy, pos, theta, phi, forward, log_T, dedx,
    )
    # Block until ready
    for s in truth_signals:
        jax.block_until_ready(s)
    t1 = time.time()

    n_active = int(jnp.sum(truth_de > 0))
    total_de = float(jnp.sum(truth_de))
    track_len_cm = n_active * STEP_SIZE_MM / 10.0
    print(f"  Active segments: {n_active}/{N_SEGMENTS}")
    print(f"  Total dE: {total_de:.1f} MeV")
    print(f"  Track length: {track_len_cm:.1f} cm")
    print(f"  First forward pass: {t1 - t0:.2f}s (includes JIT)")

    # =========================================================================
    # 1. Event display (6 planes)
    # =========================================================================
    print("\nGenerating event display...")
    wire_dict = signals_to_dict(truth_signals)
    fig = visualize_wire_signals(
        wire_dict, detector_config,
        sparse_data=False, threshold_enc=0, gamma=0.3,
    )
    fig.suptitle(
        f'Muon Event Display — {TRUTH_ENERGY:.0f} MeV, '
        f'{n_active} segments, track {track_len_cm:.0f} cm',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fname = os.path.join(OUT_DIR, 'muon_event_display.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {fname}")

    # =========================================================================
    # 1b. Zoomed event display (east-side only, cropped to signal region)
    # =========================================================================
    print("Generating zoomed event display...")
    plane_names = ['1st Induction (U)', '2nd Induction (V)', 'Collection (Y)']
    fig_zoom, axes_zoom = plt.subplots(1, 3, figsize=(18, 5))
    for pi in range(3):
        sig_np = np.array(truth_signals[pi])  # already in ADC (kernel includes conversion)
        nz = np.nonzero(np.abs(sig_np) > 0.01)
        if len(nz[0]) > 0:
            wlo = max(0, nz[0].min() - 20)
            whi = min(sig_np.shape[0], nz[0].max() + 20)
            tlo = max(0, nz[1].min() - 40)
            thi = min(sig_np.shape[1], nz[1].max() + 40)
        else:
            wlo, whi = 0, sig_np.shape[0]
            tlo, thi = 0, sig_np.shape[1]
        crop = sig_np[wlo:whi, tlo:thi]
        vabs = np.abs(crop).max()
        if vabs < 1e-10:
            vabs = 1.0
        im = axes_zoom[pi].imshow(
            crop.T, aspect='auto', origin='lower',
            extent=[wlo, whi, tlo, thi],
            cmap='RdBu_r', vmin=-vabs, vmax=vabs,
            interpolation='nearest',
        )
        axes_zoom[pi].set_title(f'East {plane_names[pi]}')
        axes_zoom[pi].set_xlabel('Wire index')
        axes_zoom[pi].set_ylabel('Time bin')
        fig_zoom.colorbar(im, ax=axes_zoom[pi], label='ADC', shrink=0.8)

    fig_zoom.suptitle(
        f'Muon Event Display (Zoomed) — {TRUTH_ENERGY:.0f} MeV, '
        f'{n_active} segments, track {track_len_cm:.0f} cm',
        fontsize=14, fontweight='bold',
    )
    fig_zoom.tight_layout()
    fname = os.path.join(OUT_DIR, 'muon_event_display_zoomed.png')
    fig_zoom.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig_zoom)
    print(f"  Saved {fname}")

    # =========================================================================
    # 2. 3D track visualization
    # =========================================================================
    print("Generating 3D track plot...")
    pos_np = np.array(truth_pos)
    de_np = np.array(truth_de)
    active = de_np > 0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        pos_np[active, 0], pos_np[active, 1], pos_np[active, 2],
        c=de_np[active], cmap='hot', s=2, alpha=0.8,
    )
    fig.colorbar(sc, ax=ax, label='dE (MeV)', shrink=0.6)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title(f'Muon Track — {TRUTH_ENERGY:.0f} MeV')
    fname = os.path.join(OUT_DIR, 'muon_track_3d.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")

    # =========================================================================
    # 3. Parameter variation figures (zoomed to signal region)
    # =========================================================================
    print("\nGenerating parameter variation figures...")

    # Use east-side Y plane (index 2) for variation visualization
    truth_Y = np.array(truth_signals[2])  # east Y plane
    truth_Y_adc = truth_Y  # already in ADC (kernel includes conversion)

    # Find bounding box of the signal region (with generous padding)
    nonzero = np.nonzero(np.abs(truth_Y_adc) > 0.01)
    if len(nonzero[0]) > 0:
        w_lo = max(0, nonzero[0].min() - 30)
        w_hi = min(truth_Y_adc.shape[0], nonzero[0].max() + 30)
        t_lo = max(0, nonzero[1].min() - 50)
        t_hi = min(truth_Y_adc.shape[1], nonzero[1].max() + 50)
    else:
        w_lo, w_hi = 0, truth_Y_adc.shape[0]
        t_lo, t_hi = 0, truth_Y_adc.shape[1]

    print(f"  Signal region: wire [{w_lo}, {w_hi}], time [{t_lo}, {t_hi}]")

    param_configs = [
        ('x',      lambda d: make_muon_params(x=TRUTH_X + d)),
        ('y',      lambda d: make_muon_params(y=TRUTH_Y + d)),
        ('z',      lambda d: make_muon_params(z=TRUTH_Z + d)),
        ('theta',  lambda d: make_muon_params(theta=TRUTH_THETA + d)),
        ('phi',    lambda d: make_muon_params(phi=TRUTH_PHI + d)),
        ('energy', lambda d: make_muon_params(energy=TRUTH_ENERGY + d)),
    ]

    for pname, make_fn in param_configs:
        delta = PERTURBATIONS[pname]
        unit = 'mm' if pname in ('x', 'y', 'z') else ('rad' if pname in ('theta', 'phi') else 'MeV')

        # Perturbed signal
        e_p, pos_p, th_p, ph_p = make_fn(+delta)
        sig_plus, _, _ = simulate_muon(e_p, pos_p, th_p, ph_p, forward, log_T, dedx)
        jax.block_until_ready(sig_plus[2])
        perturbed_Y_adc = np.array(sig_plus[2])  # already in ADC

        diff_Y = perturbed_Y_adc - truth_Y_adc

        # Crop to signal region (with extra margin for perturbed signal)
        p_nz = np.nonzero(np.abs(perturbed_Y_adc) > 0.01)
        if len(p_nz[0]) > 0:
            crop_w_lo = max(0, min(w_lo, p_nz[0].min() - 30))
            crop_w_hi = min(truth_Y_adc.shape[0], max(w_hi, p_nz[0].max() + 30))
            crop_t_lo = max(0, min(t_lo, p_nz[1].min() - 50))
            crop_t_hi = min(truth_Y_adc.shape[1], max(t_hi, p_nz[1].max() + 50))
        else:
            crop_w_lo, crop_w_hi, crop_t_lo, crop_t_hi = w_lo, w_hi, t_lo, t_hi

        truth_crop = truth_Y_adc[crop_w_lo:crop_w_hi, crop_t_lo:crop_t_hi]
        pert_crop = perturbed_Y_adc[crop_w_lo:crop_w_hi, crop_t_lo:crop_t_hi]
        diff_crop = diff_Y[crop_w_lo:crop_w_hi, crop_t_lo:crop_t_hi]

        # Plot: truth | perturbed | difference
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        extent = [crop_w_lo, crop_w_hi, crop_t_lo, crop_t_hi]
        vmax_sig = max(np.abs(truth_crop).max(), np.abs(pert_crop).max())
        if vmax_sig < 1e-10:
            vmax_sig = 1.0
        vmax_diff = np.abs(diff_crop).max()
        if vmax_diff < 1e-10:
            vmax_diff = 1.0

        im0 = axes[0].imshow(
            truth_crop.T, aspect='auto', origin='lower', extent=extent,
            cmap='RdBu_r', vmin=-vmax_sig, vmax=vmax_sig,
            interpolation='nearest',
        )
        axes[0].set_title('Truth (East Y)')
        axes[0].set_xlabel('Wire index')
        axes[0].set_ylabel('Time bin')
        fig.colorbar(im0, ax=axes[0], label='ADC', shrink=0.8)

        im1 = axes[1].imshow(
            pert_crop.T, aspect='auto', origin='lower', extent=extent,
            cmap='RdBu_r', vmin=-vmax_sig, vmax=vmax_sig,
            interpolation='nearest',
        )
        axes[1].set_title(f'Perturbed ({pname} + {delta} {unit})')
        axes[1].set_xlabel('Wire index')
        axes[1].set_ylabel('Time bin')
        fig.colorbar(im1, ax=axes[1], label='ADC', shrink=0.8)

        im2 = axes[2].imshow(
            diff_crop.T, aspect='auto', origin='lower', extent=extent,
            cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff,
            interpolation='nearest',
        )
        axes[2].set_title(f'Difference (perturbed - truth)')
        axes[2].set_xlabel('Wire index')
        axes[2].set_ylabel('Time bin')
        fig.colorbar(im2, ax=axes[2], label='ADC', shrink=0.8)

        fig.suptitle(
            f'Parameter Variation: {pname} (delta = {delta} {unit})',
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f'muon_variation_{pname}.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")

    print("\nDone!")


if __name__ == '__main__':
    main()
