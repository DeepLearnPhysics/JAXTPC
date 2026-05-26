"""
Diagnose anomalous SW loss peaks at specific parameter values.

Investigates three anomalies:
  1. Energy < 140 MeV:  track shortens, fewer nonzero signal pixels vs K=10000
  2. Phi ~ 1.2 rad:     track gains x-component, crosses x=0 to west side
  3. Theta ~ 0.55 rad:  track geometry changes projection onto wire planes

For each anomalous point, generates:
  - Signal pixel statistics (nonzero count per plane vs K)
  - East vs west segment counts
  - Event display comparisons (truth vs anomalous)
  - Pointcloud visualizations

Run from project root:
    python3 closure_analysis_muon/diagnose_loss_peaks.py
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
from tools.pointcloud import signal_to_pointcloud

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
K = 10000

TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 2
TRUTH_ENERGY = 200.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Anomalous parameter values to investigate
ANOMALOUS = {
    'energy': [120.0, 130.0, 140.0, 150.0, 160.0],
    'phi':    [1.0, 1.1, 1.2, 1.3, 1.4],
    'theta':  [0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
}


def main():
    print("=" * 60, flush=True)
    print("DIAGNOSE SW LOSS PEAKS", flush=True)
    print("=" * 60, flush=True)

    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)



    def simulate(x, y, z, theta, phi, energy):
        pos = jnp.array([x, y, z], dtype=jnp.float32)
        positions, des = generate_muon_segments(
            jnp.float32(energy), pos, jnp.float32(theta), jnp.float32(phi),
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        signals = forward(positions, des)
        return signals, positions, des

    # --- Compile ---
    print("Compiling...", flush=True)
    t0 = time.time()
    sim_jit = jax.jit(simulate)
    truth_result = sim_jit(TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY)
    for s in truth_result[0]:
        jax.block_until_ready(s)
    print(f"  Compiled ({time.time()-t0:.1f}s)", flush=True)

    truth_signals, truth_pos, truth_de = truth_result
    truth_pos_np = np.array(truth_pos)
    truth_de_np = np.array(truth_de)

    # --- Analyze truth ---
    print("\n" + "=" * 60, flush=True)
    print("TRUTH ANALYSIS", flush=True)
    print("=" * 60, flush=True)
    _analyze_result("Truth", truth_signals, truth_pos_np, truth_de_np, K)

    # --- Analyze each anomalous point ---
    all_results = {}

    for param_name, values in ANOMALOUS.items():
        print(f"\n{'=' * 60}", flush=True)
        print(f"SWEEPING {param_name.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

        for val in values:
            # Build args with this parameter changed
            args = {
                'x': TRUTH_X, 'y': TRUTH_Y, 'z': TRUTH_Z,
                'theta': TRUTH_THETA, 'phi': TRUTH_PHI, 'energy': TRUTH_ENERGY,
            }
            args[param_name] = val

            signals, pos, des = sim_jit(**args)
            for s in signals:
                jax.block_until_ready(s)
            pos_np = np.array(pos)
            de_np = np.array(des)

            label = f"{param_name}={val:.2f}"
            _analyze_result(label, signals, pos_np, de_np, K)

            all_results[(param_name, val)] = (signals, pos_np, de_np)

    # --- Generate comparison event displays ---
    print(f"\n{'=' * 60}", flush=True)
    print("GENERATING COMPARISON PLOTS", flush=True)
    print(f"{'=' * 60}", flush=True)

    # 1. Energy comparison
    _plot_comparison(
        truth_signals,
        all_results[('energy', 130.0)][0],
        "Truth (E=200 MeV)", "E=130 MeV",
        os.path.join(OUT_DIR, "diagnose_energy_130.png"),
    )
    _plot_comparison(
        truth_signals,
        all_results[('energy', 140.0)][0],
        "Truth (E=200 MeV)", "E=140 MeV",
        os.path.join(OUT_DIR, "diagnose_energy_140.png"),
    )

    # 2. Phi comparison
    _plot_comparison(
        truth_signals,
        all_results[('phi', 1.2)][0],
        f"Truth (phi={TRUTH_PHI:.2f})", "phi=1.20",
        os.path.join(OUT_DIR, "diagnose_phi_1.2.png"),
    )

    # 3. Theta comparison
    _plot_comparison(
        truth_signals,
        all_results[('theta', 0.55)][0],
        f"Truth (theta={TRUTH_THETA:.2f})", "theta=0.55",
        os.path.join(OUT_DIR, "diagnose_theta_0.55.png"),
    )

    # --- 3D track comparisons ---
    _plot_tracks_3d(
        truth_pos_np, truth_de_np,
        all_results[('energy', 130.0)][1], all_results[('energy', 130.0)][2],
        "Truth (E=200)", "E=130 MeV",
        os.path.join(OUT_DIR, "diagnose_track_energy_130.png"),
    )
    _plot_tracks_3d(
        truth_pos_np, truth_de_np,
        all_results[('phi', 1.2)][1], all_results[('phi', 1.2)][2],
        f"Truth (phi={TRUTH_PHI:.2f})", "phi=1.20",
        os.path.join(OUT_DIR, "diagnose_track_phi_1.2.png"),
    )
    _plot_tracks_3d(
        truth_pos_np, truth_de_np,
        all_results[('theta', 0.55)][1], all_results[('theta', 0.55)][2],
        f"Truth (theta={TRUTH_THETA:.2f})", "theta=0.55",
        os.path.join(OUT_DIR, "diagnose_track_theta_0.55.png"),
    )

    # --- Pointcloud comparison ---
    _plot_pointclouds(
        truth_signals, all_results[('energy', 130.0)][0],
        "Truth (E=200)", "E=130 MeV", K,
        os.path.join(OUT_DIR, "diagnose_pc_energy_130.png"),
    )
    _plot_pointclouds(
        truth_signals, all_results[('phi', 1.2)][0],
        "Truth (phi=1.57)", "phi=1.20", K,
        os.path.join(OUT_DIR, "diagnose_pc_phi_1.2.png"),
    )
    _plot_pointclouds(
        truth_signals, all_results[('theta', 0.55)][0],
        "Truth (theta=0.79)", "theta=0.55", K,
        os.path.join(OUT_DIR, "diagnose_pc_theta_0.55.png"),
    )

    print("\nDone!", flush=True)


def _analyze_result(label, signals, pos_np, de_np, K):
    """Print diagnostic statistics for one simulation result."""
    n_active = int(np.sum(de_np > 0))
    n_east = int(np.sum((de_np > 0) & (pos_np[:, 0] < 0)))
    n_west = int(np.sum((de_np > 0) & (pos_np[:, 0] >= 0)))
    total_de = float(np.sum(de_np))

    x_range = (float(pos_np[de_np > 0, 0].min()), float(pos_np[de_np > 0, 0].max())) if n_active > 0 else (0, 0)

    print(f"\n  --- {label} ---", flush=True)
    print(f"  Active segments: {n_active}/{len(de_np)}", flush=True)
    print(f"  East/West split: {n_east} / {n_west}", flush=True)
    print(f"  Total dE: {total_de:.1f} MeV", flush=True)
    print(f"  x range: [{x_range[0]:.1f}, {x_range[1]:.1f}] mm", flush=True)

    plane_names = ['East U', 'East V', 'East Y', 'West U', 'West V', 'West Y']
    for pi, pname in enumerate(plane_names):
        sig_np = np.array(signals[pi])
        abs_sig = np.abs(sig_np)
        n_nonzero = int(np.sum(abs_sig > 1e-6))
        max_val = float(abs_sig.max())
        # Check how many pixels above the K-th largest
        flat = abs_sig.ravel()
        if len(flat) > K:
            kth_val = float(np.partition(flat, -K)[-K]) if n_nonzero >= K else 0.0
        else:
            kth_val = 0.0
        below_k = "OK" if n_nonzero >= K else f"BELOW K={K}!"
        print(f"    {pname}: nonzero={n_nonzero:,}, max={max_val:.1f}, "
              f"kth_val={kth_val:.2f}, {below_k}", flush=True)


def _plot_comparison(truth_sigs, test_sigs,
                     label_truth, label_test, fname):
    """3-column (U/V/Y) × 3-row (truth/test/diff) event display comparison."""
    plane_names = ['East U (Induction)', 'East V (Induction)', 'East Y (Collection)']
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    for col, pi in enumerate(range(3)):
        truth_adc = np.array(truth_sigs[pi])  # already in ADC
        test_adc = np.array(test_sigs[pi])
        diff = test_adc - truth_adc

        # Find combined bounding box
        all_nz = np.nonzero(np.abs(truth_adc) + np.abs(test_adc) > 0.01)
        if len(all_nz[0]) > 0:
            wlo = max(0, all_nz[0].min() - 15)
            whi = min(truth_adc.shape[0], all_nz[0].max() + 15)
            tlo = max(0, all_nz[1].min() - 30)
            thi = min(truth_adc.shape[1], all_nz[1].max() + 30)
        else:
            wlo, whi = 0, truth_adc.shape[0]
            tlo, thi = 0, truth_adc.shape[1]

        crops = [
            truth_adc[wlo:whi, tlo:thi],
            test_adc[wlo:whi, tlo:thi],
            diff[wlo:whi, tlo:thi],
        ]
        extent = [wlo, whi, tlo, thi]
        vmax_sig = max(np.abs(crops[0]).max(), np.abs(crops[1]).max())
        if vmax_sig < 1e-10:
            vmax_sig = 1.0
        vmax_diff = np.abs(crops[2]).max()
        if vmax_diff < 1e-10:
            vmax_diff = 1.0

        row_labels = [label_truth, label_test, 'Difference']
        vmaxes = [vmax_sig, vmax_sig, vmax_diff]

        for row in range(3):
            im = axes[row, col].imshow(
                crops[row].T, aspect='auto', origin='lower', extent=extent,
                cmap='RdBu_r', vmin=-vmaxes[row], vmax=vmaxes[row],
                interpolation='nearest',
            )
            if row == 0:
                axes[row, col].set_title(plane_names[col], fontsize=12, fontweight='bold')
            axes[row, col].set_ylabel(f'{row_labels[row]}\nTime bin')
            axes[row, col].set_xlabel('Wire index')
            fig.colorbar(im, ax=axes[row, col], label='ADC', shrink=0.8)

    fig.suptitle(f'{label_truth}  vs  {label_test}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)


def _plot_tracks_3d(pos1, de1, pos2, de2, label1, label2, fname):
    """Side-by-side 3D track comparison with x=0 plane."""
    fig = plt.figure(figsize=(16, 6))

    for i, (pos, de, label) in enumerate([(pos1, de1, label1), (pos2, de2, label2)]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        active = de > 0
        if np.any(active):
            sc = ax.scatter(
                pos[active, 0], pos[active, 1], pos[active, 2],
                c=de[active], cmap='hot', s=3, alpha=0.8,
            )
            fig.colorbar(sc, ax=ax, label='dE (MeV)', shrink=0.6)

            # Draw x=0 plane indicator
            y_range = [pos[active, 1].min(), pos[active, 1].max()]
            z_range = [pos[active, 2].min(), pos[active, 2].max()]
            yy, zz = np.meshgrid(
                np.linspace(y_range[0] - 20, y_range[1] + 20, 2),
                np.linspace(z_range[0] - 20, z_range[1] + 20, 2),
            )
            ax.plot_surface(np.zeros_like(yy), yy, zz, alpha=0.15, color='blue')
            ax.text(0, y_range[0], z_range[1] + 30, 'x=0\n(anode)', color='blue', fontsize=8)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.set_title(label)

    fig.suptitle('3D Track Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)


def _plot_pointclouds(truth_sigs, test_sigs, label1, label2, K, fname):
    """Compare pointclouds extracted from truth and test signals."""
    plane_names = ['East U', 'East V', 'East Y']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, pi in enumerate(range(3)):
        pts_t, w_t = signal_to_pointcloud(truth_sigs[pi], K)
        pts_p, w_p = signal_to_pointcloud(test_sigs[pi], K)

        pts_t_np = np.array(pts_t)
        w_t_np = np.array(w_t)
        pts_p_np = np.array(pts_p)
        w_p_np = np.array(w_p)

        # Count how many weights are effectively zero
        n_zero_truth = int(np.sum(w_t_np < 1e-6))
        n_zero_test = int(np.sum(w_p_np < 1e-6))

        for row, (pts, w, label, n_zero) in enumerate([
            (pts_t_np, w_t_np, label1, n_zero_truth),
            (pts_p_np, w_p_np, label2, n_zero_test),
        ]):
            nonzero = w > 1e-6
            if np.any(nonzero):
                sc = axes[row, col].scatter(
                    pts[nonzero, 0], pts[nonzero, 1],
                    c=w[nonzero], cmap='hot', s=1, alpha=0.5,
                )
                fig.colorbar(sc, ax=axes[row, col], label='weight', shrink=0.8)
            if n_zero > 0:
                # Show zero-weight points in grey
                axes[row, col].scatter(
                    pts[~nonzero, 0], pts[~nonzero, 1],
                    c='grey', s=1, alpha=0.2, label=f'{n_zero} zero-weight',
                )
                axes[row, col].legend(fontsize=8)

            axes[row, col].set_xlabel('wire (norm)')
            axes[row, col].set_ylabel('time (norm)')
            title_extra = f" ({n_zero} zeros)" if n_zero > 0 else ""
            if row == 0:
                axes[row, col].set_title(f'{plane_names[col]}\n{label}{title_extra}', fontsize=10)
            else:
                axes[row, col].set_title(f'{label}{title_extra}', fontsize=10)

    fig.suptitle(f'Pointcloud Comparison (K={K})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)


if __name__ == '__main__':
    main()
