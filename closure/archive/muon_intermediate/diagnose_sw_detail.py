"""
Detailed diagnosis of SW loss anomalies at low energy and theta~0.5.

1. Energy: shows how V-plane nonzero pixel count vs K causes top_k to grab zeros
2. Theta: shows the track is parallel to V wires at theta=atan(0.5/0.866)=0.524

Run from project root:
    python3 closure_analysis_muon/diagnose_sw_detail.py
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
from ott_test.ot_losses import sliced_wasserstein_loss_jit

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments,
    build_muon_forward,
)

N_SEGMENTS = 2800
STEP_SIZE_MM = 0.5
K = 10000
N_PROJ = 200

TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 2
TRUTH_ENERGY = 200.0

PLANES = [0, 1, 2]
PLANE_NAMES = ['East U (+60°)', 'East V (-60°)', 'East Y (0°)']
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60, flush=True)
    print("DETAILED SW LOSS DIAGNOSIS", flush=True)
    print("=" * 60, flush=True)

    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    ot_key = jax.random.PRNGKey(42)

    def sim_forward(x, y, z, theta, phi, energy):
        pos, de = generate_muon_segments(
            energy, jnp.array([x, y, z]), theta, phi,
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        return forward(pos, de)

    print("Compiling...", flush=True)
    t0 = time.time()
    sim_jit = jax.jit(sim_forward)
    truth_sigs = sim_jit(TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY)
    for s in truth_sigs:
        jax.block_until_ready(s)
    print(f"  sim compiled ({time.time()-t0:.1f}s)", flush=True)

    # Truth pointclouds
    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(truth_sigs[p], K)
        target_clouds[p] = (pts, w)

    def per_plane_sw(sigs):
        """Return per-plane SW losses and pixel stats."""
        losses = []
        stats = []
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            loss = sliced_wasserstein_loss_jit(
                pts, w, target_clouds[p][0], target_clouds[p][1],
                ot_key, n_projections=N_PROJ,
            )
            losses.append(float(loss))
            # Stats
            sig_np = np.array(sigs[p])
            n_nz = int(np.sum(np.abs(sig_np) > 1e-6))
            w_np = np.array(w)
            n_zero_w = int(np.sum(w_np < 1e-6))
            stats.append({'nonzero_pixels': n_nz, 'zero_weight_in_topk': n_zero_w})
        return losses, stats

    # =========================================================================
    # 1. V-wire angle analysis
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("V-WIRE ANGLE ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # V wires at -60 deg: wire coord = -sin(60)*y + cos(60)*z = -0.866*y + 0.5*z
    # Track direction at phi=pi/2: [0, sin(theta), cos(theta)]
    # dr/ds = -0.866*sin(theta) + 0.5*cos(theta)
    # Zero crossing: tan(theta) = 0.5/0.866 => theta = 0.5236 rad (30 deg)
    critical_theta = np.arctan(0.5 / 0.866)
    print(f"  V-wire angle: -60 deg", flush=True)
    print(f"  Wire coord: r = -0.866*y + 0.5*z", flush=True)
    print(f"  Track direction (phi=pi/2): [0, sin(theta), cos(theta)]", flush=True)
    print(f"  dr/ds = -0.866*sin(theta) + 0.5*cos(theta)", flush=True)
    print(f"  dr/ds = 0 at theta = arctan(0.5/0.866) = {critical_theta:.4f} rad "
          f"({np.degrees(critical_theta):.1f} deg)", flush=True)
    print(f"  => Track is PARALLEL to V wires at theta ~ 0.524 rad!", flush=True)

    # Show dr/ds for a range of theta
    thetas = np.linspace(0.3, 1.2, 50)
    drds = -0.866 * np.sin(thetas) + 0.5 * np.cos(thetas)

    print(f"\n  theta (rad) | dr/ds (V wire projection rate)", flush=True)
    print(f"  {'-'*45}", flush=True)
    for th in [0.3, 0.4, 0.5, critical_theta, 0.55, 0.6, 0.7, 0.785, 0.9, 1.0, 1.2]:
        d = -0.866 * np.sin(th) + 0.5 * np.cos(th)
        marker = " <-- PARALLEL" if abs(th - critical_theta) < 0.02 else ""
        print(f"  {th:10.3f}   | {d:+.4f}{marker}", flush=True)

    # =========================================================================
    # 2. Per-plane SW loss vs theta
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("PER-PLANE SW LOSS vs THETA", flush=True)
    print("=" * 60, flush=True)

    theta_vals = np.linspace(0.3, 1.2, 30)
    plane_losses_theta = {p: [] for p in PLANES}
    total_losses_theta = []
    v_nonzero_theta = []
    v_zero_weight_theta = []

    t0 = time.time()
    for th in theta_vals:
        sigs = sim_jit(TRUTH_X, TRUTH_Y, TRUTH_Z, float(th), TRUTH_PHI, TRUTH_ENERGY)
        losses, stats = per_plane_sw(sigs)
        for p in PLANES:
            plane_losses_theta[p].append(losses[p])
        total_losses_theta.append(sum(losses))
        v_nonzero_theta.append(stats[1]['nonzero_pixels'])
        v_zero_weight_theta.append(stats[1]['zero_weight_in_topk'])
    print(f"  Theta sweep done ({time.time()-t0:.1f}s)", flush=True)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: per-plane losses
    for p in PLANES:
        axes[0, 0].plot(theta_vals, plane_losses_theta[p], lw=1.5, label=PLANE_NAMES[p])
    axes[0, 0].axvline(critical_theta, color='red', ls=':', lw=2,
                        label=f'V-parallel (θ={critical_theta:.3f})')
    axes[0, 0].axvline(TRUTH_THETA, color='green', ls='--', lw=2, label='Truth')
    axes[0, 0].set_xlabel('theta (rad)')
    axes[0, 0].set_ylabel('SW Loss (per plane)')
    axes[0, 0].set_title('Per-Plane SW Loss vs Theta')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: total loss
    axes[0, 1].plot(theta_vals, total_losses_theta, 'b-', lw=2)
    axes[0, 1].axvline(critical_theta, color='red', ls=':', lw=2, label=f'V-parallel')
    axes[0, 1].axvline(TRUTH_THETA, color='green', ls='--', lw=2, label='Truth')
    axes[0, 1].set_xlabel('theta (rad)')
    axes[0, 1].set_ylabel('Total SW Loss (U+V+Y)')
    axes[0, 1].set_title('Total SW Loss vs Theta')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: V plane nonzero pixels
    axes[1, 0].plot(theta_vals, v_nonzero_theta, 'b-', lw=2)
    axes[1, 0].axhline(K, color='red', ls='--', lw=2, label=f'K={K}')
    axes[1, 0].axvline(critical_theta, color='red', ls=':', lw=2, label='V-parallel')
    axes[1, 0].axvline(TRUTH_THETA, color='green', ls='--', lw=2, label='Truth')
    axes[1, 0].set_xlabel('theta (rad)')
    axes[1, 0].set_ylabel('East V nonzero pixels')
    axes[1, 0].set_title('V-Plane Signal Occupancy vs Theta')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: zero-weight points in top-K
    axes[1, 1].plot(theta_vals, v_zero_weight_theta, 'r-', lw=2)
    axes[1, 1].axvline(critical_theta, color='red', ls=':', lw=2, label='V-parallel')
    axes[1, 1].axvline(TRUTH_THETA, color='green', ls='--', lw=2, label='Truth')
    axes[1, 1].set_xlabel('theta (rad)')
    axes[1, 1].set_ylabel('Zero-weight points in top-K')
    axes[1, 1].set_title('V-Plane Top-K Zero Contamination')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f'Theta Anomaly: Track parallel to V wires at θ={critical_theta:.3f} rad ({np.degrees(critical_theta):.1f}°)\n'
        f'V wire coord: r = -0.866·y + 0.5·z, dr/ds = 0 when tan(θ) = 0.577',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'diagnose_theta_v_wire.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)

    # =========================================================================
    # 3. Per-plane SW loss vs energy
    # =========================================================================
    print("\n" + "=" * 60, flush=True)
    print("PER-PLANE SW LOSS vs ENERGY", flush=True)
    print("=" * 60, flush=True)

    energy_vals = np.linspace(100, 280, 30)
    plane_losses_e = {p: [] for p in PLANES}
    total_losses_e = []
    nonzero_per_plane_e = {p: [] for p in PLANES}
    zero_weight_per_plane_e = {p: [] for p in PLANES}

    t0 = time.time()
    for en in energy_vals:
        sigs = sim_jit(TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, float(en))
        losses, stats = per_plane_sw(sigs)
        for p in PLANES:
            plane_losses_e[p].append(losses[p])
            nonzero_per_plane_e[p].append(stats[p]['nonzero_pixels'])
            zero_weight_per_plane_e[p].append(stats[p]['zero_weight_in_topk'])
        total_losses_e.append(sum(losses))
    print(f"  Energy sweep done ({time.time()-t0:.1f}s)", flush=True)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: per-plane losses
    for p in PLANES:
        axes[0, 0].plot(energy_vals, plane_losses_e[p], lw=1.5, label=PLANE_NAMES[p])
    axes[0, 0].axvline(TRUTH_ENERGY, color='green', ls='--', lw=2, label='Truth')
    axes[0, 0].set_xlabel('Energy (MeV)')
    axes[0, 0].set_ylabel('SW Loss (per plane)')
    axes[0, 0].set_title('Per-Plane SW Loss vs Energy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: total loss
    axes[0, 1].plot(energy_vals, total_losses_e, 'b-', lw=2)
    axes[0, 1].axvline(TRUTH_ENERGY, color='green', ls='--', lw=2, label='Truth')
    axes[0, 1].set_xlabel('Energy (MeV)')
    axes[0, 1].set_ylabel('Total SW Loss (U+V+Y)')
    axes[0, 1].set_title('Total SW Loss vs Energy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: nonzero pixels per plane
    for p in PLANES:
        axes[1, 0].plot(energy_vals, nonzero_per_plane_e[p], lw=1.5, label=PLANE_NAMES[p])
    axes[1, 0].axhline(K, color='red', ls='--', lw=2, label=f'K={K}')
    axes[1, 0].axvline(TRUTH_ENERGY, color='green', ls='--', lw=2, label='Truth')
    axes[1, 0].set_xlabel('Energy (MeV)')
    axes[1, 0].set_ylabel('Nonzero pixels')
    axes[1, 0].set_title('Signal Occupancy vs Energy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: zero-weight in top-K
    for p in PLANES:
        axes[1, 1].plot(energy_vals, zero_weight_per_plane_e[p], lw=1.5, label=PLANE_NAMES[p])
    axes[1, 1].axvline(TRUTH_ENERGY, color='green', ls='--', lw=2, label='Truth')
    axes[1, 1].set_xlabel('Energy (MeV)')
    axes[1, 1].set_ylabel('Zero-weight points in top-K')
    axes[1, 1].set_title('Top-K Zero Contamination vs Energy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f'Energy Anomaly: V-plane drops below K={K} nonzero pixels at low energy\n'
        f'top_k grabs zero-valued pixels → corrupts SW distance',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'diagnose_energy_v_plane.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)

    # =========================================================================
    # 4. Wire projection rate dr/ds vs theta plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    thetas_fine = np.linspace(0.3, 1.2, 200)

    u_drds = 0.866 * np.sin(thetas_fine) + 0.5 * np.cos(thetas_fine)  # U at +60
    v_drds = -0.866 * np.sin(thetas_fine) + 0.5 * np.cos(thetas_fine)  # V at -60
    y_drds = np.sin(thetas_fine)  # Y at 0 (wires along y, measuring z... actually wires vertical, measuring y)

    # Y wires at 0 deg: r = sin(0)*y + cos(0)*z = z
    # dr/ds = cos(theta) for the z-component
    y_drds = np.cos(thetas_fine)

    ax.plot(thetas_fine, np.abs(u_drds), 'b-', lw=2, label='|dr/ds| U (+60°)')
    ax.plot(thetas_fine, np.abs(v_drds), 'r-', lw=2, label='|dr/ds| V (-60°)')
    ax.plot(thetas_fine, np.abs(y_drds), 'g-', lw=2, label='|dr/ds| Y (0°)')
    ax.axvline(critical_theta, color='red', ls=':', lw=2, alpha=0.7,
               label=f'V parallel (θ={critical_theta:.3f})')
    ax.axvline(TRUTH_THETA, color='green', ls='--', lw=2, alpha=0.7, label='Truth')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('theta (rad)')
    ax.set_ylabel('|dr/ds| (wire crossing rate)')
    ax.set_title('Wire Crossing Rate vs Track Angle (phi=π/2)\n'
                 'When |dr/ds|→0, track is parallel to wires → signal concentrates on few wires')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, 'diagnose_wire_crossing_rate.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
