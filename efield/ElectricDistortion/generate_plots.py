#!/usr/bin/env python3
"""Generate all advanced SCE visualisation plots.

Usage
-----
    python3 -m ElectricDistortion.generate_plots
    python3 -m ElectricDistortion.generate_plots --detector jaxtpc
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io.config_loader import build_params
from .core.physics import (
    compute_charge_density, solve_poisson_dst, compute_efield,
    compute_vion_profile,
)
from .run_sce import run
from .plotting.advanced import (
    plot_warped_grid, plot_total_distortion, plot_quiver_transverse,
    plot_face_maps, plot_profile_fan, plot_efield_streamlines,
    plot_convergence, plot_detector_comparison, plot_sc_correction,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _mid_index(arr):
    return int(np.argmin(np.abs(arr - (arr[0] + arr[-1]) / 2.0)))


def _nearest_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


def _run_sim(detector, nx_out=21, nx_poi=51, **extra):
    """Run a simulation with the given grid sizes."""
    overrides = {
        'Nx_poisson': nx_poi, 'Ny_poisson': nx_poi, 'Nz_poisson': nx_poi,
        'Nx_output': nx_out, 'Ny_output': nx_out, 'Nz_output': nx_out,
    }
    overrides.update(extra)
    params = build_params(preset=detector, overrides=overrides)
    return run(params)


def _collect_convergence(detector):
    """Run self-consistent iteration and collect per-step history."""
    params = build_params(preset=detector, overrides={
        'Nx_poisson': 51, 'Ny_poisson': 51, 'Nz_poisson': 51,
    })

    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    Nx = params['Nx_poisson']
    Ny = params['Ny_poisson']
    Nz = params['Nz_poisson']
    E0 = params['E0']
    Q = params['Q_charge_production']
    epsilon = params['epsilon']
    mu_ion = params['mu_ion']
    v_ion = params['v_ion']

    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)
    z_grid = np.linspace(0, Lz, Nz)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    iy = _mid_index(y_grid)
    iz = _mid_index(z_grid)

    alpha = 0.5
    max_iter = 10
    tol = 1e-3

    rho = compute_charge_density(x_grid, y_grid, z_grid, Q, v_ion)
    dphi = solve_poisson_dst(rho, Lx, Ly, Lz, epsilon)
    Ex, Ey, Ez, E_mag = compute_efield(dphi, E0, dx, dy, dz)

    history = {
        'rel_change': [],
        'rho_profiles': [rho[:, iy, iz].copy()],
        'x_grid': x_grid.copy(),
        'v_ion_profiles': [np.full_like(x_grid, v_ion)],
        'tol': tol,
    }

    for _ in range(max_iter):
        v_ion_profile = compute_vion_profile(Ex, mu_ion)
        rho_new = compute_charge_density(x_grid, y_grid, z_grid, Q,
                                         v_ion_profile)
        rho = alpha * rho_new + (1.0 - alpha) * rho

        dphi = solve_poisson_dst(rho, Lx, Ly, Lz, epsilon)
        Ex, Ey, Ez, E_mag = compute_efield(dphi, E0, dx, dy, dz)

        rho_max = np.max(np.abs(rho))
        rel_change = (np.max(np.abs(rho_new - rho)) / rho_max
                      if rho_max > 0 else 0.0)

        history['rel_change'].append(rel_change)
        history['rho_profiles'].append(rho[:, iy, iz].copy())
        history['v_ion_profiles'].append(v_ion_profile.copy())

        if rel_change < tol:
            break

    return history


# ---------------------------------------------------------------------- #
#  Main generator                                                        #
# ---------------------------------------------------------------------- #

def generate_all(detector="jaxtpc"):
    os.makedirs(PLOT_DIR, exist_ok=True)
    t_start = time.time()

    # Clear old plots
    for f in os.listdir(PLOT_DIR):
        if f.endswith('.png'):
            os.remove(os.path.join(PLOT_DIR, f))

    # ── 1. Main simulation ─────────────────────────────────────────────
    print(f"[1/6] Running {detector} simulation (21^3 output) ...")
    r = _run_sim(detector, nx_out=21, nx_poi=51)

    ox = r['output_x']
    oy = r['output_y']
    oz = r['output_z']
    dxm = r['delta_x']
    dym = r['delta_y']
    dzm = r['delta_z']
    xp = r['x_poisson']
    yp = r['y_poisson']
    zp = r['z_poisson']
    Ex = r['Ex']
    Ey = r['Ey']
    Ez = r['Ez']
    E0 = r['params']['E0']
    Lx = r['params']['Lx']

    # Pre-compute useful indices
    ix_half = _nearest_index(ox, Lx / 2)
    ix_cath = len(ox) - 1

    delta_mag = np.sqrt(dxm**2 + dym**2 + dzm**2)
    delta_max = delta_mag.max()
    levels = [v for v in [1, 2, 5, 10, 15, 20, 25] if v < delta_max * 0.9]

    # ── 2. Warped grids ───────────────────────────────────────────────
    print("[2/6] Generating warped grids + total distortion maps ...")

    plot_warped_grid(ox, oy, oz, dxm, dym, dzm, plane="xz",
                     save_path=os.path.join(PLOT_DIR,
                                            "01_warped_grid_xz.png"))
    plt.close("all")

    plot_warped_grid(ox, oy, oz, dxm, dym, dzm, plane="xy",
                     save_path=os.path.join(PLOT_DIR,
                                            "02_warped_grid_xy.png"))
    plt.close("all")

    plot_warped_grid(ox, oy, oz, dxm, dym, dzm, plane="yz",
                     index=ix_half,
                     save_path=os.path.join(PLOT_DIR,
                                            "03_warped_grid_yz_mid.png"))
    plt.close("all")

    plot_warped_grid(ox, oy, oz, dxm, dym, dzm, plane="yz",
                     index=ix_cath,
                     save_path=os.path.join(PLOT_DIR,
                                            "04_warped_grid_yz_cathode.png"))
    plt.close("all")

    # ── 3. Total distortion maps ──────────────────────────────────────
    plot_total_distortion(ox, oy, oz, dxm, dym, dzm, plane="xz",
                          contour_levels=levels or None,
                          save_path=os.path.join(PLOT_DIR,
                                                 "05_total_distortion_xz.png"))
    plt.close("all")

    plot_total_distortion(ox, oy, oz, dxm, dym, dzm, plane="xy",
                          contour_levels=levels or None,
                          save_path=os.path.join(PLOT_DIR,
                                                 "06_total_distortion_xy.png"))
    plt.close("all")

    plot_total_distortion(ox, oy, oz, dxm, dym, dzm, plane="yz",
                          index=ix_half, contour_levels=levels or None,
                          save_path=os.path.join(PLOT_DIR,
                                                 "07_total_distortion_yz_mid.png"))
    plt.close("all")

    plot_total_distortion(ox, oy, oz, dxm, dym, dzm, plane="yz",
                          index=ix_cath, contour_levels=levels or None,
                          save_path=os.path.join(PLOT_DIR,
                                                 "08_total_distortion_yz_cathode.png"))
    plt.close("all")

    # ── 4. Vector fields + face maps ──────────────────────────────────
    print("[3/6] Generating vector fields, profiles, streamlines ...")

    plot_quiver_transverse(ox, oy, oz, dym, dzm, x_index=ix_cath,
                           save_path=os.path.join(PLOT_DIR,
                                                  "09_quiver_cathode.png"))
    plt.close("all")

    plot_face_maps(ox, oy, oz, dxm, dym, dzm,
                   save_path=os.path.join(PLOT_DIR, "10_face_maps.png"))
    plt.close("all")

    # ── 5. Profile fans ───────────────────────────────────────────────
    iz_mid = _mid_index(oz)
    Nox = len(ox)
    x_indices = [Nox // 4, Nox // 2, 3 * Nox // 4, Nox - 1]
    profiles = [dym[ix, :, iz_mid] for ix in x_indices]
    labels = [f"x = {ox[ix]:.0f} cm" for ix in x_indices]
    plot_profile_fan(oy, profiles, labels,
                     xlabel="y (cm)", ylabel="delta_y (cm)",
                     title="Transverse distortion buildup with drift distance",
                     save_path=os.path.join(PLOT_DIR,
                                            "11_profile_fan_dy_vs_y.png"))
    plt.close("all")

    Noy = len(oy)
    y_indices = [0, Noy // 4, Noy // 2, 3 * Noy // 4, Noy - 1]
    profiles = [dym[:, iy, iz_mid] for iy in y_indices]
    labels = [f"y = {oy[iy]:.0f} cm" for iy in y_indices]
    plot_profile_fan(ox, profiles, labels,
                     xlabel="x (cm)", ylabel="delta_y (cm)",
                     title="delta_y(x) at different y positions",
                     save_path=os.path.join(PLOT_DIR,
                                            "12_profile_fan_dy_vs_x.png"))
    plt.close("all")

    # ── 6. E-field streamlines ────────────────────────────────────────
    plot_efield_streamlines(xp, yp, zp, Ex, Ey, Ez, E0, plane="xz",
                            save_path=os.path.join(PLOT_DIR,
                                                   "13_streamlines_xz.png"))
    plt.close("all")
    plot_efield_streamlines(xp, yp, zp, Ex, Ey, Ez, E0, plane="xy",
                            save_path=os.path.join(PLOT_DIR,
                                                   "14_streamlines_xy.png"))
    plt.close("all")

    # ── 7. Convergence diagnostics ────────────────────────────────────
    print(f"[4/6] Running {detector} self-consistent iteration ...")
    history = _collect_convergence(detector)
    plot_convergence(history,
                     save_path=os.path.join(PLOT_DIR, "15_convergence.png"))
    plt.close("all")

    # ── 8. Self-consistent correction ─────────────────────────────────
    print(f"[5/6] Running {detector} SC simulation for correction map ...")
    r_sc = _run_sim(detector, nx_out=21, nx_poi=51, self_consistent=True)
    plot_sc_correction(ox, oy, oz, dym, r_sc['delta_y'], plane="xy",
                       save_path=os.path.join(PLOT_DIR,
                                              "16_sc_correction_xy.png"))
    plt.close("all")

    # ── 9. Multi-detector comparison ──────────────────────────────────
    print("[6/6] Running multi-detector comparison (11^3) ...")
    comp_detectors = ['microboone', 'sbnd', 'icarus', 'jaxtpc']
    det_results = {}
    for det in comp_detectors:
        r_det = _run_sim(det, nx_out=11, nx_poi=51)
        xp_d = r_det['x_poisson']
        E_mag_d = r_det['E_mag']
        iy_d = _mid_index(r_det['y_poisson'])
        iz_d = _mid_index(r_det['z_poisson'])
        E_ratio_profile = E_mag_d[:, iy_d, iz_d] / r_det['params']['E0']

        dx_d = r_det['delta_x']
        dy_d = r_det['delta_y']
        dz_d = r_det['delta_z']
        dmag = np.sqrt(dx_d**2 + dy_d**2 + dz_d**2)

        det_results[det] = {
            'x_norm': xp_d / r_det['params']['Lx'],
            'E_ratio_profile': E_ratio_profile,
            'max_delta_y': np.max(np.abs(dy_d)),
            'max_delta_y_frac': np.max(np.abs(dy_d)) / r_det['params']['Ly'],
            'max_delta': np.max(dmag),
        }

    plot_detector_comparison(det_results,
                             save_path=os.path.join(PLOT_DIR,
                                                    "17_detector_comparison.png"))
    plt.close("all")

    # ── README ────────────────────────────────────────────────────────
    _write_readme(detector, r)

    elapsed = time.time() - t_start
    n_files = len([f for f in os.listdir(PLOT_DIR)
                   if f.endswith(('.png', '.md'))])
    print(f"\nDone! {n_files} files in {PLOT_DIR}/  ({elapsed:.0f} s)")


# ---------------------------------------------------------------------- #
#  README                                                                #
# ---------------------------------------------------------------------- #

def _write_readme(detector, results):
    params = results['params']
    dx = results['delta_x']
    dy = results['delta_y']
    dz = results['delta_z']
    dmag = np.sqrt(dx**2 + dy**2 + dz**2)
    readme = f"""\
# SCE Visualization Plots

Generated for **{detector}** detector:
{params['Lx']:.0f} x {params['Ly']:.0f} x {params['Lz']:.0f} cm,
E0 = {params['E0']:.0f} V/cm,
Q = {params['Q_charge_production']:.2e} C/m^3/s

Key metrics:
- max |delta_x| = {np.max(np.abs(dx)):.1f} cm
- max |delta_y| = {np.max(np.abs(dy)):.1f} cm
- max |delta_z| = {np.max(np.abs(dz)):.1f} cm
- max |delta|   = {np.max(dmag):.1f} cm
- |E|/E0 range  = [{results['E_ratio'].min():.4f}, {results['E_ratio'].max():.4f}]

All spatial axes are in **cm**. All distortions are in **cm**.
E-field ratios are dimensionless. Charge density in C/m^3.

## Plot Descriptions

### Warped Grids (01-04)
Gray lines = undistorted grid. Coloured lines = reconstructed positions
after SCE. Colour = in-plane displacement magnitude (cm).

- **01_warped_grid_xz** -- x-z slice at y midplane (drift vs length)
- **02_warped_grid_xy** -- x-y slice at z midplane (drift vs height)
- **03_warped_grid_yz_mid** -- y-z slice at mid-drift (transverse pattern
  halfway through the drift volume)
- **04_warped_grid_yz_cathode** -- y-z slice at cathode (maximum transverse
  distortion; shows the full "squeezing" pattern for electrons with the
  longest drift path)

### Total Distortion Maps (05-08)
|delta| = sqrt(dx^2 + dy^2 + dz^2) in cm. White contour lines mark
thresholds.

- **05_total_distortion_xz** -- x-z slice at y midplane
- **06_total_distortion_xy** -- x-y slice at z midplane
- **07_total_distortion_yz_mid** -- y-z at mid-drift
- **08_total_distortion_yz_cathode** -- y-z at cathode (maximum distortion)

### Vector Fields & Face Maps (09-10)
- **09_quiver_cathode** -- Arrow plot of (delta_y, delta_z) at the cathode
  face. Arrow direction = displacement direction, colour = magnitude.
- **10_face_maps** -- 2x2: cathode |delta|, cathode quiver, top face, side
  face.

### Profile Fans (11-12)
- **11_profile_fan_dy_vs_y** -- delta_y(y) at 4 drift distances. Shows
  linear buildup from midplane, steepening with drift distance.
- **12_profile_fan_dy_vs_x** -- delta_y(x) at 5 y positions. Shows how
  distortion grows from anode to cathode.

### E-field Streamlines (13-14)
Coloured by |E|/E0 (coolwarm: blue = weakened, red = enhanced). Electrons
drift opposite to these lines.

- **13_streamlines_xz** -- x-z plane
- **14_streamlines_xy** -- x-y plane

### Self-Consistent Iteration (15-16)
- **15_convergence** -- Three panels: convergence curve (log), charge
  density rho(x) per iteration, ion drift speed v_ion(x) evolution.
- **16_sc_correction_xy** -- delta_y comparison: base vs self-consistent
  vs correction (difference).

### Multi-Detector Comparison (17)
- **17_detector_comparison** -- |E|/E0 profiles (normalised x), fractional
  transverse distortion (%), and absolute |delta| for four surface LArTPC
  detectors.
"""
    with open(os.path.join(PLOT_DIR, "README.md"), 'w') as f:
        f.write(readme)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all advanced SCE visualisation plots"
    )
    parser.add_argument("--detector", default="jaxtpc",
                        help="Primary detector preset (default: jaxtpc)")
    args = parser.parse_args()
    generate_all(detector=args.detector)
