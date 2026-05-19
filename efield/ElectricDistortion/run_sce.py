#!/usr/bin/env python3
"""Run the full Space Charge Effect simulation pipeline.

Usage
-----
    python -m ElectricDistortion --detector microboone
    python -m ElectricDistortion --detector sbnd --E0 500
    python -m ElectricDistortion --detector microboone --quick
"""

import argparse
import time

import numpy as np

from .io.config_loader import build_params
from .io.map_io import save_maps_npz, save_maps_hdf5
from .core.physics import (
    compute_charge_density, solve_poisson_dst, compute_efield,
    compute_vion_profile,
)
from .core.drift_velocity import drift_velocity
from .core.electron_drift import trace_electrons_parallel


def run(params):
    """Execute the full SCE pipeline.

    Parameters
    ----------
    params : dict
        Complete parameter set (as returned by ``build_params``).

    Returns
    -------
    results : dict
        All computed arrays and grid coordinates.
    """
    Lx = params["Lx"]
    Ly = params["Ly"]
    Lz = params["Lz"]
    E0 = params["E0"]
    Q = params["Q_charge_production"]
    epsilon = params["epsilon"]
    v_ion = params["v_ion"]
    temperature = params.get("temperature", 89.0)

    Nx = params["Nx_poisson"]
    Ny = params["Ny_poisson"]
    Nz = params["Nz_poisson"]

    Nxo = params["Nx_output"]
    Nyo = params["Ny_output"]
    Nzo = params["Nz_output"]

    # ------------------------------------------------------------------ grids
    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)
    z_grid = np.linspace(0, Lz, Nz)

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    output_x = np.linspace(0, Lx, Nxo)
    output_y = np.linspace(0, Ly, Nyo)
    output_z = np.linspace(0, Lz, Nzo)

    # ------------------------------------------------- Step 2: charge density
    t0 = time.time()
    rho = compute_charge_density(x_grid, y_grid, z_grid, Q, v_ion)
    print(f"[Step 2] Charge density:  rho_max = {rho[-1, 0, 0]:.3e} C/m^3  "
          f"({time.time() - t0:.2f} s)")

    # ------------------------------------------------- Step 3: Poisson solve
    t0 = time.time()
    dphi = solve_poisson_dst(rho, Lx, Ly, Lz, epsilon)
    print(f"[Step 3] Poisson solve:   dphi_max = {np.max(np.abs(dphi)):.3e} V  "
          f"({time.time() - t0:.2f} s)")

    # ------------------------------------------------- Step 4: E-field
    t0 = time.time()
    Ex, Ey, Ez, E_mag = compute_efield(dphi, E0, dx, dy, dz)
    E_ratio = E_mag / E0
    print(f"[Step 4] E-field:         |E|/E0 range = "
          f"[{E_ratio.min():.4f}, {E_ratio.max():.4f}]  "
          f"({time.time() - t0:.2f} s)")

    # --------------------------------- Step 4b: self-consistent v_ion iteration
    if params.get("self_consistent", False):
        mu_ion = params["mu_ion"]
        sc_max_iter = params.get("sc_max_iter", 10)
        sc_tol = params.get("sc_tol", 1e-3)
        sc_alpha = params.get("sc_alpha", 0.5)

        print(f"[Step 4b] Self-consistent iteration "
              f"(max_iter={sc_max_iter}, tol={sc_tol}, alpha={sc_alpha})")

        for iteration in range(1, sc_max_iter + 1):
            t_iter = time.time()

            v_ion_profile = compute_vion_profile(Ex, mu_ion)
            rho_new = compute_charge_density(
                x_grid, y_grid, z_grid, Q, v_ion_profile
            )

            # Under-relaxation
            rho = sc_alpha * rho_new + (1.0 - sc_alpha) * rho

            # Solve updated system
            dphi = solve_poisson_dst(rho, Lx, Ly, Lz, epsilon)
            Ex, Ey, Ez, E_mag = compute_efield(dphi, E0, dx, dy, dz)

            # Convergence check
            rho_max = np.max(np.abs(rho))
            if rho_max > 0:
                rel_change = np.max(np.abs(rho_new - rho)) / rho_max
            else:
                rel_change = 0.0

            E_ratio = E_mag / E0
            print(f"  iter {iteration}: rel_change={rel_change:.2e}, "
                  f"|E|/E0=[{E_ratio.min():.4f}, {E_ratio.max():.4f}]  "
                  f"({time.time() - t_iter:.2f} s)")

            if rel_change < sc_tol:
                print(f"  Converged after {iteration} iteration(s).")
                break
        else:
            print(f"  Warning: did not converge after {sc_max_iter} iterations "
                  f"(rel_change={rel_change:.2e})")

        E_ratio = E_mag / E0

    # ------------------------------------------------- Step 5-7: trace + maps
    t0 = time.time()
    n_traces = Nxo * Nyo * Nzo
    print(f"[Step 5] Tracing {n_traces} electrons "
          f"(grid {Nxo}x{Nyo}x{Nzo}, "
          f"workers={params.get('n_workers', 'auto')}) ...")

    t_drift, y_anode, z_anode = trace_electrons_parallel(
        output_x, output_y, output_z,
        x_grid, y_grid, z_grid, Ex, Ey, Ez,
        temperature=temperature,
        method=params.get("tracing_method", "RK45"),
        rtol=params.get("tracing_rtol", 1e-6),
        atol=params.get("tracing_atol", 1e-6),
        t_max=params.get("tracing_t_max", 20000.0),
        n_workers=params.get("n_workers"),
    )

    v_nominal = drift_velocity(E0, T=temperature)
    x_true = output_x[:, None, None]
    y_true = output_y[None, :, None]
    z_true = output_z[None, None, :]

    delta_x = v_nominal * t_drift - x_true
    delta_y = y_anode - y_true
    delta_z = z_anode - z_true

    elapsed = time.time() - t0
    print(f"[Step 7] Distortion maps: "
          f"dx=[{delta_x.min():.3f}, {delta_x.max():.3f}] cm, "
          f"dy=[{delta_y.min():.3f}, {delta_y.max():.3f}] cm, "
          f"dz=[{delta_z.min():.3f}, {delta_z.max():.3f}] cm  "
          f"({elapsed:.1f} s, {elapsed / n_traces:.3f} s/trace)")

    return dict(
        x_poisson=x_grid, y_poisson=y_grid, z_poisson=z_grid,
        output_x=output_x, output_y=output_y, output_z=output_z,
        rho=rho, dphi=dphi,
        Ex=Ex, Ey=Ey, Ez=Ez, E_mag=E_mag, E_ratio=E_ratio,
        delta_x=delta_x, delta_y=delta_y, delta_z=delta_z,
        params=params,
    )


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Space Charge Effect simulation for LArTPCs"
    )
    parser.add_argument("--detector", type=str, default=None,
                        help="Detector preset name (e.g. microboone, sbnd)")
    parser.add_argument("--E0", type=float, default=None,
                        help="Override nominal field (V/cm)")
    parser.add_argument("--Q", type=float, default=None,
                        help="Override charge production rate (C/m^3/s)")
    parser.add_argument("--Lx", type=float, default=None)
    parser.add_argument("--Ly", type=float, default=None)
    parser.add_argument("--Lz", type=float, default=None)
    parser.add_argument("--Nx", type=int, default=None,
                        help="Poisson grid Nx")
    parser.add_argument("--Ny", type=int, default=None,
                        help="Poisson grid Ny")
    parser.add_argument("--Nz", type=int, default=None,
                        help="Poisson grid Nz")
    parser.add_argument("--Nxo", type=int, default=None,
                        help="Output grid Nx")
    parser.add_argument("--Nyo", type=int, default=None,
                        help="Output grid Ny")
    parser.add_argument("--Nzo", type=int, default=None,
                        help="Output grid Nz")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="sce_maps.npz",
                        help="Output file path (.npz or .h5)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: coarse grids (11^3 output)")
    parser.add_argument("--self-consistent", action="store_true",
                        help="Enable self-consistent v_ion iteration")
    parser.add_argument("--sc-max-iter", type=int, default=None,
                        help="Max iterations for self-consistent loop")
    parser.add_argument("--sc-tol", type=float, default=None,
                        help="Convergence tolerance for self-consistent loop")

    args = parser.parse_args()

    # Build overrides from CLI flags
    _CLI_MAP = {
        "E0": "E0", "Q": "Q_charge_production",
        "Lx": "Lx", "Ly": "Ly", "Lz": "Lz",
        "Nx": "Nx_poisson", "Ny": "Ny_poisson", "Nz": "Nz_poisson",
        "Nxo": "Nx_output", "Nyo": "Ny_output", "Nzo": "Nz_output",
        "workers": "n_workers",
    }
    overrides = {v: getattr(args, k) for k, v in _CLI_MAP.items()
                 if getattr(args, k) is not None}

    if args.self_consistent:
        overrides["self_consistent"] = True
    if args.sc_max_iter is not None:
        overrides["sc_max_iter"] = args.sc_max_iter
    if args.sc_tol is not None:
        overrides["sc_tol"] = args.sc_tol

    if args.quick:
        overrides.setdefault("Nx_poisson", 51)
        overrides.setdefault("Ny_poisson", 51)
        overrides.setdefault("Nz_poisson", 51)
        overrides.setdefault("Nx_output", 11)
        overrides.setdefault("Ny_output", 11)
        overrides.setdefault("Nz_output", 11)

    params = build_params(preset=args.detector, overrides=overrides)

    # Print configuration summary
    print("=" * 60)
    print("Space Charge Effect Simulation")
    print("=" * 60)
    print(f"  Detector:      {args.detector or 'custom'}")
    print(f"  Drift volume:  {params['Lx']:.1f} x {params['Ly']:.1f} "
          f"x {params['Lz']:.1f} cm")
    print(f"  E0:            {params['E0']:.1f} V/cm")
    print(f"  Q:             {params['Q_charge_production']:.2e} C/m^3/s")
    print(f"  v_ion:         {params['v_ion']:.4e} cm/s")
    print(f"  Poisson grid:  {params['Nx_poisson']}x{params['Ny_poisson']}"
          f"x{params['Nz_poisson']}")
    print(f"  Output grid:   {params['Nx_output']}x{params['Ny_output']}"
          f"x{params['Nz_output']}")
    v_nom = drift_velocity(params["E0"], T=params.get("temperature", 89.0))
    print(f"  v_drift(E0):   {v_nom:.4f} cm/us  ({v_nom*10:.3f} mm/us)")
    print("=" * 60)

    # Run
    results = run(params)

    # Save
    out_path = args.output
    save_fn = save_maps_hdf5 if out_path.endswith((".h5", ".hdf5")) else save_maps_npz
    save_fn(
        out_path,
        results["output_x"], results["output_y"], results["output_z"],
        results["delta_x"], results["delta_y"], results["delta_z"],
        results["Ex"], results["Ey"], results["Ez"],
        results["E_mag"], results["E_ratio"],
        params=params,
    )
    print(f"\nMaps saved to {out_path}")

    # Optional plots
    if args.plot:
        from .plotting.visualize import plot_summary
        plot_summary(
            results["x_poisson"], results["y_poisson"], results["z_poisson"],
            results["E_mag"], params["E0"],
            results["output_x"], results["output_y"], results["output_z"],
            results["delta_x"], results["delta_y"], results["delta_z"],
            save_path="sce_summary.png",
        )
        print("Summary plot saved to sce_summary.png")

    return results


if __name__ == "__main__":
    main()
