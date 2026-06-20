#!/usr/bin/env python3
"""Generate a VALID edge-distorted SCE field (field-cage defect) for the 40 cm
geometry, ray-trace it, and fit a SIREN — a curl-free truth with near-wall
structure to test edge recovery on.

Field = bulk space charge (ρ) + a field-cage non-uniformity: a potential ripple
on the y-walls that varies along the drift coordinate (rings deviating from
ideal grading). Solved with the general Dirichlet Poisson solver, so it is a
genuine electrostatic field (curl-free by construction) — not invented wiggles.
"""
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'efield'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from ElectricDistortion.io.config_loader import build_params
from ElectricDistortion.core.physics import compute_charge_density, compute_efield
from ElectricDistortion.core.edge_physics import solve_poisson_dirichlet
from ElectricDistortion.core.electron_drift import trace_electrons_parallel
from ElectricDistortion.core.drift_velocity import drift_velocity
import tools.sce_siren as S


def curl_rms(Ex, Ey, Ez, dx, dy, dz):
    cx = np.gradient(Ez, dy, axis=1) - np.gradient(Ey, dz, axis=2)
    cy = np.gradient(Ex, dz, axis=2) - np.gradient(Ez, dx, axis=0)
    cz = np.gradient(Ey, dx, axis=0) - np.gradient(Ex, dy, axis=1)
    return float(np.sqrt(np.mean(cx**2 + cy**2 + cz**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Q', type=float, default=3e-8)
    ap.add_argument('--ring-amp', type=float, default=150.0,
                    help='field-cage defect (V). Keep modest: large amps drive '
                         'Ex<0 (field reversal) → invalid drift field.')
    ap.add_argument('--n-rings', type=int, default=3)
    ap.add_argument('--omega', type=float, default=3.0)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'truth_40cm_edge.npz'))
    args = ap.parse_args()

    Lx, Ly, Lz, E0, T = 20.0, 40.0, 40.0, 500.0, 89.0
    params = build_params(preset='jaxtpc', overrides=dict(
        Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=args.Q,
        Nx_poisson=33, Ny_poisson=33, Nz_poisson=33,
        Nx_output=15, Ny_output=15, Nz_output=15))
    Nx = Ny = Nz = 33
    xg = np.linspace(0, Lx, Nx); yg = np.linspace(0, Ly, Ny); zg = np.linspace(0, Lz, Nz)
    dx = Lx/(Nx-1); dy = Ly/(Ny-1); dz = Lz/(Nz-1)
    eps = params['epsilon']; v_ion = params['v_ion']

    # bulk space charge
    rho = compute_charge_density(xg, yg, zg, args.Q, v_ion)

    # field-cage defect: potential ripple along x on BOTH y-walls
    ripple = args.ring_amp * np.sin(args.n_rings * np.pi * xg / Lx)   # (Nx,)
    phi_bc = np.zeros((Nx, Ny, Nz))
    phi_bc[:, 0, :] = ripple[:, None]      # y = 0 wall
    phi_bc[:, -1, :] = ripple[:, None]     # y = Ly wall
    print(f"Solving Dirichlet Poisson (33^3) with field-cage ripple "
          f"{args.ring_amp:.0f} V x {args.n_rings} rings ...")
    dphi = solve_poisson_dirichlet(rho, Lx, Ly, Lz, eps, phi_bc)
    Ex, Ey, Ez, Emag = compute_efield(dphi, E0, dx, dy, dz)

    cr = curl_rms(Ex, Ey, Ez, dx, dy, dz)
    # near-wall vs bulk |E| spread
    bulk = Emag[5:-5, 8:-8, 8:-8]; wall = Emag[:, :3, :]
    print(f"  |E| range [{Emag.min():.1f}, {Emag.max():.1f}] V/cm  "
          f"(bulk spread ±{np.abs(bulk-E0).max():.0f}, near-wall ±{np.abs(wall-E0).max():.0f})")
    # Validity needs BOTH curl≈0 AND Ex>0 everywhere: a reversed drift field
    # (Ex<0) breaks the ray-trace (electrons never reach the anode) and the
    # v(Ex) inversion, even though it is curl-free. Curl-free alone is NOT enough.
    ex_ok = Ex.min() > 0
    print(f"  TRUTH curl: {cr:.4f} V/cm/cm,  Ex range [{Ex.min():.0f}, {Ex.max():.0f}] V/cm "
          f"-> {'VALID' if (cr < 1e-2 and ex_ok) else 'INVALID'}")
    if not ex_ok:
        raise ValueError(
            f"Field-cage ripple {args.ring_amp:.0f} V reverses the drift field "
            f"(Ex_min={Ex.min():.0f} < 0). Lower --ring-amp for a valid field.")

    # ray-trace -> distortions
    ox = np.linspace(0, Lx, 15); oy = np.linspace(0, Ly, 15); oz = np.linspace(0, Lz, 15)
    print("Tracing electrons through the edge field ...")
    t_drift, y_anode, z_anode = trace_electrons_parallel(
        ox, oy, oz, xg, yg, zg, Ex, Ey, Ez, temperature=T, n_workers=params.get('n_workers'))
    v0 = drift_velocity(E0, T=T)
    dxm = v0*t_drift - ox[:, None, None]
    dym = y_anode - oy[None, :, None]
    dzm = z_anode - oz[None, None, :]
    print(f"  Δ ranges: dx[{dxm.min():.3f},{dxm.max():.3f}] dy[{dym.min():.3f},{dym.max():.3f}] cm")

    # fit SIREN (higher omega for the edge structure)
    interp = [RegularGridInterpolator((ox, oy, oz), d, bounds_error=False, fill_value=0.0)
              for d in (dxm, dym, dzm)]
    rng = np.random.RandomState(0)
    P = rng.uniform([0, 0, 0], [Lx, Ly, Lz], size=(60000, 3)).astype(np.float32)
    D = np.stack([f(P) for f in interp], -1).astype(np.float32)
    norm = np.array([Lx/2, Ly/2, Lz/2], np.float32)
    print(f"Fitting SIREN (omega={args.omega}) ...")
    sp = S.train_siren(P, D, jnp.array(norm), jnp.array(norm), omega_0=args.omega,
                       n_epochs=args.epochs, n_per_line=None, peak_lr=1e-3, verbose=False)
    S.save_siren_npz(args.out, sp, args.omega, norm, norm, E0, T,
                     extra=dict(Lx=Lx, Ly=Ly, Lz=Lz, Q=args.Q, ring_amp=args.ring_amp))

    # SIREN fit error vs the true (Poisson) E on the output grid
    efi = [RegularGridInterpolator((xg, yg, zg), f, bounds_error=False,
           fill_value=(E0 if i == 0 else 0.0)) for i, f in enumerate((Ex, Ey, Ez))]
    GX, GY, GZ = np.meshgrid(ox, oy, oz, indexing='ij')
    ep = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)
    vt, et = S.build_vinv_table(T)
    Er = np.array(S.recover_efield(sp, jnp.array(ep), E0, float(v0), vt, et,
                                   jnp.array(norm), jnp.array(norm), args.omega))
    Et = np.stack([efi[i](ep) for i in range(3)], -1)
    emag_t = np.sqrt((Et**2).sum(-1)); emag_r = np.sqrt((Er**2).sum(-1))
    print(f"\nSIREN representation of the EDGE field:")
    print(f"  |E| fit MAE = {np.mean(np.abs(emag_r-emag_t)):.3f} V/cm "
          f"({np.mean(np.abs(emag_r-emag_t))/E0:.3%})")
    print(f"  saved -> {args.out}")


if __name__ == '__main__':
    main()
