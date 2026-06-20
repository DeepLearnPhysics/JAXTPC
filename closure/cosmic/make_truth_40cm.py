#!/usr/bin/env python3
"""Generate the first-principles SCE field for the 40 cm closure geometry and
fit a SIREN to it (rung 1 on the REAL field). Saves a truth npz for the ladder.

The truth we ultimately want to learn is the generated SCE field, not a random
one. This (a) runs the ElectricDistortion Poisson + ray-trace generator for the
closure detector (drift 20 cm, transverse 40 cm), (b) fits a SIREN by direct
dense regression on Δ, and (c) reports the fit error = how well the SIREN
represents the real field (the achievable floor for everything downstream).
"""
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'efield'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from ElectricDistortion.io.config_loader import build_params
from ElectricDistortion.run_sce import run
from ElectricDistortion.core.drift_velocity import drift_velocity
import tools.sce_siren as S
from scipy.interpolate import RegularGridInterpolator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Q', type=float, default=2e-9, help='charge production rate')
    ap.add_argument('--omega', type=float, default=2.0)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'truth_40cm.npz'))
    args = ap.parse_args()

    Lx, Ly, Lz, E0, T = 20.0, 40.0, 40.0, 500.0, 89.0
    params = build_params(preset='jaxtpc', overrides=dict(
        Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=args.Q,
        Nx_poisson=41, Ny_poisson=41, Nz_poisson=41,
        Nx_output=15, Ny_output=15, Nz_output=15))
    print(f"Generating 40cm SCE field (Q={args.Q:.1e}) ...")
    maps = run(params)
    v0 = float(drift_velocity(E0, T=T))
    print(f"  Δ ranges: dx[{maps['delta_x'].min():.3f},{maps['delta_x'].max():.3f}] "
          f"dy[{maps['delta_y'].min():.3f},{maps['delta_y'].max():.3f}] cm")

    # dense regression dataset: sample Δ across the volume (generator frame)
    ox, oy, oz = maps['output_x'], maps['output_y'], maps['output_z']
    interp = [RegularGridInterpolator((ox, oy, oz), maps[k], bounds_error=False,
              fill_value=0.0) for k in ('delta_x', 'delta_y', 'delta_z')]
    rng = np.random.RandomState(0)
    P = rng.uniform([0, 0, 0], [Lx, Ly, Lz], size=(40000, 3)).astype(np.float32)
    D = np.stack([f(P) for f in interp], -1).astype(np.float32)

    norm = np.array([Lx / 2, Ly / 2, Lz / 2], np.float32)
    print(f"Fitting SIREN (omega={args.omega}) by direct regression ...")
    sp = S.train_siren(P, D, jnp.array(norm), jnp.array(norm), omega_0=args.omega,
                       n_epochs=args.epochs, n_per_line=None, peak_lr=1e-3, verbose=False)
    S.save_siren_npz(args.out, sp, args.omega, norm, norm, E0, T,
                     extra=dict(Lx=Lx, Ly=Ly, Lz=Lz, Q=args.Q))

    # rung-1 fit error: recovered E vs first-principles E on the Poisson grid
    xp, yp, zp = maps['x_poisson'], maps['y_poisson'], maps['z_poisson']
    efi = [RegularGridInterpolator((xp, yp, zp), maps[k], bounds_error=False,
           fill_value=(E0 if k == 'Ex' else 0.0)) for k in ('Ex', 'Ey', 'Ez')]
    GX, GY, GZ = np.meshgrid(ox, oy, oz, indexing='ij')
    ep = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)
    vt, et = S.build_vinv_table(T)
    Er = np.array(S.recover_efield(sp, jnp.array(ep), E0, v0, vt, et,
                                   jnp.array(norm), jnp.array(norm), args.omega))
    Et = np.stack([efi[i](ep) for i in range(3)], -1)
    Emag_t = np.sqrt((Et ** 2).sum(-1)); Emag_r = np.sqrt((Er ** 2).sum(-1))
    print(f"\nRUNG 1 (real field):  SIREN fit to first-principles SCE")
    print(f"  |E| true range : [{Emag_t.min():.1f}, {Emag_t.max():.1f}] V/cm "
          f"(SCE amplitude ±{max(abs(Emag_t.min()-E0),abs(Emag_t.max()-E0)):.1f})")
    print(f"  |E| fit MAE    : {np.mean(np.abs(Emag_r-Emag_t)):.3f} V/cm "
          f"({np.mean(np.abs(Emag_r-Emag_t))/E0:.4%})")
    print(f"  saved truth → {args.out}")


if __name__ == '__main__':
    main()
