#!/usr/bin/env python3
"""
End-to-end validation of the differentiable SCE-SIREN field used by the
forward sim (tools/sce_siren.py).

Pipeline
--------
1. Generate first-principles SCE maps with ``ElectricDistortion`` (Poisson →
   E-field → electron ray-trace → distortion maps Δ).
2. Build line-sampled training data from Δ (cosmic-track-like lines), exactly
   as the reference notebook does.
3. Fit the pure-JAX SIREN to Δ.
4. Recover the E-field by autodiff (∂Δ/∂x) + Walkowiak v-inversion and compare
   Ex, Ey, Ez, |E| and the Box recombination R against the first-principles
   maps. This is the number that matters: if |E| and R match, the field that
   reaches recombination in the sim is correct.
5. Save the trained SIREN to an .npz consumable by
   ``DetectorSimulator(..., electric_dist_siren_path=...)``.

Run (coarse, ~1-2 min CPU):
    JAX_PLATFORM_NAME=cpu python3 efield/validate_siren_wiring.py --quick
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(__file__))  # make ElectricDistortion importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root → tools

import jax  # noqa: E402

from ElectricDistortion.io.config_loader import build_params  # noqa: E402
from ElectricDistortion.run_sce import run  # noqa: E402
from ElectricDistortion.core.drift_velocity import drift_velocity  # noqa: E402

from tools.sce_siren import (  # noqa: E402
    train_siren, recover_efield, save_siren_npz,
    build_vinv_table, drift_velocity_jax)
import jax.numpy as jnp  # noqa: E402


def generate_maps(quick):
    ov = {}
    if quick:
        ov = dict(Nx_poisson=51, Ny_poisson=51, Nz_poisson=51,
                  Nx_output=17, Ny_output=17, Nz_output=17)
    else:
        ov = dict(Nx_poisson=81, Ny_poisson=81, Nz_poisson=81,
                  Nx_output=25, Ny_output=25, Nz_output=25)
    params = build_params(preset='jaxtpc', overrides=ov)
    print(f"Generating SCE maps: Poisson {params['Nx_poisson']}^3, "
          f"output {params['Nx_output']}^3 ...")
    return run(params), params


def build_line_dataset(maps, Lx, Ly, Lz, n_lines=800, n_per_line=80, seed=42):
    x_out, y_out, z_out = maps['output_x'], maps['output_y'], maps['output_z']
    interps = [
        RegularGridInterpolator((x_out, y_out, z_out), maps[k],
                                method='linear', bounds_error=False, fill_value=0.0)
        for k in ('delta_x', 'delta_y', 'delta_z')]
    rng = np.random.RandomState(seed)
    bmax = np.array([Lx, Ly, Lz])
    sizes = bmax
    areas = np.array([sizes[1]*sizes[2], sizes[1]*sizes[2],
                      sizes[0]*sizes[2], sizes[0]*sizes[2],
                      sizes[0]*sizes[1], sizes[0]*sizes[1]])
    probs = areas / areas.sum()

    def _surface_point():
        face = rng.choice(6, p=probs)
        u, v = rng.uniform(), rng.uniform()
        c = {0: [0, u*sizes[1], v*sizes[2]], 1: [bmax[0], u*sizes[1], v*sizes[2]],
             2: [u*sizes[0], 0, v*sizes[2]], 3: [u*sizes[0], bmax[1], v*sizes[2]],
             4: [u*sizes[0], v*sizes[1], 0], 5: [u*sizes[0], v*sizes[1], bmax[2]]}
        return np.array(c[face])

    starts = np.array([_surface_point() for _ in range(n_lines)])
    ends = np.array([_surface_point() for _ in range(n_lines)])
    t = np.linspace(0, 1, n_per_line)[None, :, None]
    pos = (starts[:, None, :] * (1 - t) + ends[:, None, :] * t).reshape(-1, 3)
    corr = np.stack([f(pos) for f in interps], axis=-1)
    return pos.astype(np.float32), corr.astype(np.float32), n_per_line


def box_recomb(E_Vcm, dedx=2.0):
    xi = (0.212 / 1.396) * dedx / np.maximum(E_Vcm / 1000.0, 1e-10)
    return np.log(0.93 + xi) / xi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--epochs', type=int, default=1200)
    ap.add_argument('--out', type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         'sce_siren_jaxtpc.npz'))
    args = ap.parse_args()

    maps, params = generate_maps(args.quick)
    Lx, Ly, Lz, E0, T = (params['Lx'], params['Ly'], params['Lz'],
                         params['E0'], params.get('temperature', 89.0))
    v0 = float(drift_velocity(E0, T=T))

    # Training frame normalisation (generator frame: [0,L])
    norm_off = np.array([Lx/2, Ly/2, Lz/2], np.float32)
    norm_sc = np.array([Lx/2, Ly/2, Lz/2], np.float32)

    pos, corr, n_per_line = build_line_dataset(maps, Lx, Ly, Lz)
    print(f"Training SIREN on {pos.shape[0]:,} samples, {args.epochs} epochs ...")
    t0 = time.time()
    sp = train_siren(pos, corr, jnp.array(norm_off), jnp.array(norm_sc),
                     omega_0=5.0, n_epochs=args.epochs, n_per_line=n_per_line)
    print(f"Training time: {time.time()-t0:.1f}s")

    save_siren_npz(args.out, sp, 5.0, norm_off, norm_sc, E0, T,
                   extra=dict(Lx=Lx, Ly=Ly, Lz=Lz))
    print(f"Saved SIREN → {args.out}")

    # ---- recover E on the output grid, compare to first-principles maps ----
    x_p, y_p, z_p = maps['x_poisson'], maps['y_poisson'], maps['z_poisson']
    ef_interp = [RegularGridInterpolator((x_p, y_p, z_p), maps[k],
                 method='linear', bounds_error=False,
                 fill_value=(E0 if k == 'Ex' else 0.0))
                 for k in ('Ex', 'Ey', 'Ez')]

    xg, yg, zg = maps['output_x'], maps['output_y'], maps['output_z']
    GX, GY, GZ = np.meshgrid(xg, yg, zg, indexing='ij')
    eval_pos = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)

    v_tab, E_tab = build_vinv_table(T)
    E_rec = np.array(recover_efield(sp, jnp.array(eval_pos), E0, v0, v_tab, E_tab,
                                    jnp.array(norm_off), jnp.array(norm_sc), 5.0))
    Ex_t = ef_interp[0](eval_pos); Ey_t = ef_interp[1](eval_pos); Ez_t = ef_interp[2](eval_pos)
    Emag_t = np.sqrt(Ex_t**2 + Ey_t**2 + Ez_t**2)
    Emag_r = np.sqrt((E_rec**2).sum(-1))
    R_t, R_r = box_recomb(Emag_t), box_recomb(Emag_r)

    def mae(a, b): return float(np.mean(np.abs(a - b)))
    print("\n" + "=" * 60)
    print(f"SCE-SIREN recovery (JAXTPC {Lx:.0f}x{Ly:.0f}x{Lz:.0f}, E0={E0:.0f} V/cm)")
    print("=" * 60)
    print(f"  Ex MAE        : {mae(E_rec[:,0], Ex_t):8.3f} V/cm  "
          f"(true range [{Ex_t.min():.1f},{Ex_t.max():.1f}])")
    print(f"  Ey MAE        : {mae(E_rec[:,1], Ey_t):8.3f} V/cm")
    print(f"  Ez MAE        : {mae(E_rec[:,2], Ez_t):8.3f} V/cm")
    print(f"  |E| rel error : {np.mean(np.abs(Emag_r-Emag_t)/Emag_t):8.4%} mean, "
          f"{np.max(np.abs(Emag_r-Emag_t)/Emag_t):.4%} max")
    print(f"  R   rel error : {np.mean(np.abs(R_r-R_t)/R_t):8.4%} mean, "
          f"{np.max(np.abs(R_r-R_t)/R_t):.4%} max")
    print("=" * 60)


if __name__ == '__main__':
    main()
