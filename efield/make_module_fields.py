#!/usr/bin/env python3
"""
Generate and validate a per-module SCE-SIREN field for each detector module.

This is the production tool behind the per-module SCE path: it produces one
trained SIREN ``.npz`` per module, ready to pass to the simulator as a list

    DetectorSimulator(..., distortion=['mod0.npz', 'mod1.npz', ...])

Each module can have its own space-charge conditions (e.g. a different charge
production rate Q for a module nearer a high-cosmic-flux region, or a different
drift length / E0). The fields are first-principles-generated (Poisson + ray
trace), SIREN-fit, and *each module's recovery is validated against its own
ground-truth maps* — so "different cases for different modules" is proven
physically, not just structurally.

Because the simulator stacks these into one uniform body, the modules differ
only in the trained weights/metadata — no recompile, no per-module code.

Quick 2-module demo (~5-8 min CPU), two different Q values:
    JAX_PLATFORM_NAME=cpu python3 efield/make_module_fields.py --quick \
        --module surface:3.0e-11 --module hotspot:2.0e-10
"""
import argparse
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ElectricDistortion.io.config_loader import build_params      # noqa: E402
from ElectricDistortion.run_sce import run                        # noqa: E402
from ElectricDistortion.core.drift_velocity import drift_velocity  # noqa: E402
from tools.sce_siren import (                                      # noqa: E402
    train_siren, recover_efield, save_siren_npz, build_vinv_table)

from validate_siren_wiring import build_line_dataset, box_recomb   # noqa: E402


def gen_and_fit(label, Q, quick, epochs, outdir):
    """First-principles maps → SIREN fit → save → recovery validation."""
    grid = (dict(Nx_poisson=51, Ny_poisson=51, Nz_poisson=51,
                 Nx_output=17, Ny_output=17, Nz_output=17) if quick else
            dict(Nx_poisson=81, Ny_poisson=81, Nz_poisson=81,
                 Nx_output=25, Ny_output=25, Nz_output=25))
    params = build_params(preset='jaxtpc',
                          overrides={**grid, 'Q_charge_production': Q})
    Lx, Ly, Lz, E0, T = (params['Lx'], params['Ly'], params['Lz'],
                         params['E0'], params.get('temperature', 89.0))
    print(f"\n=== module '{label}': Q={Q:.2e} C/m^3/s, "
          f"{Lx:.0f}x{Ly:.0f}x{Lz:.0f} cm, E0={E0:.0f} V/cm ===")
    maps = run(params)
    v0 = float(drift_velocity(E0, T=T))

    norm = np.array([Lx / 2, Ly / 2, Lz / 2], np.float32)
    pos, corr, n_per_line = build_line_dataset(maps, Lx, Ly, Lz)
    t0 = time.time()
    sp = train_siren(pos, corr, jnp.array(norm), jnp.array(norm),
                     omega_0=5.0, n_epochs=epochs, n_per_line=n_per_line,
                     verbose=False)
    out = os.path.join(outdir, f'sce_siren_{label}.npz')
    save_siren_npz(out, sp, 5.0, norm, norm, E0, T,
                   extra=dict(Lx=Lx, Ly=Ly, Lz=Lz, Q=Q))
    print(f"  trained in {time.time()-t0:.0f}s → {out}")

    # Recovery validation vs this module's own ground truth
    x_p, y_p, z_p = maps['x_poisson'], maps['y_poisson'], maps['z_poisson']
    from scipy.interpolate import RegularGridInterpolator
    efi = [RegularGridInterpolator((x_p, y_p, z_p), maps[k], method='linear',
           bounds_error=False, fill_value=(E0 if k == 'Ex' else 0.0))
           for k in ('Ex', 'Ey', 'Ez')]
    GX, GY, GZ = np.meshgrid(maps['output_x'], maps['output_y'], maps['output_z'],
                             indexing='ij')
    ep = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)
    v_tab, E_tab = build_vinv_table(T)
    Er = np.array(recover_efield(sp, jnp.array(ep), E0, v0, v_tab, E_tab,
                                 jnp.array(norm), jnp.array(norm), 5.0))
    Et = np.stack([efi[i](ep) for i in range(3)], -1)
    Emag_t = np.sqrt((Et ** 2).sum(-1)); Emag_r = np.sqrt((Er ** 2).sum(-1))
    R_t, R_r = box_recomb(Emag_t), box_recomb(Emag_r)
    return dict(label=label, Q=Q, out=out,
                Ex_mae=float(np.mean(np.abs(Er[:, 0] - Et[:, 0]))),
                Emag_rel=float(np.mean(np.abs(Emag_r - Emag_t) / Emag_t)),
                R_rel=float(np.mean(np.abs(R_r - R_t) / R_t)),
                sce_pct=float(max(abs(1 - Emag_t.min() / E0),
                                  abs(Emag_t.max() / E0 - 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', action='append', default=[],
                    help='label:Q  (repeatable). Default: two modules.')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--outdir', default=os.path.dirname(__file__))
    args = ap.parse_args()

    specs = args.module or ['surface:3.0e-11', 'hotspot:2.0e-10']
    mods = [(s.split(':')[0], float(s.split(':')[1])) for s in specs]

    results = [gen_and_fit(lbl, Q, args.quick, args.epochs, args.outdir)
               for lbl, Q in mods]

    print("\n" + "=" * 74)
    print(f"Per-module SCE-SIREN fields  ({len(results)} module(s))")
    print("=" * 74)
    print(f"{'module':<12}{'Q (C/m^3/s)':>14}{'SCE %':>9}"
          f"{'Ex MAE':>10}{'|E| rel':>10}{'R rel':>10}")
    print("-" * 74)
    for r in results:
        print(f"{r['label']:<12}{r['Q']:>14.2e}{r['sce_pct']:>8.2%}"
              f"{r['Ex_mae']:>9.3f}V{r['Emag_rel']:>9.4%}{r['R_rel']:>9.4%}")
    print("=" * 74)
    print("Pass to the simulator (order = volume order):")
    print("  distortion=[" +
          ", ".join(f"'{r['out']}'" for r in results) + "]")


if __name__ == '__main__':
    main()
