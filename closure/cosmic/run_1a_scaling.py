#!/usr/bin/env python3
"""1a scaling: how the recovered-vs-REAL (first-principles) error scales with
muons and optimization steps. Records recovered|E|-vs-first-principles each step
(the honest metric), alongside the vs-SIREN error, for one truth/omega.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'efield'))
from closure.cosmic.recover_field import build, recover_accum
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints
import tools.sce_siren as S
from ElectricDistortion.io.config_loader import build_params
from ElectricDistortion.run_sce import run
from ElectricDistortion.core.drift_velocity import drift_velocity
from scipy.interpolate import RegularGridInterpolator

LO, HI = (-200., -200., -200.), (0., 200., 200.)
Lx, Ly, Lz, E0, T, Q = 20., 40., 40., 500., 89., 3e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', required=True)
    ap.add_argument('--n-muons', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    d = dict(np.load(args.truth, allow_pickle=True)); om = float(d['omega_0'])
    gnorm = jnp.array([Lx / 2, Ly / 2, Lz / 2], np.float32)
    vt, et = S.build_vinv_table(T); v0 = float(drift_velocity(E0, T=T))

    # first-principles |E| on an interior generator-frame grid
    params = build_params(preset='jaxtpc', overrides=dict(
        Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=Q,
        Nx_poisson=41, Ny_poisson=41, Nz_poisson=41, Nx_output=15, Ny_output=15, Nz_output=15))
    maps = run(params)
    xp, yp, zp = maps['x_poisson'], maps['y_poisson'], maps['z_poisson']
    efi = [RegularGridInterpolator((xp, yp, zp), maps[k], bounds_error=False,
           fill_value=(E0 if k == 'Ex' else 0.)) for k in ('Ex', 'Ey', 'Ez')]
    g = np.linspace(0.05, 0.95, 12)
    GX, GY, GZ = np.meshgrid(g * Lx, g * Ly, g * Lz, indexing='ij')
    gp = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)
    Efp = jnp.asarray(np.sqrt((np.stack([efi[i](gp) for i in range(3)], -1) ** 2).sum(-1)))
    gp_j = jnp.asarray(gp)

    def extra_metric(stacked):
        wb = {'weights': [w[0] for w in stacked['weights']],
              'biases': [b[0] for b in stacked['biases']]}
        E = S.recover_efield(wb, gp_j, E0, v0, vt, et, gnorm, gnorm, om)
        return jnp.mean(jnp.abs(jnp.sqrt((E ** 2).sum(-1)) - Efp))

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    P, D, Sm = [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, dd, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., 160,
                                               logT, dedx, half_extents_mm=(200, 200, 200), step_mm=4.0)
        P.append(p); D.append(dd); Sm.append(float(s))
    hist, _ = recover_accum(sim, jnp.stack(P), jnp.stack(D), np.asarray(Sm, np.float32),
                            steps=args.steps, lr=3e-4, batch=16, record_every=1000,
                            val_frac=0.25, real_noise=True, extra_metric=extra_metric)
    res = dict(truth=args.truth, omega=om, n_muons=args.n_muons, steps=args.steps,
               emae_siren=hist['emae'], emae_real=hist['emae_real'])
    json.dump(res, open(args.out, 'w'))
    er = np.array(hist['emae_real'])
    print(f"[omega={om} M={args.n_muons}] recovered-vs-REAL: {er[0]:.2f} -> {er[-1]:.2f} V/cm "
          f"(min {er.min():.2f}); vs-SIREN last {hist['emae'][-1]:.2f}")


if __name__ == '__main__':
    main()
