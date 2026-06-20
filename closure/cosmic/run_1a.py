#!/usr/bin/env python3
"""Test 1a: recover the SCE field through the sim and score it against the
FIRST-PRINCIPLES field (the real Poisson+ray-trace map), not the SIREN fit.

Exposes the representation limit hidden by the inverse crime: the recovery is a
SIREN, so its distance to the real field is bounded by how well a SIREN of that
capacity can represent the field — independent of optimisation / noise.

Usage: run_1a.py --truth <npz> --steps N  (truth made by make_truth_40cm at a
given omega; we recover it and compare to the first-principles field at Q=3e-8).
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


def firstprinciples_E(gen_pts):
    params = build_params(preset='jaxtpc', overrides=dict(
        Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=Q,
        Nx_poisson=41, Ny_poisson=41, Nz_poisson=41, Nx_output=15, Ny_output=15, Nz_output=15))
    maps = run(params)
    xp, yp, zp = maps['x_poisson'], maps['y_poisson'], maps['z_poisson']
    efi = [RegularGridInterpolator((xp, yp, zp), maps[k], bounds_error=False,
           fill_value=(E0 if k == 'Ex' else 0.)) for k in ('Ex', 'Ey', 'Ez')]
    return np.stack([efi[i](gen_pts) for i in range(3)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', required=True)
    ap.add_argument('--n-muons', type=int, default=512)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    P, D, S_ = [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., 160,
                                              logT, dedx, half_extents_mm=(200, 200, 200), step_mm=4.0)
        P.append(p); D.append(d); S_.append(float(s))
    hist, learned = recover_accum(sim, jnp.stack(P), jnp.stack(D), np.asarray(S_, np.float32),
                                  steps=args.steps, lr=3e-4, batch=16, record_every=500,
                                  val_frac=0.25, real_noise=True)

    # evaluate recovered, SIREN-truth, and first-principles |E| on a generator-frame grid
    d = dict(np.load(args.truth, allow_pickle=True)); om = float(d['omega_0'])
    nl = int(d['n_layers'])
    twb = {'weights': [jnp.asarray(d[f'w_{i}']) for i in range(nl)],
           'biases': [jnp.asarray(d[f'b_{i}']) for i in range(nl)]}
    gnorm = jnp.array([Lx / 2, Ly / 2, Lz / 2], np.float32)
    vt, et = S.build_vinv_table(T); v0 = float(drift_velocity(E0, T=T))
    g = np.linspace(0.05, 0.95, 12)
    GX, GY, GZ = np.meshgrid(g * Lx, g * Ly, g * Lz, indexing='ij')
    gp = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)

    def Emag(wb):
        E = np.array(S.recover_efield({'weights': wb['weights'], 'biases': wb['biases']},
                     jnp.array(gp), E0, v0, vt, et, gnorm, gnorm, om))
        return np.sqrt((E ** 2).sum(-1))
    rwb = {'weights': learned['weights'], 'biases': learned['biases']}
    Er, Etru = Emag(rwb), Emag(twb)
    Efp = np.sqrt((firstprinciples_E(gp) ** 2).sum(-1))
    res = dict(truth=args.truth, omega=om, n_muons=args.n_muons, steps=args.steps,
               mae_recovered_vs_firstprinciples=float(np.mean(np.abs(Er - Efp))),
               mae_recovered_vs_siren=float(np.mean(np.abs(Er - Etru))),
               mae_siren_vs_firstprinciples=float(np.mean(np.abs(Etru - Efp))),
               last_loss_emae=float(hist['emae'][-1]))
    json.dump(res, open(args.out, 'w'))
    print(f"[omega={om}] recovered vs FIRST-PRINCIPLES = {res['mae_recovered_vs_firstprinciples']:.3f} | "
          f"vs SIREN = {res['mae_recovered_vs_siren']:.3f} | SIREN repr = {res['mae_siren_vs_firstprinciples']:.3f} V/cm")


if __name__ == '__main__':
    main()
