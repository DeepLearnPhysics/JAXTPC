#!/usr/bin/env python3
"""Expressivity vs the REAL (first-principles Poisson) field: does PROPER MODELING
(a curl-free potential, E=-grad phi) represent the real field better per DOF than an
unconstrained vector model? Offline, linear least-squares, out-of-sample.

The real SCE field is conservative (E=-grad phi from Poisson). An unconstrained vector
model spends DOF on the non-conservative subspace it can never need; a potential model
spends all DOF on physical (curl-free) fields. If the potential reaches the same |E|
accuracy at fewer DOF (or lower at equal DOF), that's the proper-modeling payoff.
We have >>enough muons, so the operative limit is this model expressivity floor.
"""
import os, sys, itertools
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'efield'))
from ElectricDistortion.io.config_loader import build_params
from ElectricDistortion.run_sce import run
from scipy.interpolate import RegularGridInterpolator

Lx, Ly, Lz, E0, Q = 20., 40., 40., 500., 3e-8


def firstprinciples_E(gen_pts):
    params = build_params(preset='jaxtpc', overrides=dict(
        Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=Q,
        Nx_poisson=41, Ny_poisson=41, Nz_poisson=41, Nx_output=15, Ny_output=15, Nz_output=15))
    maps = run(params)
    xp, yp, zp = maps['x_poisson'], maps['y_poisson'], maps['z_poisson']
    efi = [RegularGridInterpolator((xp, yp, zp), maps[k], bounds_error=False,
           fill_value=(E0 if k == 'Ex' else 0.)) for k in ('Ex', 'Ey', 'Ez')]
    return np.stack([efi[i](gen_pts) for i in range(3)], -1)


def grid(n):
    g = np.linspace(0.08, 0.92, n)
    GX, GY, GZ = np.meshgrid(g * Lx, g * Ly, g * Lz, indexing='ij')
    return np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)


def main():
    gfit, gev = grid(12), grid(17)
    Ef, Ee = firstprinciples_E(gfit), firstprinciples_E(gev)
    mag_e = np.sqrt((Ee ** 2).sum(-1))
    print(f"real |E| on eval grid: mean={mag_e.mean():.1f} std={mag_e.std():.1f} min={mag_e.min():.1f} max={mag_e.max():.1f} V/cm")
    print("(SIREN-Delta reference floor vs real ~1.06-1.66 V/cm from run_1a)\n")
    cen = jnp.array([Lx / 2, Ly / 2, Lz / 2]); scl = jnp.array([Lx / 2, Ly / 2, Lz / 2])

    def make_feats(d):                       # INTEGER-exponent monomials (grad-safe for negative coords)
        exps = [(a, b, c) for a in range(d + 1) for b in range(d + 1) for c in range(d + 1) if a + b + c <= d]
        def feats(pts):
            xn = (jnp.asarray(pts) - cen) / scl
            cols = [xn[..., 0] ** a * xn[..., 1] ** b * xn[..., 2] ** c for (a, b, c) in exps]
            return jnp.stack(cols, -1)
        return feats, len(exps)

    def emae(Epred): return float(np.mean(np.abs(np.sqrt((Epred ** 2).sum(-1)) - mag_e)))

    print(f"{'model':>22} {'E-deg':>5} {'DOF':>5} {'|E|MAE(out-of-sample)':>22}")
    # (A) unconstrained vector model: 3 independent polynomials for Ex,Ey,Ez
    for d in [2, 3, 4, 5, 6]:
        feats, ncoef = make_feats(d)
        Pf, Pe = np.asarray(feats(gfit)), np.asarray(feats(gev))
        C = np.stack([np.linalg.lstsq(Pf, Ef[:, i], rcond=None)[0] for i in range(3)], -1)
        print(f"{'vector E (uncon)':>22} {d:>5} {ncoef*3:>5} {emae(Pe @ C):>22.3f}")
    print()
    # (B) potential model: phi poly of degree d+1 -> E = -grad phi (curl-free), so E is degree d
    for d in [2, 3, 4, 5, 6]:
        feats, nc = make_feats(d + 1)
        def Emodel_mat(pts):                 # E = -grad phi ; linear in phi coeffs -> design (N,3,ncoef)
            J = jax.vmap(jax.jacfwd(lambda x: feats(x[None])[0]))(jnp.asarray(pts))  # (N,ncoef,3)
            return -np.asarray(jnp.transpose(J, (0, 2, 1)))                          # (N,3,ncoef)
        Mf, Me = Emodel_mat(gfit), Emodel_mat(gev)
        A = Mf.reshape(-1, nc); y = Ef.reshape(-1)
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        Epred = (Me.reshape(-1, nc) @ c).reshape(-1, 3)
        print(f"{'potential -grad phi':>22} {d:>5} {nc:>5} {emae(Epred):>22.3f}")


if __name__ == '__main__':
    main()
