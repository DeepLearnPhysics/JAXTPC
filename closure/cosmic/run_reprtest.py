#!/usr/bin/env python3
"""Is the field intrinsically low-DOF, and is the error at the BOUNDARIES?

Fit the TRUTH E-field with a plain 3D polynomial basis (linear in coeffs -> convex,
provably more learnable than the SIREN; no Poisson/integration) at increasing degree.
- DOF-vs-|E|MAE curve: if a low-degree poly (~100 DOF) reaches <few V/cm, the field
  is intrinsically low-DOF and the SIREN's thousands are redundant -> the K-subspace
  failure was a BASIS problem, and a direct low-DOF spatial basis is the fix (no
  physics/integration needed). If it needs high degree, the field is genuinely high-DOF.
- Residual vs distance-to-boundary: tells us whether the hard part is the boundaries
  (fast-varying near-wall SCE) or the bulk -- instead of assuming.
Fit on a coarse grid, EVALUATE on a finer grid (out-of-sample) so it's a real
representation test, not an interpolation artifact.
"""
import os, sys, itertools
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build
import tools.sce_siren as S


def efield(sim, stk, grid):
    sb = sim.distortion_state(); p0 = jax.tree.map(lambda x: x[0], stk)
    return np.asarray(S.recover_efield({'weights': p0['weights'], 'biases': p0['biases']}, jnp.asarray(grid),
                                       p0['E0'], p0['v0'], sb['v_table'], sb['E_table'],
                                       p0['norm_offsets'], p0['norm_scales'], sb['omega_0']))


def make_grid(n):
    gx, gy, gz = np.meshgrid(np.linspace(0.5, 19.5, n), np.linspace(-19.5, 19.5, n),
                             np.linspace(-19.5, 19.5, n), indexing='ij')
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1).astype(np.float32)


def poly_features(g, d):
    # normalize coords to ~[-1,1] for conditioning
    x = (g[:, 0] - 10.) / 10.; y = g[:, 1] / 20.; z = g[:, 2] / 20.
    cols = [x ** a * y ** b * z ** c for a, b, c in itertools.product(range(d + 1), repeat=3) if a + b + c <= d]
    return np.stack(cols, -1)


def main():
    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=os.path.join(HERE, 'truth_40cm.npz'))
    truth = sim._default_sim_params.distortion_field
    gfit, gev = make_grid(12), make_grid(18)
    Ef, Ee = efield(sim, truth, gfit), efield(sim, truth, gev)
    mag_e = np.sqrt((Ee ** 2).sum(-1))
    print(f"truth |E| on eval grid: mean={mag_e.mean():.1f}  std={mag_e.std():.1f}  min={mag_e.min():.1f} max={mag_e.max():.1f} V/cm")
    print("deg  DOF   |E|MAE(out-of-sample)   (field-only K=48 random floor was 20.3, SVD 32.1)")
    best = None
    for d in [1, 2, 3, 4, 5, 6, 8]:
        Pf, Pe = poly_features(gfit, d), poly_features(gev, d)
        Efit = np.stack([np.linalg.lstsq(Pf, Ef[:, i], rcond=None)[0] for i in range(3)], -1)  # (ncoef,3)
        Epred = Pe @ Efit
        mae = np.mean(np.abs(np.sqrt((Epred ** 2).sum(-1)) - mag_e))
        print(f" {d}  {Pf.shape[1]*3:4d}   {mae:8.2f} V/cm")
        if d == 4:
            best = np.abs(np.sqrt((Epred ** 2).sum(-1)) - mag_e)
    # residual vs distance-to-nearest-boundary (deg-4 fit)
    db = np.minimum.reduce([gev[:, 0] - 0.5, 19.5 - gev[:, 0], gev[:, 1] + 19.5, 19.5 - gev[:, 1],
                            gev[:, 2] + 19.5, 19.5 - gev[:, 2]])
    print("\ndeg-4 residual vs distance-to-boundary (is the error at the walls?):")
    edges = [0, 2, 5, 10, 20]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (db >= lo) & (db < hi)
        if m.sum(): print(f"  boundary-dist [{lo:2d},{hi:2d}):  |E|MAE={best[m].mean():6.2f}  (n={int(m.sum())})")


if __name__ == '__main__':
    main()
