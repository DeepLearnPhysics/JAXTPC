#!/usr/bin/env python3
"""Recover a field and show truth/learned/residual |E| across several slice
planes (z-depths and center-vs-near-wall) to inspect where recovery holds."""
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tools.sce_siren as S
from closure.cosmic.recover_field import build, recover_accum
from tools.particle_generator import (
    load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints)

HALF = (200.0, 200.0, 200.0)
LO, HI = (-200.0, -200.0, -200.0), (0.0, 200.0, 200.0)   # actual drift box (no clip)


def emag_on(sim, stacked, pts):
    sb = sim._sce_siren
    p0 = jax.tree.map(lambda x: x[0], stacked)
    par = {'weights': p0['weights'], 'biases': p0['biases']}
    E = S.recover_efield(par, jnp.asarray(pts, jnp.float32), p0['E0'], p0['v0'],
                         sb['v_table'], sb['E_table'], p0['norm_offsets'],
                         p0['norm_scales'], sb['omega_0'])
    return np.array(jnp.sqrt((E ** 2).sum(-1)))


def plane_grid(kind, fixed, n=60):
    a = np.linspace(0.5, 19.5, n)       # drift x
    b = np.linspace(-19, 19, n)         # transverse
    if kind == 'xy':   # vary x,y at z=fixed
        X, Y = np.meshgrid(a, b, indexing='ij'); Z = np.full(X.size, fixed)
        return a, b, np.stack([X.ravel(), Y.ravel(), Z], -1), 'drift x (cm)', 'y (cm)'
    if kind == 'xz':   # vary x,z at y=fixed
        X, Z = np.meshgrid(a, b, indexing='ij'); Y = np.full(X.size, fixed)
        return a, b, np.stack([X.ravel(), Y, Z.ravel()], -1), 'drift x (cm)', 'z (cm)'
    if kind == 'yz':   # vary y,z at x=fixed
        Y, Z = np.meshgrid(b, b, indexing='ij'); Xc = np.full(Y.size, fixed)
        return b, b, np.stack([Xc, Y.ravel(), Z.ravel()], -1), 'y (cm)', 'z (cm)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', default='closure/cosmic/truth_40cm_edge.npz')
    ap.add_argument('--n-muons', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=10000)
    args = ap.parse_args()

    sim, _, _, _ = build(32, 1.0, n_tracks=1, truth_npz=args.truth)
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    P, D, S = [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., 32,
                                              logT, dedx, half_extents_mm=HALF)
        P.append(p); D.append(d); S.append(float(s))
    truth = sim._default_sim_params.sce_models
    print(f"recovering M={args.n_muons}, {args.steps} steps ...")
    hist, learned = recover_accum(sim, jnp.stack(P), jnp.stack(D), np.asarray(S, np.float32),
                                  steps=args.steps, lr=3e-4, record_every=200)
    print(f"  |E| MAE → {hist['emae'][-1]:.2f} V/cm")

    # slice planes: 3 z-depths (x-y), center vs near-wall (x-z), mid-drift (y-z)
    planes = [('xy', 0.0, 'x–y  z=0'),
              ('xy', 15.0, 'x–y  z=+15'),
              ('xz', 0.0, 'x–z  y=0 (center)'),
              ('xz', 17.0, 'x–z  y=+17 (near wall)'),
              ('yz', 10.0, 'y–z  x=10 (mid-drift)')]
    fig, ax = plt.subplots(len(planes), 3, figsize=(11, 3.0 * len(planes)))
    for r, (kind, fx, lab) in enumerate(planes):
        u, v, pts, xl, yl = plane_grid(kind, fx)
        n = len(u)
        Et = emag_on(sim, truth, pts).reshape(n, n)
        El = emag_on(sim, learned, pts).reshape(n, n)
        ext = [u[0], u[-1], v[0], v[-1]]
        vmin, vmax = min(Et.min(), El.min()), max(Et.max(), El.max())
        for c, (F, t) in enumerate([(Et, 'truth'), (El, 'learned')]):
            im = ax[r, c].imshow(F.T, origin='lower', extent=ext, aspect='auto',
                                 vmin=vmin, vmax=vmax, cmap='viridis')
            ax[r, c].set_title(f'|E| {t}  [{lab}]', fontsize=9)
            ax[r, c].set_xlabel(xl, fontsize=8); ax[r, c].set_ylabel(yl, fontsize=8)
            fig.colorbar(im, ax=ax[r, c])
        res = El - Et; m = float(np.abs(res).max()) or 1.0
        im = ax[r, 2].imshow(res.T, origin='lower', extent=ext, aspect='auto',
                             vmin=-m, vmax=m, cmap='RdBu_r')
        ax[r, 2].set_title(f'learned−truth  (max {m:.1f})', fontsize=9)
        ax[r, 2].set_xlabel(xl, fontsize=8); fig.colorbar(im, ax=ax[r, 2])

    fig.suptitle(f'Edge-field recovery across slices  ({args.n_muons} muons, '
                 f'|E| MAE {hist["emae"][-1]:.2f} V/cm)', fontsize=12)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edge_slices.png')
    fig.savefig(out, dpi=120, bbox_inches='tight'); print(f"saved {out}")


if __name__ == '__main__':
    main()
