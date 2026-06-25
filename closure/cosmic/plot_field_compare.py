#!/usr/bin/env python3
"""Render the LEARNED SCE field against the first-principles TRUTH after
per-event accumulation recovery: |E| and distortion Δ slices + scatter."""
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

HALF = (200.0, 200.0, 200.0); E0, T = 500.0, 89.0
LO, HI = (-200.0, -200.0, -200.0), (0.0, 200.0, 200.0)   # actual drift box (no clip)


def field_slice(sim, stacked, z=0.0, n=60):
    sb = sim.distortion_state()
    xs = np.linspace(0.5, 19.5, n); ys = np.linspace(-19, 19, n)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    g = jnp.array(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z)], -1), jnp.float32)
    p0 = jax.tree.map(lambda x: x[0], stacked)
    par = {'weights': p0['weights'], 'biases': p0['biases']}
    E = S.recover_efield(par, g, p0['E0'], p0['v0'], sb['v_table'], sb['E_table'],
                         p0['norm_offsets'], p0['norm_scales'], sb['omega_0'])
    d = S.siren_delta(par, g, p0['norm_offsets'], p0['norm_scales'], sb['omega_0'])
    Emag = np.array(jnp.sqrt((E ** 2).sum(-1))).reshape(n, n)
    dx = np.array(d[:, 0]).reshape(n, n)
    return xs, ys, Emag, dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=128)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--truth', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'truth_40cm.npz'))
    args = ap.parse_args()

    sim, _, _, _ = build(32, truth_scale=1.0, n_tracks=1, truth_npz=args.truth)
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    P, D = [], []
    S = []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., 32,
                                              logT, dedx, half_extents_mm=HALF)
        P.append(p); D.append(d); S.append(float(s))
    pos_all, de_all = jnp.stack(P), jnp.stack(D)
    truth = sim._default_sim_params.distortion_field

    print(f"recovering full field from M={args.n_muons} muons, {args.steps} steps ...")
    hist, learned = recover_accum(sim, pos_all, de_all, np.asarray(S, np.float32),
                                  steps=args.steps, lr=3e-4)
    print(f"  |E| MAE init {hist['emae'][0]:.2f} → {hist['emae'][-1]:.2f} V/cm")

    xs, ys, Et, dxt = field_slice(sim, truth)
    _, _, El, dxl = field_slice(sim, learned)
    ext = [xs[0], xs[-1], ys[0], ys[-1]]

    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for r, (Ft, Fl, lab, unit) in enumerate([(Et, El, '|E|', 'V/cm'),
                                             (dxt, dxl, r'$\Delta_x$', 'cm')]):
        vmin, vmax = min(Ft.min(), Fl.min()), max(Ft.max(), Fl.max())
        for c, (F, t) in enumerate([(Ft, f'{lab} TRUTH'), (Fl, f'{lab} LEARNED')]):
            im = ax[r, c].imshow(F.T, origin='lower', extent=ext, aspect='auto',
                                 vmin=vmin, vmax=vmax, cmap='viridis')
            ax[r, c].set(title=f'{t} [{unit}]', xlabel='drift x (cm)', ylabel='y (cm)')
            fig.colorbar(im, ax=ax[r, c])
        res = Fl - Ft; m = float(np.abs(res).max()) or 1.0
        im = ax[r, 2].imshow(res.T, origin='lower', extent=ext, aspect='auto',
                             vmin=-m, vmax=m, cmap='RdBu_r')
        ax[r, 2].set(title=f'{lab} learned−truth [{unit}]', xlabel='drift x', ylabel='y')
        fig.colorbar(im, ax=ax[r, 2])
        ax[r, 3].scatter(Ft.ravel(), Fl.ravel(), s=2, alpha=0.3)
        lim = [min(Ft.min(), Fl.min()), max(Ft.max(), Fl.max())]
        ax[r, 3].plot(lim, lim, 'k--', lw=1)
        ax[r, 3].set(title=f'{lab}: truth vs learned', xlabel=f'truth [{unit}]',
                     ylabel=f'learned [{unit}]')

    tag = os.path.splitext(os.path.basename(args.truth))[0]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'learned_vs_{tag}.png')
    fig.suptitle(f'Learned vs truth SCE field  (full SIREN, {args.n_muons} accumulated muons, '
                 f'|E| MAE {hist["emae"][-1]:.2f} V/cm)', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == '__main__':
    main()
