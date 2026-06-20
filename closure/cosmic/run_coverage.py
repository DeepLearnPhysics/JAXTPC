#!/usr/bin/env python3
"""Rung 3: recover the (real, first-principles) SCE field through the sim at a
given observation coverage, and write a JSON result.

Coverage kinds:
  dense:G    — G^3 deposits filling the volume (full coverage; rung-2 baseline)
  cosmic:K   — K surface-to-surface cosmic tracks (sparse, realistic)
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from closure.cosmic.recover_field import build, recover_full
from tools.particle_generator import (
    load_dedx_table_jax, generate_cosmic_chord, sample_surface_endpoints)

HALF = (200.0, 200.0, 200.0)


def dense_obs(g):
    xs = np.linspace(0.8, 19.2, g); ts = np.linspace(-18, 18, g)
    GX, GY, GZ = np.meshgrid(xs, ts, ts, indexing='ij')
    pos = np.stack([-GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype('float32') * 10.0
    return jnp.array(pos), jnp.full(pos.shape[0], 2.0, jnp.float32), 2.0


def cosmic_obs(k, seg=16):
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    P, D = [], []
    for _ in range(k):
        a, b = sample_surface_endpoints(rng, HALF)
        a[0] = np.clip(a[0], -200, 0); b[0] = np.clip(b[0], -200, 0)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., seg,
                                              logT, dedx, half_extents_mm=HALF)
        P.append(p); D.append(d)
    pos = jnp.concatenate(P); de = jnp.concatenate(D)
    return pos, de, float(jnp.linalg.norm(pos[1] - pos[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coverage', required=True, help='dense:G or cosmic:K')
    ap.add_argument('--truth', required=True)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--init-noise', type=float, default=0.10)
    ap.add_argument('--init-out-scale', type=float, default=0.5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    kind, n = args.coverage.split(':'); n = int(n)
    n_seg = n ** 3 if kind == 'dense' else n * 16
    sim, _, _, _ = build(n_seg, truth_scale=1.0, n_tracks=max(1, n if kind == 'cosmic' else 8),
                         truth_npz=args.truth)
    pos, de, step = dense_obs(n) if kind == 'dense' else cosmic_obs(n)

    hist, _ = recover_full(sim, pos, de, step, steps=args.steps, lr=args.lr, seed=7,
                           init_noise=args.init_noise, init_out_scale=args.init_out_scale)
    e = np.array(hist['emae'])
    res = dict(coverage=args.coverage, kind=kind, n=n, n_seg=n_seg, steps=args.steps,
               init_mae=hist['emae'][0], best_mae=float(e.min()),
               best_step=int(e.argmin()), last_mae=hist['emae'][-1],
               final_loss=hist['loss'][-1], emae=hist['emae'], loss=hist['loss'])
    json.dump(res, open(args.out, 'w'))
    print(f"[{args.coverage}] |E|MAE init {res['init_mae']:.2f} → best {res['best_mae']:.2f} "
          f"(last {res['last_mae']:.2f}) V/cm")


if __name__ == '__main__':
    main()
