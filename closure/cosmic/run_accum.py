#!/usr/bin/env python3
"""Per-event accumulation: recover the SCE field from a POOL of M cosmic muons,
each treated as its own event (separate forward/image), mini-batched.
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

from closure.cosmic.recover_field import build, recover_accum
from tools.particle_generator import (
    load_dedx_table_jax, generate_cosmic_chord, sample_surface_endpoints)

HALF = (200.0, 200.0, 200.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, required=True)
    ap.add_argument('--seg', type=int, default=32)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--curl', type=float, default=0.0)
    ap.add_argument('--record-every', type=int, default=1)
    ap.add_argument('--noise', type=float, default=0.0, help='intrinsic noise sigma (ENC) on observed data')
    ap.add_argument('--truth', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # one sim compiled for a single S-segment event
    sim, _, _, _ = build(args.seg, truth_scale=1.0, n_tracks=1, truth_npz=args.truth)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    P, D, steps_mm = [], [], []
    for _ in range(args.n_muons):
        a, b = sample_surface_endpoints(rng, HALF)
        a[0] = np.clip(a[0], -200, 0); b[0] = np.clip(b[0], -200, 0)
        p, d, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., args.seg,
                                              logT, dedx, half_extents_mm=HALF)
        P.append(p); D.append(d); steps_mm.append(float(s))
    pos_all = jnp.stack(P); de_all = jnp.stack(D)
    step = float(np.mean(steps_mm))   # ~uniform chord segment length

    hist, _ = recover_accum(sim, pos_all, de_all, step, steps=args.steps,
                            lr=args.lr, batch=args.batch, curl_weight=args.curl,
                            record_every=args.record_every, noise_sigma=args.noise)
    e = np.array(hist['emae'])
    res = dict(n_muons=args.n_muons, seg=args.seg, batch=args.batch, steps=args.steps,
               curl_weight=args.curl, noise=args.noise,
               init_mae=hist['emae'][0], best_mae=float(e.min()),
               best_step=int(e.argmin()), last_mae=hist['emae'][-1],
               init_curl=hist['curl'][0], last_curl=hist['curl'][-1],
               final_loss=hist['loss'][-1], emae=hist['emae'], curl=hist['curl'])
    json.dump(res, open(args.out, 'w'))
    print(f"[M={args.n_muons} λ={args.curl}] |E|MAE {res['init_mae']:.2f}→{res['last_mae']:.2f} "
          f"curl {res['init_curl']:.3f}→{res['last_curl']:.3f}")


if __name__ == '__main__':
    main()
