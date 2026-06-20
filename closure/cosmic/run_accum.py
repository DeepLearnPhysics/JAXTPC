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
    load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints)

HALF = (200.0, 200.0, 200.0)
# actual drift box (mm): x in [-200,0] (anode at 0). Sample its real faces — do
# NOT clip a symmetric cube (that collapses x>0 endpoints onto the anode plane,
# making ~25% of tracks degenerate flat-at-anode with zero drift extent).
LO, HI = (-200.0, -200.0, -200.0), (0.0, 200.0, 200.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, required=True)
    ap.add_argument('--seg', type=int, default=32)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--curl', type=float, default=0.0)
    ap.add_argument('--record-every', type=int, default=1)
    ap.add_argument('--noise', type=float, default=0.0, help='IID white noise sigma (ENC) on observed data')
    ap.add_argument('--real-noise', action='store_true', help='use the actual MicroBooNE intrinsic noise model')
    ap.add_argument('--whitened', action='store_true', help='noise-whitened chi^2 loss (proper likelihood)')
    ap.add_argument('--zero-suppress', type=float, default=0.0, help='readout threshold (ENC) on noisy data')
    ap.add_argument('--val-frac', type=float, default=0.0, help='held-out muon fraction for early stopping')
    ap.add_argument('--step-mm', type=float, default=0.0, help='fixed segment dx (mm); 0 => chord/seg')
    ap.add_argument('--wd', type=float, default=0.0, help='weight decay (capacity control)')
    ap.add_argument('--truth', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # one sim compiled for a single S-segment event
    sim, _, _, _ = build(args.seg, truth_scale=1.0, n_tracks=1, truth_npz=args.truth)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    smm = args.step_mm if args.step_mm > 0 else None   # small fixed dx, or chord/seg
    P, D, steps_mm = [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., args.seg,
                                              logT, dedx, half_extents_mm=HALF, step_mm=smm)
        P.append(p); D.append(d); steps_mm.append(float(s))
    pos_all = jnp.stack(P); de_all = jnp.stack(D)
    step = float(steps_mm[0]) if smm else np.asarray(steps_mm, np.float32)

    hist, _ = recover_accum(sim, pos_all, de_all, step, steps=args.steps,
                            lr=args.lr, batch=args.batch, curl_weight=args.curl,
                            record_every=args.record_every, noise_sigma=args.noise,
                            zero_suppress=args.zero_suppress, val_frac=args.val_frac,
                            real_noise=args.real_noise, weight_decay=args.wd,
                            whitened=args.whitened)
    e = np.array(hist['emae']); v = np.array(hist['val'])
    # HONEST metric: |E| MAE at the step that minimises the held-out VAL loss
    # (no peeking at truth). Falls back to last-step when no validation set.
    val_step = int(v.argmin()) if args.val_frac > 0 else len(e) - 1
    res = dict(n_muons=args.n_muons, seg=args.seg, batch=args.batch, steps=args.steps,
               curl_weight=args.curl, noise=args.noise, zero_suppress=args.zero_suppress,
               val_frac=args.val_frac, step_mm=args.step_mm,
               init_mae=hist['emae'][0], best_mae=float(e.min()), best_step=int(e.argmin()),
               last_mae=hist['emae'][-1], val_step=val_step, val_mae=float(e[val_step]),
               final_loss=hist['loss'][-1], emae=hist['emae'], curl=hist['curl'], val=hist['val'])
    json.dump(res, open(args.out, 'w'))
    print(f"[M={args.n_muons} σ={args.noise} zs={args.zero_suppress}] |E|MAE init {res['init_mae']:.2f} "
          f"-> val-selected {res['val_mae']:.2f} (last {res['last_mae']:.2f}, oracle-best {res['best_mae']:.2f})")


if __name__ == '__main__':
    main()
