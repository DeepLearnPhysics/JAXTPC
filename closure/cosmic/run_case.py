#!/usr/bin/env python3
"""Run ONE full-field SCE recovery case and write a JSON result.

Designed for parallel scaling studies: launch several instances pinned to
different GPUs (CUDA_VISIBLE_DEVICES) to sweep the number of cosmic tracks.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import jax
from closure.cosmic.recover_field import build, recover_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tracks', type=int, required=True)
    ap.add_argument('--seg-per-track', type=int, default=16)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--omega', type=float, default=1.5)
    ap.add_argument('--seed-init', type=int, default=7)
    ap.add_argument('--reg', type=float, default=0.0)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    n_seg = args.n_tracks * args.seg_per_track
    t0 = time.time()
    sim, pos, de, step = build(n_seg, truth_scale=1.0,
                               n_tracks=args.n_tracks, omega_0=args.omega)
    hist, _ = recover_full(sim, pos, de, step, steps=args.steps, lr=args.lr,
                           seed=args.seed_init, reg_weight=args.reg)
    import numpy as _np
    res = {
        'n_tracks': args.n_tracks, 'seg_per_track': args.seg_per_track,
        'n_seg': n_seg, 'steps': args.steps, 'lr': args.lr, 'omega': args.omega,
        'reg': args.reg,
        'init_mae': hist['emae'][0], 'final_mae': min(hist['emae']),
        'best_step': int(_np.argmin(hist['emae'])),
        'last_mae': hist['emae'][-1],
        'init_loss': hist['loss'][0], 'final_loss': hist['loss'][-1],
        'emae': hist['emae'], 'loss': hist['loss'],
        'seconds': time.time() - t0, 'device': str(jax.devices()[0]),
    }
    with open(args.out, 'w') as f:
        json.dump(res, f)
    print(f"[n_tracks={args.n_tracks}] |E|MAE {res['init_mae']:.2f} → "
          f"{res['final_mae']:.2f} V/cm  ({res['seconds']:.0f}s, {res['device']})")


if __name__ == '__main__':
    main()
