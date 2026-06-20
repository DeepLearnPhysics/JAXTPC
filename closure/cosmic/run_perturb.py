#!/usr/bin/env python3
"""Test 2: robustness to cosmic-track reconstruction error.

The OBSERVED signal comes from the TRUE tracks; the recovery MODEL only knows the
reconstructed tracks, emulated by jittering each muon's entrance/exit endpoints by
sigma (mm). Jittering endpoints induces correlated position + angle + direction
errors. We sweep sigma and measure how the field recovery degrades.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build, recover_accum
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints

LO, HI = (-200., -200., -200.), (0., 200., 200.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, required=True, help='endpoint jitter (mm)')
    ap.add_argument('--n-muons', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    jr = np.random.RandomState(123)  # separate stream for the reco jitter
    Pt, Dt, St, Pm, Dm, Sm = [], [], [], [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., 160,
                                              logT, dedx, half_extents_mm=(200, 200, 200), step_mm=4.0)
        Pt.append(p); Dt.append(d); St.append(float(s))
        # perturbed (reconstructed) endpoints -> model track
        am = a + jr.normal(size=3) * args.sigma
        bm = b + jr.normal(size=3) * args.sigma
        pm, dm, _, _, sm = generate_cosmic_chord(jnp.array(am), jnp.array(bm), 4000., 160,
                                                 logT, dedx, half_extents_mm=(200, 200, 200), step_mm=4.0)
        Pm.append(pm); Dm.append(dm); Sm.append(float(sm))

    hist, _ = recover_accum(
        sim, jnp.stack(Pt), jnp.stack(Dt), np.asarray(St, np.float32),
        steps=args.steps, lr=3e-4, batch=16, record_every=500, val_frac=0.25, real_noise=True,
        pos_model=jnp.stack(Pm), de_model=jnp.stack(Dm), step_model=np.asarray(Sm, np.float32))

    e = np.array(hist['emae']); v = np.array(hist['val'])
    vstep = int(v.argmin()) if len(v) and v.min() < np.inf else len(e) - 1
    res = dict(sigma_mm=args.sigma, n_muons=args.n_muons, steps=args.steps,
               val_mae=float(e[vstep]), last_mae=float(e[-1]), best_mae=float(e.min()),
               emae=hist['emae'])
    json.dump(res, open(args.out, 'w'))
    print(f"[sigma={args.sigma}mm] val-selected |E|MAE = {res['val_mae']:.2f} "
          f"(last {res['last_mae']:.2f}, best {res['best_mae']:.2f}) V/cm")


if __name__ == '__main__':
    main()
