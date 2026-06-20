#!/usr/bin/env python3
"""Test 3: known endpoints (CRT), but the track ISN'T straight — multiple
Coulomb scattering (energy-dependent). The OBS uses the true scattered trajectory;
the recovery MODEL assumes a STRAIGHT line between the (known) endpoints. Sweep the
scattering RMS, which maps to muon momentum via Highland.

Highland (LAr, ~50 cm, X0=14 cm): lateral RMS ~ 7800/p[MeV] mm, so
  scatter_rms 1/2/5/10 mm <-> p ~ 7.8 / 3.9 / 1.6 / 0.8 GeV.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build, recover_accum
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints

LO, HI = (-200., -200., -200.), (0., 200., 200.)
HALF = (200., 200., 200.)


def scatter(pos, rms, rng):
    """Add a pinned (both-ends) Brownian-bridge transverse deviation, RMS=rms mm."""
    pos = np.asarray(pos); N = len(pos)
    d = pos[-1] - pos[0]; L = np.linalg.norm(d) + 1e-9; dirv = d / L
    ref = np.array([0., 0., 1.]) if abs(dirv[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(dirv, ref); e1 /= np.linalg.norm(e1) + 1e-9
    e2 = np.cross(dirv, e1)
    W = np.cumsum(rng.normal(size=(N, 2)), 0)
    s = np.arange(N) / (N - 1)
    bridge = W - s[:, None] * W[-1]                      # pinned at 0 and L
    delta = bridge[:, 0:1] * e1[None, :] + bridge[:, 1:2] * e2[None, :]
    delta *= rms / (np.sqrt(np.mean(delta ** 2)) + 1e-9)  # set RMS lateral deviation
    return jnp.asarray(pos + delta, jnp.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scatter', type=float, required=True, help='lateral scattering RMS (mm)')
    ap.add_argument('--n-muons', type=int, default=512)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); sr = np.random.RandomState(7)
    Ptrue, Pstr, De, St = [], [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)            # KNOWN endpoints (CRT)
        p, d, _, _, s = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., 160, logT, dedx,
                                              half_extents_mm=HALF, step_mm=4.0)
        Pstr.append(p); De.append(d); St.append(float(s))
        Ptrue.append(scatter(p, args.scatter, sr) if args.scatter > 0 else p)  # true = scattered
    pe = 7800.0 / max(args.scatter, 1e-9)                   # equiv momentum (MeV)

    hist, _ = recover_accum(
        sim, jnp.stack(Ptrue), jnp.stack(De), np.asarray(St, np.float32),   # obs: scattered (true)
        steps=args.steps, lr=3e-4, batch=16, record_every=500, val_frac=0.25, real_noise=True,
        pos_model=jnp.stack(Pstr), de_model=jnp.stack(De), step_model=np.asarray(St, np.float32))  # model: straight
    e = np.array(hist['emae']); v = np.array(hist['val'])
    vstep = int(v.argmin()) if len(v) and v.min() < np.inf else len(e) - 1
    res = dict(scatter_mm=args.scatter, equiv_p_GeV=pe / 1000, n_muons=args.n_muons, steps=args.steps,
               val_mae=float(e[vstep]), last_mae=float(e[-1]), best_mae=float(e.min()), emae=hist['emae'])
    json.dump(res, open(args.out, 'w'))
    print(f"[scatter={args.scatter}mm ~{pe/1000:.1f}GeV] field |E|MAE val-selected {res['val_mae']:.2f} "
          f"(last {res['last_mae']:.2f}, best {res['best_mae']:.2f}) V/cm")


if __name__ == '__main__':
    main()
