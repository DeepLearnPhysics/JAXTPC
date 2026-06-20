#!/usr/bin/env python3
"""Identifiability / Fisher analysis of the joint inverse (field + track nuisances).

Parameters: K random directions in SIREN field-weight space (the "field" block) +
per-muon endpoints (the "track" nuisance block). Build J = d(signal)/d(params),
F = J^T J / sigma^2 (Fisher for the real noise). Then compare:
  F_ff               -- field information with tracks KNOWN
  F_ff - F_ft Ftt^-1 F_tf  -- field information with tracks MARGINALIZED (unknown)
The eigenvalue collapse between them is the field<->track degeneracy, quantified.
Directions with eigenvalue >> 1 are constrained by the data; << 1 are not.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints

LO, HI = (-200., -200., -200.), (0., 200., 200.); STEP, NSEG = 4.0, 160
SIGMA = 1.25   # real intrinsic noise (ADC), test-validated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tracks', type=int, default=6)
    ap.add_argument('--k-field', type=int, default=16)
    ap.add_argument('--shape-modes', type=int, default=5, help='per-track scattering shape modes')
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', default=os.path.join(HERE, 'fisher.json'))
    args = ap.parse_args()

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.sce_models
    FIXED = {k: truth[k] for k in ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    w0 = [np.asarray(w[0]) for w in truth['weights']]; b0 = [np.asarray(b[0]) for b in truth['biases']]
    shapes = [w.shape for w in w0]; sizes = [w.size for w in w0]; P = sum(sizes)

    # K random orthonormal directions in field-weight space
    rg = np.random.RandomState(1)
    D = rg.normal(size=(args.k_field, P)); D, _ = np.linalg.qr(D.T); D = jnp.asarray(D.T[:args.k_field])  # (K,P)

    def field_from_a(a):                       # a:(K,) -> stacked weights pytree
        dw = (a @ D)                            # (P,)
        ws, off = [], 0
        for sh, sz, w in zip(shapes, sizes, w0):
            ws.append((jnp.asarray(w) + dw[off:off + sz].reshape(sh))[None]); off += sz
        return {**FIXED, 'weights': ws, 'biases': [jnp.asarray(b)[None] for b in b0]}

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    EP, DE = [], []
    for _ in range(args.n_tracks):
        a, b = sample_box_endpoints(rng, LO, HI)
        _, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx,
                                              half_extents_mm=(200, 200, 200), step_mm=STEP)
        EP.append(np.stack([a, b])); DE.append(d)
    EP0 = jnp.asarray(np.stack(EP)); DE = jnp.stack(DE)

    Ks = args.shape_modes
    s_ = jnp.arange(NSEG, dtype=jnp.float32) / (NSEG - 1)
    basis = jnp.sin(jnp.pi * jnp.arange(1, Ks + 1)[None, :] * s_[:, None])   # (NSEG,Ks)

    def track(ep, c):                            # ep:(2,3), c:(Ks,2) shape modes
        a, b = ep[0], ep[1]; dirv = (b - a) / (jnp.linalg.norm(b - a) + 1e-6)
        ref = jnp.where(jnp.abs(dirv[2]) < 0.9, jnp.array([0., 0., 1.]), jnp.array([1., 0., 0.]))
        e1 = jnp.cross(dirv, ref); e1 = e1 / (jnp.linalg.norm(e1) + 1e-9); e2 = jnp.cross(dirv, e1)
        i = jnp.arange(NSEG, dtype=jnp.float32)
        straight = a[None, :] + i[:, None] * STEP * dirv[None, :]
        dperp = basis @ c
        return straight + dperp[:, 0:1] * e1[None, :] + dperp[:, 1:2] * e2[None, :]

    def signal(a, ep, c):
        fld = field_from_a(a)
        sg = jax.vmap(lambda e, cc, d: sim.forward_segments(
            base._replace(sce_models=fld), track(e, cc), d, dx=STEP))(ep, c, DE)
        return jnp.concatenate([s.reshape(-1) for s in sg])

    a0 = jnp.zeros(args.k_field); C0 = jnp.zeros((args.n_tracks, Ks, 2))
    Ja, Je, Jc = jax.jacfwd(signal, argnums=(0, 1, 2))(a0, EP0, C0)
    S = Ja.shape[0]
    Ja = np.asarray(Ja); Je = np.asarray(Je).reshape(S, -1); Jc = np.asarray(Jc).reshape(S, -1)
    Kf, ne, nc = args.k_field, Je.shape[1], Jc.shape[1]
    J = np.concatenate([Ja, Je, Jc], 1) / SIGMA
    F = J.T @ J
    Fff = F[:Kf, :Kf]
    def marg(idx):                               # field info after marginalizing nuisance block `idx`
        Ffn = F[:Kf, idx]; Fnn = F[np.ix_(idx, idx)]
        return Fff - Ffn @ np.linalg.inv(Fnn + 1e-6 * np.eye(len(idx))) @ Ffn.T
    ei = np.arange(Kf, Kf + ne); ci = np.arange(Kf + ne, Kf + ne + nc)
    ev = lambda M: np.sort(np.linalg.eigvalsh(M))[::-1]
    e_known, e_ep, e_sh, e_both = ev(Fff), ev(marg(ei)), ev(marg(ci)), ev(marg(np.concatenate([ei, ci])))
    def loss(a): return float(np.median(e_known / np.maximum(a, 1e-12)))
    res = dict(n_tracks=args.n_tracks, k_field=Kf, shape_modes=Ks, sigma=SIGMA,
               n_constr_known=int((e_known > 1).sum()), n_constr_ep=int((e_ep > 1).sum()),
               n_constr_shape=int((e_sh > 1).sum()), n_constr_both=int((e_both > 1).sum()),
               ev_known=e_known.tolist(), ev_shape=e_sh.tolist())
    json.dump(res, open(args.out, 'w'))
    print(f"Field directions constrained (Fisher eig>1), {args.n_tracks} tracks, {Kf} field dirs:")
    print(f"  all nuisances KNOWN:          {res['n_constr_known']}/{Kf}   (eig {e_known[0]:.1e}..{e_known[-1]:.1e})")
    print(f"  ENDPOINTS marginalized:       {res['n_constr_ep']}/{Kf}   median field-info loss x{loss(e_ep):.1f}")
    print(f"  SHAPE modes marginalized:     {res['n_constr_shape']}/{Kf}   median field-info loss x{loss(e_sh):.1f}")
    print(f"  BOTH marginalized:            {res['n_constr_both']}/{Kf}   median field-info loss x{loss(e_both):.1f}")
    print(f"  -> if SHAPE collapses field dirs (n_constr drops / loss huge), it's a real degeneracy (explains 3b);")
    print(f"     endpoints staying constrained with bounded loss = variance-only (explains 2b).")


if __name__ == '__main__':
    main()
