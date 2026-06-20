#!/usr/bin/env python3
"""Coupled Schur-complement Gauss-Newton (the diagnosis-motivated fix).

Field in a K-dim subspace (coeffs a) + per-track endpoints (theta_i, 6). Each LM
iteration takes the JOINT step whose field direction accounts for how the tracks
re-adjust (Schur complement) -- the one direction first-order / per-track GN miss:
  S    = F_aa - sum_i F_at_i (F_tt_i+lam)^-1 F_at_i^T
  g_s  = g_a  - sum_i F_at_i (F_tt_i+lam)^-1 g_t_i
  da   = -(S + mu I)^-1 g_s
  dth_i= -(F_tt_i+lam)^-1 (g_t_i + F_at_i^T da)
LM-adapts mu on accept/reject. Endpoint-only (matches run_diag); obs from truth
field + TRUE endpoints + noise. Compares to the first-order plateau (~16-20).
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build, emag_grid
from tools.particle_generator import (load_dedx_table_jax, generate_cosmic_chord,
                                       sample_box_endpoints, mask_outside_volume)
from tools.noise import load_noise_params, _get_noise_spectrum_shape, _generate_noise_for_plane

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=48)
    ap.add_argument('--k-field', type=int, default=48)
    ap.add_argument('--ep-sigma', type=float, default=10.0)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--ep-prior', type=float, default=0.03)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', default=os.path.join(HERE, 'schur.json'))
    args = ap.parse_args()
    M, Kf = args.n_muons, args.k_field

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.sce_models
    FIXED = {k: truth[k] for k in ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    w0 = [np.asarray(w[0]) for w in truth['weights']]; b0 = [jnp.asarray(b) for b in truth['biases']]
    shapes = [w.shape for w in w0]; sizes = [w.size for w in w0]; P = sum(sizes)
    rgw = np.random.RandomState(1); D = rgw.normal(size=(Kf, P)); D, _ = np.linalg.qr(D.T); D = jnp.asarray(D.T[:Kf])
    # cold init: truth + noise, expressed as a base weight set; a perturbs along D
    w_init = [jnp.asarray(w + 0.5 * np.abs(w) * np.random.RandomState(s).normal(size=w.shape)) for s, w in enumerate(w0)]

    def field(a):
        dw = a @ D; ws, off = [], 0
        for sh, sz, w in zip(shapes, sizes, w_init):
            ws.append((w + dw[off:off + sz].reshape(sh))[None]); off += sz
        return {**FIXED, 'weights': ws, 'biases': b0}
    def fwd(stk, pos, de): return sim.forward_segments(base._replace(sce_models=stk), pos, de, dx=STEP)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); jr = np.random.RandomState(123)
    De, TH_true, TH_reco = [], [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        _, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        De.append(d); TH_true.append(np.concatenate([a, b]))
        TH_reco.append(np.concatenate([a + jr.normal(size=3) * args.ep_sigma, b + jr.normal(size=3) * args.ep_sigma]))
    De = jnp.stack(De); TH_true = jnp.asarray(np.stack(TH_true)); TH_reco = jnp.asarray(np.stack(TH_reco))
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))

    def positions(th):
        a = th[:3]; b = th[3:]; dirv = (b - a) / (jnp.linalg.norm(b - a) + 1e-6)
        ii = jnp.arange(NSEG, dtype=jnp.float32)
        return a[None, :] + ii[:, None] * STEP * dirv[None, :]
    def model_flat(a, th, d):
        pos = positions(th)
        sg = fwd(field(a), pos, mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF))
        return jnp.concatenate([s.reshape(-1) for s in sg])

    cfg = sim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path); spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    # obs: truth (a=0 on TRUTH weights, but our base is w_init; use exact truth field) + true tracks + noise
    a_truth = jnp.zeros(Kf)  # placeholder; build obs from the exact truth field
    def fwd_truth(th, d):
        pos = positions(th); return fwd(truth, pos, mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF))
    obs_planes = jax.vmap(fwd_truth)(TH_true, De)
    knz = jax.random.PRNGKey(0); obs = []
    for pl in range(len(obs_planes)):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
        keys = jax.random.split(jax.random.fold_in(knz, pl), M)
        obs.append(obs_planes[pl] + jax.vmap(lambda k: _generate_noise_for_plane(k, obs_planes[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys))
    obs_flat = jnp.stack([jnp.concatenate([obs[pl][i].reshape(-1) for pl in range(len(obs))]) for i in range(M)])

    # per-track GN blocks, computed ON GPU (return small K/6-sized blocks, not the
    # huge Jacobians -- so the curvature accumulation batches to large M cheaply).
    def per_track(a, th, d, of):
        Ja = jax.jacfwd(lambda aa: model_flat(aa, th, d))(a)     # (S,Kf)
        Jt = jax.jacfwd(lambda tt: model_flat(a, tt, d))(th)     # (S,6)
        r = model_flat(a, th, d) - of                            # (S,)
        return (Ja.T @ Ja, Ja.T @ Jt, Jt.T @ Jt, Ja.T @ r, Jt.T @ r)   # Faa,Fat,Ftt,ga,gt
    pt_batch = jax.jit(jax.vmap(per_track, in_axes=(None, 0, 0, 0)))

    def total_loss(a, TH):
        s = 0.0
        for i0 in range(0, M, 16):
            sl = slice(i0, min(i0 + 16, M))
            rr = jax.vmap(lambda th, d, of: model_flat(a, th, d) - of)(TH[sl], De[sl], obs_flat[sl])
            s += float(jnp.sum(rr ** 2))
        return s + args.ep_prior * float(jnp.sum((TH - TH_reco) ** 2))

    Et = emag_grid(sim, truth)
    def fmae(a): return float(jnp.mean(jnp.abs(emag_grid(sim, field(a)) - Et)))

    a = jnp.zeros(Kf); TH = TH_reco; lam = 1e-1; step = 0.5; tsvd_rel = 1e-2
    hist = [(fmae(a), float(jnp.mean(jnp.abs(TH - TH_true))), total_loss(a, TH))]
    for it in range(args.iters):
        # accumulate Schur blocks over track mini-batches (bounded memory, any M)
        Faa = np.zeros((Kf, Kf)); ga = np.zeros(Kf); S = np.zeros((Kf, Kf)); gs = np.zeros(Kf)
        Ftt_l, Fat_l, gt_l = [None] * M, [None] * M, [None] * M
        for i0 in range(0, M, 4):
            sl = slice(i0, min(i0 + 4, M))
            Faa_b, Fat_b, Ftt_b, ga_b, gt_b = pt_batch(a, TH[sl], De[sl], obs_flat[sl])
            Faa += np.asarray(Faa_b.sum(0)); ga += np.asarray(ga_b.sum(0))
            Fat_b = np.asarray(Fat_b); Ftt_b = np.asarray(Ftt_b); gt_b = np.asarray(gt_b)
            for j in range(Fat_b.shape[0]):
                gi = i0 + j
                Ftt = Ftt_b[j] + args.ep_prior * np.eye(6)
                gt = gt_b[j] + args.ep_prior * (np.asarray(TH[gi]) - np.asarray(TH_reco[gi]))
                Finv = np.linalg.inv(Ftt + lam * np.eye(6))
                S += -Fat_b[j] @ Finv @ Fat_b[j].T; gs += -Fat_b[j] @ Finv @ gt
                Ftt_l[gi] = Finv; Fat_l[gi] = Fat_b[j]; gt_l[gi] = gt
        S = Faa + S; gs = ga + gs
        # Truncated-eigenvalue GN: move the field ONLY in data-constrained directions
        # (eigenvalues of the Schur curvature above a relative threshold). The flat
        # directions of the random subspace are dropped so they cannot run away.
        evals, evecs = np.linalg.eigh(S)
        keep = evals > tsvd_rel * evals.max()
        V = evecs[:, keep]
        da = -step * (V @ ((V.T @ gs) / evals[keep]))
        TH_new = np.asarray(TH).copy()
        for gi in range(M):
            TH_new[gi] += -step * (Ftt_l[gi] @ (gt_l[gi] + Fat_l[gi].T @ da))
        a_new = a + jnp.asarray(da); TH_new = jnp.asarray(TH_new)
        L_old = hist[-1][2]; L_new = total_loss(a_new, TH_new)
        if L_new < L_old:
            a, TH = a_new, TH_new; step = min(step * 1.2, 1.0)
        else:
            step = max(step * 0.5, 0.05)
        hist.append((fmae(a), float(jnp.mean(jnp.abs(TH - TH_true))), total_loss(a, TH)))
        print(f"  iter {it+1}: field_mae={hist[-1][0]:.2f}  track_err={hist[-1][1]:.2f}mm  L={hist[-1][2]:.3f}  rank={int(keep.sum())} step={step:.2f}")
    res = dict(n_muons=M, k_field=Kf, ep_sigma=args.ep_sigma, field_mae=[h[0] for h in hist],
               track_err=[h[1] for h in hist], loss=[h[2] for h in hist])
    json.dump(res, open(args.out, 'w'))
    print(f"[SCHUR M={M} K={Kf}] field_mae {hist[0][0]:.1f}->{hist[-1][0]:.2f}  track_err {hist[0][1]:.1f}->{hist[-1][1]:.2f}mm")


if __name__ == '__main__':
    main()
