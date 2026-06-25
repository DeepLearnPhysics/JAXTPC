#!/usr/bin/env python3
"""#2: Gauss-Newton (curvature-preconditioned) track phase.

Alternating, but the per-track nuisance update is an exact GN/Levenberg-Marquardt
step using each track's own 16x16 curvature J^T J (endpoints+shape), instead of
fixed-LR Adam (which drifted/plateaued in run_altmin). The field stays on Adam.
Same realistic combined error (CRT endpoints + scattering). Tests whether
second-order track updates break the ~16 V/cm first-order plateau toward the
Fisher optimum (~1).
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp, optax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build, emag_grid
from tools.particle_generator import (load_dedx_table_jax, generate_cosmic_chord,
                                       sample_box_endpoints, mask_outside_volume)
from tools.losses import make_sobolev_weight, sobolev_loss_single
from tools.noise import load_noise_params, _get_noise_spectrum_shape, _generate_noise_for_plane
from closure.cosmic.run_scatter import scatter

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=512)
    ap.add_argument('--ep-sigma', type=float, default=10.0)
    ap.add_argument('--scatter', type=float, default=2.0)
    ap.add_argument('--n-modes', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=14)
    ap.add_argument('--field-steps', type=int, default=1500)
    ap.add_argument('--gn-sweeps', type=int, default=3, help='GN passes over all tracks per round')
    ap.add_argument('--lm', type=float, default=1e-2, help='LM damping (relative to mean diag)')
    ap.add_argument('--ep-prior', type=float, default=0.03)
    ap.add_argument('--c-prior', type=float, default=0.05)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    M, K = args.n_muons, args.n_modes; NP = 6 + 2 * K   # track params: endpoints(6)+shape(2K)

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.distortion_field
    FIXED = {k: truth[k] for k in ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    def full(fp): return {**FIXED, 'weights': fp['weights'], 'biases': fp['biases']}
    def fwd(stk, pos, de): return sim.forward_segments(base._replace(distortion_field=stk), pos, de, dx=STEP)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); jr = np.random.RandomState(123); sr = np.random.RandomState(7)
    Ptrue, De, EP0, E1, E2 = [], [], [], [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        De.append(d); Ptrue.append(scatter(p, args.scatter, sr) if args.scatter > 0 else jnp.asarray(p))
        EP0.append(np.stack([a + jr.normal(size=3) * args.ep_sigma, b + jr.normal(size=3) * args.ep_sigma]))
        dd = (np.asarray(p)[-1] - np.asarray(p)[0]); dd /= np.linalg.norm(dd) + 1e-9
        ref = np.array([0., 0., 1.]) if abs(dd[2]) < 0.9 else np.array([1., 0., 0.])
        e1 = np.cross(dd, ref); e1 /= np.linalg.norm(e1) + 1e-9
        E1.append(e1.astype(np.float32)); E2.append(np.cross(dd, e1).astype(np.float32))
    Ptrue = jnp.stack(Ptrue); De = jnp.stack(De); EP0 = jnp.asarray(np.stack(EP0))
    E1 = jnp.asarray(np.stack(E1)); E2 = jnp.asarray(np.stack(E2))
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))
    s_ = jnp.arange(NSEG, dtype=jnp.float32) / (NSEG - 1)
    basis = jnp.sin(jnp.pi * jnp.arange(1, K + 1)[None, :] * s_[:, None])

    cfg = sim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path); spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    obs = None
    for i in range(0, M, 256):
        o = jax.vmap(lambda p, d: fwd(truth, p, d))(Ptrue[i:i + 256], De[i:i + 256])
        if obs is None: obs = [[] for _ in o]
        for pl in range(len(o)): obs[pl].append(o[pl])
    obs = [jnp.concatenate(a, 0) for a in obs]
    knz = jax.random.PRNGKey(0)
    for pl in range(len(obs)):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
        keys = jax.random.split(jax.random.fold_in(knz, pl), obs[pl].shape[0])
        obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(k, obs[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]; nplanes = len(obs)

    # track params theta = [ep(6), c(2K)]; build positions, then flat residual vs obs
    def positions(i, th):
        a = th[:3]; b = th[3:6]; c = th[6:].reshape(K, 2)
        dirv = (b - a) / (jnp.linalg.norm(b - a) + 1e-6); ii = jnp.arange(NSEG, dtype=jnp.float32)
        pos = a[None, :] + ii[:, None] * STEP * dirv[None, :]
        dperp = basis @ c
        return pos + dperp[:, 0:1] * E1[i][None, :] + dperp[:, 1:2] * E2[i][None, :]
    th0 = jnp.concatenate([EP0.reshape(M, 6), jnp.zeros((M, 2 * K))], 1)  # (M, NP)
    th_ref = th0                                                          # prior anchor

    def model_flat(fp, i, th):
        pos = positions(i, th); de = mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF)
        return jnp.concatenate([s.reshape(-1) for s in fwd(full(fp), pos, de)])
    obs_flat = jnp.stack([jnp.concatenate([obs[pl][i].reshape(-1) for pl in range(nplanes)]) for i in range(M)])

    # GN step for one track: theta -= (J^T J + lm I)^-1 (J^T r + prior)
    def gn_track(fp, i, th):
        f = lambda t: model_flat(fp, i, t)
        J = jax.jacfwd(f)(th)                      # (S, NP)
        res = f(th) - obs_flat[i]                  # (S,)
        JTJ = J.T @ J; g = J.T @ res
        pri = args.ep_prior * 2 * jnp.concatenate([th[:6] - th_ref[i, :6], jnp.zeros(2 * K)]) \
            + args.c_prior * 2 * jnp.concatenate([jnp.zeros(6), th[6:]])
        H = JTJ + (args.lm * jnp.mean(jnp.diag(JTJ)) + 1e-3) * jnp.eye(NP)
        return th - jnp.linalg.solve(H, g + pri)
    gn_batch = jax.jit(jax.vmap(gn_track, in_axes=(None, 0, 0)))

    # field: sobolev loss (smooth), Adam
    spec = [make_sobolev_weight(*obs[pl].shape[1:], max_pad=128, s=1.5) for pl in range(nplanes)]
    def field_loss(fp, idx, TH):
        def one(i, th):
            pos = positions(i, th)
            return fwd(full(fp), pos, mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF))
        sg = jax.vmap(one)(idx, TH[idx])
        tot = 0.0
        for pl in range(nplanes):
            tot += jnp.mean(jax.vmap(lambda u, v: sobolev_loss_single(u, v, spec[pl]))(sg[pl], obs[pl][idx]))
        return tot / nplanes
    k = jax.random.PRNGKey(7); nw, nb = [], []
    for w in truth['weights']:
        k, ss = jax.random.split(k); nw.append(w + 0.5 * jnp.abs(w) * jax.random.normal(ss, w.shape))
    for b in truth['biases']:
        k, ss = jax.random.split(k); nb.append(b + 0.5 * jnp.abs(b) * jax.random.normal(ss, b.shape))
    fp = {'weights': nw, 'biases': nb}; TH = th0
    of = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4)); sf = of.init(fp)

    @jax.jit
    def field_step(fp, sf, idx, TH):
        g = jax.grad(lambda p: field_loss(p, idx, TH))(fp)
        u, sf = of.update(g, sf, fp); return optax.apply_updates(fp, u), sf

    Et = emag_grid(sim, truth)
    def fmae(fp): return float(jnp.mean(jnp.abs(emag_grid(sim, full(fp)) - Et)))
    rng2 = np.random.RandomState(0); B = min(16, M); hist = [fmae(fp)]
    for r in range(args.rounds):
        for _ in range(args.field_steps):
            fp, sf = field_step(fp, sf, jnp.asarray(rng2.choice(M, B, replace=False)), TH)
        for _ in range(args.gn_sweeps):                       # GN over all tracks (small batches: jacfwd is heavy)
            for i0 in range(0, M, 8):
                idx = jnp.arange(i0, min(i0 + 8, M))
                TH = TH.at[idx].set(gn_batch(fp, idx, TH[idx]))
        hist.append(fmae(fp))
    res = dict(n_muons=M, ep_sigma=args.ep_sigma, scatter=args.scatter, rounds=args.rounds,
               field_init=hist[0], field_final=hist[-1], field_best=min(hist), fmae=hist)
    json.dump(res, open(args.out, 'w'))
    print(f"[GN M={M} ep={args.ep_sigma} sc={args.scatter}] field |E|MAE {hist[0]:.1f}->{hist[-1]:.2f} (best {min(hist):.2f}) per-round {[round(float(x),1) for x in hist]}")


if __name__ == '__main__':
    main()
