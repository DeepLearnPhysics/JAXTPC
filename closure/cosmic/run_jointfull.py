#!/usr/bin/env python3
"""Full joint fit (the Fisher-motivated approach): recover field + per-track
endpoints + per-track scattering shape, under the REALISTIC combined error
(endpoints known to ~CRT sigma, track scattered). Staged optimizer: warm up the
field with nuisances frozen, THEN release endpoints+shape jointly (fixes the 3b
cold-start failure). Sweep coverage (N_tracks) to test the Fisher prediction that
the well-posed optimum is reachable with enough tracks.
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
    ap.add_argument('--ep-sigma', type=float, default=10.0, help='CRT endpoint reco error (mm)')
    ap.add_argument('--scatter', type=float, default=2.0, help='scattering RMS (mm)')
    ap.add_argument('--n-modes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--warmup', type=int, default=4000)
    ap.add_argument('--ep-lr', type=float, default=0.1)
    ap.add_argument('--c-lr', type=float, default=0.2)
    ap.add_argument('--ep-prior', type=float, default=0.03)
    ap.add_argument('--c-prior', type=float, default=0.1)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    M, K = args.n_muons, args.n_modes

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.distortion_field
    FIXED = {k: truth[k] for k in ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    def full(fp): return {**FIXED, 'weights': fp['weights'], 'biases': fp['biases']}
    def fwd(stk, pos, de): return sim.forward_segments(base._replace(distortion_field=stk), pos, de, dx=STEP)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); jr = np.random.RandomState(123); sr = np.random.RandomState(7)
    Ptrue, De, EP0, E1, E2 = [], [], [], [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx,
                                              half_extents_mm=HALF, step_mm=STEP)
        De.append(d)
        Ptrue.append(scatter(p, args.scatter, sr) if args.scatter > 0 else jnp.asarray(p))
        EP0.append(np.stack([a + jr.normal(size=3) * args.ep_sigma, b + jr.normal(size=3) * args.ep_sigma]))
        d_ = (np.asarray(p)[-1] - np.asarray(p)[0]); d_ /= np.linalg.norm(d_) + 1e-9
        ref = np.array([0., 0., 1.]) if abs(d_[2]) < 0.9 else np.array([1., 0., 0.])
        e1 = np.cross(d_, ref); e1 /= np.linalg.norm(e1) + 1e-9
        E1.append(e1.astype(np.float32)); E2.append(np.cross(d_, e1).astype(np.float32))
    Ptrue = jnp.stack(Ptrue); De = jnp.stack(De); EP0 = jnp.asarray(np.stack(EP0))
    E1 = jnp.asarray(np.stack(E1)); E2 = jnp.asarray(np.stack(E2))
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))
    s_ = jnp.arange(NSEG, dtype=jnp.float32) / (NSEG - 1)
    basis = jnp.sin(jnp.pi * jnp.arange(1, K + 1)[None, :] * s_[:, None])

    # obs: true scattered forward + real noise
    cfg = sim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path)
    spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    # chunk the obs precompute (vmapping the forward over all M tracks OOMs at M~1000s)
    obs = None
    for i in range(0, M, 256):
        o = jax.vmap(lambda p, d: fwd(truth, p, d))(Ptrue[i:i + 256], De[i:i + 256])
        if obs is None:
            obs = [[] for _ in o]
        for pl in range(len(o)):
            obs[pl].append(o[pl])
    obs = [jnp.concatenate(a, 0) for a in obs]
    knz = jax.random.PRNGKey(0)
    for pl in range(len(obs)):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
        keys = jax.random.split(jax.random.fold_in(knz, pl), obs[pl].shape[0])
        obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(k, obs[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]; nplanes = len(obs)
    spec = [make_sobolev_weight(*obs[pl].shape[1:], max_pad=128, s=1.5) for pl in range(nplanes)]

    def model_one(fp, i, ep, c):
        a, b = ep[0], ep[1]; dirv = (b - a) / (jnp.linalg.norm(b - a) + 1e-6)
        ii = jnp.arange(NSEG, dtype=jnp.float32)
        pos = a[None, :] + ii[:, None] * STEP * dirv[None, :]
        dperp = basis @ c
        pos = pos + dperp[:, 0:1] * E1[i][None, :] + dperp[:, 1:2] * E2[i][None, :]
        de = mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF)
        return fwd(full(fp), pos, de)

    def loss(par, idx):
        fp = {'weights': par['weights'], 'biases': par['biases']}
        sg = jax.vmap(lambda i, ep, c: model_one(fp, i, ep, c))(idx, par['ep'][idx], par['c'][idx])
        tot = 0.0
        for pl in range(nplanes):
            tot += jnp.mean(jax.vmap(lambda u, v: sobolev_loss_single(u, v, spec[pl]))(sg[pl], obs[pl][idx]))
        return (tot / nplanes + args.ep_prior * jnp.mean((par['ep'][idx] - EP0[idx]) ** 2)
                + args.c_prior * jnp.mean(par['c'][idx] ** 2))

    k = jax.random.PRNGKey(7); nw, nb = [], []
    for w in truth['weights']:
        k, ss = jax.random.split(k); nw.append(w + 0.5 * jnp.abs(w) * jax.random.normal(ss, w.shape))
    for b in truth['biases']:
        k, ss = jax.random.split(k); nb.append(b + 0.5 * jnp.abs(b) * jax.random.normal(ss, b.shape))
    par = {'weights': nw, 'biases': nb, 'ep': EP0, 'c': jnp.zeros((M, K, 2))}

    sched = optax.warmup_cosine_decay_schedule(0., 3e-4, args.steps // 10, args.steps, 1.5e-5)
    of = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    oe = optax.adam(args.ep_lr); oc = optax.adam(args.c_lr)
    sf = of.init({'weights': par['weights'], 'biases': par['biases']})
    se = oe.init({'ep': par['ep']}); sc = oc.init({'c': par['c']})

    def make_step(fit_nuis):
        @jax.jit
        def step(par, sf, se, sc, idx):
            g = jax.grad(loss)(par, idx)
            uf, sf2 = of.update({'weights': g['weights'], 'biases': g['biases']}, sf,
                                {'weights': par['weights'], 'biases': par['biases']})
            fp = optax.apply_updates({'weights': par['weights'], 'biases': par['biases']}, uf)
            if fit_nuis:
                ue, se2 = oe.update({'ep': g['ep']}, se, {'ep': par['ep']})
                uc, sc2 = oc.update({'c': g['c']}, sc, {'c': par['c']})
                ep = optax.apply_updates({'ep': par['ep']}, ue)['ep']
                c = optax.apply_updates({'c': par['c']}, uc)['c']
                return {**fp, 'ep': ep, 'c': c}, sf2, se2, sc2
            return {**fp, 'ep': par['ep'], 'c': par['c']}, sf2, se, sc
        return step
    step_warm, step_joint = make_step(False), make_step(True)

    Et = emag_grid(sim, truth)
    def fmae(p): return float(jnp.mean(jnp.abs(emag_grid(sim, full(p)) - Et)))
    rng2 = np.random.RandomState(0); B = min(16, M); hist = [fmae(par)]
    for i in range(args.steps):
        idx = jnp.asarray(rng2.choice(M, B, replace=False))
        stp = step_warm if i < args.warmup else step_joint
        par, sf, se, sc = stp(par, sf, se, sc, idx)
        if (i + 1) % 500 == 0 or i == args.steps - 1:
            hist.append(fmae(par))
    res = dict(n_muons=M, ep_sigma=args.ep_sigma, scatter=args.scatter, n_modes=K, steps=args.steps,
               warmup=args.warmup, field_init=hist[0], field_final=hist[-1], field_best=min(hist), fmae=hist)
    json.dump(res, open(args.out, 'w'))
    print(f"[M={M} ep_sig={args.ep_sigma} scat={args.scatter}] field |E|MAE {hist[0]:.1f}->{hist[-1]:.2f} (best {min(hist):.2f}) V/cm")


if __name__ == '__main__':
    main()
