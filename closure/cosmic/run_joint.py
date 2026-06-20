#!/usr/bin/env python3
"""Test 2b: JOINT track + field fit.

Track-reco error (test 2) is the dominant confound. Here we fit the per-muon
track endpoints AND the shared SCE field simultaneously: obs from the TRUE tracks,
the model track is parameterised by optimisable endpoints initialised from the
(perturbed) reconstructed endpoints. The field is coherent across muons while the
track errors are independent, so the joint fit can in principle separate them.

Reports field recovery (|E| MAE vs truth) and track recovery (endpoint error)
before/after, for a given reco error sigma.
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

LO, HI = (-200., -200., -200.), (0., 200., 200.)
HALF = (200., 200., 200.)
STEP, NSEG = 4.0, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--n-muons', type=int, default=512)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--ep-lr', type=float, default=0.1, help='endpoint LR (mm/step scale)')
    ap.add_argument('--ep-prior', type=float, default=0.0, help='prior weight pulling endpoints to the reco')
    ap.add_argument('--fit-tracks', type=int, default=1, help='1=joint, 0=fixed (baseline)')
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.sce_models
    FIXED = {k: truth[k] for k in ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    def full(fp): return {**FIXED, 'weights': fp['weights'], 'biases': fp['biases']}
    def fwd(stk, pos, de): return sim.forward_segments(base._replace(sce_models=stk), pos, de, dx=STEP)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); jr = np.random.RandomState(123)
    Pt, De, EPt, EPm = [], [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx,
                                              half_extents_mm=HALF, step_mm=STEP)
        Pt.append(p); De.append(d); EPt.append(np.stack([a, b]))
        EPm.append(np.stack([a + jr.normal(size=3) * args.sigma, b + jr.normal(size=3) * args.sigma]))
    Pt = jnp.stack(Pt); De = jnp.stack(De)
    EP_true = jnp.asarray(np.stack(EPt)); EP0 = jnp.asarray(np.stack(EPm))   # (M,2,3)
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))

    # obs: true-track forward + real intrinsic noise (ADC scale)
    cfg = sim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path)
    spec_noise = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    obs = list(jax.vmap(lambda p, d: fwd(truth, p, d))(Pt, De))
    knz = jax.random.PRNGKey(0)
    for pl in range(len(obs)):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32); series = ny + nz * L
        keys = jax.random.split(jax.random.fold_in(knz, pl), obs[pl].shape[0])
        obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(
            k, obs[pl].shape[1], nt, spec_noise, series, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]
    nplanes = len(obs)
    spec = [make_sobolev_weight(*obs[pl].shape[1:], max_pad=128, s=1.5) for pl in range(nplanes)]

    def track(ep):                                   # (2,3) -> (NSEG,3), differentiable in ep
        a, b = ep[0], ep[1]; dirv = (b - a) / (jnp.linalg.norm(b - a) + 1e-6)
        i = jnp.arange(NSEG, dtype=jnp.float32)
        return a[None, :] + i[:, None] * STEP * dirv[None, :]

    def model_one(fp, ep):
        pos = track(ep)
        de = mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF)
        return fwd(full(fp), pos, de)

    def loss(par, idx):
        sg = jax.vmap(lambda ep: model_one({'weights': par['weights'], 'biases': par['biases']}, ep))(par['ep'][idx])
        tot = 0.0
        for pl in range(nplanes):
            tot += jnp.mean(jax.vmap(lambda a, b: sobolev_loss_single(a, b, spec[pl]))(sg[pl], obs[pl][idx]))
        data = tot / nplanes
        # prior: the reco IS a measurement of the endpoints with ~sigma uncertainty;
        # anchor the fitted endpoints to it so they refine within sigma, not drift.
        prior = args.ep_prior * jnp.mean((par['ep'][idx] - EP0[idx]) ** 2)
        return data + prior

    # init: field = truth+noise, endpoints = reconstructed (perturbed)
    k = jax.random.PRNGKey(7); nw, nb = [], []
    for w in truth['weights']:
        k, s = jax.random.split(k); nw.append(w + 0.5 * jnp.abs(w) * jax.random.normal(s, w.shape))
    for b in truth['biases']:
        k, s = jax.random.split(k); nb.append(b + 0.5 * jnp.abs(b) * jax.random.normal(s, b.shape))
    par = {'weights': nw, 'biases': nb, 'ep': EP0}

    sched = optax.warmup_cosine_decay_schedule(0., 3e-4, args.steps // 10, args.steps, 1.5e-5)
    opt_f = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    opt_e = optax.adam(args.ep_lr) if args.fit_tracks else optax.set_to_zero()
    sf = opt_f.init({'weights': par['weights'], 'biases': par['biases']})
    se = opt_e.init({'ep': par['ep']})

    @jax.jit
    def step(par, sf, se, idx):
        g = jax.grad(loss)(par, idx)
        uf, sf = opt_f.update({'weights': g['weights'], 'biases': g['biases']}, sf,
                              {'weights': par['weights'], 'biases': par['biases']})
        ue, se = opt_e.update({'ep': g['ep']}, se, {'ep': par['ep']})
        fp = optax.apply_updates({'weights': par['weights'], 'biases': par['biases']}, uf)
        ep = optax.apply_updates({'ep': par['ep']}, ue)['ep']
        return {**fp, 'ep': ep}, sf, se

    Et = emag_grid(sim, truth)
    def fmae(par): return float(jnp.mean(jnp.abs(emag_grid(sim, full(par)) - Et)))
    def tmae(par): return float(jnp.mean(jnp.abs(par['ep'] - EP_true)))   # endpoint error (mm)

    rng2 = np.random.RandomState(0); B = min(16, args.n_muons)
    hist = {'fmae': [fmae(par)], 'tmae': [tmae(par)]}
    for i in range(args.steps):
        idx = jnp.asarray(rng2.choice(args.n_muons, B, replace=False))
        par, sf, se = step(par, sf, se, idx)
        if (i + 1) % 500 == 0 or i == args.steps - 1:
            hist['fmae'].append(fmae(par)); hist['tmae'].append(tmae(par))
    res = dict(sigma_mm=args.sigma, fit_tracks=args.fit_tracks, n_muons=args.n_muons, steps=args.steps,
               field_mae_init=hist['fmae'][0], field_mae_final=hist['fmae'][-1], field_mae_best=min(hist['fmae']),
               track_mae_init=hist['tmae'][0], track_mae_final=hist['tmae'][-1], fmae=hist['fmae'], tmae=hist['tmae'])
    json.dump(res, open(args.out, 'w'))
    print(f"[sigma={args.sigma} joint={args.fit_tracks}] field |E|MAE {res['field_mae_init']:.2f}->{res['field_mae_final']:.2f} "
          f"(best {res['field_mae_best']:.2f}) | track err {res['track_mae_init']:.2f}->{res['track_mae_final']:.2f} mm")


if __name__ == '__main__':
    main()
