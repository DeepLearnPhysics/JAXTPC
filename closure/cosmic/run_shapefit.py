#!/usr/bin/env python3
"""Test 3b: fit the TRACK SHAPE to handle scattering (known endpoints).

True track is scattered (test 3). The recovery keeps the known endpoints fixed but
gives each muon K low-order transverse deflection modes (sin(k*pi*s/L), pinned at
both ends -> smooth, endpoint-preserving) fit jointly with the shared field. The
field is coherent across muons; the per-muon shape modes absorb that muon's
scattering. Compares to the straight-track baseline (test 3).
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

LO, HI = (-200., -200., -200.), (0., 200., 200.)
HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160


def perp_basis(p):
    p = np.asarray(p); d = p[-1] - p[0]; d = d / (np.linalg.norm(d) + 1e-9)
    ref = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1) + 1e-9
    return e1.astype(np.float32), np.cross(d, e1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scatter', type=float, required=True)
    ap.add_argument('--n-modes', type=int, default=5)
    ap.add_argument('--c-lr', type=float, default=0.3)
    ap.add_argument('--c-prior', type=float, default=0.0)
    ap.add_argument('--fit-shape', type=int, default=1)
    ap.add_argument('--n-muons', type=int, default=512)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.distortion_field
    FIXED = {k: truth[k] for k in ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    def full(fp): return {**FIXED, 'weights': fp['weights'], 'biases': fp['biases']}
    def fwd(stk, pos, de): return sim.forward_segments(base._replace(distortion_field=stk), pos, de, dx=STEP)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); sr = np.random.RandomState(7)
    Pstr, Ptrue, De, E1, E2 = [], [], [], [], []
    for _ in range(args.n_muons):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx,
                                              half_extents_mm=HALF, step_mm=STEP)
        Pstr.append(np.asarray(p)); De.append(d)
        Ptrue.append(scatter(p, args.scatter, sr) if args.scatter > 0 else jnp.asarray(p))
        e1, e2 = perp_basis(p); E1.append(e1); E2.append(e2)
    Pstr = jnp.asarray(np.stack(Pstr)); De = jnp.stack(De)
    Ptrue = jnp.stack(Ptrue); E1 = jnp.asarray(np.stack(E1)); E2 = jnp.asarray(np.stack(E2))
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))

    # mode basis: (NSEG, K) sin(k pi s), pinned at both ends
    s = jnp.arange(NSEG, dtype=jnp.float32) / (NSEG - 1)
    basis = jnp.sin(jnp.pi * jnp.arange(1, args.n_modes + 1)[None, :] * s[:, None])   # (NSEG,K)

    # obs: true scattered forward + real noise
    cfg = sim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path)
    spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    obs = list(jax.vmap(lambda p, d: fwd(truth, p, d))(Ptrue, De))
    knz = jax.random.PRNGKey(0)
    for pl in range(len(obs)):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
        keys = jax.random.split(jax.random.fold_in(knz, pl), obs[pl].shape[0])
        obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(
            k, obs[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]
    nplanes = len(obs)
    spec = [make_sobolev_weight(*obs[pl].shape[1:], max_pad=128, s=1.5) for pl in range(nplanes)]

    def model_pos(i):                                   # straight_i + modes_i . c_i
        return Pstr[i]
    def model_one(fp, i, c):                            # c: (K,2)
        dperp = basis @ c                               # (NSEG,2)
        pos = Pstr[i] + dperp[:, 0:1] * E1[i][None, :] + dperp[:, 1:2] * E2[i][None, :]
        de = mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF)
        return fwd(full(fp), pos, de)

    def loss(par, idx):
        sg = jax.vmap(lambda i, c: model_one({'weights': par['weights'], 'biases': par['biases']}, i, c))(idx, par['c'][idx])
        tot = 0.0
        for pl in range(nplanes):
            tot += jnp.mean(jax.vmap(lambda a, b: sobolev_loss_single(a, b, spec[pl]))(sg[pl], obs[pl][idx]))
        return tot / nplanes + args.c_prior * jnp.mean(par['c'][idx] ** 2)

    k = jax.random.PRNGKey(7); nw, nb = [], []
    for w in truth['weights']:
        k, ss = jax.random.split(k); nw.append(w + 0.5 * jnp.abs(w) * jax.random.normal(ss, w.shape))
    for b in truth['biases']:
        k, ss = jax.random.split(k); nb.append(b + 0.5 * jnp.abs(b) * jax.random.normal(ss, b.shape))
    par = {'weights': nw, 'biases': nb, 'c': jnp.zeros((args.n_muons, args.n_modes, 2))}

    sched = optax.warmup_cosine_decay_schedule(0., 3e-4, args.steps // 10, args.steps, 1.5e-5)
    opt_f = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    opt_c = optax.adam(args.c_lr) if args.fit_shape else optax.set_to_zero()
    sf = opt_f.init({'weights': par['weights'], 'biases': par['biases']}); sc = opt_c.init({'c': par['c']})

    @jax.jit
    def step(par, sf, sc, idx):
        g = jax.grad(loss)(par, idx)
        uf, sf = opt_f.update({'weights': g['weights'], 'biases': g['biases']}, sf,
                              {'weights': par['weights'], 'biases': par['biases']})
        uc, sc = opt_c.update({'c': g['c']}, sc, {'c': par['c']})
        fp = optax.apply_updates({'weights': par['weights'], 'biases': par['biases']}, uf)
        return {**fp, 'c': optax.apply_updates({'c': par['c']}, uc)['c']}, sf, sc

    Et = emag_grid(sim, truth)
    def fmae(par): return float(jnp.mean(jnp.abs(emag_grid(sim, full(par)) - Et)))
    rng2 = np.random.RandomState(0); B = min(16, args.n_muons)
    hist = [fmae(par)]
    for i in range(args.steps):
        idx = jnp.asarray(rng2.choice(args.n_muons, B, replace=False))
        par, sf, sc = step(par, sf, sc, idx)
        if (i + 1) % 500 == 0 or i == args.steps - 1:
            hist.append(fmae(par))
    res = dict(scatter_mm=args.scatter, fit_shape=args.fit_shape, n_modes=args.n_modes,
               field_mae_init=hist[0], field_mae_final=hist[-1], field_mae_best=min(hist), fmae=hist)
    json.dump(res, open(args.out, 'w'))
    print(f"[scatter={args.scatter} shapefit={args.fit_shape} K={args.n_modes}] field |E|MAE "
          f"{hist[0]:.1f}->{hist[-1]:.2f} (best {min(hist):.2f}) V/cm")


if __name__ == '__main__':
    main()
