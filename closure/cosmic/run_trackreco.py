#!/usr/bin/env python3
"""How well does the DATA reconstruct a track (given the field)? The differentiable
sim IS a track-reco tool: invert track->image. We kept anchoring tracks to cm-CRT;
this tests whether the wire image (high-SNR) constrains the track far better, which
would mean the CRT anchor was the artificial limit.

Given the TRUTH field, for each track fit its endpoints to its own wire image (Adam,
NO CRT anchor, init = true + CRT noise). Report recovered endpoint error vs CRT sigma.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp, optax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints, mask_outside_volume
from tools.noise import load_noise_params, _get_noise_spectrum_shape, _generate_noise_for_plane

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=64)
    ap.add_argument('--ep-sigma', type=float, default=10.0)   # CRT init noise (mm)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=2.0)
    ap.add_argument('--noise', action='store_true', default=True)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', default=os.path.join(HERE, 'trackreco.json'))
    args = ap.parse_args()
    M = args.n_muons

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.distortion_field
    def fwd(pos, de): return sim.forward_segments(base, pos, de, dx=STEP)  # truth field fixed

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); jr = np.random.RandomState(123)
    P, De, EPtrue = [], [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        P.append(p); De.append(d); EPtrue.append(np.stack([np.asarray(p)[0], np.asarray(p)[-1]]))
    P = jnp.stack(P); De = jnp.stack(De); EPtrue = jnp.asarray(np.stack(EPtrue))
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))
    EPinit = EPtrue + jr.normal(size=EPtrue.shape).astype(np.float32) * args.ep_sigma

    cfg = sim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path); spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    obs = list(jax.vmap(lambda p, d: fwd(p, d))(P, De)); nplanes = len(obs)
    knz = jax.random.PRNGKey(0)
    if args.noise:
        for pl in range(nplanes):
            L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
            keys = jax.random.split(jax.random.fold_in(knz, pl), M)
            obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(k, obs[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]

    def track(ep):
        a, b = ep[0], ep[1]; s = (jnp.arange(NSEG, dtype=jnp.float32) / (NSEG - 1))[:, None]
        return a[None, :] + s * (b - a)[None, :]
    def loss_one(ep, d, ob):                       # fit ONE track's endpoints to its image (field fixed=truth)
        pos = track(ep); sg = fwd(pos, mask_outside_volume(pos, d, HALF))
        return sum(jnp.mean((sg[pl] - ob[pl]) ** 2) for pl in range(nplanes))

    opt = optax.adam(args.lr)
    @jax.jit
    def fit(ep0, d, ob):
        st = opt.init(ep0)
        def body(c, _):
            ep, st = c; g = jax.grad(loss_one)(ep, d, ob); u, st = opt.update(g, st, ep); return (optax.apply_updates(ep, u), st), None
        (ep, _), _ = jax.lax.scan(body, (ep0, st), None, length=args.steps)
        return ep
    fit_batch = jax.jit(jax.vmap(fit, in_axes=(0, 0, 0)))

    init_err = float(jnp.mean(jnp.abs(EPinit - EPtrue)))
    EPrec = []
    for i0 in range(0, M, 16):
        sl = slice(i0, min(i0 + 16, M))
        EPrec.append(np.asarray(fit_batch(EPinit[sl], De[sl], [o[sl] for o in obs])))
    EPrec = jnp.asarray(np.concatenate(EPrec, 0))
    rec_err = float(jnp.mean(jnp.abs(EPrec - EPtrue)))
    # split: anode-side vs cathode-side endpoint (drift x near 0 = anode)
    xa = np.abs(np.asarray(EPtrue)[:, :, 0])  # |x| ~ drift distance
    print(f"endpoint error (mm): CRT init {init_err:.2f} -> DATA-reconstructed {rec_err:.2f}  (noise={args.noise})")
    per = np.abs(np.asarray(EPrec) - np.asarray(EPtrue)).mean(-1)  # (M,2)
    print(f"  per-endpoint: mean {per.mean():.2f}mm  | shallow-drift end {per[xa<np.median(xa)].mean():.2f}  deep-drift end {per[xa>=np.median(xa)].mean():.2f}")
    json.dump(dict(n_muons=M, ep_sigma=args.ep_sigma, init_err=init_err, rec_err=rec_err), open(args.out, 'w'))


if __name__ == '__main__':
    main()
