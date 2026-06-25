#!/usr/bin/env python3
"""Plot the FULL-MODE cosmic SCE-field closure (all SIREN weights, Sobolev loss).

Recovers the full space-charge field shape — not just a global scale — by
gradient descent through the differentiable simulator on a batch of cosmic
muons, using the Sobolev (screened-Poisson H^{-s}) signal loss the muon
closure uses. Renders convergence curves + true/init/recovered |E| slices.
"""
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tools.sce_siren as S
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
from closure.cosmic.recover_field import build, emag_grid

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closure_full.png')


def emag_slice(sim, stacked, z_cm=0.0, n=48):
    sb = sim.distortion_state()
    xs = np.linspace(0.5, 19.5, n)
    ys = np.linspace(-19, 19, n)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    grid = jnp.array(np.stack([XX.ravel(), YY.ravel(),
                               np.full(XX.size, z_cm)], -1), jnp.float32)
    p0 = jax.tree.map(lambda x: x[0], stacked)
    E = S.recover_efield({'weights': p0['weights'], 'biases': p0['biases']}, grid,
                         p0['E0'], p0['v0'], sb['v_table'], sb['E_table'],
                         p0['norm_offsets'], p0['norm_scales'], sb['omega_0'])
    return xs, ys, np.array(jnp.sqrt((E ** 2).sum(-1))).reshape(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--n-tracks', type=int, default=48)
    ap.add_argument('--n-seg', type=int, default=768)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--omega', type=float, default=2.0)
    args = ap.parse_args()

    sim, pos, de, step = build(args.n_seg, truth_scale=1.0,
                               n_tracks=args.n_tracks, omega_0=args.omega)
    base = sim._default_sim_params
    truth = base.distortion_field

    def fwd(stk):
        return sim.forward_segments(base._replace(distortion_field=stk), pos, de, dx=step)
    obs = [jax.lax.stop_gradient(s) for s in fwd(truth)]

    planes = tuple(range(len(obs)))
    spec_w = tuple(make_sobolev_weight(*obs[p].shape, max_pad=256, s=1.5)
                   for p in planes)

    def loss(stk):
        return sobolev_loss_geomean_log1p(fwd(stk), obs, spec_w, planes)

    Et_grid = emag_grid(sim, truth)

    def e_mae(stk):
        return float(jnp.mean(jnp.abs(emag_grid(sim, stk) - Et_grid)))

    # Init: a COHERENT magnitude error (output layer ×0.5 ⇒ ~half-strength SCE,
    # in-regime) plus 10% noise on every weight. The coherent component is what
    # the cosmic tracks actually constrain (so descent is well-posed), while all
    # parameters still differ from truth (a genuine full-field optimisation).
    # NB: with a finite number of tracks the field OFF the tracks is only weakly
    # constrained — global recovery improves with denser angular coverage.
    k = jax.random.PRNGKey(7)
    leaves, tdef = jax.tree.flatten(truth)
    nl = []
    for lf in leaves:
        k, sub = jax.random.split(k)
        nl.append(lf + 0.10 * jnp.abs(lf) * jax.random.normal(sub, lf.shape))
    init = jax.tree.unflatten(tdef, nl)
    init['weights'][-1] = init['weights'][-1] * 0.5     # coherent strength error
    # Gradient clipping + modest LR: a high-DOF field is easily kicked out of
    # the physical (un-clamped) regime by a single large step, after which the
    # v-inversion clamp gives zero gradient and the optimiser is stuck. Clipping
    # the global grad norm keeps every step in-regime so descent is monotone.
    sched = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, max(1, args.steps // 10), args.steps, args.lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    st = opt.init(init); p = init
    vg = jax.jit(jax.value_and_grad(loss))

    hist = {'loss': [float(loss(p))], 'emae': [e_mae(p)]}
    print(f"[full+Sobolev] {args.n_tracks} muons, {args.n_seg} segs, {args.steps} steps")
    print(f"  init: loss={hist['loss'][0]:.3e}  |E|MAE={hist['emae'][0]:.2f} V/cm")
    for i in range(args.steps):
        l, g = vg(p); u, st = opt.update(g, st, p); p = optax.apply_updates(p, u)
        hist['loss'].append(float(l)); hist['emae'].append(e_mae(p))
        if i % max(1, args.steps // 8) == 0 or i == args.steps - 1:
            print(f"  step {i:3d}: loss={float(l):.3e}  |E|MAE={hist['emae'][-1]:.2f}")

    xs, ys, Et = emag_slice(sim, truth)
    _, _, Ei = emag_slice(sim, init)
    _, _, Er = emag_slice(sim, p)
    e0, ef = hist['emae'][0], hist['emae'][-1]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)
    it = np.arange(len(hist['loss']))
    a0 = fig.add_subplot(gs[0, 0]); a0.semilogy(it, hist['loss'], '-')
    a0.set(title='Sobolev signal loss', xlabel='step', ylabel='loss')
    a1 = fig.add_subplot(gs[0, 1]); a1.plot(it, hist['emae'], '-', c='C3')
    a1.set(title='|E| recovery error vs truth', xlabel='step', ylabel='|E| MAE (V/cm)')
    a1.axhline(e0, ls=':', c='gray'); a1.text(0, e0, f' init {e0:.1f}', va='bottom', c='gray')
    # init vs recovered scatter against truth on the slice
    a2 = fig.add_subplot(gs[0, 2])
    a2.scatter(Et.ravel(), Ei.ravel(), s=2, alpha=0.3, label=f'init (MAE {e0:.1f})')
    a2.scatter(Et.ravel(), Er.ravel(), s=2, alpha=0.4, label=f'recovered (MAE {ef:.1f})')
    lim = [min(Et.min(), Ei.min(), Er.min()), max(Et.max(), Ei.max(), Er.max())]
    a2.plot(lim, lim, 'k--', lw=1); a2.set(title='|E|: truth vs estimate',
            xlabel='true |E| (V/cm)', ylabel='estimated |E| (V/cm)'); a2.legend()

    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    vmin = float(min(Et.min(), Ei.min(), Er.min()))
    vmax = float(max(Et.max(), Ei.max(), Er.max()))
    pl = np.array(pos) / 10.0
    xl, yl = pl[:, 0] + 20.0, pl[:, 1]
    for col, (F, ttl) in enumerate([(Et, '|E| true'),
                                    (Ei, '|E| init (+20% noise)'),
                                    (Er, '|E| recovered')]):
        AX = fig.add_subplot(gs[1, col])
        im = AX.imshow(F.T, origin='lower', extent=ext, aspect='auto',
                       vmin=vmin, vmax=vmax, cmap='viridis')
        if col == 0:
            AX.scatter(xl[np.array(de) > 0], yl[np.array(de) > 0], s=0.5, c='w', alpha=0.2)
        AX.set(title=ttl, xlabel='drift x (cm)', ylabel='y (cm)')
        fig.colorbar(im, ax=AX, label='V/cm')

    fig.suptitle('Full SCE-field recovery from cosmic muons — Sobolev loss '
                 '(differentiable simulator)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f"recovered: |E|MAE {e0:.2f} → {ef:.2f} V/cm;  saved {OUT}")


if __name__ == '__main__':
    main()
