#!/usr/bin/env python3
"""#2: polynomial-Delta field recovered THROUGH the sim (the proposed fix, tested).

The representation test (run_reprtest) showed a polynomial basis CAN represent the
field offline; this tests RECOVERABILITY through the forward sim. Self-consistent
mechanism: disable the SIREN-SCE (zero field) and shift deposit positions by Delta
DIRECTLY -- obs uses the truth SIREN's Delta, the model uses a polynomial Delta
(anode-BC enforced via the (x_norm+1) factor, linear in coeffs). Field-only, tracks
fixed at truth: does it beat the random-subspace floor (|E|=20) and reach the
representation floor (~0.65)? If yes, a good low-DOF basis IS the fix.
"""
import os, sys, argparse, json
import numpy as np, jax, jax.numpy as jnp, optax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build, emag_grid
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints
from tools.sce_siren import siren_delta, efield_from_dDdx

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=128)
    ap.add_argument('--deg', type=int, default=5)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', default=os.path.join(HERE, 'poly.json'))
    args = ap.parse_args()
    M, deg = args.n_muons, args.deg

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.sce_models
    tp0 = jax.tree.map(lambda x: x[0], truth); sb = sim._sce_siren
    no, ns, om = tp0['norm_offsets'], tp0['norm_scales'], sb['omega_0']
    E0, v0, vt, et = tp0['E0'], tp0['v0'], sb['v_table'], sb['E_table']
    zero_field = {**truth, 'weights': [jnp.zeros_like(w) for w in truth['weights']],
                  'biases': [jnp.zeros_like(b) for b in truth['biases']]}
    def fwd(field, pos, de): return sim.forward_segments(base._replace(sce_models=field), pos, de, dx=STEP)

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    POS, DE = [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        POS.append(p); DE.append(d)
    POS = jnp.stack(POS); DE = jnp.stack(DE)              # mm

    tparams = {'weights': tp0['weights'], 'biases': tp0['biases']}
    def truthD_mm(pos_mm): return siren_delta(tparams, pos_mm / 10., no, ns, om) * 10.
    obs = [jax.lax.stop_gradient(s) for s in jax.vmap(lambda p, d: fwd(zero_field, p + truthD_mm(p), d))(POS, DE)]
    nplanes = len(obs)

    # polynomial basis (cm coords, normalized), anode-BC factor (x_norm+1)
    exps = jnp.array([(a, b, c) for a in range(deg + 1) for b in range(deg + 1) for c in range(deg + 1) if a + b + c <= deg], jnp.float32)
    ncoef = exps.shape[0]
    def polyD_cm(coeffs, pos_cm):                          # coeffs (ncoef,3) -> Delta_cm (...,3)
        xn = (pos_cm - no) / ns
        mon = jnp.prod(xn[..., None, :] ** exps, axis=-1)  # (...,ncoef)
        return (mon @ coeffs) * (xn[..., 0:1] + 1.0)
    def model(coeffs):
        return jax.vmap(lambda p, d: fwd(zero_field, p + polyD_cm(coeffs, p / 10.) * 10., d))(POS, DE)
    def loss(coeffs):
        sg = model(coeffs)
        return sum(jnp.mean((a - b) ** 2) for a, b in zip(sg, obs)) / nplanes

    # metrics on the emag grid
    Et = emag_grid(sim, truth)
    gx, gy, gz = np.meshgrid(np.linspace(0.5, 19.5, 10), np.linspace(-19.5, 19.5, 10), np.linspace(-19.5, 19.5, 10), indexing='ij')
    grid = jnp.array(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1), jnp.float32)
    Dt_grid = siren_delta(tparams, grid, no, ns, om); Dtmag = float(jnp.mean(jnp.abs(Dt_grid)))
    def poly_emag(coeffs):
        dDdx = jax.vmap(lambda x: jax.jvp(lambda xx: polyD_cm(coeffs, xx), (x,), (jnp.array([1., 0., 0.]),))[1])(grid)
        return jnp.sqrt((efield_from_dDdx(dDdx, E0, v0, vt, et) ** 2).sum(-1))
    def emae(coeffs): return float(jnp.mean(jnp.abs(poly_emag(coeffs) - Et)))
    def dmae(coeffs): return float(jnp.mean(jnp.abs(polyD_cm(coeffs, grid) - Dt_grid)))

    coeffs = jnp.zeros((ncoef, 3))
    opt = optax.adam(args.lr); st = opt.init(coeffs)
    sl = jax.jit(jax.value_and_grad(loss))
    print(f"deg={deg} ncoef={ncoef*3} DOF | truth |E| mean=~500, Delta={Dtmag*1e4:.0f}um | random-subspace floor was |E|=20")
    print(f"  init: |E|MAE={emae(coeffs):.2f}  Delta MAE={dmae(coeffs)*1e4:.0f}um")
    hist = []
    for i in range(args.steps):
        L, g = sl(coeffs); u, st = opt.update(g, st, coeffs); coeffs = optax.apply_updates(coeffs, u)
        if (i + 1) % 500 == 0:
            em, dm = emae(coeffs), dmae(coeffs); hist.append((i + 1, em, dm, float(L)))
            print(f"  step {i+1}: |E|MAE={em:.2f}  Delta MAE={dm*1e4:.0f}um ({100*dm/Dtmag:.0f}%)  loss={float(L):.4e}")
    json.dump(dict(n_muons=M, deg=deg, ncoef3=ncoef * 3, truth_delta_um=Dtmag * 1e4,
                   emae=[h[1] for h in hist], dmae_um=[h[2] * 1e4 for h in hist]), open(args.out, 'w'))
    print(f"[POLY M={M} deg={deg}] |E|MAE {emae(jnp.zeros((ncoef,3))):.1f} -> {hist[-1][1]:.2f}  (rand-subspace floor 20; repr floor ~0.65)")


if __name__ == '__main__':
    main()
