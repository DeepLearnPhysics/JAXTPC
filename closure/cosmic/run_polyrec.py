#!/usr/bin/env python3
"""Polynomial-field recovery THROUGH the sim (the validated poly _apply_distortion).

Field-only (true tracks), obs from the SIREN truth; recover a polynomial Delta field
via Adam. Metric = poly |E| vs the SIREN-truth |E| on the local emag grid. This is
the end-to-end test that the (linear, well-conditioned) polynomial field recovers
through the real sim pipeline. Pairs with run_express (poly ~ real field to 0.14):
together they say a polynomial recovery would reach ~0.14 vs real (vs the SIREN's 1.6),
modulo a real-field truth (frame-careful next step).
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp, optax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from closure.cosmic.recover_field import build
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints, mask_outside_volume
from tools.sce_siren import recover_efield_poly, poly_exps

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=128)
    ap.add_argument('--deg', type=int, default=6)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', default=os.path.join(HERE, 'polyrec.json'))
    args = ap.parse_args()
    M, deg = args.n_muons, args.deg

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth, distortion_poly_deg=deg)
    base = sim._default_sim_params; truth = base.distortion_field; tp0 = jax.tree.map(lambda x: x[0], truth)
    no, ns, sb = tp0['norm_offsets'], tp0['norm_scales'], sim.distortion_state()
    exps = poly_exps(deg); ncoef = len(exps)
    def fwd(sm, p, d): return sim.forward_segments(base._replace(distortion_field=sm), p, d, dx=STEP)
    def polysm(coeffs):
        return {'poly_coeffs': coeffs[None], 'norm_offsets': no[None], 'norm_scales': ns[None],
                'E0': tp0['E0'][None], 'v0': tp0['v0'][None], 'drift_direction': tp0['drift_direction'][None]}

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0)
    P, D = [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        P.append(p); D.append(d)
    P = jnp.stack(P); D = jnp.stack(D)
    obs = [jax.lax.stop_gradient(s) for s in jax.vmap(lambda p, d: fwd(truth, p, d))(P, D)]
    nplanes = len(obs)

    # metric grid (local frame, same as emag_grid): poly |E| vs SIREN-truth |E|
    from closure.cosmic.recover_field import emag_grid
    Et = emag_grid(sim, truth)                      # SIREN-truth |E|
    gx, gy, gz = np.meshgrid(np.linspace(0.5, 19.5, 10), np.linspace(-19.5, 19.5, 10), np.linspace(-19.5, 19.5, 10), indexing='ij')
    grid = jnp.array(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1), jnp.float32)
    def poly_emag(coeffs):
        E = recover_efield_poly(coeffs, grid, tp0['E0'], tp0['v0'], sb['v_table'], sb['E_table'], no, ns, exps)
        return jnp.sqrt((E ** 2).sum(-1))
    def emae(coeffs): return float(jnp.mean(jnp.abs(poly_emag(coeffs) - Et)))

    def loss(coeffs):
        sg = jax.vmap(lambda p, d: fwd(polysm(coeffs), p, d))(P, D)
        return sum(jnp.mean((a - b) ** 2) for a, b in zip(sg, obs)) / nplanes

    coeffs = jnp.zeros((ncoef, 3)); opt = optax.adam(args.lr); st = opt.init(coeffs)
    sl = jax.jit(jax.value_and_grad(loss))
    print(f"poly recovery deg={deg} ({ncoef*3} DOF), M={M}; metric vs SIREN-truth |E| (SIREN floor vs real ~1.6; poly vs real ~0.14 offline)")
    print(f"  init: |E|MAE vs SIREN-truth = {emae(coeffs):.2f}")
    hist = []
    for i in range(args.steps):
        L, g = sl(coeffs); u, st = opt.update(g, st, coeffs); coeffs = optax.apply_updates(coeffs, u)
        if (i + 1) % 500 == 0:
            em = emae(coeffs); hist.append((i + 1, em, float(L)))
            print(f"  step {i+1}: |E|MAE vs SIREN-truth = {em:.3f}  loss={float(L):.3e}")
    json.dump(dict(deg=deg, n_muons=M, emae_vs_siren=[h[1] for h in hist]), open(args.out, 'w'))
    print(f"[POLYREC deg={deg}] |E|MAE vs SIREN-truth {emae(jnp.zeros((ncoef,3))):.2f} -> {hist[-1][1]:.3f}")


if __name__ == '__main__':
    main()
