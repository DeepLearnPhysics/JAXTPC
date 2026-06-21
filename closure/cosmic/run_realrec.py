#!/usr/bin/env python3
"""CAPSTONE: recover the REAL (first-principles) field end-to-end, measured vs real.

Breaks the inverse crime: truth = a high-degree (deg-8) polynomial fit to the REAL
Poisson Delta (~real, no SIREN 1.6 cap); generate obs through the validated poly sim;
recover with a LOWER-degree (deg-6) polynomial (so it can't trivially match the truth);
measure recovered |E| vs the real field. Tests whether the polynomial path reaches its
expressivity floor (~0.14 deg-6) vs real -- beating the SIREN's ~1.6.
Frame validated: local = (x_gen, y_gen-Ly/2, z_gen-Ly/2); SIREN ~ real Delta to 0.01cm.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp, optax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'efield'))
from closure.cosmic.recover_field import build
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints
from tools.sce_siren import recover_efield_poly, poly_exps
from ElectricDistortion.io.config_loader import build_params
from ElectricDistortion.run_sce import run
from scipy.interpolate import RegularGridInterpolator

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160
Lx, Ly, Lz, E0, Q = 20., 40., 40., 500., 3e-8
OFF = np.array([0., Ly / 2, Lz / 2], np.float32)   # gen -> local centering


def fit_poly(pts_local, vals, deg, no, ns):         # LSQ poly_delta coeffs (anode-BC)
    exps = poly_exps(deg); xn = (pts_local - np.asarray(no)) / np.asarray(ns)
    mon = np.stack([xn[:, 0] ** a * xn[:, 1] ** b * xn[:, 2] ** c for (a, b, c) in exps], -1)
    A = mon * (xn[:, 0:1] + 1.0)
    return np.stack([np.linalg.lstsq(A, vals[:, i], rcond=None)[0] for i in range(3)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=256); ap.add_argument('--truth-deg', type=int, default=8)
    ap.add_argument('--rec-deg', type=int, default=6); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--out', default=os.path.join(HERE, 'realrec.json'))
    args = ap.parse_args()
    M = args.n_muons

    tsim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=os.path.join(HERE, 'truth_40cm.npz'), sce_poly_deg=args.truth_deg)
    rsim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=os.path.join(HERE, 'truth_40cm.npz'), sce_poly_deg=args.rec_deg)
    tp0 = jax.tree.map(lambda x: x[0], tsim._default_sim_params.sce_models)
    no, ns, sb = tp0['norm_offsets'], tp0['norm_scales'], tsim._sce_siren

    # real field maps
    m = run(build_params(preset='jaxtpc', overrides=dict(Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=Q,
            Nx_poisson=41, Ny_poisson=41, Nz_poisson=41, Nx_output=15, Ny_output=15, Nz_output=15)))
    ox, oy, oz = m['output_x'], m['output_y'], m['output_z']
    GX, GY, GZ = np.meshgrid(ox, oy, oz, indexing='ij'); gen = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)
    realD = np.stack([m['delta_x'].ravel(), m['delta_y'].ravel(), m['delta_z'].ravel()], -1)
    # truth = deg-8 poly fit to REAL Delta in local frame
    Ctruth = fit_poly(gen - OFF, realD, args.truth_deg, no, ns)

    def smodel(coeffs):
        return {'poly_coeffs': jnp.asarray(coeffs)[None], 'norm_offsets': no[None], 'norm_scales': ns[None],
                'E0': tp0['E0'][None], 'v0': tp0['v0'][None], 'drift_direction': tp0['drift_direction'][None]}

    # real |E| metric grid (gen frame, interior) + interpolators
    xp, yp, zp = m['x_poisson'], m['y_poisson'], m['z_poisson']
    efi = [RegularGridInterpolator((xp, yp, zp), m[k], bounds_error=False, fill_value=None) for k in ('Ex', 'Ey', 'Ez')]
    g = np.linspace(0.1, 0.9, 11); MX, MY, MZ = np.meshgrid(g * Lx, g * Ly, g * Lz, indexing='ij')
    mgen = np.stack([MX.ravel(), MY.ravel(), MZ.ravel()], -1).astype(np.float32)
    real_Emag = np.sqrt(sum(efi[i](mgen) ** 2 for i in range(3)))
    mloc = jnp.asarray(mgen - OFF)
    exps_r = poly_exps(args.rec_deg)
    def emae(coeffs):
        E = recover_efield_poly(jnp.asarray(coeffs), mloc, tp0['E0'], tp0['v0'], sb['v_table'], sb['E_table'], no, ns, exps_r)
        return float(np.mean(np.abs(np.sqrt(np.asarray((E ** 2).sum(-1))) - real_Emag)))
    # truth deg-8 |E| vs real (obs faithfulness reference)
    exps_t = poly_exps(args.truth_deg)
    Et = recover_efield_poly(jnp.asarray(Ctruth), mloc, tp0['E0'], tp0['v0'], sb['v_table'], sb['E_table'], no, ns, exps_t)
    truth_vs_real = float(np.mean(np.abs(np.sqrt(np.asarray((Et ** 2).sum(-1))) - real_Emag)))

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); P, D = [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        P.append(p); D.append(d)
    P = jnp.stack(P); D = jnp.stack(D)
    obs = [jax.lax.stop_gradient(s) for s in jax.vmap(lambda p, d: tsim.forward_segments(tsim._default_sim_params._replace(sce_models=smodel(Ctruth)), p, d, dx=STEP))(P, D)]
    nplanes = len(obs)
    rbase = rsim._default_sim_params
    def loss(coeffs):
        sg = jax.vmap(lambda p, d: rsim.forward_segments(rbase._replace(sce_models=smodel(coeffs)), p, d, dx=STEP))(P, D)
        return sum(jnp.mean((a - b) ** 2) for a, b in zip(sg, obs)) / nplanes

    nc = len(exps_r); coeffs = jnp.zeros((nc, 3)); opt = optax.adam(args.lr); st = opt.init(coeffs)
    sl = jax.jit(jax.value_and_grad(loss))
    print(f"truth deg-{args.truth_deg} (~real) | recover deg-{args.rec_deg} ({nc*3} DOF) | M={M}")
    print(f"  truth deg-{args.truth_deg} |E| vs REAL = {truth_vs_real:.3f} V/cm (obs faithfulness; SIREN was ~1.6)")
    print(f"  init recover |E| vs REAL = {emae(coeffs):.2f}")
    hist = []
    for i in range(args.steps):
        L, gd = sl(coeffs); u, st = opt.update(gd, st, coeffs); coeffs = optax.apply_updates(coeffs, u)
        if (i + 1) % 500 == 0:
            em = emae(coeffs); hist.append((i + 1, em, float(L)))
            print(f"  step {i+1}: recover |E| vs REAL = {em:.3f}  loss={float(L):.3e}")
    json.dump(dict(truth_deg=args.truth_deg, rec_deg=args.rec_deg, n_muons=M, truth_vs_real=truth_vs_real,
                   emae_vs_real=[h[1] for h in hist]), open(args.out, 'w'))
    print(f"[REALREC truth{args.truth_deg}/rec{args.rec_deg}] recover |E| vs REAL -> {hist[-1][1]:.3f} V/cm  (SIREN floor ~1.6)")


if __name__ == '__main__':
    main()
