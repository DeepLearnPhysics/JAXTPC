#!/usr/bin/env python3
"""SCE joint fit with PHYSICS-DOF muons (the closure/muon reconstruction) + poly field.

The track problem is solved by the muon closure: parameterize each muon by physics DOF
(vertex x,y,z + direction sin/cos theta,phi + energy) -- the data reconstructs these to
sub-mm (no free-endpoint sliding). A physics-DOF muon is a RIGID straight line: it CANNOT
absorb the smooth SCE curvature, so the field gets the SCE cleanly. Truth = deg-8 poly ~
real; obs = forward(truth, truth-muons) + noise; recover deg-6 poly field + per-muon
physics DOF (init perturbed = rough reco), jointly. Measure field vs REAL.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp, optax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'efield'))
from closure.cosmic.recover_field import build
from tools.particle_generator import (load_dedx_table_jax, generate_muon_segments_trig,
                                       sample_box_endpoints, mask_outside_volume, get_half_extents_mm)
from tools.sce_siren import recover_efield_poly, poly_exps
from tools.losses import make_sobolev_weight, sobolev_loss_single
from tools.noise import load_noise_params, _get_noise_spectrum_shape, _generate_noise_for_plane
from ElectricDistortion.io.config_loader import build_params
from ElectricDistortion.run_sce import run
from scipy.interpolate import RegularGridInterpolator

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160
Lx, Ly, Lz, E0, Q = 20., 40., 40., 500., 3e-8
OFF = np.array([0., Ly / 2, Lz / 2], np.float32)
SC = jnp.array([200., 200., 200., 1., 1., 1., 1., 500.], jnp.float32)  # phys scales for optimizer


def fit_poly(pts_local, vals, deg, no, ns):
    exps = poly_exps(deg); xn = (pts_local - np.asarray(no)) / np.asarray(ns)
    mon = np.stack([xn[:, 0] ** a * xn[:, 1] ** b * xn[:, 2] ** c for (a, b, c) in exps], -1)
    A = mon * (xn[:, 0:1] + 1.0)
    return np.stack([np.linalg.lstsq(A, vals[:, i], rcond=None)[0] for i in range(3)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=256); ap.add_argument('--truth-deg', type=int, default=8)
    ap.add_argument('--rec-deg', type=int, default=6); ap.add_argument('--vtx-sigma', type=float, default=10.0)
    ap.add_argument('--dir-sigma', type=float, default=0.02); ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--field-lr', type=float, default=3e-3); ap.add_argument('--mu-lr', type=float, default=1e-3)
    ap.add_argument('--mu-prior', type=float, default=0.1, help='anchor muon params to reco (breaks low-order field<->muon degeneracy)')
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--out', default=os.path.join(HERE, 'muonsce.json'))
    args = ap.parse_args()
    M = args.n_muons; logT, dedx = load_dedx_table_jax()

    tsim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=os.path.join(HERE, 'truth_40cm.npz'), sce_poly_deg=args.truth_deg)
    rsim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=os.path.join(HERE, 'truth_40cm.npz'), sce_poly_deg=args.rec_deg)
    tp0 = jax.tree.map(lambda x: x[0], tsim._default_sim_params.sce_models)
    no, ns, sb = tp0['norm_offsets'], tp0['norm_scales'], tsim._sce_siren
    tbase, rbase = tsim._default_sim_params, rsim._default_sim_params

    m = run(build_params(preset='jaxtpc', overrides=dict(Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=Q,
            Nx_poisson=41, Ny_poisson=41, Nz_poisson=41, Nx_output=15, Ny_output=15, Nz_output=15)))
    GX, GY, GZ = np.meshgrid(m['output_x'], m['output_y'], m['output_z'], indexing='ij')
    gen = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)
    realD = np.stack([m['delta_x'].ravel(), m['delta_y'].ravel(), m['delta_z'].ravel()], -1)
    Ctruth = fit_poly(gen - OFF, realD, args.truth_deg, no, ns)

    def smodel(coeffs):
        return {'poly_coeffs': jnp.asarray(coeffs)[None], 'norm_offsets': no[None], 'norm_scales': ns[None],
                'E0': tp0['E0'][None], 'v0': tp0['v0'][None], 'drift_direction': tp0['drift_direction'][None]}

    xp, yp, zp = m['x_poisson'], m['y_poisson'], m['z_poisson']
    efi = [RegularGridInterpolator((xp, yp, zp), m[k], bounds_error=False, fill_value=None) for k in ('Ex', 'Ey', 'Ez')]
    g = np.linspace(0.1, 0.9, 11); MX, MY, MZ = np.meshgrid(g * Lx, g * Ly, g * Lz, indexing='ij')
    mgen = np.stack([MX.ravel(), MY.ravel(), MZ.ravel()], -1).astype(np.float32)
    real_Emag = np.sqrt(sum(efi[i](mgen) ** 2 for i in range(3))); mloc = jnp.asarray(mgen - OFF)
    exps_r = poly_exps(args.rec_deg)
    def emae(coeffs):
        E = recover_efield_poly(jnp.asarray(coeffs), mloc, tp0['E0'], tp0['v0'], sb['v_table'], sb['E_table'], no, ns, exps_r)
        return float(np.mean(np.abs(np.sqrt(np.asarray((E ** 2).sum(-1))) - real_Emag)))

    # truth muons in physics DOF; reco-init = truth + perturbation
    rng = np.random.RandomState(0); jr = np.random.RandomState(7)
    PHYS = []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI); d = (b - a); d = d / (np.linalg.norm(d) + 1e-9)
        st = float(np.hypot(d[0], d[1])); ct = float(d[2]); sp = float(d[1] / (st + 1e-9)); cp = float(d[0] / (st + 1e-9))
        PHYS.append([a[0], a[1], a[2], st, ct, sp, cp, 4000.])
    PHYS = jnp.asarray(np.array(PHYS, np.float32))                       # truth physics (M,8)
    pert = np.zeros((M, 8), np.float32)
    pert[:, :3] = jr.normal(size=(M, 3)) * args.vtx_sigma
    pert[:, 3:7] = jr.normal(size=(M, 4)) * args.dir_sigma
    PHYS0 = PHYS + jnp.asarray(pert)                                     # rough-reco init

    def muon(phys):                                                      # physics DOF -> positions, de
        pos, de = generate_muon_segments_trig(phys[7], phys[:3], phys[3], phys[4], phys[5], phys[6], STEP, NSEG, logT, dedx)
        return pos, mask_outside_volume(pos, de, HALF)

    # obs = truth field through truth muons + noise
    cfg = tsim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path); spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    obs = None
    for i0 in range(0, M, 128):
        o = jax.vmap(lambda ph: (lambda pd: tsim.forward_segments(tbase._replace(sce_models=smodel(Ctruth)), pd[0], pd[1], dx=STEP))(muon(ph)))(PHYS[i0:i0 + 128])
        if obs is None: obs = [[] for _ in o]
        for pl in range(len(o)): obs[pl].append(o[pl])
    obs = [jnp.concatenate(a, 0) for a in obs]; nplanes = len(obs); knz = jax.random.PRNGKey(0)
    for pl in range(nplanes):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
        keys = jax.random.split(jax.random.fold_in(knz, pl), M)
        obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(k, obs[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]
    spec = [make_sobolev_weight(*obs[pl].shape[1:], max_pad=128, s=1.5) for pl in range(nplanes)]

    def model(coeffs, phys):
        pos, de = muon(phys)
        return rsim.forward_segments(rbase._replace(sce_models=smodel(coeffs)), pos, de, dx=STEP)
    N0 = PHYS0 / SC                                                      # reco anchor (normalized)
    def loss(coeffs, N, idx):                                            # N = phys/SC (normalized for optimizer)
        sg = jax.vmap(lambda n: model(coeffs, n * SC))(N[idx])
        data = sum(jnp.mean(jax.vmap(lambda u, v: sobolev_loss_single(u, v, spec[pl]))(sg[pl], obs[pl][idx])) for pl in range(nplanes)) / nplanes
        return data + args.mu_prior * jnp.mean((N[idx] - N0[idx]) ** 2)   # anchor muons to reco

    nc = len(exps_r); coeffs = jnp.zeros((nc, 3)); N = PHYS0 / SC
    of = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.field_lr)); om = optax.adam(args.mu_lr)
    sf = of.init(coeffs); sm = om.init({'N': N})
    @jax.jit
    def step(coeffs, N, sf, sm, idx):
        gc, gN = jax.grad(loss, argnums=(0, 1))(coeffs, N, idx)
        u, sf = of.update(gc, sf, coeffs); coeffs = optax.apply_updates(coeffs, u)
        un, sm = om.update({'N': gN}, sm, {'N': N}); N = optax.apply_updates({'N': N}, un)['N']
        return coeffs, N, sf, sm
    def vtxerr(N): return float(jnp.mean(jnp.abs((N * SC)[:, :3] - PHYS[:, :3])))

    rng2 = np.random.RandomState(0); B = min(args.batch, M)
    print(f"physics-DOF muons | truth deg{args.truth_deg} rec deg{args.rec_deg} | M={M} vtx_sig={args.vtx_sigma}mm dir_sig={args.dir_sigma}")
    print(f"  init: |E| vs REAL = {emae(coeffs):.2f}  vtx_err = {vtxerr(N):.2f}mm")
    hist = []
    for i in range(args.steps):
        idx = jnp.asarray(rng2.choice(M, B, replace=False))
        coeffs, N, sf, sm = step(coeffs, N, sf, sm, idx)
        if (i + 1) % 1000 == 0:
            em, ve = emae(coeffs), vtxerr(N); hist.append((i + 1, em, ve))
            print(f"  step {i+1}: |E| vs REAL = {em:.3f}  vtx_err = {ve:.2f}mm")
    json.dump(dict(n_muons=M, vtx_sigma=args.vtx_sigma, emae_vs_real=[h[1] for h in hist], vtx_err=[h[2] for h in hist]), open(args.out, 'w'))
    print(f"[MUONSCE M={M} vtx_sig={args.vtx_sigma}] |E| vs REAL -> {hist[-1][1]:.3f} V/cm (SIREN-field-recovery was ~6 with sliding endpoints)")


if __name__ == '__main__':
    main()
