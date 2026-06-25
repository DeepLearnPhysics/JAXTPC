#!/usr/bin/env python3
"""INTEGRATED end-to-end: polynomial field + two-timescale joint fit + reco endpoint
error + noise, recovered vs the REAL field. Combines run_realrec (poly field, real-
field truth, vs-real metric) + run_twoscale (per-track endpoints, slow field / fast
anchored tracks, noise). The genuine realistic number.

Truth = deg-8 poly fit to the REAL Poisson Delta (~real). obs = forward(truth, TRUE
tracks) + noise. Recover a deg-6 poly field (SLOW) + per-track endpoints (init=reco
perturbed, anchored, FAST). Measure recovered |E| vs the REAL field.
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp, optax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'efield'))
from closure.cosmic.recover_field import build
from tools.particle_generator import load_dedx_table_jax, generate_cosmic_chord, sample_box_endpoints, mask_outside_volume
from tools.sce_siren import recover_efield_poly, poly_exps
from tools.losses import make_sobolev_weight, sobolev_loss_single
from tools.noise import load_noise_params, _get_noise_spectrum_shape, _generate_noise_for_plane
from closure.cosmic.run_scatter import scatter
from ElectricDistortion.io.config_loader import build_params
from ElectricDistortion.run_sce import run
from scipy.interpolate import RegularGridInterpolator

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160
Lx, Ly, Lz, E0, Q = 20., 40., 40., 500., 3e-8
OFF = np.array([0., Ly / 2, Lz / 2], np.float32)


def fit_poly(pts_local, vals, deg, no, ns):
    exps = poly_exps(deg); xn = (pts_local - np.asarray(no)) / np.asarray(ns)
    mon = np.stack([xn[:, 0] ** a * xn[:, 1] ** b * xn[:, 2] ** c for (a, b, c) in exps], -1)
    A = mon * (xn[:, 0:1] + 1.0)
    return np.stack([np.linalg.lstsq(A, vals[:, i], rcond=None)[0] for i in range(3)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=512)
    ap.add_argument('--truth-deg', type=int, default=8); ap.add_argument('--rec-deg', type=int, default=6)
    ap.add_argument('--ep-sigma', type=float, default=10.0); ap.add_argument('--scatter', type=float, default=0.0)
    ap.add_argument('--field-lr', type=float, default=3e-4); ap.add_argument('--ep-lr', type=float, default=0.1)
    ap.add_argument('--ep-prior', type=float, default=0.03); ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=32); ap.add_argument('--out', default=os.path.join(HERE, 'full.json'))
    args = ap.parse_args()
    M = args.n_muons

    tsim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=os.path.join(HERE, 'truth_40cm.npz'), distortion_poly_deg=args.truth_deg)
    rsim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=os.path.join(HERE, 'truth_40cm.npz'), distortion_poly_deg=args.rec_deg)
    tp0 = jax.tree.map(lambda x: x[0], tsim._default_sim_params.distortion_field)
    no, ns, sb = tp0['norm_offsets'], tp0['norm_scales'], tsim.distortion_state()
    tbase, rbase = tsim._default_sim_params, rsim._default_sim_params

    m = run(build_params(preset='jaxtpc', overrides=dict(Lx=Lx, Ly=Ly, Lz=Lz, E0=E0, Q_charge_production=Q,
            Nx_poisson=41, Ny_poisson=41, Nz_poisson=41, Nx_output=15, Ny_output=15, Nz_output=15)))
    ox, oy, oz = m['output_x'], m['output_y'], m['output_z']
    GX, GY, GZ = np.meshgrid(ox, oy, oz, indexing='ij'); gen = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(np.float32)
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

    # tracks: true (straight or scattered) + reco-perturbed endpoints
    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); jr = np.random.RandomState(123); sr = np.random.RandomState(7)
    Ptrue, De, TH_reco = [], [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        p, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        pt = scatter(p, args.scatter, sr) if args.scatter > 0 else jnp.asarray(p)
        De.append(d); Ptrue.append(pt)
        # perturb the ACTUAL chord endpoints (track(th) must match the obs track at sigma=0)
        e0, e1 = np.asarray(pt[0]), np.asarray(pt[-1])
        TH_reco.append(np.stack([e0 + jr.normal(size=3) * args.ep_sigma, e1 + jr.normal(size=3) * args.ep_sigma]))
    Ptrue = jnp.stack(Ptrue); De = jnp.stack(De); TH_reco = jnp.asarray(np.stack(TH_reco))
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))
    # true endpoints for track-error reporting (first/last true positions)
    TH_true = jnp.stack([jnp.stack([Ptrue[i, 0], Ptrue[i, -1]]) for i in range(M)])

    # obs = forward(truth, TRUE tracks) + noise
    cfg = tsim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path); spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    obs = None  # chunk the obs precompute (vmapping all M OOMs at thousands)
    for i0 in range(0, M, 256):
        o = jax.vmap(lambda p, d: tsim.forward_segments(tbase._replace(distortion_field=smodel(Ctruth)), p, d, dx=STEP))(Ptrue[i0:i0 + 256], De[i0:i0 + 256])
        if obs is None: obs = [[] for _ in o]
        for pl in range(len(o)): obs[pl].append(o[pl])
    obs = [jnp.concatenate(a, 0) for a in obs]
    knz = jax.random.PRNGKey(0); nplanes = len(obs)
    for pl in range(nplanes):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
        keys = jax.random.split(jax.random.fold_in(knz, pl), M)
        obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(k, obs[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]
    spec = [make_sobolev_weight(*obs[pl].shape[1:], max_pad=128, s=1.5) for pl in range(nplanes)]

    def track(th):
        a, b = th[0], th[1]; dirv = (b - a) / (jnp.linalg.norm(b - a) + 1e-6)
        ii = jnp.arange(NSEG, dtype=jnp.float32)
        return a[None, :] + ii[:, None] * STEP * dirv[None, :]
    def model(coeffs, th, d):
        pos = track(th)
        return rsim.forward_segments(rbase._replace(distortion_field=smodel(coeffs)), pos, mask_outside_volume(pos, d, HALF), dx=STEP)
    def loss(coeffs, TH, idx):
        sg = jax.vmap(lambda th, d: model(coeffs, th, d))(TH[idx], De[idx])
        tot = sum(jnp.mean(jax.vmap(lambda u, v: sobolev_loss_single(u, v, spec[pl]))(sg[pl], obs[pl][idx])) for pl in range(nplanes))
        return tot / nplanes + args.ep_prior * jnp.mean((TH[idx] - TH_reco[idx]) ** 2)

    nc = len(exps_r); coeffs = jnp.zeros((nc, 3)); TH = TH_reco
    of = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.field_lr)); oe = optax.adam(args.ep_lr)
    sf = of.init(coeffs); se = oe.init({'TH': TH})
    @jax.jit
    def step(coeffs, TH, sf, se, idx):
        g = jax.grad(loss, argnums=(0, 1))(coeffs, TH, idx)
        u, sf = of.update(g[0], sf, coeffs); coeffs = optax.apply_updates(coeffs, u)
        ut, se = oe.update({'TH': g[1]}, se, {'TH': TH}); TH = optax.apply_updates({'TH': TH}, ut)['TH']
        return coeffs, TH, sf, se

    truth_vs_real = emae(Ctruth_to_rec := fit_poly(gen - OFF, realD, args.rec_deg, no, ns))  # best deg-rec fit to real = floor
    rng2 = np.random.RandomState(0); B = min(args.batch, M)
    def terr(TH): return float(jnp.mean(jnp.abs(TH - TH_true)))
    print(f"poly truth deg{args.truth_deg} | recover deg{args.rec_deg} | M={M} sigma={args.ep_sigma} scatter={args.scatter} noise=on")
    print(f"  deg-{args.rec_deg} best-fit-to-real floor = {truth_vs_real:.3f} V/cm (SIREN was ~1.6)")
    print(f"  init: |E| vs REAL = {emae(coeffs):.2f}  track_err = {terr(TH):.2f}mm")
    hist = []
    for i in range(args.steps):
        idx = jnp.asarray(rng2.choice(M, B, replace=False))
        coeffs, TH, sf, se = step(coeffs, TH, sf, se, idx)
        if (i + 1) % 1000 == 0:
            em, te = emae(coeffs), terr(TH); hist.append((i + 1, em, te))
            print(f"  step {i+1}: |E| vs REAL = {em:.3f}  track_err = {te:.2f}mm")
    json.dump(dict(n_muons=M, ep_sigma=args.ep_sigma, scatter=args.scatter, rec_deg=args.rec_deg,
                   floor=truth_vs_real, emae_vs_real=[h[1] for h in hist], track_err=[h[2] for h in hist]), open(args.out, 'w'))
    print(f"[FULL M={M} sig={args.ep_sigma} sc={args.scatter}] |E| vs REAL -> {hist[-1][1]:.3f} V/cm (floor {truth_vs_real:.2f}; SIREN ~1.6)")


if __name__ == '__main__':
    main()
