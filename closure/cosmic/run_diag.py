#!/usr/bin/env python3
"""Systematic diagnosis: is the joint-fit plateau a BAD LANDSCAPE or BAD OPTIMIZER?

Endpoint-only case (true track exactly representable, so 'truth' is unambiguous).
Probes:
  L(truth field, TRUE endpoints)   -- the ideal data fit
  L(truth field, RECO endpoints)   -- truth field, wrong tracks
  cold-start joint fit  -> plateau: compare L_plateau vs L_truth, and mae
  truth-init joint fit  -> does the field STAY at truth (mae~0) or DRIFT?
  truth-init field-only (reco tracks fixed) -> does the field drift = bias test
Reports data-loss (no prior) at each point + |E| MAE, to separate the cases.
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

LO, HI = (-200., -200., -200.), (0., 200., 200.); HALF = (200., 200., 200.); STEP, NSEG = 4.0, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-muons', type=int, default=256)
    ap.add_argument('--ep-sigma', type=float, default=10.0)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--ep-lr', type=float, default=0.1)
    ap.add_argument('--ep-prior', type=float, default=0.03)
    ap.add_argument('--truth', default=os.path.join(HERE, 'truth_40cm.npz'))
    ap.add_argument('--out', default=os.path.join(HERE, 'diag.json'))
    args = ap.parse_args()
    M = args.n_muons

    sim, _, _, _ = build(160, 1.0, n_tracks=1, truth_npz=args.truth)
    base = sim._default_sim_params; truth = base.distortion_field
    FIXED = {k: truth[k] for k in ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    def full(fp): return {**FIXED, 'weights': fp['weights'], 'biases': fp['biases']}
    def fwd(stk, pos, de): return sim.forward_segments(base._replace(distortion_field=stk), pos, de, dx=STEP)
    fp_truth = {'weights': [w for w in truth['weights']], 'biases': [b for b in truth['biases']]}

    logT, dedx = load_dedx_table_jax(); rng = np.random.RandomState(0); jr = np.random.RandomState(123)
    De, TH_true, TH_reco = [], [], []
    for _ in range(M):
        a, b = sample_box_endpoints(rng, LO, HI)
        _, d, _, _, _ = generate_cosmic_chord(jnp.array(a), jnp.array(b), 4000., NSEG, logT, dedx, half_extents_mm=HALF, step_mm=STEP)
        De.append(d); TH_true.append(np.stack([a, b]))
        TH_reco.append(np.stack([a + jr.normal(size=3) * args.ep_sigma, b + jr.normal(size=3) * args.ep_sigma]))
    De = jnp.stack(De); TH_true = jnp.asarray(np.stack(TH_true)); TH_reco = jnp.asarray(np.stack(TH_reco))
    de_mip = float(np.mean(np.asarray(De)[np.asarray(De) > 0]))

    def positions(th):
        a, b = th[0], th[1]; dirv = (b - a) / (jnp.linalg.norm(b - a) + 1e-6)
        ii = jnp.arange(NSEG, dtype=jnp.float32)
        return a[None, :] + ii[:, None] * STEP * dirv[None, :]

    def model(fp, th, d):
        pos = positions(th); return fwd(full(fp), pos, mask_outside_volume(pos, jnp.full(NSEG, de_mip), HALF))
    # obs: truth field through the TRUE tracks + real noise
    cfg = sim.config; nt = cfg.num_time_steps
    nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path); spn = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
    obs = None
    for i in range(0, M, 256):
        o = jax.vmap(lambda th, d: model(fp_truth, th, d))(TH_true[i:i + 256], De[i:i + 256])
        if obs is None: obs = [[] for _ in o]
        for pl in range(len(o)): obs[pl].append(o[pl])
    obs = [jnp.concatenate(a, 0) for a in obs]
    knz = jax.random.PRNGKey(0)
    for pl in range(len(obs)):
        L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
        keys = jax.random.split(jax.random.fold_in(knz, pl), obs[pl].shape[0])
        obs[pl] = obs[pl] + jax.vmap(lambda k: _generate_noise_for_plane(k, obs[pl].shape[1], nt, spn, ny + nz * L, float(nx)))(keys)
    obs = [jax.lax.stop_gradient(o) for o in obs]; nplanes = len(obs)
    spec = [make_sobolev_weight(*obs[pl].shape[1:], max_pad=128, s=1.5) for pl in range(nplanes)]

    def data_loss(fp, TH, idx):                 # pure data fit (no prior) over batch idx
        sg = jax.vmap(lambda th, d: model(fp, th, d))(TH[idx], De[idx])
        tot = 0.0
        for pl in range(nplanes):
            tot += jnp.mean(jax.vmap(lambda u, v: sobolev_loss_single(u, v, spec[pl]))(sg[pl], obs[pl][idx]))
        return tot / nplanes
    allidx = jnp.arange(M)
    def Lfull(fp, TH): return float(jax.jit(lambda f, T: data_loss(f, T, allidx))(fp, TH))
    Et = emag_grid(sim, truth)
    def fmae(fp): return float(jnp.mean(jnp.abs(emag_grid(sim, full(fp)) - Et)))
    # Delta-space metric (what the data integrates; |E| is its x-derivative)
    from tools.sce_siren import siren_delta
    _sb = sim.distortion_state()
    _gx, _gy, _gz = np.meshgrid(np.linspace(0.5, 19.5, 10), np.linspace(-19.5, 19.5, 10),
                                np.linspace(-19.5, 19.5, 10), indexing='ij')
    _gd = jnp.array(np.stack([_gx.ravel(), _gy.ravel(), _gz.ravel()], -1), jnp.float32)
    _tp0 = jax.tree.map(lambda x: x[0], truth)
    _Dt = siren_delta({'weights': _tp0['weights'], 'biases': _tp0['biases']}, _gd,
                      _tp0['norm_offsets'], _tp0['norm_scales'], _sb['omega_0'])
    _Dtmag = float(jnp.mean(jnp.abs(_Dt)))
    def dmae(fp):
        w = [x[0] if x.ndim == 3 else x for x in fp['weights']]
        b = [x[0] if x.ndim == 2 else x for x in fp['biases']]
        Dr = siren_delta({'weights': w, 'biases': b}, _gd,
                         _tp0['norm_offsets'], _tp0['norm_scales'], _sb['omega_0'])
        return float(jnp.mean(jnp.abs(Dr - _Dt)))

    # --- landscape reference points (data loss, field mae) ---
    pts = {'truth+TRUEtracks': (Lfull(fp_truth, TH_true), 0.0),
           'truth+RECOtracks': (Lfull(fp_truth, TH_reco), 0.0)}

    # --- optimizer: joint field+endpoints from a given init ---
    def run_joint(fp0, TH0, fit_tracks=True, tag=''):
        fp = {'weights': [w for w in fp0['weights']], 'biases': [b for b in fp0['biases']]}; TH = TH0
        of = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4)); oe = optax.adam(args.ep_lr)
        sf = of.init(fp); se = oe.init({'TH': TH})
        @jax.jit
        def step(fp, TH, sf, se, idx):
            g = jax.grad(lambda f, T: data_loss(f, T, idx) + args.ep_prior * jnp.mean((T[idx] - TH_reco[idx]) ** 2), argnums=(0, 1))(fp, TH)
            u, sf = of.update(g[0], sf, fp); fp = optax.apply_updates(fp, u)
            if fit_tracks:
                ut, se = oe.update({'TH': g[1]}, se, {'TH': TH}); TH = optax.apply_updates({'TH': TH}, ut)['TH']
            return fp, TH, sf, se
        rng2 = np.random.RandomState(0); B = min(16, M); traj = [(fmae(fp), Lfull(fp, TH), dmae(fp))]
        for i in range(args.steps):
            fp, TH, sf, se = step(fp, TH, sf, se, jnp.asarray(rng2.choice(M, B, replace=False)))
            if (i + 1) % 1000 == 0: traj.append((fmae(fp), Lfull(fp, TH), dmae(fp)))
        return fp, TH, traj

    fp_init = {'weights': [w + 0.5 * jnp.abs(w) * jax.random.normal(jax.random.PRNGKey(int(s)), w.shape)
                           for s, w in enumerate(truth['weights'])],
               'biases': [b for b in truth['biases']]}
    _, _, traj_cold = run_joint(fp_init, TH_reco, True, 'cold')           # cold start joint
    _, _, traj_tinit = run_joint(fp_truth, TH_reco, True, 'truthinit')    # truth-init joint
    _, _, traj_fonly = run_joint(fp_truth, TH_reco, False, 'fieldonly')   # truth-init, tracks fixed wrong

    res = dict(ep_sigma=args.ep_sigma, n_muons=M, steps=args.steps,
               L_truth_true=pts['truth+TRUEtracks'][0], L_truth_reco=pts['truth+RECOtracks'][0],
               cold=dict(mae=traj_cold[-1][0], L=traj_cold[-1][1], traj=traj_cold),
               truthinit=dict(mae=traj_tinit[-1][0], L=traj_tinit[-1][1], traj=traj_tinit),
               fieldonly=dict(mae=traj_fonly[-1][0], L=traj_fonly[-1][1], traj=traj_fonly))
    json.dump(res, open(args.out, 'w'))
    print(f"DATA-LOSS reference:  truth+true={res['L_truth_true']:.4f}   truth+reco={res['L_truth_reco']:.4f}")
    print(f"truth Delta mean = {_Dtmag*1e4:.1f} um  (|E|_mae in V/cm ; Delta_mae in um and %truthDelta)")
    def line(tag, tr):
        print(f"  {tag:>26}: |E|_mae {tr[0][0]:.2f}->{tr[-1][0]:.2f}   Delta_mae {tr[0][2]*1e4:.1f}->{tr[-1][2]*1e4:.1f}um "
              f"({100*tr[0][2]/_Dtmag:.0f}%->{100*tr[-1][2]/_Dtmag:.0f}%)")
    line('cold-start joint', traj_cold)
    line('TRUTH-INIT joint', traj_tinit)
    line('truth-init FIELDONLY(wrong tk)', traj_fonly)
    print("-> if TRUTH-INIT Delta barely moves but |E| swings, the drift is a DERIVATIVE-METRIC artifact.")


if __name__ == '__main__':
    main()
