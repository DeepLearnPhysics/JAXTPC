#!/usr/bin/env python3
"""
SCE-field closure from cosmic-ray muons via the differentiable simulator.

Demonstrates the payoff of making the SCE field a differentiable simulator
parameter (`sim_params.sce_models`): given cosmic muons with *known* straight
trajectories (entrance/exit on detector surfaces, GeV/MIP so they cross
without stopping), we can recover the space-charge field by gradient descent
on the field through the full forward simulation — no hand-derived inverse.

Why cosmics: a cosmic muon's true path is a known straight line. SCE displaces
where its ionisation lands on the wires; that displacement (in the simulated
signal) is the observable that constrains the field. Out-of-detector segments
are zeroed (value and gradient) by `generate_cosmic_chord(..., half_extents)`.

Two regimes (selectable):
  * ``strength`` (default, robust): recover a global SCE-strength scale α that
    multiplies a fixed field shape. 1-D, smooth, converges reliably — the clean
    "recover how strong the space charge is from cosmic data" demonstration.
  * ``full``: optimise all SIREN weights from a perturbed init. Signal-space is
    a weak/ill-conditioned constraint on the full field, so this needs many
    tracks and is shown as a stretch (loss reduction, partial shape recovery).

The synthetic *true* field is a smooth few-percent perturbation (|E| within
~±15% of E0) — the physical SCE regime where the v(E) inversion is
well-conditioned. (An unphysically strong field saturates the inversion clamp
and has zero gradient there — uninvertible by construction.)

Run (CPU, ~2-4 min):
    JAX_PLATFORM_NAME=cpu python3 closure/cosmic/recover_field.py
    JAX_PLATFORM_NAME=cpu python3 closure/cosmic/recover_field.py --mode full --steps 150
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tools.sce_siren as S
from tools.simulation import DetectorSimulator
from tools.particle_generator import (
    load_dedx_table_jax, generate_cosmic_chord, sample_surface_endpoints)

HALF_MM = (200.0, 200.0, 200.0)   # 40 cm cube
E0, T = 500.0, 89.0


def _wire_cfg():
    return {'volumes': [{'id': 0,
            'geometry': {'ranges': [[-20., 0.], [-20., 20.], [-20., 20.]],
                         'drift_direction': -1},
            'planes': [{'plane_id': p, 'type': t, 'angle': a, 'wire_spacing': 0.3,
                        'distance_from_anode': d, 'bias_voltage': bv}
                       for p, (t, a, d, bv) in enumerate(
                           [('first_induction', 60., 0.6, -200.),
                            ('second_induction', -60., 0.3, -200.),
                            ('collection', 0., 0., 500.)])]}],
            'readout': {'sampling_rate': 2.0, 'electrons_per_adc': 182},
            'simulation': {'drift': {'velocity': 1.6, 'longitudinal_diffusion': 6.2,
                           'transverse_diffusion': 16.3, 'electron_lifetime': 10.0},
            'charge_recombination': {'model': 'modified_box',
                                     'recomb_parameters': {'alpha': 0.93, 'beta': 0.212}}},
            'medium': {'type': 'liquid_argon',
                       'properties': {'density': 1.396, 'ionization_energy': 23.6,
                                      'excitation_ratio': 0.21},
                       'temperature': 87., 'pressure': 1.0},
            'electric_field': {'field_strength': E0}}


def build(n_seg, truth_scale, seed=1, n_tracks=8, omega_0=2.0, truth_npz=None, sce_poly_deg=None):
    """Build sim with a truth SCE field + a batch of cosmics.

    ``truth_npz``: path to a trained SIREN to use as truth (e.g. the
    first-principles SCE field from ``make_truth_40cm.py``). If None, generate a
    smooth random SIREN truth (for optimizer/ladder sanity rungs).

    ``omega_0`` sets the random truth's spatial frequency (ignored when
    ``truth_npz`` is given — the file carries its own omega). Real SCE is a very
    smooth Poisson solution, so a LOW omega is both physical and far better
    conditioned: a high omega makes ∂Δ/∂x large, pushing the v-inversion toward
    its clamp (zero-gradient) and destabilising the optimisation.
    """
    if truth_npz is not None:
        fp = truth_npz
    else:
        truth = S.init_siren(jax.random.PRNGKey(seed), hidden_features=48,
                             hidden_layers=2, omega_0=omega_0)
        truth['weights'][-1] = truth['weights'][-1] * truth_scale
        fp = os.path.join(tempfile.mkdtemp(), 'truth.npz')
        S.save_siren_npz(fp, truth, omega_0, np.array([10., 20., 20.]),
                         np.array([10., 20., 20.]), E0, T)
    sim = DetectorSimulator(_wire_cfg(), total_pad=n_seg, response_chunk_size=n_seg,
                            include_track_hits=False, differentiable=True,
                            n_segments=n_seg, iterate_mode='scan',
                            include_electric_dist=True, electric_dist_siren_path=fp,
                            sce_poly_deg=sce_poly_deg)

    logT, dedx = load_dedx_table_jax()
    rng = np.random.RandomState(0)
    ntr, seg = n_tracks, n_seg // n_tracks
    P, D = [], []
    for _ in range(ntr):
        a, b = sample_surface_endpoints(rng, HALF_MM)
        a[0] = np.clip(a[0], -200, 0); b[0] = np.clip(b[0], -200, 0)  # this volume
        p, d, _, _, _ = generate_cosmic_chord(
            jnp.array(a), jnp.array(b), 4000., seg, logT, dedx, half_extents_mm=HALF_MM)
        P.append(p); D.append(d)
    pos, de = jnp.concatenate(P), jnp.concatenate(D)
    step = float(jnp.linalg.norm(pos[1] - pos[0]))
    return sim, pos, de, step


def recover_full(sim, pos, de, step, steps=200, lr=3e-4, init_out_scale=0.5,
                 init_noise=0.10, seed=7, record_every=1,
                 reg_weight=0.0, reg_grid_n=10):
    """Full-field recovery: optimise all SIREN weights/biases (Sobolev loss).

    Only the field shape (``weights``/``biases``) is a free parameter — the
    per-volume ``E0``/``v0``/``norm_*``/``drift_direction`` are fixed physics
    and geometry and are frozen at their truth values. Returns
    ``(history, recovered_stacked_field)``.

    ``reg_weight`` adds a smoothness prior: λ · mean of the squared 2nd spatial
    differences of Δ on a ``reg_grid_n³`` grid. This penalises high-frequency
    structure — the null-space modes that line-tracks cannot constrain — while
    leaving the smooth, large-scale physical field free. It both regularises the
    under-determined inverse and suppresses the late-time signal-overfitting.
    """
    import optax
    from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p

    base = sim._default_sim_params
    truth = base.sce_models
    FIXED = {k: truth[k] for k in
             ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}
    omega = sim._sce_siren['omega_0']

    def full(par):
        return {**FIXED, 'weights': par['weights'], 'biases': par['biases']}

    def fwd(stk):
        return sim.forward_segments(base._replace(sce_models=stk), pos, de, dx=step)
    obs = [jax.lax.stop_gradient(s) for s in fwd(truth)]
    planes = tuple(range(len(obs)))
    spec_w = tuple(make_sobolev_weight(*obs[p].shape, max_pad=256, s=1.5)
                   for p in planes)

    # smoothness-prior grid (local frame, interior of the volume)
    g1 = jnp.linspace(1.0, 19.0, reg_grid_n)
    gt = jnp.linspace(-18.0, 18.0, reg_grid_n)
    GX, GY, GZ = jnp.meshgrid(g1, gt, gt, indexing='ij')
    REG_GRID = jnp.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1).astype(jnp.float32)

    def smooth_penalty(par):
        f = full(par); p0 = jax.tree.map(lambda x: x[0], f)
        d = S.siren_delta({'weights': p0['weights'], 'biases': p0['biases']},
                          REG_GRID, p0['norm_offsets'], p0['norm_scales'], omega)
        d = d.reshape(reg_grid_n, reg_grid_n, reg_grid_n, 3)
        return sum(jnp.mean(jnp.diff(d, n=2, axis=ax) ** 2) for ax in range(3))

    def loss(par):
        data = sobolev_loss_geomean_log1p(fwd(full(par)), obs, spec_w, planes)
        return data + reg_weight * smooth_penalty(par) if reg_weight > 0 else data

    Et = emag_grid(sim, truth)

    def e_mae(par):
        return float(jnp.mean(jnp.abs(emag_grid(sim, full(par)) - Et)))

    # init: perturb weights/biases (10% noise) + a coherent half-strength error
    k = jax.random.PRNGKey(seed)
    nw, nb = [], []
    for w in truth['weights']:
        k, s = jax.random.split(k)
        nw.append(w + init_noise * jnp.abs(w) * jax.random.normal(s, w.shape))
    for b in truth['biases']:
        k, s = jax.random.split(k)
        nb.append(b + init_noise * jnp.abs(b) * jax.random.normal(s, b.shape))
    nw[-1] = nw[-1] * init_out_scale
    par = {'weights': nw, 'biases': nb}

    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, max(1, steps // 10), steps, lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    st = opt.init(par)
    vg = jax.jit(jax.value_and_grad(loss))
    hist = {'loss': [float(loss(par))], 'emae': [e_mae(par)]}
    for i in range(steps):
        l, g = vg(par); u, st = opt.update(g, st, par); par = optax.apply_updates(par, u)
        if (i + 1) % record_every == 0 or i == steps - 1:
            hist['loss'].append(float(l)); hist['emae'].append(e_mae(par))
    return hist, full(par)


def recover_accum(sim, pos_all, de_all, step, steps=300, lr=3e-4, batch=8,
                  init_noise=0.5, init_out_scale=1.0, seed=7, record_every=1,
                  curl_weight=0.0, curl_grid_n=10,
                  noise_sigma=0.0, noise_seed=0, zero_suppress=0.0, val_frac=0.0,
                  real_noise=False, weight_decay=0.0, whitened=False,
                  extra_metric=None, pos_model=None, de_model=None, step_model=None):
    """Per-EVENT accumulation: each muon is its own forward/image; the loss
    averages per-event Sobolev losses over a mini-batch of muons each step.

    This is the physically-correct formulation (each cosmic is a separate
    event), and unlike cramming all tracks into one combined image it lets
    coverage actually accumulate: more muons in the pool ⇒ more independent
    constraints ⇒ better field recovery.

    pos_all : (M, S, 3) — M muons, each S segments.  de_all : (M, S).
    """
    import optax
    from tools.losses import make_sobolev_weight, sobolev_loss_single

    base = sim._default_sim_params
    truth = base.sce_models
    FIXED = {k: truth[k] for k in
             ('norm_offsets', 'norm_scales', 'E0', 'v0', 'drift_direction')}

    def full(par):
        return {**FIXED, 'weights': par['weights'], 'biases': par['biases']}

    M = pos_all.shape[0]
    # dx per muon: each cosmic chord has its own segment length (they differ by
    # up to ~1.6x), so the recombination dE/dx must use the per-muon step, not a
    # single mean (which mis-scales charge for most tracks). Accept scalar or (M,).
    step_arr = jnp.broadcast_to(jnp.asarray(step, jnp.float32), (M,))

    # Optional track-reconstruction mismatch: the OBS uses the true tracks
    # (pos_all/de_all); the recovery MODEL uses pos_model/de_model (perturbed
    # entrance/exit -> position+angle+direction errors). Defaults to the true
    # tracks (no mismatch). This tests robustness to cosmic-reco uncertainty.
    pm_all = pos_all if pos_model is None else pos_model
    dm_all = de_all if de_model is None else de_model
    sm_arr = step_arr if step_model is None else jnp.broadcast_to(
        jnp.asarray(step_model, jnp.float32), (M,))

    def fwd1(stk, p, d, s):
        return sim.forward_segments(base._replace(sce_models=stk), p, d, dx=s)

    # Plane shapes from a single-muon forward (for spec / noise-PSD sizing).
    obs0 = fwd1(truth, pos_all[0], de_all[0], step_arr[0])
    nplanes = len(obs0)
    plane_shapes = [tuple(o.shape) for o in obs0]              # (num_wires, num_time) per plane

    # STREAMING observations: we do NOT materialise all M noisy images (that OOMs
    # at M~thousands). Instead obs_for(idx) recomputes the truth forward + a
    # DETERMINISTIC per-event noise realisation for each mini-batch — keyed by the
    # GLOBAL event index, so the realisation is fixed across steps (identical to
    # adding noise once). Memory then scales with the batch, not with M ("batch
    # and accumulate"). The model forward is the only graph that carries gradient.
    noise_psd = None
    series_list = white = spectrum = nt = None
    if real_noise:
        # The ACTUAL simulator intrinsic noise: MicroBooNE model (arXiv:1705.07341)
        # — per-wire ENC = sqrt(x^2 + (y + z*L)^2) with the empirical spectral
        # shape, L = wire length. Intrinsic-only => per-wire-independent => the
        # noise covariance is diagonal, so whitening is a per-(wire,freq) reweight.
        from tools.noise import (load_noise_params, _get_noise_spectrum_shape,
                                  _generate_noise_for_plane)
        cfg = sim.config; nt = cfg.num_time_steps
        nx, ny, nz, ef, es = load_noise_params(cfg.noise_spectrum_path)
        spectrum = jnp.array(_get_noise_spectrum_shape(nt, ef, es))
        # noise RMS per wire in the SIM'S OUTPUT UNITS (ADC). The sim output is
        # already ADC, and the MicroBooNE params x,y,z are ADC — so add the noise
        # DIRECTLY, exactly like tools.noise.add_noise. (Earlier this multiplied
        # by electrons_per_adc, which made the noise 182x too large.)
        white = float(nx)
        knz = jax.random.PRNGKey(noise_seed); series_list = []; noise_psd = []
        for pl in range(nplanes):
            nw = plane_shapes[pl][0]
            L = jnp.asarray(cfg.volumes[0].wire_lengths_m[pl], jnp.float32)
            series = (ny + nz * L)                             # (num_wires,) ADC
            series_list.append(series)
            pk = jax.random.split(jax.random.fold_in(knz, 1000 + pl), 256)
            ns = jax.vmap(lambda k: _generate_noise_for_plane(
                k, nw, nt, spectrum, series, white))(pk)
            noise_psd.append(jnp.maximum(
                jnp.mean(jnp.abs(jnp.fft.rfft(ns, axis=2)) ** 2, axis=0), 1e-3))

    knoise = jax.random.PRNGKey(noise_seed)

    def obs_for(idx):
        """Observed images for events ``idx`` (global indices), on the fly:
        truth forward + deterministic per-event noise (+ zero-suppression),
        stop_gradient. Recomputed each step so nothing M-sized is stored."""
        sg = jax.vmap(lambda p, d, s: fwd1(truth, p, d, s))(
            pos_all[idx], de_all[idx], step_arr[idx])
        out = []
        for pl in range(nplanes):
            o = sg[pl]
            if real_noise or noise_sigma > 0:
                ek = jax.vmap(lambda e: jax.random.fold_in(
                    jax.random.fold_in(knoise, pl), e))(idx)
                if real_noise:
                    nw = plane_shapes[pl][0]; series = series_list[pl]
                    noise = jax.vmap(lambda k: _generate_noise_for_plane(
                        k, nw, nt, spectrum, series, white))(ek)
                else:
                    noise = jax.vmap(lambda k: noise_sigma * jax.random.normal(
                        k, plane_shapes[pl]))(ek)
                o = o + noise
            if zero_suppress > 0:
                o = jnp.where(jnp.abs(o) > zero_suppress, o, 0.0)
            out.append(jax.lax.stop_gradient(o))
        return out

    # Held-out validation muons for HONEST early stopping (the loss keeps falling
    # by fitting noise; the val loss on unseen events turns up when the field
    # starts overfitting — stop there, no peeking at truth).
    n_val = int(val_frac * M)
    n_train = M - n_val
    spec = [make_sobolev_weight(*plane_shapes[pl], max_pad=128, s=1.5)
            for pl in range(nplanes)]

    # Per-plane, per-event loss term. Two choices:
    #  - Sobolev (default): smoothness-weighted MSE — noise-UNAWARE, weights all
    #    bins by frequency, so under noise it fits the noise-dominated bins.
    #  - whitened χ² (whitened=True): the proper Gaussian likelihood for the
    #    KNOWN noise — residual whitened by the noise PSD per (wire, freq), so
    #    noisy bins/planes are down-weighted automatically and only signal above
    #    the noise drives the gradient. Requires real_noise (needs the PSD).
    if whitened:
        if noise_psd is None:
            raise ValueError("whitened loss requires real_noise=True (needs the noise PSD)")

        def lterm(model, obs_ev, pl):
            R = model - obs_ev                                  # (num_wires, num_time)
            Rhat = jnp.fft.rfft(R, axis=-1)
            return jnp.mean(jnp.abs(Rhat) ** 2 / noise_psd[pl])
    else:
        def lterm(model, obs_ev, pl):
            return sobolev_loss_single(model, obs_ev, spec[pl])

    # Soft validity (curl) penalty: push the *derived* E toward curl-free
    # (∇×E=0) without paying for the potential parameterization's transport
    # integral. E is evaluated on a regular grid and curl taken by central
    # differences. λ>0 trades data-fit for electrostatic validity; λ=0 is the
    # current setup. This lets us TEST whether validity helps recovery.
    sb = sim._sce_siren
    Lx = float(2 * truth['norm_scales'][0, 0]) if truth['norm_scales'].ndim == 2 \
        else float(2 * truth['norm_scales'][0])
    _cn = curl_grid_n
    _cx = jnp.linspace(0.1 * Lx, 0.9 * Lx, _cn)
    _ct = jnp.linspace(-0.45 * Lx, 0.45 * Lx, _cn)
    _CX, _CY, _CZ = jnp.meshgrid(_cx, _ct, _ct, indexing='ij')
    _cgrid = jnp.stack([_CX.ravel(), _CY.ravel(), _CZ.ravel()], -1).astype(jnp.float32)
    _dx = float(_cx[1] - _cx[0]); _dt = float(_ct[1] - _ct[0])

    def curl_penalty(par):
        p0 = jax.tree.map(lambda x: x[0], full(par))
        E = S.recover_efield({'weights': p0['weights'], 'biases': p0['biases']}, _cgrid,
                             p0['E0'], p0['v0'], sb['v_table'], sb['E_table'],
                             p0['norm_offsets'], p0['norm_scales'], sb['omega_0'])
        E = E.reshape(_cn, _cn, _cn, 3)
        Ex, Ey, Ez = E[..., 0], E[..., 1], E[..., 2]
        cx = jnp.gradient(Ez, _dt, axis=1) - jnp.gradient(Ey, _dt, axis=2)
        cy = jnp.gradient(Ex, _dt, axis=2) - jnp.gradient(Ez, _dx, axis=0)
        cz = jnp.gradient(Ey, _dx, axis=0) - jnp.gradient(Ex, _dt, axis=1)
        return jnp.mean(cx ** 2 + cy ** 2 + cz ** 2)

    def data_loss_idx(par, idx):
        # MODEL forward uses the (possibly perturbed) recovery tracks; obs_for
        # uses the true tracks. Equal by default (pm_all is pos_all).
        sg = jax.vmap(lambda p, d, s: fwd1(full(par), p, d, s))(
            pm_all[idx], dm_all[idx], sm_arr[idx])
        ob = obs_for(idx)
        tot = 0.0
        for pl in range(nplanes):
            per = jax.vmap(lambda a, b: lterm(a, b, pl))(sg[pl], ob[pl])
            tot = tot + jnp.mean(per)
        return tot / nplanes

    def loss(par, idx):
        data = data_loss_idx(par, idx)
        return data + curl_weight * curl_penalty(par) if curl_weight > 0 else data

    # cap the val set so the (jitted, full-batch) val forward stays cheap at large M
    _val_idx = jnp.arange(n_train, min(M, n_train + 128))

    @jax.jit
    def val_loss(par):
        return data_loss_idx(par, _val_idx)

    Et = emag_grid(sim, truth)

    def e_mae(par):
        return float(jnp.mean(jnp.abs(emag_grid(sim, full(par)) - Et)))

    def curl_rms(par):
        return float(jnp.sqrt(curl_penalty(par)))

    k = jax.random.PRNGKey(seed); nw, nb = [], []
    for w in truth['weights']:
        k, s = jax.random.split(k)
        nw.append(w + init_noise * jnp.abs(w) * jax.random.normal(s, w.shape))
    for b in truth['biases']:
        k, s = jax.random.split(k)
        nb.append(b + init_noise * jnp.abs(b) * jax.random.normal(s, b.shape))
    nw[-1] = nw[-1] * init_out_scale
    par = {'weights': nw, 'biases': nb}

    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, steps // 10),
                                               steps, lr * 0.05)
    # Weight decay = capacity control: shrinks the SIREN weights toward the
    # smooth-init prior so the field can't grow the high-frequency modes that fit
    # noise (the real noise lever — batch size / curl don't remove that freedom).
    if weight_decay > 0:
        opt = optax.chain(optax.clip_by_global_norm(0.5),
                          optax.adamw(sched, weight_decay=weight_decay))
    else:
        opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    st = opt.init(par)

    @jax.jit
    def stepf(par, st, idx):
        l, g = jax.value_and_grad(lambda p: loss(p, idx))(par)
        u, st = opt.update(g, st, par)
        return optax.apply_updates(par, u), st, l

    B = min(batch, n_train)
    rng = np.random.RandomState(0)
    # extra_metric(stacked_field) -> float: an INDEPENDENT score recorded each
    # step (e.g. recovered |E| vs the first-principles field, not the SIREN).
    hist = {'loss': [float(loss(par, jnp.arange(B)))], 'emae': [e_mae(par)],
            'curl': [curl_rms(par)],
            'val': [float(val_loss(par)) if n_val else 0.0]}
    if extra_metric is not None:
        hist['emae_real'] = [float(extra_metric(full(par)))]
    for i in range(steps):
        idx = jnp.asarray(rng.choice(n_train, size=B, replace=False))  # train only
        par, st, l = stepf(par, st, idx)
        if (i + 1) % record_every == 0 or i == steps - 1:
            hist['loss'].append(float(l)); hist['emae'].append(e_mae(par))
            hist['curl'].append(curl_rms(par))
            hist['val'].append(float(val_loss(par)) if n_val else 0.0)
            if extra_metric is not None:
                hist['emae_real'].append(float(extra_metric(full(par))))
    return hist, full(par)


def emag_grid(sim, stacked):
    """|E| (V/cm) of a field on a probe grid, for a recovery metric.

    The grid spans nearly the full volume (x in (0.5,19.5), y,z in (-19.5,19.5))
    so edge structure is not under-weighted — important for edge fields, where a
    truncated ±18 grid would optimistically miss the near-wall region.
    """
    sb = sim._sce_siren
    gx, gy, gz = np.meshgrid(np.linspace(0.5, 19.5, 10), np.linspace(-19.5, 19.5, 10),
                             np.linspace(-19.5, 19.5, 10), indexing='ij')
    grid = jnp.array(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1), jnp.float32)
    p0 = jax.tree.map(lambda x: x[0], stacked)
    E = S.recover_efield({'weights': p0['weights'], 'biases': p0['biases']}, grid,
                         p0['E0'], p0['v0'], sb['v_table'], sb['E_table'],
                         p0['norm_offsets'], p0['norm_scales'], sb['omega_0'])
    return jnp.sqrt((E ** 2).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['strength', 'full'], default='strength')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--n-seg', type=int, default=256)
    ap.add_argument('--n-tracks', type=int, default=8)
    ap.add_argument('--truth-scale', type=float, default=1.0)
    args = ap.parse_args()

    sim, pos, de, step = build(args.n_seg, args.truth_scale, n_tracks=args.n_tracks)
    base = sim._default_sim_params
    truth_stk = base.sce_models

    def fwd(stk):
        return sim.forward_segments(base._replace(sce_models=stk), pos, de, dx=step)
    obs = [jax.lax.stop_gradient(s) for s in fwd(truth_stk)]

    def sig_loss(stk):
        sg = fwd(stk)
        return sum(jnp.mean((a - b) ** 2) for a, b in zip(sg, obs)) / len(sg)

    Et = emag_grid(sim, truth_stk)

    def e_mae(stk):
        return float(jnp.mean(jnp.abs(emag_grid(sim, stk) - Et)))

    if args.mode == 'strength':
        # Recover a global SCE-strength scale α (truth α = 1).
        def scaled(alpha):
            s = dict(truth_stk); wl = list(s['weights'])
            wl[-1] = wl[-1] * alpha; s['weights'] = wl
            return s

        def loss(alpha):
            return sig_loss(scaled(alpha))

        a = jnp.array(0.4)
        sched = optax.cosine_decay_schedule(0.05, args.steps)
        opt = optax.adam(sched); st = opt.init(a)
        vg = jax.jit(jax.value_and_grad(loss))
        l0 = float(loss(a))
        print(f"[strength] recover global SCE scale (truth α=1.0)")
        print(f"  init α=0.400  loss={l0:.3e}  |E|MAE={e_mae(scaled(a)):.2f} V/cm")
        for i in range(args.steps):
            l, g = vg(a); u, st = opt.update(g, st, a); a = optax.apply_updates(a, u)
            if i % max(1, args.steps // 6) == 0 or i == args.steps - 1:
                print(f"  step {i:3d}: α={float(a):.4f}  loss={float(l):.3e}  "
                      f"|E|MAE={e_mae(scaled(a)):.2f}")
        print(f"  RECOVERED α={float(a):.4f} (truth 1.0); "
              f"loss {l0:.3e} → {float(loss(a)):.3e}")
    else:
        # Full-field: optimise all SIREN weights from a perturbed (0.5×) init,
        # using the Sobolev (screened-Poisson, H^{-s}) signal loss — same loss
        # the muon closure uses. The 1/|k|^{2s} weighting amplifies the low-
        # frequency (large-scale) signal differences a spatial SCE shift
        # produces, giving a far better-conditioned objective than per-pixel
        # MSE for recovering the field shape.
        from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p
        planes = tuple(range(len(obs)))
        spec_w = tuple(make_sobolev_weight(*obs[p].shape, max_pad=256, s=1.5)
                       for p in planes)

        def sob_loss(stk):
            return sobolev_loss_geomean_log1p(fwd(stk), obs, spec_w, planes)

        init = jax.tree.map(lambda x: 0.5 * x, truth_stk)
        sched = optax.cosine_decay_schedule(3e-3, args.steps)
        opt = optax.adam(sched); st = opt.init(init); p = init
        vg = jax.jit(jax.value_and_grad(sob_loss))
        l0, e0 = float(sob_loss(p)), e_mae(p)
        print(f"[full] optimise all SIREN weights (init 0.5×truth), Sobolev s=1.5 loss")
        print(f"  init loss={l0:.3e}  |E|MAE={e0:.2f} V/cm")
        for i in range(args.steps):
            l, g = vg(p); u, st = opt.update(g, st, p); p = optax.apply_updates(p, u)
            if i % max(1, args.steps // 6) == 0 or i == args.steps - 1:
                print(f"  step {i:3d}: loss={float(l):.3e}  |E|MAE={e_mae(p):.2f}")
        print(f"  loss {l0:.3e} → {float(sob_loss(p)):.3e};  "
              f"|E|MAE {e0:.2f} → {e_mae(p):.2f} V/cm")


if __name__ == '__main__':
    main()
