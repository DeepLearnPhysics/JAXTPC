"""Systematic study of MCS scattering-angle recoverability from wire signals.

Builds the truth track + differentiable wire forward ONCE, then sweeps the
scattering-angle regularisation knobs and reports physically-interpretable
metrics.  Motivation (see FINDINGS_MCS.md):

  * Globals-only (vertex / direction / energy with scattering fixed) is
    essentially solved (~0.15 mm, ~0.02 deg, ~1 % E).
  * The 2N free per-segment scattering angles are badly ill-posed: the wire
    pitch (~3 mm) is 6x coarser than the 0.5 mm step, so wire signals only
    constrain the *integrated trajectory*, not the fine angles.  With the
    stock setup (lambda_prior=1e-3, no LR decay) the optimiser builds a
    low-frequency coherent bend that overfits the wire signal — cumulative
    deflection blows up to ~13x Highland and the global energy degrades.

This harness tests three levers against that failure:
  1. coarse-graining G : one scattering angle per block of G segments
     (M = N/G free angles), directly cutting low-frequency DOF.
  2. lambda_prior      : strength of the Highland prior (traced -> no recompile).
  3. LR decay + early-stop at best wire-loss.

Metrics per run:
  wire_loss      final wire (Sobolev) loss at the best (early-stopped) params
  E_err_MeV      global energy error (full mode only)
  traj_rms_mm    RMS position residual between fit and truth track (mm)
  cum_ratio      |cumulative deflection|_fit / _truth  (1.0 = right magnitude)
  cum_corr       correlation of cumulative deflection (trajectory shape)
  win_corr       correlation of 3 mm-windowed deflection (fine recovery)

Run:
    python3 -m closure.mcs.study --study scattering
    python3 -m closure.mcs.study --study full
"""

import argparse
import time
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from closure.mcs.forward import (
    mcs_cumsum_forward, generate_mcs_truth, highland_prior, build_mcs_forward,
    mask_outside_volume,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight

# ---- truth config (matches run.py) ----
N_SEGMENTS = 2000
STEP_SIZE_MM = 0.5
STEP_SIZE_CM = STEP_SIZE_MM / 10.0
TRUTH_X, TRUTH_Y, TRUTH_Z = -200.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 6
TRUTH_ENERGY = 500.0
ANGLE_SCALE = 0.01
ENERGY_SCALE = 500.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRUTH_SIN_TH = float(np.sin(TRUTH_THETA))
TRUTH_COS_TH = float(np.cos(TRUTH_THETA))
TRUTH_SIN_PH = float(np.sin(TRUTH_PHI))
TRUTH_COS_PH = float(np.cos(TRUTH_PHI))


def _direction(sth, cth, sph, cph):
    d = jnp.array([sth * cph, sth * sph, cth])
    return d / jnp.linalg.norm(d)


def expand_coarse(coarse, G, M):
    """(2M,) coarse angles -> (N,),(N,) per-segment angles, constant per block."""
    a1 = jnp.repeat(coarse[:M], G)
    a2 = jnp.repeat(coarse[M:], G)
    return a1 * ANGLE_SCALE, a2 * ANGLE_SCALE


def metrics(dt1_fit, dt2_fit, dt1_truth, dt2_truth, pos_fit, pos_truth):
    dt1_t, dt2_t = np.asarray(dt1_truth), np.asarray(dt2_truth)
    dt1_f, dt2_f = np.asarray(dt1_fit), np.asarray(dt2_fit)
    N = len(dt1_t)

    # trajectory position residual (mm)
    traj_rms = float(np.sqrt(np.mean(np.sum((np.asarray(pos_fit) -
                                             np.asarray(pos_truth)) ** 2, axis=1))))

    # cumulative deflection
    cum1_t, cum2_t = np.cumsum(dt1_t), np.cumsum(dt2_t)
    cum1_f, cum2_f = np.cumsum(dt1_f), np.cumsum(dt2_f)
    tot_t = np.sqrt(cum1_t[-1] ** 2 + cum2_t[-1] ** 2)
    tot_f = np.sqrt(cum1_f[-1] ** 2 + cum2_f[-1] ** 2)
    cum_ratio = tot_f / (tot_t + 1e-12)
    cum_corr = 0.5 * (np.corrcoef(cum1_f, cum1_t)[0, 1] +
                      np.corrcoef(cum2_f, cum2_t)[0, 1])

    # 3 mm-windowed deflection
    window = max(1, int(round(3.0 / STEP_SIZE_MM)))
    nw = N // window
    def winsum(x):
        return x[:nw * window].reshape(nw, window).sum(1)
    win_corr = 0.5 * (np.corrcoef(winsum(dt1_f), winsum(dt1_t))[0, 1] +
                      np.corrcoef(winsum(dt2_f), winsum(dt2_t))[0, 1])

    return dict(traj_rms_mm=traj_rms, cum_ratio=cum_ratio,
                cum_corr=cum_corr, win_corr=win_corr)


def run_adam(loss_and_grad, init, n_steps, lr, lam, decay=1.0,
             project=None, early_stop=True):
    """Adam with optional exp LR decay + early stop at best loss. Returns best params."""
    if decay < 1.0:
        sched = optax.exponential_decay(lr, transition_steps=1, decay_rate=decay)
        opt = optax.adam(sched)
    else:
        opt = optax.adam(lr)
    state = opt.init(init)
    params = init
    best_loss, best_params = np.inf, init
    for step in range(n_steps):
        loss, grad = loss_and_grad(params, lam)
        lv = float(loss)
        if early_stop and lv < best_loss:
            best_loss, best_params = lv, params
        upd, state = opt.update(grad, state, params)
        params = optax.apply_updates(params, upd)
        if project is not None:
            params = project(params)
    if not early_stop:
        best_loss, best_params = float(loss_and_grad(params, lam)[0]), params
    return best_params, best_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', choices=['scattering', 'full'], default='scattering')
    ap.add_argument('--steps', type=int, default=400)
    args = ap.parse_args()

    print("=" * 78)
    print(f"MCS SCATTERING STUDY — mode={args.study}, N={N_SEGMENTS}, steps={args.steps}")
    print("=" * 78, flush=True)

    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    # truth
    key = jax.random.PRNGKey(42)
    pos_truth, de_truth, dt1_truth, dt2_truth = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    de_truth_m = mask_outside_volume(pos_truth, de_truth, half_ext)
    t0 = time.time()
    truth_signals = jax.jit(forward)(pos_truth, de_truth_m)
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"truth signals compiled ({time.time()-t0:.1f}s)", flush=True)
    weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=1.5)
                    for p in range(6))

    pos_truth_np = np.asarray(pos_truth)
    cum_truth = np.sqrt(np.cumsum(dt1_truth)[-1] ** 2 + np.cumsum(dt2_truth)[-1] ** 2)
    print(f"truth cumulative deflection = {cum_truth*1000:.1f} mrad", flush=True)

    truth_start = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32)

    # sweep grid
    if args.study == 'scattering':
        G_list = [1, 5, 10, 20]
        lam_list = [1e-3, 3e-2, 0.3]
        decay_list = [1.0, 0.99]
    else:
        # full mode: focus on the informative region found in the scattering sweep
        G_list = [1, 10]
        lam_list = [1e-3, 3e-2, 0.3]
        decay_list = [1.0, 0.99]

    header = (f"{'G':>3} {'M':>5} {'lam':>7} {'decay':>6} {'wire_loss':>10} "
              f"{'E_err':>7} {'traj_mm':>8} {'cum_ratio':>9} {'cum_corr':>9} {'win_corr':>9}")
    print("\n" + header)
    print("-" * len(header), flush=True)

    results = []
    for G in G_list:
        M = N_SEGMENTS // G
        assert M * G == N_SEGMENTS

        if args.study == 'scattering':
            # params = 2M coarse angles; globals fixed at truth
            def make_loss(G=G, M=M):
                def loss(params, lam):
                    dt1, dt2 = expand_coarse(params, G, M)
                    pos, de = mcs_cumsum_forward(
                        jnp.float32(TRUTH_ENERGY), truth_start,
                        TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
                        dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
                    de = mask_outside_volume(pos, de, half_ext)
                    wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_signals, weights)
                    pr = highland_prior(dt1, dt2, jnp.float32(TRUTH_ENERGY),
                                        STEP_SIZE_CM, N_SEGMENTS, log_T, dedx)
                    return wl + lam * pr / N_SEGMENTS
                return loss
            init = jnp.zeros(2 * M)
            project = None
        else:
            # full: params = [8 globals_norm] + 2M coarse angles
            def make_loss(G=G, M=M):
                SC = jnp.array([200., 200., 200., 1., 1., 1., 1., ENERGY_SCALE])
                def loss(params, lam):
                    g = params[:8] * SC
                    dt1, dt2 = expand_coarse(params[8:], G, M)
                    pos, de = mcs_cumsum_forward(
                        g[7], jnp.array([g[0], g[1], g[2]]),
                        g[3], g[4], g[5], g[6],
                        dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
                    de = mask_outside_volume(pos, de, half_ext)
                    wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_signals, weights)
                    pr = highland_prior(dt1, dt2, g[7], STEP_SIZE_CM, N_SEGMENTS, log_T, dedx)
                    return wl + lam * pr / N_SEGMENTS
                return loss
            SC = np.array([200., 200., 200., 1., 1., 1., 1., ENERGY_SCALE])
            init_g = np.array([TRUTH_X + 100, TRUTH_Y + 100, TRUTH_Z - 100,
                               np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
                               np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
                               TRUTH_ENERGY + 100]) / SC
            init = jnp.concatenate([jnp.array(init_g, jnp.float32), jnp.zeros(2 * M)])

            def project(p):
                st, ct = p[3], p[4]
                nt = jnp.maximum(jnp.sqrt(st**2 + ct**2), 1e-8)
                sp, cp = p[5], p[6]
                npp = jnp.maximum(jnp.sqrt(sp**2 + cp**2), 1e-8)
                return p.at[3].set(st/nt).at[4].set(ct/nt).at[5].set(sp/npp).at[6].set(cp/npp)

        loss_fn = make_loss()
        lg = jax.jit(jax.value_and_grad(loss_fn))
        _ = lg(init, jnp.float32(lam_list[0]))  # compile
        jax.block_until_ready(_[0])

        for lam in lam_list:
            for decay in decay_list:
                best, bl = run_adam(lg, init, args.steps, 0.01, jnp.float32(lam),
                                    decay=decay, project=project)
                # extract fit
                if args.study == 'scattering':
                    dt1_f, dt2_f = expand_coarse(best, G, M)
                    pos_f, de_f = mcs_cumsum_forward(
                        jnp.float32(TRUTH_ENERGY), truth_start,
                        TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
                        dt1_f, dt2_f, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
                    e_err = 0.0
                else:
                    SCj = jnp.array([200., 200., 200., 1., 1., 1., 1., ENERGY_SCALE])
                    g = best[:8] * SCj
                    dt1_f, dt2_f = expand_coarse(best[8:], G, M)
                    pos_f, de_f = mcs_cumsum_forward(
                        g[7], jnp.array([g[0], g[1], g[2]]), g[3], g[4], g[5], g[6],
                        dt1_f, dt2_f, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
                    e_err = float(g[7]) - TRUTH_ENERGY
                m = metrics(dt1_f, dt2_f, dt1_truth, dt2_truth, pos_f, pos_truth_np)
                results.append((G, M, lam, decay, bl, e_err, m))
                print(f"{G:>3} {M:>5} {lam:>7.0e} {decay:>6.2f} {bl:>10.4f} "
                      f"{e_err:>7.1f} {m['traj_rms_mm']:>8.2f} {m['cum_ratio']:>9.2f} "
                      f"{m['cum_corr']:>9.3f} {m['win_corr']:>9.3f}", flush=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
