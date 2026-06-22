"""Decompose the MCS loss (wire vs Highland prior) across the stage transition,
and sweep the prior weight lambda.

Part A: log the wire term and prior term separately, per iteration, across
stage-1 (wire-only, scattering=truth) -> stage-2 (wire + lambda*prior, scattering
free from 0). Shows what actually jumps when stage 2 turns on.

Part B: sweep lambda; report the converged wire term, prior term, angle recovery
by scale, net-deflection ratio, and energy error -> what lambda to use.

    python3 -m closure.mcs.decompose_sweep
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import closure.mcs.run as R
from closure.mcs.run import (
    build_globals_only_loss, project_unit_circle,
    N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, SCALES, ANGLE_SCALE, SOBOLEV_S,
    TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY,
)
from closure.mcs.forward import (
    generate_mcs_truth, mcs_cumsum_forward, build_mcs_forward, mask_outside_volume,
    highland_prior,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p

OUT = os.path.dirname(os.path.abspath(__file__))
SCj = jnp.array(SCALES, jnp.float32)
STAGE1, STAGE2 = 250, 1000
LAMBDAS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
DECOMP_LAMBDA = 3e-2


def main():
    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    fwd_j = jax.jit(forward)

    key = jax.random.PRNGKey(32)
    pos_t, de_t, dt1_t, dt2_t = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    truth_signals = fwd_j(pos_t, mask_outside_volume(pos_t, de_t, half_ext))
    for s in truth_signals:
        jax.block_until_ready(s)
    weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=SOBOLEV_S) for p in range(6))
    t1n, t2n = np.asarray(dt1_t), np.asarray(dt2_t)
    cum_truth = np.sqrt(np.sum(t1n) ** 2 + np.sum(t2n) ** 2) * 1000

    # ---- stage 1 (globals-only, wire-only loss, scattering=truth) ----
    R.LAMBDA_PRIOR = DECOMP_LAMBDA
    init_g = np.array([TRUTH_X + 100, TRUTH_Y + 100, TRUTH_Z - 100,
                       np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
                       np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
                       TRUTH_ENERGY + 100])
    loss_g = build_globals_only_loss(forward, log_T, dedx, truth_signals, weights,
                                     dt1_t, dt2_t, half_ext)
    vg_g = jax.jit(jax.value_and_grad(loss_g))
    opt1 = optax.adam(0.015)
    pg = jnp.array(init_g / SCALES, jnp.float32)
    st1 = opt1.init(pg)
    wire1 = []
    print("--- stage 1 (wire-only) ---", flush=True)
    for _ in range(STAGE1):
        l, gr = vg_g(pg)
        wire1.append(float(l))           # stage-1 loss IS the wire term
        u, st1 = opt1.update(gr, st1, pg)
        pg = project_unit_circle(optax.apply_updates(pg, u))

    # ---- stage 2: wire + lambda*prior, traced lambda ----
    def terms(p):
        g = p[:8] * SCj
        dt1 = p[8:8 + N_SEGMENTS] * ANGLE_SCALE
        dt2 = p[8 + N_SEGMENTS:] * ANGLE_SCALE
        pos, de = mcs_cumsum_forward(g[7], jnp.array([g[0], g[1], g[2]]),
                                     g[3], g[4], g[5], g[6],
                                     dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        de = mask_outside_volume(pos, de, half_ext)
        wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_signals, weights)
        pr = highland_prior(dt1, dt2, g[7], STEP_SIZE_CM, N_SEGMENTS, log_T, dedx) / N_SEGMENTS
        return wl, pr

    def full(p, lam):
        wl, pr = terms(p)
        return wl + lam * pr
    vg = jax.jit(jax.value_and_grad(full))
    terms_j = jax.jit(terms)

    def corr(p, W):
        a1 = np.asarray(p[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        a2 = np.asarray(p[8 + N_SEGMENTS:]) * ANGLE_SCALE
        def agg(x):
            n = len(x) // W
            return x[:n * W].reshape(n, W).sum(1)
        at = np.concatenate([agg(t1n), agg(t2n)]); af = np.concatenate([agg(a1), agg(a2)])
        return float(np.corrcoef(af, at)[0, 1]) if len(at) > 1 else np.nan

    def run_stage2(lam, log=False):
        opt = optax.adam(optax.exponential_decay(0.003, 1, 0.9995))
        p = jnp.concatenate([pg, jnp.zeros(2 * N_SEGMENTS)])
        st = opt.init(p)
        W, P = [], []
        for _ in range(STAGE2):
            l, gr = vg(p, jnp.float32(lam))
            if log:
                wl, pr = terms_j(p)
                W.append(float(wl)); P.append(float(pr))
            u, st = opt.update(gr, st, p)
            u = u.at[8:].multiply(0.01 / 0.003)
            p = p.at[:8].set(project_unit_circle(p[:8]))
            p = optax.apply_updates(p, u)
            p = p.at[:8].set(project_unit_circle(p[:8]))
        return p, W, P

    # Part A: decomposition history at lambda=DECOMP_LAMBDA
    print(f"--- stage 2 decomposition (lambda={DECOMP_LAMBDA}) ---", flush=True)
    p_dec, W2, P2 = run_stage2(DECOMP_LAMBDA, log=True)

    # Part B: sweep
    print(f"\n{'lambda':>8} {'wire':>9} {'prior_w':>9} {'total':>9} {'dE':>6} "
          f"{'cum/tru':>8} {'c@12.5':>7} {'c@50':>7} {'c@200':>7}")
    sweep = []
    for lam in LAMBDAS:
        p, _, _ = run_stage2(lam, log=False)
        wl, pr = terms_j(p)
        wl, pr = float(wl), float(pr)
        g = np.asarray(p[:8] * SCj)
        a1 = np.asarray(p[8:8 + N_SEGMENTS]); a2 = np.asarray(p[8 + N_SEGMENTS:])
        cum = np.sqrt(np.sum(a1 * ANGLE_SCALE) ** 2 + np.sum(a2 * ANGLE_SCALE) ** 2) * 1000
        row = (lam, wl, lam * pr, wl + lam * pr, g[7] - TRUTH_ENERGY,
               cum / cum_truth, corr(p, 25), corr(p, 100), corr(p, 400))
        sweep.append(row)
        print(f"{lam:>8.0e} {wl:>9.5f} {lam*pr:>9.5f} {wl+lam*pr:>9.5f} {g[7]-TRUTH_ENERGY:>+6.1f} "
              f"{cum/cum_truth:>8.2f} {corr(p,25):>7.3f} {corr(p,100):>7.3f} {corr(p,400):>7.3f}",
              flush=True)

    # ---- decomposition plot ----
    total1 = np.array(wire1)                       # stage-1: prior=0, loss=wire
    it1 = np.arange(STAGE1)
    it2 = STAGE1 + np.arange(len(W2))
    W2 = np.array(W2); P2 = np.array(P2) * DECOMP_LAMBDA   # weighted prior
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].semilogy(it1, total1, color='grey', lw=1.0, label='stage1 loss (=wire)')
    ax[0].semilogy(it2, W2, color='#3d6aa6', lw=1.0, label='wire term')
    ax[0].semilogy(it2, np.maximum(P2, 1e-9), color='#d9772a', lw=1.0,
                   label=f'prior term (x{DECOMP_LAMBDA:g})')
    ax[0].semilogy(it2, W2 + P2, color='k', lw=0.8, alpha=0.6, label='total')
    ax[0].axvline(STAGE1, color='crimson', ls='--', lw=1.5)
    ax[0].text(STAGE1, ax[0].get_ylim()[1], ' stage 2 on', color='crimson',
               va='top', fontsize=9)
    ax[0].set_xlabel('iteration'); ax[0].set_ylabel('loss term'); ax[0].grid(True, alpha=0.3)
    ax[0].set_title('Loss decomposition across the stage transition')
    ax[0].legend(fontsize=9)

    lams = np.array([max(r[0], 1e-4) for r in sweep])
    ax[1].semilogx(lams, [r[6] for r in sweep], 'o-', label='corr@12.5mm')
    ax[1].semilogx(lams, [r[7] for r in sweep], 's-', label='corr@50mm')
    ax[1].semilogx(lams, [r[8] for r in sweep], '^-', label='corr@200mm')
    ax[1].semilogx(lams, [r[5] for r in sweep], 'd--', color='grey', label='net defl ratio')
    ax[1].axhline(1.0, color='grey', ls=':', lw=0.8)
    ax[1].set_xlabel('lambda (prior weight)'); ax[1].set_ylabel('recovery')
    ax[1].set_title('Prior-weight sweep'); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=9)
    fig.suptitle(f'MCS loss decomposition + prior sweep (s={SOBOLEV_S})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fn = os.path.join(OUT, 'mcs_decompose_sweep.png')
    fig.savefig(fn, dpi=140, bbox_inches='tight')
    print(f"\nsaved {fn}")


if __name__ == '__main__':
    main()
