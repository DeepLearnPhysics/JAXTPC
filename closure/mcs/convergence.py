"""Save + plot the MCS joint-fit convergence history (why it takes many steps).

Runs the converged joint fit (angle_lr=0.01, seed-32 median event) and records,
per step, the loss, the global errors, and the angle correlation at several
spatial scales. Saves the history to .npz and plots it. The point is to show the
MULTI-TIMESCALE convergence: the coarse/trajectory modes converge in a few
hundred steps while the fine-angle modes crawl for thousands — the signature of
the ill-conditioned double-integral forward (dtheta -> direction -> position).

    python3 -m closure.mcs.convergence
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
    build_globals_only_loss, run_optimization, project_unit_circle,
    N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, SCALES, ANGLE_SCALE,
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
N_STEPS = 2500
ANGLE_LR, GLOBAL_LR, DECAY = 0.01, 0.003, 0.9995
LOG_EVERY = 25
SCALES_SEG = [(6, 3.0), (12, 6.0), (25, 12.5), (50, 25.0), (100, 50.0), (400, 200.0)]


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
    weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=1.5) for p in range(6))
    t1n, t2n = np.asarray(dt1_t), np.asarray(dt2_t)

    R.LAMBDA_PRIOR = 0.03
    init_g = np.array([TRUTH_X + 100, TRUTH_Y + 100, TRUTH_Z - 100,
                       np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
                       np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
                       TRUTH_ENERGY + 100])
    loss_g = build_globals_only_loss(forward, log_T, dedx, truth_signals, weights,
                                     dt1_t, dt2_t, half_ext)
    params_g, *_ = run_optimization(loss_g, jnp.array(init_g / SCALES, jnp.float32),
                                    250, 0.015, 'stage1', project_fn=project_unit_circle)

    def full_loss(p):
        g = p[:8] * SCj
        dt1 = p[8:8 + N_SEGMENTS] * ANGLE_SCALE
        dt2 = p[8 + N_SEGMENTS:] * ANGLE_SCALE
        pos, de = mcs_cumsum_forward(g[7], jnp.array([g[0], g[1], g[2]]),
                                     g[3], g[4], g[5], g[6],
                                     dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        de = mask_outside_volume(pos, de, half_ext)
        wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_signals, weights)
        pr = highland_prior(dt1, dt2, g[7], STEP_SIZE_CM, N_SEGMENTS, log_T, dedx)
        return wl + R.LAMBDA_PRIOR * pr / N_SEGMENTS

    vg = jax.jit(jax.value_and_grad(full_loss))

    def corr(p, W):
        a1 = np.asarray(p[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        a2 = np.asarray(p[8 + N_SEGMENTS:]) * ANGLE_SCALE
        def agg(x):
            n = len(x) // W
            return x[:n * W].reshape(n, W).sum(1)
        at = np.concatenate([agg(t1n), agg(t2n)]); af = np.concatenate([agg(a1), agg(a2)])
        return float(np.corrcoef(af, at)[0, 1]) if len(at) > 1 else np.nan

    opt = optax.adam(optax.exponential_decay(ANGLE_LR, 1, DECAY))
    p = jnp.concatenate([params_g, jnp.zeros(2 * N_SEGMENTS)])
    state = opt.init(p)

    loss_hist = np.zeros(N_STEPS)
    log_steps, corr_hist, dE_hist, dph_hist = [], [], [], []
    print(f"joint fit: angle_lr={ANGLE_LR}, global_lr={GLOBAL_LR}, {N_STEPS} steps", flush=True)
    for step in range(N_STEPS):
        loss, grad = vg(p)
        loss_hist[step] = float(loss)
        upd, state = opt.update(grad, state, p)
        upd = upd.at[:8].multiply(GLOBAL_LR / ANGLE_LR)
        p = optax.apply_updates(p, upd)
        p = p.at[:8].set(project_unit_circle(p[:8]))
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            log_steps.append(step)
            corr_hist.append([corr(p, W) for W, _ in SCALES_SEG])
            g = np.asarray(p[:8] * SCj)
            dE_hist.append(g[7] - TRUTH_ENERGY)
            dph_hist.append(np.degrees(np.arctan2(g[5], g[6])) - np.degrees(TRUTH_PHI))
    corr_hist = np.array(corr_hist)
    log_steps = np.array(log_steps)

    npz = os.path.join(OUT, 'mcs_convergence_history.npz')
    np.savez(npz, loss=loss_hist, log_steps=log_steps, corr=corr_hist,
             scales_mm=np.array([mm for _, mm in SCALES_SEG]),
             dE=np.array(dE_hist), dphi=np.array(dph_hist))
    print(f"saved history -> {npz}")

    # report the per-scale step-to-reach-0.5 (or 0.8) to quantify the timescales
    for j, (_, mm) in enumerate(SCALES_SEG):
        c = corr_hist[:, j]
        s50 = log_steps[np.argmax(c >= 0.5)] if (c >= 0.5).any() else None
        s80 = log_steps[np.argmax(c >= 0.8)] if (c >= 0.8).any() else None
        print(f"  scale {mm:>5.1f} mm: corr_final={c[-1]:.3f}, "
              f"steps->0.5={s50}, steps->0.8={s80}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].semilogy(np.arange(N_STEPS), loss_hist, lw=0.8, color='#3d6aa6')
    ax[0].set_xlabel('Adam step'); ax[0].set_ylabel('loss (Sobolev + prior)')
    ax[0].set_title('Loss vs step'); ax[0].grid(True, alpha=0.3)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(SCALES_SEG)))
    for j, (_, mm) in enumerate(SCALES_SEG):
        ax[1].plot(log_steps, corr_hist[:, j], color=cmap[j], lw=1.6,
                   label=f'{mm:.0f} mm')
    ax[1].axhline(0.5, color='grey', ls=':')
    ax[1].set_xlabel('Adam step'); ax[1].set_ylabel('angle corr (learned vs truth)')
    ax[1].set_title('Angle recovery by scale — coarse fast, fine slow')
    ax[1].legend(title='aggregation', fontsize=8, ncol=2); ax[1].grid(True, alpha=0.3)
    fig.suptitle('MCS joint-fit convergence (angle_lr=0.01): the fine scales are '
                 'slow because the forward double-integrates angles -> positions',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fn = os.path.join(OUT, 'mcs_convergence.png')
    fig.savefig(fn, dpi=140, bbox_inches='tight')
    print(f"saved plot -> {fn}")


if __name__ == '__main__':
    main()
