"""Why does the loss dip (~step 150) then rise? Decompose wire vs prior + globals.

Logs, per step, the wire-data term and the Highland-prior term separately, plus
the global errors and the angle RMS, through the dip-and-rise region. Pinpoints
whether the rise is the wire fit worsening (angles/energy transient) or just the
prior penalty growing as angles leave zero.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

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

SCj = jnp.array(SCALES, jnp.float32)


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

    def terms(p):
        """Return (wire_term, prior_term) separately."""
        g = p[:8] * SCj
        dt1 = p[8:8 + N_SEGMENTS] * ANGLE_SCALE
        dt2 = p[8 + N_SEGMENTS:] * ANGLE_SCALE
        pos, de = mcs_cumsum_forward(g[7], jnp.array([g[0], g[1], g[2]]),
                                     g[3], g[4], g[5], g[6],
                                     dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        de = mask_outside_volume(pos, de, half_ext)
        wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_signals, weights)
        pr = R.LAMBDA_PRIOR * highland_prior(dt1, dt2, g[7], STEP_SIZE_CM,
                                             N_SEGMENTS, log_T, dedx) / N_SEGMENTS
        return wl, pr

    full = lambda p: sum(terms(p))
    vg = jax.jit(jax.value_and_grad(full))
    terms_j = jax.jit(terms)

    ANGLE_LR, GLOBAL_LR, DECAY = 0.01, 0.003, 0.9995
    opt = optax.adam(optax.exponential_decay(ANGLE_LR, 1, DECAY))
    p = jnp.concatenate([params_g, jnp.zeros(2 * N_SEGMENTS)])
    state = opt.init(p)

    print(f"\n{'step':>4} {'wire':>8} {'prior':>8} {'total':>8} {'dE':>6} {'dphi':>6} "
          f"{'a_rms':>6} {'|grad_g|':>8} {'|grad_a|':>8}")
    for step in range(500):
        (loss, grad) = vg(p)
        if step % 20 == 0 or step in (5, 10, 150, 175):
            wl, pr = terms_j(p)
            g = np.asarray(p[:8] * SCj)
            a1 = np.asarray(p[8:8+N_SEGMENTS]) * ANGLE_SCALE
            a2 = np.asarray(p[8+N_SEGMENTS:]) * ANGLE_SCALE
            gg = float(jnp.linalg.norm(grad[:8]))
            ga = float(jnp.linalg.norm(grad[8:]))
            print(f"{step:>4} {float(wl):>8.5f} {float(pr):>8.5f} {float(loss):>8.5f} "
                  f"{g[7]-TRUTH_ENERGY:>+6.1f} "
                  f"{np.degrees(np.arctan2(g[5],g[6]))-np.degrees(TRUTH_PHI):>+6.2f} "
                  f"{np.sqrt(np.mean(a1**2+a2**2))*1000:>6.2f} {gg:>8.4f} {ga:>8.4f}",
                  flush=True)
        upd, state = opt.update(grad, state, p)
        upd = upd.at[:8].multiply(GLOBAL_LR / ANGLE_LR)
        p = optax.apply_updates(p, upd)
        p = p.at[:8].set(project_unit_circle(p[:8]))
    print("\nDone.")


if __name__ == '__main__':
    main()
