"""Does lowering the Sobolev order s improve the loss landscape (conditioning)?

First verifies the eps<->screening-length math numerically, then runs the joint
MCS fit at several s values and compares: the dip-then-rise bump, the loss
descent, the angle recovery by scale, and global stability. Lower s shrinks the
weight's low/high-frequency dynamic range -> better-conditioned, so the question
is whether the degenerate-flat-basin + slow-tail pathology eases (and whether
the angle recovery improves or overfits).

    python3 -m closure.mcs.sobolev_s_scan
"""
import math
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


def verify_eps_math(max_pad=1024):
    print("=== eps / screening-length check (max_pad=%d) ===" % max_pad)
    eps = 1.0 / (math.pi ** 2 * max_pad ** 2)
    f_screen = math.sqrt(eps)                 # cycles/sample where freq^2 ~ eps
    kappa = 2 * math.pi * f_screen            # angular, rad/sample
    L_screen = 1.0 / kappa                    # samples
    print(f"  eps = 1/(pi^2 max_pad^2) = {eps:.3e}")
    print(f"  f_screen = sqrt(eps)     = {f_screen:.3e} cycles/sample")
    print(f"  screening length 1/kappa = {L_screen:.1f} samples  (expect max_pad/2 = {max_pad/2:.0f})")
    for s in [1.5, 1.0, 0.5]:
        lowf = (1.0 / eps) ** s
        nyq = 1.0 / (0.25 + eps) ** s          # weight at |f|~0.5 (Nyquist, 1D)
        print(f"  s={s}: weight(f=0)={lowf:.2e}, weight(Nyquist)={nyq:.2e}, "
              f"dynamic range={lowf/nyq:.1e}")
    print()


def main():
    verify_eps_math(1024)

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
    t1n, t2n = np.asarray(dt1_t), np.asarray(dt2_t)
    shapes = [truth_signals[p].shape for p in range(6)]

    R.LAMBDA_PRIOR = 0.03
    init_g = np.array([TRUTH_X + 100, TRUTH_Y + 100, TRUTH_Z - 100,
                       np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
                       np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
                       TRUTH_ENERGY + 100])

    def corr(p, W):
        a1 = np.asarray(p[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        a2 = np.asarray(p[8 + N_SEGMENTS:]) * ANGLE_SCALE
        def agg(x):
            n = len(x) // W
            return x[:n * W].reshape(n, W).sum(1)
        at = np.concatenate([agg(t1n), agg(t2n)]); af = np.concatenate([agg(a1), agg(a2)])
        return float(np.corrcoef(af, at)[0, 1]) if len(at) > 1 else np.nan

    N_STAGE2 = 800
    for s_order in [1.5, 1.0, 0.5]:
        weights = tuple(make_sobolev_weight(*shapes[p], s=s_order) for p in range(6))
        loss_g = build_globals_only_loss(forward, log_T, dedx, truth_signals, weights,
                                         dt1_t, dt2_t, half_ext)
        params_g, *_ = run_optimization(loss_g, jnp.array(init_g / SCALES, jnp.float32),
                                        250, 0.015, f's1(s={s_order})',
                                        project_fn=project_unit_circle)

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
        opt = optax.adam(optax.exponential_decay(0.01, 1, 0.9995))
        p = jnp.concatenate([params_g, jnp.zeros(2 * N_SEGMENTS)])
        state = opt.init(p)
        loss_hist = np.zeros(N_STAGE2)
        for step in range(N_STAGE2):
            loss, grad = vg(p)
            loss_hist[step] = float(loss)
            upd, state = opt.update(grad, state, p)
            upd = upd.at[:8].multiply(0.003 / 0.01)
            p = optax.apply_updates(p, upd)
            p = p.at[:8].set(project_unit_circle(p[:8]))

        # normalize trajectory by its own dip (scale-independent shape)
        dip_i = int(np.argmin(loss_hist[:250]))
        dip = loss_hist[dip_i]
        peak = float(np.max(loss_hist[dip_i:min(dip_i + 400, N_STAGE2)]))
        g = np.asarray(p[:8] * SCj)
        print(f"\n=== s={s_order} ===")
        print(f"  dip loss={dip:.5f} @step {dip_i};  post-dip peak={peak:.5f}  "
              f"(bump x{peak/dip:.1f});  final={loss_hist[-1]:.5f}")
        print(f"  globals: dE={g[7]-TRUTH_ENERGY:+.1f} MeV, "
              f"dphi={np.degrees(np.arctan2(g[5],g[6]))-np.degrees(TRUTH_PHI):+.2f} deg")
        print(f"  angle corr: @12.5mm={corr(p,25):.3f}, @50mm={corr(p,100):.3f}, "
              f"@200mm={corr(p,400):.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
