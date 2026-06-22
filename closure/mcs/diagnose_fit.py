"""Systematic diagnosis of the JOINT MCS full fit (globals + 2N angles).

Studies how the loss, global errors, and angle-recovery actually evolve over
steps, scanning the scattering-angle learning rate. Globals move at a gentle
fixed rate; only the angle LR is scanned. The point is to SEE whether each
setting descends smoothly (loss moving-average trending down, angle correlation
climbing) or oscillates (LR too high), instead of guessing.

    python3 -m closure.mcs.diagnose_fit
"""
import time
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
TRUTH = dict(x=TRUTH_X, y=TRUTH_Y, z=TRUTH_Z,
             th=np.degrees(TRUTH_THETA), ph=np.degrees(TRUTH_PHI), E=TRUTH_ENERGY)


def project_full(p):
    return p.at[:8].set(project_unit_circle(p[:8]))


def main():
    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    fwd_j = jax.jit(forward)

    # truth (median-scattering seed 32)
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

    # stage 1: lock globals
    R.LAMBDA_PRIOR = 0.03
    init_globals = np.array([
        TRUTH_X + 100.0, TRUTH_Y + 100.0, TRUTH_Z - 100.0,
        np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
        np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
        TRUTH_ENERGY + 100.0])
    loss_g = build_globals_only_loss(forward, log_T, dedx, truth_signals, weights,
                                     dt1_t, dt2_t, half_ext)
    print("locking globals (stage 1)...", flush=True)
    params_g, *_ = run_optimization(loss_g, jnp.array(init_globals / SCALES, jnp.float32),
                                    250, 0.015, 'stage1', project_fn=project_unit_circle)

    # joint full loss
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

    def angle_corr(p, W):
        a1 = np.asarray(p[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        a2 = np.asarray(p[8 + N_SEGMENTS:]) * ANGLE_SCALE
        def agg(x):
            n = len(x) // W
            return x[:n * W].reshape(n, W).sum(1)
        at = np.concatenate([agg(t1n), agg(t2n)]); af = np.concatenate([agg(a1), agg(a2)])
        return np.corrcoef(af, at)[0, 1] if len(at) > 1 else float('nan')

    def gstate(p):
        g = np.asarray(p[:8] * SCj)
        a1 = np.asarray(p[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        a2 = np.asarray(p[8 + N_SEGMENTS:]) * ANGLE_SCALE
        return dict(dx=g[0]-TRUTH['x'], dE=g[7]-TRUTH['E'],
                    dth=np.degrees(np.arctan2(g[3], g[4]))-TRUTH['th'],
                    dph=np.degrees(np.arctan2(g[5], g[6]))-TRUTH['ph'],
                    arms=np.sqrt(np.mean(a1**2 + a2**2))*1000)

    GLOBAL_LR = 0.003
    DECAY = 0.9995
    N_STEPS = 1000
    print(f"\nJOINT fit diagnosis: global_lr={GLOBAL_LR}, decay={DECAY}, "
          f"truth angle RMS={np.sqrt(np.mean(t1n**2+t2n**2))*1000:.2f} mrad")

    for angle_lr in [0.01, 0.02, 0.04]:
        opt = optax.adam(optax.exponential_decay(angle_lr, 1, DECAY))
        p = jnp.concatenate([params_g, jnp.zeros(2 * N_SEGMENTS)])
        state = opt.init(p)
        losses = []
        print(f"\n=== angle_lr={angle_lr} ===")
        print(f"{'step':>5} {'loss_ma50':>9} {'Δma':>8} {'dE':>7} {'dphi':>6} "
              f"{'a_rms':>6} {'c@12mm':>7} {'c@50mm':>7}")
        prev_ma = None
        for step in range(N_STEPS):
            loss, grad = vg(p)
            losses.append(float(loss))
            upd, state = opt.update(grad, state, p)
            upd = upd.at[:8].multiply(GLOBAL_LR / angle_lr)   # globals move slower
            p = optax.apply_updates(p, upd)
            p = project_full(p)
            if step % 100 == 0 or step == N_STEPS - 1:
                ma = float(np.mean(losses[-50:]))
                dma = (ma - prev_ma) if prev_ma is not None else 0.0
                prev_ma = ma
                gs = gstate(p)
                print(f"{step:>5} {ma:>9.4f} {dma:>+8.4f} {gs['dE']:>7.1f} "
                      f"{gs['dph']:>+6.2f} {gs['arms']:>6.2f} "
                      f"{angle_corr(p, 25):>7.3f} {angle_corr(p, 100):>7.3f}", flush=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
