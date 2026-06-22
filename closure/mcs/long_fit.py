"""Is the wire MCS fine-angle floor optimization-limited? Run long, log corr vs step.

The oracle test showed the from-zero fit stops at wire-loss ~0.014 while the
TRUTH angles give loss 0.0 — i.e. a strictly better, perfectly-correlated
solution exists that the 350-step fit does not reach.  If the fine-scale
correlation keeps climbing as we drive the loss lower with more steps, the
floor is optimization-limited (we CAN resolve finer), not an information limit.

Logs wire-loss and aggregate-angle correlation at 3 / 12.5 / 25 mm vs step, for
a lightly-regularized (lam=1e-3, near data-only) and a moderately-regularized
(lam=3e-2) long run.

    python3 -m closure.mcs.long_fit
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

from closure.mcs.forward import (
    mcs_cumsum_forward, generate_mcs_truth, highland_prior, build_mcs_forward,
    mask_outside_volume,
)
from closure.mcs.study import (
    N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, TRUTH_X, TRUTH_Y, TRUTH_Z,
    TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY, ANGLE_SCALE,
    TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight


def wcorr(f1, f2, t1, t2, W):
    def agg(x):
        n = len(x) // W
        return x[:n * W].reshape(n, W).sum(1)
    at = np.concatenate([agg(t1), agg(t2)])
    af = np.concatenate([agg(f1), agg(f2)])
    return np.corrcoef(af, at)[0, 1] if len(at) > 1 else float('nan')


def main():
    print("=" * 72)
    print("MCS wire fine-angle: is the floor optimization-limited? (long fit)")
    print("=" * 72, flush=True)

    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    key = jax.random.PRNGKey(42)
    pos_t, de_t, dt1_t, dt2_t = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    de_tm = mask_outside_volume(pos_t, de_t, half_ext)
    t0 = time.time()
    truth_sig = jax.jit(forward)(pos_t, de_tm)
    for s in truth_sig:
        jax.block_until_ready(s)
    print(f"compiled ({time.time()-t0:.1f}s)", flush=True)
    weights = tuple(make_sobolev_weight(*truth_sig[p].shape, s=1.5) for p in range(6))
    start = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32)
    t1, t2 = np.asarray(dt1_t), np.asarray(dt2_t)

    def make_lg(lam):
        def loss(params):
            dt1 = params[:N_SEGMENTS] * ANGLE_SCALE
            dt2 = params[N_SEGMENTS:] * ANGLE_SCALE
            pos, de = mcs_cumsum_forward(
                jnp.float32(TRUTH_ENERGY), start,
                TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
                dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
            de = mask_outside_volume(pos, de, half_ext)
            wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_sig, weights)
            pr = highland_prior(dt1, dt2, jnp.float32(TRUTH_ENERGY),
                                STEP_SIZE_CM, N_SEGMENTS, log_T, dedx)
            return wl + lam * pr / N_SEGMENTS
        return jax.jit(jax.value_and_grad(loss))

    cum_t = np.sqrt(np.sum(t1) ** 2 + np.sum(t2) ** 2)
    N_STEPS = 9000
    checkpoints = [350, 1000, 2500, 4000, 6000, 8000, 8999]
    for lam in [3e-2]:
        lg = make_lg(lam)
        opt = optax.adam(optax.exponential_decay(0.02, 1, 0.99975))
        params = jnp.zeros(2 * N_SEGMENTS)
        state = opt.init(params)
        print(f"\n--- lam={lam:.0e} ---")
        print(f"{'step':>5} {'wire+pri':>9} {'c@3mm':>7} {'c@6mm':>7} {'c@12.5':>7} "
              f"{'c@25mm':>7} {'cum_ratio':>9}", flush=True)
        for step in range(N_STEPS):
            l, g = lg(params)
            upd, state = opt.update(g, state, params)
            params = optax.apply_updates(params, upd)
            if step in checkpoints:
                f1 = np.asarray(params[:N_SEGMENTS]) * ANGLE_SCALE
                f2 = np.asarray(params[N_SEGMENTS:]) * ANGLE_SCALE
                cum_f = np.sqrt(np.sum(f1) ** 2 + np.sum(f2) ** 2)
                print(f"{step:>5} {float(l):>9.5f} "
                      f"{wcorr(f1,f2,t1,t2,6):>7.3f} "
                      f"{wcorr(f1,f2,t1,t2,12):>7.3f} "
                      f"{wcorr(f1,f2,t1,t2,25):>7.3f} "
                      f"{wcorr(f1,f2,t1,t2,50):>7.3f} "
                      f"{cum_f/cum_t:>9.3f}", flush=True)

    print("\nReading: if c@12.5/c@25 keep climbing as wire+pri falls, the floor is")
    print("optimization-limited and finer scales are recoverable with more steps.")
    print("\nDone.")


if __name__ == '__main__':
    main()
