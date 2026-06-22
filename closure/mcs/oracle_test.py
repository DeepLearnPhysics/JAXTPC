"""Information limit vs optimization limit for the wire MCS fine-angle floor.

The ~13 mm aggregate-angle crossover is invariant to loss order, drift depth and
parameterization, suggesting it is an information limit of the 3-plane wire
forward rather than an optimization failure.  This settles it decisively:

  * Init the scattering angles AT TRUTH (the global-truth point).
  * Run the same closure.
  * If the fine-window correlation DEGRADES from 1.0 toward the ~0.5@13 mm floor,
    the wire signal cannot hold the fine angles -> INFORMATION limit (the loss
    has flat/degenerate directions at fine scale).
  * If it STAYS ~1.0, the from-zero fit was merely optimization-limited.

Also reports, as an even cleaner probe, the loss at truth vs the loss the
from-zero fit reaches: if from-zero reaches a LOWER loss than truth while having
worse angles, the forward is provably degenerate at fine scale.

    python3 -m closure.mcs.oracle_test
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

from closure.mcs.forward import (
    mcs_cumsum_forward, generate_mcs_truth, highland_prior, build_mcs_forward,
    mask_outside_volume,
)
from closure.mcs.study import (
    N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, TRUTH_X, TRUTH_Y, TRUTH_Z,
    TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY, ANGLE_SCALE,
    TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
    run_adam,
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
    print("=" * 70)
    print("MCS wire fine-angle: information limit vs optimization limit")
    print("=" * 70, flush=True)

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

    def loss(params, lam_):
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
        return wl + lam_ * pr / N_SEGMENTS

    lg = jax.jit(jax.value_and_grad(loss))

    # truth params (normalized) and its loss
    truth_params = jnp.concatenate([
        jnp.asarray(t1 / ANGLE_SCALE, jnp.float32),
        jnp.asarray(t2 / ANGLE_SCALE, jnp.float32)])
    truth_loss = float(lg(truth_params, jnp.float32(0.0))[0])  # wire-only at truth
    print(f"\nwire loss at TRUTH angles          = {truth_loss:.6f}")

    Wlist = [6, 25, 50, 100]   # 3, 12.5, 25, 50 mm

    # 1) from-zero fit
    z, _ = run_adam(lg, jnp.zeros(2 * N_SEGMENTS), 350, 0.01, jnp.float32(3e-2),
                    decay=0.99, early_stop=True)
    z_loss = float(lg(z, jnp.float32(0.0))[0])
    z1 = np.asarray(z[:N_SEGMENTS]) * ANGLE_SCALE
    z2 = np.asarray(z[N_SEGMENTS:]) * ANGLE_SCALE
    print(f"wire loss reached from ZERO init   = {z_loss:.6f}")

    # 2) oracle fit: init AT truth, run the same closure (no prior, pure data,
    #    to test whether the DATA term alone holds the fine angles).
    def loss_data(params, lam_):
        dt1 = params[:N_SEGMENTS] * ANGLE_SCALE
        dt2 = params[N_SEGMENTS:] * ANGLE_SCALE
        pos, de = mcs_cumsum_forward(
            jnp.float32(TRUTH_ENERGY), start,
            TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
            dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        de = mask_outside_volume(pos, de, half_ext)
        return sobolev_loss_geomean_log1p(forward(pos, de), truth_sig, weights)
    lg_data = jax.jit(jax.value_and_grad(loss_data))
    orc, _ = run_adam(lg_data, truth_params, 350, 0.01, jnp.float32(0.0),
                      decay=1.0, early_stop=False)
    o1 = np.asarray(orc[:N_SEGMENTS]) * ANGLE_SCALE
    o2 = np.asarray(orc[N_SEGMENTS:]) * ANGLE_SCALE
    orc_loss = float(lg(orc, jnp.float32(0.0))[0])
    print(f"wire loss after ORACLE drift       = {orc_loss:.6f}")

    print(f"\n{'window':>8} | {'from-zero':>9} | {'oracle(truth-init,data-only)':>28}")
    print("-" * 54)
    for W in Wlist:
        cz = wcorr(z1, z2, t1, t2, W)
        co = wcorr(o1, o2, t1, t2, W)
        print(f"{W*STEP_SIZE_MM:>6.1f}mm | {cz:>9.3f} | {co:>28.3f}", flush=True)

    print("\nReading: if the data-only oracle, STARTING at truth, drifts the fine")
    print("windows well below 1.0 (toward the from-zero floor) at NO higher wire")
    print("loss, the wire forward is degenerate at fine scale -> information limit.")
    print("\nDone.")


if __name__ == '__main__':
    main()
