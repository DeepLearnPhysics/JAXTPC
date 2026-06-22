"""Direct coarse-scale fit vs fine-fit-then-aggregate for MCS angles.

The ~13 mm aggregate-angle crossover is invariant to loss order and drift depth.
One untested lever remains: so far we fit 2000 free per-segment angles and then
*sum* them into windows — summing noisy per-segment estimates loses coherence.
Fitting ONE angle per window directly (M = N/G DOF) is far better conditioned
(no fine null-space to wander in).  This compares, at each scale G:

  direct  : fit 2M coarse angles, correlate vs truth aggregated at G
  aggregate: fit 2N fine angles (s=1.5 best), sum to windows of G, correlate

If `direct` beats `aggregate` at small G, we *can* resolve finer than the
fine-fit crossover by matching the parameterization to the scale.

    python3 -m closure.mcs.coarse_direct
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
    run_adam, expand_coarse,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight


def agg(x, W):
    n = len(x) // W
    return x[:n * W].reshape(n, W).sum(1)


def main():
    print("=" * 70)
    print("Direct coarse fit vs fine-then-aggregate (scattering-only)")
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
    dt1_t, dt2_t = np.asarray(dt1_t), np.asarray(dt2_t)

    def make_loss(G, M):
        def loss(params, lam_):
            dt1, dt2 = expand_coarse(params, G, M)   # constant within block
            pos, de = mcs_cumsum_forward(
                jnp.float32(TRUTH_ENERGY), start,
                TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
                dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
            de = mask_outside_volume(pos, de, half_ext)
            wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_sig, weights)
            pr = highland_prior(dt1, dt2, jnp.float32(TRUTH_ENERGY),
                                STEP_SIZE_CM, N_SEGMENTS, log_T, dedx)
            return wl + lam_ * pr / N_SEGMENTS
        return loss

    # fine baseline (G=1)
    lg_fine = jax.jit(jax.value_and_grad(make_loss(1, N_SEGMENTS)))
    best_fine, _ = run_adam(lg_fine, jnp.zeros(2 * N_SEGMENTS), 350, 0.01,
                            jnp.float32(3e-2), decay=0.99, early_stop=True)
    f1 = np.asarray(best_fine[:N_SEGMENTS]) * ANGLE_SCALE
    f2 = np.asarray(best_fine[N_SEGMENTS:]) * ANGLE_SCALE

    print(f"\n{'G':>4} {'L_mm':>6} | {'direct_corr':>11} | {'aggregate_corr':>14}")
    print("-" * 44, flush=True)
    for G in [5, 10, 25, 50, 100]:   # must divide N_SEGMENTS=2000
        M = N_SEGMENTS // G
        # direct coarse fit
        lg = jax.jit(jax.value_and_grad(make_loss(G, M)))
        best, _ = run_adam(lg, jnp.zeros(2 * M), 350, 0.01,
                           jnp.float32(3e-2), decay=0.99, early_stop=True)
        # coarse fit angle per block = sum over block of expanded constant value
        c1 = np.asarray(best[:M]) * ANGLE_SCALE * G   # block-summed deflection
        c2 = np.asarray(best[M:]) * ANGLE_SCALE * G
        at1, at2 = agg(dt1_t, G), agg(dt2_t, G)
        # match lengths
        n = min(len(c1), len(at1))
        direct = np.corrcoef(np.concatenate([c1[:n], c2[:n]]),
                             np.concatenate([at1[:n], at2[:n]]))[0, 1]
        # aggregate fine fit
        af1, af2 = agg(f1, G), agg(f2, G)
        aggc = np.corrcoef(np.concatenate([af1, af2]),
                           np.concatenate([at1, at2]))[0, 1]
        print(f"{G:>4} {G*STEP_SIZE_MM:>6.1f} | {direct:>11.3f} | {aggc:>14.3f}",
              flush=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
