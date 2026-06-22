"""Coarse-to-fine Sobolev schedule: can we beat the ~6.7 mm converged crossover?

§7 of FINDINGS_MCS.md showed the wire fine-angle floor is optimization-limited:
the Sobolev 1/k^{2s} weight starves the high-frequency gradients that drive the
fine angles, so they converge very slowly.  Hypothesis: lock the coarse
trajectory at s=1.5, then SHARPEN at a smaller s (flatter spectral weight ->
stronger high-frequency gradients) once the coarse solution is in place.  Lower s
overfits *from zero* (resolve_scan.py), but starting from a converged coarse
solution it should sharpen the fine angles without global wandering (globals are
fixed at truth here anyway).

Compares single-s=1.5 (8000 steps) vs coarse-to-fine (3000 @ s=1.5, then 5000
@ s=0.75).

    python3 -m closure.mcs.coarse2fine
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


def crossover(scales_mm, corrs, thr=0.5):
    for i in range(len(corrs) - 1):
        if corrs[i] < thr <= corrs[i + 1]:
            t = (thr - corrs[i]) / (corrs[i + 1] - corrs[i])
            return float(np.exp(np.log(scales_mm[i]) +
                                t * (np.log(scales_mm[i+1]) - np.log(scales_mm[i]))))
    return None


def main():
    print("=" * 72)
    print("Coarse-to-fine Sobolev schedule for MCS fine-angle resolution")
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
    start = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32)
    t1, t2 = np.asarray(dt1_t), np.asarray(dt2_t)

    def make_lg(weights, lam):
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

    w15 = tuple(make_sobolev_weight(*truth_sig[p].shape, s=1.5) for p in range(6))
    w075 = tuple(make_sobolev_weight(*truth_sig[p].shape, s=0.75) for p in range(6))

    Wsc = [(6, 3.0), (12, 6.0), (25, 12.5), (50, 25.0), (100, 50.0)]

    def run(steps, lg, params, opt, state):
        for _ in range(steps):
            l, g = lg(params)
            upd, state = opt.update(g, state, params)
            params = optax.apply_updates(params, upd)
        return params, state, float(l)

    def report(tag, params):
        f1 = np.asarray(params[:N_SEGMENTS]) * ANGLE_SCALE
        f2 = np.asarray(params[N_SEGMENTS:]) * ANGLE_SCALE
        cs = [wcorr(f1, f2, t1, t2, W) for W, _ in Wsc]
        cross = crossover([m for _, m in Wsc], cs)
        cum = np.sqrt(np.sum(f1)**2 + np.sum(f2)**2) / \
            (np.sqrt(np.sum(t1)**2 + np.sum(t2)**2) + 1e-12)
        print(f"{tag:>22} | " + " ".join(f"{c:>6.3f}" for c in cs) +
              f" | L@0.5={cross:.1f}mm  cum={cum:.2f}" if cross else
              f"{tag:>22} | " + " ".join(f"{c:>6.3f}" for c in cs) +
              f" | L@0.5=>50mm  cum={cum:.2f}", flush=True)
        return cs

    print(f"\n{'config':>22} | " + " ".join(f"{m:>5.1f}mm" for _, m in Wsc) + " | crossover")
    print("-" * 86, flush=True)

    # Baseline for reference (from a prior run): single s=1.5 @ 8000 steps gives
    # crossover ~7.0 mm, cum 0.63.  Here we test whether sharpening beats it.

    # --- coarse-to-fine: converge stage-1 @ s=1.5, then sharpen @ s=0.75 ---
    lg15 = make_lg(w15, 3e-2)
    opt = optax.adam(optax.exponential_decay(0.02, 1, 0.99975))
    p = jnp.zeros(2 * N_SEGMENTS)
    st = opt.init(p)
    p, st, _ = run(5000, lg15, p, opt, st)
    report("  after 5000 @ s=1.5", p)   # should reproduce ~6.7 mm crossover
    lg075 = make_lg(w075, 3e-2)
    opt2 = optax.adam(optax.exponential_decay(0.008, 1, 0.9997))
    st2 = opt2.init(p)
    p, st2, _ = run(4000, lg075, p, opt2, st2)
    report("  +4000 @ s=0.75 (C2F)", p)

    print("\nDone.")


if __name__ == '__main__':
    main()
