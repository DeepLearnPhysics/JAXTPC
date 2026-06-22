"""Is the MCS fine-scale resolution limited by charge diffusion (drift depth)?

The Sobolev-order sweep (resolve_scan.py) showed the ~13 mm aggregate-angle
crossover is NOT set by the loss smoothing (reducing s overfits and does worse).
The remaining suspect is physical: transverse/longitudinal diffusion blurs the
fine track structure before it reaches the wires, and diffusion grows with drift
distance.  If that is the limiter, a track near the anode (small drift, sharp)
should resolve scattering at a finer scale than one near the cathode.

This runs the same scattering-only closure at several truth drift depths and
reports the aggregate-angle correlation vs window length + the corr=0.5
crossover for each.

    python3 -m closure.mcs.depth_scan
"""

import time
import os
import jax
import jax.numpy as jnp
import numpy as np

from closure.mcs.forward import (
    mcs_cumsum_forward, generate_mcs_truth, highland_prior, build_mcs_forward,
    mask_outside_volume,
)
from closure.mcs.study import (
    N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, TRUTH_Y, TRUTH_Z,
    TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY, ANGLE_SCALE,
    TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
    run_adam, OUT_DIR,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight


def fit_at_depth(forward, half_ext, log_T, dedx, truth_x, weights):
    """Run scattering-only closure for a truth track whose vertex x = truth_x."""
    start = jnp.array([truth_x, TRUTH_Y, TRUTH_Z], jnp.float32)
    key = jax.random.PRNGKey(42)
    pos_t, de_t, dt1_t, dt2_t = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY), start,
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    de_tm = mask_outside_volume(pos_t, de_t, half_ext)
    truth_sig = forward(pos_t, de_tm)
    for s in truth_sig:
        jax.block_until_ready(s)

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
    best, bl = run_adam(lg, jnp.zeros(2 * N_SEGMENTS), 350, 0.01,
                        jnp.float32(3e-2), decay=0.99, early_stop=True)
    dt1_f = np.asarray(best[:N_SEGMENTS]) * ANGLE_SCALE
    dt2_f = np.asarray(best[N_SEGMENTS:]) * ANGLE_SCALE
    return np.asarray(dt1_t), np.asarray(dt2_t), dt1_f, dt2_f, bl


def window_corr(dt1_f, dt2_f, dt1_t, dt2_t, W):
    def agg(x):
        n = len(x) // W
        return x[:n * W].reshape(n, W).sum(1)
    at = np.concatenate([agg(dt1_t), agg(dt2_t)])
    af = np.concatenate([agg(dt1_f), agg(dt2_f)])
    return np.corrcoef(af, at)[0, 1] if len(at) > 1 else float('nan')


def crossover(Lmm, cs, thr=0.5):
    for i in range(len(cs) - 1):
        if cs[i] < thr <= cs[i + 1]:
            t = (thr - cs[i]) / (cs[i + 1] - cs[i])
            return float(np.exp(np.log(Lmm[i]) + t * (np.log(Lmm[i+1]) - np.log(Lmm[i]))))
    return None


def main():
    print("=" * 78)
    print("MCS fine-scale resolution vs drift depth (diffusion test)")
    print("=" * 78, flush=True)

    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward_raw = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    forward = jax.jit(forward_raw)

    # need weights sized to the signal shape; shapes are depth-independent
    # (same detector), so build once from a probe.
    probe_start = jnp.array([-100., TRUTH_Y, TRUTH_Z], jnp.float32)
    pp, pde, *_ = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY), probe_start,
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, jax.random.PRNGKey(0))
    t0 = time.time()
    ps = forward(pp, mask_outside_volume(pp, pde, half_ext))
    for s in ps:
        jax.block_until_ready(s)
    print(f"compiled ({time.time()-t0:.1f}s)", flush=True)
    weights = tuple(make_sobolev_weight(*ps[p].shape, s=1.5) for p in range(6))

    # anode is at x=0, drift_direction=-1 -> drift distance = |x|.  Max ~216 mm.
    depths_x = [-20.0, -60.0, -120.0, -200.0]
    W_report = [2, 6, 12, 25, 50, 100]
    Lmm = np.array([W * STEP_SIZE_MM for W in W_report])

    hdr = (f"{'drift_mm':>8} {'wloss':>8} | " +
           " | ".join(f"{L:>5.1f}mm" for L in Lmm) + f" | {'L@0.5':>7}")
    print("\n" + hdr)
    print("-" * len(hdr), flush=True)

    rows = []
    for x in depths_x:
        dt1_t, dt2_t, dt1_f, dt2_f, bl = fit_at_depth(
            forward, half_ext, log_T, dedx, x, weights)
        cs = [window_corr(dt1_f, dt2_f, dt1_t, dt2_t, W) for W in W_report]
        cross = crossover(Lmm, cs)
        rows.append((abs(x), cs, cross))
        print(f"{abs(x):>8.0f} {bl:>8.4f} | " +
              " | ".join(f"{c:>6.3f}" for c in cs) +
              f" | {(f'{cross:.1f}mm' if cross else '>50mm'):>7}", flush=True)

    print("\nInterpretation: if L@0.5 shrinks as drift -> 0, the fine-scale")
    print("resolution is diffusion-limited (sharper signal near the anode).")
    print("\nDone.")


if __name__ == '__main__':
    main()
