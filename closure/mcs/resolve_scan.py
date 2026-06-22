"""Can a less-smoothing loss resolve MCS scattering at a finer scale?

The Sobolev weight 1/(k^2+eps)^s suppresses high spatial frequencies as
~k^{-2s}.  At s=1.5 the loss is dominated by the coarse trajectory and barely
sees the fine kinks that carry small-scale scattering information — that is what
caps the aggregate-angle crossover at ~13 mm (window_scan.py).

In scattering-only mode the globals are fixed at truth, so there is no coarse
misalignment to protect: we can use a smaller s (retaining high-frequency
content) to sharpen the fine-scale angle recovery.  This sweeps s and reports
the aggregate-angle correlation as a function of window length for each, so we
can see the crossover move.

    python3 -m closure.mcs.resolve_scan
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
    N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, TRUTH_X, TRUTH_Y, TRUTH_Z,
    TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY, ANGLE_SCALE,
    TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
    run_adam, OUT_DIR,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight


def window_corr(dt1_f, dt2_f, dt1_t, dt2_t, W):
    def agg(x):
        n = len(x) // W
        return x[:n * W].reshape(n, W).sum(1)
    at = np.concatenate([agg(dt1_t), agg(dt2_t)])
    af = np.concatenate([agg(dt1_f), agg(dt2_f)])
    if len(at) < 2:
        return float('nan'), float('nan')
    corr = np.corrcoef(af, at)[0, 1]
    rms_ratio = np.sqrt(np.mean(af ** 2)) / (np.sqrt(np.mean(at ** 2)) + 1e-12)
    return corr, rms_ratio


def main():
    print("=" * 78)
    print("MCS fine-scale resolution vs Sobolev order s (scattering-only)")
    print("=" * 78, flush=True)

    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    key = jax.random.PRNGKey(42)
    pos_truth, de_truth, dt1_truth, dt2_truth = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    de_truth_m = mask_outside_volume(pos_truth, de_truth, half_ext)
    t0 = time.time()
    truth_signals = jax.jit(forward)(pos_truth, de_truth_m)
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"truth signals compiled ({time.time()-t0:.1f}s)", flush=True)

    truth_start = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32)
    dt1_t, dt2_t = np.asarray(dt1_truth), np.asarray(dt2_truth)

    # windows to report (segments) -> mm
    W_report = [2, 6, 12, 25, 50, 100]
    s_list = [1.5, 1.0, 0.5, 0.25]
    # lower s sees more high-freq -> use smaller LR and a stronger prior to
    # keep the now-sharper-but-noisier gradient from injecting spurious wiggle.
    cfg = {1.5: (0.01, 3e-2), 1.0: (0.01, 3e-2),
           0.5: (0.008, 1e-1), 0.25: (0.006, 3e-1)}

    print(f"\nwindow lengths (mm): " +
          ", ".join(f"{W*STEP_SIZE_MM:.1f}" for W in W_report))
    hdr = f"{'s':>5} {'lr':>6} {'lam':>6} {'wloss':>8} | " + \
          " | ".join(f"{W*STEP_SIZE_MM:>4.1f}mm" for W in W_report)
    print("\n" + hdr)
    print("-" * len(hdr), flush=True)

    curves = {}
    for s in s_list:
        weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=s)
                        for p in range(6))
        lr, lam = cfg[s]

        def loss(params, lam_):
            dt1 = params[:N_SEGMENTS] * ANGLE_SCALE
            dt2 = params[N_SEGMENTS:] * ANGLE_SCALE
            pos, de = mcs_cumsum_forward(
                jnp.float32(TRUTH_ENERGY), truth_start,
                TRUTH_SIN_TH, TRUTH_COS_TH, TRUTH_SIN_PH, TRUTH_COS_PH,
                dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
            de = mask_outside_volume(pos, de, half_ext)
            wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_signals, weights)
            pr = highland_prior(dt1, dt2, jnp.float32(TRUTH_ENERGY),
                                STEP_SIZE_CM, N_SEGMENTS, log_T, dedx)
            return wl + lam_ * pr / N_SEGMENTS

        lg = jax.jit(jax.value_and_grad(loss))
        init = jnp.zeros(2 * N_SEGMENTS)
        best, bl = run_adam(lg, init, 400, lr, jnp.float32(lam),
                            decay=0.99, early_stop=True)
        dt1_f = np.asarray(best[:N_SEGMENTS]) * ANGLE_SCALE
        dt2_f = np.asarray(best[N_SEGMENTS:]) * ANGLE_SCALE

        cs = []
        for W in W_report:
            c, _ = window_corr(dt1_f, dt2_f, dt1_t, dt2_t, W)
            cs.append(c)
        curves[s] = cs
        print(f"{s:>5.2f} {lr:>6.3f} {lam:>6.0e} {bl:>8.4f} | " +
              " | ".join(f"{c:>6.3f}" for c in cs), flush=True)

    # crossover (corr>0.5) window per s, linear-interp in log(L)
    print("\ncrossover window L where aggregate-angle corr = 0.5:")
    Lmm = np.array([W * STEP_SIZE_MM for W in W_report])
    for s in s_list:
        cs = np.array(curves[s])
        cross = None
        for i in range(len(cs) - 1):
            if cs[i] < 0.5 <= cs[i + 1]:
                t = (0.5 - cs[i]) / (cs[i + 1] - cs[i])
                cross = np.exp(np.log(Lmm[i]) + t * (np.log(Lmm[i + 1]) - np.log(Lmm[i])))
                break
        print(f"  s={s:.2f}: L_0.5 = " +
              (f"{cross:.1f} mm" if cross else f">{Lmm[-1]:.0f} or <{Lmm[0]:.0f} mm"))

    print("\nDone.")


if __name__ == '__main__':
    main()
