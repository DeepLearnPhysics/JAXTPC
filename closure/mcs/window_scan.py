"""How coarse must we aggregate before MCS scattering angles become recoverable?

The per-3 mm (wire-pitch) scattering angle is unrecoverable (corr ~0.25), but
that is the *evaluation* scale, not a statement about the physics. A net
deflection over a window of length L moves the track by ~L*theta, so the wire
position signal that constrains it grows with L. This scans the aggregate
(window-summed) scattering angle recovery as a function of window length and
locates the crossover where it becomes well-measured.

Runs ONE scattering-only closure (globals fixed at truth, best regularization),
then aggregates the per-segment fit/truth angles at many window sizes.

    python3 -m closure.mcs.window_scan
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


def window_aggregate(x, W):
    """Sum array into non-overlapping windows of W elements (drop remainder)."""
    n = len(x) // W
    return x[:n * W].reshape(n, W).sum(1)


def main():
    print("=" * 76)
    print("MCS aggregate-angle recovery vs window length (scattering-only)")
    print("=" * 76, flush=True)

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
    weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=1.5)
                    for p in range(6))

    truth_start = jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32)

    # scattering-only loss (globals fixed at truth), best regularization
    def loss(params, lam):
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
        return wl + lam * pr / N_SEGMENTS

    lg = jax.jit(jax.value_and_grad(loss))
    init = jnp.zeros(2 * N_SEGMENTS)
    print("compiling + optimizing (lam=3e-2, decay=0.99, early-stop, 300 steps)...",
          flush=True)
    best, bl = run_adam(lg, init, 300, 0.01, jnp.float32(3e-2),
                        decay=0.99, early_stop=True)
    print(f"  best wire+prior loss = {bl:.5f}", flush=True)

    dt1_f = np.asarray(best[:N_SEGMENTS]) * ANGLE_SCALE
    dt2_f = np.asarray(best[N_SEGMENTS:]) * ANGLE_SCALE
    dt1_t, dt2_t = np.asarray(dt1_truth), np.asarray(dt2_truth)

    # --- aggregate-angle recovery vs window length ---
    # windows in segments; 0.5 mm/segment
    W_list = [1, 2, 6, 12, 25, 50, 100, 200, 400]
    print(f"\n{'W_seg':>6} {'L_mm':>7} {'n_win':>6} {'corr':>7} "
          f"{'rms_t_mrad':>11} {'rms_f_mrad':>11} {'rms_f/rms_t':>11} {'resid_mrad':>11}")
    print("-" * 80)
    rows = []
    for W in W_list:
        a1t, a2t = window_aggregate(dt1_t, W), window_aggregate(dt2_t, W)
        a1f, a2f = window_aggregate(dt1_f, W), window_aggregate(dt2_f, W)
        at = np.concatenate([a1t, a2t])
        af = np.concatenate([a1f, a2f])
        nwin = len(a1t)
        corr = np.corrcoef(af, at)[0, 1] if nwin > 1 else float('nan')
        rms_t = np.sqrt(np.mean(at ** 2)) * 1000
        rms_f = np.sqrt(np.mean(af ** 2)) * 1000
        resid = np.sqrt(np.mean((af - at) ** 2)) * 1000
        rows.append((W, W * STEP_SIZE_MM, nwin, corr, rms_t, rms_f, resid))
        print(f"{W:>6} {W*STEP_SIZE_MM:>7.1f} {nwin:>6} {corr:>7.3f} "
              f"{rms_t:>11.3f} {rms_f:>11.3f} {rms_f/rms_t:>11.3f} {resid:>11.3f}",
              flush=True)

    # cumulative (whole track) deflection
    cum_t = np.sqrt(np.sum(dt1_t) ** 2 + np.sum(dt2_t) ** 2) * 1000
    cum_f = np.sqrt(np.sum(dt1_f) ** 2 + np.sum(dt2_f) ** 2) * 1000
    print(f"\nwhole-track net deflection: truth={cum_t:.1f} mrad, "
          f"fit={cum_f:.1f} mrad (ratio {cum_f/cum_t:.2f})")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        Wm = np.array([r[1] for r in rows])
        corrs = np.array([r[3] for r in rows])
        ratio = np.array([r[5] / r[4] for r in rows])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].semilogx(Wm, corrs, 'o-', color='#3d6aa6')
        ax[0].axhline(0.5, color='grey', ls=':')
        ax[0].set_xlabel('aggregation window L (mm)')
        ax[0].set_ylabel('aggregate-angle correlation')
        ax[0].set_title('Recovery vs window length')
        ax[0].grid(True, alpha=0.3)
        ax[1].semilogx(Wm, ratio, 's-', color='#8e5572')
        ax[1].axhline(1.0, color='grey', ls=':')
        ax[1].set_xlabel('aggregation window L (mm)')
        ax[1].set_ylabel('RMS(fit) / RMS(truth)')
        ax[1].set_title('Aggregate-angle magnitude bias')
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fn = os.path.join(OUT_DIR, 'mcs_aggregate_angle_recovery.png')
        fig.savefig(fn, dpi=140, bbox_inches='tight')
        print(f"saved {fn}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\nDone.")


if __name__ == '__main__':
    main()
