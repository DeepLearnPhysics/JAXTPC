"""MCS closure event display: truth vs reconstructed wire signals + difference.

Runs the staged MCS closure (globals-only warm-up, then converged joint fit at
s=1.0, lambda=1e-3, angle_lr=0.01), then renders, for every wire plane, the
truth signal, the reconstructed-track signal, and their difference — the
post-reconstruction residual.

    python3 -m closure.mcs.render_closure
"""

import time
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import closure.mcs.run as R
from closure.mcs.run import (
    build_globals_only_loss, build_full_loss, run_optimization, project_unit_circle,
    N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, SCALES, ANGLE_SCALE, SOBOLEV_S,
    TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY, PLANE_NAMES,
)
from closure.mcs.forward import (
    generate_mcs_truth, mcs_cumsum_forward, build_mcs_forward, mask_outside_volume,
    highland_prior,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p

OUT = os.path.dirname(os.path.abspath(__file__))


def bbox(img, pad=25, thr=0.03):
    """Active (wire, time) window of a plane image."""
    a = np.abs(img)
    if a.max() <= 0:
        return 0, img.shape[0], 0, img.shape[1]
    m = a > thr * a.max()
    rows = np.where(m.any(1))[0]
    cols = np.where(m.any(0))[0]
    if not len(rows) or not len(cols):
        return 0, img.shape[0], 0, img.shape[1]
    return (max(rows[0] - pad, 0), min(rows[-1] + pad, img.shape[0]),
            max(cols[0] - pad, 0), min(cols[-1] + pad, img.shape[1]))


def main():
    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    fwd_j = jax.jit(forward)

    # --- truth ---
    # seed 32 is a median-scattering realization (net ~70 mrad, vs the seed-42
    # default which is a low ~12th-percentile ~32 mrad event); use it so the
    # displayed MCS curvature is representative. See check_mcs_amount.py.
    key = jax.random.PRNGKey(32)
    pos_t, de_t, dt1_t, dt2_t = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    de_tm = mask_outside_volume(pos_t, de_t, half_ext)
    t0 = time.time()
    truth_signals = fwd_j(pos_t, de_tm)
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"truth signals compiled ({time.time()-t0:.1f}s)", flush=True)
    weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=SOBOLEV_S) for p in range(6))

    # --- staged closure (tuned) ---
    R.LAMBDA_PRIOR = 1e-3   # optimal at s=1.0 (decompose_sweep.py); was 0.03
    init_globals = np.array([
        TRUTH_X + 100.0, TRUTH_Y + 100.0, TRUTH_Z - 100.0,
        np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
        np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
        TRUTH_ENERGY + 100.0])
    init_norm = jnp.array(init_globals / SCALES, jnp.float32)

    print("\n--- stage 1: globals-only ---", flush=True)
    loss_g = build_globals_only_loss(forward, log_T, dedx, truth_signals, weights,
                                     dt1_t, dt2_t, half_ext)
    params_g, *_ = run_optimization(loss_g, init_norm, 250, 0.015, 'stage1',
                                    project_fn=project_unit_circle)

    # Stage 2: CONVERGED joint full fit (globals + 2N angles together). The LR
    # scan (diagnose_fit.py) showed angle_lr=0.01 descends monotonically and the
    # angle correlation climbs cleanly, while higher LRs (0.02, 0.04) excite the
    # angle<->direction degeneracy (phi drifts, loss oscillates/diverges). So:
    # globals gentle (lr=0.003), angles at 0.01 (via lr_scat), slow decay, no
    # early-stop, ~3000 steps for the (spectral-bias-slow) angles to converge.
    print("\n--- stage 2: full JOINT fit, CONVERGED (s=1.0, angle_lr=0.01, 1500 steps) ---", flush=True)
    loss_f = build_full_loss(forward, log_T, dedx, truth_signals, weights, half_ext)
    init_full = jnp.concatenate([params_g, jnp.zeros(2 * N_SEGMENTS)])

    def proj_full(p):
        return p.at[:8].set(project_unit_circle(p[:8]))

    params, *_ = run_optimization(loss_f, init_full, 1500, 0.003, 'stage2-joint',
                                  project_fn=proj_full, lr_scat=0.01,
                                  decay=0.9995, early_stop=False)

    # --- reconstructed signals ---
    g = params[:8] * jnp.array(SCALES, jnp.float32)
    dt1 = params[8:8 + N_SEGMENTS] * ANGLE_SCALE
    dt2 = params[8 + N_SEGMENTS:] * ANGLE_SCALE
    pos_f, de_f = mcs_cumsum_forward(
        g[7], jnp.array([g[0], g[1], g[2]]), g[3], g[4], g[5], g[6],
        dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
    de_fm = mask_outside_volume(pos_f, de_f, half_ext)
    recon_signals = fwd_j(pos_f, de_fm)
    for s in recon_signals:
        jax.block_until_ready(s)

    gp = np.array(g)
    print(f"\nrecon globals: x={gp[0]:.1f} y={gp[1]:.1f} z={gp[2]:.1f} "
          f"th={np.degrees(np.arctan2(gp[3],gp[4])):.1f} "
          f"ph={np.degrees(np.arctan2(gp[5],gp[6])):.1f} E={gp[7]:.1f}", flush=True)

    # --- proof the angles are LEARNED (init at 0), not the truth angles ---
    a1 = np.asarray(dt1); a2 = np.asarray(dt2)
    t1 = np.asarray(dt1_t); t2 = np.asarray(dt2_t)
    print(f"init scattering angles were ZERO; learned RMS={np.sqrt(np.mean(a1**2+a2**2))*1000:.2f} "
          f"mrad vs truth RMS={np.sqrt(np.mean(t1**2+t2**2))*1000:.2f} mrad")
    cum_t = np.sqrt(np.sum(t1)**2 + np.sum(t2)**2)
    cum_a = np.sqrt(np.sum(a1)**2 + np.sum(a2)**2)
    print(f"net (cumulative) deflection: truth={cum_t*1000:.1f} mrad, "
          f"learned={cum_a*1000:.1f} mrad (ratio {cum_a/(cum_t+1e-12):.2f})")
    print(f"learned-vs-truth angle (NOT identical -> learned, not the truth seed):")
    for W, mm in [(1, 0.5), (25, 12.5), (100, 50.0), (400, 200.0)]:
        def agg(x):
            n = len(x) // W
            return x[:n*W].reshape(n, W).sum(1)
        at = np.concatenate([agg(t1), agg(t2)]); af = np.concatenate([agg(a1), agg(a2)])
        c = np.corrcoef(af, at)[0, 1] if len(at) > 1 else float('nan')
        print(f"   window {mm:>5.1f} mm: corr(learned, truth) = {c:+.3f}")

    T = [np.asarray(s) for s in truth_signals]
    Rc = [np.asarray(s) for s in recon_signals]
    Df = [Rc[i] - T[i] for i in range(6)]

    # --- render: 3 rows (truth/reco/diff) x 6 cols (planes) ---
    fig, axes = plt.subplots(3, 6, figsize=(22, 9.5))
    row_lbl = ['Truth', 'Reconstructed', 'Difference (reco − truth)']
    for c in range(6):
        # common crop per plane from truth+reco
        r0, r1, c0, c1 = bbox(np.abs(T[c]) + np.abs(Rc[c]))
        mt = max(float(np.abs(T[c]).max()), 1e-9)
        md = max(float(np.abs(Df[c]).max()), 1e-9)
        resid = float(np.sqrt(np.sum(Df[c] ** 2)) / (np.sqrt(np.sum(T[c] ** 2)) + 1e-12))
        panels = [(T[c], mt, 'RdBu_r'), (Rc[c], mt, 'RdBu_r'), (Df[c], md, 'PuOr_r')]
        for r, (data, vmax, cmap) in enumerate(panels):
            ax = axes[r][c]
            sub = data[r0:r1, c0:c1]
            ax.imshow(sub.T, aspect='auto', origin='lower', cmap=cmap,
                      vmin=-vmax, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(PLANE_NAMES[c], fontsize=12)
            if r == 2:
                ax.set_xlabel(f'resid {resid*100:.1f}%', fontsize=10)
            if c == 0:
                ax.set_ylabel(row_lbl[r], fontsize=12, fontweight='bold')

    fig.suptitle('MCS closure — truth vs reconstructed wire signals (rows: '
                 'truth / reconstructed / difference; columns: wire planes)\n'
                 'Time (vertical) vs Wire index (horizontal), cropped to the active region',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = os.path.join(OUT, 'mcs_closure_event_display.png')
    fig.savefig(fn, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {fn}")


if __name__ == '__main__':
    main()
