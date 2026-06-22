"""3D event display of the MCS closure: truth vs reconstructed track.

Runs the staged closure (s=SOBOLEV_S=1.0), then visualizes in 3D:
  (top)    the truth and reconstructed muon tracks overlaid, two view angles
           -> confirms the globals + overall trajectory are right.
  (bottom) the TRANSVERSE deviation from the straight start->end axis, in the
           two perpendicular directions, vs arc length, truth vs reconstruction
           -> this is the scattering itself; confirms the recon follows the
           coarse bend (and shows the unresolved fine wiggle).

    python3 -m closure.mcs.render_3d
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from closure.mcs.run import (
    build_globals_only_loss, build_full_loss, run_optimization, project_unit_circle,
    N_SEGMENTS, STEP_SIZE_MM, SCALES, ANGLE_SCALE, SOBOLEV_S,
    TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY,
)
from closure.mcs.forward import (
    generate_mcs_truth, mcs_cumsum_forward, build_mcs_forward, mask_outside_volume,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import make_sobolev_weight

OUT = os.path.dirname(os.path.abspath(__file__))
SCj = jnp.array(SCALES, jnp.float32)


def perp_basis(d):
    d = d / np.linalg.norm(d)
    sign = 1.0 if d[2] >= 0 else -1.0
    a = -1.0 / (sign + d[2]); b = d[0] * d[1] * a
    e1 = np.array([1 + sign * d[0] ** 2 * a, sign * b, -sign * d[0]])
    e2 = np.array([b, sign + d[1] ** 2 * a, -d[1]])
    return e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)


def transverse(pos, p0, axis, e1, e2):
    rel = pos - p0
    along = rel @ axis
    dev = rel - np.outer(along, axis)
    return along, dev @ e1, dev @ e2


def corr(a1, a2, t1, t2, W):
    def agg(x):
        n = len(x) // W
        return x[:n * W].reshape(n, W).sum(1)
    at = np.concatenate([agg(t1), agg(t2)]); af = np.concatenate([agg(a1), agg(a2)])
    return float(np.corrcoef(af, at)[0, 1]) if len(at) > 1 else np.nan


def main():
    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    fwd_j = jax.jit(forward)

    key = jax.random.PRNGKey(32)   # median-scattering realization
    pos_t, de_t, dt1_t, dt2_t = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    truth_signals = fwd_j(pos_t, mask_outside_volume(pos_t, de_t, half_ext))
    for s in truth_signals:
        jax.block_until_ready(s)
    weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=SOBOLEV_S) for p in range(6))
    print(f"using Sobolev s={SOBOLEV_S}", flush=True)

    import closure.mcs.run as R
    R.LAMBDA_PRIOR = 1e-3   # optimal at s=1.0 (decompose_sweep.py); was 0.03
    init_g = np.array([TRUTH_X + 100, TRUTH_Y + 100, TRUTH_Z - 100,
                       np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
                       np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
                       TRUTH_ENERGY + 100])
    print("--- stage 1: globals-only ---", flush=True)
    loss_g = build_globals_only_loss(forward, log_T, dedx, truth_signals, weights,
                                     dt1_t, dt2_t, half_ext)
    params_g, *_ = run_optimization(loss_g, jnp.array(init_g / SCALES, jnp.float32),
                                    250, 0.015, 'stage1', project_fn=project_unit_circle)
    print("--- stage 2: joint, s=1.0, 1500 steps ---", flush=True)
    loss_f = build_full_loss(forward, log_T, dedx, truth_signals, weights, half_ext)
    init_full = jnp.concatenate([params_g, jnp.zeros(2 * N_SEGMENTS)])
    params, *_ = run_optimization(
        loss_f, init_full, 1500, 0.003, 'stage2',
        project_fn=lambda p: p.at[:8].set(project_unit_circle(p[:8])),
        lr_scat=0.01, decay=0.9995, early_stop=False)

    # reconstructed track positions
    g = params[:8] * SCj
    dt1 = params[8:8 + N_SEGMENTS] * ANGLE_SCALE
    dt2 = params[8 + N_SEGMENTS:] * ANGLE_SCALE
    pos_f, _ = mcs_cumsum_forward(g[7], jnp.array([g[0], g[1], g[2]]),
                                  g[3], g[4], g[5], g[6],
                                  dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
    P = np.asarray(pos_t) / 10.0     # mm -> cm
    Q = np.asarray(pos_f) / 10.0
    gp = np.asarray(g)
    a1, a2 = np.asarray(dt1), np.asarray(dt2)
    t1, t2 = np.asarray(dt1_t), np.asarray(dt2_t)
    print(f"\nrecon globals: x={gp[0]:.1f} y={gp[1]:.1f} z={gp[2]:.1f} "
          f"th={np.degrees(np.arctan2(gp[3],gp[4])):.1f} ph={np.degrees(np.arctan2(gp[5],gp[6])):.1f} "
          f"E={gp[7]:.1f}")
    cum_t = np.sqrt(np.sum(t1)**2 + np.sum(t2)**2) * 1000
    cum_a = np.sqrt(np.sum(a1)**2 + np.sum(a2)**2) * 1000
    print(f"scattering: net deflection truth={cum_t:.1f} mrad, recon={cum_a:.1f} mrad")
    print(f"angle corr @12.5mm={corr(a1,a2,t1,t2,25):.3f}, @50mm={corr(a1,a2,t1,t2,100):.3f}, "
          f"@200mm={corr(a1,a2,t1,t2,400):.3f}")

    # transverse deviation in the TRUTH start->end frame (shared reference)
    p0 = P[0]; axis = P[-1] - P[0]; axis /= np.linalg.norm(axis)
    e1, e2 = perp_basis(axis)
    sT, d1T, d2T = transverse(P, p0, axis, e1, e2)
    sQ, d1Q, d2Q = transverse(Q, p0, axis, e1, e2)
    traj_rms = np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))) * 10  # cm->mm
    print(f"3D track residual (truth vs recon) = {traj_rms:.2f} mm RMS")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(16, 9))
    for k, azim in enumerate([-60, 30]):
        ax = fig.add_subplot(2, 2, k + 1, projection='3d')
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color='#3d6aa6', lw=1.6, label='truth')
        ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], color='#d9772a', lw=1.2, ls='--', label='reco')
        ax.scatter(*P[0], color='k', s=30)
        ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)'); ax.set_zlabel('z (cm)')
        ax.view_init(elev=18, azim=azim)
        ax.set_title(f'3D track (azim={azim}°)')
        if k == 0:
            ax.legend(loc='upper left', fontsize=10)
    for j, (dT, dQ, lbl) in enumerate([(d1T, d1Q, 'transverse-1'), (d2T, d2Q, 'transverse-2')]):
        ax = fig.add_subplot(2, 2, 3 + j)
        ax.plot(sT, dT * 10, color='#3d6aa6', lw=1.2, label='truth')
        ax.plot(sQ, dQ * 10, color='#d9772a', lw=1.2, label='reco')
        ax.set_xlabel('distance along track (cm)')
        ax.set_ylabel(f'{lbl} deviation (mm)')
        ax.set_title(f'Scattering: {lbl} deviation from straight line')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f'MCS closure 3D — truth vs reconstructed (s={SOBOLEV_S}); '
                 f'net deflection {cum_t:.0f}->{cum_a:.0f} mrad, '
                 f'3D residual {traj_rms:.1f} mm', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = os.path.join(OUT, 'mcs_closure_3d.png')
    fig.savefig(fn, dpi=140, bbox_inches='tight')
    print(f"saved {fn}")


if __name__ == '__main__':
    main()
