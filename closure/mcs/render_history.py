"""Plot the MCS closure optimization history: loss + parameters vs iteration.

Logs, per iteration across BOTH stages (globals-only warm-up, then joint), the
loss and every global parameter (x, y, z, theta, phi, E) plus the scattering
angle RMS. A vertical line marks where stage 2 (joint) turns on. Truth values
are dashed. s = SOBOLEV_S (=1.0).

    python3 -m closure.mcs.render_history
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import closure.mcs.run as R
from closure.mcs.run import (
    build_globals_only_loss, build_full_loss, project_unit_circle,
    N_SEGMENTS, STEP_SIZE_MM, SCALES, ANGLE_SCALE, SOBOLEV_S,
    TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY,
)
from closure.mcs.forward import (
    generate_mcs_truth, build_mcs_forward, mask_outside_volume,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import make_sobolev_weight

OUT = os.path.dirname(os.path.abspath(__file__))
SCj = jnp.array(SCALES, jnp.float32)
STAGE1_STEPS, STAGE2_STEPS = 250, 800


def globals_phys(p8):
    g = np.asarray(p8) * SCALES[:8]
    return dict(x=g[0], y=g[1], z=g[2],
                th=np.degrees(np.arctan2(g[3], g[4])),
                ph=np.degrees(np.arctan2(g[5], g[6])), E=g[7])


def run_stage(loss_fn, init, n_steps, lr, lr_scat, decay, has_angles):
    """Manual Adam loop logging per-step loss, globals, angle RMS."""
    sched = optax.exponential_decay(lr, 1, decay) if decay < 1 else lr
    opt = optax.adam(sched)
    p = init
    state = opt.init(p)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    L, G, A = [], [], []
    for _ in range(n_steps):
        loss, grad = vg(p)
        L.append(float(loss))
        G.append(globals_phys(p[:8]))
        if has_angles:
            a1 = np.asarray(p[8:8 + N_SEGMENTS]) * ANGLE_SCALE
            a2 = np.asarray(p[8 + N_SEGMENTS:]) * ANGLE_SCALE
            A.append(np.sqrt(np.mean(a1 ** 2 + a2 ** 2)) * 1000)
        else:
            A.append(np.nan)
        upd, state = opt.update(grad, state, p)
        if lr_scat is not None:
            upd = upd.at[8:].multiply(lr_scat / lr)
        p = optax.apply_updates(p, upd)
        p = p.at[:8].set(project_unit_circle(p[:8]))
    return p, L, G, A


def main():
    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    fwd_j = jax.jit(forward)

    key = jax.random.PRNGKey(32)
    pos_t, de_t, dt1_t, dt2_t = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY),
        jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, key)
    truth_signals = fwd_j(pos_t, mask_outside_volume(pos_t, de_t, half_ext))
    for s in truth_signals:
        jax.block_until_ready(s)
    weights = tuple(make_sobolev_weight(*truth_signals[p].shape, s=SOBOLEV_S) for p in range(6))
    truth_arms = float(np.sqrt(np.mean(np.asarray(dt1_t) ** 2 + np.asarray(dt2_t) ** 2)) * 1000)

    R.LAMBDA_PRIOR = 1e-3   # optimal at s=1.0 (decompose_sweep.py); was 0.03
    init_g = np.array([TRUTH_X + 100, TRUTH_Y + 100, TRUTH_Z - 100,
                       np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
                       np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
                       TRUTH_ENERGY + 100])

    print("--- stage 1: globals-only ---", flush=True)
    loss_g = build_globals_only_loss(forward, log_T, dedx, truth_signals, weights,
                                     dt1_t, dt2_t, half_ext)
    p1, L1, G1, A1 = run_stage(loss_g, jnp.array(init_g / SCALES, jnp.float32),
                               STAGE1_STEPS, 0.015, None, 1.0, has_angles=False)

    print("--- stage 2: joint ---", flush=True)
    loss_f = build_full_loss(forward, log_T, dedx, truth_signals, weights, half_ext)
    init2 = jnp.concatenate([p1, jnp.zeros(2 * N_SEGMENTS)])
    p2, L2, G2, A2 = run_stage(loss_f, init2, STAGE2_STEPS, 0.003, 0.01, 0.9995,
                               has_angles=True)

    # concatenate (continuous iteration axis); stage-2 begins at STAGE1_STEPS
    loss = np.array(L1 + L2)
    G = G1 + G2
    arms = np.array(A1 + A2)
    it = np.arange(len(loss))
    t2 = STAGE1_STEPS
    gf = globals_phys(p2[:8])
    print(f"\nfinal globals: x={gf['x']:.1f} y={gf['y']:.1f} z={gf['z']:.1f} "
          f"th={gf['th']:.1f} ph={gf['ph']:.1f} E={gf['E']:.1f}")

    truth = dict(x=TRUTH_X, y=TRUTH_Y, z=TRUTH_Z,
                 th=np.degrees(TRUTH_THETA), ph=np.degrees(TRUTH_PHI), E=TRUTH_ENERGY)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    def vline(ax):
        ax.axvline(t2, color='crimson', ls='--', lw=1.5)

    ax = axes[0, 0]
    ax.semilogy(it, loss, color='#3d6aa6', lw=0.8); vline(ax)
    ax.set_title('loss'); ax.set_xlabel('iteration'); ax.grid(True, alpha=0.3)
    ax.text(t2, ax.get_ylim()[1], ' stage 2 on', color='crimson', va='top', fontsize=9)

    panels = [('x', 'x (mm)'), ('y', 'y (mm)'), ('z', 'z (mm)'),
              ('th', 'theta (deg)'), ('ph', 'phi (deg)'), ('E', 'energy (MeV)')]
    pos = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    for (kkey, lbl), (r, c) in zip(panels, pos):
        ax = axes[r, c]
        ax.plot(it, [g[kkey] for g in G], color='#3d6aa6', lw=1.1)
        ax.axhline(truth[kkey], color='green', ls=':', lw=1.5, label='truth')
        vline(ax)
        ax.set_title(lbl); ax.set_xlabel('iteration'); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[1, 3]
    ax.plot(it, arms, color='#8e5572', lw=1.1)
    ax.axhline(truth_arms, color='green', ls=':', lw=1.5, label='truth')
    vline(ax)
    ax.set_title('scattering angle RMS (mrad)'); ax.set_xlabel('iteration')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.text(t2, ax.get_ylim()[1], ' angles freed (init 0)', color='crimson',
            va='top', fontsize=8)

    fig.suptitle(f'MCS closure optimization history (s={SOBOLEV_S}); '
                 f'red dashed = stage 2 (joint) turns on at iter {t2}; '
                 f'green dotted = truth', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = os.path.join(OUT, 'mcs_closure_history.png')
    fig.savefig(fn, dpi=140, bbox_inches='tight')
    np.savez(os.path.join(OUT, 'mcs_closure_history.npz'),
             loss=loss, arms=arms, stage2_start=t2,
             **{k: np.array([g[k] for g in G]) for k in truth})
    print(f"saved {fn}")


if __name__ == '__main__':
    main()
