"""Convergence speed + reconstruction quality across a few MCS events.

Runs the staged closure (s=1.0, lambda=1e-3, angle_lr=0.01) on several scattering
realizations (seeds) and reports, per event: the scattering amount, the global
errors (vertex / direction / energy), the scattering recovery (corr@50mm,
net-deflection ratio), the 3D track residual, the stage-2 steps-to-plateau, and
the wall time. Truth signals are passed as a traced argument so the JIT compiles
once and is reused for every event.

    python3 -m closure.mcs.multi_event
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

from closure.mcs.run import (
    project_unit_circle, N_SEGMENTS, STEP_SIZE_MM, STEP_SIZE_CM, SCALES, ANGLE_SCALE,
    SOBOLEV_S, TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY,
)
from closure.mcs.forward import (
    generate_mcs_truth, mcs_cumsum_forward, build_mcs_forward, mask_outside_volume,
    highland_prior,
)
from tools.particle_generator import load_dedx_table_jax, get_half_extents_mm
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p

SCj = jnp.array(SCALES, jnp.float32)
SEEDS = [0, 7, 13, 32, 42]
LAMBDA = 1e-3
STAGE1, STAGE2 = 250, 800
GLR, ALR, DECAY = 0.003, 0.01, 0.9995
TRUTH_DEG = dict(x=TRUTH_X, y=TRUTH_Y, z=TRUTH_Z,
                 th=np.degrees(TRUTH_THETA), ph=np.degrees(TRUTH_PHI), E=TRUTH_ENERGY)


def main():
    log_T, dedx = load_dedx_table_jax()
    det = generate_detector('config/cubic_wireplane_config.yaml')
    half_ext = get_half_extents_mm(det)
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS,
                            recombination_model='modified_box')
    forward = build_mcs_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    fwd_j = jax.jit(forward)

    # weights are shape-based (same for every event)
    probe, dprobe, *_ = generate_mcs_truth(
        jnp.float32(TRUTH_ENERGY), jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
        jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, jax.random.PRNGKey(0))
    sig0 = fwd_j(probe, mask_outside_volume(probe, dprobe, half_ext))
    weights = tuple(make_sobolev_weight(*sig0[p].shape, s=SOBOLEV_S) for p in range(6))

    def track(p_full):
        g = p_full[:8] * SCj
        dt1 = p_full[8:8 + N_SEGMENTS] * ANGLE_SCALE
        dt2 = p_full[8 + N_SEGMENTS:] * ANGLE_SCALE
        return mcs_cumsum_forward(g[7], jnp.array([g[0], g[1], g[2]]),
                                  g[3], g[4], g[5], g[6],
                                  dt1, dt2, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)

    # --- losses with truth passed as a traced arg (compile once) ---
    def loss_g(p8, truth_sigs, dt1t, dt2t):
        g = p8 * SCj
        pos, de = mcs_cumsum_forward(g[7], jnp.array([g[0], g[1], g[2]]),
                                     g[3], g[4], g[5], g[6],
                                     dt1t, dt2t, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        de = mask_outside_volume(pos, de, half_ext)
        return sobolev_loss_geomean_log1p(forward(pos, de), truth_sigs, weights)

    def loss_f(p, truth_sigs):
        pos, de = track(p)
        de = mask_outside_volume(pos, de, half_ext)
        wl = sobolev_loss_geomean_log1p(forward(pos, de), truth_sigs, weights)
        g = p[:8] * SCj
        dt1 = p[8:8 + N_SEGMENTS] * ANGLE_SCALE
        dt2 = p[8 + N_SEGMENTS:] * ANGLE_SCALE
        pr = highland_prior(dt1, dt2, g[7], STEP_SIZE_CM, N_SEGMENTS, log_T, dedx)
        return wl + LAMBDA * pr / N_SEGMENTS

    vg_g = jax.jit(jax.value_and_grad(loss_g, argnums=0))
    vg_f = jax.jit(jax.value_and_grad(loss_f, argnums=0))

    init_g = jnp.array(np.array([
        TRUTH_X + 100, TRUTH_Y + 100, TRUTH_Z - 100,
        np.sin(TRUTH_THETA + 0.5), np.cos(TRUTH_THETA + 0.5),
        np.sin(TRUTH_PHI + 0.5), np.cos(TRUTH_PHI + 0.5),
        TRUTH_ENERGY + 100]) / SCALES, jnp.float32)

    def winsum(x, W):
        n = len(x) // W
        return x[:n * W].reshape(n, W).sum(1)

    hdr = (f"{'seed':>4} {'net_mrad':>8} {'dvtx_mm':>7} {'dE_MeV':>6} {'dth':>5} "
           f"{'dph':>5} {'c@50mm':>6} {'cum/tru':>7} {'traj_mm':>7} {'plateau':>7} {'t2_s':>6}")
    print(f"\nMCS closure across {len(SEEDS)} events (s={SOBOLEV_S}, lambda={LAMBDA})")
    print(hdr); print('-' * len(hdr), flush=True)

    rows = []
    for seed in SEEDS:
        pos_t, de_t, dt1_t, dt2_t = generate_mcs_truth(
            jnp.float32(TRUTH_ENERGY), jnp.array([TRUTH_X, TRUTH_Y, TRUTH_Z], jnp.float32),
            jnp.float32(TRUTH_THETA), jnp.float32(TRUTH_PHI),
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx, jax.random.PRNGKey(seed))
        truth_sigs = fwd_j(pos_t, mask_outside_volume(pos_t, de_t, half_ext))
        t1, t2 = np.asarray(dt1_t), np.asarray(dt2_t)
        net_truth = np.sqrt(np.sum(t1) ** 2 + np.sum(t2) ** 2) * 1000

        # stage 1
        opt1 = optax.adam(0.015); p = init_g; st = opt1.init(p)
        for _ in range(STAGE1):
            _, gr = vg_g(p, truth_sigs, dt1_t, dt2_t)
            u, st = opt1.update(gr, st, p)
            p = project_unit_circle(optax.apply_updates(p, u))
        # stage 2 (timed)
        opt2 = optax.adam(optax.exponential_decay(GLR, 1, DECAY))
        pf = jnp.concatenate([p, jnp.zeros(2 * N_SEGMENTS)]); st2 = opt2.init(pf)
        losses = np.zeros(STAGE2)
        _ = vg_f(pf, truth_sigs)  # ensure compiled before timing (first event only)
        jax.block_until_ready(_[0])
        t0 = time.time()
        for k in range(STAGE2):
            l, gr = vg_f(pf, truth_sigs)
            losses[k] = float(l)
            u, st2 = opt2.update(gr, st2, pf)
            u = u.at[8:].multiply(ALR / GLR)
            pf = pf.at[:8].set(project_unit_circle(pf[:8]))
            pf = optax.apply_updates(pf, u)
            pf = pf.at[:8].set(project_unit_circle(pf[:8]))
        jax.block_until_ready(pf)
        t2_s = time.time() - t0

        # metrics
        g = np.asarray(pf[:8] * SCj)
        a1 = np.asarray(pf[8:8 + N_SEGMENTS]) * ANGLE_SCALE
        a2 = np.asarray(pf[8 + N_SEGMENTS:]) * ANGLE_SCALE
        dvtx = np.sqrt((g[0]-TRUTH_X)**2 + (g[1]-TRUTH_Y)**2 + (g[2]-TRUTH_Z)**2)
        dE = g[7] - TRUTH_ENERGY
        dth = np.degrees(np.arctan2(g[3], g[4])) - TRUTH_DEG['th']
        dph = np.degrees(np.arctan2(g[5], g[6])) - TRUTH_DEG['ph']
        c50 = np.corrcoef(np.concatenate([winsum(a1,100), winsum(a2,100)]),
                          np.concatenate([winsum(t1,100), winsum(t2,100)]))[0,1]
        cum = np.sqrt(np.sum(a1)**2 + np.sum(a2)**2) * 1000
        pos_f, _ = track(pf)
        traj = np.sqrt(np.mean(np.sum((np.asarray(pos_f) - np.asarray(pos_t))**2, axis=1)))
        final = np.mean(losses[-50:])
        plateau = int(np.argmax(losses <= 1.3 * final))  # first step within 1.3x final

        rows.append((seed, net_truth, dvtx, dE, dth, dph, c50, cum/net_truth, traj, plateau, t2_s))
        print(f"{seed:>4} {net_truth:>8.1f} {dvtx:>7.2f} {dE:>+6.1f} {dth:>+5.2f} "
              f"{dph:>+5.2f} {c50:>6.3f} {cum/net_truth:>7.2f} {traj:>7.2f} "
              f"{plateau:>7d} {t2_s:>6.0f}", flush=True)

    R = np.array([(r[2], r[3], r[6], r[7], r[8], r[9], r[10]) for r in rows])
    print("\nsummary (mean +/- std over events):")
    for j, name in enumerate(['dvtx_mm', 'dE_MeV', 'c@50mm', 'cum/tru', 'traj_mm',
                              'plateau_steps', 't2_sec']):
        print(f"  {name:>14}: {R[:,j].mean():.2f} +/- {R[:,j].std():.2f}")
    print("\nDone.")


if __name__ == '__main__':
    main()
