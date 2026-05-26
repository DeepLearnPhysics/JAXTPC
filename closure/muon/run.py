"""Run muon optimization(s) from specified initial guesses.

Each run saves its own NPZ file. Can run one or many at a time.

Usage:
    python3 closure/muon/run.py A          # single run
    python3 closure/muon/run.py A B C D    # all four
    python3 closure/muon/run.py A --steps 800 --lr 0.02
"""
import sys, os, argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight
from tools.particle_generator import (
    load_dedx_table_jax,
    generate_muon_segments_trig,
    mask_outside_volume,
    get_half_extents_mm,
    build_muon_forward,
)

# =============================================================================
# Truth and constants
# =============================================================================

N_SEGMENTS = 4000
STEP_SIZE_MM = 0.5

TRUTH_X = -200.0
TRUTH_Y = 0.0
TRUTH_Z = 100.0
TRUTH_THETA = np.pi / 4
TRUTH_PHI = np.pi / 6
TRUTH_ENERGY = 500.0

TRUTH_PHYS = np.array([
    TRUTH_X, TRUTH_Y, TRUTH_Z,
    np.sin(TRUTH_THETA), np.cos(TRUTH_THETA),
    np.sin(TRUTH_PHI), np.cos(TRUTH_PHI),
    TRUTH_ENERGY,
])

SCALES = np.array([200.0, 200.0, 200.0, 1.0, 1.0, 1.0, 1.0, 500.0])
SCALES_JAX = jnp.array(SCALES, dtype=jnp.float32)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

INIT_CONFIGS = {
    'A': {'dx': +400.0, 'dy': +300.0, 'dz': -300.0,
           'dtheta': +0.4, 'dphi': +0.4, 'dE': +100.0},
    'B': {'dx': -300.0, 'dy': -400.0, 'dz': +400.0,
           'dtheta': -0.4, 'dphi': -0.4, 'dE': -100.0},
    'C': {'dx': +500.0, 'dy': -300.0, 'dz': +200.0,
           'dtheta': +0.5, 'dphi': -0.3, 'dE': +150.0},
    'D': {'dx': -200.0, 'dy': +500.0, 'dz': -400.0,
           'dtheta': -0.3, 'dphi': +0.5, 'dE': -150.0},
}

# =============================================================================
# Helpers
# =============================================================================

def make_init_phys(cfg):
    theta_init = TRUTH_THETA + cfg['dtheta']
    phi_init = TRUTH_PHI + cfg['dphi']
    return np.array([
        TRUTH_X + cfg['dx'], TRUTH_Y + cfg['dy'], TRUTH_Z + cfg['dz'],
        np.sin(theta_init), np.cos(theta_init),
        np.sin(phi_init), np.cos(phi_init),
        TRUTH_ENERGY + cfg['dE'],
    ])


def project_unit_circle(p):
    st, ct = p[3], p[4]
    nt = jnp.maximum(jnp.sqrt(st**2 + ct**2), 1e-8)
    sp, cp = p[5], p[6]
    np_ = jnp.maximum(jnp.sqrt(sp**2 + cp**2), 1e-8)
    return p.at[3].set(st/nt).at[4].set(ct/nt).at[5].set(sp/np_).at[6].set(cp/np_)


# =============================================================================
# Setup (shared across runs)
# =============================================================================

def setup(config_path='config/cubic_wireplane_config.yaml'):
    """Compile everything, return loss_and_grad and truth data."""
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector(config_path)
    sim = DetectorSimulator(detector_config, differentiable=True,
                            n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)
    half_ext = get_half_extents_mm(detector_config)

    def fwd(phys):
        pos, de = generate_muon_segments_trig(
            phys[7], jnp.array([phys[0], phys[1], phys[2]]),
            phys[3], phys[4], phys[5], phys[6],
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        de = mask_outside_volume(pos, de, half_ext)
        sigs = forward(pos, de)
        active = de > 0
        ep_start = jax.lax.stop_gradient(pos[jnp.argmax(active)])
        ep_end = jax.lax.stop_gradient(pos[N_SEGMENTS - 1 - jnp.argmax(active[::-1])])
        return sigs, ep_start, ep_end

    print("Compiling forward...", flush=True)
    t0 = time.time()
    truth_sigs, truth_start, truth_end = jax.jit(fwd)(
        jnp.array(TRUTH_PHYS, dtype=jnp.float32))
    for s in truth_sigs:
        jax.block_until_ready(s)
    truth_start = np.array(truth_start)
    truth_end = np.array(truth_end)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    spec_w = tuple(make_sobolev_weight(*truth_sigs[p].shape, s=1.5) for p in range(6))

    def loss_fn(n):
        sigs, ep_start, ep_end = fwd(n * SCALES_JAX)
        loss = sobolev_loss_geomean_log1p(sigs, truth_sigs, spec_w)
        return loss, (ep_start, ep_end)

    print("Compiling loss + gradient...", flush=True)
    t0 = time.time()
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    dummy = jnp.array(TRUTH_PHYS / SCALES, dtype=jnp.float32)
    (_, _), g = loss_and_grad(dummy)
    jax.block_until_ready(g)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    return loss_and_grad, truth_start, truth_end


# =============================================================================
# Single optimization run
# =============================================================================

def run_one(loss_and_grad, init_phys, n_steps, lr, b1, b2):
    """Run optimization, return (phist, lhist, ep_starts, ep_ends)."""
    init_n = jnp.array(init_phys / SCALES, dtype=jnp.float32)

    (init_loss, (init_ep_s, init_ep_e)), init_grad = loss_and_grad(init_n)
    jax.block_until_ready(init_grad)
    print(f"  Initial loss: {float(init_loss):.4f}")

    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(init_n)
    params = init_n

    phist = np.empty((n_steps + 1, 8))
    lhist = np.empty(n_steps + 1)
    ep_starts = np.empty((n_steps + 1, 3))
    ep_ends = np.empty((n_steps + 1, 3))

    phist[0] = np.array(params * SCALES_JAX)
    lhist[0] = float(init_loss)
    ep_starts[0] = np.array(init_ep_s)
    ep_ends[0] = np.array(init_ep_e)

    t_start = time.time()
    for step in range(n_steps):
        (loss, (ep_s, ep_e)), grad = loss_and_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = project_unit_circle(params)

        p = np.array(params * SCALES_JAX)
        phist[step + 1] = p
        lhist[step + 1] = float(loss)
        ep_starts[step + 1] = np.array(ep_s)
        ep_ends[step + 1] = np.array(ep_e)

        if (step + 1) % 50 == 0 or step == n_steps - 1:
            th = np.degrees(np.arctan2(p[3], p[4]))
            ph = np.degrees(np.arctan2(p[5], p[6]))
            elapsed = time.time() - t_start
            print(f"    Step {step+1:4d}: loss={float(loss):.4f}, "
                  f"x={p[0]:7.1f} y={p[1]:7.1f} z={p[2]:7.1f} "
                  f"th={th:5.1f} ph={ph:5.1f} E={p[7]:6.1f}  "
                  f"({elapsed:.0f}s)", flush=True)

    total = time.time() - t_start
    print(f"  Done in {total:.0f}s ({total/n_steps:.2f}s/step)")
    return phist, lhist, ep_starts, ep_ends


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inits', nargs='+', help='Init labels (A, B, C, D)')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.99)
    parser.add_argument('--config', default='config/cubic_wireplane_config.yaml',
                        help='Detector config YAML path')
    args = parser.parse_args()

    for label in args.inits:
        if label not in INIT_CONFIGS:
            print(f"Unknown init '{label}'. Available: {list(INIT_CONFIGS.keys())}")
            sys.exit(1)

    print(f"Runs: {args.inits}, steps={args.steps}, lr={args.lr}, "
          f"b1={args.b1}, b2={args.b2}")

    loss_and_grad, truth_start, truth_end = setup(args.config)

    for label in args.inits:
        cfg = INIT_CONFIGS[label]
        init_phys = make_init_phys(cfg)
        init_theta = TRUTH_THETA + cfg['dtheta']
        init_phi = TRUTH_PHI + cfg['dphi']

        print(f"\n{'='*60}")
        print(f"Run {label}: x={init_phys[0]:.0f} y={init_phys[1]:.0f} "
              f"z={init_phys[2]:.0f} th={np.degrees(init_theta):.1f} "
              f"ph={np.degrees(init_phi):.1f} E={init_phys[7]:.0f}")

        phist, lhist, ep_starts, ep_ends = run_one(
            loss_and_grad, init_phys, args.steps, args.lr, args.b1, args.b2)

        out_path = os.path.join(OUT_DIR, f'optimization_{label}.npz')
        np.savez(out_path,
                 label=label,
                 truth_phys=TRUTH_PHYS,
                 truth_start=truth_start,
                 truth_end=truth_end,
                 init_config=cfg,
                 n_steps=args.steps,
                 lr=args.lr, b1=args.b1, b2=args.b2,
                 param_history=phist,
                 loss_history=lhist,
                 endpoints_starts=ep_starts,
                 endpoints_ends=ep_ends)
        print(f"Saved {out_path} ({os.path.getsize(out_path)/1e3:.1f} KB)")

    print("\nAll done!")


if __name__ == '__main__':
    main()
