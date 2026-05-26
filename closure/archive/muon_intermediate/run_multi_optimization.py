"""
Run 4 muon optimizations from different initial guesses, save histories,
and plot comparison of convergence (loss, position, theta, phi, energy).

Run from project root:
    python3 closure_analysis_muon/run_multi_optimization.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optax
import time

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.losses import sobolev_loss_geomean, make_sobolev_weight

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments_trig,
    mask_outside_volume,
    build_muon_forward,
)

# =============================================================================
# Configuration
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

N_STEPS = 1000
LR = 0.01
B1 = 0.9
B2 = 0.99

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 4 different initial guesses — moderate perturbations
INIT_CONFIGS = [
    {
        'label': 'Init A',
        'dx': +400.0, 'dy': +300.0, 'dz': -300.0,
        'dtheta': +0.4, 'dphi': +0.4, 'dE': +100.0,
    },
    {
        'label': 'Init B',
        'dx': -300.0, 'dy': -400.0, 'dz': +400.0,
        'dtheta': -0.4, 'dphi': -0.4, 'dE': -100.0,
    },
    {
        'label': 'Init C',
        'dx': +500.0, 'dy': -300.0, 'dz': +200.0,
        'dtheta': +0.5, 'dphi': -0.3, 'dE': +150.0,
    },
    {
        'label': 'Init D',
        'dx': -200.0, 'dy': +500.0, 'dz': -400.0,
        'dtheta': -0.3, 'dphi': +0.5, 'dE': -150.0,
    },
]


# =============================================================================
# Helpers
# =============================================================================

def to_physical(norm_params):
    return norm_params * SCALES_JAX


def project_unit_circle(norm_params):
    st, ct = norm_params[3], norm_params[4]
    norm_t = jnp.maximum(jnp.sqrt(st**2 + ct**2), 1e-8)
    sp, cp = norm_params[5], norm_params[6]
    norm_p = jnp.maximum(jnp.sqrt(sp**2 + cp**2), 1e-8)
    return norm_params.at[3].set(st / norm_t).at[4].set(ct / norm_t) \
                       .at[5].set(sp / norm_p).at[6].set(cp / norm_p)


def _make_sim_forward(forward, log_T, dedx):
    def sim_forward(phys):
        pos, de = generate_muon_segments_trig(
            phys[7], jnp.array([phys[0], phys[1], phys[2]]),
            phys[3], phys[4], phys[5], phys[6],
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        )
        de = mask_outside_volume(pos, de)
        return forward(pos, de)
    return sim_forward


def make_init_phys(cfg):
    """Build initial physical params from perturbation config."""
    theta_init = TRUTH_THETA + cfg['dtheta']
    phi_init = TRUTH_PHI + cfg['dphi']
    return np.array([
        TRUTH_X + cfg['dx'],
        TRUTH_Y + cfg['dy'],
        TRUTH_Z + cfg['dz'],
        np.sin(theta_init),
        np.cos(theta_init),
        np.sin(phi_init),
        np.cos(phi_init),
        TRUTH_ENERGY + cfg['dE'],
    ])


def run_single_optimization(loss_and_grad, init_phys, run_idx):
    """Run one optimization, return (param_history, loss_history)."""
    init_norm = init_phys / SCALES
    init_n = jnp.array(init_norm, dtype=jnp.float32)

    init_loss, init_grad = loss_and_grad(init_n)
    jax.block_until_ready(init_grad)
    print(f"  Initial loss: {float(init_loss):.6f}")

    optimizer = optax.adam(learning_rate=LR, b1=B1, b2=B2)
    opt_state = optimizer.init(init_n)
    params_n = init_n

    param_history_phys = np.empty((N_STEPS + 1, 8))
    loss_history = np.empty(N_STEPS + 1)

    param_history_phys[0] = np.array(to_physical(params_n))
    loss_history[0] = float(init_loss)

    t_start = time.time()
    for step in range(N_STEPS):
        loss, grad = loss_and_grad(params_n)
        updates, opt_state = optimizer.update(grad, opt_state, params_n)
        params_n = optax.apply_updates(params_n, updates)
        params_n = project_unit_circle(params_n)

        p = np.array(to_physical(params_n))
        param_history_phys[step + 1] = p
        loss_history[step + 1] = float(loss)

        if (step + 1) % 50 == 0 or step == N_STEPS - 1:
            eff_theta = float(jnp.arctan2(params_n[3], params_n[4]))
            eff_phi = float(jnp.arctan2(params_n[5], params_n[6]))
            elapsed = time.time() - t_start
            print(f"    Step {step+1:3d}: loss={float(loss):.6f}, "
                  f"x={p[0]:7.1f}, y={p[1]:7.1f}, z={p[2]:7.1f}, "
                  f"th={np.degrees(eff_theta):5.1f} deg, "
                  f"ph={np.degrees(eff_phi):5.1f} deg, "
                  f"E={p[7]:6.1f} MeV  ({elapsed:.1f}s)", flush=True)

    total = time.time() - t_start
    print(f"  Done in {total:.1f}s ({total/N_STEPS:.2f}s/step)")
    return param_history_phys, loss_history


# =============================================================================
# Plotting
# =============================================================================

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_multi_convergence(all_results, all_labels):
    """5-panel comparison plot: loss, |dr|, dtheta, dphi, dE."""
    steps = np.arange(N_STEPS + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Loss ---
    ax = axes[0, 0]
    for i, (phist, lhist) in enumerate(all_results):
        ax.semilogy(steps, lhist, color=COLORS[i], lw=1.5,
                    label=all_labels[i], alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('Sobolev Geomean Loss')
    ax.set_title('Loss Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Position difference (Euclidean |dr|) ---
    ax = axes[0, 1]
    for i, (phist, _) in enumerate(all_results):
        dr = np.sqrt((phist[:, 0] - TRUTH_X)**2 +
                     (phist[:, 1] - TRUTH_Y)**2 +
                     (phist[:, 2] - TRUTH_Z)**2)
        ax.semilogy(steps, dr + 1e-3, color=COLORS[i], lw=1.5,
                    label=all_labels[i], alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('|r - r_truth| (mm)')
    ax.set_title('Position Difference')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Theta difference ---
    ax = axes[0, 2]
    for i, (phist, _) in enumerate(all_results):
        eff_theta = np.arctan2(phist[:, 3], phist[:, 4])
        dtheta = np.abs(np.degrees(eff_theta - TRUTH_THETA))
        ax.semilogy(steps, dtheta + 1e-4, color=COLORS[i], lw=1.5,
                    label=all_labels[i], alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('|theta - truth| (deg)')
    ax.set_title('Theta Difference')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Phi difference ---
    ax = axes[1, 0]
    for i, (phist, _) in enumerate(all_results):
        eff_phi = np.arctan2(phist[:, 5], phist[:, 6])
        dphi = np.abs(np.degrees(eff_phi - TRUTH_PHI))
        ax.semilogy(steps, dphi + 1e-4, color=COLORS[i], lw=1.5,
                    label=all_labels[i], alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('|phi - truth| (deg)')
    ax.set_title('Phi Difference')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Energy difference ---
    ax = axes[1, 1]
    for i, (phist, _) in enumerate(all_results):
        dE = np.abs(phist[:, 7] - TRUTH_ENERGY)
        ax.semilogy(steps, dE + 1e-3, color=COLORS[i], lw=1.5,
                    label=all_labels[i], alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('|E - E_truth| (MeV)')
    ax.set_title('Energy Difference')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Final residuals table ---
    ax = axes[1, 2]
    ax.axis('off')
    headers = ['Run', '|dr| mm', 'dth deg', 'dph deg', 'dE MeV', 'Loss']
    rows = []
    for i, (phist, lhist) in enumerate(all_results):
        fp = phist[-1]
        dr = np.sqrt((fp[0]-TRUTH_X)**2 + (fp[1]-TRUTH_Y)**2 + (fp[2]-TRUTH_Z)**2)
        dth = np.abs(np.degrees(np.arctan2(fp[3], fp[4]) - TRUTH_THETA))
        dph = np.abs(np.degrees(np.arctan2(fp[5], fp[6]) - TRUTH_PHI))
        dE = np.abs(fp[7] - TRUTH_ENERGY)
        label_short = chr(ord('A') + i)
        rows.append([label_short, f'{dr:.2f}', f'{dth:.2f}', f'{dph:.2f}',
                     f'{dE:.2f}', f'{lhist[-1]:.6f}'])
    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#d4e6f1')
    for i in range(len(rows)):
        table[i + 1, 0].set_facecolor(COLORS[i] + '33')  # light tint
    ax.set_title('Final Residuals', fontsize=12, pad=20)

    fig.suptitle(
        f'Multi-Start Muon Optimization: {N_STEPS} steps, Adam LR={LR} b1={B1} b2={B2}, '
        f'Sobolev s=1.5\n'
        f'Truth: x={TRUTH_X}, y={TRUTH_Y}, z={TRUTH_Z}, '
        f'theta={np.degrees(TRUTH_THETA):.1f} deg, '
        f'phi={np.degrees(TRUTH_PHI):.1f} deg, E={TRUTH_ENERGY} MeV',
        fontsize=11, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join(OUT_DIR, 'multi_optimization_convergence.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {fname}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-START MUON OPTIMIZATION (4 INITIAL GUESSES)")
    print("=" * 70)
    print(f"Truth: x={TRUTH_X}, y={TRUTH_Y}, z={TRUTH_Z}")
    print(f"       theta={np.degrees(TRUTH_THETA):.1f} deg, "
          f"phi={np.degrees(TRUTH_PHI):.1f} deg, E={TRUTH_ENERGY} MeV")
    print(f"N_SEGMENTS={N_SEGMENTS}, N_STEPS={N_STEPS}, LR={LR}")
    print()

    # --- Load resources (shared across all runs) ---
    log_T, dedx = load_dedx_table_jax()
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(detector_config, differentiable=True,
                            n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    sim_forward = _make_sim_forward(forward, log_T, dedx)

    # --- Generate truth signals ---
    print("Compiling forward...", flush=True)
    t0 = time.time()
    truth_signals = jax.jit(sim_forward)(
        jnp.array(TRUTH_PHYS, dtype=jnp.float32))
    for s in truth_signals:
        jax.block_until_ready(s)
    print(f"  Compiled ({time.time()-t0:.1f}s)", flush=True)

    # --- Precompute Sobolev spectral weights ---
    print("Precomputing Sobolev spectral weights (s=1.5)...", flush=True)
    spec_weights_tuple = tuple(
        make_sobolev_weight(*truth_signals[p].shape, s=1.5) for p in range(6)
    )

    # --- Loss + gradient ---
    def loss_fn(norm_params):
        sigs = sim_forward(to_physical(norm_params))
        return sobolev_loss_geomean(sigs, truth_signals, spec_weights_tuple)

    print("Compiling loss + gradient...", flush=True)
    t0 = time.time()
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    dummy_init = jnp.array(make_init_phys(INIT_CONFIGS[0]) / SCALES,
                           dtype=jnp.float32)
    _, dummy_g = loss_and_grad(dummy_init)
    jax.block_until_ready(dummy_g)
    print(f"  Compiled ({time.time()-t0:.1f}s)", flush=True)

    # --- Run 4 optimizations ---
    all_results = []
    all_labels = []
    total_t0 = time.time()

    for run_idx, cfg in enumerate(INIT_CONFIGS):
        init_phys = make_init_phys(cfg)
        init_theta = TRUTH_THETA + cfg['dtheta']
        init_phi = TRUTH_PHI + cfg['dphi']

        print(f"\n{'='*70}")
        print(f"Run {run_idx+1}/4: {cfg['label']}")
        print(f"  Init: x={init_phys[0]:.0f}, y={init_phys[1]:.0f}, "
              f"z={init_phys[2]:.0f}, "
              f"th={np.degrees(init_theta):.1f} deg, "
              f"ph={np.degrees(init_phi):.1f} deg, "
              f"E={init_phys[7]:.0f} MeV")

        phist, lhist = run_single_optimization(loss_and_grad, init_phys, run_idx)
        all_results.append((phist, lhist))
        all_labels.append(cfg['label'])

    total_time = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"All 4 optimizations complete in {total_time:.1f}s "
          f"({total_time/4:.1f}s avg)")

    # --- Compute endpoints from param history ---
    print(f"\nComputing track endpoints...", flush=True)

    @jax.jit
    def get_endpoints(phys):
        pos, de = generate_muon_segments_trig(
            phys[7], jnp.array([phys[0], phys[1], phys[2]]),
            phys[3], phys[4], phys[5], phys[6],
            STEP_SIZE_MM, N_SEGMENTS, log_T, dedx)
        de = mask_outside_volume(pos, de)
        active = de > 0
        first_idx = jnp.argmax(active)
        last_idx = N_SEGMENTS - 1 - jnp.argmax(active[::-1])
        return pos[first_idx], pos[last_idx]

    # Truth endpoints
    truth_start, truth_end = get_endpoints(
        jnp.array(TRUTH_PHYS, dtype=jnp.float32))
    truth_start = np.array(truth_start)
    truth_end = np.array(truth_end)

    endpoints_starts = []
    endpoints_ends = []
    t0 = time.time()

    for ri, (phist, _) in enumerate(all_results):
        starts = np.empty((N_STEPS + 1, 3))
        ends = np.empty((N_STEPS + 1, 3))
        for step in range(N_STEPS + 1):
            ep_s, ep_e = get_endpoints(
                jnp.array(phist[step], dtype=jnp.float32))
            jax.block_until_ready(ep_e)
            starts[step] = np.array(ep_s)
            ends[step] = np.array(ep_e)
        endpoints_starts.append(starts)
        endpoints_ends.append(ends)
        elapsed = time.time() - t0
        print(f"  Run {ri+1}/4 done ({elapsed:.0f}s)", flush=True)

    print(f"  All endpoints computed in {time.time()-t0:.0f}s")

    # --- Save ---
    out_path = os.path.join(OUT_DIR, 'multi_optimization_history.npz')
    save_dict = {
        'truth_phys': TRUTH_PHYS,
        'truth_start': truth_start,
        'truth_end': truth_end,
        'scales': SCALES,
        'n_steps': N_STEPS,
        'lr': LR,
        'b1': B1,
        'b2': B2,
    }
    for i, (phist, lhist) in enumerate(all_results):
        save_dict[f'param_history_{i}'] = phist
        save_dict[f'loss_history_{i}'] = lhist
        save_dict[f'init_phys_{i}'] = make_init_phys(INIT_CONFIGS[i])
        save_dict[f'label_{i}'] = all_labels[i]
        save_dict[f'endpoints_starts_{i}'] = endpoints_starts[i]
        save_dict[f'endpoints_ends_{i}'] = endpoints_ends[i]
    np.savez(out_path, **save_dict)
    print(f"Saved {out_path} ({os.path.getsize(out_path) / 1e3:.1f} KB)")

    # --- Plot ---
    plot_multi_convergence(all_results, all_labels)
    print("Done!")


if __name__ == '__main__':
    main()
