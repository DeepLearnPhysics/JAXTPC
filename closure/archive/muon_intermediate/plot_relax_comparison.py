"""
Compare softplus relaxation values against the hard-clamp (true) baseline.

For each relax value, produces a separate figure with:
  - Left panel: loss landscape (softplus vs hard clamp)
  - Right panel: gradient (softplus AD + softplus FD vs hard-clamp FD)

Run from project root:
    python3 closure_analysis_muon/plot_relax_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax, _get_consistent_csda, build_muon_forward,
)

# =============================================================================
# Config
# =============================================================================

N_SEGMENTS = 2800
STEP_SIZE_MM = 0.5
STEP_SIZE_CM = STEP_SIZE_MM / 10.0

TRUTH_X, TRUTH_Y, TRUTH_Z = -500.0, 0.0, 100.0
TRUTH_THETA, TRUTH_PHI = np.pi / 4, np.pi / 2
TRUTH_ENERGY = 200.0

PLANES = [0, 1, 2]
N_SWEEP = 120
ENERGY_RANGE = (100.0, 280.0)
FD_EPS = 0.5

RELAX_FACTORS = [1, 2, 5, 10]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CSDA builders
# =============================================================================

def make_csda_hard(R_cm, T_MeV):
    def csda(kinetic_energy_mev):
        indices = jnp.arange(N_SEGMENTS)
        R_initial = jnp.interp(jnp.log(kinetic_energy_mev), jnp.log(T_MeV), R_cm)
        R_at_start = R_initial - indices * STEP_SIZE_CM
        R_at_end = R_initial - (indices + 1) * STEP_SIZE_CM
        R_floor = R_cm[0]
        E_start = jnp.interp(jnp.maximum(R_at_start, R_floor), R_cm, T_MeV)
        E_end = jnp.interp(jnp.maximum(R_at_end, R_floor), R_cm, T_MeV)
        return jnp.maximum(E_start - E_end, 0.0)
    return csda


def make_csda_soft(R_cm, T_MeV, relax_cm):
    def csda(kinetic_energy_mev):
        indices = jnp.arange(N_SEGMENTS)
        R_initial = jnp.interp(jnp.log(kinetic_energy_mev), jnp.log(T_MeV), R_cm)
        R_at_start = R_initial - indices * STEP_SIZE_CM
        R_at_end = R_initial - (indices + 1) * STEP_SIZE_CM
        R_floor = R_cm[0]
        R_s = R_floor + jax.nn.softplus((R_at_start - R_floor) / relax_cm) * relax_cm
        R_e = R_floor + jax.nn.softplus((R_at_end - R_floor) / relax_cm) * relax_cm
        E_start = jnp.interp(R_s, R_cm, T_MeV)
        E_end = jnp.interp(R_e, R_cm, T_MeV)
        return jnp.maximum(E_start - E_end, 0.0)
    return csda


def make_sim_fn(csda_fn, forward_fn):
    def sim_forward(x, y, z, theta, phi, energy):
        sin_t = jnp.sin(theta); cos_t = jnp.cos(theta)
        sin_p = jnp.sin(phi);   cos_p = jnp.cos(phi)
        dir_vec = jnp.array([sin_t * cos_p, sin_t * sin_p, cos_t])
        step_vec = dir_vec * STEP_SIZE_MM
        indices = jnp.arange(N_SEGMENTS)
        positions = jnp.array([x, y, z])[None, :] + indices[:, None] * step_vec[None, :]
        de = csda_fn(energy)
        return forward_fn(positions, de)
    return sim_forward


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RELAX COMPARISON: per-relax figures, energy sweep")
    print("=" * 70)

    log_T, dedx = load_dedx_table_jax()
    R_cm, T_MeV = _get_consistent_csda(log_T, dedx)

    det = generate_detector('config/cubic_wireplane_config.yaml')
    sim = DetectorSimulator(det, differentiable=True, n_segments=N_SEGMENTS)
    forward = build_muon_forward(sim, N_SEGMENTS, STEP_SIZE_MM)

    truth_args = tuple(jnp.float32(v) for v in
                       [TRUTH_X, TRUTH_Y, TRUTH_Z, TRUTH_THETA, TRUTH_PHI, TRUTH_ENERGY])

    # --- Hard-clamp truth signals (the ground truth target) ---
    print("\nBuilding hard-clamp truth signals...", flush=True)
    csda_hard = make_csda_hard(R_cm, T_MeV)
    sim_hard = make_sim_fn(csda_hard, forward)
    truth_sigs = jax.jit(sim_hard)(*truth_args)
    for s in truth_sigs:
        jax.block_until_ready(s)
    target_sigs = {p: truth_sigs[p] for p in PLANES}

    def make_mse(sim_fn):
        def mse_loss(x, y, z, theta, phi, energy):
            sigs = sim_fn(x, y, z, theta, phi, energy)
            return sum(jnp.mean((sigs[p] - target_sigs[p]) ** 2) for p in PLANES)
        return mse_loss

    # --- Compile hard-clamp loss (FD only) ---
    print("  Compiling hard clamp (FD only)...", flush=True)
    t0 = time.time()
    mse_hard = make_mse(sim_hard)
    loss_hard_jit = jax.jit(mse_hard)
    _ = loss_hard_jit(*truth_args)
    print(f"    done ({time.time()-t0:.1f}s)", flush=True)

    # --- Energy sweep points ---
    e_vals = np.linspace(*ENERGY_RANGE, N_SWEEP)

    def _args(E):
        a = list(truth_args)
        a[5] = jnp.float32(E)
        return tuple(a)

    # --- Hard-clamp: loss only (no FD needed) ---
    print(f"\nSweeping hard clamp loss ({N_SWEEP} pts)...", flush=True)
    t0 = time.time()
    hard_losses = np.empty(N_SWEEP)
    for i, E in enumerate(e_vals):
        hard_losses[i] = float(loss_hard_jit(*_args(E)))
    print(f"  done ({time.time()-t0:.1f}s)", flush=True)

    # --- Per-relax: compile, sweep, plot ---
    for rf in RELAX_FACTORS:
        relax_cm = STEP_SIZE_CM * rf
        label = f'{rf}x step ({relax_cm*10:.1f} mm)'

        print(f"\n--- relax = {rf}x step ---", flush=True)
        print(f"  Compiling...", flush=True)
        t0 = time.time()
        csda_soft = make_csda_soft(R_cm, T_MeV, relax_cm)
        sim_soft = make_sim_fn(csda_soft, forward)
        mse_soft = make_mse(sim_soft)
        loss_soft_jit = jax.jit(mse_soft)
        loss_soft_vg = jax.jit(jax.value_and_grad(mse_soft, argnums=tuple(range(6))))
        _ = loss_soft_jit(*truth_args)
        _ = loss_soft_vg(*truth_args)
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)

        print(f"  Sweeping ({N_SWEEP} pts)...", flush=True)
        t0 = time.time()
        soft_losses = np.empty(N_SWEEP)
        soft_ad = np.empty(N_SWEEP)
        soft_fd = np.empty(N_SWEEP)
        for i, E in enumerate(e_vals):
            loss, grads = loss_soft_vg(*_args(E))
            soft_losses[i] = float(loss)
            soft_ad[i] = float(grads[5])
            soft_fd[i] = (float(loss_soft_jit(*_args(E + FD_EPS)))
                           - float(loss_soft_jit(*_args(E - FD_EPS)))) / (2 * FD_EPS)
        print(f"    done ({time.time()-t0:.1f}s)", flush=True)

        # --- Figure: loss (left) + gradient (right) ---
        fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(16, 6))

        # Loss panel
        ax_loss.plot(e_vals, hard_losses, 'k-', lw=2, label='Hard clamp (true)')
        ax_loss.plot(e_vals, soft_losses, 'r-', lw=1.5, alpha=0.9,
                     label=f'Softplus {label}')
        ax_loss.axvline(TRUTH_ENERGY, color='green', ls='--', lw=1.5, alpha=0.7)
        ax_loss.set_xlabel('Energy (MeV)', fontsize=13)
        ax_loss.set_ylabel('MSE Loss', fontsize=13)
        ax_loss.set_title('Loss Landscape', fontsize=14)
        ax_loss.legend(fontsize=11)
        ax_loss.grid(True, alpha=0.3)

        # Gradient panel
        ax_grad.plot(e_vals, soft_ad, 'r-', lw=1.5, alpha=0.9,
                     label=f'Softplus AD')
        ax_grad.plot(e_vals, soft_fd, 'b:', lw=1.5, alpha=0.7,
                     label=f'Softplus FD')
        ax_grad.axhline(0, color='k', ls='-', lw=0.5)
        ax_grad.axvline(TRUTH_ENERGY, color='green', ls='--', lw=1.5, alpha=0.7)
        ax_grad.set_xlabel('Energy (MeV)', fontsize=13)
        ax_grad.set_ylabel('dLoss/dEnergy', fontsize=13)
        ax_grad.set_title('Gradient', fontsize=14)
        ax_grad.legend(fontsize=11)
        ax_grad.grid(True, alpha=0.3)

        fig.suptitle(
            f'Softplus relax = {label}\n'
            f'target = hard-clamp truth, MSE loss, {N_SEGMENTS} seg, step={STEP_SIZE_MM}mm',
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        fname = os.path.join(OUT_DIR, f'relax_energy_{rf}x.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}", flush=True)

    print("\nDone!")


if __name__ == '__main__':
    main()
