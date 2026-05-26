"""
MCMC-inspired optimizer for N-segment closure analysis.

Based on 3DGS-MCMC (Kheradmand et al., NeurIPS 2024) adapted for TPC simulation.

Pipeline per step:
  1. Adam update on (pos, dE) with reconstruction gradients only
  2. Decoupled L1 energy drain (fixed rate, bypasses Adam — like AdamW)
  3. Position noise (coupled to LR schedule, no gating)
  4. Track gradient EMA per segment (diagnostic)
  5. Relocate dead segments via energy-proportional multinomial sampling

Key design choices vs original 3DGS-MCMC paper:
  - Decoupled L1 ensures fixed drain rate independent of Adam state
  - No noise gating: energy != convergence quality in our physics
  - No separate noise decay: coupled to cosine LR schedule
  - Donor selection: multinomial sampling proportional to energy (mirrors paper)
  - 50/50 energy split gives clone enough survival time to explore

Supports overcomplete mode (n_seg > n_truth):
  Extra segments initialized at random positions with median energy.
  L1 drains extras; noise breaks symmetry between competing segments.
  Dead segments cycle through relocation until they find uncovered positions.

Modes:
  baseline — plain Adam + cosine schedule
  noise    — + position noise (no gating)
  full     — + decoupled L1 + relocation (default)

Run from project root:
    python3 closure_analysis/sgld_closure.py N [--n_truth T] [--steps S] [--mode MODE]

Examples:
    python3 closure_analysis/sgld_closure.py 5 --steps 1500 --mode baseline
    python3 closure_analysis/sgld_closure.py 5 --steps 3000 --mode full
    python3 closure_analysis/sgld_closure.py 8 --n_truth 5 --steps 3000
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax

from tools.geometry import generate_detector
from tools.config import SegmentData
from tools.simulation import DetectorSimulator
from tools.pointcloud import signal_to_pointcloud
from ott_test.ot_losses import sliced_wasserstein_loss_jit

# =============================================================================
# Truth bank for 0.5mm segments (energies in 0.05–0.20 MeV)
# Positions unchanged from original (mm), energies scaled to 0.5mm dE/dx range
# =============================================================================

TRUTH_BANK = np.array([
    [-100.0,   50.0, 100.0, 0.10],
    [-150.0,  -30.0, 200.0, 0.10],
    [ -50.0,  -80.0,  50.0, 0.08],
    [-120.0,  100.0, 280.0, 0.12],
    [ -80.0, -120.0, 150.0, 0.06],
    [-170.0,   20.0,  30.0, 0.09],
    [ -30.0,   80.0, 250.0, 0.11],
    [-140.0,  -90.0, 180.0, 0.07],
    [ -60.0,  130.0, 320.0, 0.13],
    [-180.0,  -50.0, 120.0, 0.05],
    [ -90.0,   70.0, 350.0, 0.10],
    [-160.0, -100.0,  80.0, 0.08],
    [ -40.0,  -20.0, 270.0, 0.14],
    [-110.0,  110.0, 170.0, 0.09],
    [ -70.0, -140.0, 230.0, 0.07],
    [-130.0,   40.0, 310.0, 0.11],
])

INIT_OFFSET = np.array([30.0, -30.0, 30.0, 0.05])  # ~50% energy perturbation

# Detector volume bounds for random init of extra segments (mm)
EXTRA_INIT_BOUNDS = {
    'x': (-200.0, -20.0),    # drift direction
    'y': (-150.0, 150.0),
    'z': (20.0, 380.0),
}

# =============================================================================
# Hyperparameters
# =============================================================================

PLANES = [0, 1, 2]
K = 10000
N_PROJ = 200
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Optimizer
LR_POSITION = 0.5
LR_ENERGY_MULT = 0.01   # energy learns slower than positions
B1 = 0.95
B2 = 0.999

# L1 on energy (coupled to LR schedule — strong early, safe late)
LR_L1 = 0.00005          # MeV/step fixed rate
MIN_ENERGY = 0.001        # MeV numerical floor
DEATH_THRESH = 0.005      # MeV — segments at or near MIN_ENERGY get relocated

# Noise
NOISE_LR = 5.0           # noise ∝ lr² — strong early exploration, settles late

# Relocation — continuous (every step after warmup)
WARMUP = 500
SPLIT_RATIO = 0.5        # 50/50 energy split

# EMA
GRAD_EMA_BETA = 0.95

TOTAL_STEPS = 3000


# =============================================================================
# Helpers
# =============================================================================

def best_permutation(final, truth_params):
    """Match optimizer segments to truth via Hungarian algorithm.

    Handles overcomplete case (n_seg > n_truth): returns best n_truth matches.
    Remaining segments are unmatched extras.

    Returns
    -------
    assignment : (n_seg, 4) — matched truth for each optimizer segment (zeros for extras)
    errors : (n_seg, 4) — error for each optimizer segment (NaN for extras)
    matched_idx : list of optimizer segment indices that matched truth
    """
    from scipy.optimize import linear_sum_assignment
    n_seg = len(final)
    n_truth = len(truth_params)

    # Cost matrix: (n_seg, n_truth) — L1 distance in position+energy space
    cost = np.sum(np.abs(final[:, None, :] - truth_params[None, :, :]), axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build assignment array: matched segments get their truth, extras get zeros
    assignment = np.zeros_like(final)
    errors = np.full_like(final, np.nan)
    for r, c in zip(row_ind, col_ind):
        assignment[r] = truth_params[c]
        errors[r] = final[r] - truth_params[c]

    return assignment, errors, list(row_ind)


# =============================================================================
# Loss function (reconstruction only — no L1 in the loss)
# =============================================================================

def build_loss_fn(forward, target_clouds, key):
    """Build SW loss summed over 3 planes. No regularization terms."""
    def loss_fn(params):
        seg = SegmentData(positions_mm=params[:, :3], de=params[:, 3])
        sigs = forward(seg)
        loss = 0.0
        for p in PLANES:
            pts, w = signal_to_pointcloud(sigs[p], K)
            loss = loss + sliced_wasserstein_loss_jit(
                pts, w, target_clouds[p][0], target_clouds[p][1],
                key, n_projections=N_PROJ,
            )
        return loss, loss
    return loss_fn


# =============================================================================
# Adam state manipulation
# =============================================================================

def reset_adam_moments(opt_state, target_idx):
    """Zero out Adam moments for a specific segment index."""
    adam_state = opt_state[0]
    new_mu = adam_state.mu.at[target_idx].set(0.0)
    new_nu = adam_state.nu.at[target_idx].set(0.0)
    new_adam_state = adam_state._replace(mu=new_mu, nu=new_nu)
    return (new_adam_state,) + opt_state[1:]


# =============================================================================
# Segment relocation
# =============================================================================

def relocate_segments(params, opt_state, rng_key):
    """Relocate dead segments via energy-proportional multinomial sampling.

    Mirrors the paper's _sample_alives(): donors sampled proportional to energy,
    processed sequentially so each split halves the donor's sampling weight
    (self-regulating — prevents donor massacre).

    - 50/50 energy split per clone
    - Clone position = donor + small offset (break symmetry)
    - Clone's Adam state is reset (fresh start)

    Returns (params, opt_state, rng_key, n_relocated).
    """
    energies = np.array(params[:, 3])
    dead_mask = energies <= DEATH_THRESH
    dead_indices = np.where(dead_mask)[0]
    alive_mask = ~dead_mask
    alive_indices = np.where(alive_mask)[0]

    if len(dead_indices) == 0 or len(alive_indices) == 0:
        return params, opt_state, rng_key, 0

    # Mutable energy weights — updated after each pick (self-regulating)
    alive_energies = np.array(params[alive_indices, 3], dtype=np.float64)
    n_relocated = 0

    for d_idx in dead_indices:
        # Skip if all alive segments are now too weak to donate
        if alive_energies.sum() <= 0:
            break

        # Multinomial sampling proportional to energy
        probs = alive_energies / alive_energies.sum()
        rng_key, sample_key = jax.random.split(rng_key)
        donor_local = int(jax.random.choice(sample_key, len(alive_indices), p=jnp.array(probs)))
        donor_idx = alive_indices[donor_local]

        donor_energy = float(params[donor_idx, 3])

        # Position: donor + small Gaussian offset to break symmetry
        rng_key, offset_key = jax.random.split(rng_key)
        offset = jax.random.normal(offset_key, shape=(3,)) * 3.0  # ~3mm std
        new_pos = params[donor_idx, :3] + offset

        # Energy split: clone gets small fraction, donor mostly preserved
        clone_energy = donor_energy * (1.0 - SPLIT_RATIO)
        donor_new_energy = donor_energy * SPLIT_RATIO

        params = params.at[d_idx, :3].set(new_pos)
        params = params.at[d_idx, 3].set(clone_energy)
        params = params.at[donor_idx, 3].set(donor_new_energy)

        # Update sampling weight (self-regulating: halved after each pick)
        alive_energies[donor_local] = donor_new_energy

        # Reset clone's Adam state (fresh start)
        opt_state = reset_adam_moments(opt_state, d_idx)
        n_relocated += 1

        print(f"    Relocated seg {d_idx} (dE={energies[d_idx]:.4f}) "
              f"-> donor {donor_idx} (dE={donor_energy:.4f}->{donor_new_energy:.4f})")

    return params, opt_state, rng_key, n_relocated


# =============================================================================
# Main training
# =============================================================================

def run_sgld_closure(n_seg, n_truth=None, total_steps=TOTAL_STEPS, mode='full'):
    """Run MCMC-inspired closure test.

    Parameters
    ----------
    n_seg : int
        Number of optimizer segments.
    n_truth : int or None
        Number of truth segments. Defaults to n_seg (equal mode).
        If n_truth < n_seg, runs in overcomplete mode.
    total_steps : int
        Total optimization steps.
    mode : str
        'baseline' (Adam only), 'noise' (+ noise), 'full' (+ L1 + relocation)
    """
    if n_truth is None:
        n_truth = n_seg
    assert n_truth <= len(TRUTH_BANK), f"Only {len(TRUTH_BANK)} truth segments defined"
    assert n_seg >= n_truth, f"n_seg ({n_seg}) must be >= n_truth ({n_truth})"
    assert mode in ('baseline', 'noise', 'full'), f"Unknown mode: {mode}"

    overcomplete = n_seg > n_truth
    enable_noise = mode in ('noise', 'full')
    enable_l1 = mode == 'full'
    enable_relocation = mode == 'full'

    print(f"\n{'=' * 70}")
    oc_str = f"  [{n_seg} opt / {n_truth} truth — OVERCOMPLETE]" if overcomplete else ""
    print(f"{n_seg}-SEGMENT MCMC CLOSURE TEST  [mode={mode}]{oc_str}")
    print(f"{'=' * 70}")
    print(f"Hyperparams:")
    print(f"  lr_pos={LR_POSITION}, lr_e_mult={LR_ENERGY_MULT}, b1={B1}")
    print(f"  noise_lr={NOISE_LR}, lr_l1={LR_L1}, death_thresh={DEATH_THRESH}")
    print(f"  split_ratio={SPLIT_RATIO}, warmup={WARMUP}, reloc=continuous, steps={total_steps}")
    print(f"  K={K}, N_PROJ={N_PROJ}")
    print(f"  noise={enable_noise}, l1={enable_l1}, relocation={enable_relocation}")

    truth_params = TRUTH_BANK[:n_truth]

    # --- Initialize optimizer segments ---
    rng_init = np.random.RandomState(42)
    init_params = np.zeros((n_seg, 4))

    # First n_truth segments: near truth + offset
    init_params[:n_truth] = truth_params + INIT_OFFSET

    # Extra segments: random positions, median energy
    if overcomplete:
        median_energy = float(np.median(truth_params[:, 3]))
        for i in range(n_truth, n_seg):
            init_params[i, 0] = rng_init.uniform(*EXTRA_INIT_BOUNDS['x'])
            init_params[i, 1] = rng_init.uniform(*EXTRA_INIT_BOUNDS['y'])
            init_params[i, 2] = rng_init.uniform(*EXTRA_INIT_BOUNDS['z'])
            init_params[i, 3] = median_energy

    init_params = jnp.array(init_params)

    print(f"\nTruth segments ({n_truth}):")
    for i in range(n_truth):
        print(f"  Seg {i}: truth={truth_params[i]}  init={np.array(init_params[i])}")
    if overcomplete:
        print(f"Extra segments ({n_seg - n_truth}):")
        for i in range(n_truth, n_seg):
            print(f"  Seg {i}: init={np.array(init_params[i])}  (random)")

    # --- Build simulators ---
    print("Building simulator...")
    detector_config = generate_detector('config/cubic_wireplane_config.yaml')

    # Truth simulator (may differ from optimizer if overcomplete)
    if overcomplete:
        sim_truth = DetectorSimulator(
            detector_config, differentiable=True, n_segments=n_truth)
        fwd_truth = sim_truth.build_forward()
    else:
        sim_truth = None  # reuse optimizer sim

    sim_opt = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_seg)
    fwd_opt = sim_opt.build_forward()

    if not overcomplete:
        fwd_truth = fwd_opt

    # --- Generate target signals and pointclouds ---
    truth_seg = SegmentData(
        positions_mm=jnp.array(truth_params[:, :3]),
        de=jnp.array(truth_params[:, 3]),
    )
    target_signals = fwd_truth(truth_seg)
    key = jax.random.PRNGKey(42)

    target_clouds = {}
    for p in PLANES:
        pts, w = signal_to_pointcloud(target_signals[p], K)
        target_clouds[p] = (pts, w)

    # --- Build loss (uses optimizer's forward) ---
    loss_fn = build_loss_fn(fwd_opt, target_clouds, key)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # --- Optimizer (exponential decay: 0.3 → ~0.005 over total_steps) ---
    schedule = optax.exponential_decay(
        init_value=LR_POSITION, transition_steps=1, decay_rate=0.9995)
    optimizer = optax.adam(schedule, b1=B1, b2=B2)

    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    # Gradient EMA per segment (diagnostic)
    grad_ema = jnp.zeros(n_seg)

    # JIT warmup
    print("Warming up JIT...")
    _ = grad_fn(params)
    print("JIT warm-up done.")

    # Logging
    losses = []
    param_history = []
    dead_counts = []
    relocation_steps = []     # (step, n_relocated)
    cumulative_relocs = 0
    print_every = max(20, total_steps // 30)

    t0 = time.time()

    for step in range(total_steps):
        # --- 1. Forward + backward (reconstruction loss only) ---
        (total_loss, _), grads = grad_fn(params)

        # --- 2. Adam update with slower energy LR ---
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(LR_ENERGY_MULT)
        params = optax.apply_updates(params, updates)

        # --- 3. Energy floor (always enforced) ---
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], MIN_ENERGY))

        # --- 3b. Decoupled L1 energy drain (fixed rate) ---
        if enable_l1 and step >= WARMUP:
            params = params.at[:, 3].add(-LR_L1)
            params = params.at[:, 3].set(jnp.maximum(params[:, 3], MIN_ENERGY))

        # --- 4. Noise on positions (quadratic LR coupling: ∝ lr²) ---
        if enable_noise:
            lr_current = float(schedule(step))
            noise_scale = (lr_current ** 2 / LR_POSITION) * NOISE_LR
            rng_key, noise_key = jax.random.split(rng_key)
            noise_vec = jax.random.normal(noise_key, shape=(n_seg, 3))
            params = params.at[:, :3].add(noise_scale * noise_vec)

        # --- 5. Track gradient EMA (diagnostic) ---
        grad_pos_norm = jnp.linalg.norm(grads[:, :3], axis=-1)
        grad_ema = GRAD_EMA_BETA * grad_ema + (1 - GRAD_EMA_BETA) * grad_pos_norm

        # --- 6. Relocation — continuous check every step ---
        n_dead = int(jnp.sum(params[:, 3] <= DEATH_THRESH))
        if (enable_relocation and step >= WARMUP and n_dead > 0):
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key)
            if n_reloc > 0:
                cumulative_relocs += n_reloc
                relocation_steps.append((step, n_reloc))

        # --- 7. Log ---
        loss_val = float(total_loss)
        losses.append(loss_val)
        param_history.append(np.array(params))
        dead_counts.append(n_dead)

        if step % print_every == 0 or step == total_steps - 1:
            energies_str = ", ".join(f"{float(params[i, 3]):.4f}" for i in range(n_seg))
            print(f"  Step {step:4d}: loss={loss_val:.6f}  "
                  f"dead={n_dead}  E=[{energies_str}]")
            if step % (print_every * 3) == 0:
                ema_str_sci = ", ".join(
                    f"{float(grad_ema[i]):.2e}" for i in range(n_seg))
                print(f"             grad_ema=[{ema_str_sci}]")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s "
          f"({elapsed / total_steps * 1000:.1f} ms/step)")

    # --- Match segments to truth ---
    final = np.array(params)
    param_history = np.array(param_history)
    assignment, errors, matched_idx = best_permutation(final, truth_params)

    print(f"\nResults (matched to {n_truth} truth segments):")
    for i in matched_idx:
        e = errors[i]
        print(f"  Seg {i}: x={e[0]:+.3f} mm, y={e[1]:+.3f} mm, "
              f"z={e[2]:+.3f} mm, dE={e[3]*1000:+.1f} keV")
    matched_errors = errors[matched_idx]
    max_pos = np.max(np.abs(matched_errors[:, :3]))
    mean_pos = np.mean(np.sqrt(np.sum(matched_errors[:, :3] ** 2, axis=1)))
    max_de = np.max(np.abs(matched_errors[:, 3])) * 1000
    print(f"  Max position error:  {max_pos:.3f} mm")
    print(f"  Mean position error: {mean_pos:.3f} mm")
    print(f"  Max dE error:        {max_de:.1f} keV")

    if overcomplete:
        extra_idx = [i for i in range(n_seg) if i not in matched_idx]
        extra_energies = [float(final[i, 3]) for i in extra_idx]
        print(f"\n  Extra segments ({len(extra_idx)}): "
              f"energies={[f'{e:.4f}' for e in extra_energies]}")
        n_extra_dead = sum(1 for e in extra_energies if e <= DEATH_THRESH)
        print(f"  Extra dead: {n_extra_dead}/{len(extra_idx)}")

    print(f"  Total relocations: {cumulative_relocs}")

    # Plot
    _plot_sgld_closure(n_seg, n_truth, total_steps, mode, np.array(losses),
                       param_history, assignment, matched_idx,
                       np.array(dead_counts), relocation_steps)

    return errors, matched_idx


# =============================================================================
# Plotting (2x2 diagnostic panels)
# =============================================================================

def _plot_sgld_closure(n_seg, n_truth, total_steps, mode, losses, param_history,
                       assignment, matched_idx, dead_counts, relocation_steps):
    seg_colors = plt.cm.tab10(np.linspace(0, 1, max(n_seg, 3)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    extra_idx = [i for i in range(n_seg) if i not in matched_idx]

    LABEL_SIZE = 13
    TITLE_SIZE = 14
    TICK_SIZE = 11
    LEGEND_SIZE = 10

    # --- Top-left: loss curve ---
    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.0, alpha=0.7, label='SW loss')
    if mode != 'baseline':
        ax.axvline(WARMUP, color='gray', ls='--', lw=0.8, alpha=0.5,
                   label=f'Warmup={WARMUP}')
    for rs, nr in relocation_steps:
        ax.axvline(rs, color='red', ls=':', lw=0.5, alpha=0.3)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Loss', fontsize=LABEL_SIZE)
    ax.set_title('Loss Convergence (red = relocation)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Top-right: position error per matched segment ---
    ax = axes[0, 1]
    for s in matched_idx:
        pos_err = np.sqrt(np.sum(
            (param_history[:, s, :3] - assignment[s, :3]) ** 2, axis=1))
        ax.plot(pos_err, color=seg_colors[s], lw=1.0, label=f'Seg {s}')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('||pos error|| (mm)', fontsize=LABEL_SIZE)
    ax.set_title(f'Position Error (matched {n_truth} segments)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE - 2, ncol=2)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: total energy + per-segment energy ---
    ax = axes[1, 0]
    total_e = param_history[:, :, 3].sum(axis=1) * 1000  # keV
    truth_total_e = sum(assignment[s, 3] for s in matched_idx) * 1000
    ax.plot(total_e, 'k-', lw=2.0, alpha=0.8, label='Total energy')
    ax.axhline(truth_total_e, color='k', ls=':', lw=1.5, alpha=0.5,
               label=f'Truth total={truth_total_e:.0f} keV')
    for s in range(n_seg):
        is_extra = s in extra_idx
        ls = '--' if is_extra else '-'
        ax.plot(param_history[:, s, 3] * 1000, color=seg_colors[s],
                lw=0.6, ls=ls, alpha=0.4)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Energy (keV)', fontsize=LABEL_SIZE)
    ax.set_title('Energy (total=black, segments=thin)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: per-segment energy trajectories (detailed) ---
    ax = axes[1, 1]
    for s in range(n_seg):
        is_extra = s in extra_idx
        ls = '--' if is_extra else '-'
        alpha_val = 0.5 if is_extra else 1.0
        lbl = f'Seg {s}' + (' (extra)' if is_extra else '')
        ax.plot(param_history[:, s, 3] * 1000, color=seg_colors[s],
                lw=1.0, ls=ls, alpha=alpha_val, label=lbl)
        if not is_extra:
            ax.axhline(assignment[s, 3] * 1000, color=seg_colors[s],
                       ls=':', lw=0.8, alpha=0.4)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('dE (keV)', fontsize=LABEL_SIZE)
    ax.set_title('Per-Segment Energy (solid=matched, dashed=extra, dotted=truth)',
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE - 3, ncol=2)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    mode_desc = {'baseline': 'Adam only', 'noise': 'Adam + noise',
                 'full': 'Adam + noise + L1 + reloc'}[mode]
    oc_str = f' ({n_seg}opt/{n_truth}truth)' if n_seg > n_truth else ''
    n_reloc_total = sum(nr for _, nr in relocation_steps)
    fig.suptitle(
        f'{n_seg}-Seg MCMC Closure [{mode_desc}]{oc_str}  |  lr={LR_POSITION}, '
        f'lr_l1={LR_L1}, noise={NOISE_LR}, steps={total_steps}, '
        f'relocs={n_reloc_total}',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(
        OUT_DIR,
        f'sgld_closure_{n_seg}seg_{n_truth}truth_{mode}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCMC closure test')
    parser.add_argument('n_seg', type=int, nargs='?', default=5,
                        help='Number of optimizer segments (default: 5)')
    parser.add_argument('--n_truth', type=int, default=None,
                        help='Number of truth segments (default: same as n_seg)')
    parser.add_argument('--steps', type=int, default=TOTAL_STEPS,
                        help=f'Total optimization steps (default: {TOTAL_STEPS})')
    parser.add_argument('--mode', choices=['baseline', 'noise', 'full'],
                        default='full',
                        help='Validation mode (default: full)')
    args = parser.parse_args()
    run_sgld_closure(args.n_seg, n_truth=args.n_truth,
                     total_steps=args.steps, mode=args.mode)
