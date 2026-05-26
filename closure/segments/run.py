"""
Full event closure analysis — real event reconstruction.

Loads a real event from HDF5, generates truth wire signals via full simulation,
then reconstructs using N_opt point charges with Adam + MCMC optimizer.

Produces intermediate plots every --plot-every steps:
- Event display (6-plane wire signals, same as run_simulation.ipynb)
- Progress plot (loss, energy, signal comparison, energy histogram)

Run from project root:
    python3 closure/segments/run.py --n-seg 10000 --steps 1000
    python3 closure/segments/run.py --data mpvmpr_20.h5 --n-seg 10000 --steps 1000 \
        --e-scale 5.0 --lr-e-mult 0.1 --mode baseline
"""

import os, argparse, time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax

from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data, build_deposit_data
from tools.losses import sobolev_loss_geomean_log1p, make_sobolev_weight
from tools.visualization import visualize_wire_signals
from functools import partial

# =============================================================================
# Constants
# =============================================================================

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Plot font sizes (from run_overcomplete_combos.py)
LABEL_SIZE = 16
TITLE_SIZE = 17
TICK_SIZE = 13
LEGEND_SIZE = 12
SUPTITLE_SIZE = 16

PLANE_NAMES = ['east_U', 'east_V', 'east_Y', 'west_U', 'west_V', 'west_Y']

DEFAULTS = {
    'lr': 1.0,
    'decay_rate': 0.999,
    'lr_e_mult': 0.01,
    'b1': 0.9,
    'b2': 0.999,
    'warmup': 100,
    'death_thresh': 0.012,
    'min_energy': 0.001,
    'noise_lr': 0.3,
    'l1': 0.0,
    'split_ratio': 0.8,
    'e_scale': None,
    'recomb_model': 'modified_box',
    'pos_jitter_mm': 50.0,
    'e_jitter_frac': 0.8,
    'threshold_enc': 1000,
    'max_reloc': 1000,
    'reloc_every': 25,
}


# =============================================================================
# Loss function (all active planes)
# =============================================================================

def build_loss_fn(forward, target_signals, spectral_weights, active_planes):
    """Build Sobolev geomean log1p loss over active wire planes."""
    planes_tuple = tuple(active_planes)
    def loss_fn(params):
        positions_mm = params[:, :3]
        de = params[:, 3]
        sigs = forward(positions_mm, de)
        loss = sobolev_loss_geomean_log1p(
            sigs, target_signals, spectral_weights, planes=planes_tuple)
        return loss, loss
    return loss_fn


# =============================================================================
# Parameterized relocation
# =============================================================================

@partial(jax.jit, static_argnums=(4,))
def relocate_segments(params, opt_state, rng_key, recomb_constants, max_reloc):
    """JIT-compiled relocation with charge-conserving K=1 split.

    Each dead segment is paired with a unique alive donor (no stacking).
    The donor's energy is split in two via the Q-space formula:

        Q ~ ln(max(alpha + xi, 1))   (natural extension)
        Q_new = Q_old / 2  ==>  dE_new = (dx/B)*[sqrt(alpha + B*dE_d/dx) - alpha]

    With the natural extension, any alive donor (alpha+xi > 1) stays alive
    after splitting (sqrt(alpha+xi) > 1), so no donor threshold is needed.

    Adam moments are zeroed for both donor and clone (as in 3DGS-MCMC)
    to prevent stale momentum from causing post-relocation cascades.

    Parameters
    ----------
    params : (N, 4) array [x, y, z, dE]
    opt_state : optax optimizer state
    rng_key : JAX PRNG key
    recomb_constants : (death_thresh, alpha, B, dx_cm) tuple
    max_reloc : int (static)
        Compiled array shape (typically n_seg).
    """
    death_thresh, alpha, B, dx_cm = recomb_constants

    N = params.shape[0]
    energies = params[:, 3]
    dead_mask = energies <= death_thresh
    alive_mask = ~dead_mask

    n_dead = jnp.sum(dead_mask)
    n_alive = jnp.sum(alive_mask)

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

    # Select dead segments (random shuffle via priority)
    dead_priority = jax.random.uniform(sk1, shape=(N,))
    dead_priority = jnp.where(dead_mask, dead_priority, -1.0)
    dead_indices = jnp.argsort(-dead_priority)[:max_reloc]

    # Select unique donors (random -- uniform selection preserves optimizer energy placement)
    alive_priority = jax.random.uniform(sk2, shape=(N,))
    alive_priority = jnp.where(alive_mask, alive_priority, -1.0)
    donor_indices = jnp.argsort(-alive_priority)[:max_reloc]

    # Relocate min(n_dead, n_alive, max_reloc)
    n_reloc = jnp.minimum(jnp.minimum(n_dead, n_alive), max_reloc)
    valid = jnp.arange(max_reloc) < n_reloc
    valid_3 = valid[:, None]

    # K=1 charge-conserving split in Q-space: Q_new = Q_old/2
    # With natural extension Q ~ ln(alpha+xi), so xi_new = sqrt(alpha+xi_d) - alpha
    donor_energies = energies[donor_indices]
    xi_donor = B * donor_energies / dx_cm
    xi_new = jnp.sqrt(alpha + xi_donor) - alpha
    split_de = xi_new * dx_cm / B

    # Place clone at donor's exact position
    clone_pos = params[donor_indices, :3]

    # Write clones into dead slots
    orig_pos = params[dead_indices, :3]
    orig_de = params[dead_indices, 3]
    write_pos = jnp.where(valid_3, clone_pos, orig_pos)
    write_de = jnp.where(valid, split_de, orig_de)

    new_params = params.at[dead_indices, :3].set(write_pos)
    new_params = new_params.at[dead_indices, 3].set(write_de)

    # Update donor energies (also get split_de)
    donor_orig_de = new_params[donor_indices, 3]
    donor_write_de = jnp.where(valid, split_de, donor_orig_de)
    new_params = new_params.at[donor_indices, 3].set(donor_write_de)

    # Zero optimizer state for both clones and donors (3DGS-MCMC approach)
    adam_state = opt_state[0]
    mu, nu = adam_state.mu, adam_state.nu
    zeros = jnp.zeros((max_reloc, params.shape[1]))

    # Zero clone (dead slot) moments
    new_mu = mu.at[dead_indices].set(
        jnp.where(valid_3, zeros, mu[dead_indices]))
    new_nu = nu.at[dead_indices].set(
        jnp.where(valid_3, zeros, nu[dead_indices]))

    # Zero donor moments (their energy just changed)
    new_mu = new_mu.at[donor_indices].set(
        jnp.where(valid_3, zeros, new_mu[donor_indices]))
    new_nu = new_nu.at[donor_indices].set(
        jnp.where(valid_3, zeros, new_nu[donor_indices]))

    new_adam = adam_state._replace(mu=new_mu, nu=new_nu)
    new_opt_state = (new_adam, opt_state[1])

    return new_params, new_opt_state, rng_key, n_reloc


# =============================================================================
# Training loop (reusable by sweep.py -- no intermediate plotting)
# =============================================================================

def run_training_loop(grad_fn, init_params, n_seg, total_steps, mode, cfg):
    """Run optimization loop with given hyperparameters.

    Parameters
    ----------
    grad_fn : callable
        JIT-compiled value_and_grad function.
    init_params : jnp.ndarray
        (n_seg, 4) initial parameters [x, y, z, dE].
    n_seg : int
        Number of optimizer segments.
    total_steps : int
        Total optimization steps.
    mode : str
        'baseline', 'noise', or 'full'.
    cfg : dict
        Hyperparameters (lr, decay_rate, etc.).

    Returns
    -------
    dict with losses, total_energies, final_params, dead_counts,
    relocation_steps, cumulative_relocs.
    """
    enable_noise = mode in ('noise', 'full')
    enable_l1 = mode == 'full'
    enable_relocation = mode == 'full'

    schedule = optax.exponential_decay(
        init_value=cfg['lr'], transition_steps=1, decay_rate=cfg['decay_rate'])
    optimizer = optax.adam(schedule, b1=cfg['b1'], b2=cfg['b2'])

    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    losses = []
    total_energies = []
    dead_counts = []
    relocation_steps = []
    cumulative_relocs = 0
    max_reloc = cfg.get('max_reloc', None) or n_seg
    reloc_every = cfg.get('reloc_every', 50)
    print_every = max(20, total_steps // 30)

    t0 = time.time()

    for step in range(total_steps):
        # Forward + backward
        (total_loss, _), grads = grad_fn(params)

        # Adam update with slower energy LR
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(cfg['lr_e_mult'])
        params = optax.apply_updates(params, updates)

        # Energy floor
        params = params.at[:, 3].set(jnp.maximum(params[:, 3], cfg['min_energy']))

        # L1 drain after warmup
        if enable_l1 and step >= cfg['warmup']:
            params = params.at[:, 3].add(-cfg['l1'])
            params = params.at[:, 3].set(jnp.maximum(params[:, 3], cfg['min_energy']))

        # Position noise (linear coupling)
        if enable_noise:
            lr_cur = float(schedule(step))
            noise_scale = lr_cur * cfg['noise_lr']
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                noise_scale * jax.random.normal(nk, shape=(n_seg, 3)))

        # Relocation (JIT-compiled, ~2ms, fixed shapes -> no retracing)
        if enable_relocation and step >= cfg['warmup'] and step % reloc_every == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key,
                cfg['recomb_constants'], max_reloc)
            cumulative_relocs += int(n_reloc)

        # Log (sync once per step for loss)
        loss_val = float(total_loss)
        losses.append(loss_val)

        if step % print_every == 0 or step == total_steps - 1:
            n_dead = int(jnp.sum(params[:, 3] <= cfg['death_thresh']))
            total_e = float(jnp.sum(params[:, 3]))
            total_energies.append(total_e)
            dead_counts.append(n_dead)
            print(f"  Step {step:5d}: loss={loss_val:.6f}  "
                  f"total_dE={total_e:.2f}  dead={n_dead}  "
                  f"relocs={cumulative_relocs}")
        else:
            total_energies.append(total_energies[-1] if total_energies else 0)
            dead_counts.append(dead_counts[-1] if dead_counts else 0)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / total_steps * 1000:.1f} ms/step)")

    return {
        'losses': np.array(losses),
        'total_energies': np.array(total_energies),
        'final_params': np.array(params),
        'dead_counts': np.array(dead_counts),
        'relocation_steps': relocation_steps,
        'cumulative_relocs': cumulative_relocs,
    }


# =============================================================================
# Event display helper
# =============================================================================

def save_event_display(signals_by_idx, active_planes, sim_config,
                       title, filepath, threshold_enc=0):
    """Save 6-plane event display using visualize_wire_signals (response path)."""
    plane_dict = {(p // 3, p % 3): np.array(signals_by_idx[p])
                  for p in active_planes}
    fig = visualize_wire_signals(plane_dict, sim_config,
                                 sparse=False, gamma=0.2,
                                 threshold_enc=threshold_enc)
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {filepath}")


# =============================================================================
# Plotting (2x2 diagnostic)
# =============================================================================

def plot_full_closure(result, truth_signals, recon_signals, active_planes,
                      truth_total_de, n_seg, total_steps, mode, cfg,
                      tag=''):
    """4x2 diagnostic plot for full closure analysis.

    Top-left:     Loss curve (with warmup/relocation markers)
    Top-right:    Signal comparison -- wire-summed profiles for best plane
    Mid-left:     Signal comparison -- west_Y
    Mid-right:    Q ratio trajectory
    Row3-left:    Final segment energy histogram
    Row3-right:   dE waterfall
    Bottom-left:  Per-plane signal ratio (recon/truth)
    Bottom-right: Signal error distribution (truth - recon)
    Bottom-right: (empty or future use)
    """
    fig, axes = plt.subplots(4, 2, figsize=(18, 26))

    losses = result['losses']
    total_energies = result['total_energies']
    relocation_steps = result['relocation_steps']

    # Pick the most active plane for signal comparison
    best_plane = max(active_planes,
                     key=lambda p: float(jnp.sum(jnp.abs(truth_signals[p]))))

    # --- Top-left: loss curve ---
    ax = axes[0, 0]
    ax.semilogy(losses, 'b-', lw=1.2, alpha=0.7, label='Sobolev geomean log1p')
    if mode != 'baseline':
        ax.axvline(cfg['warmup'], color='gray', ls='--', lw=1.0, alpha=0.5,
                   label=f'Warmup={cfg["warmup"]}')
    for rs, nr in relocation_steps:
        ax.axvline(rs, color='red', ls=':', lw=0.5, alpha=0.3)
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Loss', fontsize=LABEL_SIZE)
    ax.set_title('Loss Convergence (red = relocation)', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Top-right: signal comparison (wire-summed profile) ---
    ax = axes[0, 1]
    truth_sig = np.array(truth_signals[best_plane])
    recon_sig = np.array(recon_signals[best_plane])
    truth_profile = np.sum(np.abs(truth_sig), axis=1)
    recon_profile = np.sum(np.abs(recon_sig), axis=1)
    # Zoom to active wire range
    nonzero = np.where(truth_profile > 0)[0]
    if len(nonzero) > 0:
        w_lo = max(0, nonzero[0] - 10)
        w_hi = min(len(truth_profile), nonzero[-1] + 11)
    else:
        w_lo, w_hi = 0, len(truth_profile)
    wires = np.arange(w_lo, w_hi)
    ax.plot(wires, truth_profile[w_lo:w_hi], 'b-', lw=1.5, label='Truth')
    ax.plot(wires, recon_profile[w_lo:w_hi], 'r--', lw=1.5, label='Recon')
    ax.set_xlabel('Wire', fontsize=LABEL_SIZE)
    ax.set_ylabel('|Signal| (summed over time)', fontsize=LABEL_SIZE)
    ax.set_title(f'Signal Comparison ({PLANE_NAMES[best_plane]})',
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Mid-left: signal comparison west_Y ---
    y_plane = 5  # west_Y
    ax = axes[1, 0]
    if y_plane in active_planes:
        truth_y = np.array(truth_signals[y_plane])
        recon_y = np.array(recon_signals[y_plane])
        truth_y_prof = np.sum(np.abs(truth_y), axis=1)
        recon_y_prof = np.sum(np.abs(recon_y), axis=1)
        nz_y = np.where(truth_y_prof > 0)[0]
        if len(nz_y) > 0:
            yl, yh = max(0, nz_y[0] - 10), min(len(truth_y_prof), nz_y[-1] + 11)
        else:
            yl, yh = 0, len(truth_y_prof)
        yw = np.arange(yl, yh)
        ax.plot(yw, truth_y_prof[yl:yh], 'b-', lw=1.5, label='Truth')
        ax.plot(yw, recon_y_prof[yl:yh], 'r--', lw=1.5, label='Recon')
        ax.set_xlabel('Wire', fontsize=LABEL_SIZE)
        ax.set_ylabel('|Signal| (summed over time)', fontsize=LABEL_SIZE)
        ax.set_title(f'Signal Comparison ({PLANE_NAMES[y_plane]})',
                     fontsize=TITLE_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
    else:
        ax.set_title('west_Y: inactive', fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Mid-right: Q ratio trajectory ---
    ax = axes[1, 1]
    q_ratios = result.get('q_ratios', np.array([]))
    if len(q_ratios) > 0:
        ax.plot(q_ratios, 'b-', lw=2.0, alpha=0.8, label='Q ratio (sim/truth)')
        ax.axhline(1.0, color='green', ls='--', lw=1.5, alpha=0.5, label='Q=1')
    ax.set_xlabel('Step', fontsize=LABEL_SIZE)
    ax.set_ylabel('Q Ratio', fontsize=LABEL_SIZE)
    ax.set_title('Charge Conservation', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: final energy histogram ---
    ax = axes[2, 0]
    final_de = result['final_params'][:, 3] * 1000  # keV
    alive_de = final_de[final_de > cfg['death_thresh'] * 1000]
    n_dead_final = len(final_de) - len(alive_de)
    ax.hist(alive_de, bins=100, alpha=0.7, color='steelblue',
            label=f'Alive ({len(alive_de):,})')
    ax.axvline(cfg['death_thresh'] * 1000, color='r', ls='--', lw=1.5,
               label=f'Death thresh ({n_dead_final:,} dead)')
    ax.set_xlabel('dE (keV)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Count', fontsize=LABEL_SIZE)
    ax.set_title('Final Segment Energy Distribution', fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: dE waterfall ---
    import matplotlib.colors as mcolors
    ax = axes[2, 1]
    de_history = result.get('de_history', [])
    de_history_steps = result.get('de_history_steps', np.array([]))
    if len(de_history) > 1:
        all_alive_kev = np.concatenate([h[0] * 1000 for h in de_history if len(h[0]) > 0])
        if len(all_alive_kev) > 0:
            p99_kev = np.percentile(all_alive_kev, 99)
            n_bins = 60
            bin_edges = np.linspace(cfg['death_thresh'] * 1000, p99_kev, n_bins + 1)
            waterfall = np.zeros((n_bins + 2, len(de_history)))
            for j, (alive_de_j, n_dead_j) in enumerate(de_history):
                alive_kev = alive_de_j * 1000
                hist, _ = np.histogram(alive_kev, bins=bin_edges)
                overflow = int(np.sum(alive_kev > p99_kev))
                waterfall[0, j] = n_dead_j
                waterfall[1:n_bins+1, j] = hist
                waterfall[n_bins+1, j] = overflow
            step_arr = de_history_steps
            im = ax.imshow(waterfall, aspect='auto', origin='lower',
                           extent=[step_arr[0], step_arr[-1], 0, n_bins + 2],
                           cmap='inferno',
                           norm=mcolors.LogNorm(vmin=1, vmax=max(waterfall.max(), 2)),
                           interpolation='none')
            tick_pos = [0.5]
            tick_lab = ['Dead']
            for kev_val in [50, 200, 500, 1000]:
                if kev_val <= p99_kev:
                    bp = 1 + (kev_val - cfg['death_thresh'] * 1000) / (p99_kev - cfg['death_thresh'] * 1000) * n_bins
                    tick_pos.append(bp)
                    tick_lab.append(f'{kev_val}')
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_lab)
            ax.set_xlabel('Step', fontsize=LABEL_SIZE)
            ax.set_ylabel('dE (keV)', fontsize=LABEL_SIZE)
            ax.set_title('dE Waterfall', fontsize=TITLE_SIZE)
            ax.tick_params(labelsize=TICK_SIZE)
    else:
        ax.axis('off')

    # --- Row 3, left: per-plane signal ratio ---
    ax = axes[3, 0]
    plane_ratios = []
    plane_labels = []
    for p in active_planes:
        t_arr = np.array(truth_signals[p])
        r_arr = np.array(recon_signals[p])
        t_sum = np.sum(np.abs(t_arr))
        r_sum = np.sum(np.abs(r_arr))
        if t_sum > 1:
            plane_ratios.append(r_sum / t_sum)
            plane_labels.append(PLANE_NAMES[p])
    if plane_ratios:
        colors = ['#e74c3c' if abs(r - 1) > 0.02 else '#2ecc71' for r in plane_ratios]
        bars = ax.bar(range(len(plane_ratios)), plane_ratios, color=colors, alpha=0.8)
        ax.axhline(1.0, color='black', ls='--', lw=1.5)
        ax.set_xticks(range(len(plane_labels)))
        ax.set_xticklabels(plane_labels, fontsize=TICK_SIZE)
        ax.set_ylabel('|Recon| / |Truth|', fontsize=LABEL_SIZE)
        ax.set_title('Per-Plane Signal Ratio', fontsize=TITLE_SIZE)
        ax.set_ylim(min(0.9, min(plane_ratios) - 0.02), max(1.1, max(plane_ratios) + 0.02))
        for i, r in enumerate(plane_ratios):
            ax.text(i, r + 0.002, f'{r:.4f}', ha='center', fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Row 3, right: signal error distribution ---
    ax = axes[3, 1]
    all_errors = []
    for p in active_planes:
        t_arr = np.array(truth_signals[p])
        r_arr = np.array(recon_signals[p])
        diff = (t_arr - r_arr).ravel()
        nonzero = diff[np.abs(diff) > 0.01]
        if len(nonzero) > 0:
            all_errors.append(nonzero)
    if all_errors:
        all_err = np.concatenate(all_errors)
        p1, p99 = np.percentile(all_err, [1, 99])
        clipped = all_err[(all_err >= p1) & (all_err <= p99)]
        ax.hist(clipped, bins=100, alpha=0.7, color='steelblue', density=True)
        ax.axvline(0, color='red', ls='--', lw=1.5)
        rms = np.sqrt(np.mean(all_err**2))
        ax.set_xlabel('Truth - Recon (ADC)', fontsize=LABEL_SIZE)
        ax.set_ylabel('Density', fontsize=LABEL_SIZE)
        ax.set_title(f'Signal Error Distribution (RMS={rms:.2f})', fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # Suptitle
    mode_desc = {'baseline': 'Adam only', 'noise': 'Adam + noise',
                 'full': 'Adam + noise + L1 + reloc'}[mode]
    q_final = q_ratios[-1] if len(q_ratios) > 0 else 0
    fig.suptitle(
        f'Full Closure [{mode_desc}]  |  N={n_seg:,}, steps={total_steps}, '
        f'lr={cfg["lr"]}, d={cfg["decay_rate"]}\n'
        f'Loss={losses[-1]:.6f}, Q={q_final:.3f}, '
        f'relocs={result["cumulative_relocs"]}, '
        f'dead={result["dead_counts"][-1]}',
        fontsize=SUPTITLE_SIZE, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(OUT_DIR,
                         f'full_closure_{n_seg}seg_{mode}{tag}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Main pipeline
# =============================================================================

def run_full_closure(h5_path, event_idx=0, n_seg=40000, total_steps=3000,
                     mode='baseline', config=None, plot_every=100,
                     config_yaml='config/cubic_wireplane_config.yaml',
                     dx_mm=0.3, sobolev_s=1.0, tag='', schedule_type='exponential'):
    """Run full event closure analysis with intermediate plotting.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 event file.
    event_idx : int
        Event index in HDF5 file.
    n_seg : int
        Number of optimizer segments.
    total_steps : int
        Total optimization steps.
    mode : str
        'baseline' (Adam only), 'noise' (+ noise), 'full' (+ L1 + relocation).
    config : dict or None
        Hyperparameter overrides (merged with DEFAULTS).
    plot_every : int
        Save event display + progress plot every N steps. 0 to disable.
    config_yaml : str
        Path to detector YAML config file.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    enable_noise = mode in ('noise', 'full')
    enable_l1 = mode == 'full'
    enable_relocation = mode == 'full'

    print(f"\n{'=' * 70}")
    print(f"FULL EVENT CLOSURE  [mode={mode}, n_seg={n_seg:,}, "
          f"steps={total_steps}]")
    print(f"{'=' * 70}")
    print(f"File: {h5_path}, event={event_idx}")
    print(f"  lr={cfg['lr']}, decay={cfg['decay_rate']}, "
          f"e_mult={cfg['lr_e_mult']}, e_scale={cfg['e_scale']}")
    print(f"  noise_lr={cfg['noise_lr']}, l1={cfg['l1']}, "
          f"death={cfg['death_thresh']}, split={cfg['split_ratio']}")
    print(f"  warmup={cfg['warmup']}, plot_every={plot_every}, "
          f"recomb={cfg['recomb_model']}")
    print(f"  pos_jitter={cfg['pos_jitter_mm']}mm, "
          f"e_jitter=+/-{cfg['e_jitter_frac'] * 100:.0f}%")
    print(f"  threshold_enc={cfg['threshold_enc']}, "
          f"max_reloc={cfg['max_reloc']}, reloc_every={cfg['reloc_every']}")
    print(f"  fixed_dx={dx_mm}mm, sobolev_s={sobolev_s}")

    # --- 1. Load data ---
    print("\nLoading event data...")
    raw = load_particle_step_data(h5_path, event_idx, verbose=False)
    n_truth = raw['positions_mm'].shape[0]
    truth_total_de = float(np.sum(raw['de']))
    print(f"  {n_truth:,} segments, total dE={truth_total_de:.2f} MeV")

    # --- 2. Truth simulation (dense, no noise/electronics/track_hits) ---
    print("Generating truth signals...")
    detector_config = generate_detector(config_yaml)

    total_pad = max(200_000, n_truth + 1000)
    total_pad = ((total_pad + 9999) // 10000) * 10000

    sim_truth = DetectorSimulator(
        detector_config,
        use_bucketed=False,
        total_pad=total_pad,
        response_chunk_size=50_000,
        include_noise=False,
        include_electronics=False,
        include_track_hits=False,
        recombination_model=cfg['recomb_model'],
    )

    # Compute recombination constants for charge-conserving relocation
    # Q = (dx/B) * ln(alpha + B*dE/dx), so split uses (K+1)-th root formula
    dx_cm = dx_mm / 10.0
    rp = sim_truth.default_sim_params.recomb_params
    dens = float(rp.density)
    alpha_r = float(rp.alpha)
    recomb_model = 'emb' if hasattr(rp, 'beta_90') else 'modified_box'
    if recomb_model == 'emb':
        beta_r = float(rp.beta_90)
    else:
        beta_r = float(rp.beta)
    field_kVcm = float(rp.field_strength_Vcm) / 1000.0
    B_eff = beta_r / dens / field_kVcm
    cfg['recomb_constants'] = (cfg['death_thresh'], alpha_r, B_eff, dx_cm)
    print(f"  Recomb split: model={recomb_model}, alpha={alpha_r}, "
          f"B={B_eff:.4f}, dx={dx_cm}cm, death={cfg['death_thresh']}")

    # Build DepositData — use real dx from HDF5, force t0=0 (no interaction time offset)
    forced_t0 = np.zeros(n_truth, dtype=np.float32)
    deposits = build_deposit_data(
        raw['positions_mm'], raw['de'], raw['dx'], sim_truth.config,
        theta=raw['theta'], phi=raw['phi'], track_ids=raw['track_ids'],
        t0_us=forced_t0, interaction_ids=raw.get('interaction_ids'),
        root_track_ids=raw.get('root_track_ids'), pdg=raw.get('pdg'))

    t0 = time.time()
    response_signals, _, _ = sim_truth.process_event(deposits)
    print(f"  Truth sim done in {time.time() - t0:.1f}s")

    # --- 3. Build target signals and spectral weights ---
    truth_signals_dict = {}
    active_planes = []
    sob_weights_dict = {}

    for (side, plane), signal in response_signals.items():
        plane_idx = side * 3 + plane
        signal = jnp.asarray(signal)
        truth_signals_dict[plane_idx] = signal
        if jnp.any(signal != 0):
            active_planes.append(plane_idx)
            H, W = signal.shape
            sob_weights_dict[plane_idx] = make_sobolev_weight(H, W, s=sobolev_s)

    active_planes.sort()

    # Build 6-element tuples (sobolev_loss_geomean_log1p indexes by int 0-5)
    truth_signals = tuple(
        truth_signals_dict[i] if i in truth_signals_dict
        else jnp.zeros((1, 1))
        for i in range(6)
    )
    spectral_weights = tuple(
        sob_weights_dict[i] if i in sob_weights_dict
        else jnp.zeros((1, 1))
        for i in range(6)
    )
    print(f"  Active planes: {[PLANE_NAMES[p] for p in active_planes]}")
    for p in active_planes:
        sig = truth_signals[p]
        print(f"    {PLANE_NAMES[p]}: {sig.shape}, "
              f"max={float(jnp.max(jnp.abs(sig))):.4f}")

    # --- 3b. Plot truth event display ---
    t_enc = cfg['threshold_enc']
    save_event_display(
        truth_signals, active_planes, sim_truth.config,
        f'Truth Event (event {event_idx}, {n_truth:,} segments)',
        os.path.join(OUT_DIR, f'truth_event{tag}.png'),
        threshold_enc=t_enc)

    # --- 4. Initialize params (subsample + jitter + energy scaling) ---
    print("Initializing optimizer segments...")
    rng = np.random.RandomState(42)
    truth_pos = np.asarray(raw['positions_mm'])
    truth_de = np.asarray(raw['de'])


    replace = n_seg > n_truth
    indices = rng.choice(n_truth, size=n_seg, replace=replace)

    # Auto e_scale: each segment represents n_truth/n_seg truth segments
    e_scale = cfg['e_scale']
    if e_scale is None:
        e_scale = n_truth / n_seg
    print(f"  e_scale={e_scale:.2f} (n_truth/n_seg={n_truth/n_seg:.2f})")

    init_pos = truth_pos[indices].copy()
    init_de = truth_de[indices].copy() * e_scale

    # Per-track position jitter (one offset per track, preserves within-track coherence)
    truth_tids = np.asarray(raw['track_ids'])
    init_tids = truth_tids[indices]
    for tid in np.unique(init_tids):
        init_pos[init_tids == tid] += rng.normal(0, cfg['pos_jitter_mm'], size=3)
    # Energy jitter
    init_de *= rng.uniform(
        1.0 - cfg['e_jitter_frac'], 1.0 + cfg['e_jitter_frac'], size=n_seg)
    init_de = np.maximum(init_de, cfg['min_energy'])

    init_params = jnp.array(np.column_stack([init_pos, init_de]))
    init_total_de = float(jnp.sum(init_params[:, 3]))
    print(f"  Init total dE={init_total_de:.2f} MeV "
          f"(truth={truth_total_de:.2f} MeV, e_scale={e_scale:.2f})")

    # --- 5. Build differentiable optimizer simulator ---
    print(f"\nBuilding differentiable simulator (n_seg={n_seg:,})...")
    sim_opt = DetectorSimulator(
        detector_config, differentiable=True, n_segments=n_seg,
        recombination_model=cfg['recomb_model'])
    sim_params = sim_opt.default_sim_params
    def fwd_opt(positions_mm, de):
        return sim_opt.forward_segments(sim_params, positions_mm, de, dx=dx_mm)

    # --- 6. Loss + gradient function ---
    loss_fn = build_loss_fn(fwd_opt, truth_signals, spectral_weights,
                            active_planes)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # --- 7. JIT warmup ---
    print("Warming up JIT...")
    t0 = time.time()
    _ = grad_fn(init_params)
    print(f"JIT warm-up done in {time.time() - t0:.1f}s")

    # --- 8. Training loop with intermediate plotting ---
    print(f"\nOptimizing ({total_steps} steps, mode={mode})...")

    if schedule_type == 'cosine':
        schedule = optax.cosine_decay_schedule(
            init_value=cfg['lr'], decay_steps=total_steps, alpha=0.0)
    elif schedule_type == 'warmup_cosine':
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=cfg['lr'],
            warmup_steps=min(50, total_steps // 20),
            decay_steps=total_steps, end_value=0.0)
    else:
        schedule = optax.exponential_decay(
            init_value=cfg['lr'], transition_steps=1, decay_rate=cfg['decay_rate'])
    print(f"  Schedule: {schedule_type}, lr={cfg['lr']}")
    optimizer = optax.adam(schedule, b1=cfg['b1'], b2=cfg['b2'])

    params = init_params
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(123)

    def compute_Q(de_arr, dx_cm_arr=None):
        d = dx_cm if dx_cm_arr is None else dx_cm_arr
        return (d / B_eff) * np.log(np.maximum(alpha_r + B_eff * de_arr / d, 1.0))
    truth_dx_cm = np.maximum(np.asarray(raw['dx']) / 10.0, 1e-10)
    truth_total_Q = float(compute_Q(np.asarray(raw['de']), truth_dx_cm).sum())

    losses = []
    total_energies = []
    q_ratios = []
    dead_counts = []
    de_history = []
    de_history_steps = []
    relocation_steps = []
    cumulative_relocs = 0
    max_reloc = cfg.get('max_reloc', None) or n_seg
    reloc_every = cfg.get('reloc_every', 50)
    print_every = max(20, total_steps // 30)
    de_hist_every = max(1, total_steps // 200)

    t0 = time.time()

    for step in range(total_steps):
        (total_loss, _), grads = grad_fn(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = updates.at[:, 3].multiply(cfg['lr_e_mult'])
        params = optax.apply_updates(params, updates)

        params = params.at[:, 3].set(jnp.maximum(params[:, 3], cfg['min_energy']))

        if enable_l1 and step >= cfg['warmup']:
            params = params.at[:, 3].add(-cfg['l1'])
            params = params.at[:, 3].set(jnp.maximum(params[:, 3], cfg['min_energy']))

        if enable_noise:
            lr_cur = float(schedule(step))
            noise_scale = lr_cur * cfg['noise_lr']
            rng_key, nk = jax.random.split(rng_key)
            params = params.at[:, :3].add(
                noise_scale * jax.random.normal(nk, shape=(n_seg, 3)))

        if enable_relocation and step >= cfg['warmup'] and step % reloc_every == 0:
            params, opt_state, rng_key, n_reloc = relocate_segments(
                params, opt_state, rng_key,
                cfg['recomb_constants'], max_reloc)
            cumulative_relocs += int(n_reloc)

        loss_val = float(total_loss)
        losses.append(loss_val)

        # Q ratio + stats
        if step % print_every == 0 or step == total_steps - 1:
            p_np = np.array(params)
            n_dead = int(np.sum(p_np[:, 3] <= cfg['death_thresh']))
            alive_mask = p_np[:, 3] > cfg['death_thresh']
            sim_Q = float(compute_Q(p_np[alive_mask, 3]).sum())
            q_ratio = sim_Q / truth_total_Q if truth_total_Q > 0 else 0
            total_e = float(np.sum(p_np[:, 3]))
            total_energies.append(total_e)
            q_ratios.append(q_ratio)
            dead_counts.append(n_dead)
            print(f"  Step {step:5d}: loss={loss_val:.6f}  "
                  f"Q={q_ratio:.3f}  dead={n_dead}  "
                  f"relocs={cumulative_relocs}  ({time.time()-t0:.0f}s)",
                  flush=True)
        else:
            total_energies.append(total_energies[-1] if total_energies else 0)
            q_ratios.append(q_ratios[-1] if q_ratios else 0)
            dead_counts.append(dead_counts[-1] if dead_counts else 0)

        # dE histogram snapshot
        if step % de_hist_every == 0 or step == total_steps - 1:
            p_np = np.array(params[:, 3])
            alive_de = p_np[p_np > cfg['death_thresh']]
            n_dead_h = int(np.sum(p_np <= cfg['death_thresh']))
            de_history.append((alive_de.copy(), n_dead_h))
            de_history_steps.append(step)

        # --- Intermediate plotting ---
        should_plot = (plot_every > 0 and
                       (step > 0 and step % plot_every == 0
                        or step == total_steps - 1))
        if should_plot:
            # Forward pass for visualization
            recon_raw = fwd_opt(params[:, :3], params[:, 3])
            recon_signals = {p: jnp.asarray(recon_raw[p])
                             for p in active_planes}

            # Event display
            de_ratio_cur = float(jnp.sum(params[:, 3])) / truth_total_de
            save_event_display(
                recon_signals, active_planes, sim_truth.config,
                f'Recon step {step}  (loss={loss_val:.4f}, '
                f'dE_ratio={de_ratio_cur:.3f})',
                os.path.join(OUT_DIR, f'recon_step_{step:04d}{tag}.png'),
                threshold_enc=t_enc)

            # Diff event display (truth - recon)
            diff_signals = {p: truth_signals[p] - recon_signals[p]
                            for p in active_planes}
            save_event_display(
                diff_signals, active_planes, sim_truth.config,
                f'Diff (truth - recon) step {step}',
                os.path.join(OUT_DIR, f'diff_step_{step:04d}{tag}.png'),
                threshold_enc=t_enc)

            # Progress plot
            result_so_far = {
                'losses': np.array(losses),
                'total_energies': np.array(total_energies),
                'q_ratios': np.array(q_ratios),
                'final_params': np.array(params),
                'dead_counts': np.array(dead_counts),
                'de_history': de_history,
                'de_history_steps': np.array(de_history_steps),
                'relocation_steps': relocation_steps,
                'cumulative_relocs': cumulative_relocs,
            }
            plot_full_closure(
                result_so_far, truth_signals, recon_signals, active_planes,
                truth_total_de, n_seg, step + 1, mode, cfg,
                tag=f'{tag}_progress')

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s "
          f"({elapsed / total_steps * 1000:.1f} ms/step)")

    # --- Summary ---
    final_params = np.array(params)
    final_total_de = float(np.sum(final_params[:, 3]))
    de_ratio = final_total_de / truth_total_de if truth_total_de > 0 else 0
    final_q = q_ratios[-1] if q_ratios else 0
    print(f"\nResults:")
    print(f"  Final loss:     {losses[-1]:.6f}")
    print(f"  Final Q ratio:  {final_q:.3f}")
    print(f"  Final total dE: {final_total_de:.2f} MeV "
          f"(truth={truth_total_de:.2f}, ratio={de_ratio:.3f})")
    print(f"  Total relocs:   {cumulative_relocs}")
    print(f"  Final dead:     {dead_counts[-1]}")

    result = {
        'losses': np.array(losses),
        'total_energies': np.array(total_energies),
        'final_params': final_params,
        'dead_counts': np.array(dead_counts),
        'relocation_steps': relocation_steps,
        'cumulative_relocs': cumulative_relocs,
        'truth_total_de': truth_total_de,
        'de_ratio': de_ratio,
        'active_planes': active_planes,
    }
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Full event closure analysis')
    parser.add_argument('--data', default='mpvmpr_20.h5',
                        help='Path to HDF5 event file (default: mpvmpr_20.h5)')
    parser.add_argument('--config-yaml', default='config/cubic_wireplane_config.yaml',
                        help='Path to detector YAML config (default: config/cubic_wireplane_config.yaml)')
    parser.add_argument('--event', type=int, default=0,
                        help='Event index (default: 0)')
    parser.add_argument('--n-seg', type=int, default=40000,
                        help='Number of optimizer segments (default: 40000)')
    parser.add_argument('--steps', type=int, default=3000,
                        help='Total optimization steps (default: 3000)')
    parser.add_argument('--mode', choices=['baseline', 'noise', 'full'],
                        default='baseline',
                        help='Optimizer mode (default: baseline)')
    parser.add_argument('--plot-every', type=int, default=100,
                        help='Plot interval in steps (default: 100, 0=off)')
    # Hyperparameter overrides
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr-e-mult', type=float, default=None)
    parser.add_argument('--decay-rate', type=float, default=None)
    parser.add_argument('--noise-lr', type=float, default=None)
    parser.add_argument('--l1', type=float, default=None)
    parser.add_argument('--split', type=float, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--e-scale', type=float, default=None,
                        help='Energy init scale factor (default: auto=n_truth/n_seg)')
    parser.add_argument('--recomb', type=str, default=None,
                        choices=['modified_box', 'emb'],
                        help='Recombination model (default: config default)')
    parser.add_argument('--pos-jitter', type=float, default=None)
    parser.add_argument('--e-jitter', type=float, default=None)
    parser.add_argument('--threshold-enc', type=int, default=None,
                        help='Deadband threshold for event displays (default: 1000)')
    parser.add_argument('--reloc-every', type=int, default=None,
                        help='Relocation interval in steps (default: 50)')
    parser.add_argument('--max-reloc', type=int, default=None,
                        help='Max relocations per round (default: n_seg)')
    parser.add_argument('--dx', type=float, default=0.3,
                        help='Fixed segment dx in mm (default: 0.3)')
    parser.add_argument('--sobolev-s', type=float, default=1.0,
                        help='Sobolev exponent s (default: 1.0)')
    parser.add_argument('--schedule', type=str, default='exponential',
                        choices=['exponential', 'cosine', 'warmup_cosine'],
                        help='LR schedule (default: exponential)')
    parser.add_argument('--tag', type=str, default='',
                        help='Tag appended to output filenames (e.g. --tag _v2)')

    args = parser.parse_args()

    # Build config from CLI overrides
    overrides = {}
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.lr_e_mult is not None:
        overrides['lr_e_mult'] = args.lr_e_mult
    if args.decay_rate is not None:
        overrides['decay_rate'] = args.decay_rate
    if args.noise_lr is not None:
        overrides['noise_lr'] = args.noise_lr
    if args.l1 is not None:
        overrides['l1'] = args.l1
    if args.split is not None:
        overrides['split_ratio'] = args.split
    if args.warmup is not None:
        overrides['warmup'] = args.warmup
    if args.e_scale is not None:
        overrides['e_scale'] = args.e_scale
    if args.pos_jitter is not None:
        overrides['pos_jitter_mm'] = args.pos_jitter
    if args.e_jitter is not None:
        overrides['e_jitter_frac'] = args.e_jitter
    if args.recomb is not None:
        overrides['recomb_model'] = args.recomb
    if args.threshold_enc is not None:
        overrides['threshold_enc'] = args.threshold_enc
    if args.reloc_every is not None:
        overrides['reloc_every'] = args.reloc_every
    if args.max_reloc is not None:
        overrides['max_reloc'] = args.max_reloc

    run_full_closure(
        args.data,
        event_idx=args.event,
        n_seg=args.n_seg,
        total_steps=args.steps,
        mode=args.mode,
        config=overrides,
        plot_every=args.plot_every,
        config_yaml=args.config_yaml,
        dx_mm=args.dx,
        sobolev_s=args.sobolev_s,
        tag=args.tag,
        schedule_type=args.schedule,
    )
