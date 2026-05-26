"""Slide 2: Sequential scan method — dE per segment showing the Bragg peak."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'lines.linewidth': 2.0,
})

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    generate_muon_segments_scan,
)

STEP_SIZE_MM = 0.5
N_SEGMENTS = 4000
MIN_ENERGY = 10.0
ENERGIES = [100.0, 200.0, 500.0]
COLORS = ['#D6604D', '#2166AC', '#4DAF4A']

log_T, dedx = load_dedx_table_jax()
start = jnp.zeros(3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left panel: dE per segment for multiple energies ──
for E, color in zip(ENERGIES, COLORS):
    _, de = generate_muon_segments_scan(
        jnp.float32(E), start, jnp.float32(0.8), jnp.float32(1.2),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
    jax.block_until_ready(de)
    de_np = np.array(de)

    # Convert segment index to distance in cm
    dist_cm = np.arange(N_SEGMENTS) * STEP_SIZE_MM / 10.0
    n_active = int(np.sum(de_np > 1e-6))
    range_cm = n_active * STEP_SIZE_MM / 10.0

    ax1.plot(dist_cm[:n_active+50], de_np[:n_active+50],
             color=color, lw=1.8,
             label=f'{E:.0f} MeV  (R = {range_cm:.0f} cm)')

ax1.set_xlabel('Distance Along Track (cm)')
ax1.set_ylabel('Energy Deposit  $\\Delta E$  (MeV / segment)')
ax1.set_title('Energy Deposit per Segment')
ax1.legend(loc='upper left', framealpha=0.9, edgecolor='0.7')
ax1.grid(True, which='major', ls='-', alpha=0.25)
ax1.set_ylim(bottom=-0.005)

# Add annotation for Bragg peak
ax1.annotate('Bragg peak', xy=(34, 0.17), fontsize=12,
             color='#D6604D', ha='center',
             arrowprops=dict(arrowstyle='->', color='#D6604D', lw=1.2),
             xytext=(25, 0.22))

# Add annotation for MIP region
ax1.annotate('MIP region\n(~flat dE/dx)', xy=(50, 0.076),
             fontsize=11, color='#2166AC', ha='center', va='bottom',
             xytext=(50, 0.16),
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.2))

# ── Right panel: zoom on Bragg peak for 200 MeV ──
_, de200 = generate_muon_segments_scan(
    jnp.float32(200.0), start, jnp.float32(0.8), jnp.float32(1.2),
    STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
jax.block_until_ready(de200)
de200_np = np.array(de200)
n_active_200 = int(np.sum(de200_np > 1e-6))
dist_cm = np.arange(N_SEGMENTS) * STEP_SIZE_MM / 10.0

# Show energy evolution on right y-axis
# Reconstruct energy along track
energies_along = np.zeros(n_active_200 + 20)
energies_along[0] = 200.0
for i in range(1, len(energies_along)):
    energies_along[i] = max(energies_along[i-1] - de200_np[i-1], MIN_ENERGY)

zoom_lo = max(0, n_active_200 - 200)
zoom_hi = n_active_200 + 20
sl = slice(zoom_lo, zoom_hi)

ax2.plot(dist_cm[sl], de200_np[sl], color='#2166AC', lw=2,
         label='$\\Delta E$ per segment')
ax2.set_xlabel('Distance Along Track (cm)')
ax2.set_ylabel('$\\Delta E$ (MeV / segment)', color='#2166AC')
ax2.tick_params(axis='y', labelcolor='#2166AC')

# Energy on secondary axis
ax2r = ax2.twinx()
ax2r.plot(dist_cm[sl], energies_along[:zoom_hi - zoom_lo],
          color='#B2182B', lw=2, ls='--', label='Kinetic energy')
ax2r.set_ylabel('Kinetic Energy $T$ (MeV)', color='#B2182B')
ax2r.tick_params(axis='y', labelcolor='#B2182B')
ax2r.axhline(MIN_ENERGY, color='gray', ls=':', lw=1.2, alpha=0.6)
ax2r.text(dist_cm[zoom_hi-5], MIN_ENERGY + 2,
          f'$E_{{\\min}}$ = {MIN_ENERGY} MeV', fontsize=10,
          color='gray', ha='right', va='bottom')

ax2.set_title('Bragg Peak Region (200 MeV)')
ax2.grid(True, which='major', ls='-', alpha=0.25)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', framealpha=0.9, edgecolor='0.7')

fig.tight_layout(w_pad=3)

out = os.path.join(os.path.dirname(__file__), 'slide2_scan_method.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
