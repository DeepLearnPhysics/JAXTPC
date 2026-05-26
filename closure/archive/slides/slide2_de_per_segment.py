"""Slide 2: Energy deposit per segment for muon tracks — showing Bragg peak."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'lines.linewidth': 2.2,
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

fig, ax = plt.subplots(figsize=(9, 6))

for E, color in zip(ENERGIES, COLORS):
    _, de = generate_muon_segments_scan(
        jnp.float32(E), start, jnp.float32(0.8), jnp.float32(1.2),
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
    jax.block_until_ready(de)
    de_np = np.array(de)
    dist_cm = np.arange(N_SEGMENTS) * STEP_SIZE_MM / 10.0
    n_active = int(np.sum(de_np > 1e-6))
    range_cm = n_active * STEP_SIZE_MM / 10.0

    # Plot up to just past where the muon stops
    end = min(n_active + 30, N_SEGMENTS)
    ax.plot(dist_cm[:end], de_np[:end], color=color, lw=2.2,
            label=f'{E:.0f} MeV  ($R$ = {range_cm:.0f} cm)')

ax.set_xlabel('Distance Along Track (cm)')
ax.set_ylabel('Energy Deposit $\\Delta E$ per Segment (MeV)')
ax.set_title('Muon Energy Deposit Profile')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.7')
ax.grid(True, which='major', ls='-', alpha=0.25)
ax.set_ylim(bottom=-0.005)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'slide2_de_per_segment.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
