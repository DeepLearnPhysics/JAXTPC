"""Slide 4: Scan vs CSDA — identical results, massive speedup."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'lines.linewidth': 2.0,
})

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    build_consistent_csda_table,
    generate_muon_segments_scan,
    generate_muon_segments_csda,
)

STEP_SIZE_MM = 0.5
N_SEGMENTS = 4000
MIN_ENERGY = 10.0
ENERGY = 200.0

log_T, dedx = load_dedx_table_jax()
R_cm, T_MeV = build_consistent_csda_table(log_T, dedx, n_points=10000)

start = jnp.zeros(3)
theta, phi = jnp.float32(0.8), jnp.float32(1.2)
E = jnp.float32(ENERGY)

# Generate with both methods
_, de_scan = generate_muon_segments_scan(
    E, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
jax.block_until_ready(de_scan)

_, de_csda = generate_muon_segments_csda(
    E, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    R_cm, T_MeV, relax_steps=2.0)
jax.block_until_ready(de_csda)

de_scan_np = np.array(de_scan)
de_csda_np = np.array(de_csda)
dist_cm = np.arange(N_SEGMENTS) * STEP_SIZE_MM / 10.0
n_active = int(np.sum(de_scan_np > 1e-6))

# Speed benchmark
fwd_scan = jax.jit(lambda: generate_muon_segments_scan(
    E, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    min_energy_mev=MIN_ENERGY, smooth_temperature=0.0))
fwd_csda = jax.jit(lambda: generate_muon_segments_csda(
    E, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    R_cm, T_MeV))

# Warm up
jax.block_until_ready(fwd_scan())
jax.block_until_ready(fwd_csda())

N_RUNS = 100
t0 = time.perf_counter()
for _ in range(N_RUNS):
    jax.block_until_ready(fwd_scan())
t_scan = (time.perf_counter() - t0) / N_RUNS * 1000

t0 = time.perf_counter()
for _ in range(N_RUNS):
    jax.block_until_ready(fwd_csda())
t_csda = (time.perf_counter() - t0) / N_RUNS * 1000

speedup = t_scan / t_csda
print(f'Scan: {t_scan:.2f} ms, CSDA: {t_csda:.3f} ms, Speedup: {speedup:.0f}x')

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: overlay of dE profiles
end = n_active + 30
ax1.plot(dist_cm[:end], de_scan_np[:end], color='#2166AC', lw=2.5,
         label='Sequential scan', zorder=3)
ax1.plot(dist_cm[:end], de_csda_np[:end], color='#D6604D', lw=1.5,
         ls='--', label='CSDA parallel', zorder=4)

ax1.set_xlabel('Distance Along Track (cm)')
ax1.set_ylabel('$\\Delta E$ per Segment (MeV)')
ax1.set_title('Energy Deposits: Scan vs CSDA')
ax1.legend(framealpha=0.9, edgecolor='0.7')
ax1.grid(True, alpha=0.25)
ax1.set_ylim(bottom=-0.005)

# Inset: residual over the active region
ax_in = ax1.inset_axes([0.3, 0.45, 0.55, 0.4])
active_end = n_active + 5
residual = de_scan_np[:active_end] - de_csda_np[:active_end]
ax_in.plot(dist_cm[:active_end], residual * 1e3, color='0.3', lw=1.2)
ax_in.set_ylabel('Residual ($\\times 10^{-3}$ MeV)', fontsize=10)
ax_in.set_xlabel('Distance (cm)', fontsize=10)
ax_in.tick_params(labelsize=9)
max_res = np.nanmax(np.abs(residual))
ax_in.set_title(f'max |res| = {max_res:.1e} MeV', fontsize=10)
ax_in.grid(True, alpha=0.2)
ax_in.axhline(0, color='k', lw=0.5)

# Right: speed comparison bar chart
methods = ['Scan\n(sequential)', 'CSDA\n(parallel)']
times = [t_scan, t_csda]
colors = ['#2166AC', '#D6604D']
bars = ax2.bar(methods, times, color=colors, width=0.5, edgecolor='0.3', lw=1.2)

# Add time labels on bars
for bar, t in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + t_scan*0.03,
             f'{t:.2f} ms', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.set_ylabel('Runtime (ms)')
ax2.set_title(f'Forward Pass Speed ({speedup:.0f}x faster)')
ax2.grid(True, axis='y', alpha=0.25)
ax2.set_ylim(0, t_scan * 1.25)

fig.tight_layout(w_pad=3)
out = os.path.join(os.path.dirname(__file__), 'slide4_scan_vs_csda.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
