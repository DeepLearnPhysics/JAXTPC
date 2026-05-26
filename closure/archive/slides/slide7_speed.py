"""Slide 7: Speed comparison — scan vs CSDA for forward and gradient passes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

plt.rcParams.update({
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
N_RUNS = 100

log_T, dedx = load_dedx_table_jax()
R_cm, T_MeV = build_consistent_csda_table(log_T, dedx, n_points=10000)

start = jnp.zeros(3)
E = jnp.float32(500.0)
theta, phi = jnp.float32(0.8), jnp.float32(1.2)

# Forward functions
fwd_scan = jax.jit(lambda e: generate_muon_segments_scan(
    e, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    min_energy_mev=MIN_ENERGY, smooth_temperature=0.0))
fwd_csda = jax.jit(lambda e: generate_muon_segments_csda(
    e, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    R_cm, T_MeV))

# Gradient functions
grad_scan = jax.jit(jax.value_and_grad(
    lambda e: jnp.sum(generate_muon_segments_scan(
        e, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        min_energy_mev=MIN_ENERGY, smooth_temperature=0.2)[1])))
grad_csda = jax.jit(jax.value_and_grad(
    lambda e: jnp.sum(generate_muon_segments_csda(
        e, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        R_cm, T_MeV)[1])))

# Warm up
jax.block_until_ready(fwd_scan(E))
jax.block_until_ready(fwd_csda(E))
jax.block_until_ready(grad_scan(E))
jax.block_until_ready(grad_csda(E))

# Benchmark
def bench(fn, E, n):
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn(E))
    return (time.perf_counter() - t0) / n * 1000

t_fwd_scan = bench(fwd_scan, E, N_RUNS)
t_fwd_csda = bench(fwd_csda, E, N_RUNS)
t_grad_scan = bench(grad_scan, E, N_RUNS)
t_grad_csda = bench(grad_csda, E, N_RUNS)

print(f'Forward:  scan={t_fwd_scan:.1f}ms  csda={t_fwd_csda:.2f}ms  '
      f'speedup={t_fwd_scan/t_fwd_csda:.0f}x')
print(f'Gradient: scan={t_grad_scan:.1f}ms  csda={t_grad_csda:.2f}ms  '
      f'speedup={t_grad_scan/t_grad_csda:.0f}x')

# ── Figure ──
fig, ax = plt.subplots(figsize=(9, 6))

labels = ['Forward\n(scan)', 'Forward\n(CSDA)', 'Gradient\n(scan)', 'Gradient\n(CSDA)']
times = [t_fwd_scan, t_fwd_csda, t_grad_scan, t_grad_csda]
colors = ['#2166AC', '#D6604D', '#2166AC', '#D6604D']
hatches = [None, None, '///', '///']

bars = ax.bar(labels, times, color=colors, width=0.55, edgecolor='0.2', lw=1.2)
for bar, h in zip(bars, hatches):
    if h:
        bar.set_hatch(h)

# Time labels
for bar, t in zip(bars, times):
    if t > 100:
        txt = f'{t:.0f} ms'
    else:
        txt = f'{t:.1f} ms'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
            txt, ha='center', va='bottom', fontsize=14, fontweight='bold')

# Speedup annotations
fwd_speedup = t_fwd_scan / t_fwd_csda
grad_speedup = t_grad_scan / t_grad_csda
mid_fwd = np.sqrt(t_fwd_scan * t_fwd_csda)
mid_grad = np.sqrt(t_grad_scan * t_grad_csda)
ax.text(0.5, mid_fwd, f'{fwd_speedup:.0f}x',
        fontsize=22, fontweight='bold', color='#4DAF4A', ha='center', va='center')
ax.text(2.5, mid_grad, f'{grad_speedup:.0f}x',
        fontsize=22, fontweight='bold', color='#4DAF4A', ha='center', va='center')

ax.set_ylabel('Runtime (ms)')
ax.set_title('Sequential Scan vs Parallel CSDA')
ax.set_yscale('log')
ax.grid(True, axis='y', alpha=0.25, which='both')

# Legend for scan/csda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2166AC', edgecolor='0.2', label='Sequential scan'),
                   Patch(facecolor='#D6604D', edgecolor='0.2', label='Parallel CSDA')]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, edgecolor='0.7')

fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
out = os.path.join(os.path.dirname(__file__), 'slide7_speed.png')
fig.savefig(out, dpi=200)
plt.close(fig)
print(f'Saved {out}')
