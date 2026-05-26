"""Slide 3: Hard clamp vs softplus smoothing at the stopping boundary."""
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
ENERGY = 200.0

log_T, dedx = load_dedx_table_jax()
start = jnp.zeros(3)
theta, phi = jnp.float32(0.8), jnp.float32(1.2)

# ── Generate hard and smooth (multiple temperatures) ──
_, de_hard = generate_muon_segments_scan(
    jnp.float32(ENERGY), start, theta, phi,
    STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
    min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
jax.block_until_ready(de_hard)

smooth_temps = [0.05, 0.2, 1.0]
smooth_colors = ['#4DAF4A', '#FF7F00', '#984EA3']
de_smooth = {}
for T in smooth_temps:
    _, de_s = generate_muon_segments_scan(
        jnp.float32(ENERGY), start, theta, phi,
        STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        min_energy_mev=MIN_ENERGY, smooth_temperature=T)
    jax.block_until_ready(de_s)
    de_smooth[T] = np.array(de_s)

de_hard_np = np.array(de_hard)
n_active = int(np.sum(de_hard_np > 1e-6))
dist_cm = np.arange(N_SEGMENTS) * STEP_SIZE_MM / 10.0

# ── Compute gradients w.r.t. energy for hard and smooth ──
def loss_hard(E):
    _, de = generate_muon_segments_scan(
        E, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        min_energy_mev=MIN_ENERGY, smooth_temperature=0.0)
    return jnp.sum(de)

def loss_smooth_02(E):
    _, de = generate_muon_segments_scan(
        E, start, theta, phi, STEP_SIZE_MM, N_SEGMENTS, log_T, dedx,
        min_energy_mev=MIN_ENERGY, smooth_temperature=0.2)
    return jnp.sum(de)

grad_hard_fn = jax.jit(jax.value_and_grad(loss_hard))
grad_smooth_fn = jax.jit(jax.value_and_grad(loss_smooth_02))

# Warm up
_ = grad_hard_fn(jnp.float32(ENERGY))
_ = grad_smooth_fn(jnp.float32(ENERGY))

E_sweep = np.linspace(100, 280, 80)
grad_hard_vals = np.empty(len(E_sweep))
grad_smooth_vals = np.empty(len(E_sweep))
loss_hard_vals = np.empty(len(E_sweep))
loss_smooth_vals = np.empty(len(E_sweep))

for i, E in enumerate(E_sweep):
    lh, gh = grad_hard_fn(jnp.float32(E))
    ls, gs = grad_smooth_fn(jnp.float32(E))
    grad_hard_vals[i] = float(gh)
    grad_smooth_vals[i] = float(gs)
    loss_hard_vals[i] = float(lh)
    loss_smooth_vals[i] = float(ls)

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# --- Panel 1: dE per segment near stopping ---
ax = axes[0]
zoom_lo = max(0, n_active - 150)
zoom_hi = n_active + 30
sl = slice(zoom_lo, zoom_hi)

ax.plot(dist_cm[sl], de_hard_np[sl], color='#2166AC', lw=2.5,
        label='Hard clamp ($T=0$)', zorder=5)
for T, color in zip(smooth_temps, smooth_colors):
    ax.plot(dist_cm[sl], de_smooth[T][sl], color=color, lw=1.8,
            alpha=0.85, label=f'Softplus $T={T}$')

ax.set_xlabel('Distance Along Track (cm)')
ax.set_ylabel('$\\Delta E$ (MeV / segment)')
ax.set_title('Stopping Boundary')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.7', fontsize=10)
ax.grid(True, alpha=0.25)

# --- Panel 2: total dE vs initial energy (forward) ---
ax = axes[1]
ax.plot(E_sweep, loss_hard_vals, color='#2166AC', lw=2.5,
        label='Hard clamp')
ax.plot(E_sweep, loss_smooth_vals, color='#FF7F00', lw=2, ls='--',
        label='Softplus $T=0.2$')
ax.set_xlabel('Initial Kinetic Energy (MeV)')
ax.set_ylabel('Total Energy Deposited (MeV)')
ax.set_title('Forward: $\\sum \\Delta E$ vs Energy')
ax.legend(framealpha=0.9, edgecolor='0.7')
ax.grid(True, alpha=0.25)

# --- Panel 3: gradient d(sum dE)/dE ---
ax = axes[2]
ax.plot(E_sweep, grad_hard_vals, color='#2166AC', lw=2.5,
        label='Hard clamp (AD)')
ax.plot(E_sweep, grad_smooth_vals, color='#FF7F00', lw=2,
        label='Softplus $T=0.2$ (AD)')
ax.axhline(1.0, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.text(280, 1.02, 'Ideal = 1.0', fontsize=10, color='gray',
        ha='right', va='bottom')

ax.set_xlabel('Initial Kinetic Energy (MeV)')
ax.set_ylabel('$\\partial (\\sum \\Delta E) / \\partial E_0$')
ax.set_title('Autodiff Gradient')
ax.legend(framealpha=0.9, edgecolor='0.7')
ax.grid(True, alpha=0.25)

# Annotate the hard-clamp gradient problem
ax.annotate('Hard clamp:\nnon-smooth gradient',
            xy=(130, grad_hard_vals[np.argmin(np.abs(E_sweep - 130))]),
            fontsize=10, color='#2166AC',
            xytext=(140, 0.4),
            arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.2))

fig.tight_layout(w_pad=2.5)

out = os.path.join(os.path.dirname(__file__), 'slide3_smoothing_problem.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
