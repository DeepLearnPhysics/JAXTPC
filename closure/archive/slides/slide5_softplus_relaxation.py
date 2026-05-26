"""Slide 5: Softplus relaxation — the smooth clamp and its effect on the Bragg peak."""
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
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'lines.linewidth': 2.2,
})

from closure_analysis_muon.diff_muon_generator import (
    load_dedx_table_jax,
    build_consistent_csda_table,
)

STEP_SIZE_MM = 0.5
STEP_SIZE_CM = STEP_SIZE_MM / 10.0
N_SEGMENTS = 4000
ENERGY = 200.0

log_T, dedx = load_dedx_table_jax()
R_cm, T_MeV = build_consistent_csda_table(log_T, dedx, n_points=10000)
R_cm_j = jnp.array(R_cm)
T_MeV_j = jnp.array(T_MeV)
E = jnp.float32(ENERGY)

# ── Hard clamp CSDA baseline (no softplus) ──
def csda_hard(energy):
    indices = jnp.arange(N_SEGMENTS)
    R_initial = jnp.interp(jnp.log(energy), jnp.log(T_MeV_j), R_cm_j)
    R_at_start = R_initial - indices * STEP_SIZE_CM
    R_at_end = R_initial - (indices + 1) * STEP_SIZE_CM
    R_floor = R_cm_j[0]
    E_start = jnp.interp(jnp.maximum(R_at_start, R_floor), R_cm_j, T_MeV_j)
    E_end = jnp.interp(jnp.maximum(R_at_end, R_floor), R_cm_j, T_MeV_j)
    return jnp.maximum(E_start - E_end, 0.0)

def csda_soft(energy, relax_steps):
    indices = jnp.arange(N_SEGMENTS)
    relax = STEP_SIZE_CM * relax_steps
    R_initial = jnp.interp(jnp.log(energy), jnp.log(T_MeV_j), R_cm_j)
    R_at_start = R_initial - indices * STEP_SIZE_CM
    R_at_end = R_initial - (indices + 1) * STEP_SIZE_CM
    R_floor = R_cm_j[0]
    R_s = R_floor + jax.nn.softplus((R_at_start - R_floor) / relax) * relax
    R_e = R_floor + jax.nn.softplus((R_at_end - R_floor) / relax) * relax
    E_start = jnp.interp(R_s, R_cm_j, T_MeV_j)
    E_end = jnp.interp(R_e, R_cm_j, T_MeV_j)
    return jnp.maximum(E_start - E_end, 0.0)

de_hard_np = np.array(csda_hard(E))
n_active = int(np.sum(de_hard_np > 1e-6))
dist_cm = np.arange(N_SEGMENTS) * STEP_SIZE_CM

relax_vals = [1.0, 2.0, 5.0, 10.0]
relax_colors = ['#4DAF4A', '#2166AC', '#FF7F00', '#984EA3']
de_relax = {}
for rv in relax_vals:
    de_relax[rv] = np.array(csda_soft(E, rv))

# ── Figure: two panels ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left: softplus vs max illustration ---
x = np.linspace(-4, 4, 500)
ax1.plot(x, np.maximum(x, 0), color='0.3', lw=2.5, label='$\\max(x, 0)$')
for relax_w, color, label in [(0.5, '#4DAF4A', '$\\lambda = 0.5$'),
                                (1.0, '#2166AC', '$\\lambda = 1.0$'),
                                (2.0, '#FF7F00', '$\\lambda = 2.0$')]:
    sp = relax_w * np.log(1 + np.exp(x / relax_w))
    ax1.plot(x, sp, color=color, lw=2, label=f'softplus, {label}')

ax1.set_xlabel('$x$')
ax1.set_ylabel('$f(x)$')
ax1.set_title('Softplus Approximation')
ax1.legend(framealpha=0.9, edgecolor='0.7', fontsize=12)
ax1.grid(True, alpha=0.25)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-0.3, 4.5)

ax1.text(0.03, 0.97,
         r'$\mathrm{softplus}_\lambda(x) = \lambda \ln(1 + e^{x/\lambda})$',
         transform=ax1.transAxes, fontsize=14,
         va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='0.7', alpha=0.9))

# --- Right: Bragg peak region for different relax values ---
zoom_lo = max(0, n_active - 120)
zoom_hi = n_active + 40
sl = slice(zoom_lo, zoom_hi)

ax2.plot(dist_cm[sl], de_hard_np[sl], color='0.3', lw=2.5,
         label='Hard clamp (max)', zorder=5)
for rv, color in zip(relax_vals, relax_colors):
    is_default = (rv == 2.0)
    lw = 2.5 if is_default else 1.5
    suffix = ' (default)' if is_default else ''
    ax2.plot(dist_cm[sl], de_relax[rv][sl], color=color, lw=lw,
             label=f'relax = {rv:.0f}$\\times$ step{suffix}',
             zorder=4 if is_default else 3)

ax2.set_xlabel('Distance Along Track (cm)')
ax2.set_ylabel('$\\Delta E$ per Segment (MeV)')
ax2.set_title('Effect on Bragg Peak')
ax2.legend(framealpha=0.9, edgecolor='0.7', fontsize=11)
ax2.grid(True, alpha=0.25)

fig.tight_layout(w_pad=3)
out = os.path.join(os.path.dirname(__file__), 'slide5_softplus_relaxation.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
