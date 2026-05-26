"""Slide 6: Hard clamp vs softplus — per-segment gradient w.r.t. energy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
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
    'lines.linewidth': 2.0,
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
R_cm = jnp.array(R_cm)
T_MeV = jnp.array(T_MeV)

def csda_hard(energy):
    indices = jnp.arange(N_SEGMENTS)
    log_T_csda = jnp.log(T_MeV)
    R_initial = jnp.interp(jnp.log(energy), log_T_csda, R_cm)
    R_at_start = R_initial - indices * STEP_SIZE_CM
    R_at_end = R_initial - (indices + 1) * STEP_SIZE_CM
    R_floor = R_cm[0]
    E_start = jnp.interp(jnp.maximum(R_at_start, R_floor), R_cm, T_MeV)
    E_end = jnp.interp(jnp.maximum(R_at_end, R_floor), R_cm, T_MeV)
    return jnp.maximum(E_start - E_end, 0.0)

relax_val = STEP_SIZE_CM * 2.0
def csda_soft(energy):
    indices = jnp.arange(N_SEGMENTS)
    log_T_csda = jnp.log(T_MeV)
    R_initial = jnp.interp(jnp.log(energy), log_T_csda, R_cm)
    R_at_start = R_initial - indices * STEP_SIZE_CM
    R_at_end = R_initial - (indices + 1) * STEP_SIZE_CM
    R_floor = R_cm[0]
    R_s = R_floor + jax.nn.softplus((R_at_start - R_floor) / relax_val) * relax_val
    R_e = R_floor + jax.nn.softplus((R_at_end - R_floor) / relax_val) * relax_val
    E_start = jnp.interp(R_s, R_cm, T_MeV)
    E_end = jnp.interp(R_e, R_cm, T_MeV)
    return jnp.maximum(E_start - E_end, 0.0)

# Jacobian: d(dE_i)/d(E0) for each segment i
jac_hard_fn = jax.jit(jax.jacfwd(csda_hard))
jac_soft_fn = jax.jit(jax.jacfwd(csda_soft))

E0 = jnp.float32(ENERGY)
jac_hard = np.array(jac_hard_fn(E0))
jac_soft = np.array(jac_soft_fn(E0))

de_hard = np.array(csda_hard(E0))
de_soft = np.array(csda_soft(E0))

n_active = int(np.sum(de_hard > 1e-6))
dist_cm = np.arange(N_SEGMENTS) * STEP_SIZE_CM

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

zoom_lo = max(0, n_active - 120)
zoom_hi = n_active + 30
sl = slice(zoom_lo, zoom_hi)

# Left: dE per segment near stopping
ax1.plot(dist_cm[sl], de_hard[sl], color='0.3', lw=2.5, label='Hard clamp')
ax1.plot(dist_cm[sl], de_soft[sl], color='#D6604D', lw=2, ls='--',
         label='Softplus (relax = 2x)')
ax1.set_xlabel('Distance Along Track (cm)')
ax1.set_ylabel('dE per segment (MeV)')
ax1.set_title('Energy Deposits Near Stopping')
ax1.legend(framealpha=0.9, edgecolor='0.7')
ax1.grid(True, alpha=0.25)

# Right: per-segment gradient d(dE_i)/d(E0)
ax2.plot(dist_cm[sl], jac_hard[sl], color='0.3', lw=2.5, label='Hard clamp')
ax2.plot(dist_cm[sl], jac_soft[sl], color='#D6604D', lw=2, label='Softplus (relax = 2x)')
ax2.set_xlabel('Distance Along Track (cm)')
ax2.set_ylabel('d(dE_i) / d(E0)')
ax2.set_title('Per-Segment Gradient w.r.t. Energy')
ax2.legend(framealpha=0.9, edgecolor='0.7')
ax2.grid(True, alpha=0.25)
ax2.axhline(0, color='k', lw=0.5)

fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.92, wspace=0.25)
out = os.path.join(os.path.dirname(__file__), 'slide6_gradient_fix.png')
fig.savefig(out, dpi=200)
plt.close(fig)
print(f'Saved {out}')
