"""Slide 1: Muon dE/dx (Bethe-Bloch) curve and CSDA range in liquid argon."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Professional style ──
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'lines.linewidth': 2.0,
})

LAR_DENSITY = 1.396  # g/cm^3

# ── Load data ──
data = np.loadtxt('tools/data/muon_dedx_lar.csv', delimiter=',', comments='#')
T_MeV = data[:, 0]
dedx_MeVcm2g = data[:, 2]
dedx_MeVcm = dedx_MeVcm2g * LAR_DENSITY
R_gcm2 = data[:, 3]
R_cm = R_gcm2 / LAR_DENSITY

# MIP location
mip_idx = np.argmin(dedx_MeVcm2g)
T_mip = T_MeV[mip_idx]
dedx_mip = dedx_MeVcm[mip_idx]

# ── Figure: two panels ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left: dE/dx vs T ---
ax1.loglog(T_MeV, dedx_MeVcm, color='#2166AC', lw=2.5, zorder=3)
ax1.axhline(dedx_mip, color='#B2182B', ls='--', lw=1.5, alpha=0.7, zorder=2)
ax1.plot(T_mip, dedx_mip, 'o', color='#B2182B', ms=8, zorder=4,
         label=f'MIP: {dedx_mip:.2f} MeV/cm at {T_mip:.0f} MeV')

# Shade unreliable region
ax1.axvspan(T_MeV[0], 10, color='gray', alpha=0.12, zorder=1)
ax1.text(3, 25, 'PDG\nunreliable', fontsize=10, color='gray',
         ha='center', va='top', style='italic')

# Bragg peak annotation
ax1.annotate('Bragg peak\n(rising dE/dx)', xy=(5, 12), fontsize=11,
             color='#2166AC', ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.2),
             xytext=(20, 25))

# Relativistic rise annotation
ax1.annotate('Relativistic\nrise', xy=(5e4, dedx_MeVcm[-3]),
             fontsize=11, color='#2166AC', ha='center',
             xytext=(1e4, 10),
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.2))

ax1.set_xlabel('Kinetic Energy $T$ (MeV)')
ax1.set_ylabel('Stopping Power  $-dE/dx$  (MeV/cm)')
ax1.set_title('Muon Stopping Power in LAr')
ax1.legend(loc='upper right', framealpha=0.9, edgecolor='0.7')
ax1.set_xlim(T_MeV[0], T_MeV[-1])
ax1.set_ylim(1.5, 45)
ax1.grid(True, which='major', ls='-', alpha=0.25)
ax1.grid(True, which='minor', ls=':', alpha=0.12)

# --- Right: CSDA range vs T ---
ax2.loglog(T_MeV, R_cm, color='#D6604D', lw=2.5, zorder=3)

# Mark a few reference energies
ref_energies = [50, 200, 500, 2000]
for E_ref in ref_energies:
    idx = np.argmin(np.abs(T_MeV - E_ref))
    R_ref = R_cm[idx]
    ax2.plot(T_MeV[idx], R_ref, 's', color='#D6604D', ms=6, zorder=4)
    if E_ref == 200:
        ax2.annotate(f'{E_ref} MeV\nR = {R_ref:.0f} cm',
                     xy=(T_MeV[idx], R_ref), fontsize=10,
                     ha='left', va='top',
                     xytext=(T_MeV[idx]*1.8, R_ref*0.4),
                     arrowprops=dict(arrowstyle='->', color='0.4', lw=1.0))
    elif E_ref == 500:
        ax2.annotate(f'{E_ref} MeV\nR = {R_ref:.0f} cm',
                     xy=(T_MeV[idx], R_ref), fontsize=10,
                     ha='left', va='bottom',
                     xytext=(T_MeV[idx]*1.5, R_ref*2),
                     arrowprops=dict(arrowstyle='->', color='0.4', lw=1.0))

ax2.set_xlabel('Kinetic Energy $T$ (MeV)')
ax2.set_ylabel('CSDA Range $R$ (cm)')
ax2.set_title('CSDA Range in LAr')
ax2.set_xlim(T_MeV[0], T_MeV[-1])
ax2.grid(True, which='major', ls='-', alpha=0.25)
ax2.grid(True, which='minor', ls=':', alpha=0.12)

fig.tight_layout(w_pad=3)

out = os.path.join(os.path.dirname(__file__), 'slide1_dedx_and_range.png')
fig.savefig(out)
plt.close(fig)
print(f'Saved {out}')
