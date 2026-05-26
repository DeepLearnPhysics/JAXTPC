"""Slide 1: Muon dE/dx (Bethe-Bloch) curve in liquid argon."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'lines.linewidth': 2.5,
})

LAR_DENSITY = 1.396

data = np.loadtxt('tools/data/muon_dedx_lar.csv', delimiter=',', comments='#')
T_MeV = data[:, 0]
dedx_MeVcm = data[:, 2] * LAR_DENSITY

mip_idx = np.argmin(data[:, 2])
T_mip = T_MeV[mip_idx]
dedx_mip = dedx_MeVcm[mip_idx]

fig, ax = plt.subplots(figsize=(9, 6))

ax.loglog(T_MeV, dedx_MeVcm, color='#2166AC', lw=2.5, zorder=3)

# MIP point
ax.axhline(dedx_mip, color='#B2182B', ls='--', lw=1.5, alpha=0.5, zorder=2)
ax.plot(T_mip, dedx_mip, 'o', color='#B2182B', ms=9, zorder=4)
ax.annotate(f'MIP: {dedx_mip:.2f} MeV/cm\nat $T$ = {T_mip:.0f} MeV',
            xy=(T_mip, dedx_mip), fontsize=13,
            xytext=(3000, 3.5),
            arrowprops=dict(arrowstyle='->', color='#B2182B', lw=1.3),
            color='#B2182B')

# Shade unreliable region
ax.axvspan(T_MeV[0], 10, color='gray', alpha=0.1, zorder=1)
ax.text(3.5, 30, 'PDG\nunreliable', fontsize=12, color='0.5',
        ha='center', va='top', style='italic')

ax.set_xlabel('Kinetic Energy $T$ (MeV)')
ax.set_ylabel('Stopping Power  $-dE/dx$  (MeV/cm)')
ax.set_title('Muon Stopping Power in Liquid Argon')
ax.set_xlim(T_MeV[0], T_MeV[-1])
ax.set_ylim(1.5, 45)
ax.grid(True, which='major', ls='-', alpha=0.25)
ax.grid(True, which='minor', ls=':', alpha=0.12)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'slide1_dedx_curve.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
