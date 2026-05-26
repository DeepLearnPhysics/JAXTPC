"""Slide 3: CSDA range curve — the key concept enabling parallelization."""
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
R_cm = data[:, 3] / LAR_DENSITY

fig, ax = plt.subplots(figsize=(9, 6))

ax.loglog(T_MeV, R_cm, color='#D6604D', lw=2.5, zorder=3)

# Annotate a few reference energies
refs = [(200, 'left'), (500, 'left'), (1000, 'right')]
for E_ref, ha in refs:
    idx = np.argmin(np.abs(T_MeV - E_ref))
    R_ref = R_cm[idx]
    ax.plot(T_MeV[idx], R_ref, 's', color='#D6604D', ms=7, zorder=4)

    if E_ref == 200:
        ax.annotate(f'{E_ref} MeV,  $R$ = {R_ref:.0f} cm',
                    xy=(T_MeV[idx], R_ref), fontsize=13,
                    xytext=(8, R_ref * 4), ha='left',
                    arrowprops=dict(arrowstyle='->', color='0.35', lw=1.2))
    elif E_ref == 500:
        ax.annotate(f'{E_ref} MeV,  $R$ = {R_ref:.0f} cm',
                    xy=(T_MeV[idx], R_ref), fontsize=13,
                    xytext=(T_MeV[idx] * 3, R_ref * 0.15), ha='left',
                    arrowprops=dict(arrowstyle='->', color='0.35', lw=1.2))
    elif E_ref == 1000:
        ax.annotate(f'{E_ref} MeV,  $R$ = {R_ref:.0f} cm',
                    xy=(T_MeV[idx], R_ref), fontsize=13,
                    xytext=(T_MeV[idx] * 3, R_ref * 0.35), ha='left',
                    arrowprops=dict(arrowstyle='->', color='0.35', lw=1.2))

ax.set_xlabel('Kinetic Energy $T$ (MeV)')
ax.set_ylabel('CSDA Range $R(T)$ (cm)')
ax.set_title('CSDA Range in Liquid Argon')
ax.set_xlim(T_MeV[0], T_MeV[-1])
ax.grid(True, which='major', ls='-', alpha=0.25)
ax.grid(True, which='minor', ls=':', alpha=0.12)

# Add the key formula
ax.text(0.03, 0.95,
        r'$R(T) = \int_0^T \frac{1}{dE/dx}\, dT^\prime$',
        transform=ax.transAxes, fontsize=17,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='0.7', alpha=0.9))

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'slide3_csda_range.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
