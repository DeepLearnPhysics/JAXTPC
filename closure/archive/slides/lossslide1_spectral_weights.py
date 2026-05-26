"""Lossslide 1: Spectral weights in frequency domain — MSE vs Blur MSE vs Sobolev."""
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
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'lines.linewidth': 2.5,
})

# ---------------------------------------------------------------------------
# Build 1D radial spectral weights for N = 4096 (matches padded size)
# ---------------------------------------------------------------------------
N = 4096
freqs = np.fft.fftfreq(N)
# Take positive half only for plotting
pos = freqs[:N // 2]

# --- MSE: flat weight = 1 ---
w_mse = np.ones_like(pos)

# --- Blur MSE: sum of Gaussian weights ---
sigmas = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
w_blur = np.zeros_like(pos)
for s in sigmas:
    if s == 0:
        w_blur += 1.0
    else:
        w_blur += s**2 * np.exp(-4 * np.pi**2 * s**2 * pos**2)

# --- Sobolev s=1.5 ---
max_pad = 1024
eps = 1.0 / (np.pi**2 * max_pad**2)
w_sob15 = 1.0 / (pos**2 + eps)**1.5

# --- Sobolev s=2.0 ---
w_sob20 = 1.0 / (pos**2 + eps)**2.0

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(pos, w_mse, color='0.5', lw=2, ls='--', label='MSE  ($W = 1$)', zorder=2)
ax.loglog(pos, w_blur, color='#4393C3', lw=2.5, label='Gaussian Pyramid', zorder=3)
ax.loglog(pos, w_sob15, color='#D6604D', lw=2.5, label=r'Sobolev $H^{-1.5}$', zorder=4)
ax.loglog(pos, w_sob20, color='#B2182B', lw=2.5, ls='-.', label=r'Sobolev $H^{-2}$', zorder=4)

ax.set_xlabel('Frequency  $|f|$  (cycles / pixel)')
ax.set_ylabel('Spectral Weight  $W(f)$')
ax.set_title('Frequency-Domain Loss Weights')
ax.set_xlim(pos[1], pos[-1])
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, which='major', ls='-', alpha=0.25)
ax.grid(True, which='minor', ls=':', alpha=0.12)

# Annotate low-frequency amplification
ax.annotate('Low-freq amplification\n(large-scale structure)',
            xy=(3e-3, 1e8), fontsize=12, color='#B2182B',
            ha='center', style='italic')

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'lossslide1_spectral_weights.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
