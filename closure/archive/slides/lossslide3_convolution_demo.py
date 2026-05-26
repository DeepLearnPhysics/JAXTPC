"""Lossslide 3: Convolution demo — a displaced 2D Gaussian before/after Sobolev kernel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# ---------------------------------------------------------------------------
# Create a simple 2D test case: Gaussian blob, truth vs displaced
# ---------------------------------------------------------------------------
H, W = 256, 256
y, x = np.mgrid[:H, :W]

# Truth: Gaussian at center
cx_t, cy_t = 128, 128
sig = 12.0
truth = np.exp(-((x - cx_t)**2 + (y - cy_t)**2) / (2 * sig**2))

# Sim: same blob displaced by 60 pixels
shift = 60
cx_s, cy_s = cx_t + shift, cy_t + int(shift * 0.5)
sim = np.exp(-((x - cx_s)**2 + (y - cy_s)**2) / (2 * sig**2))

diff = sim - truth

# ---------------------------------------------------------------------------
# Build Sobolev spectral weights and convolve the difference
# ---------------------------------------------------------------------------
max_pad = 128
H_pad, W_pad = H + 2 * max_pad, W + 2 * max_pad
fy = np.fft.fftfreq(H_pad)
fx = np.fft.fftfreq(W_pad)
freq_sq = fy[:, None]**2 + fx[None, :]**2
eps = 1.0 / (np.pi**2 * max_pad**2)

# K_hat = sqrt(W) = 1/(freq_sq + eps)^{s/2}
def convolve_sobolev(diff_2d, s, max_pad):
    H, W = diff_2d.shape
    H_pad, W_pad = H + 2 * max_pad, W + 2 * max_pad
    fy = np.fft.fftfreq(H_pad)
    fx = np.fft.fftfreq(W_pad)
    freq_sq = fy[:, None]**2 + fx[None, :]**2
    eps = 1.0 / (np.pi**2 * max_pad**2)
    K_hat = 1.0 / (freq_sq + eps)**(s / 2.0)
    diff_pad = np.pad(diff_2d, max_pad)
    convolved_pad = np.fft.ifft2(np.fft.fft2(diff_pad) * K_hat).real
    return convolved_pad[max_pad:max_pad + H, max_pad:max_pad + W]


conv_15 = convolve_sobolev(diff, s=1.5, max_pad=max_pad)
conv_20 = convolve_sobolev(diff, s=2.0, max_pad=max_pad)

# ---------------------------------------------------------------------------
# Plot: 2 rows x 3 cols
# Row 1: Truth, Sim, Difference
# Row 2: (empty), Convolved s=1.5, Convolved s=2.0
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Row 1
im0 = axes[0, 0].imshow(truth, cmap='inferno', origin='lower', vmin=0, vmax=1)
axes[0, 0].set_title('Truth Signal')

im1 = axes[0, 1].imshow(sim, cmap='inferno', origin='lower', vmin=0, vmax=1)
axes[0, 1].set_title('Simulated Signal (shifted)')

vmax_d = np.max(np.abs(diff))
norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
im2 = axes[0, 2].imshow(diff, cmap='RdBu_r', origin='lower', norm=norm_d)
axes[0, 2].set_title('Difference (MSE input)')
fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

# Row 2
axes[1, 0].axis('off')

vmax_15 = np.max(np.abs(conv_15))
norm_15 = TwoSlopeNorm(vmin=-vmax_15, vcenter=0, vmax=vmax_15)
im3 = axes[1, 1].imshow(conv_15, cmap='RdBu_r', origin='lower', norm=norm_15)
axes[1, 1].set_title(r'$K_{1.5} \ast$ Difference  ($s=1.5$)')
fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)

vmax_20 = np.max(np.abs(conv_20))
norm_20 = TwoSlopeNorm(vmin=-vmax_20, vcenter=0, vmax=vmax_20)
im4 = axes[1, 2].imshow(conv_20, cmap='RdBu_r', origin='lower', norm=norm_20)
axes[1, 2].set_title(r'$K_{2} \ast$ Difference  ($s=2$)')
fig.colorbar(im4, ax=axes[1, 2], shrink=0.8)

for ax in axes.flat:
    if ax.get_visible() and ax.images:
        ax.set_xlabel('Wire')
        ax.set_ylabel('Time')

fig.suptitle('Sobolev Convolution: Displaced Gaussian Blob',
             fontsize=20, fontweight='bold', y=1.01)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'lossslide3_convolution_demo.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
