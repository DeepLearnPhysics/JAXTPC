"""Lossslide 2: 2D spatial-domain kernels — IFFT of spectral weights."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

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
# Build 2D spectral weights, IFFT to get spatial kernels
# ---------------------------------------------------------------------------
N = 1024
max_pad = 512
eps = 1.0 / (np.pi**2 * max_pad**2)

fy = np.fft.fftfreq(N)
fx = np.fft.fftfreq(N)
freq_sq = fy[:, None]**2 + fx[None, :]**2

# Gaussian Pyramid
sigmas = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
W_blur = np.zeros((N, N))
for s in sigmas:
    if s == 0:
        W_blur += 1.0
    else:
        W_blur += s**2 * np.exp(-4 * np.pi**2 * s**2 * freq_sq)

# Sobolev
W_sob15 = 1.0 / (freq_sq + eps)**1.5
W_sob20 = 1.0 / (freq_sq + eps)**2.0

# K_hat = sqrt(W), then IFFT for spatial kernel
K_blur = np.fft.fftshift(np.fft.ifft2(np.sqrt(W_blur)).real)
K_sob15 = np.fft.fftshift(np.fft.ifft2(np.sqrt(W_sob15)).real)
K_sob20 = np.fft.fftshift(np.fft.ifft2(np.sqrt(W_sob20)).real)

# Normalize to peak=1
K_blur /= K_blur.max()
K_sob15 /= K_sob15.max()
K_sob20 /= K_sob20.max()

# Crop to +/- crop_r pixels around center
crop_r = 250
c = N // 2
sl = slice(c - crop_r, c + crop_r)
extent = [-crop_r, crop_r, -crop_r, crop_r]

K_blur_c = K_blur[sl, sl]
K_sob15_c = K_sob15[sl, sl]
K_sob20_c = K_sob20[sl, sl]

# ---------------------------------------------------------------------------
# Plot: 3 panels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

vmin = 1e-3
titles = ['Gaussian Pyramid', r'Sobolev  $s = 1.5$', r'Sobolev  $s = 2$']
kernels = [K_blur_c, K_sob15_c, K_sob20_c]

for ax, K, title in zip(axes, kernels, titles):
    K_pos = np.clip(K, vmin, None)
    im = ax.imshow(K_pos, cmap='inferno', origin='lower', extent=extent,
                   norm=LogNorm(vmin=vmin, vmax=1.0))
    ax.set_title(title)
    ax.set_xlabel('$\\Delta x$  (pixels)')
    ax.set_ylabel('$\\Delta y$  (pixels)')
    fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

fig.suptitle('Spatial Convolution Kernels  $K(\\Delta x, \\Delta y)$',
             fontsize=20, fontweight='bold', y=1.02)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'lossslide2_spatial_kernels.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
