"""
Multi-scale spectral blur MSE loss for wireplane signal comparison.

All blur scales are combined into a single precomputed spectral weight via
Parseval's theorem: sum_s sigma_s^2 * ||G_s * D||^2 = (1/N) sum_f |D^|^2 W(f),
where W(f) = sum_s sigma_s^2 * |G^_s(f)|^2. One FFT per plane, O(N log N).

Usage:
    from tools.losses import blur_mse_loss, make_spectral_weight, DEFAULT_BLUR_SIGMAS

    # Precompute spectral weight once per plane shape (call before JIT)
    sw = tuple(make_spectral_weight(H, W, DEFAULT_BLUR_SIGMAS)
               for H, W in plane_shapes)

    def loss_fn(params):
        sigs = forward(SegmentData(...))
        return blur_mse_loss(sigs, target_signals, sw)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    loss, grads = grad_fn(init_params)
"""

import math
from functools import partial

import jax
import jax.numpy as jnp


# Full-resolution blur sigmas: from pixel-exact to detector-scale.
# sigma=0 means raw MSE (no blur).
# Reach in pixels = 4*sigma in each direction.
# At 0.8 mm/time-bin: sigma=256 -> reach=1024 bins = 820mm.
# At 3 mm/wire: sigma=256 -> reach=1024 wires = 3072mm.
DEFAULT_BLUR_SIGMAS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def make_spectral_weight(H, W, sigmas):
    """Precompute the combined spectral weight for all blur scales.

    Combines all Gaussian blur scales into one frequency-domain weight
    via Parseval's theorem. Call once per unique (H, W) shape, reuse
    across forward/backward passes.

    Parameters
    ----------
    H, W : int
        Signal dimensions (wires, time bins).
    sigmas : tuple of float
        Blur sigma values. sigma=0 contributes a flat weight of 1
        (raw MSE via Parseval).

    Returns
    -------
    spectral_weight : jnp.ndarray, shape (H_pad, W_pad)
        Frequency-domain weight array.
    """
    max_sigma = max(sigmas)
    max_pad = math.ceil(4.0 * max_sigma) if max_sigma > 0 else 0

    H_pad = H + 2 * max_pad
    W_pad = W + 2 * max_pad
    fy = jnp.fft.fftfreq(H_pad)
    fx = jnp.fft.fftfreq(W_pad)
    freq_sq = fy[:, None] ** 2 + fx[None, :] ** 2

    spectral_weight = jnp.zeros((H_pad, W_pad), dtype=jnp.float32)
    for s in sigmas:
        if s == 0:
            # Delta kernel: |G_hat|^2 = 1, weight = 1 (raw MSE)
            spectral_weight = spectral_weight + 1.0
        else:
            spectral_weight = spectral_weight + s ** 2 * jnp.exp(
                -4 * math.pi ** 2 * s ** 2 * freq_sq
            )
    return spectral_weight


def blur_mse_loss_single(A, B, spectral_weight):
    """Multi-scale spectral blur MSE loss for a single plane.

    Single FFT implementation via Parseval's theorem. Both A and B are
    normalized by sum(|B|) before computation, making the loss
    dimensionless and O(1).

    Parameters
    ----------
    A, B : jnp.ndarray
        2D arrays of shape (H, W), the simulated and target signals.
    spectral_weight : jnp.ndarray
        Precomputed spectral weight from make_spectral_weight(),
        shape (H_pad, W_pad). Padding is inferred from shape difference.

    Returns
    -------
    jnp.ndarray
        Scalar loss.
    """
    # Infer padding from shape difference (known at trace time)
    pad_h = (spectral_weight.shape[0] - A.shape[0]) // 2
    pad_w = (spectral_weight.shape[1] - A.shape[1]) // 2

    norm = jnp.sum(jnp.abs(B)) + 1e-12
    diff = (A - B) / norm
    diff_pad = jnp.pad(diff, ((pad_h, pad_h), (pad_w, pad_w)))
    diff_fft = jnp.fft.fft2(diff_pad)
    power = diff_fft.real ** 2 + diff_fft.imag ** 2
    N = diff_pad.shape[0] * diff_pad.shape[1]
    return jnp.sum(power * spectral_weight) / N


@partial(jax.jit, static_argnums=(3,))
def blur_mse_loss(signals_a, signals_b, spectral_weights, planes=(0, 1, 2, 3, 4, 5)):
    """Multi-scale spectral blur MSE loss summed over wire planes.

    Single FFT per plane via Parseval's theorem — all blur scales
    combined into precomputed spectral weights.

    Parameters
    ----------
    signals_a, signals_b : tuple of jnp.ndarray
        Tuples of 6 arrays (one per plane) from forward().
        Each array has shape (W_i, T_i). Traced (differentiable).
    spectral_weights : tuple of jnp.ndarray
        Per-plane spectral weight arrays from make_spectral_weight().
        Shape (H_pad, W_pad) for each plane — padding inferred from
        shape difference with corresponding signal.
    planes : tuple of int
        Which plane indices to include (default all 6). Static.

    Returns
    -------
    jnp.ndarray
        Scalar total loss.
    """
    loss = 0.0
    for p in planes:
        loss = loss + blur_mse_loss_single(
            signals_a[p], signals_b[p], spectral_weights[p],
        )
    return loss


# ---------------------------------------------------------------------------
# Sobolev (screened Poisson) spectral weight — drop-in replacement
# ---------------------------------------------------------------------------
#
# Replaces the sum-of-Gaussians weight with W(f) = 1/(|f|^2 + eps),
# the regularised H^{-1} Sobolev norm.  Equivalent to the Wasserstein-2
# distance for small perturbations (Peyre 2018).
#
# eps is set so the screening length L = max_pad/2, ensuring the spatial
# kernel decays to ~2% at the periodic boundary (2*max_pad away).
#
#   L = 1/(2*pi*sqrt(eps))  =>  eps = 1/(pi^2 * max_pad^2)
#
# Gradient behaviour vs distance d from target:
#   d < L  (~512 px):  constant magnitude  (pure H^{-1} / Wasserstein)
#   d > L:             decays as exp(-d/L)  (still directionally correct)
#
# With max_pad=1024 (same as current Gaussian setup):
#   L = 512 px, strong gradients to ~500 px, useful to ~1500 px.
#   Ghost contamination exp(-4) ~ 2%.


def make_sobolev_weight(H, W, max_pad=1024, s=2.0):
    """H^{-s} Sobolev spectral weight.

    s=1: log(d) loss growth, 1/d gradient (Laplacian)
    s=3/2: |d| loss growth, constant gradient (W1-like)
    s=2: d^2 loss growth, linear gradient (W2^2-like)

    Parameters
    ----------
    H, W : int
        Signal dimensions (wires, time bins).
    max_pad : int
        Zero-padding per side. Screening length L = max_pad/2.
        Default 1024 matches the current Gaussian setup (4*256).
    s : float
        Sobolev exponent. Default 2.0.

    Returns
    -------
    spectral_weight : jnp.ndarray, shape (H + 2*max_pad, W + 2*max_pad)
        Frequency-domain weight array.
    """
    H_pad = H + 2 * max_pad
    W_pad = W + 2 * max_pad
    fy = jnp.fft.fftfreq(H_pad)
    fx = jnp.fft.fftfreq(W_pad)
    freq_sq = fy[:, None] ** 2 + fx[None, :] ** 2

    eps = 1.0 / (math.pi ** 2 * max_pad ** 2)

    return 1 / (freq_sq + eps) ** s


def sobolev_loss_single(A, B, spectral_weight):
    """Sobolev H^{-1} loss for a single plane.

    Same structure as blur_mse_loss_single: infers padding from the
    spectral weight shape, single FFT, Parseval summation.

    Parameters
    ----------
    A, B : jnp.ndarray
        2D arrays of shape (H, W), the simulated and target signals.
    spectral_weight : jnp.ndarray
        Precomputed weight from make_sobolev_weight().

    Returns
    -------
    jnp.ndarray
        Scalar loss.
    """
    pad_h = (spectral_weight.shape[0] - A.shape[0]) // 2
    pad_w = (spectral_weight.shape[1] - A.shape[1]) // 2

    norm = jnp.sum(jnp.abs(B)) + 1e-12
    diff = (A - B) / norm
    diff_pad = jnp.pad(diff, ((pad_h, pad_h), (pad_w, pad_w)))
    diff_fft = jnp.fft.fft2(diff_pad)
    power = diff_fft.real ** 2 + diff_fft.imag ** 2
    N = diff_pad.shape[0] * diff_pad.shape[1]
    return jnp.sum(power * spectral_weight) / N


@partial(jax.jit, static_argnums=(3,))
def sobolev_loss(signals_a, signals_b, spectral_weights, planes=(0, 1, 2, 3, 4, 5)):
    """Sobolev H^{-1} loss summed over wire planes.

    Drop-in replacement for blur_mse_loss().

    Parameters
    ----------
    signals_a, signals_b : tuple of jnp.ndarray
        Tuples of 6 arrays (one per plane).
    spectral_weights : tuple of jnp.ndarray
        Per-plane weights from make_sobolev_weight().
    planes : tuple of int
        Which plane indices to include (default all 6). Static.

    Returns
    -------
    jnp.ndarray
        Scalar total loss.
    """
    loss = 0.0
    for p in planes:
        loss = loss + sobolev_loss_single(
            signals_a[p], signals_b[p], spectral_weights[p],
        )
    return loss


@partial(jax.jit, static_argnums=(3,))
def sobolev_loss_geomean(signals_a, signals_b, spectral_weights,
                         planes=(0, 1, 2, 3, 4, 5), eps=1.0):
    """Sobolev loss with geometric mean over planes.

    Geometric mean automatically rebalances plane contributions:
    each plane's gradient is weighted by 1/(L_plane + eps),
    so dominant planes get downweighted.

    Uses logsumexp-style computation for numerical stability.

    Parameters
    ----------
    signals_a, signals_b : tuple of jnp.ndarray
        Tuples of 6 arrays (one per plane).
    spectral_weights : tuple of jnp.ndarray
        Per-plane weights from make_sobolev_weight().
    planes : tuple of int
        Which plane indices to include (default all 6). Static.
    eps : float
        Floor to prevent log(0). Should be near the converged loss scale.

    Returns
    -------
    jnp.ndarray
        Scalar geometric-mean loss.
    """
    n = len(planes)
    log_sum = 0.0
    for p in planes:
        lp = sobolev_loss_single(signals_a[p], signals_b[p], spectral_weights[p])
        log_sum = log_sum + jnp.log(lp + eps)
    return jnp.exp(log_sum / n)


@partial(jax.jit, static_argnums=(3,))
def sobolev_loss_geomean_log1p(signals_a, signals_b, spectral_weights,
                                planes=(0, 1, 2, 3, 4, 5)):
    """Sobolev loss with log1p/expm1 geometric mean over planes.

    Parameter-free variant of sobolev_loss_geomean. Uses the
    Kolmogorov-Nagumo quasi-arithmetic mean with generator log1p:

        loss = expm1( mean( log1p(L_p) ) )
             = prod(1 + L_p)^(1/n) - 1

    For L_p >> 1: behaves as geometric mean (scale-normalizing).
    For L_p << 1: behaves as arithmetic mean (stable gradients).
    Gradient weight per plane is 1/(1 + L_p), bounded in [0, 1].

    Parameters
    ----------
    signals_a, signals_b : tuple of jnp.ndarray
        Tuples of 6 arrays (one per plane).
    spectral_weights : tuple of jnp.ndarray
        Per-plane weights from make_sobolev_weight().
    planes : tuple of int
        Which plane indices to include (default all 6). Static.

    Returns
    -------
    jnp.ndarray
        Scalar loss.
    """
    log_sum = 0.0
    for p in planes:
        lp = sobolev_loss_single(signals_a[p], signals_b[p], spectral_weights[p])
        log_sum = log_sum + jnp.log1p(lp)
    return jnp.expm1(log_sum / len(planes))
