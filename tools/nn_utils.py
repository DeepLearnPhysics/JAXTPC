"""
NN inference utilities for JAXTPC simulation.

Extracted from siren_response_training/ to avoid runtime dependency on
training code. Contains transforms and kernel operations needed at inference.
"""

import jax.numpy as jnp


def inv_symlog(x, eps=1e-10):
    """Inverse symlog transform: symlog(y) = sign(y) * log(|y| + eps).

    inv_symlog(x) = sign(x) * (exp(|x|) - eps)
    """
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - eps)


def unfold_kernel(folded):
    """Mirror folded kernel to full width using wire symmetry.

    Parameters
    ----------
    folded : (N, H, half_W) or (H, half_W)
        Folded kernel (right half including center wire).

    Returns
    -------
    full : (N, H, 2*half_W - 1) or (H, 2*half_W - 1)
        Full symmetric kernel.
    """
    # Left half is the mirror of folded[:, :, 1:] (excluding center to avoid duplication)
    left_half = jnp.flip(folded[..., 1:], axis=-1)
    return jnp.concatenate([left_half, folded], axis=-1)


def normalize_positions(positions_cm, origin_cm, extent_cm):
    """Map physical coordinates to [0,1]^3 for NN input.

    Parameters
    ----------
    positions_cm : (N, 3) positions in cm
    origin_cm : (3,) or tuple, lower corner of normalization box
    extent_cm : (3,) or tuple, size of normalization box

    Returns
    -------
    normalized : (N, 3) in [0,1]^3
    """
    origin = jnp.array(origin_cm)
    extent = jnp.array(extent_cm)
    return (positions_cm - origin) / extent
