"""Pixel response kernel lookup on the bin-centre grid (tools/kernels.py).

The pixel NPZ samples the quarter plane at x_k = (k + 1/2) * bin from the
pixel centre. Read correctly, the charge collected by all output pixels
equals the table's pad-sum for every sub-pixel deposit offset.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.kernels import (load_pixel_response_kernel,
                           interpolate_pixel_response_kernel)

pytestmark = pytest.mark.requires_kernels

NPZ = "config/pixel_response.npz"
DT = 0.5  # simulation time step, us
OFFSETS = (np.arange(20) + 0.5) / 20 - 0.5  # 20 sub-pixel offsets in [-0.5, 0.5)


def _kernel(sigma_t):
    z = np.load(NPZ)
    bins_per_pitch = 10
    pitch = bins_per_pitch * float(z["pixel_bin_size"])  # integer bins per pitch
    k = load_pixel_response_kernel(
        NPZ, num_s=4, time_spacing=DT, pixel_pitch_cm=pitch,
        max_sigma_trans_unitless=sigma_t, max_sigma_long_unitless=0.0)
    # Charge collected by the pad: quarter-plane cells inside the pad, x4,
    # divided by the cells per pixel.
    half = bins_per_pitch // 2
    pad_q = (4 * z["response"][:half, :half].sum() * float(z["time_bin_size"])
             / bins_per_pitch ** 2)
    return k, pad_q


@pytest.mark.parametrize("sigma_t", [0.0, 0.2])  # transverse sigma, pixel pitches
def test_total_charge_equals_pad_sum_at_every_offset(sigma_t):
    k, pad_q = _kernel(sigma_t)
    s = 1.0 if sigma_t else 0.0

    def total_q(u, v):
        out = interpolate_pixel_response_kernel(
            k.DKernel, s, u, v, 0.0, k.pixel_spacing,
            k.kernel_py, k.kernel_pz, k.rebin_factor)
        return out.sum() * DT

    grid = jnp.asarray(OFFSETS, dtype=jnp.float32)
    q = np.asarray(jax.jit(jax.vmap(jax.vmap(total_q, (None, 0)), (0, None)))(grid, grid))
    np.testing.assert_allclose(q, pad_q, atol=1e-3)
