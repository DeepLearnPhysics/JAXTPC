"""Pixel response table applied at a pitch other than its own (response_pixel_bin)."""

import numpy as np
import jax.numpy as jnp
import pytest

from tools.config import create_sim_config
from tools.geometry import generate_detector
from tools.kernels import apply_pixel_diffusion_response, load_pixel_response_kernel

pytestmark = [pytest.mark.requires_kernels, pytest.mark.requires_config]

NPZ = "config/pixel_response.npz"
DT = 0.5


def _charge(config):
    """Charge collected over all pixels, at 16 sub-pixel offsets, unit deposits."""
    vol = create_sim_config(generate_detector(config), total_pad=100).volumes[0]
    k = load_pixel_response_kernel(
        NPZ, num_s=4, time_spacing=DT, pixel_pitch_cm=vol.pixel_pitch_cm,
        max_sigma_trans_unitless=0.2, max_sigma_long_unitless=0.5,
        pixel_bin_cm=vol.pixel_response_bin_cm)
    u = np.linspace(-0.45, 0.45, 16)
    out = apply_pixel_diffusion_response(
        k.DKernel, jnp.ones(16), jnp.array(u), jnp.array(u[::-1]),
        jnp.full(16, 0.3), k.pixel_spacing, k.kernel_py, k.kernel_pz, k.rebin_factor)
    return vol, np.asarray(out.sum(axis=(1, 2, 3)) * DT)


@pytest.mark.parametrize("config", ["config/cubic_pixel_config.yaml",
                                    "config/dune_ndlar_config.yaml"])
def test_all_charge_is_collected(config):
    _, q = _charge(config)
    np.testing.assert_allclose(q, 1.0, atol=2e-3)


def test_table_bin_matches_pitch():
    for config in ("config/cubic_pixel_config.yaml", "config/dune_ndlar_config.yaml"):
        vol, _ = _charge(config)
        bin_cm = vol.pixel_response_bin_cm or float(np.load(NPZ)["pixel_bin_size"])
        assert vol.pixel_pitch_cm / bin_cm == pytest.approx(10.0)
