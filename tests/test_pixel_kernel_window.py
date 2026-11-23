"""Pixel response output window (tools/kernels.py).

The output window must be centred on the deposit's pixel: a deposit at
sub-pixel offset u gives on pixel m the same waveform as a deposit at -u
gives on pixel -m.
"""

import numpy as np
import pytest

from tools.kernels import (load_pixel_response_kernel,
                           interpolate_pixel_response_kernel)

pytestmark = pytest.mark.requires_kernels

NPZ = "config/pixel_response.npz"
DT = 0.5  # simulation time step, us


@pytest.mark.parametrize("u", [0.13, 0.37, 0.49])
def test_window_is_mirror_symmetric(u):
    z = np.load(NPZ)
    pitch = 10 * float(z["pixel_bin_size"])  # integer bins per pitch
    k = load_pixel_response_kernel(
        NPZ, num_s=4, time_spacing=DT, pixel_pitch_cm=pitch,
        max_sigma_trans_unitless=0.2, max_sigma_long_unitless=0.0)

    def waveforms(uy):
        return np.asarray(interpolate_pixel_response_kernel(
            k.DKernel, 1.0, uy, 0.1, 0.0, k.pixel_spacing,
            k.kernel_py, k.kernel_pz, k.rebin_factor))

    pos, neg = waveforms(u), waveforms(-u)
    # Output index i is pixel offset i - py_zero_bin; mirror m -> -m.
    m = np.arange(k.kernel_py) - k.py_zero_bin
    mirror = k.py_zero_bin - m
    assert np.all((mirror >= 0) & (mirror < k.kernel_py)), \
        f"window offsets {m.tolist()} are not symmetric about the deposit pixel"
    np.testing.assert_allclose(pos, neg[mirror], rtol=0, atol=1e-6)
