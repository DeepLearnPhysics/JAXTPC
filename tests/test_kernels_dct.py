"""
Tests for response kernel diffusion table generation and interpolation.

Files under test: tools/kernels.py
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from tools.kernels import (
    generate_dkernel_table,
    interpolate_diffusion_kernel,
    interpolate_diffusion_kernel_batch,
)


@pytest.fixture
def synthetic_kernel():
    """Synthetic base kernel for testing without real kernel files.

    21x15 kernel with a centered Gaussian-like peak (wire_spacing=0.1).
    """
    H, W = 21, 15
    ty = jnp.arange(H, dtype=jnp.float32) - H // 2
    wx = jnp.arange(W, dtype=jnp.float32) - W // 2
    tt, ww = jnp.meshgrid(ty, wx, indexing='ij')
    kernel = jnp.exp(-0.5 * (ww ** 2 / 4.0 + tt ** 2 / 9.0))
    return kernel, 0.1, 0.5  # kernel, dx, dy


class TestGenerateDKernelTable:
    """Tests for reflect-pad + spatial convolution DKernel generation."""

    def test_output_shape(self, synthetic_kernel):
        """Output shape should be (num_s, H, W)."""
        base, dx, dy = synthetic_kernel
        H, W = base.shape
        s_levels = jnp.linspace(0, 1, 8)

        dk = generate_dkernel_table(0.3, 0.5, base, dx, dy, s_levels)

        assert dk.shape == (8, H, W)

    def test_s0_matches_base(self, synthetic_kernel):
        """DKernel at s=0 (zero diffusion) should match the base kernel."""
        base, dx, dy = synthetic_kernel
        s_levels = jnp.linspace(0, 1, 8)

        dk = generate_dkernel_table(0.3, 0.5, base, dx, dy, s_levels)

        np.testing.assert_allclose(dk[0], base, atol=1e-5,
                                   err_msg="s=0 should match base kernel")

    def test_sum_preserved(self, synthetic_kernel):
        """Gaussian convolution preserves the DC component (total sum)."""
        base, dx, dy = synthetic_kernel
        s_levels = jnp.linspace(0, 1, 8)

        dk = generate_dkernel_table(0.3, 0.5, base, dx, dy, s_levels)

        sum_base = float(jnp.sum(base))
        for i in range(len(s_levels)):
            np.testing.assert_allclose(
                float(jnp.sum(dk[i])), sum_base, rtol=5e-3,
                err_msg=f"Sum not preserved at s_level {i}")

    def test_monotonic_peak_reduction(self, synthetic_kernel):
        """Peak value should decrease monotonically with increasing diffusion."""
        base, dx, dy = synthetic_kernel
        s_levels = jnp.linspace(0, 1, 8)

        dk = generate_dkernel_table(0.3, 0.5, base, dx, dy, s_levels)

        peaks = [float(jnp.max(jnp.abs(dk[i]))) for i in range(len(s_levels))]
        for i in range(1, len(peaks)):
            assert peaks[i] <= peaks[i - 1] + 1e-6, \
                f"Peak at s[{i}]={peaks[i]} > s[{i-1}]={peaks[i-1]}"

    def test_zero_sigma_all_identical(self, synthetic_kernel):
        """Zero diffusion sigmas should make all s-levels identical to base."""
        base, dx, dy = synthetic_kernel
        s_levels = jnp.linspace(0, 1, 4)

        dk = generate_dkernel_table(0.0, 0.0, base, dx, dy, s_levels)

        for i in range(len(s_levels)):
            np.testing.assert_allclose(dk[i], base, atol=1e-5,
                                       err_msg=f"s[{i}] should match base with zero sigma")


class TestInterpolateDiffusionKernel:
    """Tests for runtime kernel interpolation."""

    @pytest.fixture
    def dkernel_and_meta(self, synthetic_kernel):
        """Build a DKernel table from the synthetic kernel."""
        base, dx, dy = synthetic_kernel
        s_levels = jnp.linspace(0, 1, 8)
        dk = generate_dkernel_table(0.3, 0.5, base, dx, dy, s_levels)
        wire_spacing = dx
        num_wires = 4
        return dk, wire_spacing, num_wires

    def test_output_shape(self, dkernel_and_meta):
        """Interpolated kernel should have shape (num_wires, kernel_height - 1)."""
        dk, ws, nw = dkernel_and_meta
        result = interpolate_diffusion_kernel(dk, 0.5, 0.0, 0.0, ws, nw)

        assert result.shape == (nw, dk.shape[1] - 1)

    def test_s_boundary_values(self, dkernel_and_meta):
        """s=0 and s=1 should produce valid (finite, non-NaN) outputs."""
        dk, ws, nw = dkernel_and_meta

        r0 = interpolate_diffusion_kernel(dk, 0.0, 0.0, 0.0, ws, nw)
        r1 = interpolate_diffusion_kernel(dk, 1.0, 0.0, 0.0, ws, nw)

        assert jnp.all(jnp.isfinite(r0))
        assert jnp.all(jnp.isfinite(r1))

    def test_batch_equals_sequential(self, dkernel_and_meta):
        """Batch interpolation should match sequential calls."""
        dk, ws, nw = dkernel_and_meta
        N = 5
        s_vals = jnp.linspace(0.1, 0.9, N)
        w_vals = jnp.zeros(N)
        t_vals = jnp.zeros(N)

        batch_result = interpolate_diffusion_kernel_batch(
            dk, s_vals, w_vals, t_vals, ws, nw)

        for i in range(N):
            single = interpolate_diffusion_kernel(
                dk, s_vals[i], w_vals[i], t_vals[i], ws, nw)
            np.testing.assert_allclose(
                batch_result[i], single, atol=1e-5,
                err_msg=f"Batch[{i}] doesn't match sequential")

    def test_jit_compatible(self, dkernel_and_meta):
        """Interpolation should work under jax.jit."""
        dk, ws, nw = dkernel_and_meta

        @jax.jit
        def interp(s):
            return interpolate_diffusion_kernel(dk, s, 0.0, 0.0, ws, nw)

        result = interp(0.5)
        assert jnp.all(jnp.isfinite(result))
