"""
Tests for wire geometry and accumulation functions.

Files under test: tools/wires.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tools.wires import (
    compute_wire_distances,
    accumulate_response_signals,
    build_bucket_mapping,
    scatter_contributions_to_buckets,
    sparse_buckets_to_dense,
)


class TestComputeWireDistances:
    """Test wire index projection for different plane angles."""

    def test_y_plane_angle_zero(self):
        """Y-plane (angle=0): wire index depends only on z."""
        positions_yz = jnp.array([[0.0, 0.3]], dtype=jnp.float32)
        angle_rad = 0.0
        wire_spacing = 0.3
        max_wire_idx_abs = 100
        index_offset = 100

        idx, dist = compute_wire_distances(
            positions_yz, angle_rad, wire_spacing, max_wire_idx_abs, index_offset)

        assert int(idx[0]) == 101
        assert abs(float(dist[0])) < 1e-5

    def test_y_plane_two_wire_spacings(self):
        """Deposit at z=2*spacing should give index_offset+2."""
        positions_yz = jnp.array([[0.0, 0.6]], dtype=jnp.float32)
        idx, dist = compute_wire_distances(
            positions_yz, 0.0, 0.3, 100, 100)

        assert int(idx[0]) == 102
        assert abs(float(dist[0])) < 1e-5

    def test_y_plane_half_spacing_offset(self):
        """Deposit halfway between wires should have distance = half spacing."""
        positions_yz = jnp.array([[0.0, 0.15]], dtype=jnp.float32)
        idx, dist = compute_wire_distances(
            positions_yz, 0.0, 0.3, 100, 100)

        assert abs(float(dist[0])) == pytest.approx(0.15, abs=1e-5)

    def test_u_v_symmetry(self):
        """U (+60deg) and V (-60deg) should give same index for deposit at origin."""
        positions_yz = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        u_angle = jnp.radians(60.0)
        v_angle = jnp.radians(-60.0)

        idx_u, _ = compute_wire_distances(positions_yz, u_angle, 0.3, 100, 100)
        idx_v, _ = compute_wire_distances(positions_yz, v_angle, 0.3, 100, 100)

        assert int(idx_u[0]) == int(idx_v[0]) == 100

    def test_vectorized(self):
        """Multiple deposits should be processed in parallel."""
        rng = np.random.RandomState(42)
        N = 50
        positions_yz = jnp.array(rng.uniform(-10, 10, (N, 2)), dtype=jnp.float32)

        idx, dist = compute_wire_distances(positions_yz, 0.0, 0.3, 200, 200)

        assert idx.shape == (N,)
        assert dist.shape == (N,)
        assert jnp.all(jnp.abs(dist) <= 0.15 + 1e-5)

    def test_negative_z_gives_lower_index(self):
        """Y-plane: negative z should give index below offset."""
        positions_yz = jnp.array([[0.0, -0.3]], dtype=jnp.float32)
        idx, _ = compute_wire_distances(positions_yz, 0.0, 0.3, 100, 100)

        assert int(idx[0]) == 99


class TestAccumulateResponseSignals:
    """Test dense signal accumulation from response kernels."""

    def test_single_deposit_center(self):
        """Single deposit at center should accumulate full kernel sum."""
        num_wires = 20
        num_time = 50
        kW = 3
        kH = 5
        wire_zero = kW // 2
        time_zero = kH // 2

        wire_indices = jnp.array([10])
        time_indices = jnp.array([25])
        intensities = jnp.array([1.0])
        kernel = jnp.ones((1, kW, kH), dtype=jnp.float32)

        signals = accumulate_response_signals(
            wire_indices, time_indices, intensities, kernel,
            num_wires, num_time, kW, kH, wire_zero, time_zero)

        assert float(signals.sum()) == pytest.approx(kW * kH, abs=1e-3)

    def test_intensity_scaling(self):
        """Signal sum should scale linearly with intensity."""
        num_wires = 20
        num_time = 50
        kW = 3
        kH = 5
        wire_zero = kW // 2
        time_zero = kH // 2

        wire_indices = jnp.array([10])
        time_indices = jnp.array([25])
        kernel = jnp.ones((1, kW, kH), dtype=jnp.float32)

        signals_1x = accumulate_response_signals(
            wire_indices, time_indices, jnp.array([1.0]), kernel,
            num_wires, num_time, kW, kH, wire_zero, time_zero)
        signals_3x = accumulate_response_signals(
            wire_indices, time_indices, jnp.array([3.0]), kernel,
            num_wires, num_time, kW, kH, wire_zero, time_zero)

        np.testing.assert_allclose(signals_3x, 3.0 * signals_1x, atol=1e-5)

    def test_boundary_clipping(self):
        """Deposits at the grid boundary should clip (drop out-of-bounds parts)."""
        num_wires = 10
        num_time = 10
        kW = 5
        kH = 5
        wire_zero = kW // 2
        time_zero = kH // 2

        wire_indices = jnp.array([0])
        time_indices = jnp.array([0])
        intensities = jnp.array([1.0])
        kernel = jnp.ones((1, kW, kH), dtype=jnp.float32)

        signals = accumulate_response_signals(
            wire_indices, time_indices, intensities, kernel,
            num_wires, num_time, kW, kH, wire_zero, time_zero)

        # Should be less than full sum due to clipping
        full_sum = kW * kH
        actual_sum = float(signals.sum())
        assert actual_sum > 0.0
        assert actual_sum < full_sum

    def test_two_deposits_additive(self):
        """Two non-overlapping deposits should be additive."""
        num_wires = 30
        num_time = 50
        kW = 3
        kH = 3
        wire_zero = kW // 2
        time_zero = kH // 2

        wire_indices = jnp.array([5, 25])
        time_indices = jnp.array([10, 40])
        intensities = jnp.array([1.0, 2.0])
        kernel = jnp.ones((2, kW, kH), dtype=jnp.float32)

        signals = accumulate_response_signals(
            wire_indices, time_indices, intensities, kernel,
            num_wires, num_time, kW, kH, wire_zero, time_zero)

        assert float(signals.sum()) == pytest.approx(
            (1.0 + 2.0) * kW * kH, abs=1e-3)

    def test_zero_intensity_no_contribution(self):
        """Zero intensity deposit should not change the signal array."""
        num_wires = 10
        num_time = 10
        kW = 3
        kH = 3
        wire_zero = kW // 2
        time_zero = kH // 2

        wire_indices = jnp.array([5])
        time_indices = jnp.array([5])
        intensities = jnp.array([0.0])
        kernel = jnp.ones((1, kW, kH), dtype=jnp.float32)

        signals = accumulate_response_signals(
            wire_indices, time_indices, intensities, kernel,
            num_wires, num_time, kW, kH, wire_zero, time_zero)

        assert float(signals.sum()) == 0.0


class TestBucketedAccumulation:
    """Test sparse bucketed accumulation matches dense."""

    def test_dense_equals_bucketed(self):
        """Bucketed accumulation reconstructed to dense should match direct dense."""
        num_wires = 30
        num_time = 50
        kW = 5
        kH = 7
        wire_zero = kW // 2
        time_zero = kH // 2
        B1 = 2 * kW
        B2 = 2 * kH
        max_buckets = 20

        rng = np.random.RandomState(42)
        N = 8
        wire_indices = jnp.array(rng.randint(kW, num_wires - kW, N), dtype=jnp.int32)
        time_indices = jnp.array(rng.randint(kH, num_time - kH, N), dtype=jnp.int32)
        intensities = jnp.array(rng.uniform(0.5, 5.0, N), dtype=jnp.float32)
        contributions = jnp.array(rng.randn(N, kW, kH), dtype=jnp.float32)

        # Dense path
        dense = accumulate_response_signals(
            wire_indices, time_indices, intensities, contributions,
            num_wires, num_time, kW, kH, wire_zero, time_zero)

        # Bucketed path
        p2c, num_active, c2k = build_bucket_mapping(
            wire_indices, time_indices, B1, B2,
            num_wires, num_time, max_buckets, wire_zero, time_zero)

        buckets = scatter_contributions_to_buckets(
            wire_indices, time_indices, intensities, contributions,
            p2c, max_buckets, kW, kH, B1, B2, wire_zero, time_zero,
            num_wires=num_wires, num_time_steps=num_time)

        reconstructed = sparse_buckets_to_dense(
            buckets, c2k, num_active, B1, B2, num_wires, num_time, max_buckets)

        np.testing.assert_allclose(
            np.array(reconstructed), np.array(dense), atol=1e-4,
            err_msg="Bucketed reconstruction doesn't match dense accumulation")
