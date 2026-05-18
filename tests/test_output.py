"""
Tests for output format conversion.

File under test: tools/output.py
"""

import numpy as np
import jax.numpy as jnp
import pytest

from tools.output import to_dense, to_sparse
from tools.wires import (
    build_bucket_mapping,
    scatter_contributions_to_buckets,
)


@pytest.fixture
def mock_config():
    """Minimal config for output tests."""
    from collections import namedtuple
    VG = namedtuple('VG', ['num_wires', 'n_planes'])
    Cfg = namedtuple('Cfg', ['volumes', 'n_volumes', 'num_time_steps', 'time_step_us', 'electrons_per_adc'])
    return Cfg(
        volumes=(VG(num_wires=(100, 100, 80), n_planes=3),
                 VG(num_wires=(100, 100, 80), n_planes=3)),
        n_volumes=2,
        num_time_steps=200,
        time_step_us=0.5,
        electrons_per_adc=182.0,
    )


class TestToDense:
    """Tests for to_dense conversion."""

    def test_dense_passthrough(self, mock_config):
        """Dense arrays should pass through unchanged."""
        arr = np.random.randn(100, 200).astype(np.float32)
        signals = {(0, 0): arr}

        result = to_dense(signals, mock_config)
        np.testing.assert_array_equal(result[(0, 0)], arr)

    def test_dense_shape(self, mock_config):
        """Output should have correct (num_wires, num_time) shape."""
        arr = np.zeros((100, 200), dtype=np.float32)
        arr[50, 100] = 42.0
        signals = {(0, 0): arr, (1, 2): np.zeros((80, 200), dtype=np.float32)}

        result = to_dense(signals, mock_config)
        assert result[(0, 0)].shape == (100, 200)
        assert result[(1, 2)].shape == (80, 200)


class TestToSparse:
    """Tests for to_sparse conversion."""

    def test_sparse_basic(self, mock_config):
        """Nonzero entries should appear in sparse output."""
        arr = np.zeros((100, 200), dtype=np.float32)
        arr[10, 20] = 5.0
        arr[30, 40] = -3.0
        signals = {(0, 0): arr}

        result = to_sparse(signals, mock_config)
        sp = result[(0, 0)]

        assert len(sp['wire']) == 2
        assert len(sp['time']) == 2
        assert len(sp['values']) == 2
        assert set(zip(sp['wire'], sp['time'])) == {(10, 20), (30, 40)}

    def test_sparse_threshold(self, mock_config):
        """Threshold should filter small values."""
        arr = np.zeros((100, 200), dtype=np.float32)
        arr[10, 20] = 5.0
        arr[30, 40] = 1.0  # below threshold
        signals = {(0, 0): arr}

        result = to_sparse(signals, mock_config, threshold_adc=2.0)
        sp = result[(0, 0)]

        assert len(sp['values']) == 1
        assert float(sp['values'][0]) == 5.0

    def test_sparse_empty(self, mock_config):
        """All-zero input should give empty sparse output."""
        signals = {(0, 0): np.zeros((100, 200), dtype=np.float32)}

        result = to_sparse(signals, mock_config)
        sp = result[(0, 0)]
        assert len(sp['values']) == 0

    def test_dense_sparse_roundtrip(self, mock_config):
        """to_dense(to_sparse(x)) should recover nonzero entries."""
        arr = np.zeros((100, 200), dtype=np.float32)
        rng = np.random.RandomState(42)
        for _ in range(50):
            w, t = rng.randint(0, 100), rng.randint(0, 200)
            arr[w, t] = rng.randn() * 10
        signals = {(0, 0): arr}

        sparse = to_sparse(signals, mock_config)

        # Reconstruct dense from sparse
        sp = sparse[(0, 0)]
        reconstructed = np.zeros_like(arr)
        reconstructed[sp['wire'], sp['time']] = sp['values']

        np.testing.assert_array_equal(reconstructed, arr)


class TestBucketedToDense:
    """Test to_dense conversion from bucketed 5-tuple format."""

    def test_bucketed_roundtrip(self, mock_config):
        """Bucketed signal through to_dense should match reference dense."""
        num_wires = 100
        num_time = 200
        kW = 5
        kH = 7
        wire_zero = kW // 2
        time_zero = kH // 2
        B1 = 2 * kW
        B2 = 2 * kH
        max_buckets = 20

        rng = np.random.RandomState(42)
        N = 6
        wire_idx = jnp.array(rng.randint(kW, num_wires - kW, N), dtype=jnp.int32)
        time_idx = jnp.array(rng.randint(kH, num_time - kH, N), dtype=jnp.int32)
        intensities = jnp.array(rng.uniform(1.0, 5.0, N), dtype=jnp.float32)
        contributions = jnp.array(rng.randn(N, kW, kH), dtype=jnp.float32)

        # Build reference dense
        from tools.wires import accumulate_response_signals
        ref_dense = np.asarray(accumulate_response_signals(
            wire_idx, time_idx, intensities, contributions,
            num_wires, num_time, kW, kH, wire_zero, time_zero))

        # Build bucketed
        p2c, num_active, c2k = build_bucket_mapping(
            wire_idx, time_idx, B1, B2, num_wires, num_time,
            max_buckets, wire_zero, time_zero)
        buckets = scatter_contributions_to_buckets(
            wire_idx, time_idx, intensities, contributions,
            p2c, max_buckets, kW, kH, B1, B2, wire_zero, time_zero,
            num_wires=num_wires, num_time_steps=num_time)

        bucketed_signal = (buckets, num_active, c2k,
                          jnp.array(B1), jnp.array(B2))
        signals = {(0, 0): bucketed_signal}
        result = to_dense(signals, mock_config)

        np.testing.assert_allclose(
            result[(0, 0)], ref_dense, atol=1e-4,
            err_msg="to_dense(bucketed) doesn't match reference dense")


class TestWireSparseToDense:
    """Test to_dense conversion from wire_sparse 3-tuple format."""

    def test_wire_sparse_roundtrip(self, mock_config):
        """Wire-sparse signal through to_dense should reconstruct correctly."""
        num_wires = 100
        num_time = 200

        rng = np.random.RandomState(42)
        n_active = 5
        wire_indices = np.array(sorted(rng.choice(num_wires, n_active, replace=False)),
                                dtype=np.int32)

        active_signals = np.zeros((n_active, num_time), dtype=np.float32)
        for i in range(n_active):
            active_signals[i, 50:60] = rng.randn(10).astype(np.float32) * 10

        # Build reference dense
        ref_dense = np.zeros((num_wires, num_time), dtype=np.float32)
        for i in range(n_active):
            ref_dense[wire_indices[i]] = active_signals[i]

        wire_sparse = (jnp.array(active_signals), jnp.array(wire_indices),
                       jnp.array(n_active))
        signals = {(0, 0): wire_sparse}
        result = to_dense(signals, mock_config)

        np.testing.assert_allclose(
            result[(0, 0)], ref_dense, atol=1e-6,
            err_msg="to_dense(wire_sparse) doesn't match reference dense")
