"""
Tests for dense/sparse conversion utilities.

File under test: tools/sparse_utils.py
"""

import numpy as np
import jax.numpy as jnp
import pytest
from tools.sparse_utils import dense_to_sparse, sparse_to_dense


class TestDenseToSparseRoundtrip:
    """Tests for dense ↔ sparse conversions."""

    def test_roundtrip(self):
        """dense → sparse → dense should reproduce the original array."""
        rng = np.random.RandomState(10)
        num_wires, num_time = 50, 100
        dense = jnp.array(rng.randn(num_wires, num_time).astype(np.float32))

        indices, values = dense_to_sparse(dense, threshold=0.0)
        reconstructed = sparse_to_dense(indices, values, num_wires, num_time)

        np.testing.assert_allclose(np.array(reconstructed), np.array(dense), rtol=1e-5)

    def test_threshold_filtering(self):
        """Threshold should filter by absolute value but preserve sign."""
        dense = jnp.array([[0.5, 1.0, 2.5, 3.0, -2.5]], dtype=jnp.float32)

        indices, values = dense_to_sparse(dense, threshold=2.0)

        # Only |v| > 2.0 kept: 2.5, 3.0, -2.5
        assert len(values) == 3, f"Expected 3 entries above threshold, got {len(values)}"
        # Verify sign is preserved
        vals_sorted = np.sort(np.array(values))
        assert vals_sorted[0] < 0, "Negative value should be preserved"


class TestSparseAccumulation:
    """Tests for sparse_to_dense accumulation behavior."""

    def test_duplicate_index_accumulation(self):
        """Duplicate (wire, time) indices should have values summed."""
        indices = jnp.array([[1, 2], [1, 2], [3, 4]], dtype=jnp.int32)
        values = jnp.array([1.0, 2.0, 5.0], dtype=jnp.float32)

        dense = sparse_to_dense(indices, values, num_wires=5, num_time_steps=6)

        assert float(dense[1, 2]) == 3.0, "Duplicate indices should sum: 1.0 + 2.0 = 3.0"
        assert float(dense[3, 4]) == 5.0

    def test_empty_sparse_array(self):
        """Empty sparse arrays should produce all-zero dense output."""
        indices = jnp.empty((0, 2), dtype=jnp.int32)
        values = jnp.empty((0,), dtype=jnp.float32)

        dense = sparse_to_dense(indices, values, num_wires=10, num_time_steps=20)

        assert jnp.all(dense == 0.0), "Empty sparse should give all-zero dense"
        assert dense.shape == (10, 20)
