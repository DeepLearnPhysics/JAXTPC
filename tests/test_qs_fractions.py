"""
Tests for Q_s fraction computation.

File under test: tools/track_hits.py (compute_qs_fractions)
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tools.track_hits import compute_qs_fractions


class TestComputeQsFractions:
    """Tests for compute_qs_fractions inside JIT."""

    def test_fractions_sum_to_one_per_group(self):
        """Q_s fractions within each group should sum to 1.0."""
        charges = jnp.array([10.0, 20.0, 30.0, 5.0, 15.0, 0.0])
        group_ids = jnp.array([0, 0, 0, 1, 1, 2])
        num_segments = 3

        qs = compute_qs_fractions(charges, group_ids, num_segments)

        # Group 0: 10+20+30=60 → fractions 10/60, 20/60, 30/60
        np.testing.assert_allclose(float(qs[0]), 10.0 / 60.0, rtol=1e-5)
        np.testing.assert_allclose(float(qs[1]), 20.0 / 60.0, rtol=1e-5)
        np.testing.assert_allclose(float(qs[2]), 30.0 / 60.0, rtol=1e-5)

        # Group 1: 5+15=20
        np.testing.assert_allclose(float(qs[3]), 5.0 / 20.0, rtol=1e-5)
        np.testing.assert_allclose(float(qs[4]), 15.0 / 20.0, rtol=1e-5)

        # Group sums
        np.testing.assert_allclose(float(qs[0] + qs[1] + qs[2]), 1.0, rtol=1e-5)
        np.testing.assert_allclose(float(qs[3] + qs[4]), 1.0, rtol=1e-5)

    def test_zero_charges_give_zero_fractions(self):
        """Zero charges should give zero fractions (not NaN)."""
        charges = jnp.array([0.0, 0.0, 5.0, 5.0])
        group_ids = jnp.array([0, 0, 1, 1])

        qs = compute_qs_fractions(charges, group_ids, num_segments=2)

        assert jnp.all(jnp.isfinite(qs)), "Fractions should not be NaN/Inf"
        assert float(qs[0]) == 0.0
        assert float(qs[1]) == 0.0

    def test_single_deposit_per_group(self):
        """Single deposit in a group should have fraction 1.0."""
        charges = jnp.array([7.0, 3.0, 12.0])
        group_ids = jnp.array([0, 1, 2])

        qs = compute_qs_fractions(charges, group_ids, num_segments=3)

        for i in range(3):
            np.testing.assert_allclose(float(qs[i]), 1.0, rtol=1e-5)

    def test_padded_entries_are_zero(self):
        """Padded entries (charges=0, group_id=0) should not affect real groups."""
        # 3 real deposits in group 1, then 2 padded (group 0, charge 0)
        charges = jnp.array([10.0, 20.0, 30.0, 0.0, 0.0])
        group_ids = jnp.array([1, 1, 1, 0, 0])

        qs = compute_qs_fractions(charges, group_ids, num_segments=5)

        # Group 1 fractions
        np.testing.assert_allclose(float(qs[0] + qs[1] + qs[2]), 1.0, rtol=1e-5)
        # Padded entries
        assert float(qs[3]) == 0.0
        assert float(qs[4]) == 0.0

    def test_works_inside_jit(self):
        """Should be callable inside jax.jit."""
        @jax.jit
        def fn(charges, gids):
            return compute_qs_fractions(charges, gids, num_segments=10)

        charges = jnp.array([1.0, 2.0, 3.0])
        gids = jnp.array([0, 0, 1])
        qs = fn(charges, gids)

        np.testing.assert_allclose(float(qs[0] + qs[1]), 1.0, rtol=1e-5)
        np.testing.assert_allclose(float(qs[2]), 1.0, rtol=1e-5)
