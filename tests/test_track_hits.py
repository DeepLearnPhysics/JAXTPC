"""
Tests for track hit labeling module.

File under test: tools/track_hits.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from tools.track_hits import group_hits_by_track, label_hits, sparse_hits_to_dense


class TestGroupHitsByTrack:
    """Tests for group_hits_by_track."""

    def test_single_track_aggregation(self):
        """5 hits at same (wire, time) from same track → one entry, charge=sum."""
        wire_time = jnp.array([[10, 20]] * 5, dtype=jnp.int32)
        track_ids = jnp.array([1, 1, 1, 1, 1], dtype=jnp.int32)
        charges = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        hits, num_hits, boundaries, num_tracks, track_ids_out = group_hits_by_track(
            wire_time, track_ids, charges,
            min_charge_threshold=0.0,
            max_tracks=100, max_wires=100, max_time=100, max_keys=1000)
        jax.block_until_ready(hits)

        # Should have exactly 1 unique entry
        assert int(num_hits) == 1
        # Total charge should be sum
        np.testing.assert_allclose(float(hits[0, 2]), 15.0, rtol=1e-4)

    def test_threshold_filtering(self):
        """Hits with total charge below threshold should be removed."""
        wire_time = jnp.array([[10, 20], [30, 40]], dtype=jnp.int32)
        track_ids = jnp.array([1, 2], dtype=jnp.int32)
        charges = jnp.array([0.5, 5.0])  # First below threshold=1.0

        hits, num_hits, _, _, _ = group_hits_by_track(
            wire_time, track_ids, charges,
            min_charge_threshold=1.0,
            max_tracks=100, max_wires=100, max_time=100, max_keys=1000)
        jax.block_until_ready(hits)

        assert int(num_hits) == 1, f"Expected 1 hit above threshold, got {int(num_hits)}"

    def test_multiple_tracks_separated(self):
        """3 tracks at distinct locations should produce num_tracks=3."""
        wire_time = jnp.array([
            [10, 20], [10, 20],  # Track 1
            [30, 40], [30, 40],  # Track 2
            [50, 60],            # Track 3
        ], dtype=jnp.int32)
        track_ids = jnp.array([1, 1, 2, 2, 3], dtype=jnp.int32)
        charges = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        hits, num_hits, boundaries, num_tracks, track_ids_out = group_hits_by_track(
            wire_time, track_ids, charges,
            min_charge_threshold=0.0,
            max_tracks=100, max_wires=100, max_time=100, max_keys=1000)
        jax.block_until_ready(hits)

        assert int(num_tracks) == 3, f"Expected 3 tracks, got {int(num_tracks)}"
        assert int(num_hits) == 3, f"Expected 3 unique hits, got {int(num_hits)}"


class TestLabelHits:
    """Tests for label_hits (dominant track labeling)."""

    def test_dominant_track_labeling(self):
        """Two tracks at same location → label = highest charge track."""
        wire_time = jnp.array([
            [10, 20],  # Track 1, charge 2.0
            [10, 20],  # Track 2, charge 5.0 (dominant)
        ], dtype=jnp.int32)
        track_ids = jnp.array([1, 2], dtype=jnp.int32)
        charges = jnp.array([2.0, 5.0])

        hits, num_hits, boundaries, num_tracks, track_ids_out = group_hits_by_track(
            wire_time, track_ids, charges,
            min_charge_threshold=0.0,
            max_tracks=100, max_wires=100, max_time=100, max_keys=1000)
        jax.block_until_ready(hits)

        labeled, num_labeled = label_hits(
            hits, int(num_hits), track_ids_out, boundaries, int(num_tracks),
            max_keys=1000, max_time=100)
        jax.block_until_ready(labeled)

        # Should have 1 unique (wire, time) with track_id = 2 (dominant)
        assert int(num_labeled) == 1
        assert int(labeled[0, 0]) == 2, f"Expected track 2 as dominant, got {int(labeled[0, 0])}"


class TestSparseHitsToDense:
    """Tests for sparse_hits_to_dense."""

    def test_sparse_hits_to_dense(self):
        """Known hits should produce correct dense array values."""
        hits_by_track = jnp.zeros((100, 3))
        hits_by_track = hits_by_track.at[0].set(jnp.array([5, 10, 3.0]))
        hits_by_track = hits_by_track.at[1].set(jnp.array([7, 15, 2.0]))
        hits_by_track = hits_by_track.at[2].set(jnp.array([5, 10, 1.0]))  # Same location

        track_result = {
            'hits_by_track': hits_by_track,
            'num_hits': 3,
        }

        dense = sparse_hits_to_dense(track_result, num_wires=20, num_time_steps=30)

        # (5, 10) should have 3.0 + 1.0 = 4.0
        np.testing.assert_allclose(float(dense[5, 10]), 4.0, rtol=1e-4)
        np.testing.assert_allclose(float(dense[7, 15]), 2.0, rtol=1e-4)
        assert float(dense[0, 0]) == 0.0
