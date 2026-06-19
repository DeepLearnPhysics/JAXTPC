"""
Tests for drift physics calculations.

Files under test: tools/drift.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tools.drift import (
    compute_drift_to_plane,
    correct_drift_for_plane,
    apply_drift_corrections,
)


class TestComputeDriftToPlane:
    """Test basic drift distance/time calculation."""

    def test_drift_direction_negative(self):
        """drift_direction=-1: anode at x_min, cathode at x_max."""
        positions = jnp.array([[-15.0, 0.0, 0.0]], dtype=jnp.float32)
        x_anode = -20.0
        drift_dir = -1
        velocity = 1.6
        plane_dist = 0.0

        dist, time, yz = compute_drift_to_plane(
            positions, x_anode, drift_dir, velocity, plane_dist)

        assert float(dist[0]) == pytest.approx(5.0, abs=1e-5)
        assert float(time[0]) == pytest.approx(5.0 / 1.6, abs=1e-5)
        np.testing.assert_allclose(yz[0], [0.0, 0.0])

    def test_drift_direction_positive(self):
        """drift_direction=+1: anode at x_max, cathode at x_min."""
        positions = jnp.array([[15.0, 0.0, 0.0]], dtype=jnp.float32)
        x_anode = 20.0
        drift_dir = 1
        velocity = 1.6
        plane_dist = 0.0

        dist, time, yz = compute_drift_to_plane(
            positions, x_anode, drift_dir, velocity, plane_dist)

        assert float(dist[0]) == pytest.approx(5.0, abs=1e-5)
        assert float(time[0]) == pytest.approx(5.0 / 1.6, abs=1e-5)

    def test_deposit_at_anode_zero_drift(self):
        """Deposit at the anode plane should have zero drift distance."""
        positions = jnp.array([[-20.0, 5.0, -3.0]], dtype=jnp.float32)
        x_anode = -20.0
        drift_dir = -1
        velocity = 1.6
        plane_dist = 0.0

        dist, time, _ = compute_drift_to_plane(
            positions, x_anode, drift_dir, velocity, plane_dist)

        assert float(dist[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(time[0]) == pytest.approx(0.0, abs=1e-5)

    def test_plane_offset_reduces_drift(self):
        """A plane further from the anode (inward) is closer to the deposit."""
        positions = jnp.array([[-10.0, 0.0, 0.0]], dtype=jnp.float32)
        x_anode = -20.0
        drift_dir = -1
        velocity = 1.6

        dist_at_anode, _, _ = compute_drift_to_plane(
            positions, x_anode, drift_dir, velocity, 0.0)
        dist_offset, _, _ = compute_drift_to_plane(
            positions, x_anode, drift_dir, velocity, 0.6)

        # Plane offset moves it inward (toward cathode), reducing drift distance
        assert float(dist_offset[0]) < float(dist_at_anode[0])
        assert float(dist_offset[0]) == pytest.approx(9.4, abs=1e-4)

    def test_all_distances_positive(self):
        """Drift distances should always be non-negative for deposits inside the volume."""
        rng = np.random.RandomState(42)
        x = rng.uniform(-20.0, 0.0, size=100).astype(np.float32)
        yz = rng.uniform(-20.0, 20.0, size=(100, 2)).astype(np.float32)
        positions = jnp.array(np.column_stack([x, yz]))

        dist, time, _ = compute_drift_to_plane(
            positions, -20.0, -1, 1.6, 0.6)

        assert jnp.all(dist >= 0.0)
        assert jnp.all(time >= 0.0)

    def test_yz_passthrough(self):
        """yz positions should be passed through unchanged."""
        positions = jnp.array([[-10.0, 3.5, -7.2]], dtype=jnp.float32)
        _, _, yz = compute_drift_to_plane(positions, -20.0, -1, 1.6, 0.0)
        np.testing.assert_allclose(yz[0], [3.5, -7.2])

    def test_symmetric_volumes(self):
        """Mirror deposits in east/west volumes should get the same drift distance."""
        pos_east = jnp.array([[-15.0, 0.0, 0.0]], dtype=jnp.float32)
        pos_west = jnp.array([[15.0, 0.0, 0.0]], dtype=jnp.float32)

        dist_e, _, _ = compute_drift_to_plane(pos_east, -20.0, -1, 1.6, 0.0)
        dist_w, _, _ = compute_drift_to_plane(pos_west, 20.0, 1, 1.6, 0.0)

        assert float(dist_e[0]) == pytest.approx(float(dist_w[0]), abs=1e-5)


class TestCorrectDriftForPlane:
    """Test per-plane drift correction."""

    def test_correction_reduces_distance(self):
        """Closer planes should have shorter drift distances."""
        dist = jnp.array([10.0, 5.0, 1.0], dtype=jnp.float32)
        time = dist / 1.6

        corrected_dist, corrected_time = correct_drift_for_plane(
            dist, time, 1.6, 0.3)

        assert jnp.all(corrected_dist <= dist)
        assert jnp.all(corrected_time <= time)

    def test_zero_correction_identity(self):
        """Zero plane distance difference should not change anything."""
        dist = jnp.array([10.0], dtype=jnp.float32)
        time = jnp.array([6.25], dtype=jnp.float32)

        corrected_dist, corrected_time = correct_drift_for_plane(
            dist, time, 1.6, 0.0)

        np.testing.assert_allclose(corrected_dist, dist)
        np.testing.assert_allclose(corrected_time, time)

    def test_clamp_to_zero(self):
        """Correction larger than drift should clamp to zero, not go negative."""
        dist = jnp.array([0.2], dtype=jnp.float32)
        time = jnp.array([0.125], dtype=jnp.float32)

        corrected_dist, corrected_time = correct_drift_for_plane(
            dist, time, 1.6, 0.6)

        assert float(corrected_dist[0]) == 0.0
        assert float(corrected_time[0]) == 0.0


class TestApplyDriftCorrections:
    """Test space charge effect corrections."""

    # New (SCE-rework) API: apply_drift_corrections(drift_time_us,
    # positions_yz_cm, delta_t_us, delta_yz_cm, velocity_cm_us) ->
    # (corrected_distance_cm, corrected_time_us, corrected_yz_cm).
    # Time is primary; distance is derived as corrected_time * velocity.

    def test_zero_deltas_identity(self):
        """Zero SCE deltas leave time/yz unchanged; distance = time * velocity."""
        N = 10
        time = jnp.ones(N) * 3.125
        yz = jnp.stack([jnp.linspace(-5, 5, N), jnp.linspace(-5, 5, N)], axis=-1)
        zeros_t = jnp.zeros(N)
        zeros_yz = jnp.zeros((N, 2))
        velocity = 1.6

        c_dist, c_time, c_yz = apply_drift_corrections(
            time, yz, zeros_t, zeros_yz, velocity)

        np.testing.assert_allclose(c_time, time)
        np.testing.assert_allclose(c_yz, yz)
        np.testing.assert_allclose(c_dist, time * velocity, rtol=1e-6)

    def test_positive_delta_t_increases_distance(self):
        """Positive delta_t should increase drift time and derived distance."""
        time = jnp.array([3.125], dtype=jnp.float32)
        yz = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        delta_t = jnp.array([1.0], dtype=jnp.float32)
        zeros_yz = jnp.zeros((1, 2))
        velocity = 1.6

        c_dist, c_time, _ = apply_drift_corrections(
            time, yz, delta_t, zeros_yz, velocity)

        assert float(c_time[0]) == pytest.approx(4.125, abs=1e-5)
        assert float(c_dist[0]) == pytest.approx(4.125 * velocity, abs=1e-4)
        assert float(c_time[0]) > float(time[0])

    def test_negative_delta_clamps_to_zero(self):
        """Large negative delta_t should clamp time (and distance) to zero."""
        time = jnp.array([1.25], dtype=jnp.float32)
        yz = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        delta_t = jnp.array([-5.0], dtype=jnp.float32)
        zeros_yz = jnp.zeros((1, 2))

        c_dist, c_time, _ = apply_drift_corrections(
            time, yz, delta_t, zeros_yz, 1.6)

        assert float(c_time[0]) == 0.0
        assert float(c_dist[0]) == 0.0

    def test_yz_corrections_applied(self):
        """delta_yz should shift the transverse positions."""
        time = jnp.array([3.125], dtype=jnp.float32)
        yz = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        zeros_t = jnp.array([0.0])
        delta_yz = jnp.array([[0.5, -0.3]], dtype=jnp.float32)

        _, _, c_yz = apply_drift_corrections(
            time, yz, zeros_t, delta_yz, 1.6)

        np.testing.assert_allclose(c_yz[0], [1.5, 1.7], atol=1e-5)

    def test_jit_compatible(self):
        """All drift functions should work under jax.jit."""
        positions = jnp.array([[-10.0, 0.0, 0.0]], dtype=jnp.float32)

        dist, time, yz = jax.jit(
            lambda p: compute_drift_to_plane(p, -20.0, -1, 1.6, 0.6)
        )(positions)

        assert dist.shape == (1,)
        assert jnp.isfinite(dist[0])
