import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def _calculate_single_plane_drift_jit(positions_cm, detector_half_width_x, drift_velocity_cm_us, plane_dist_from_anode_cm):
    """JIT helper for drift time/distance to a single plane."""
    x = positions_cm[:, 0]
    positions_yz_cm = positions_cm[:, 1:3]  # Shape: (n_hits, 2)
    is_left_side = x < 0
    left_anode_x = -detector_half_width_x
    right_anode_x = detector_half_width_x
    plane_x_left = left_anode_x + plane_dist_from_anode_cm
    plane_x_right = right_anode_x - plane_dist_from_anode_cm
    plane_x = jnp.where(is_left_side, plane_x_left, plane_x_right)
    drift_distance_cm = jnp.abs(x - plane_x)
    drift_time_us = jnp.where(drift_velocity_cm_us > 1e-9,
                              drift_distance_cm / drift_velocity_cm_us,
                              jnp.inf)
    return drift_distance_cm, drift_time_us, positions_yz_cm


@jax.jit
def _calculate_single_plane_drift_correction(drift_distance_cm, drift_time_us, drift_velocity_cm_us, plane_dist_difference_cm):
    """
    JIT helper to correct drift time/distance for planes relative to the furthest plane.
    The correction is subtracted because planes closer to the anode have less drift distance.
    """
    # A positive plane_dist_difference_cm means this plane is closer to the anode than the furthest plane
    drift_distance_correction = plane_dist_difference_cm
    drift_time_correction = jnp.where(drift_velocity_cm_us > 1e-9,
                                    drift_distance_correction / drift_velocity_cm_us,
                                    jnp.inf)

    corrected_drift_distance_cm = drift_distance_cm - drift_distance_correction
    corrected_drift_time_us = drift_time_us - drift_time_correction

    # Ensure we don't get negative distances/times due to correction
    corrected_drift_distance_cm = jnp.maximum(corrected_drift_distance_cm, 0.0)
    corrected_drift_time_us = jnp.maximum(corrected_drift_time_us, 0.0)

    return corrected_drift_distance_cm, corrected_drift_time_us


@jax.jit
def calculate_drift_attenuation(drift_distance_cm, drift_velocity_cm_us, electron_lifetime_ms):
    """
    Calculate charge attenuation due to electron lifetime.
    """
    # Convert electron lifetime from ms to µs for consistent units
    electron_lifetime_us = electron_lifetime_ms * 1000.0

    # Calculate drift time in µs
    drift_time_us = jnp.where(drift_velocity_cm_us > 1e-9,
                             drift_distance_cm / drift_velocity_cm_us,
                             jnp.inf)

    # Calculate attenuation factor using exponential decay
    attenuation = jnp.exp(-drift_time_us / electron_lifetime_us)

    return attenuation