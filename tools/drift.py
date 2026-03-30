"""
Drift physics calculations for LArTPC simulation.

Provides JIT-compiled functions for calculating electron drift times,
distances, and space charge corrections.
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1,))
def compute_drift_to_plane(positions_cm, detector_half_width_x, drift_velocity_cm_us, plane_dist_from_anode_cm):
    """
    Calculate the drift time and distance to a single plane for each position.

    Parameters
    ----------
    positions_cm : jnp.ndarray
        Array of shape (N, 3) containing the (x, y, z) positions in cm.
    detector_half_width_x : float
        Half-width of the detector in the x-direction in cm.
    drift_velocity_cm_us : float
        Drift velocity in cm/μs.
    plane_dist_from_anode_cm : float
        Distance of the plane from the anode in cm.

    Returns
    -------
    drift_distance_cm : jnp.ndarray
        Array of shape (N,) containing the drift distances in cm.
    drift_time_us : jnp.ndarray
        Array of shape (N,) containing the drift times in μs.
    positions_yz_cm : jnp.ndarray
        Array of shape (N, 2) containing the (y, z) positions in cm.
    """
    x = positions_cm[:, 0]
    positions_yz_cm = positions_cm[:, 1:3]
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
def correct_drift_for_plane(drift_distance_cm, drift_time_us, drift_velocity_cm_us, plane_dist_difference_cm):
    """
    Correct drift time/distance for planes relative to the furthest plane.
    The correction is subtracted because planes closer to the anode have less drift distance.

    Parameters
    ----------
    drift_distance_cm : jnp.ndarray
        Array of shape (N,) containing the drift distances to the furthest plane in cm.
    drift_time_us : jnp.ndarray
        Array of shape (N,) containing the drift times to the furthest plane in μs.
    drift_velocity_cm_us : float
        Drift velocity in cm/μs.
    plane_dist_difference_cm : float
        Distance difference between the furthest plane and this plane in cm.
        Positive value means this plane is closer to the anode than the furthest plane.

    Returns
    -------
    corrected_drift_distance_cm : jnp.ndarray
        Array of shape (N,) containing the corrected drift distances in cm.
    corrected_drift_time_us : jnp.ndarray
        Array of shape (N,) containing the corrected drift times in μs.
    """
    drift_time_correction = jnp.where(drift_velocity_cm_us > 1e-9,
                                      plane_dist_difference_cm / drift_velocity_cm_us,
                                      jnp.inf)

    corrected_drift_distance_cm = drift_distance_cm - plane_dist_difference_cm
    corrected_drift_time_us = drift_time_us - drift_time_correction

    # Ensure we don't get negative distances/times due to correction
    corrected_drift_distance_cm = jnp.maximum(corrected_drift_distance_cm, 0.0)
    corrected_drift_time_us = jnp.maximum(corrected_drift_time_us, 0.0)

    return corrected_drift_distance_cm, corrected_drift_time_us


@jax.jit
def apply_drift_corrections(drift_distance_cm, drift_time_us, positions_yz_cm,
                             delta_x_cm, delta_y_cm, delta_z_cm,
                             velocity_cm_us):
    """
    Apply space charge drift corrections to nominal drift quantities.

    Parameters
    ----------
    drift_distance_cm : jnp.ndarray, shape (N,)
        Nominal drift distances in cm.
    drift_time_us : jnp.ndarray, shape (N,)
        Nominal drift times in us.
    positions_yz_cm : jnp.ndarray, shape (N, 2)
        Nominal (y, z) positions at wire planes in cm.
    delta_x_cm : jnp.ndarray, shape (N,)
        Longitudinal correction in cm (positive = longer apparent drift).
    delta_y_cm : jnp.ndarray, shape (N,)
        Transverse y correction in cm.
    delta_z_cm : jnp.ndarray, shape (N,)
        Transverse z correction in cm.
    velocity_cm_us : float
        Nominal drift velocity in cm/us.

    Returns
    -------
    corrected_distance_cm : jnp.ndarray, shape (N,)
    corrected_time_us : jnp.ndarray, shape (N,)
    corrected_yz_cm : jnp.ndarray, shape (N, 2)
    """
    corrected_distance = jnp.maximum(drift_distance_cm + delta_x_cm, 0.0)
    corrected_time = jnp.maximum(
        drift_time_us + delta_x_cm / jnp.maximum(velocity_cm_us, 1e-9),
        0.0,
    )
    corrected_yz = positions_yz_cm + jnp.stack([delta_y_cm, delta_z_cm], axis=-1)
    return corrected_distance, corrected_time, corrected_yz
