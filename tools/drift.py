"""
Drift physics calculations for LArTPC simulation.

Provides JIT-compiled functions for calculating electron drift times,
distances, and space charge corrections.
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1, 2))
def compute_drift_to_plane(positions_cm, x_anode_cm, drift_direction,
                           drift_velocity_cm_us, plane_dist_from_anode_cm):
    """
    Calculate the drift time and distance to a single plane for each position.

    Parameters
    ----------
    positions_cm : jnp.ndarray
        Array of shape (N, 3) containing the (x, y, z) positions in cm.
    x_anode_cm : float
        X-position of the anode for this volume (static).
    drift_direction : int
        +1 or -1 indicating drift direction (static).
        +1: electrons drift toward +x (anode at x_max).
        -1: electrons drift toward -x (anode at x_min).
    drift_velocity_cm_us : float
        Drift velocity in cm/us.
    plane_dist_from_anode_cm : float
        Distance of the plane from the anode in cm.

    Returns
    -------
    drift_distance_cm : jnp.ndarray
        Array of shape (N,) containing the drift distances in cm.
    drift_time_us : jnp.ndarray
        Array of shape (N,) containing the drift times in us.
    positions_yz_cm : jnp.ndarray
        Array of shape (N, 2) containing the (y, z) positions in cm.
    """
    x = positions_cm[:, 0]
    positions_yz_cm = positions_cm[:, 1:3]

    # Plane is offset from anode inward (toward cathode):
    #   dir == -1: anode at x_min, plane at x_anode + dist (toward +x)
    #   dir == +1: anode at x_max, plane at x_anode - dist (toward -x)
    plane_x = x_anode_cm - drift_direction * plane_dist_from_anode_cm

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
        Array of shape (N,) containing the drift times to the furthest plane in us.
    drift_velocity_cm_us : float
        Drift velocity in cm/us.
    plane_dist_difference_cm : float
        Distance difference between the furthest plane and this plane in cm.

    Returns
    -------
    corrected_drift_distance_cm : jnp.ndarray
    corrected_drift_time_us : jnp.ndarray
    """
    drift_time_correction = jnp.where(drift_velocity_cm_us > 1e-9,
                                      plane_dist_difference_cm / drift_velocity_cm_us,
                                      jnp.inf)

    corrected_drift_distance_cm = drift_distance_cm - plane_dist_difference_cm
    corrected_drift_time_us = drift_time_us - drift_time_correction

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
    drift_time_us : jnp.ndarray, shape (N,)
    positions_yz_cm : jnp.ndarray, shape (N, 2)
    delta_x_cm : jnp.ndarray, shape (N,)
    delta_y_cm : jnp.ndarray, shape (N,)
    delta_z_cm : jnp.ndarray, shape (N,)
    velocity_cm_us : float

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
