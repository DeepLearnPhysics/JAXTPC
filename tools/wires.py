import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(3, 4))
def _calculate_single_plane_wire_distances_jit(
    positions_yz_centered_cm, angle_rad, wire_spacing_cm, max_wire_idx_abs, index_offset
):
    """JIT helper for batched calculation of closest wire distances using broadcasting."""
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)

    # Extract y and z components from positions
    P_y_cm = positions_yz_centered_cm[:, 0]  # Shape: (n_hits,)
    P_z_cm = positions_yz_centered_cm[:, 1]  # Shape: (n_hits,)

    # Calculate r_prime (the wire coordinate) for all positions
    r_prime = P_y_cm * cos_theta + P_z_cm * sin_theta  # Shape: (n_hits,)

    # Calculate index and distance to closest wire
    idx_rel_floor = jnp.floor(r_prime / (wire_spacing_cm + 1e-9) - 1e-9)  # Shape: (n_hits,)
    idx_rel_ceil = jnp.ceil(r_prime / (wire_spacing_cm + 1e-9) + 1e-9)    # Shape: (n_hits,)

    dist_floor = r_prime - idx_rel_floor * wire_spacing_cm  # Shape: (n_hits,)
    dist_ceil = r_prime - idx_rel_ceil * wire_spacing_cm    # Shape: (n_hits,)

    is_floor_closer = jnp.abs(dist_floor) <= jnp.abs(dist_ceil)  # Shape: (n_hits,)

    closest_idx_rel = jnp.where(is_floor_closer, idx_rel_floor, idx_rel_ceil)  # Shape: (n_hits,)
    closest_distances = jnp.where(is_floor_closer, dist_floor, dist_ceil)      # Shape: (n_hits,)

    closest_indices_abs = (closest_idx_rel.astype(jnp.int32) + index_offset)   # Shape: (n_hits,)

    return closest_indices_abs, closest_distances


@jax.jit
def calculate_diffusion_response_normalized(
    wire_distance_cm, time_difference_us, drift_time_us,
    longitudinal_diffusion_cm2_us, transverse_diffusion_cm2_us, drift_velocity_cm_us
):
    """Calculates normalized 2D Gaussian response without charge scaling."""
    # Calculate drift-dependent sigmas with diffusion
    # σ² = 2Dt (where t is drift_time)

    # Spatial diffusion (transverse)
    sigma_wire_squared = 2.0 * transverse_diffusion_cm2_us * drift_time_us

    # Time diffusion (longitudinal) - convert from spatial to time units
    longitudinal_diffusion_us2_us = longitudinal_diffusion_cm2_us / (drift_velocity_cm_us ** 2)
    sigma_time_squared = 2.0 * longitudinal_diffusion_us2_us * drift_time_us

    # Ensure minimum sigma values
    min_sigma = 1e-4
    sigma_wire_squared = jnp.maximum(sigma_wire_squared, min_sigma ** 2)
    sigma_time_squared = jnp.maximum(sigma_time_squared, min_sigma ** 2)

    # Calculate Gaussian terms
    wire_term = -(wire_distance_cm**2) / (2.0 * sigma_wire_squared)
    time_term = -(time_difference_us**2) / (2.0 * sigma_time_squared)

    # Calculate normalization factor
    norm_factor = 1.0 / (2.0 * jnp.pi * jnp.sqrt(sigma_wire_squared) * jnp.sqrt(sigma_time_squared))

    # Apply Gaussian formula
    response = norm_factor * jnp.exp(wire_term + time_term)

    return jnp.maximum(response, 0.0)


@jax.jit
def calculate_angular_scaling(theta_xz, theta_y):
    """
    Calculate angular scaling factors based on angles to wire and plane.

    Args:
        theta_xz: Angle to wire in the xz-plane (in radians)
        theta_y: Angle to wire plane (in radians)

    Returns:
        scaling_factor: Scaling factor for signal calculation
    """
    # Calculate the scaling factor based on angles
    scaling_factor = (jnp.cos(theta_xz) * jnp.sin(theta_y))
    return scaling_factor

calculate_angular_scaling_vmap = jax.vmap(
    calculate_angular_scaling, in_axes=(0, 0)
)

@jax.jit
def calculate_segment_wire_angles(theta, phi, wire_angle):
    """
    Calculate angles between an segment and a wire/plane.

    Args:
        theta: Polar angle (0 to π) from the positive z-axis
        phi: Azimuthal angle (-π to π) from the positive x-axis
        wire_angle: Angle of the wire in the yz-plane (in radians)

    Returns:
        angle_to_wire (theta_xz): Acute angle between segment and wire (in radians)
        angle_to_plane (theta_y): Acute angle between segment and wire plane (in radians)
    """
    # Calculate segment direction vector
    dx = jnp.sin(theta) * jnp.cos(phi)
    dy = jnp.sin(theta) * jnp.sin(phi)
    dz = jnp.cos(theta)

    # Calculate wire direction vector
    wire_dy = jnp.cos(wire_angle)
    wire_dz = jnp.sin(wire_angle)

    # Calculate dot product for segment-to-wire angle
    dot_product = dy * wire_dy + dz * wire_dz

    # For undirected lines, use absolute value of dot product
    # This gives the acute angle between lines (0-90°)
    dot_product_abs = jnp.abs(dot_product)
    dot_product_clipped = jnp.clip(dot_product_abs, 0.0, 1.0)
    angle_to_wire = jnp.arccos(dot_product_clipped)

    # Calculate angle to plane (always the acute angle)
    dx_abs = jnp.abs(dx)
    dx_clipped = jnp.clip(dx_abs, 0.0, 1.0)
    angle_to_plane = jnp.arccos(dx_clipped)

    # theta_xz, theta_y
    return angle_to_wire, angle_to_plane


# Vectorized version that can handle arrays of inputs
calculate_segment_wire_angles_vmap = jax.vmap(
    calculate_segment_wire_angles, in_axes=(0, 0, None)
)

@partial(jax.jit, static_argnames=['K_wire', 'K_time', 'num_angles', 'num_wire_distances', 'num_wires', 'num_time_steps'])
def prepare_segment_modified(
    charge, drift_time_us, drift_distance_cm, closest_index_abs, closest_distance,
    attenuation_factor, theta_xz_rad, theta_y_rad, angular_scaling_factor, valid_hit,
    K_wire, K_time, wire_spacing_cm, time_step_size_us,
    longitudinal_diffusion_cm2_us, transverse_diffusion_cm2_us, drift_velocity_cm_us,
    num_angles, num_wire_distances, min_idx_abs, num_wires, num_time_steps
):
    """Process a single hit with detailed intermediate results for visualization."""
    # Only process if valid hit
    charge = jnp.where(valid_hit, charge, 0.0)

    # 1. Calculate time bins and offsets
    central_time_bin = jnp.floor(drift_time_us / time_step_size_us).astype(jnp.int32)
    half_k_time = (K_time - 1) // 2
    time_bin_offsets = jnp.arange(-half_k_time, half_k_time + 1)
    time_bins = central_time_bin + time_bin_offsets
    bin_center_times = (time_bins + 0.5) * time_step_size_us
    time_differences_us = drift_time_us - bin_center_times  # Shape: (K_time,)

    # 2. Calculate wire indices and distances
    half_k_wire = (K_wire - 1) // 2
    relative_indices = jnp.arange(-half_k_wire, K_wire - half_k_wire)
    wire_indices = closest_index_abs + relative_indices  # Shape: (K_wire,)
    wire_distances_cm = closest_distance + relative_indices * wire_spacing_cm  # Shape: (K_wire,)

    # 3. Apply charge scaling and attenuation
    charge_scaled = charge * angular_scaling_factor
    attenuated_charge = charge_scaled * attenuation_factor

    # 4. Calculate diffusion response array (K_wire x K_time)
    # Reshape for broadcasting
    wire_distances_2d = jnp.expand_dims(jnp.abs(wire_distances_cm), axis=1)  # Shape: (K_wire, 1)
    time_differences_2d = jnp.expand_dims(time_differences_us, axis=0)  # Shape: (1, K_time)

    # Calculate normalized diffusion response (without charge)
    diffusion_response_normalized = calculate_diffusion_response_normalized(
        wire_distances_2d,
        time_differences_2d,
        drift_time_us,
        longitudinal_diffusion_cm2_us,
        transverse_diffusion_cm2_us,
        drift_velocity_cm_us
    )  # Shape: (K_wire, K_time)

    # Apply charge for full diffusion response
    diffusion_response = diffusion_response_normalized * attenuated_charge

    # 5. Angle interpolation (theta_xz)
    theta_xz_deg = jnp.clip(theta_xz_rad * (180.0 / jnp.pi), 0.0, 90.0)
    angle_step = 90.0 / (num_angles - 1)
    normalized_pos_angle = theta_xz_deg / angle_step
    lower_idx_angle = jnp.clip(jnp.floor(normalized_pos_angle).astype(jnp.int32), 0, num_angles - 2)
    upper_idx_angle = lower_idx_angle + 1
    frac_angle = normalized_pos_angle - lower_idx_angle
    angle_indices = jnp.array([lower_idx_angle, upper_idx_angle])  # Shape: (2,)
    angle_weights = jnp.array([1.0 - frac_angle, frac_angle])  # Shape: (2,)

    # 6. Wire distance interpolation
    wire_dist_wire_units = jnp.abs(wire_distances_cm) / wire_spacing_cm  # Shape: (K_wire,)
    wire_dist_clipped = jnp.clip(wire_dist_wire_units, 0.0, 2.5)
    bin_idx = jnp.clip(jnp.floor(wire_dist_clipped / 0.5).astype(jnp.int32), 0, num_wire_distances - 2)
    lower_idx_dist = bin_idx
    upper_idx_dist = bin_idx + 1
    bin_lower = lower_idx_dist * 0.5
    frac_dist = (wire_dist_clipped - bin_lower) / 0.5

    # Wire distance indices and weights
    wire_dist_indices = jnp.stack([lower_idx_dist, upper_idx_dist], axis=1)  # Shape: (K_wire, 2)
    wire_dist_weights = jnp.stack([1.0 - frac_dist, frac_dist], axis=1)  # Shape: (K_wire, 2)

    # Initialize arrays for broadcasting across 4D
    wire_indices_4d = jnp.zeros((K_wire, K_time, 2, 2), dtype=jnp.int32)
    time_indices_4d = jnp.zeros((K_wire, K_time, 2, 2), dtype=jnp.int32)
    angle_indices_4d = jnp.zeros((K_wire, K_time, 2, 2), dtype=jnp.int32)
    wire_dist_indices_4d = jnp.zeros((K_wire, K_time, 2, 2), dtype=jnp.int32)

    # Prepare 4D arrays of indices - similar to original code
    for w in range(K_wire):
        wire_indices_4d = wire_indices_4d.at[w, :, :, :].set(wire_indices[w])

    for t in range(K_time):
        time_indices_4d = time_indices_4d.at[:, t, :, :].set(time_bins[t])

    for a in range(2):
        angle_indices_4d = angle_indices_4d.at[:, :, a, :].set(angle_indices[a])

    for w in range(K_wire):
        for d in range(2):
            wire_dist_indices_4d = wire_dist_indices_4d.at[w, :, :, d].set(wire_dist_indices[w, d])

    # Calculate signals using broadcasting
    # Reshape for broadcasting
    diffusion_2d = diffusion_response[:, :, jnp.newaxis, jnp.newaxis]  # (K_wire, K_time, 1, 1)
    angle_weights_2d = angle_weights.reshape((1, 1, 2, 1))  # (1, 1, 2, 1)
    wire_dist_weights_4d = wire_dist_weights.reshape((K_wire, 1, 1, 2))  # (K_wire, 1, 1, 2)

    # Compute signals with broadcasting
    signal_values_4d = diffusion_2d * angle_weights_2d * wire_dist_weights_4d

    # Create validity mask
    wire_valid = (wire_indices_4d >= 0) & (wire_indices_4d < num_wires)
    time_valid = (time_indices_4d >= 0) & (time_indices_4d < num_time_steps)

    # Combine validity masks
    valid_mask = wire_valid & time_valid & valid_hit

    # Apply validity mask to zero out invalid entries
    wire_indices_rel = jnp.where(valid_mask, wire_indices_4d - min_idx_abs, 0).reshape(-1)
    time_indices_out = jnp.where(valid_mask, time_indices_4d, 0).reshape(-1)
    angle_indices_out = jnp.where(valid_mask, angle_indices_4d, 0).reshape(-1)
    wire_dist_indices_out = jnp.where(valid_mask, wire_dist_indices_4d, 0).reshape(-1)
    signal_values_out = jnp.where(valid_mask, signal_values_4d, 0.0).reshape(-1)

    # Return processed indices (already with min_idx offset) and values
    return wire_indices_rel, time_indices_out, angle_indices_out, wire_dist_indices_out, signal_values_out


@partial(jax.jit, static_argnames=["num_wires", "num_time_steps", "num_angles", "num_wire_distances"])
def fill_signals_array(indices_and_values, num_wires, num_time_steps, num_angles, num_wire_distances):
    """
    Simple function to fill output array with calculated signal values.

    Args:
        indices_and_values: Tuple of (wire_indices, time_indices, angle_indices,
                           wire_dist_indices, signal_values)
        output_shape: Shape of the output array

    Returns:
        Filled signals array
    """
    wire_indices, time_indices, angle_indices, wire_dist_indices, signal_values = indices_and_values

    # Create output array
    wire_signals = jnp.zeros((num_wires, num_time_steps, num_angles, num_wire_distances))

    # Fill array using scatter_add
    wire_signals = wire_signals.at[
        wire_indices, time_indices, angle_indices, wire_dist_indices
    ].add(signal_values)

    return wire_signals