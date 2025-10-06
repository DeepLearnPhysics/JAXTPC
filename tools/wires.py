import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(3, 4))
def _calculate_single_plane_wire_distances_jit(
    positions_yz_centered_cm, angle_rad, wire_spacing_cm, max_wire_idx_abs, index_offset
):
    """
    Calculate the closest wire indices and distances for each hit in a detector plane.
    
    Parameters
    ----------
    positions_yz_centered_cm : jnp.ndarray
        Array of shape (n_hits, 2) containing the (y, z) positions in cm.
    angle_rad : float
        Wire angle in radians, measured in the YZ plane.
    wire_spacing_cm : float
        Spacing between wires in cm.
    max_wire_idx_abs : int
        Maximum absolute wire index.
    index_offset : int
        Wire index offset.
        
    Returns
    -------
    closest_indices_abs : jnp.ndarray
        Array of shape (n_hits,) containing the absolute indices of the closest wires.
    closest_distances : jnp.ndarray
        Array of shape (n_hits,) containing the distances to the closest wires in cm.
    """
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)

    # Extract y and z components from positions
    P_y_cm = positions_yz_centered_cm[:, 0]  # Shape: (n_hits,)
    P_z_cm = positions_yz_centered_cm[:, 1]  # Shape: (n_hits,)

    # Calculate r_prime (the wire coordinate) for all positions
    r_prime = P_y_cm * sin_theta + P_z_cm * cos_theta  # Shape: (n_hits,)

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


@partial(jax.jit, static_argnums=(3,))
def calculate_k_nearest_wires(positions_yz_cm, angle_rad, wire_spacing_cm,
                              K, max_wire_idx_abs, index_offset):
    """
    Calculate K nearest wire indices and distances for each hit in a detector plane.

    Parameters
    ----------
    positions_yz_cm : jnp.ndarray
        Array of shape (n_hits, 2) containing the (y, z) positions in cm.
    angle_rad : float
        Wire angle in radians, measured in the YZ plane.
    wire_spacing_cm : float
        Spacing between wires in cm.
    K : int
        Number of nearest wires to find.
    max_wire_idx_abs : int
        Maximum absolute wire index.
    index_offset : int
        Wire index offset.

    Returns
    -------
    wire_indices : jnp.ndarray
        Array of shape (n_hits, K) containing the indices of the K nearest wires.
    wire_distances : jnp.ndarray
        Array of shape (n_hits, K) containing the distances to K nearest wires.
    """
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)

    # Extract y and z components
    P_y_cm = positions_yz_cm[:, 0]
    P_z_cm = positions_yz_cm[:, 1]

    # Calculate r_prime (the wire coordinate) for all positions
    r_prime = P_y_cm * sin_theta + P_z_cm * cos_theta

    # Find the closest wire index
    closest_idx_rel = jnp.round(r_prime / wire_spacing_cm).astype(jnp.int32)
    closest_idx_abs = closest_idx_rel + index_offset

    # Calculate how many wires to take on each side
    half_k = (K - 1) // 2

    # Create offsets array
    offsets = jnp.arange(-half_k, K - half_k)

    # Calculate indices for K nearest wires using broadcasting
    # Shape: (n_hits, 1) + (K,) -> (n_hits, K)
    wire_indices = closest_idx_abs[:, jnp.newaxis] + offsets

    # Calculate distances for each wire
    # Shape: (n_hits, 1) - (n_hits, K) * scalar -> (n_hits, K)
    wire_r_values = (wire_indices - index_offset) * wire_spacing_cm
    wire_distances = r_prime[:, jnp.newaxis] - wire_r_values

    # Replace out-of-bounds indices with -1
    valid_mask = (wire_indices >= 0) & (wire_indices <= max_wire_idx_abs)
    wire_indices = jnp.where(valid_mask, wire_indices, -1)
    wire_distances = jnp.where(valid_mask, wire_distances, jnp.nan)

    return wire_indices, wire_distances


@jax.jit
def calculate_diffusion_response_normalized(
    wire_distance_cm, time_difference_us, drift_time_us,
    longitudinal_diffusion_cm2_us, transverse_diffusion_cm2_us, drift_velocity_cm_us
):
    """
    Calculate normalized 2D Gaussian response for charge diffusion without charge scaling.
    
    Parameters
    ----------
    wire_distance_cm : jnp.ndarray
        Distance from the hit to the wire in cm.
    time_difference_us : jnp.ndarray
        Time difference between drift time and bin center in μs.
    drift_time_us : float
        Drift time in μs.
    longitudinal_diffusion_cm2_us : float
        Longitudinal diffusion coefficient in cm²/μs.
    transverse_diffusion_cm2_us : float
        Transverse diffusion coefficient in cm²/μs.
    drift_velocity_cm_us : float
        Drift velocity in cm/μs.
        
    Returns
    -------
    response : jnp.ndarray
        Normalized diffusion response.
    """
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
def calculate_wire_diffusion_normalized(
    wire_distance_cm, drift_time_us, transverse_diffusion_cm2_us
):
    """
    Calculate normalized 1D Gaussian response for wire (transverse) diffusion.
    
    Parameters
    ----------
    wire_distance_cm : jnp.ndarray
        Distance from the hit to the wire in cm.
    drift_time_us : float
        Drift time in μs.
    transverse_diffusion_cm2_us : float
        Transverse diffusion coefficient in cm²/μs.
        
    Returns
    -------
    response : jnp.ndarray
        Normalized wire diffusion response.
    """
    # Spatial diffusion (transverse)
    sigma_wire_squared = 2.0 * transverse_diffusion_cm2_us * drift_time_us

    # Ensure minimum sigma value
    min_sigma = 1e-4
    sigma_wire_squared = jnp.maximum(sigma_wire_squared, min_sigma ** 2)

    # Calculate Gaussian term
    wire_term = -(wire_distance_cm**2) / (2.0 * sigma_wire_squared)

    # Calculate normalization factor for 1D Gaussian
    norm_factor = 1.0 / (jnp.sqrt(2.0 * jnp.pi * sigma_wire_squared))

    # Apply Gaussian formula
    response = norm_factor * jnp.exp(wire_term)

    return jnp.maximum(response, 0.0)


@jax.jit
def calculate_time_diffusion_normalized(
    time_difference_us, drift_time_us,
    longitudinal_diffusion_cm2_us, drift_velocity_cm_us
):
    """
    Calculate normalized 1D Gaussian response for time (longitudinal) diffusion.
    
    Parameters
    ----------
    time_difference_us : jnp.ndarray
        Time difference between drift time and bin center in μs.
    drift_time_us : float
        Drift time in μs.
    longitudinal_diffusion_cm2_us : float
        Longitudinal diffusion coefficient in cm²/μs.
    drift_velocity_cm_us : float
        Drift velocity in cm/μs.
        
    Returns
    -------
    response : jnp.ndarray
        Normalized time diffusion response.
    """
    # Time diffusion (longitudinal) - convert from spatial to time units
    longitudinal_diffusion_us2_us = longitudinal_diffusion_cm2_us / (drift_velocity_cm_us ** 2)
    sigma_time_squared = 2.0 * longitudinal_diffusion_us2_us * drift_time_us

    # Ensure minimum sigma value
    min_sigma = 1e-4
    sigma_time_squared = jnp.maximum(sigma_time_squared, min_sigma ** 2)

    # Calculate Gaussian term
    time_term = -(time_difference_us**2) / (2.0 * sigma_time_squared)

    # Calculate normalization factor for 1D Gaussian
    norm_factor = 1.0 / (jnp.sqrt(2.0 * jnp.pi * sigma_time_squared))

    # Apply Gaussian formula
    response = norm_factor * jnp.exp(time_term)

    return jnp.maximum(response, 0.0)


@partial(jax.jit, static_argnames=['clipping_value_deg'])
def calculate_angular_scaling(theta_xz, theta_y, clipping_value_deg=5.0):
    """
    Calculate angular scaling factors based on angles to wire and plane.
    
    Parameters
    ----------
    theta_xz : float
        Angle to wire in the xz-plane in radians.
    theta_y : float
        Angle to wire plane in radians.
    clipping_value_deg : float, optional
        Clipping value in degrees to avoid extreme angles, by default 5.0.
        
    Returns
    -------
    scaling_factor : float
        Scaling factor for signal calculation.
    """
    # Clip angles to avoid extreme values
    clipping_value_rad = jnp.radians(clipping_value_deg)

    theta_xz = jnp.abs(theta_xz)
    theta_y = jnp.abs(theta_y)

    theta_xz = jnp.clip(theta_xz, clipping_value_rad, jnp.pi/2 - clipping_value_rad)
    theta_y = jnp.clip(theta_y, clipping_value_rad, jnp.pi/2 - clipping_value_rad)

    scaling_factor = 1/(jnp.cos(theta_xz) * jnp.sin(theta_y))

    return scaling_factor

calculate_angular_scaling_vmap = jax.vmap(
    calculate_angular_scaling, in_axes=(0, 0)
)

@jax.jit
def calculate_segment_wire_angles(theta, phi, wire_angle):
    """
    Calculate angles between a segment and a wire/plane.
    
    Parameters
    ----------
    theta : float
        Polar angle (0 to π) from the positive z-axis in radians.
    phi : float
        Azimuthal angle (-π to π) from the positive x-axis in radians.
    wire_angle : float
        Angle of the wire in the yz-plane in radians.
        
    Returns
    -------
    angle_to_wire : float
        Acute angle between segment and wire in radians (theta_xz).
    angle_to_plane : float
        Acute angle between segment and wire plane in radians (theta_y).
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

@partial(jax.jit, static_argnames=['num_wires'])
def prepare_segment_no_diffusion(
    charge, drift_time_us, closest_index_abs, closest_distance,
    attenuation_factor, valid_hit,
    wire_spacing_cm, time_step_size_us,
    min_idx_abs, num_wires
):
    """
    Process a single hit without diffusion (handled by response kernels).
    
    Parameters
    ----------
    charge : float
        Charge deposited by the hit.
    drift_time_us : float
        Drift time of the hit in μs.
    closest_index_abs : int
        Absolute index of the closest wire.
    closest_distance : float
        Distance to the closest wire in cm.
    attenuation_factor : float
        Attenuation factor due to electron lifetime.
    valid_hit : bool
        Whether the hit is valid.
    wire_spacing_cm : float
        Spacing between wires in cm.
    time_step_size_us : float
        Size of time step in μs.
    min_idx_abs : int
        Minimum absolute wire index.
    num_wires : int
        Number of wires in the plane.
        
    Returns
    -------
    tuple
        Tuple containing indices, offsets, and intensity:
        (wire_index, wire_offset, time_index, time_offset, intensity)
    """
    # Only process if valid hit
    charge = jnp.where(valid_hit, charge, 0.0)

    # Calculate central time bin and offset
    time_index = jnp.floor(drift_time_us / time_step_size_us).astype(jnp.int32)
    time_offset = (drift_time_us / time_step_size_us) - time_index

    # Calculate wire offset (fractional part of wire position)
    wire_offset = jnp.abs(closest_distance) / wire_spacing_cm

    # Apply charge scaling and attenuation
    intensity = charge * attenuation_factor

    # Convert to relative wire index
    wire_index_rel = jnp.where(
        (closest_index_abs >= 0) & (closest_index_abs < num_wires),
        closest_index_abs - min_idx_abs,
        -1  # Invalid index
    )
    
    # Set intensity to 0 for invalid hits
    intensity = jnp.where(
        valid_hit & (wire_index_rel >= 0),
        intensity,
        0.0
    )

    return wire_index_rel, wire_offset, time_index, time_offset, intensity


@partial(jax.jit, static_argnames=['K_wire', 'K_time', 'num_wires', 'num_time_steps'])
def prepare_segment_modified(
    charge, drift_time_us, drift_distance_cm, closest_index_abs, closest_distance,
    attenuation_factor, theta_xz_rad, theta_y_rad, angular_scaling_factor, valid_hit,
    K_wire, K_time, wire_spacing_cm, time_step_size_us,
    longitudinal_diffusion_cm2_us, transverse_diffusion_cm2_us, drift_velocity_cm_us,
    min_idx_abs, num_wires, num_time_steps
):
    """
    Process a single hit with detailed intermediate results for visualization.
    
    Parameters
    ----------
    charge : float
        Charge deposited by the hit.
    drift_time_us : float
        Drift time of the hit in μs.
    drift_distance_cm : float
        Drift distance of the hit in cm.
    closest_index_abs : int
        Absolute index of the closest wire.
    closest_distance : float
        Distance to the closest wire in cm.
    attenuation_factor : float
        Attenuation factor due to electron lifetime.
    theta_xz_rad : float
        Angle to wire in the xz-plane in radians.
    theta_y_rad : float
        Angle to wire plane in radians.
    angular_scaling_factor : float
        Angular scaling factor for the hit.
    valid_hit : bool
        Whether the hit is valid.
    K_wire : int
        Number of wire neighbors to consider.
    K_time : int
        Number of time bins to consider.
    wire_spacing_cm : float
        Spacing between wires in cm.
    time_step_size_us : float
        Size of time step in μs.
    longitudinal_diffusion_cm2_us : float
        Longitudinal diffusion coefficient in cm²/μs.
    transverse_diffusion_cm2_us : float
        Transverse diffusion coefficient in cm²/μs.
    drift_velocity_cm_us : float
        Drift velocity in cm/μs.
    min_idx_abs : int
        Minimum absolute wire index.
    num_wires : int
        Number of wires in the plane.
    num_time_steps : int
        Number of time steps in the simulation.
        
    Returns
    -------
    tuple
        Tuple containing indices and values for the hit:
        (wire_indices_rel, time_indices_out, signal_values_out)
    """
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

    # Calculate normalized wire diffusion response
    wire_diffusion = calculate_wire_diffusion_normalized(
        wire_distances_2d,
        drift_time_us,
        transverse_diffusion_cm2_us
    )  # Shape: (K_wire, 1)
    
    # Calculate normalized time diffusion response
    time_diffusion = calculate_time_diffusion_normalized(
        time_differences_2d,
        drift_time_us,
        longitudinal_diffusion_cm2_us,
        drift_velocity_cm_us
    )  # Shape: (1, K_time)
    
    # Combine wire and time diffusion responses using broadcasting
    diffusion_response_normalized = wire_diffusion * time_diffusion  # Shape: (K_wire, K_time)
    
    # Apply charge for full diffusion response
    diffusion_response = diffusion_response_normalized * attenuated_charge

    # Create validity mask
    wire_valid = (wire_indices >= 0) & (wire_indices < num_wires)
    time_valid = (time_bins >= 0) & (time_bins < num_time_steps)

    # Combine validity masks - expand to 2D
    wire_valid_2d = wire_valid[:, jnp.newaxis]  # Shape: (K_wire, 1)
    time_valid_2d = time_valid[jnp.newaxis, :]  # Shape: (1, K_time)
    valid_mask_2d = wire_valid_2d & time_valid_2d & valid_hit  # Shape: (K_wire, K_time)

    # Apply validity mask to zero out invalid entries
    wire_indices_rel = jnp.where(wire_valid, wire_indices - min_idx_abs, 0)
    time_indices_out = time_bins
    signal_values_2d = jnp.where(valid_mask_2d, diffusion_response, 0.0)

    # Flatten the arrays for output
    wire_indices_rel_flat = jnp.repeat(wire_indices_rel, K_time)
    time_indices_out_flat = jnp.tile(time_indices_out, K_wire)
    signal_values_out = signal_values_2d.reshape(-1)

    # Return processed indices and values
    return wire_indices_rel_flat, time_indices_out_flat, signal_values_out


@partial(jax.jit, static_argnames=["num_wires", "num_time_steps"])
def fill_signals_array(indices_and_values, num_wires, num_time_steps):
    """
    Fill output array with calculated signal values.
    
    Parameters
    ----------
    indices_and_values : tuple
        Tuple of (wire_indices, time_indices, signal_values).
    num_wires : int
        Number of wires in the plane.
    num_time_steps : int
        Number of time steps in the simulation.
        
    Returns
    -------
    jnp.ndarray
        Filled signals array of shape (num_wires, num_time_steps).
    """
    wire_indices, time_indices, signal_values = indices_and_values

    # Create output array
    wire_signals = jnp.zeros((num_wires, num_time_steps))

    # Fill array using scatter_add
    wire_signals = wire_signals.at[
        wire_indices, time_indices
    ].add(signal_values)

    return wire_signals


@partial(jax.jit, static_argnames=('num_wires', 'num_time_steps', 'kernel_num_wires', 'kernel_height'))
def fill_signals_from_kernels(wire_indices, time_indices, intensities, contributions,
                            num_wires, num_time_steps, kernel_num_wires, kernel_height):
    """
    Fill signals array from kernel contributions for multiple segments.
    
    This function efficiently accumulates kernel contributions from multiple segments
    into a signals array without explicit loops, using JAX's vectorized operations.
    
    Args:
        wire_indices: (N,) center wire index for each segment
        time_indices: (N,) start time index for each segment
        intensities: (N,) intensity scaling factor for each segment
        contributions: (N, kernel_num_wires, kernel_height) kernel response for each segment
        num_wires: Total number of wires in output (static)
        num_time_steps: Total number of time steps in output (static)
        kernel_num_wires: Number of wires in kernel (static)
        kernel_height: Number of time bins in kernel (static)
    
    Returns:
        signals: (num_wires, num_time_steps) accumulated wire signals
    """
    # Initialize output array
    signals = jnp.zeros((num_wires, num_time_steps))
    
    # Calculate wire offset to center kernel on wire_indices
    wire_offset = kernel_num_wires // 2
    
    # Create kernel index offsets (reused for all segments)
    kernel_wire_offsets = jnp.arange(kernel_num_wires)  # shape: (kernel_num_wires,)
    kernel_time_offsets = jnp.arange(kernel_height)     # shape: (kernel_height,)
    
    # Compute absolute wire positions for all segments
    # Shape: (N, kernel_num_wires)
    wire_positions = wire_indices[:, None] - wire_offset + kernel_wire_offsets[None, :]
    
    # Compute absolute time positions for all segments  
    # Shape: (N, kernel_height)
    time_positions = time_indices[:, None] + kernel_time_offsets[None, :]
    
    # Scale all contributions by their intensities
    # Shape: (N, kernel_num_wires, kernel_height)
    scaled_contributions = contributions * intensities[:, None, None]
    
    # Create flattened indices for scatter operation
    # Flatten wire positions: (N * kernel_num_wires,)
    flat_wire_indices = wire_positions.reshape(-1)
    
    # Create corresponding time indices for each wire
    # We need to broadcast time_positions to match wire dimensions
    # Shape: (N, kernel_num_wires, kernel_height) -> (N * kernel_num_wires * kernel_height,)
    time_positions_broadcast = jnp.broadcast_to(
        time_positions[:, None, :], 
        (wire_indices.shape[0], kernel_num_wires, kernel_height)
    )
    flat_time_indices = time_positions_broadcast.reshape(-1)
    
    # Flatten contributions
    # Shape: (N * kernel_num_wires * kernel_height,)
    flat_contributions = scaled_contributions.reshape(-1)
    
    # Create 2D indices for scatter
    # Wire indices need to be repeated for each time bin
    wire_indices_repeated = jnp.repeat(flat_wire_indices, kernel_height)
    
    # Use .at[].add() to accumulate all contributions at once
    # mode='drop' will silently ignore out-of-bounds indices
    signals = signals.at[wire_indices_repeated, flat_time_indices].add(flat_contributions, mode='drop')
    
    return signals


# @partial(jax.jit, static_argnames=["num_wires", "num_time_steps", "num_angles", "num_wire_distances"])
# def fill_signals_array(indices_and_values, num_wires, num_time_steps, num_angles, num_wire_distances):
#     """Represent as sparse matrix and convert to dense at the end"""
#     # Calculate strides (following row-major ordering)
#     # For a 4D array with dimensions [W, T, A, D]
#     # stride_W = T * A * D
#     # stride_T = A * D
#     # stride_A = D
#     # stride_D = 1 (implicit)

#     wire_indices, time_indices, angle_indices, wire_dist_indices, signal_values = indices_and_values

#     wire_indices = wire_indices.reshape(-1)
#     time_indices = time_indices.reshape(-1)
#     angle_indices = angle_indices.reshape(-1)
#     wire_dist_indices = wire_dist_indices.reshape(-1)
#     signal_values = signal_values.reshape(-1)
    
#     stride_time = num_angles * num_wire_distances
#     stride_wire = num_time_steps * stride_time
    
#     # Convert 4D indices to 1D indices
#     linear_indices = (
#         wire_indices * stride_wire +
#         time_indices * stride_time +
#         angle_indices * num_wire_distances +
#         wire_dist_indices
#     )
    
#     # Use segment_sum to aggregate values
#     total_size = num_wires * num_time_steps * num_angles * num_wire_distances
#     output_flat = jax.ops.segment_sum(
#         signal_values, linear_indices, total_size
#     )
    
#     # Reshape to final dimensions
#     output = output_flat.reshape((num_wires, num_time_steps, num_angles, num_wire_distances))
#     return output