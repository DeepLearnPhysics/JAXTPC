"""
Wire signal calculations for LArTPC simulation.

This module contains all wire-related calculations for processing charge deposits
into wire signals. It supports two parallel simulation paths:

Sections:
1. SHARED: Wire geometry calculations
2. RESPONSE PATH: Kernel-based signal generation
3. HIT PATH: Direct diffusion calculation
"""

import jax
import jax.numpy as jnp
from functools import partial


# ============================================================================
# SHARED: Wire Geometry
# ============================================================================

@partial(jax.jit, static_argnums=(3, 4))
def compute_wire_distances(
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
    closest_wire_distances : jnp.ndarray
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
    closest_wire_distances = jnp.where(is_floor_closer, dist_floor, dist_ceil)  # Shape: (n_hits,)

    closest_indices_abs = (closest_idx_rel.astype(jnp.int32) + index_offset)   # Shape: (n_hits,)

    return closest_indices_abs, closest_wire_distances


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
def compute_deposit_wire_angles(theta, phi, wire_angle):
    """
    Calculate angles between a deposit segment and a wire/plane.

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
compute_deposit_wire_angles_vmap = jax.vmap(
    compute_deposit_wire_angles, in_axes=(0, 0, None)
)


@partial(jax.jit, static_argnames=['clipping_value_deg'])
def compute_angular_scaling(theta_xz, theta_y, clipping_value_deg=5.0):
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


compute_angular_scaling_vmap = jax.vmap(
    compute_angular_scaling, in_axes=(0, 0)
)


# ============================================================================
# HIT PATH: Direct Diffusion Calculation
# ============================================================================

@jax.jit
def compute_gaussian_diffusion(
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
def compute_transverse_diffusion(
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
def compute_longitudinal_diffusion(
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


@partial(jax.jit, static_argnames=['K_wire', 'K_time', 'num_wires', 'num_time_steps'])
def prepare_deposit_with_diffusion(
    charge, drift_time_us, drift_distance_cm, closest_index_abs, closest_wire_distance,
    attenuation_factor, theta_xz_rad, theta_y_rad, angular_scaling_factor, valid_hit,
    K_wire, K_time, wire_spacing_cm, time_step_size_us,
    longitudinal_diffusion_cm2_us, transverse_diffusion_cm2_us, drift_velocity_cm_us,
    min_idx_abs, num_wires, num_time_steps
):
    """
    Prepare deposit data WITH K_wire x K_time diffusion (hit path).

    This function calculates diffusion directly in a K_wire x K_time grid
    for the hit path. The diffusion is computed without detector
    response to maintain accurate particle-to-signal attribution.

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
    closest_wire_distance : float
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
        Half-width of wire neighbors to consider.
    K_time : int
        Half-width of time bins to consider.
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

    # 1. Calculate time bins and offsets (using K as half-width)
    central_time_bin = jnp.floor(drift_time_us / time_step_size_us).astype(jnp.int32)
    time_bin_offsets = jnp.arange(-K_time, K_time + 1)  # 2K+1 values
    time_bins = central_time_bin + time_bin_offsets
    bin_center_times = (time_bins + 0.5) * time_step_size_us
    time_differences_us = drift_time_us - bin_center_times  # Shape: (2*K_time+1,)

    # 2. Calculate wire indices and distances (using K as half-width)
    relative_indices = jnp.arange(-K_wire, K_wire + 1)  # 2K+1 values
    wire_indices = closest_index_abs + relative_indices  # Shape: (2*K_wire+1,)
    wire_distances_cm = closest_wire_distance + relative_indices * wire_spacing_cm  # Shape: (2*K_wire+1,)

    # 3. Apply charge scaling and attenuation
    charge_scaled = charge * angular_scaling_factor
    attenuated_charge = charge_scaled * attenuation_factor

    # 4. Calculate diffusion response array ((2*K_wire+1) x (2*K_time+1))
    # Reshape for broadcasting
    wire_distances_2d = jnp.expand_dims(jnp.abs(wire_distances_cm), axis=1)  # Shape: (2*K_wire+1, 1)
    time_differences_2d = jnp.expand_dims(time_differences_us, axis=0)  # Shape: (1, 2*K_time+1)

    # Calculate normalized wire diffusion response
    wire_diffusion = compute_transverse_diffusion(
        wire_distances_2d,
        drift_time_us,
        transverse_diffusion_cm2_us
    )  # Shape: (2*K_wire+1, 1)

    # Calculate normalized time diffusion response
    time_diffusion = compute_longitudinal_diffusion(
        time_differences_2d,
        drift_time_us,
        longitudinal_diffusion_cm2_us,
        drift_velocity_cm_us
    )  # Shape: (1, 2*K_time+1)

    # Combine wire and time diffusion responses using broadcasting
    diffusion_response_normalized = wire_diffusion * time_diffusion  # Shape: (2*K_wire+1, 2*K_time+1)

    # Apply charge for full diffusion response
    diffusion_response = diffusion_response_normalized * attenuated_charge

    # Create validity mask
    wire_valid = (wire_indices >= 0) & (wire_indices < num_wires)
    time_valid = (time_bins >= 0) & (time_bins < num_time_steps)

    # Combine validity masks - expand to 2D
    wire_valid_2d = wire_valid[:, jnp.newaxis]  # Shape: (2*K_wire+1, 1)
    time_valid_2d = time_valid[jnp.newaxis, :]  # Shape: (1, 2*K_time+1)
    valid_mask_2d = wire_valid_2d & time_valid_2d & valid_hit  # Shape: (2*K_wire+1, 2*K_time+1)

    # Apply validity mask to zero out invalid entries
    wire_indices_rel = jnp.where(wire_valid, wire_indices - min_idx_abs, 0)
    time_indices_out = time_bins
    signal_values_2d = jnp.where(valid_mask_2d, diffusion_response, 0.0)

    # Calculate total number of elements
    num_wire_bins = 2 * K_wire + 1
    num_time_bins = 2 * K_time + 1

    # Flatten the arrays for output
    wire_indices_rel_flat = jnp.repeat(wire_indices_rel, num_time_bins)
    time_indices_out_flat = jnp.tile(time_indices_out, num_wire_bins)
    signal_values_out = signal_values_2d.reshape(-1)

    # Return processed indices and values
    return wire_indices_rel_flat, time_indices_out_flat, signal_values_out


@partial(jax.jit, static_argnames=["num_wires", "num_time_steps"])
def accumulate_signals(indices_and_values, num_wires, num_time_steps):
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


# ============================================================================
# RESPONSE PATH: Kernel-based Signal Generation
# ============================================================================

@partial(jax.jit, static_argnames=['num_wires'])
def prepare_deposit_for_response(
    charge, drift_time_us, closest_index_abs, closest_wire_distance,
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
    closest_wire_distance : float
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
    wire_offset = jnp.abs(closest_wire_distance) / wire_spacing_cm

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


@partial(jax.jit, static_argnames=('num_wires', 'num_time_steps', 'kernel_num_wires', 'kernel_height'))
def accumulate_response_signals(wire_indices, time_indices, intensities, contributions,
                                num_wires, num_time_steps, kernel_num_wires, kernel_height):
    """
    Fill signals array from kernel contributions for multiple segments.

    This function efficiently accumulates kernel contributions from multiple segments
    into a signals array without explicit loops, using JAX's vectorized operations.

    Parameters
    ----------
    wire_indices : jnp.ndarray
        (N,) center wire index for each segment.
    time_indices : jnp.ndarray
        (N,) start time index for each segment.
    intensities : jnp.ndarray
        (N,) intensity scaling factor for each segment.
    contributions : jnp.ndarray
        (N, kernel_num_wires, kernel_height) kernel response for each segment.
    num_wires : int
        Total number of wires in output (static).
    num_time_steps : int
        Total number of time steps in output (static).
    kernel_num_wires : int
        Number of wires in kernel (static).
    kernel_height : int
        Number of time bins in kernel (static).

    Returns
    -------
    signals : jnp.ndarray
        (num_wires, num_time_steps) accumulated wire signals.
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


# ============================================================================
# SPARSE BUCKETED ACCUMULATION (for very large detectors)
# ============================================================================

@partial(jax.jit, static_argnames=('B1', 'B2', 'num_wires', 'num_time_steps', 'max_buckets'))
def build_bucket_mapping(wire_indices, time_indices,
                         B1, B2, num_wires, num_time_steps, max_buckets):
    """
    Build point_to_compact mapping for sparse bucketing.

    Phase 1 of the sparse bucketing algorithm. For each segment at (wire, time),
    its kernel can touch up to 4 buckets. This function computes which 4 buckets
    each segment touches and creates a compact mapping to active bucket indices.

    Args:
        wire_indices: (N,) center wire index for each segment
        time_indices: (N,) start time index for each segment
        B1: Bucket size in wire direction (= 2 * kernel_num_wires)
        B2: Bucket size in time direction (= 2 * kernel_height)
        num_wires: Total wires (X dimension)
        num_time_steps: Total time steps (Y dimension)
        max_buckets: Maximum active buckets to allocate

    Returns:
        point_to_compact: (N, 4) int32 - maps (segment, quadrant) to compact index
        num_active: scalar int32 - number of unique active buckets
        compact_to_key: (max_buckets,) int32 - maps compact index to bucket key
    """
    N = wire_indices.shape[0]
    # Use ceiling division to ensure all buckets have unique keys
    # Floor division causes collisions when dimensions don't divide evenly
    NUM_BUCKETS_T = (num_time_steps + B2 - 1) // B2

    # Wire offset: kernel_num_wires // 2, and B1 = 2 * kernel_num_wires
    # So kernel_num_wires = B1 // 2, and wire_offset = B1 // 4
    wire_offset = B1 // 4

    # Home bucket for each segment, based on kernel START position (not center)
    # This must match scatter_contributions_to_buckets which uses (wire_indices - wire_offset)
    home_bw = (wire_indices - wire_offset) // B1  # (N,)
    home_bt = time_indices // B2  # (N,)

    # 4 potential buckets per segment (quadrants)
    # Quadrant layout:
    #   0: (home_bw, home_bt)       - home bucket
    #   1: (home_bw, home_bt + 1)   - right neighbor (time)
    #   2: (home_bw + 1, home_bt)   - bottom neighbor (wire)
    #   3: (home_bw + 1, home_bt + 1) - bottom-right neighbor
    offsets_w = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    offsets_t = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

    bucket_bw = home_bw[:, None] + offsets_w  # (N, 4)
    bucket_bt = home_bt[:, None] + offsets_t  # (N, 4)

    # Pack into single key: key = bw * NUM_BUCKETS_T + bt
    bucket_keys = bucket_bw * NUM_BUCKETS_T + bucket_bt  # (N, 4)

    # Flatten to (4N,) for sorting
    flat_keys = bucket_keys.ravel()  # (4N,)

    # Sort keys to group duplicates together
    sorted_idx = jnp.argsort(flat_keys)
    sorted_keys = flat_keys[sorted_idx]

    # Detect boundaries where key changes (marks unique entries)
    is_new = jnp.concatenate([
        jnp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1]
    ])

    # Assign compact indices: cumsum of is_new gives 1, 1, 1, 2, 2, 3, ...
    # Subtract 1 to get 0-based indices: 0, 0, 0, 1, 1, 2, ...
    compact_idx_sorted = jnp.cumsum(is_new.astype(jnp.int32)) - 1

    # Invert the sort to get compact index for each original position
    compact_idx_flat = jnp.zeros(4 * N, dtype=jnp.int32)
    compact_idx_flat = compact_idx_flat.at[sorted_idx].set(compact_idx_sorted)

    # Reshape back to (N, 4)
    point_to_compact = compact_idx_flat.reshape(N, 4)

    # Count number of active buckets
    num_active = jnp.sum(is_new.astype(jnp.int32))

    # Build reverse mapping: compact_idx -> bucket_key
    compact_to_key = jnp.zeros(max_buckets, dtype=jnp.int32)
    compact_to_key = compact_to_key.at[compact_idx_sorted].set(sorted_keys, mode='drop')

    return point_to_compact, num_active, compact_to_key


@partial(jax.jit, static_argnames=('max_buckets', 'kernel_num_wires', 'kernel_height', 'B1', 'B2'))
def scatter_contributions_to_buckets(
    wire_indices, time_indices, intensities, contributions,
    point_to_compact, max_buckets, kernel_num_wires, kernel_height, B1, B2
):
    """
    Scatter (N, kernel_num_wires, kernel_height) contributions to sparse buckets.

    Phase 2 of the sparse bucketing algorithm. Scatters pre-computed kernel
    contributions to sparse buckets using the mapping from Phase 1.

    Args:
        wire_indices: (N,) center wire index for each segment
        time_indices: (N,) start time index for each segment
        intensities: (N,) intensity scaling factor for each segment
        contributions: (N, kernel_num_wires, kernel_height) kernel response for each segment
        point_to_compact: (N, 4) mapping from (segment, quadrant) to compact index
        max_buckets: Maximum number of buckets (static)
        kernel_num_wires: Number of wires in kernel (static)
        kernel_height: Number of time bins in kernel (static)
        B1: Bucket size in wire direction (static)
        B2: Bucket size in time direction (static)

    Returns:
        (max_buckets, B1, B2) array of sparse bucket contributions
    """
    N = wire_indices.shape[0]
    wire_offset = kernel_num_wires // 2

    # Scale contributions by intensity
    scaled = contributions * intensities[:, None, None]

    # Create index grids
    n_idx = jnp.arange(N)[:, None, None]
    kw_idx = jnp.arange(kernel_num_wires)[None, :, None]
    kt_idx = jnp.arange(kernel_height)[None, None, :]

    # Global coordinates (wire_indices is center, offset to get start)
    gw = (wire_indices[:, None, None] - wire_offset + kw_idx).astype(jnp.int32)
    gt = (time_indices[:, None, None] + kt_idx).astype(jnp.int32)

    # Compute which quadrant each kernel element falls into
    # Home bucket based on kernel start position
    home_bw = (wire_indices[:, None, None] - wire_offset) // B1
    home_bt = time_indices[:, None, None] // B2

    # Cell bucket for each kernel element
    cell_bw = gw // B1
    cell_bt = gt // B2

    # Quadrant offset: 0 or 1 in each direction
    which_w = jnp.clip(cell_bw - home_bw, 0, 1).astype(jnp.int32)
    which_t = jnp.clip(cell_bt - home_bt, 0, 1).astype(jnp.int32)

    # Quadrant index: 0, 1, 2, or 3
    quadrant = which_w * 2 + which_t

    # Gather bucket index from precomputed mapping
    n_idx_expanded = jnp.broadcast_to(n_idx, (N, kernel_num_wires, kernel_height))
    bucket_idx = point_to_compact[n_idx_expanded, quadrant]

    # Local coordinates within bucket
    lw = (gw % B1).astype(jnp.int32)
    lt = (gt % B2).astype(jnp.int32)

    # Scatter to buckets
    output = jnp.zeros((max_buckets, B1, B2), dtype=jnp.float32)
    output = output.at[bucket_idx, lw, lt].add(scaled, mode='drop')

    return output


@partial(jax.jit, static_argnames=('max_buckets', 'kernel_num_wires', 'kernel_height', 'B1', 'B2', 'batch_size'))
def scatter_contributions_to_buckets_batched(
    wire_indices, time_indices, intensities, contributions,
    point_to_compact, max_buckets, kernel_num_wires, kernel_height, B1, B2,
    batch_size=1000
):
    """
    Batched scatter to reduce memory from O(N × K1 × K2) to O(batch_size × K1 × K2).

    Uses jax.lax.scan to process segments in batches, avoiding materialization
    of 14 arrays of shape (N, K1, K2) simultaneously. Use this for large N
    where the original scatter would OOM.

    Args:
        wire_indices: (N,) center wire index for each segment
        time_indices: (N,) start time index for each segment
        intensities: (N,) intensity scaling factor for each segment
        contributions: (N, kernel_num_wires, kernel_height) kernel response
        point_to_compact: (N, 4) mapping from (segment, quadrant) to compact index
        max_buckets: Maximum number of buckets (static)
        kernel_num_wires: Number of wires in kernel (static)
        kernel_height: Number of time bins in kernel (static)
        B1: Bucket size in wire direction (static)
        B2: Bucket size in time direction (static)
        batch_size: Number of segments to process per batch (static)

    Returns:
        (max_buckets, B1, B2) array of sparse bucket contributions
    """
    N = wire_indices.shape[0]
    wire_offset = kernel_num_wires // 2

    # Pad to multiple of batch_size
    n_batches = (N + batch_size - 1) // batch_size
    padded_N = n_batches * batch_size
    pad = padded_N - N

    # Pad all inputs
    wire_indices_pad = jnp.pad(wire_indices, (0, pad), constant_values=0)
    time_indices_pad = jnp.pad(time_indices, (0, pad), constant_values=0)
    intensities_pad = jnp.pad(intensities, (0, pad), constant_values=0.0)
    contributions_pad = jnp.pad(contributions, ((0, pad), (0, 0), (0, 0)), constant_values=0.0)
    point_to_compact_pad = jnp.pad(point_to_compact, ((0, pad), (0, 0)), constant_values=0)
    valid_mask = jnp.arange(padded_N) < N

    # Reshape for batched processing
    wire_batched = wire_indices_pad.reshape(n_batches, batch_size)
    time_batched = time_indices_pad.reshape(n_batches, batch_size)
    int_batched = intensities_pad.reshape(n_batches, batch_size)
    contrib_batched = contributions_pad.reshape(n_batches, batch_size, kernel_num_wires, kernel_height)
    p2c_batched = point_to_compact_pad.reshape(n_batches, batch_size, 4)
    valid_batched = valid_mask.reshape(n_batches, batch_size)

    # Static kernel index grids
    kw_idx = jnp.arange(kernel_num_wires)
    kt_idx = jnp.arange(kernel_height)

    def process_batch(output, batch_data):
        bw, bt, bi, bc, bp, bv = batch_data

        # Scale contributions - shape (batch_size, K1, K2)
        scaled = bc * bi[:, None, None]

        # Global coordinates - shape (batch_size, K1, K2)
        gw = (bw[:, None, None] - wire_offset + kw_idx[None, :, None]).astype(jnp.int32)
        gt = (bt[:, None, None] + kt_idx[None, None, :]).astype(jnp.int32)

        # Home bucket
        home_bw = (bw[:, None, None] - wire_offset) // B1
        home_bt = bt[:, None, None] // B2

        # Cell bucket
        cell_bw = gw // B1
        cell_bt = gt // B2

        # Quadrant offset and index
        which_w = jnp.clip(cell_bw - home_bw, 0, 1).astype(jnp.int32)
        which_t = jnp.clip(cell_bt - home_bt, 0, 1).astype(jnp.int32)
        quadrant = which_w * 2 + which_t

        # Gather bucket index
        batch_idx = jnp.arange(batch_size)[:, None, None]
        batch_idx_expanded = jnp.broadcast_to(batch_idx, (batch_size, kernel_num_wires, kernel_height))
        bucket_idx = bp[batch_idx_expanded, quadrant]

        # Local coordinates within bucket
        lw = (gw % B1).astype(jnp.int32)
        lt = (gt % B2).astype(jnp.int32)

        # Mask invalid segments
        valid_3d = jnp.broadcast_to(bv[:, None, None], (batch_size, kernel_num_wires, kernel_height))
        masked_scaled = jnp.where(valid_3d, scaled, 0.0)

        # Scatter to output
        output = output.at[bucket_idx, lw, lt].add(masked_scaled, mode='drop')

        return output, None

    # Initialize output and run batched scatter
    output = jnp.zeros((max_buckets, B1, B2), dtype=jnp.float32)
    output, _ = jax.lax.scan(
        process_batch,
        output,
        (wire_batched, time_batched, int_batched, contrib_batched, p2c_batched, valid_batched)
    )

    return output


@partial(jax.jit, static_argnames=('num_wires', 'num_time_steps', 'kernel_num_wires',
                                    'kernel_height', 'max_buckets', 'batch_size'))
def accumulate_response_signals_sparse_bucketed(
    wire_indices, time_indices, intensities, contributions,
    num_wires, num_time_steps, kernel_num_wires, kernel_height,
    max_buckets, batch_size=1000
):
    """
    Sparse bucketed accumulation using the two-phase approach.

    Memory-efficient version that only allocates active buckets. Use this
    for large detectors where the dense output array would exceed GPU memory.

    Phase 1: Build bucket mapping (O(N) with sort)
    Phase 2: Batched scatter to sparse buckets (O(N * K1 * K2))

    Args:
        wire_indices: (N,) center wire index for each segment
        time_indices: (N,) start time index for each segment
        intensities: (N,) intensity scaling factor for each segment
        contributions: (N, kernel_num_wires, kernel_height) kernel response for each segment
        num_wires: Total number of wires in output
        num_time_steps: Total number of time steps in output
        kernel_num_wires: Number of wires in kernel
        kernel_height: Number of time bins in kernel
        max_buckets: Maximum number of active buckets to allocate
        batch_size: Batch size for scatter. Recommended: 500-2000.

    Returns:
        buckets: (max_buckets, B1, B2) sparse bucket contributions
        num_active: Number of active buckets
        compact_to_key: (max_buckets,) mapping to original bucket keys
    """
    # Bucket sizes: 2x kernel size
    B1 = 2 * kernel_num_wires
    B2 = 2 * kernel_height

    # Phase 1: Build mapping
    point_to_compact, num_active, compact_to_key = build_bucket_mapping(
        wire_indices, time_indices, B1, B2, num_wires, num_time_steps, max_buckets
    )

    # Phase 2: Batched scatter to sparse buckets
    buckets = scatter_contributions_to_buckets_batched(
        wire_indices, time_indices, intensities, contributions,
        point_to_compact, max_buckets, kernel_num_wires, kernel_height, B1, B2,
        batch_size=batch_size
    )

    return buckets, num_active, compact_to_key


@partial(jax.jit, static_argnames=('B1', 'B2', 'num_wires', 'num_time_steps', 'max_buckets'))
def sparse_buckets_to_dense(buckets, compact_to_key, num_active,
                            B1, B2, num_wires, num_time_steps, max_buckets):
    """
    Convert sparse buckets back to dense (num_wires, num_time_steps) array.

    Args:
        buckets: (max_buckets, B1, B2) sparse bucket contributions
        compact_to_key: (max_buckets,) mapping to original bucket keys
        num_active: Number of active buckets
        B1: Bucket size in wire direction
        B2: Bucket size in time direction
        num_wires: Total number of wires
        num_time_steps: Total number of time steps
        max_buckets: Maximum number of buckets

    Returns:
        (num_wires, num_time_steps) dense array
    """
    # Use ceiling division to match build_bucket_mapping
    NUM_BUCKETS_T = (num_time_steps + B2 - 1) // B2

    def add_bucket(i, output):
        key = compact_to_key[i]
        bw = key // NUM_BUCKETS_T
        bt = key % NUM_BUCKETS_T

        w_start = bw * B1
        t_start = bt * B2

        w_indices = w_start + jnp.arange(B1)[:, None]
        t_indices = t_start + jnp.arange(B2)[None, :]

        valid = i < num_active
        bucket_data = jnp.where(valid, buckets[i], 0.0)

        output = output.at[w_indices, t_indices].add(bucket_data)
        return output

    output = jnp.zeros((num_wires, num_time_steps), dtype=jnp.float32)
    output = jax.lax.fori_loop(0, max_buckets, add_bucket, output)

    return output
