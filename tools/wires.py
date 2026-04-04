"""
Wire and pixel signal calculations for LArTPC simulation.

This module contains all readout-related calculations for processing charge deposits
into wire or pixel signals. It supports two parallel simulation paths:

Sections:
1. SHARED: Wire geometry calculations
2. RESPONSE PATH: Kernel-based signal generation (wire)
3. HIT PATH: Direct diffusion calculation (wire)
4. 3D PIXEL BUCKETING: Tile-based signal accumulation (pixel)
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
    closest_wire_idx : jnp.ndarray
        Array of shape (n_hits,) containing the wire index of the closest wire.
    closest_wire_dist : jnp.ndarray
        Array of shape (n_hits,) containing the distances to the closest wires in cm.
    """
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)

    P_y_cm = positions_yz_centered_cm[:, 0]
    P_z_cm = positions_yz_centered_cm[:, 1]

    r_prime = P_y_cm * sin_theta + P_z_cm * cos_theta

    idx_offset_rel = jnp.round(r_prime / wire_spacing_cm)
    closest_wire_dist = r_prime - idx_offset_rel * wire_spacing_cm
    closest_wire_idx = idx_offset_rel.astype(jnp.int32) + index_offset

    return closest_wire_idx, closest_wire_dist


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


@partial(jax.jit, static_argnums=(2,))
def diffusion_cdf_1d(mu, sigma, size):
    """Compute charge fractions per bin via CDF integration.

    Integrates a Gaussian between bin edges to get the exact fraction
    of charge landing in each bin. Normalized to sum to 1.0.

    Works for any axis (wire, pixel, time) — just provide sigma and mu
    in bin units (wire pitches, pixel pitches, or time steps).

    Parameters
    ----------
    mu : float
        Sub-bin offset of the deposit from the central bin center,
        in bin units. Range [-0.5, 0.5).
    sigma : float
        Diffusion sigma in bin units (e.g., wire pitches or time steps).
    size : int (static)
        Number of bins (2*K+1).

    Returns
    -------
    fractions : jnp.ndarray, shape (size,)
        Charge fraction per bin, sums to 1.0.
    """
    from jax.scipy.stats import norm
    bins = jnp.arange(size + 1, dtype=jnp.float32) - size / 2
    sigma_safe = jnp.maximum(sigma, 1e-6)
    cdf = norm.cdf(bins, loc=mu, scale=sigma_safe)
    pdf = jnp.diff(cdf)
    return pdf / jnp.maximum(pdf.sum(), 1e-30)


def prepare_deposit_with_diffusion(
    charge, drift_time_us, tick_us, drift_distance_cm, wire_idx, closest_wire_distance,
    attenuation_factor, theta_xz_rad, theta_y_rad, angular_scaling_factor, valid_hit,
    K_wire, K_time, wire_spacing_cm, time_step_size_us,
    longitudinal_diffusion_cm2_us, transverse_diffusion_cm2_us, drift_velocity_cm_us,
    num_wires, num_time_steps
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
        Pure drift time in μs (for diffusion sigma calculation).
    tick_us : float
        Readout tick time in μs (drift + t0 + pre_window, for time binning).
    drift_distance_cm : float
        Drift distance of the hit in cm.
    wire_idx : int
        Wire index of the closest wire.
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
    num_wires : int
        Number of wires in the plane.
    num_time_steps : int
        Number of time steps in the simulation.

    Returns
    -------
    tuple
        Tuple containing indices and values for the hit:
        (wire_indices, time_indices_out, signal_values_out)
    """
    # Only process if valid hit
    charge = jnp.where(valid_hit, charge, 0.0)

    # 1. Calculate time bins and offsets (using K as half-width)
    # Use tick_us (readout time) for bin placement, drift_time_us for diffusion sigma
    central_time_bin = jnp.floor(tick_us / time_step_size_us).astype(jnp.int32)
    time_bin_offsets = jnp.arange(-K_time, K_time + 1)  # 2K+1 values
    time_bins = central_time_bin + time_bin_offsets
    bin_center_times = (time_bins + 0.5) * time_step_size_us
    time_differences_us = tick_us - bin_center_times  # Shape: (2*K_time+1,)

    # 2. Calculate wire indices and distances (using K as half-width)
    relative_indices = jnp.arange(-K_wire, K_wire + 1)  # 2K+1 values
    wire_indices = wire_idx + relative_indices  # Shape: (2*K_wire+1,)
    wire_distances_cm = closest_wire_distance - relative_indices * wire_spacing_cm  # Shape: (2*K_wire+1,)

    # 3. Apply charge scaling and attenuation
    # charge_scaled = charge * angular_scaling_factor
    attenuated_charge = charge * attenuation_factor

    # 4. Calculate diffusion response via CDF integration
    # Convert sigma to bin units (wire pitches / time steps)
    sigma_wire_cm = jnp.sqrt(2.0 * transverse_diffusion_cm2_us * drift_time_us)
    sigma_wire_pitches = sigma_wire_cm / wire_spacing_cm

    sigma_time_us = jnp.sqrt(
        2.0 * (longitudinal_diffusion_cm2_us / (drift_velocity_cm_us ** 2)) * drift_time_us)
    sigma_time_steps = sigma_time_us / time_step_size_us

    # Sub-bin offsets in bin units
    wire_mu = closest_wire_distance / wire_spacing_cm  # offset from nearest wire center
    time_mu = (tick_us / time_step_size_us) - central_time_bin - 0.5  # offset from bin center

    # CDF-integrated fractions per bin (charge-conserving, sum = 1.0)
    wire_fractions = diffusion_cdf_1d(wire_mu, sigma_wire_pitches, 2 * K_wire + 1)
    time_fractions = diffusion_cdf_1d(time_mu, sigma_time_steps, 2 * K_time + 1)

    # 2D separable product
    diffusion_response_normalized = wire_fractions[:, None] * time_fractions[None, :]

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
    wire_indices = jnp.where(wire_valid, wire_indices, 0)
    time_indices_out = time_bins
    signal_values_2d = jnp.where(valid_mask_2d, diffusion_response, 0.0)

    # Calculate total number of elements
    num_wire_bins = 2 * K_wire + 1
    num_time_bins = 2 * K_time + 1

    # Flatten the arrays for output
    wire_indices_flat = jnp.repeat(wire_indices, num_time_bins)
    time_indices_flat = jnp.tile(time_indices_out, num_wire_bins)
    signal_values_out = signal_values_2d.reshape(-1)

    return wire_indices_flat, time_indices_flat, signal_values_out



def prepare_pixel_deposit_with_diffusion(
    charge, drift_time_us, tick_us, pixel_y_idx, pixel_z_idx,
    pixel_y_offset, pixel_z_offset,
    attenuation_factor, valid_hit,
    K_py, K_pz, K_time, pixel_pitch_cm, time_step_size_us,
    transverse_diffusion_cm2_us, longitudinal_diffusion_cm2_us,
    drift_velocity_cm_us, num_py, num_pz, num_time_steps
):
    """
    Prepare pixel deposit data WITH 3D diffusion kernel (hit/track_hits path).

    Computes CDF-integrated Gaussian fractions across a (2K_py+1) × (2K_pz+1) × (2K_time+1)
    neighborhood. Charge is exactly conserved (fractions sum to 1.0).

    Parameters
    ----------
    charge : float
    drift_time_us : float
        Pure drift time for diffusion sigma calculation.
    tick_us : float
        Readout tick time for time bin placement.
    pixel_y_idx, pixel_z_idx : int
        Center pixel indices.
    pixel_y_offset, pixel_z_offset : float
        Sub-pixel offsets in [-0.5, 0.5) pixel pitch units.
    attenuation_factor : float
    valid_hit : bool
    K_py, K_pz, K_time : int (static)
        Half-widths in each dimension.
    pixel_pitch_cm : float
    time_step_size_us : float
    transverse_diffusion_cm2_us : float
    longitudinal_diffusion_cm2_us : float
    drift_velocity_cm_us : float
    num_py, num_pz, num_time_steps : int (static)

    Returns
    -------
    spatial_keys : jnp.ndarray, shape (K_total,), int32
        Flattened pixel IDs: py * num_pz + pz.
    time_indices : jnp.ndarray, shape (K_total,), int32
    signal_values : jnp.ndarray, shape (K_total,), float32
    """
    charge = jnp.where(valid_hit, charge, 0.0)
    attenuated_charge = charge * attenuation_factor

    # Time bins
    central_time_bin = jnp.floor(tick_us / time_step_size_us).astype(jnp.int32)
    time_offsets = jnp.arange(-K_time, K_time + 1)
    time_bins = central_time_bin + time_offsets

    # Pixel indices
    py_offsets = jnp.arange(-K_py, K_py + 1)
    pz_offsets = jnp.arange(-K_pz, K_pz + 1)
    py_indices = pixel_y_idx + py_offsets
    pz_indices = pixel_z_idx + pz_offsets

    # Diffusion sigmas in bin units
    sigma_T_cm = jnp.sqrt(2.0 * transverse_diffusion_cm2_us * drift_time_us)
    sigma_T_pitches = sigma_T_cm / pixel_pitch_cm

    sigma_L_us = jnp.sqrt(
        2.0 * (longitudinal_diffusion_cm2_us / (drift_velocity_cm_us ** 2)) * drift_time_us)
    sigma_L_steps = sigma_L_us / time_step_size_us

    # Sub-bin offsets
    time_mu = (tick_us / time_step_size_us) - central_time_bin - 0.5

    # CDF-integrated fractions (charge-conserving)
    py_fractions = diffusion_cdf_1d(pixel_y_offset, sigma_T_pitches, 2 * K_py + 1)
    pz_fractions = diffusion_cdf_1d(pixel_z_offset, sigma_T_pitches, 2 * K_pz + 1)
    time_fractions = diffusion_cdf_1d(time_mu, sigma_L_steps, 2 * K_time + 1)

    # 3D separable product
    diffusion_3d = (py_fractions[:, None, None]
                    * pz_fractions[None, :, None]
                    * time_fractions[None, None, :])  # (2K_py+1, 2K_pz+1, 2K_time+1)

    diffusion_3d = diffusion_3d * attenuated_charge

    # Validity masks
    py_valid = (py_indices >= 0) & (py_indices < num_py)
    pz_valid = (pz_indices >= 0) & (pz_indices < num_pz)
    time_valid = (time_bins >= 0) & (time_bins < num_time_steps)

    valid_3d = (py_valid[:, None, None]
                & pz_valid[None, :, None]
                & time_valid[None, None, :]
                & valid_hit)

    diffusion_3d = jnp.where(valid_3d, diffusion_3d, 0.0)
    py_indices = jnp.where(py_valid, py_indices, 0)
    pz_indices = jnp.where(pz_valid, pz_indices, 0)

    # Flatten to (K_total,) arrays
    num_py_bins = 2 * K_py + 1
    num_pz_bins = 2 * K_pz + 1
    num_t_bins = 2 * K_time + 1
    K_total = num_py_bins * num_pz_bins * num_t_bins

    # Spatial key: py * num_pz + pz (for track_hits merge)
    # Broadcast to 3D then flatten
    py_3d = jnp.broadcast_to(py_indices[:, None, None], (num_py_bins, num_pz_bins, num_t_bins))
    pz_3d = jnp.broadcast_to(pz_indices[None, :, None], (num_py_bins, num_pz_bins, num_t_bins))
    t_3d = jnp.broadcast_to(time_bins[None, None, :], (num_py_bins, num_pz_bins, num_t_bins))

    spatial_keys = (py_3d * num_pz + pz_3d).reshape(K_total).astype(jnp.int32)
    time_indices_out = t_3d.reshape(K_total).astype(jnp.int32)
    signal_values_out = diffusion_3d.reshape(K_total)

    return spatial_keys, time_indices_out, signal_values_out


# ============================================================================
# RESPONSE PATH: Kernel-based Signal Generation
# ============================================================================

@partial(jax.jit, static_argnames=['num_wires'])
def prepare_deposit_for_response(
    charge, tick_us, wire_idx, closest_wire_distance,
    attenuation_factor, valid_hit,
    wire_spacing_cm, time_step_size_us,
    num_wires
):
    """
    Process a single hit without diffusion (handled by response kernels).

    Parameters
    ----------
    charge : float
        Charge deposited by the hit.
    tick_us : float
        Readout tick time in μs (drift + t0 + pre_window).
    wire_idx : int
        Wire index of the closest wire.
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
    time_index = jnp.floor(tick_us / time_step_size_us).astype(jnp.int32)
    time_offset = (tick_us / time_step_size_us) - time_index

    # Calculate wire offset (fractional part of wire position)
    wire_offset = closest_wire_distance / wire_spacing_cm

    # Apply charge scaling and attenuation
    intensity = charge * attenuation_factor

    wire_idx_out = jnp.where(
        (wire_idx >= 0) & (wire_idx < num_wires),
        wire_idx,
        -1  # Invalid index
    )

    intensity = jnp.where(
        valid_hit & (wire_idx_out >= 0),
        intensity,
        0.0
    )

    return wire_idx_out, wire_offset, time_index, time_offset, intensity


@partial(jax.jit, static_argnames=('num_wires', 'num_time_steps', 'kernel_num_wires', 'kernel_height',
                                    'wire_zero_bin', 'time_zero_bin'))
def accumulate_response_signals(wire_indices, time_indices, intensities, contributions,
                                num_wires, num_time_steps, kernel_num_wires, kernel_height,
                                wire_zero_bin, time_zero_bin):
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
    wire_zero_bin : int
        Index where wire=0 is in the kernel (static). For symmetric kernels,
        this is kernel_num_wires // 2.
    time_zero_bin : int
        Index where t=0 is in the kernel (static). For symmetric kernels,
        this is kernel_height // 2. For asymmetric kernels (e.g., more
        negative time extent), this reflects where t=0 actually is.

    Returns
    -------
    signals : jnp.ndarray
        (num_wires, num_time_steps) accumulated wire signals.
    """
    # Initialize output array
    signals = jnp.zeros((num_wires, num_time_steps))

    # Create kernel index offsets (reused for all segments)
    kernel_wire_offsets = jnp.arange(kernel_num_wires)
    kernel_time_offsets = jnp.arange(kernel_height)

    # Compute absolute wire positions for all segments — shape: (N, kernel_num_wires)
    wire_positions = wire_indices[:, None] - wire_zero_bin + kernel_wire_offsets[None, :]

    # Compute absolute time positions — shape: (N, kernel_height)
    time_positions = time_indices[:, None] - time_zero_bin + kernel_time_offsets[None, :]

    # Scale all contributions by their intensities
    # Shape: (N, kernel_num_wires, kernel_height)
    scaled_contributions = contributions * intensities[:, None, None]

    # Zero out contributions where indices fall outside the output array.
    # JAX treats negative indices as Python-style (e.g. -1 → last element),
    # so we must mask them here rather than relying on mode='drop'.
    wire_valid = (wire_positions >= 0) & (wire_positions < num_wires)       # (N, kernel_num_wires)
    time_valid = (time_positions >= 0) & (time_positions < num_time_steps)  # (N, kernel_height)
    scaled_contributions = scaled_contributions * (wire_valid[:, :, None] & time_valid[:, None, :])

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

@partial(jax.jit, static_argnames=('B1', 'B2', 'num_wires', 'num_time_steps', 'max_buckets',
                                    'wire_zero_bin', 'time_zero_bin'))
def build_bucket_mapping(wire_indices, time_indices,
                         B1, B2, num_wires, num_time_steps, max_buckets,
                         wire_zero_bin, time_zero_bin):
    """
    Build point_to_compact mapping for sparse bucketing.

    Phase 1 of the sparse bucketing algorithm. For each segment at (wire, time),
    its kernel can touch up to 4 buckets. This function computes which 4 buckets
    each segment touches and creates a compact mapping to active bucket indices.

    Parameters
    ----------
    wire_indices : jnp.ndarray, shape (N,)
        Center wire index for each segment.
    time_indices : jnp.ndarray, shape (N,)
        Start time index for each segment.
    B1 : int
        Bucket size in wire direction (= 2 * kernel_num_wires).
    B2 : int
        Bucket size in time direction (= 2 * kernel_height).
    num_wires : int
        Total wires (X dimension).
    num_time_steps : int
        Total time steps (Y dimension).
    max_buckets : int
        Maximum active buckets to allocate.
    wire_zero_bin : int
        Where wire=0 is in output wires.
    time_zero_bin : int
        Where t=0 is in output simulation time bins.

    Returns
    -------
    point_to_compact : jnp.ndarray, shape (N, 4)
        Maps (segment, quadrant) to compact index.
    num_active : int
        Number of unique active buckets.
    compact_to_key : jnp.ndarray, shape (max_buckets,)
        Maps compact index to bucket key.
    """
    N = wire_indices.shape[0]
    # Use ceiling division to ensure all buckets have unique keys
    # Floor division causes collisions when dimensions don't divide evenly
    NUM_BUCKETS_T = (num_time_steps + B2 - 1) // B2

    # Use explicit center bins (passed as parameters)
    # This makes the calculation consistent with scatter_contributions_to_buckets
    # FIX: Previously home_bt did not use time offset, causing wrong bucket selection
    home_bw = (wire_indices - wire_zero_bin) // B1  # (N,)
    home_bt = (time_indices - time_zero_bin) // B2  # (N,)  # FIXED: Now uses time offset

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


@partial(jax.jit, static_argnames=('max_buckets', 'kernel_num_wires', 'kernel_height', 'B1', 'B2',
                                    'wire_zero_bin', 'time_zero_bin',
                                    'num_wires', 'num_time_steps'))
def scatter_contributions_to_buckets(
    wire_indices, time_indices, intensities, contributions,
    point_to_compact, max_buckets, kernel_num_wires, kernel_height, B1, B2,
    wire_zero_bin, time_zero_bin,
    num_wires=None, num_time_steps=None
):
    """
    Scatter (N, kernel_num_wires, kernel_height) contributions to sparse buckets.

    Non-batched reference implementation. Production uses
    scatter_contributions_to_buckets_batched which processes in chunks
    via jax.lax.scan to reduce peak memory.

    Phase 2 of the sparse bucketing algorithm. Scatters pre-computed kernel
    contributions to sparse buckets using the mapping from Phase 1.

    Parameters
    ----------
    wire_indices : jnp.ndarray, shape (N,)
        Center wire index for each segment.
    time_indices : jnp.ndarray, shape (N,)
        Start time index for each segment.
    intensities : jnp.ndarray, shape (N,)
        Intensity scaling factor for each segment.
    contributions : jnp.ndarray, shape (N, kernel_num_wires, kernel_height)
        Kernel response for each segment.
    point_to_compact : jnp.ndarray, shape (N, 4)
        Mapping from (segment, quadrant) to compact index.
    max_buckets : int
        Maximum number of buckets.
    kernel_num_wires : int
        Number of wires in kernel.
    kernel_height : int
        Number of time bins in kernel.
    B1 : int
        Bucket size in wire direction.
    B2 : int
        Bucket size in time direction.
    wire_zero_bin : int
        Where wire=0 is in output wires.
    time_zero_bin : int
        Where t=0 is in output simulation time bins.
    num_wires : int, optional
        Total wires for bounds checking.
    num_time_steps : int, optional
        Total time steps for bounds checking.

    Returns
    -------
    jnp.ndarray, shape (max_buckets, B1, B2)
        Sparse bucket contributions.
    """
    N = wire_indices.shape[0]

    # Scale contributions by intensity
    scaled = contributions * intensities[:, None, None]

    # Create index grids
    n_idx = jnp.arange(N)[:, None, None]
    kw_idx = jnp.arange(kernel_num_wires)[None, :, None]
    kt_idx = jnp.arange(kernel_height)[None, None, :]

    # Global coordinates (centered on wire_indices and time_indices)
    gw = (wire_indices[:, None, None] - wire_zero_bin + kw_idx).astype(jnp.int32)
    gt = (time_indices[:, None, None] - time_zero_bin + kt_idx).astype(jnp.int32)

    # Zero contributions outside detector bounds
    if num_wires is not None and num_time_steps is not None:
        valid = (gw >= 0) & (gw < num_wires) & (gt >= 0) & (gt < num_time_steps)
        scaled = scaled * valid

    # Compute which quadrant each kernel element falls into
    home_bw = (wire_indices[:, None, None] - wire_zero_bin) // B1
    home_bt = (time_indices[:, None, None] - time_zero_bin) // B2

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


@partial(jax.jit, static_argnames=('max_buckets', 'kernel_num_wires', 'kernel_height', 'B1', 'B2',
                                    'wire_zero_bin', 'time_zero_bin', 'batch_size',
                                    'num_wires', 'num_time_steps'))
def scatter_contributions_to_buckets_batched(
    wire_indices, time_indices, intensities, contributions,
    point_to_compact, max_buckets, kernel_num_wires, kernel_height, B1, B2,
    wire_zero_bin, time_zero_bin,
    batch_size=1000, num_wires=None, num_time_steps=None
):
    """
    Batched scatter to reduce memory from O(N × K1 × K2) to O(batch_size × K1 × K2).

    Uses jax.lax.scan to process segments in batches, avoiding materialization
    of 14 arrays of shape (N, K1, K2) simultaneously. Use this for large N
    where the original scatter would OOM.

    Parameters
    ----------
    wire_indices : jnp.ndarray, shape (N,)
        Center wire index for each segment.
    time_indices : jnp.ndarray, shape (N,)
        Start time index for each segment.
    intensities : jnp.ndarray, shape (N,)
        Intensity scaling factor for each segment.
    contributions : jnp.ndarray, shape (N, kernel_num_wires, kernel_height)
        Kernel response for each segment.
    point_to_compact : jnp.ndarray, shape (N, 4)
        Mapping from (segment, quadrant) to compact index.
    max_buckets : int
        Maximum number of buckets.
    kernel_num_wires : int
        Number of wires in kernel.
    kernel_height : int
        Number of time bins in kernel.
    B1 : int
        Bucket size in wire direction.
    B2 : int
        Bucket size in time direction.
    wire_zero_bin : int
        Where wire=0 is in output wires.
    time_zero_bin : int
        Where t=0 is in output simulation time bins.
    batch_size : int
        Number of segments to process per batch.
    num_wires : int, optional
        Total wires for bounds checking.
    num_time_steps : int, optional
        Total time steps for bounds checking.

    Returns
    -------
    jnp.ndarray, shape (max_buckets, B1, B2)
        Sparse bucket contributions.
    """
    N = wire_indices.shape[0]

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
        # Centered on wire and time indices
        gw = (bw[:, None, None] - wire_zero_bin + kw_idx[None, :, None]).astype(jnp.int32)
        gt = (bt[:, None, None] - time_zero_bin + kt_idx[None, None, :]).astype(jnp.int32)

        # Zero contributions outside detector bounds
        if num_wires is not None and num_time_steps is not None:
            valid_bounds = (gw >= 0) & (gw < num_wires) & (gt >= 0) & (gt < num_time_steps)
            scaled = scaled * valid_bounds

        # Home bucket
        home_bw = (bw[:, None, None] - wire_zero_bin) // B1
        home_bt = (bt[:, None, None] - time_zero_bin) // B2

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
                                    'kernel_height', 'max_buckets',
                                    'wire_zero_bin', 'time_zero_bin', 'batch_size'))
def accumulate_response_signals_sparse_bucketed(
    wire_indices, time_indices, intensities, contributions,
    num_wires, num_time_steps, kernel_num_wires, kernel_height,
    max_buckets, wire_zero_bin, time_zero_bin, batch_size=1000
):
    """
    Sparse bucketed accumulation using the two-phase approach.

    Memory-efficient version that only allocates active buckets. Use this
    for large detectors where the dense output array would exceed GPU memory.

    Phase 1: Build bucket mapping (O(N) with sort)
    Phase 2: Batched scatter to sparse buckets (O(N * K1 * K2))

    Parameters
    ----------
    wire_indices : jnp.ndarray, shape (N,)
        Center wire index for each segment.
    time_indices : jnp.ndarray, shape (N,)
        Start time index for each segment.
    intensities : jnp.ndarray, shape (N,)
        Intensity scaling factor for each segment.
    contributions : jnp.ndarray, shape (N, kernel_num_wires, kernel_height)
        Kernel response for each segment.
    num_wires : int
        Total number of wires in output.
    num_time_steps : int
        Total number of time steps in output.
    kernel_num_wires : int
        Number of wires in kernel.
    kernel_height : int
        Number of time bins in kernel.
    max_buckets : int
        Maximum number of active buckets to allocate.
    wire_zero_bin : int
        Where wire=0 is in output wires.
    time_zero_bin : int
        Where t=0 is in output simulation time bins.
    batch_size : int
        Batch size for scatter. Recommended: 500-2000.

    Returns
    -------
    buckets : jnp.ndarray, shape (max_buckets, B1, B2)
        Sparse bucket contributions.
    num_active : int
        Number of active buckets.
    compact_to_key : jnp.ndarray, shape (max_buckets,)
        Mapping to original bucket keys.
    """
    # Bucket sizes: 2x kernel size
    B1 = 2 * kernel_num_wires
    B2 = 2 * kernel_height

    # Phase 1: Build mapping
    point_to_compact, num_active, compact_to_key = build_bucket_mapping(
        wire_indices, time_indices, B1, B2, num_wires, num_time_steps, max_buckets,
        wire_zero_bin, time_zero_bin
    )

    # Phase 2: Batched scatter to sparse buckets
    buckets = scatter_contributions_to_buckets_batched(
        wire_indices, time_indices, intensities, contributions,
        point_to_compact, max_buckets, kernel_num_wires, kernel_height, B1, B2,
        wire_zero_bin, time_zero_bin,
        batch_size=batch_size, num_wires=num_wires, num_time_steps=num_time_steps
    )

    return buckets, num_active, compact_to_key


@partial(jax.jit, static_argnames=('B1', 'B2', 'num_wires', 'num_time_steps', 'max_buckets'))
def sparse_buckets_to_dense(buckets, compact_to_key, num_active,
                            B1, B2, num_wires, num_time_steps, max_buckets):
    """
    Convert sparse buckets back to dense (num_wires, num_time_steps) array.

    Parameters
    ----------
    buckets : jnp.ndarray, shape (max_buckets, B1, B2)
        Sparse bucket contributions.
    compact_to_key : jnp.ndarray, shape (max_buckets,)
        Mapping to original bucket keys.
    num_active : int
        Number of active buckets.
    B1 : int
        Bucket size in wire direction.
    B2 : int
        Bucket size in time direction.
    num_wires : int
        Total number of wires.
    num_time_steps : int
        Total number of time steps.
    max_buckets : int
        Maximum number of buckets.

    Returns
    -------
    jnp.ndarray, shape (num_wires, num_time_steps)
        Dense output array.
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

        output = output.at[w_indices, t_indices].add(bucket_data, mode='drop')
        return output

    output = jnp.zeros((num_wires, num_time_steps), dtype=jnp.float32)
    output = jax.lax.fori_loop(0, max_buckets, add_bucket, output)

    return output


# ============================================================================
# Pixel Readout Helpers
# ============================================================================

@jax.jit
def digitize_pixel_positions(positions_yz_cm, pixel_pitch_cm, pixel_origins_cm):
    """Convert physical (y, z) positions to pixel indices and sub-pixel offsets.

    Parameters
    ----------
    positions_yz_cm : jnp.ndarray, shape (N, 2)
        Physical (y, z) positions in cm.
    pixel_pitch_cm : float
        Pixel pitch size in cm.
    pixel_origins_cm : jnp.ndarray, shape (2,)
        Pixel grid origin [y_min, z_min] in cm.

    Returns
    -------
    pixel_y_idx : jnp.ndarray, shape (N,), int32
        Center pixel y indices.
    pixel_z_idx : jnp.ndarray, shape (N,), int32
        Center pixel z indices.
    pixel_y_offset : jnp.ndarray, shape (N,), float32
        Sub-pixel y offset in [-0.5, 0.5).
    pixel_z_offset : jnp.ndarray, shape (N,), float32
        Sub-pixel z offset in [-0.5, 0.5).
    """
    d_yz = positions_yz_cm - pixel_origins_cm
    offsets, centers = jnp.modf(d_yz / pixel_pitch_cm)
    offsets = offsets - 0.5
    pixel_indices = centers.astype(jnp.int32)
    return pixel_indices[:, 0], pixel_indices[:, 1], offsets[:, 0], offsets[:, 1]


@partial(jax.jit, static_argnames=['num_py', 'num_pz'])
def prepare_pixel_deposit_for_response(
    charge, tick_us, pixel_y_idx, pixel_z_idx,
    pixel_y_offset, pixel_z_offset,
    attenuation_factor, valid_hit,
    time_step_size_us, num_py, num_pz
):
    """Prepare a single pixel deposit for response kernel application.

    Parameters
    ----------
    charge : float
    tick_us : float
        Readout tick time in μs.
    pixel_y_idx, pixel_z_idx : int
        Center pixel indices.
    pixel_y_offset, pixel_z_offset : float
        Sub-pixel fractional offsets [-0.5, 0.5).
    attenuation_factor : float
    valid_hit : bool
    time_step_size_us : float
    num_py, num_pz : int
        Total pixel grid dimensions.

    Returns
    -------
    tuple
        (pixel_y_idx, pixel_z_idx, pixel_y_offset, pixel_z_offset,
         time_index, time_offset, intensity)
    """
    charge = jnp.where(valid_hit, charge, 0.0)

    time_index = jnp.floor(tick_us / time_step_size_us).astype(jnp.int32)
    time_offset = (tick_us / time_step_size_us) - time_index

    intensity = charge * attenuation_factor

    valid = (valid_hit
             & (pixel_y_idx >= 0) & (pixel_y_idx < num_py)
             & (pixel_z_idx >= 0) & (pixel_z_idx < num_pz)
             & (time_index >= 0))

    intensity = jnp.where(valid, intensity, 0.0)
    pixel_y_idx = jnp.where(valid, pixel_y_idx, 0)
    pixel_z_idx = jnp.where(valid, pixel_z_idx, 0)

    return (pixel_y_idx, pixel_z_idx, pixel_y_offset, pixel_z_offset,
            time_index, time_offset, intensity)


# ============================================================================
# 3D Pixel Bucketing
# ============================================================================

@partial(jax.jit, static_argnames=('B1', 'B2', 'B3', 'num_py', 'num_pz',
                                    'num_time_steps', 'max_buckets',
                                    'py_zero_bin', 'pz_zero_bin', 'time_zero_bin'))
def build_bucket_mapping_3d(pixel_y_indices, pixel_z_indices, time_indices,
                            B1, B2, B3, num_py, num_pz, num_time_steps,
                            max_buckets, py_zero_bin, pz_zero_bin, time_zero_bin):
    """
    Build point_to_compact mapping for 3D pixel bucketing.

    Phase 1 of sparse pixel bucketing. Each deposit's kernel can touch up to
    8 tiles (2x2x2 octants in pixel_y, pixel_z, time). This function computes
    which 8 tiles each deposit touches and creates a compact mapping.

    Parameters
    ----------
    pixel_y_indices : jnp.ndarray, shape (N,)
        Center pixel y index for each segment.
    pixel_z_indices : jnp.ndarray, shape (N,)
        Center pixel z index for each segment.
    time_indices : jnp.ndarray, shape (N,)
        Start time index for each segment.
    B1 : int
        Tile size in pixel y direction (= 2 * kernel_py).
    B2 : int
        Tile size in pixel z direction (= 2 * kernel_pz).
    B3 : int
        Tile size in time direction (= 2 * kernel_time).
    num_py, num_pz : int
        Total pixels in y and z.
    num_time_steps : int
        Total time steps.
    max_buckets : int
        Maximum active tiles to allocate.
    py_zero_bin, pz_zero_bin, time_zero_bin : int
        Kernel center offsets in each dimension.

    Returns
    -------
    point_to_compact : jnp.ndarray, shape (N, 8)
        Maps (segment, octant) to compact tile index.
    num_active : int
        Number of unique active tiles.
    compact_to_key : jnp.ndarray, shape (max_buckets,)
        Maps compact index to tile key.
    """
    N = pixel_y_indices.shape[0]
    NUM_BPZ = (num_pz + B2 - 1) // B2
    NUM_BT = (num_time_steps + B3 - 1) // B3

    home_bpy = (pixel_y_indices - py_zero_bin) // B1
    home_bpz = (pixel_z_indices - pz_zero_bin) // B2
    home_bt = (time_indices - time_zero_bin) // B3

    # 8 octants: all combinations of {0, 1} in 3 dimensions
    offsets_py = jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
    offsets_pz = jnp.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=jnp.int32)
    offsets_t = jnp.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)

    bucket_bpy = home_bpy[:, None] + offsets_py  # (N, 8)
    bucket_bpz = home_bpz[:, None] + offsets_pz
    bucket_bt = home_bt[:, None] + offsets_t

    bucket_keys = bucket_bpy * NUM_BPZ * NUM_BT + bucket_bpz * NUM_BT + bucket_bt

    flat_keys = bucket_keys.ravel()  # (8N,)

    sorted_idx = jnp.argsort(flat_keys)
    sorted_keys = flat_keys[sorted_idx]

    is_new = jnp.concatenate([
        jnp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1]
    ])

    compact_idx_sorted = jnp.cumsum(is_new.astype(jnp.int32)) - 1

    compact_idx_flat = jnp.zeros(8 * N, dtype=jnp.int32)
    compact_idx_flat = compact_idx_flat.at[sorted_idx].set(compact_idx_sorted)

    point_to_compact = compact_idx_flat.reshape(N, 8)

    num_active = jnp.sum(is_new.astype(jnp.int32))

    compact_to_key = jnp.zeros(max_buckets, dtype=jnp.int32)
    compact_to_key = compact_to_key.at[compact_idx_sorted].set(
        sorted_keys, mode='drop')

    return point_to_compact, num_active, compact_to_key


@partial(jax.jit, static_argnames=('max_buckets', 'kernel_py', 'kernel_pz',
                                    'kernel_time', 'B1', 'B2', 'B3',
                                    'py_zero_bin', 'pz_zero_bin', 'time_zero_bin',
                                    'batch_size', 'num_py', 'num_pz',
                                    'num_time_steps'))
def scatter_contributions_to_pixel_buckets_batched(
    pixel_y_indices, pixel_z_indices, time_indices,
    intensities, contributions,
    point_to_compact, max_buckets,
    kernel_py, kernel_pz, kernel_time,
    B1, B2, B3,
    py_zero_bin, pz_zero_bin, time_zero_bin,
    batch_size=1000, num_py=None, num_pz=None, num_time_steps=None
):
    """
    Batched scatter of 3D pixel kernel contributions to sparse tiles.

    Uses jax.lax.scan to process segments in batches, avoiding
    materialization of large intermediate arrays.

    Parameters
    ----------
    pixel_y_indices : jnp.ndarray, shape (N,)
        Center pixel y index for each segment.
    pixel_z_indices : jnp.ndarray, shape (N,)
        Center pixel z index for each segment.
    time_indices : jnp.ndarray, shape (N,)
        Start time index for each segment.
    intensities : jnp.ndarray, shape (N,)
        Intensity scaling factor for each segment.
    contributions : jnp.ndarray, shape (N, kernel_py, kernel_pz, kernel_time)
        Kernel response for each segment.
    point_to_compact : jnp.ndarray, shape (N, 8)
        Mapping from (segment, octant) to compact tile index.
    max_buckets : int
        Maximum number of tiles.
    kernel_py, kernel_pz, kernel_time : int
        Kernel dimensions.
    B1, B2, B3 : int
        Tile sizes in pixel_y, pixel_z, time directions.
    py_zero_bin, pz_zero_bin, time_zero_bin : int
        Kernel center offsets.
    batch_size : int
        Segments per batch in scan.
    num_py, num_pz, num_time_steps : int, optional
        Detector bounds for validity checking.

    Returns
    -------
    jnp.ndarray, shape (max_buckets, B1, B2, B3)
        Sparse tile contributions.
    """
    N = pixel_y_indices.shape[0]

    n_batches = (N + batch_size - 1) // batch_size
    padded_N = n_batches * batch_size
    pad = padded_N - N

    py_pad = jnp.pad(pixel_y_indices, (0, pad), constant_values=0)
    pz_pad = jnp.pad(pixel_z_indices, (0, pad), constant_values=0)
    t_pad = jnp.pad(time_indices, (0, pad), constant_values=0)
    int_pad = jnp.pad(intensities, (0, pad), constant_values=0.0)
    contrib_pad = jnp.pad(contributions,
                          ((0, pad), (0, 0), (0, 0), (0, 0)),
                          constant_values=0.0)
    p2c_pad = jnp.pad(point_to_compact, ((0, pad), (0, 0)), constant_values=0)
    valid_mask = jnp.arange(padded_N) < N

    py_batched = py_pad.reshape(n_batches, batch_size)
    pz_batched = pz_pad.reshape(n_batches, batch_size)
    t_batched = t_pad.reshape(n_batches, batch_size)
    int_batched = int_pad.reshape(n_batches, batch_size)
    contrib_batched = contrib_pad.reshape(
        n_batches, batch_size, kernel_py, kernel_pz, kernel_time)
    p2c_batched = p2c_pad.reshape(n_batches, batch_size, 8)
    valid_batched = valid_mask.reshape(n_batches, batch_size)

    kpy_idx = jnp.arange(kernel_py)
    kpz_idx = jnp.arange(kernel_pz)
    kt_idx = jnp.arange(kernel_time)

    def process_batch(output, batch_data):
        bpy, bpz, bt, bi, bc, bp, bv = batch_data

        # Scale contributions — shape (batch_size, K_py, K_pz, K_t)
        scaled = bc * bi[:, None, None, None]

        # Global coordinates — shape (batch_size, K_py, K_pz, K_t)
        gpy = (bpy[:, None, None, None] - py_zero_bin
               + kpy_idx[None, :, None, None]).astype(jnp.int32)
        gpz = (bpz[:, None, None, None] - pz_zero_bin
               + kpz_idx[None, None, :, None]).astype(jnp.int32)
        gt = (bt[:, None, None, None] - time_zero_bin
              + kt_idx[None, None, None, :]).astype(jnp.int32)

        # Zero contributions outside detector bounds
        if num_py is not None and num_pz is not None and num_time_steps is not None:
            valid_bounds = ((gpy >= 0) & (gpy < num_py)
                            & (gpz >= 0) & (gpz < num_pz)
                            & (gt >= 0) & (gt < num_time_steps))
            scaled = scaled * valid_bounds

        # Home tile
        home_bpy = (bpy[:, None, None, None] - py_zero_bin) // B1
        home_bpz = (bpz[:, None, None, None] - pz_zero_bin) // B2
        home_bt = (bt[:, None, None, None] - time_zero_bin) // B3

        # Cell tile
        cell_bpy = gpy // B1
        cell_bpz = gpz // B2
        cell_bt = gt // B3

        # Octant index
        which_py = jnp.clip(cell_bpy - home_bpy, 0, 1).astype(jnp.int32)
        which_pz = jnp.clip(cell_bpz - home_bpz, 0, 1).astype(jnp.int32)
        which_t = jnp.clip(cell_bt - home_bt, 0, 1).astype(jnp.int32)
        octant = which_py * 4 + which_pz * 2 + which_t

        # Gather tile index
        batch_idx = jnp.arange(batch_size)[:, None, None, None]
        batch_idx_expanded = jnp.broadcast_to(
            batch_idx, (batch_size, kernel_py, kernel_pz, kernel_time))
        bucket_idx = bp[batch_idx_expanded, octant]

        # Local coordinates within tile
        lpy = (gpy % B1).astype(jnp.int32)
        lpz = (gpz % B2).astype(jnp.int32)
        lt = (gt % B3).astype(jnp.int32)

        # Mask invalid segments
        valid_4d = jnp.broadcast_to(
            bv[:, None, None, None],
            (batch_size, kernel_py, kernel_pz, kernel_time))
        masked_scaled = jnp.where(valid_4d, scaled, 0.0)

        output = output.at[bucket_idx, lpy, lpz, lt].add(
            masked_scaled, mode='drop')

        return output, None

    output = jnp.zeros((max_buckets, B1, B2, B3), dtype=jnp.float32)
    output, _ = jax.lax.scan(
        process_batch,
        output,
        (py_batched, pz_batched, t_batched, int_batched,
         contrib_batched, p2c_batched, valid_batched)
    )

    return output


@partial(jax.jit, static_argnames=('B1', 'B2', 'B3', 'num_py', 'num_pz',
                                    'num_time_steps', 'max_buckets'))
def sparse_pixel_buckets_to_dense(buckets, compact_to_key, num_active,
                                  B1, B2, B3, num_py, num_pz,
                                  num_time_steps, max_buckets):
    """
    Convert sparse 3D pixel tiles back to dense (num_py, num_pz, num_time) array.

    Parameters
    ----------
    buckets : jnp.ndarray, shape (max_buckets, B1, B2, B3)
        Sparse tile contributions.
    compact_to_key : jnp.ndarray, shape (max_buckets,)
        Mapping to original tile keys.
    num_active : int
        Number of active tiles.
    B1, B2, B3 : int
        Tile sizes in pixel_y, pixel_z, time.
    num_py, num_pz : int
        Total pixels in y and z.
    num_time_steps : int
        Total time steps.
    max_buckets : int
        Maximum number of tiles.

    Returns
    -------
    jnp.ndarray, shape (num_py, num_pz, num_time_steps)
        Dense output array.
    """
    NUM_BPZ = (num_pz + B2 - 1) // B2
    NUM_BT = (num_time_steps + B3 - 1) // B3

    def add_tile(i, output):
        key = compact_to_key[i]
        bpy = key // (NUM_BPZ * NUM_BT)
        remainder = key % (NUM_BPZ * NUM_BT)
        bpz = remainder // NUM_BT
        bt = remainder % NUM_BT

        py_start = bpy * B1
        pz_start = bpz * B2
        t_start = bt * B3

        py_indices = py_start + jnp.arange(B1)[:, None, None]
        pz_indices = pz_start + jnp.arange(B2)[None, :, None]
        t_indices = t_start + jnp.arange(B3)[None, None, :]

        valid = i < num_active
        tile_data = jnp.where(valid, buckets[i], 0.0)

        output = output.at[py_indices, pz_indices, t_indices].add(
            tile_data, mode='drop')
        return output

    output = jnp.zeros((num_py, num_pz, num_time_steps), dtype=jnp.float32)
    output = jax.lax.fori_loop(0, max_buckets, add_tile, output)

    return output
