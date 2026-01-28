"""
Response Kernel Module

This module handles loading, creating, and applying wire response kernels with diffusion.
Uses pre-computed diffusion kernel arrays for efficient runtime interpolation.

Contents:
1. Kernel Loading - Load NPZ kernel files
2. Gaussian Convolution - Create diffusion kernels at different s levels
3. Runtime Interpolation - JIT-compiled batch interpolation
4. High-Level API - load_response_kernels(), apply_diffusion_response()
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, vmap
from functools import partial
import numpy as np


# ============================================================================
# KERNEL LOADING
# ============================================================================

def load_kernel(filename):
    """
    Load kernel from npz file (stored in actual values, not log scale).

    Parameters
    ----------
    filename : str
        Path to kernel npz file

    Returns
    -------
    kernel : np.ndarray
        Kernel array in actual current values
    kernel_x_coords : np.ndarray
        Wire coordinates
    kernel_y_coords : np.ndarray
        Time coordinates
    plane : str
        Plane name
    dx : float
        Wire spacing
    dy : float
        Time spacing
    wire_zero_bin : int
        Index of wire=0 in kernel (where kernel_x_coords is closest to 0)
    time_zero_bin : int
        Index of t=0 in kernel (where kernel_y_coords is closest to 0)
    """
    data = np.load(filename, allow_pickle=True)

    # Kernel is now stored in actual values, not log scale
    kernel = data['kernel']

    kernel_x_coords = data['kernel_x_coords']
    kernel_y_coords = data['kernel_y_coords']
    plane = str(data['plane'])

    # Get spacing from coordinates
    dx = kernel_x_coords[1] - kernel_x_coords[0] if len(kernel_x_coords) > 1 else 0.1
    dy = kernel_y_coords[1] - kernel_y_coords[0] if len(kernel_y_coords) > 1 else 0.5

    # Read zero-bin indices from file, or compute from coordinates as fallback
    wire_zero_bin = int(data['wire_zero_bin']) if 'wire_zero_bin' in data else int(np.argmin(np.abs(kernel_x_coords)))
    time_zero_bin = int(data['time_zero_bin']) if 'time_zero_bin' in data else int(np.argmin(np.abs(kernel_y_coords)))

    return kernel, kernel_x_coords, kernel_y_coords, plane, dx, dy, wire_zero_bin, time_zero_bin


def calculate_wire_count(kernel_width, wire_spacing=0.1):
    """
    Calculate how many wire positions we can represent given kernel width.

    For wire_spacing = 0.1, we have 10 bins per unit wire spacing.
    So if kernel_width = 127, we have (127-1)/10 = 12.6, so floor(12.6) = 12 wires total.

    Parameters
    ----------
    kernel_width : int
        Width of kernel in bins
    wire_spacing : float
        Wire spacing

    Returns
    -------
    num_wires : int
        Number of representable wires
    """
    # Number of bins per wire unit
    bins_per_wire = int(1.0 / wire_spacing)  # 10

    # Total wire range (symmetric around center)
    wire_range = (kernel_width - 1) / bins_per_wire
    num_wires = int(wire_range)  # Use floor, not +1

    return num_wires


# ============================================================================
# GAUSSIAN CONVOLUTION (for creating diffusion levels)
# ============================================================================

def create_gaussian_kernel(shape, sigma_trans, sigma_long, dx, dy):
    """
    Create a 2D Gaussian kernel with given sigmas and grid spacing.

    Parameters
    ----------
    shape : tuple
        (ny, nx) kernel shape
    sigma_trans : float
        Sigma in transverse direction (wire/spatial, unitless)
    sigma_long : float
        Sigma in longitudinal direction (time/temporal, unitless)
    dx : float
        Wire grid spacing
    dy : float
        Time grid spacing

    Returns
    -------
    gaussian : np.ndarray
        Normalized Gaussian kernel
    """
    ny, nx = shape

    # Create coordinate grids centered at 0
    # Use (n-1)//2 to match convolve2d mode='same' center convention
    x = np.arange(nx) - (nx - 1) // 2
    y = np.arange(ny) - (ny - 1) // 2
    X, Y = np.meshgrid(x * dx, y * dy)

    # Handle small sigma values
    eps = 1e-6
    sigma_trans = max(sigma_trans, eps)
    sigma_long = max(sigma_long, eps)

    # Create Gaussian
    gaussian = np.exp(-(X**2 / (2 * sigma_trans**2) + Y**2 / (2 * sigma_long**2)))

    # Normalize
    gaussian = gaussian / np.sum(gaussian)

    return gaussian


def convolve_with_gaussian(kernel, sigma_trans, sigma_long, dx, dy):
    """
    Convolve kernel with Gaussian using JAX.

    Parameters
    ----------
    kernel : np.ndarray
        Input kernel
    sigma_trans : float
        Sigma in transverse direction (wire/spatial)
    sigma_long : float
        Sigma in longitudinal direction (time/temporal)
    dx : float
        Wire grid spacing
    dy : float
        Time grid spacing

    Returns
    -------
    convolved : np.ndarray
        Convolved kernel
    gaussian : np.ndarray
        Gaussian kernel used
    """
    # Create Gaussian kernel with same shape as input
    gaussian = create_gaussian_kernel(kernel.shape, sigma_trans, sigma_long, dx, dy)

    # Convert to JAX arrays
    kernel_jax = jnp.array(kernel)
    gaussian_jax = jnp.array(gaussian)

    # Perform convolution with 'same' mode to maintain shape
    convolved = jax.scipy.signal.convolve2d(kernel_jax, gaussian_jax, mode='same')

    return np.array(convolved), gaussian


def create_diffusion_kernel_array(planes=['U', 'V', 'Y'], num_s=16, kernel_dir='tools/responses',
                                 wire_spacing=0.1, time_spacing=0.5,
                                 max_sigma_trans_unitless=None, max_sigma_long_unitless=None):
    """
    Create the diffusion kernel array DKernel for each plane.
    DKernel[0] is the original kernel (s=0, no convolution)
    DKernel[1:] are progressively more diffused kernels

    Parameters
    ----------
    planes : list
        List of planes to process
    num_s : int
        Number of s values (diffusion levels)
    kernel_dir : str
        Directory containing kernel files
    wire_spacing : float
        Wire spacing in cm
    time_spacing : float
        Time spacing in us
    max_sigma_trans_unitless : float, optional
        Maximum transverse diffusion sigma in unitless grid coordinates
    max_sigma_long_unitless : float, optional
        Maximum longitudinal diffusion sigma in unitless grid coordinates

    Returns
    -------
    DKernels : dict
        Dictionary mapping plane to (DKernel, linear_s, kernel_shape, x_coords, y_coords)
    """
    # Create linear mapping from 0 to 1
    linear_s = jnp.linspace(0, 1, num_s)

    DKernels = {}

    for plane in planes:
        try:
            # Load original kernel
            filename = f'{kernel_dir}/{plane}_plane_kernel.npz'
            kernel, x_coords, y_coords, loaded_plane, dx, dy, wire_zero_bin, time_zero_bin = load_kernel(filename)

            # Scale kernel by dy (kernel time spacing) for proper discrete integration
            # The kernel represents response density; multiplying by fine dt converts to charge-per-fine-bin
            # Summing bins_per_sim_time fine bins then gives correct charge per simulation bin
            kernel = kernel * dy

            # Initialize DKernel array
            kernel_shape = kernel.shape
            DKernel = jnp.zeros((num_s, *kernel_shape))

            # First entry is original kernel (s=0)
            DKernel = DKernel.at[0].set(kernel)

            # Create progressively diffused kernels
            for i in range(1, num_s):
                s = linear_s[i]

                # Calculate sigmas based on physics: sigma = sigma_max * sqrt(s)
                # Since sigma ~ sqrt(drift_time) and s ~ drift_time, then sigma(s) = sigma_max * sqrt(s)
                # sigma_trans is in wire pitches (matches kernel x-axis units)
                # sigma_long must be in μs (matches kernel y-axis units),
                #   so convert from sim-time-bin units by multiplying by time_spacing
                if max_sigma_trans_unitless is not None and max_sigma_long_unitless is not None:
                    sigma_trans = max_sigma_trans_unitless * np.sqrt(s)                  # wire pitches
                    sigma_long = max_sigma_long_unitless * np.sqrt(s) * time_spacing    # μs
                else:
                    # Fallback to old hardcoded values if not provided
                    sigma_trans = 0.7 * s + 1e-3
                    sigma_long = 1.0 * s + 1e-3

                # Convolve with Gaussian
                convolved, _ = convolve_with_gaussian(kernel, sigma_trans, sigma_long, dx, dy)
                DKernel = DKernel.at[i].set(convolved)

            DKernels[plane] = (DKernel, linear_s, kernel_shape, x_coords, y_coords, dx, dy, wire_zero_bin, time_zero_bin)

        except FileNotFoundError:
            print(f"Warning: Could not find kernel file for {plane} plane")
            continue

    return DKernels


# ============================================================================
# RUNTIME INTERPOLATION (JIT-compiled)
# ============================================================================

@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))  # wire_stride, wire_spacing, kernel_time_spacing, sim_time_spacing, num_wires, num_sim_time_bins are static
def interpolate_diffusion_kernel(DKernel, s_observed, w_offset, t_offset,
                               wire_stride, wire_spacing,
                               kernel_time_spacing, sim_time_spacing,
                               num_wires, num_sim_time_bins):
    """
    Interpolate the diffusion kernel at given s, w, t offsets.

    This is the core runtime function for efficient kernel interpolation.
    Supports high-resolution kernels (e.g., 0.1 μs) with different simulation
    time resolution (e.g., 0.5 μs).

    Parameters
    ----------
    DKernel : jnp.ndarray
        Array of shape (num_s, kernel_height, kernel_width)
    s_observed : float
        Diffusion parameter in [0, 1]
    w_offset : float
        Wire offset in [0, 1.0) - wire offset in units of wire_spacing
    t_offset : float
        Time offset in [0, 1.0) - fractional position within simulation time bin.
    wire_stride : int
        Static wire stride (10 for 0.1 spacing to 1.0 spacing)
    wire_spacing : float
        Static wire spacing (0.1)
    kernel_time_spacing : float
        Static kernel time spacing (e.g., 0.1 μs for high-res)
    sim_time_spacing : float
        Static simulation time spacing (e.g., 0.5 μs)
    num_wires : int
        Static number of wire positions expected
    num_sim_time_bins : int
        Static number of output time bins (in simulation resolution)

    Returns
    -------
    interpolated_values : jnp.ndarray
        Interpolated kernel values with shape (num_wires, num_sim_time_bins)
    """
    num_s, kernel_height, kernel_width = DKernel.shape

    # 1. S interpolation - simple since we have linear points
    s_continuous = s_observed * (num_s - 1)  # Map to [0, num_s-1]
    s_idx = jnp.floor(s_continuous).astype(int)
    s_idx = jnp.clip(s_idx, 0, num_s - 2)  # Ensure we don't go out of bounds
    s_alpha = s_continuous - s_idx

    # 2. Wire interpolation setup
    center_w = kernel_width // 2
    bins_per_wire = int(1.0 / wire_spacing)  # 10

    # Convert w_offset to bin offset
    w_bin_offset = w_offset * bins_per_wire
    w_base_bin = jnp.floor(w_bin_offset).astype(int)
    w_alpha = w_bin_offset - w_base_bin

    # Generate wire bin indices for each output wire position
    # For num_wires=12, we want: -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 (12 total)
    if num_wires % 2 == 0:
        # Even number of wires
        half_wires = num_wires // 2
        wire_positions = jnp.arange(-half_wires, half_wires)  # -6 to 5 for num_wires=12
    else:
        # Odd number of wires
        half_wires = num_wires // 2
        wire_positions = jnp.arange(-half_wires, half_wires + 1)  # -6 to 6 for num_wires=13

    wire_base_positions = wire_positions * bins_per_wire + center_w

    # 3. Time integration setup - sum fine kernel bins into simulation time bins
    # Calculate how many kernel bins per simulation time bin
    bins_per_sim_time = int(round(sim_time_spacing / kernel_time_spacing))  # e.g., 5 for 0.5/0.1
    slice_len = num_sim_time_bins * bins_per_sim_time  # total fine bins to read

    # Integer fine-bin offset from t_offset
    t_base_bin = jnp.floor(t_offset * bins_per_sim_time).astype(int)

    # Initialize output array
    output_values = jnp.zeros((num_wires, num_sim_time_bins))

    # Process each wire position
    for wire_idx in range(num_wires):
        wire_bin_left = wire_base_positions[wire_idx] + w_base_bin
        wire_bin_right = wire_bin_left + 1
        wire_bin_left = jnp.clip(wire_bin_left, 0, kernel_width - 1)
        wire_bin_right = jnp.clip(wire_bin_right, 0, kernel_width - 1)

        values_s_n_left = DKernel[s_idx, :, wire_bin_left]
        values_s_n_plus_1_left = DKernel[s_idx + 1, :, wire_bin_left]
        values_s_n_right = DKernel[s_idx, :, wire_bin_right]
        values_s_n_plus_1_right = DKernel[s_idx + 1, :, wire_bin_right]

        values_s_interp_left = (1 - s_alpha) * values_s_n_left + s_alpha * values_s_n_plus_1_left
        values_s_interp_right = (1 - s_alpha) * values_s_n_right + s_alpha * values_s_n_plus_1_right
        values_w_interp = (1 - w_alpha) * values_s_interp_left + w_alpha * values_s_interp_right

        # Time integration: dynamic_slice + reshape + sum
        # Extracts slice_len contiguous fine bins starting at t_base_bin,
        # reshapes into (num_sim_time_bins, bins_per_sim_time), and sums
        # to produce one integrated value per simulation time bin.
        chunk = lax.dynamic_slice(values_w_interp, (t_base_bin,), (slice_len,))
        integrated = chunk.reshape(num_sim_time_bins, bins_per_sim_time).sum(axis=1)
        output_values = output_values.at[wire_idx, :].set(integrated)

    return output_values


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))  # wire_stride, wire_spacing, kernel_time_spacing, sim_time_spacing, num_wires, num_sim_time_bins are static
def interpolate_diffusion_kernel_batch(DKernel, s_observed_batch, w_offset_batch, t_offset_batch,
                                     wire_stride, wire_spacing,
                                     kernel_time_spacing, sim_time_spacing,
                                     num_wires, num_sim_time_bins):
    """
    Batch interpolation using vmap for multiple sets of parameters.

    This is the key function for efficient runtime processing of many segments.

    Parameters
    ----------
    DKernel : jnp.ndarray
        Array of shape (num_s, kernel_height, kernel_width)
    s_observed_batch : jnp.ndarray
        Array of shape (N,) with s values
    w_offset_batch : jnp.ndarray
        Array of shape (N,) with w_offset values
    t_offset_batch : jnp.ndarray
        Array of shape (N,) with t_offset values
    wire_stride : int
        Static wire stride
    wire_spacing : float
        Static wire spacing
    kernel_time_spacing : float
        Static kernel time spacing (e.g., 0.1 μs for high-res)
    sim_time_spacing : float
        Static simulation time spacing (e.g., 0.5 μs)
    num_wires : int
        Static number of wires
    num_sim_time_bins : int
        Static number of output time bins (in simulation resolution)

    Returns
    -------
    batch_results : jnp.ndarray
        Batch results with shape (N, num_wires, num_sim_time_bins)
    """
    # Vmap over the batch dimension (first axis)
    vmapped_interpolate = vmap(
        lambda s, w, t: interpolate_diffusion_kernel(
            DKernel, s, w, t, wire_stride, wire_spacing,
            kernel_time_spacing, sim_time_spacing, num_wires, num_sim_time_bins
        ),
        in_axes=(0, 0, 0),  # Vmap over first axis of s, w, t
        out_axes=0          # Output has batch dimension first
    )

    return vmapped_interpolate(s_observed_batch, w_offset_batch, t_offset_batch)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def load_response_kernels(response_path="tools/responses/", num_s=16,
                         wire_spacing=0.1, time_spacing=0.5,
                         max_sigma_trans_unitless=None, max_sigma_long_unitless=None):
    """
    Load response kernels and create diffusion kernel arrays.

    Parameters
    ----------
    response_path : str
        Path to directory containing kernel NPZ files.
    num_s : int
        Number of diffusion levels to create.
    wire_spacing : float
        Wire spacing in cm.
    time_spacing : float
        Simulation time spacing in microseconds.
    max_sigma_trans_unitless : float, optional
        Maximum transverse diffusion sigma in unitless grid coordinates.
    max_sigma_long_unitless : float, optional
        Maximum longitudinal diffusion sigma in unitless grid coordinates.

    Returns
    -------
    dict
        Dictionary mapping plane names to kernel data including:
        - DKernel: diffusion kernel array
        - num_wires: number of output wires
        - kernel_height: number of output time bins (in simulation resolution)
        - wire_spacing: wire spacing
        - time_spacing: simulation time spacing
        - wire_stride: bins per wire
        - kernel_time_spacing: fine kernel time spacing
        - wire_zero_bin: where wire=0 is in output wires
        - time_zero_bin: where t=0 is in output (simulation time bins)
    """
    planes = ['U', 'V', 'Y']

    # Create diffusion kernel arrays
    DKernels = create_diffusion_kernel_array(
        planes=planes,
        num_s=num_s,
        kernel_dir=response_path,
        wire_spacing=wire_spacing,
        time_spacing=time_spacing,
        max_sigma_trans_unitless=max_sigma_trans_unitless,
        max_sigma_long_unitless=max_sigma_long_unitless
    )

    # Extract kernel info for each plane
    response_kernels = {}
    for plane in DKernels:
        DKernel, linear_s, kernel_shape, x_coords, y_coords, kernel_dx, kernel_dy, wire_zero_bin, time_zero_bin = DKernels[plane]
        num_wires = calculate_wire_count(kernel_shape[1], wire_spacing)
        bins_per_wire = int(1.0 / wire_spacing)  # e.g., 10 for 0.1 spacing

        # Compute num_sim_time_bins accounting for offset headroom
        # dynamic_slice needs: t_base_bin + num_sim_time_bins * bps <= kernel_height
        # t_base_bin can be at most (bps - 1), so:
        bins_per_sim_time = int(round(time_spacing / kernel_dy))
        kernel_height_fine = kernel_shape[0]
        num_sim_time_bins = (kernel_height_fine - bins_per_sim_time + 1) // bins_per_sim_time

        # Convert kernel time_zero_bin to simulation time bins
        # time_zero_bin is in kernel bins, scale to simulation resolution
        time_zero_bin_sim = int(round(time_zero_bin * kernel_dy / time_spacing))

        # Calculate wire_zero_bin in output wire units
        # wire_zero_bin is in kernel bins, convert to output wire position
        wire_zero_bin_out = wire_zero_bin // bins_per_wire

        response_kernels[plane] = {
            'DKernel': DKernel,
            'num_wires': num_wires,
            'kernel_height': num_sim_time_bins,      # Now in simulation time bins
            'wire_spacing': wire_spacing,
            'time_spacing': time_spacing,            # Simulation time spacing
            'wire_stride': bins_per_wire,            # 10 for 0.1 spacing
            'kernel_time_spacing': kernel_dy,        # Fine kernel time spacing (e.g., 0.1 μs)
            'wire_zero_bin': wire_zero_bin_out,      # Where wire=0 is in output wires
            'time_zero_bin': time_zero_bin_sim,      # Where t=0 is in output (sim bins)
        }

    return response_kernels


def apply_diffusion_response(DKernel, s_values, wire_offsets, time_offsets,
                           wire_stride, wire_spacing,
                           kernel_time_spacing, sim_time_spacing,
                           num_wires, num_sim_time_bins):
    """
    Apply diffusion response using pre-computed kernels.

    Parameters
    ----------
    DKernel : jnp.ndarray
        Diffusion kernel array for the plane.
    s_values : jnp.ndarray
        Array of s values (diffusion parameters) for each segment.
    wire_offsets : jnp.ndarray
        Array of wire offsets in [0, 1) for each segment.
    time_offsets : jnp.ndarray
        Array of time offsets in [0, 1) for each segment.
    wire_stride : int
        Wire stride (static parameter).
    wire_spacing : float
        Wire spacing (static parameter).
    kernel_time_spacing : float
        Fine kernel time spacing (e.g., 0.1 μs for high-res).
    sim_time_spacing : float
        Simulation time spacing (e.g., 0.5 μs).
    num_wires : int
        Number of wires in kernel (static parameter).
    num_sim_time_bins : int
        Number of output time bins in simulation resolution.

    Returns
    -------
    jnp.ndarray
        Response contributions with shape (N, num_wires, num_sim_time_bins).
    """
    # Apply batch interpolation
    contributions = interpolate_diffusion_kernel_batch(
        DKernel, s_values, wire_offsets, time_offsets,
        wire_stride, wire_spacing,
        kernel_time_spacing, sim_time_spacing,
        num_wires, num_sim_time_bins
    )

    return contributions
