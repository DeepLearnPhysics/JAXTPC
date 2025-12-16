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

    return kernel, kernel_x_coords, kernel_y_coords, plane, dx, dy


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
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
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
            kernel, x_coords, y_coords, loaded_plane, dx, dy = load_kernel(filename)

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
                # These are in UNITLESS grid coordinates (not physical units)
                if max_sigma_trans_unitless is not None and max_sigma_long_unitless is not None:
                    sigma_trans = max_sigma_trans_unitless * np.sqrt(s)  # unitless
                    sigma_long = max_sigma_long_unitless * np.sqrt(s)    # unitless
                else:
                    # Fallback to old hardcoded values if not provided
                    sigma_trans = 0.7 * s + 1e-3
                    sigma_long = 1.0 * s + 1e-3

                # Convolve with Gaussian
                convolved, _ = convolve_with_gaussian(kernel, sigma_trans, sigma_long, dx, dy)
                DKernel = DKernel.at[i].set(convolved)

            DKernels[plane] = (DKernel, linear_s, kernel_shape, x_coords, y_coords)

        except FileNotFoundError:
            print(f"Warning: Could not find kernel file for {plane} plane")
            continue

    return DKernels


# ============================================================================
# RUNTIME INTERPOLATION (JIT-compiled)
# ============================================================================

@partial(jit, static_argnums=(4, 5, 6, 7))  # wire_stride, wire_spacing, time_spacing, num_wires are static
def interpolate_diffusion_kernel(DKernel, s_observed, w_offset, t_offset,
                               wire_stride, wire_spacing, time_spacing, num_wires):
    """
    Interpolate the diffusion kernel at given s, w, t offsets.

    This is the core runtime function for efficient kernel interpolation.

    Parameters
    ----------
    DKernel : jnp.ndarray
        Array of shape (num_s, kernel_height, kernel_width)
    s_observed : float
        Diffusion parameter in [0, 1]
    w_offset : float
        Wire offset in [0, 1.0) - wire offset in units of wire_spacing
    t_offset : float
        Time offset in [0, 0.5) - time offset in units of time_spacing
    wire_stride : int
        Static wire stride (10 for 0.1 spacing to 1.0 spacing)
    wire_spacing : float
        Static wire spacing (0.1)
    time_spacing : float
        Static time spacing (0.5)
    num_wires : int
        Static number of wire positions expected

    Returns
    -------
    interpolated_values : jnp.ndarray
        Interpolated kernel values with shape (num_wires, kernel_height-1)
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

    # Initialize output array
    output_values = jnp.zeros((num_wires, kernel_height - 1))

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

        t_alpha = t_offset / time_spacing
        interpolated_values = (1 - t_alpha) * values_w_interp[:-1] + t_alpha * values_w_interp[1:]
        output_values = output_values.at[wire_idx, :].set(interpolated_values)

    return output_values


@partial(jit, static_argnums=(4, 5, 6, 7))  # wire_stride, wire_spacing, time_spacing, num_wires are static
def interpolate_diffusion_kernel_batch(DKernel, s_observed_batch, w_offset_batch, t_offset_batch,
                                     wire_stride, wire_spacing, time_spacing, num_wires):
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
    time_spacing : float
        Static time spacing
    num_wires : int
        Static number of wires

    Returns
    -------
    batch_results : jnp.ndarray
        Batch results with shape (N, num_wires, kernel_height-1)
    """
    # Vmap over the batch dimension (first axis)
    vmapped_interpolate = vmap(
        lambda s, w, t: interpolate_diffusion_kernel(
            DKernel, s, w, t, wire_stride, wire_spacing, time_spacing, num_wires
        ),
        in_axes=(0, 0, 0),  # Vmap over first axis of s, w, t
        out_axes=0          # Output has batch dimension first
    )

    return vmapped_interpolate(s_observed_batch, w_offset_batch, t_offset_batch)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def load_response_kernels(response_path="tools_refactored/responses/", num_s=16,
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
        Time spacing in microseconds.
    max_sigma_trans_unitless : float, optional
        Maximum transverse diffusion sigma in unitless grid coordinates.
    max_sigma_long_unitless : float, optional
        Maximum longitudinal diffusion sigma in unitless grid coordinates.

    Returns
    -------
    dict
        Dictionary mapping plane names to kernel data.
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
        DKernel, linear_s, kernel_shape, x_coords, y_coords = DKernels[plane]
        num_wires = calculate_wire_count(kernel_shape[1], wire_spacing)
        kernel_height = kernel_shape[0] - 1  # Output height after interpolation

        response_kernels[plane] = {
            'DKernel': DKernel,
            'num_wires': num_wires,
            'kernel_height': kernel_height,
            'wire_spacing': wire_spacing,
            'time_spacing': time_spacing,
            'wire_stride': int(1.0 / wire_spacing)  # 10 for 0.1 spacing
        }

    return response_kernels


def apply_diffusion_response(DKernel, s_values, wire_offsets, time_offsets,
                           wire_stride, wire_spacing, time_spacing, num_wires):
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
        Array of time offsets in [0, 0.5) for each segment.
    wire_stride : int
        Wire stride (static parameter).
    wire_spacing : float
        Wire spacing (static parameter).
    time_spacing : float
        Time spacing (static parameter).
    num_wires : int
        Number of wires in kernel (static parameter).

    Returns
    -------
    jnp.ndarray
        Response contributions with shape (N, num_wires, kernel_height).
    """
    # Apply batch interpolation
    contributions = interpolate_diffusion_kernel_batch(
        DKernel, s_values, wire_offsets, time_offsets,
        wire_stride, wire_spacing, time_spacing, num_wires
    )

    return contributions
