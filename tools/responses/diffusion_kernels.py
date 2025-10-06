"""
Diffusion Kernels Module

This module handles creation of diffusion kernel arrays and runtime interpolation
for efficient wire response calculations with JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np




def load_kernel(filename):
    """
    Load kernel from npz file (now stored in actual values, not log scale).
    
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


def create_gaussian_kernel(shape, sigma_x, sigma_y, dx, dy):
    """
    Create a 2D Gaussian kernel with given sigmas and grid spacing.
    
    Parameters
    ----------
    shape : tuple
        (ny, nx) kernel shape
    sigma_x : float
        Sigma in x direction (wire units)
    sigma_y : float
        Sigma in y direction (microseconds)
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
    sigma_x = max(sigma_x, eps)
    sigma_y = max(sigma_y, eps)

    # Create Gaussian
    gaussian = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))

    # Normalize
    gaussian = gaussian / np.sum(gaussian)

    return gaussian


def convolve_with_gaussian(kernel, sigma_x, sigma_y, dx, dy):
    """
    Convolve kernel with Gaussian using JAX.
    
    Parameters
    ----------
    kernel : np.ndarray
        Input kernel
    sigma_x : float
        Sigma in x direction
    sigma_y : float
        Sigma in y direction
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
    gaussian = create_gaussian_kernel(kernel.shape, sigma_x, sigma_y, dx, dy)

    # Convert to JAX arrays
    kernel_jax = jnp.array(kernel)
    gaussian_jax = jnp.array(gaussian)

    # Perform convolution with 'same' mode to maintain shape
    convolved = jax.scipy.signal.convolve2d(kernel_jax, gaussian_jax, mode='same')

    return np.array(convolved), gaussian


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


def create_diffusion_kernel_array(planes=['U', 'V', 'Y'], num_s=16, kernel_dir='tools/responses',
                                 wire_spacing=0.1, time_spacing=0.5):
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
        Wire spacing
    time_spacing : float
        Time spacing
        
    Returns
    -------
    DKernels : dict
        Dictionary mapping plane to (DKernel, linear_s, kernel_shape, x_coords, y_coords)
    """
    # convolve_with_gaussian is defined in this module, no need to import
    
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
            
            print(f"\nCreating diffusion kernels for {plane} plane...")
            print(f"Kernel shape: {kernel_shape}")
            print(f"DKernel shape: {DKernel.shape}")
            
            # Create progressively diffused kernels
            for i in range(1, num_s):
                s = linear_s[i]
                
                # Calculate sigmas based on s (same formula as before with t=s)
                sigma_x = 0.7 * s + 1e-3
                sigma_y = 1.0 * s + 1e-3
                
                # Convolve with Gaussian
                convolved, _ = convolve_with_gaussian(kernel, sigma_x, sigma_y, dx, dy)
                DKernel = DKernel.at[i].set(convolved)
                
                print(f"  s[{i}] = {s:.3f}, σ_x = {sigma_x:.3f}, σ_y = {sigma_y:.3f}")
            
            DKernels[plane] = (DKernel, linear_s, kernel_shape, x_coords, y_coords)
            
        except FileNotFoundError:
            print(f"Warning: Could not find kernel file for {plane} plane")
            continue
    
    return DKernels


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
        # Get the two adjacent wire bin indices for interpolation
        wire_bin_left = wire_base_positions[wire_idx] + w_base_bin
        wire_bin_right = wire_bin_left + 1
        
        # Clamp to valid range
        wire_bin_left = jnp.clip(wire_bin_left, 0, kernel_width - 1)
        wire_bin_right = jnp.clip(wire_bin_right, 0, kernel_width - 1)
        
        # Extract values for s interpolation
        values_s_n_left = DKernel[s_idx, :, wire_bin_left]
        values_s_n_plus_1_left = DKernel[s_idx + 1, :, wire_bin_left]
        values_s_n_right = DKernel[s_idx, :, wire_bin_right]
        values_s_n_plus_1_right = DKernel[s_idx + 1, :, wire_bin_right]
        
        # S interpolation for both left and right wire positions
        values_s_interp_left = (1 - s_alpha) * values_s_n_left + s_alpha * values_s_n_plus_1_left
        values_s_interp_right = (1 - s_alpha) * values_s_n_right + s_alpha * values_s_n_plus_1_right
        
        # Wire interpolation
        values_w_interp = (1 - w_alpha) * values_s_interp_left + w_alpha * values_s_interp_right
        
        # Time interpolation - keep all time points - 1
        t_alpha = t_offset / time_spacing
        interpolated_values = (1 - t_alpha) * values_w_interp[:-1] + t_alpha * values_w_interp[1:]
        
        # Store result
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