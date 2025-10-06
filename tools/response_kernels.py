"""
Response Module

This module handles loading and applying wire response kernels with diffusion.
Uses pre-computed diffusion kernel arrays for efficient runtime interpolation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from tools.responses.diffusion_kernels import (
    load_kernel,
    create_diffusion_kernel_array,
    interpolate_diffusion_kernel_batch,
    calculate_wire_count
)


def load_response_kernels(response_path="tools/responses/", num_s=16, 
                         wire_spacing=0.1, time_spacing=0.5):
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
        time_spacing=time_spacing
    )
    
    # Extract kernel info for each plane
    kernel_info = {}
    for plane in DKernels:
        DKernel, linear_s, kernel_shape, x_coords, y_coords = DKernels[plane]
        num_wires = calculate_wire_count(kernel_shape[1], wire_spacing)
        kernel_height = kernel_shape[0] - 1  # Output height after interpolation
        
        kernel_info[plane] = {
            'DKernel': DKernel,
            'num_wires': num_wires,
            'kernel_height': kernel_height,
            'wire_spacing': wire_spacing,
            'time_spacing': time_spacing,
            'wire_stride': int(1.0 / wire_spacing)  # 10 for 0.1 spacing
        }
    
    return kernel_info


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