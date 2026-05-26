"""
Wire Response Processing Module

This module provides tools for extracting wire responses from images,
creating diffusion kernels, and performing efficient runtime interpolation.
"""

# Core runtime functions - what most users need
# These are now in the main kernels module
from tools.kernels import (
    create_diffusion_kernel_array,
    interpolate_diffusion_kernel_batch,
    calculate_wire_count
)

# Extraction function - only needed for initial setup
from .response_extraction import process_single_plane

# Visualization functions
from .response_visualization_utils import (
    visualize_kernel,
    visualize_interpolation_steps,
    create_parameter_sweep_gif
)

# Version
__version__ = '1.0.0'

# Note: The extract_and_create_kernels function has been removed.
# Use extract_responses_from_images.py for one-time extraction,
# then use create_diffusion_kernel_array to load and process.


__all__ = [
    # Core runtime functions
    'create_diffusion_kernel_array',
    'interpolate_diffusion_kernel_batch',
    'calculate_wire_count',
    
    # Extraction (one-time use)
    'process_single_plane',
    
    # Visualization
    'visualize_kernel',
    'visualize_interpolation_steps',
    'create_parameter_sweep_gif'
]