"""
Wire Response Processing Module

This module provides tools for wire response kernels and visualization.
"""

# Core runtime functions - what most users need
# These are now in the main kernels module
from tools.kernels import (
    create_diffusion_kernel_array,
    interpolate_diffusion_kernel_batch,
    calculate_wire_count
)

# Visualization functions
from .visualization import (
    actual_to_paper_log10,
    visualize_kernel,
    visualize_diffusion_progression,
    visualize_interpolation_steps,
    create_parameter_sweep_gif
)

# Version
__version__ = '1.0.0'

__all__ = [
    # Core runtime functions
    'create_diffusion_kernel_array',
    'interpolate_diffusion_kernel_batch',
    'calculate_wire_count',

    # Visualization
    'actual_to_paper_log10',
    'visualize_kernel',
    'visualize_diffusion_progression',
    'visualize_interpolation_steps',
    'create_parameter_sweep_gif'
]
