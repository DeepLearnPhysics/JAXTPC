"""
JAXTPC Gradient Testing Framework

This module provides utilities for testing gradients of the TPC simulation
with respect to physics parameters like diffusion and recombination.

Structure:
- physics_params.py: PhysicsParams container for differentiable parameters
- loss_functions.py: MSE and other loss functions
- gradient_utils.py: Numerical and analytical gradient computation
- visualization.py: Plotting utilities
- run_gradient_test.py: Main test script
- tools/: Modified copy of main tools/ for gradient testing
"""

from .physics_params import (
    PhysicsParams,
    create_default_params,
    perturb_param,
    scale_param,
    get_param_range,
    PARAM_INFO,
)

from .loss_functions import (
    mse_loss_dense,
    mse_loss_sparse,
    total_charge_loss,
    charge_sum_loss,
    per_plane_mse_loss,
)

from .gradient_utils import (
    compute_numerical_gradient,
    compute_all_numerical_gradients,
    try_analytical_gradient,
    compare_gradients,
    gradient_check,
    loss_landscape_1d,
)
