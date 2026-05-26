"""
Physics parameters container for gradient testing.

This module defines a PhysicsParams container that holds the physics
parameters we want to test gradients for. The parameters are stored
as JAX-compatible values for differentiation.
"""

from typing import NamedTuple
import jax.numpy as jnp


class PhysicsParams(NamedTuple):
    """
    Container for physics parameters used in gradient testing.

    All parameters are stored as floats for JAX compatibility.
    This is a NamedTuple which is automatically a valid JAX pytree.
    """
    # Diffusion parameters (in cm²/μs)
    diffusion_long: float   # Longitudinal diffusion coefficient
    diffusion_trans: float  # Transverse diffusion coefficient

    # Recombination parameters (dimensionless)
    recomb_A: float  # Box model A parameter
    recomb_B: float  # Box model B parameter

    # Medium/field parameters (for future expansion)
    field_strength: float  # V/cm
    density: float         # g/cm³
    w_value: float         # eV (ionization energy)


def create_default_params() -> PhysicsParams:
    """
    Create PhysicsParams with default values from the detector config.

    These are the default values from cubic_wireplane_config.yaml.

    Returns
    -------
    PhysicsParams
        Default physics parameters.
    """
    return PhysicsParams(
        # Diffusion: 6.2 cm²/s and 16.3 cm²/s converted to cm²/μs
        diffusion_long=6.2e-6,   # 6.2 cm²/s = 6.2e-6 cm²/μs
        diffusion_trans=16.3e-6, # 16.3 cm²/s = 16.3e-6 cm²/μs

        # Recombination
        recomb_A=0.8,
        recomb_B=0.3,

        # Medium/field
        field_strength=500.0,  # V/cm
        density=1.396,         # g/cm³
        w_value=23.6,          # eV
    )


def perturb_param(
    params: PhysicsParams,
    param_name: str,
    delta: float
) -> PhysicsParams:
    """
    Create a new PhysicsParams with one parameter perturbed.

    Parameters
    ----------
    params : PhysicsParams
        Original parameters.
    param_name : str
        Name of the parameter to perturb.
    delta : float
        Amount to add to the parameter value.

    Returns
    -------
    PhysicsParams
        New parameters with the specified field perturbed.
    """
    current_value = getattr(params, param_name)
    new_value = current_value + delta
    return params._replace(**{param_name: new_value})


def scale_param(
    params: PhysicsParams,
    param_name: str,
    scale: float
) -> PhysicsParams:
    """
    Create a new PhysicsParams with one parameter scaled.

    Parameters
    ----------
    params : PhysicsParams
        Original parameters.
    param_name : str
        Name of the parameter to scale.
    scale : float
        Multiplicative factor.

    Returns
    -------
    PhysicsParams
        New parameters with the specified field scaled.
    """
    current_value = getattr(params, param_name)
    new_value = current_value * scale
    return params._replace(**{param_name: new_value})


def get_param_range(
    params: PhysicsParams,
    param_name: str,
    relative_range: float = 0.5,
    num_points: int = 21
) -> tuple:
    """
    Generate a range of values for parameter sweeps.

    Parameters
    ----------
    params : PhysicsParams
        Base parameters.
    param_name : str
        Name of the parameter to sweep.
    relative_range : float
        Fraction of the default value for the range (e.g., 0.5 = ±50%).
    num_points : int
        Number of points in the sweep.

    Returns
    -------
    tuple
        (param_values, all_params_list) where param_values is an array of
        parameter values and all_params_list is a list of PhysicsParams.
    """
    base_value = getattr(params, param_name)

    # Create range from (1-relative_range) to (1+relative_range) times base
    scales = jnp.linspace(1.0 - relative_range, 1.0 + relative_range, num_points)
    param_values = base_value * scales

    # Create list of PhysicsParams
    all_params = [
        params._replace(**{param_name: float(val)})
        for val in param_values
    ]

    return param_values, all_params


# Parameter metadata for plotting/reporting
PARAM_INFO = {
    'diffusion_long': {
        'name': 'Longitudinal Diffusion',
        'units': 'cm²/μs',
        'latex': r'$D_L$',
    },
    'diffusion_trans': {
        'name': 'Transverse Diffusion',
        'units': 'cm²/μs',
        'latex': r'$D_T$',
    },
    'recomb_A': {
        'name': 'Recombination A',
        'units': '-',
        'latex': r'$A$',
    },
    'recomb_B': {
        'name': 'Recombination B',
        'units': '-',
        'latex': r'$B$',
    },
    'field_strength': {
        'name': 'Electric Field',
        'units': 'V/cm',
        'latex': r'$E$',
    },
    'density': {
        'name': 'LAr Density',
        'units': 'g/cm³',
        'latex': r'$\rho$',
    },
    'w_value': {
        'name': 'Ionization Energy',
        'units': 'eV',
        'latex': r'$W$',
    },
}
