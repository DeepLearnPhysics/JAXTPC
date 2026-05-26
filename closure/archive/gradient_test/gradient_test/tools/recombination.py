"""
Charge recombination calculations for LArTPC simulation.

This module implements the Box model for calculating charge recombination
in liquid argon. The recombination factor determines how much of the
ionization charge is collected versus recombined with ions.

Modified for gradient testing - accepts PhysicsParams for differentiation.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Any, Optional, Union

# Import PhysicsParams from gradient_test package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics_params import PhysicsParams


def calculate_box_model_charge(de, dx, params):
    """
    Calculate deposited charge using the standard Box model.

    This function is NOT jit-compiled to allow gradient computation
    through the physics parameters.

    Parameters
    ----------
    de : jnp.ndarray
        Array of energy depositions in MeV.
    dx : jnp.ndarray
        Array of step lengths in cm.
    params : tuple or PhysicsParams
        Either a tuple of (field_strength, density, w_value, A, B) or
        a PhysicsParams object.

    Returns
    -------
    jnp.ndarray
        Array of deposited charge (electrons) for each step.
    """
    # Handle both tuple and PhysicsParams inputs
    if isinstance(params, PhysicsParams):
        field_strength = params.field_strength
        density = params.density
        w_value = params.w_value
        A = params.recomb_A
        B = params.recomb_B
    else:
        field_strength, density, w_value, A, B = params

    # Convert w_value from eV to MeV
    w_value_mev = w_value * 1e-6

    # Calculate dE/dx (MeV/cm)
    de_dx = de / jnp.maximum(dx, 1e-10)

    # Calculate recombination factor using Box model
    denominator = 1.0 + (B * field_strength) / (density * jnp.maximum(de_dx, 1e-10))
    recombination_factor = A / denominator
    recombination_factor = jnp.clip(recombination_factor, 0.0, 1.0)

    # Calculate initial charge and apply recombination
    initial_charge = de / w_value_mev
    collected_charge = initial_charge * (1.0 - recombination_factor)

    return collected_charge


def calculate_box_model_charge_with_physics_params(
    de: jnp.ndarray,
    dx: jnp.ndarray,
    physics_params: PhysicsParams
) -> jnp.ndarray:
    """
    Calculate deposited charge using PhysicsParams.

    This is the gradient-friendly version that takes PhysicsParams directly.

    Parameters
    ----------
    de : jnp.ndarray
        Array of energy depositions in MeV.
    dx : jnp.ndarray
        Array of step lengths in cm.
    physics_params : PhysicsParams
        Physics parameters for recombination.

    Returns
    -------
    jnp.ndarray
        Array of deposited charge (electrons) for each step.
    """
    # Extract parameters
    field_strength = physics_params.field_strength
    density = physics_params.density
    w_value = physics_params.w_value
    A = physics_params.recomb_A
    B = physics_params.recomb_B

    # Convert w_value from eV to MeV
    w_value_mev = w_value * 1e-6

    # Calculate dE/dx (MeV/cm)
    de_dx = de / jnp.maximum(dx, 1e-10)

    # Calculate recombination factor using Box model
    denominator = 1.0 + (B * field_strength) / (density * jnp.maximum(de_dx, 1e-10))
    recombination_factor = A / denominator
    recombination_factor = jnp.clip(recombination_factor, 0.0, 1.0)

    # Calculate initial charge and apply recombination
    initial_charge = de / w_value_mev
    collected_charge = initial_charge * (1.0 - recombination_factor)

    return collected_charge


def extract_params_for_box_model(detector_config):
    """
    Extract recombination parameters from the detector configuration.

    Parameters
    ----------
    detector_config : dict
        Dictionary with detector configuration parameters.

    Returns
    -------
    tuple
        Tuple of parameters (field_strength, density, w_value, A, B).
    """
    field_strength = detector_config['electric_field']['field_strength']
    density = detector_config['medium']['properties']['density']
    w_value = detector_config['medium']['properties']['ionization_energy']
    recomb_params = detector_config['simulation']['charge_recombination']['recomb_parameters']
    A = recomb_params['A']
    B = recomb_params['B']

    return field_strength, density, w_value, A, B


def recombine_steps(
    step_data: Dict,
    detector_config: Dict,
    physics_params: Optional[PhysicsParams] = None
) -> jnp.ndarray:
    """
    Process particle steps to calculate deposited charge.

    Parameters
    ----------
    step_data : dict
        Dictionary containing arrays from the particle step data.
    detector_config : dict
        Dictionary with detector configuration parameters.
    physics_params : PhysicsParams, optional
        If provided, use these parameters instead of detector_config.
        This allows gradient computation through the parameters.

    Returns
    -------
    jnp.ndarray
        Array of deposited charge for each step.
    """
    # Extract de and dx arrays from step_data
    de = step_data['de']
    dx = step_data['dx']

    if physics_params is not None:
        # Use PhysicsParams for gradient-friendly computation
        return calculate_box_model_charge_with_physics_params(de, dx, physics_params)
    else:
        # Use detector_config (original behavior)
        params = extract_params_for_box_model(detector_config)
        return calculate_box_model_charge(de, dx, params)


def recombine_steps_with_physics_params(
    step_data: Dict,
    physics_params: PhysicsParams
) -> jnp.ndarray:
    """
    Convenience function for gradient testing.

    Parameters
    ----------
    step_data : dict
        Dictionary containing 'de' and 'dx' arrays.
    physics_params : PhysicsParams
        Physics parameters.

    Returns
    -------
    jnp.ndarray
        Array of deposited charge for each step.
    """
    de = step_data['de']
    dx = step_data['dx']
    return calculate_box_model_charge_with_physics_params(de, dx, physics_params)


if __name__ == "__main__":
    from geometry import generate_detector
    from loader import load_particle_step_data

    config_path = "config/cubic_wireplane_config.yaml"
    detector = generate_detector(config_path)

    data_path = "mpvmpr.h5"
    event_idx = 0

    step_data = load_particle_step_data(data_path, event_idx)

    processed_charge = recombine_steps(step_data, detector)
