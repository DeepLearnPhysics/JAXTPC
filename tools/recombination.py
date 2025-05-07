import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Any

@jit
def calculate_box_model_charge(de, dx, params):
    """
    Calculate deposited charge using the standard Box model.

    Parameters
    ----------
        de: Array of energy depositions (MeV)
        dx: Array of step lengths (cm)
        params: Tuple of parameters (field_strength, density, w_value, A, B)

    Returns
    -------
        Array of deposited charge (electrons) for each step
    """

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

def extract_params_for_box_model(detector_config):
    """
    Extract recombination parameters from the detector configuration.

    Parameters
    ----------
        detector_config: Dictionary with detector configuration parameters

    Returns
    -------
        Tuple of parameters (field_strength, density, w_value, A, B)
    """
    field_strength = detector_config['electric_field']['field_strength']
    density = detector_config['medium']['properties']['density']
    w_value = detector_config['medium']['properties']['ionization_energy']
    recomb_params = detector_config['simulation']['charge_recombination']['recomb_parameters']
    A = recomb_params['A']
    B = recomb_params['B']

    return field_strength, density, w_value, A, B

def recombine_steps(step_data, detector_config):
    """
    Process particle steps to calculate deposited charge.

    Parameters
    ----------
        step_data: Dictionary containing arrays from the particle step data
        detector_config: Dictionary with detector configuration parameters

    Returns
    -------
        Array of deposited charge for each step
    """
    params = extract_params_for_box_model(detector_config)

    # Extract de and dx arrays from step_data
    de = step_data['de']
    dx = step_data['dx']

    # No need for vmap since we're already working with arrays
    return calculate_box_model_charge(de, dx, params)

if __name__ == "__main__":
    from geometry import generate_detector
    from loader import load_particle_step_data

    config_path = "config/cubic_wireplane_config.yaml"
    detector = generate_detector(config_path)

    data_path = "mpvmpr.h5"
    event_idx = 0

    step_data = load_particle_step_data(data_path, event_idx)

    processed_charge = recombine_steps(step_data, detector)

