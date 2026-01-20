"""
Charge recombination calculations for LArTPC simulation.

This module implements the Modified Box model for calculating charge
recombination in liquid argon. The recombination factor determines how
much of the ionization charge is collected versus recombined with ions.

Modified Box Model Reference:
    ArgoNeuT Collaboration, JINST 8 (2013) P08005
    https://arxiv.org/abs/1306.1712

    R = ln(alpha + xi) / xi

    Where:
        R = survival fraction (collected electrons / initial electrons)
        alpha = 0.93 (dimensionless, ArgoNeuT fit)
        beta = 0.212 (kV/cm)(g/cm²)/MeV (ArgoNeuT fit)
        xi = beta * (dE/dx) / (rho * E)
        dE/dx = stopping power (MeV/cm)
        rho = LAr density (g/cm³)
        E = electric field (kV/cm)

    Typical survival fractions at 500 V/cm in LAr:
        MIP (dE/dx ~ 2.1 MeV/cm): R ~ 0.70 (70% survival)
        Heavily ionizing (dE/dx ~ 10 MeV/cm): R ~ 0.45 (45% survival)
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Any


@jit
def calculate_modified_box_charge(de, dx, params):
    """
    Calculate deposited charge using the Modified Box model (ArgoNeuT).

    This implementation follows LArSoft's ISCalcSeparate.cxx approach,
    which is the standard for DUNE, MicroBooNE, SBND, and ICARUS.

    Parameters
    ----------
    de : jnp.ndarray
        Array of energy depositions in MeV.
    dx : jnp.ndarray
        Array of step lengths in cm.
    params : tuple
        Tuple of parameters (field_strength_Vcm, density, w_value, alpha, beta).
        field_strength is in V/cm (converted to kV/cm internally).

    Returns
    -------
    jnp.ndarray
        Array of deposited charge (electrons) for each step.

    Notes
    -----
    Modified Box model (ArgoNeuT 2013):
        R = ln(α + ξ) / ξ
        where ξ = β × (dE/dx) / (ρ × E)

    Parameters (ArgoNeuT):
        α = 0.93 (dimensionless)
        β = 0.212 (kV/cm)(g/cm²)/MeV

    Edge case handling (matching LArSoft):
        - dE/dx is clamped to minimum of 1.0 MeV/cm to prevent formula breakdown
        - dx <= 0 returns zero charge (no valid step)

    References:
        - ArgoNeuT: JINST 8 (2013) P08005, arXiv:1306.1712
        - LArSoft: larsim/IonizationScintillation/ISCalcSeparate.cxx
    """
    field_strength_Vcm, density, w_value, alpha, beta = params

    # Convert field from V/cm to kV/cm
    field_kVcm = field_strength_Vcm / 1000.0

    # Convert w_value from eV to MeV
    w_value_mev = w_value * 1e-6

    # Ensure non-negative energy deposits
    de_safe = jnp.maximum(de, 0.0)

    # Calculate dE/dx, handling zero step length
    # Following LArSoft: dEdx = (ds <= 0.0) ? 0.0 : e / ds
    dx_positive = dx > 0.0
    de_dx_raw = jnp.where(dx_positive, de_safe / jnp.maximum(dx, 1e-10), 0.0)

    # LArSoft clamps dE/dx to minimum of 1.0 MeV/cm
    # This prevents the Modified Box formula from producing unphysical results
    # at very low ionization (where the formula gives R < 0)
    de_dx = jnp.maximum(de_dx_raw, 1.0)

    # Modified Box model: R = ln(alpha + xi) / xi
    # where xi = beta * dE/dx / (rho * E)
    # Following LArSoft: Xi = (ModBoxB / density) * dEdx / EField
    xi = (beta / density) * de_dx / jnp.maximum(field_kVcm, 1e-10)

    # Calculate survival fraction
    # LArSoft: recomb = log(fModBoxA + Xi) / Xi
    survival_fraction = jnp.log(alpha + xi) / xi

    # For dx <= 0, set survival to 0 (no valid step)
    survival_fraction = jnp.where(dx_positive, survival_fraction, 0.0)

    # Calculate initial charge and apply recombination
    initial_charge = de_safe / w_value_mev
    collected_charge = initial_charge * survival_fraction

    return collected_charge

def extract_recombination_params(detector_config):
    """
    Extract recombination parameters from the detector configuration.

    Parameters
    ----------
    detector_config : dict
        Dictionary with detector configuration parameters.

    Returns
    -------
    tuple
        Tuple of parameters (field_strength_Vcm, density, w_value, alpha, beta).
    """
    field_strength = detector_config['electric_field']['field_strength']
    density = detector_config['medium']['properties']['density']
    w_value = detector_config['medium']['properties']['ionization_energy']
    recomb_params = detector_config['simulation']['charge_recombination']['recomb_parameters']
    alpha = recomb_params['alpha']
    beta = recomb_params['beta']

    return field_strength, density, w_value, alpha, beta


def recombine_steps(step_data, detector_config):
    """
    Process particle steps to calculate deposited charge.

    Uses the Modified Box model (ArgoNeuT 2013) to calculate the
    fraction of ionization electrons that survive recombination.

    Parameters
    ----------
    step_data : dict
        Dictionary containing arrays from the particle step data.
        Must contain 'de' (energy deposits in MeV) and 'dx' (step lengths in cm).
    detector_config : dict
        Dictionary with detector configuration parameters.

    Returns
    -------
    jnp.ndarray
        Array of deposited charge (electrons) for each step.
    """
    params = extract_recombination_params(detector_config)

    # Extract de and dx arrays from step_data
    de = step_data['de']
    dx = step_data['dx']

    return calculate_modified_box_charge(de, dx, params)

if __name__ == "__main__":
    from geometry import generate_detector
    from loader import load_particle_step_data

    config_path = "config/cubic_wireplane_config.yaml"
    detector = generate_detector(config_path)

    data_path = "mpvmpr.h5"
    event_idx = 0

    step_data = load_particle_step_data(data_path, event_idx)

    processed_charge = recombine_steps(step_data, detector)

