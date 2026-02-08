"""
Charge recombination calculations for LArTPC simulation.

This module computes the fraction of ionization electrons that survive
recombination in liquid argon, converting energy deposits (dE) into
collected charge (number of electrons). Two models are available:

Available Models
----------------

1. ``'modified_box'`` — Modified Box Model (ArgoNeuT 2013)
    ArgoNeuT Collaboration, JINST 8 (2013) P08005
    https://arxiv.org/abs/1306.1712

    Survival fraction:
        R = ln(α + ξ) / ξ
        ξ = β / (ρ × E) × dE/dx

    Default parameters (ArgoNeuT):
        α = 0.93 (dimensionless)
        β = 0.212 (kV/cm)(g/cm²)/MeV

    This is the standard model used in LArSoft (ISCalcSeparate.cxx) for
    DUNE, MicroBooNE, SBND, and ICARUS. It has no angular dependence —
    recombination depends only on the local ionization density dE/dx.

2. ``'emb'`` — Ellipsoid Modified Box Model (ICARUS 2024)
    ICARUS Collaboration, arXiv:2407.12969
    https://arxiv.org/abs/2407.12969

    Survival fraction:
        R = ln(α + ξ(φ)) / ξ(φ)
        ξ(φ) = β_eff(φ) / (ρ × E) × dE/dx
        β_eff(φ) = β_90 / √(sin²φ + cos²φ / R²)

    Where φ is the angle between the track direction and the drift
    electric field (x-axis). φ = 0 means parallel, φ = π/2 perpendicular.

    Default parameters (ICARUS):
        α   = 0.904 (dimensionless)
        β_90 = 0.204 (kV/cm)(g/cm²)/MeV — beta at φ = 90°
        R   = 1.25 (ellipsoid anisotropy ratio)

    Extends the Modified Box model with an angular correction that accounts
    for the anisotropic column charge density seen by the drift field.
    Tracks parallel to the field (φ → 0) have higher effective β (more
    recombination) than perpendicular tracks (φ = 90°), matching the
    physical expectation that aligned columns produce denser charge.

Usage
-----
Use ``create_recombination_fn()`` to get a model-specific callable::

    from tools.recombination import create_recombination_fn

    recomb_fn, model_name = create_recombination_fn(detector_config)
    charges = recomb_fn(de, dx_cm, phi_drift)

The model is selected from the YAML config key ``simulation.charge_recombination.model``
or overridden via the ``model`` argument. Both models return a function with the
same signature ``fn(de, dx_cm, phi_drift) -> charges`` for drop-in interchangeability.

Edge Cases
----------
Both models share the following edge-case handling (matching LArSoft):
    - dE/dx is clamped to a minimum of 1.0 MeV/cm to prevent the log formula
      from producing unphysical (negative) survival fractions at very low ionization.
    - dx ≤ 0 returns zero charge (invalid step).
    - Negative dE values are clamped to zero.
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
    Extract Modified Box recombination parameters from the detector configuration.

    Falls back to ArgoNeuT 2013 defaults if config keys are missing.

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
    recomb_params = detector_config.get('simulation', {}).get('charge_recombination', {}).get('recomb_parameters', {})
    alpha = recomb_params.get('alpha', 0.93)
    beta = recomb_params.get('beta', 0.212)

    return field_strength, density, w_value, alpha, beta


def extract_emb_params(detector_config):
    """
    Extract Ellipsoid Modified Box (EMB) parameters from the detector configuration.

    Falls back to ICARUS 2024 (arXiv:2407.12969) defaults if config keys are missing.

    Parameters
    ----------
    detector_config : dict
        Dictionary with detector configuration parameters.

    Returns
    -------
    tuple
        Tuple of parameters (field_strength_Vcm, density, w_value, alpha, beta_90, R).
    """
    field_strength = detector_config['electric_field']['field_strength']
    density = detector_config['medium']['properties']['density']
    w_value = detector_config['medium']['properties']['ionization_energy']
    recomb_params = detector_config.get('simulation', {}).get('charge_recombination', {}).get('recomb_parameters', {})
    alpha = recomb_params.get('alpha_emb', 0.904)
    beta_90 = recomb_params.get('beta_90', 0.204)
    R = recomb_params.get('R_anisotropy', 1.25)

    return field_strength, density, w_value, alpha, beta_90, R


@jit
def calculate_emb_charge(de, dx, phi_drift, params):
    """
    Calculate deposited charge using the Ellipsoid Modified Box (EMB) model.

    The EMB model (ICARUS 2024, arXiv:2407.12969) extends the Modified Box
    model with an angular dependence on the track angle to the drift field:

        beta_eff(phi) = beta_90 / sqrt(sin²phi + cos²phi / R²)

    Parameters
    ----------
    de : jnp.ndarray
        Array of energy depositions in MeV.
    dx : jnp.ndarray
        Array of step lengths in cm.
    phi_drift : jnp.ndarray
        Array of angles between track direction and drift electric field
        in radians. phi=0 means parallel to field, phi=pi/2 means perpendicular.
    params : tuple
        Tuple of parameters (field_strength_Vcm, density, w_value, alpha, beta_90, R).

    Returns
    -------
    jnp.ndarray
        Array of deposited charge (electrons) for each step.
    """
    field_strength_Vcm, density, w_value, alpha, beta_90, R = params

    # Convert field from V/cm to kV/cm
    field_kVcm = field_strength_Vcm / 1000.0

    # Convert w_value from eV to MeV
    w_value_mev = w_value * 1e-6

    # Ensure non-negative energy deposits
    de_safe = jnp.maximum(de, 0.0)

    # Calculate dE/dx, handling zero step length
    dx_positive = dx > 0.0
    de_dx_raw = jnp.where(dx_positive, de_safe / jnp.maximum(dx, 1e-10), 0.0)

    # Clamp dE/dx to minimum of 1.0 MeV/cm (matching LArSoft)
    de_dx = jnp.maximum(de_dx_raw, 1.0)

    # EMB angular correction: effective beta depends on track-to-field angle
    # beta_eff(phi) = beta_90 / sqrt(sin²phi + cos²phi / R²)
    sin_phi = jnp.sin(phi_drift)
    cos_phi = jnp.cos(phi_drift)
    angular_factor = jnp.sqrt(sin_phi**2 + cos_phi**2 / (R**2))
    effective_beta = beta_90 / jnp.maximum(angular_factor, 1e-10)

    # Modified Box formula with angular-dependent beta
    xi = (effective_beta / density) * de_dx / jnp.maximum(field_kVcm, 1e-10)
    survival_fraction = jnp.log(alpha + xi) / xi

    # For dx <= 0, set survival to 0 (no valid step)
    survival_fraction = jnp.where(dx_positive, survival_fraction, 0.0)

    # Calculate initial charge and apply recombination
    initial_charge = de_safe / w_value_mev
    collected_charge = initial_charge * survival_fraction

    return collected_charge


def create_recombination_fn(detector_config, model=None):
    """
    Factory to create a recombination function for use in the simulation.

    Returns a JIT-compatible function with a unified signature so that models
    can be swapped without changing the calling code.

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector(). Physical constants
        (E-field, density, W-value) are always read from this dict. Model-
        specific parameters fall back to published defaults if absent.
    model : str, optional
        Which recombination model to use:

        - ``'modified_box'`` : Standard Modified Box (ArgoNeuT 2013).
          Angle-independent. Uses config keys ``alpha``, ``beta``.
          Default values: α = 0.93, β = 0.212.
        - ``'emb'`` : Ellipsoid Modified Box (ICARUS 2024).
          Angle-dependent via phi_drift. Uses config keys ``alpha_emb``,
          ``beta_90``, ``R_anisotropy``.
          Default values: α = 0.904, β_90 = 0.204, R = 1.25.

        If None, reads from ``simulation.charge_recombination.model`` in the
        config dict, falling back to ``'modified_box'`` if not specified.

    Returns
    -------
    tuple of (callable, str)
        - **recomb_fn** : function with signature
          ``fn(de, dx_cm, phi_drift) -> charges``
          where de (MeV), dx_cm (cm), phi_drift (rad) are JAX arrays and
          charges is the number of surviving electrons per step.
          For ``'modified_box'``, phi_drift is accepted but ignored.
        - **model_name** : the string name of the selected model.
    """
    if model is None:
        model = (detector_config
                 .get('simulation', {})
                 .get('charge_recombination', {})
                 .get('model', 'modified_box'))

    if model == 'modified_box':
        params = extract_recombination_params(detector_config)

        def recomb_fn(de, dx_cm, phi_drift):
            return calculate_modified_box_charge(de, dx_cm, params)

        return recomb_fn, model

    elif model == 'emb':
        params = extract_emb_params(detector_config)

        def recomb_fn(de, dx_cm, phi_drift):
            return calculate_emb_charge(de, dx_cm, phi_drift, params)

        return recomb_fn, model

    else:
        raise ValueError(
            f"Unknown recombination model: '{model}'. "
            f"Supported models: 'modified_box', 'emb'."
        )


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

