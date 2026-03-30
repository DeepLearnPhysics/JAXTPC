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

Edge Cases
----------
Both models use a natural extension at low ionization density:
    - Instead of clamping dE/dx to 1.0 MeV/cm (LArSoft default), the log
      argument is clamped: R = ln(max(α + ξ, 1)) / ξ.
    - For α + ξ ≥ 1 (dE/dx above ~0.23 MeV/cm): standard Modified Box result.
    - For α + ξ < 1 (very low ionization): R = 0, Q = 0 (zero charge).
    - This is smooth, monotone, and differentiable — suitable for gradient-based
      optimization while remaining physically sensible (no negative R).
    - dx ≤ 0 returns zero charge (invalid step).
    - Negative dE values are clamped to zero.
"""

import jax.numpy as jnp
from jax import jit


# Valid recombination model names
RECOMB_MODELS = ('modified_box', 'emb')


@jit
def calculate_modified_box_charge(de, dx, phi_drift, e_field_Vcm, params):
    """
    Calculate deposited charge using the Modified Box model (ArgoNeuT 2013).

    Parameters
    ----------
    de : jnp.ndarray, shape (N,)
        Energy depositions in MeV.
    dx : jnp.ndarray, shape (N,)
        Step lengths in cm.
    phi_drift : jnp.ndarray, shape (N,)
        Angle between track and drift field (accepted but unused).
    e_field_Vcm : scalar or jnp.ndarray, shape (N,)
        Local E-field magnitude in V/cm (from SCE or nominal).
    params : ModifiedBoxParams
        NamedTuple with fields: density, w_value, field_strength_Vcm, alpha, beta.

    Returns
    -------
    jnp.ndarray, shape (N,)
        Deposited charge (electrons) for each step.

    Notes
    -----
    Modified Box model (ArgoNeuT 2013):
        R = ln(α + ξ) / ξ
        where ξ = β × (dE/dx) / (ρ × E)

    Parameters (ArgoNeuT):
        α = 0.93 (dimensionless)
        β = 0.212 (kV/cm)(g/cm²)/MeV

    Edge case handling:
        - Natural extension: R = ln(max(α+ξ, 1))/ξ, giving R → 0 at low dE/dx
        - dx <= 0 returns zero charge (no valid step)
        - Safe denominator: max(xi, 1e-10) keeps backward pass finite

    References
    ----------
    - ArgoNeuT: JINST 8 (2013) P08005, arXiv:1306.1712
    - LArSoft: larsim/IonizationScintillation/ISCalcSeparate.cxx
    """
    field_kVcm = e_field_Vcm / 1000.0
    w_value_mev = params.w_value * 1e-6
    de_safe = jnp.maximum(de, 0.0)
    dx_positive = dx > 0.0
    de_dx_raw = jnp.where(dx_positive, de_safe / jnp.maximum(dx, 1e-10), 0.0)

    xi = (params.beta / params.density) * de_dx_raw / jnp.maximum(field_kVcm, 1e-10)
    ln_arg = jnp.maximum(params.alpha + xi, 1.0)
    safe_xi = jnp.maximum(xi, 1e-10)
    survival_fraction = jnp.where(xi > 1e-10, jnp.log(ln_arg) / safe_xi, 0.0)
    survival_fraction = jnp.where(dx_positive, survival_fraction, 0.0)

    initial_charge = de_safe / w_value_mev
    return initial_charge * survival_fraction


@jit
def calculate_emb_charge(de, dx, phi_drift, e_field_Vcm, params):
    """
    Calculate deposited charge using the Ellipsoid Modified Box (EMB) model.

    The EMB model (ICARUS 2024, arXiv:2407.12969) extends the Modified Box
    model with an angular dependence on the track angle to the drift field:

        beta_eff(phi) = beta_90 / sqrt(sin²phi + cos²phi / R²)

    Parameters
    ----------
    de : jnp.ndarray, shape (N,)
        Energy depositions in MeV.
    dx : jnp.ndarray, shape (N,)
        Step lengths in cm.
    phi_drift : jnp.ndarray, shape (N,)
        Angle between track direction and drift electric field in radians.
        phi=0 means parallel to field, phi=pi/2 means perpendicular.
    e_field_Vcm : scalar or jnp.ndarray, shape (N,)
        Local E-field magnitude in V/cm (from SCE or nominal).
    params : EMBParams
        NamedTuple with fields: density, w_value, field_strength_Vcm, alpha, beta_90, R.

    Returns
    -------
    jnp.ndarray, shape (N,)
        Deposited charge (electrons) for each step.

    Notes
    -----
    EMB model (ICARUS 2024):
        R = ln(α + ξ(φ)) / ξ(φ)
        ξ(φ) = β_eff(φ) / (ρ × E) × dE/dx
        β_eff(φ) = β_90 / √(sin²φ + cos²φ / R²)

    Parameters (ICARUS):
        α   = 0.904
        β_90 = 0.204 (kV/cm)(g/cm²)/MeV
        R   = 1.25

    References
    ----------
    - ICARUS: arXiv:2407.12969
    """
    field_kVcm = e_field_Vcm / 1000.0
    w_value_mev = params.w_value * 1e-6
    de_safe = jnp.maximum(de, 0.0)
    dx_positive = dx > 0.0
    de_dx_raw = jnp.where(dx_positive, de_safe / jnp.maximum(dx, 1e-10), 0.0)

    sin_phi = jnp.sin(phi_drift)
    cos_phi = jnp.cos(phi_drift)
    angular_factor = jnp.sqrt(sin_phi**2 + cos_phi**2 / (params.R**2))
    effective_beta = params.beta_90 / jnp.maximum(angular_factor, 1e-10)

    xi = (effective_beta / params.density) * de_dx_raw / jnp.maximum(field_kVcm, 1e-10)
    ln_arg = jnp.maximum(params.alpha + xi, 1.0)
    safe_xi = jnp.maximum(xi, 1e-10)
    survival_fraction = jnp.where(xi > 1e-10, jnp.log(ln_arg) / safe_xi, 0.0)
    survival_fraction = jnp.where(dx_positive, survival_fraction, 0.0)

    initial_charge = de_safe / w_value_mev
    return initial_charge * survival_fraction
