"""
Charge and scintillation light calculations for LArTPC simulation.

This module computes both the number of ionization electrons (charge, Q)
and scintillation photons (light, L) that result from energy deposits
in liquid argon. The charge-light split is anti-correlated: their sum
is fixed by energy conservation (Q + L = ΔE / W_ph).

Physics
-------
Energy deposited by a charged particle produces:
    - N_i = ΔE / W_ion electron-ion pairs
    - N_ex = α × N_i excitons (Ar*)

where α = N_ex/N_i = 0.21 is the excitation ratio (universal LAr constant)
and W_ion = (1 + α) × W_ph relates the two work functions:
    - W_ion = 23.6 eV (energy per ion pair)
    - W_ph  = 19.5 eV (energy per quantum — either photon or ion pair)

After recombination with survival fraction R:
    Q = N_i × R               (electrons that escape)
    L = N_ex + N_i × (1 - R)  (photons from excitation + recombination)
      = ΔE / W_ph − Q         (by energy conservation)

References: arXiv:1909.07920 (LArIAT), Eqs. 1-2.

Available Models
----------------

1. ``'modified_box'`` — Modified Box Model (ArgoNeuT 2013)
    ArgoNeuT Collaboration, JINST 8 (2013) P08005
    https://arxiv.org/abs/1306.1712

    ξ = β / (ρ × E) × dE/dx

    Default parameters (ArgoNeuT):
        α = 0.93 (dimensionless)
        β = 0.212 (kV/cm)(g/cm²)/MeV

    No angular dependence — recombination depends only on dE/dx.

2. ``'emb'`` — Ellipsoid Modified Box Model (ICARUS 2024)
    ICARUS Collaboration, arXiv:2407.12969
    https://arxiv.org/abs/2407.12969

    ξ(φ) = β_eff(φ) / (ρ × E) × dE/dx
    β_eff(φ) = β_90 / √(sin²φ + cos²φ / R²)

    Default parameters (ICARUS):
        α   = 0.904 (dimensionless)
        β_90 = 0.204 (kV/cm)(g/cm²)/MeV — beta at φ = 90°
        R   = 1.25 (ellipsoid anisotropy ratio)

    Adds angular correction: tracks parallel to the drift field
    (φ → 0) have more recombination than perpendicular tracks.

Both models share the same survival fraction formula:
    R = ln(max(α + ξ, 1)) / ξ

Edge Cases
----------
- log argument clamped: R = ln(max(α + ξ, 1)) / ξ → R = 0 at very low dE/dx
- dx ≤ 0 returns zero charge and zero light (invalid step)
- Negative dE values are clamped to zero
"""

import jax.numpy as jnp

# Valid recombination model names
RECOMB_MODELS = ('modified_box', 'emb')


def _xi_modified_box(de_dx, field_kVcm, phi_drift, params):
    """Modified Box ionization density parameter: ξ = β/(ρ·E) · dE/dx."""
    return (params.beta / params.density) * de_dx / jnp.maximum(field_kVcm, 1e-10)


def _xi_emb(de_dx, field_kVcm, phi_drift, params):
    """EMB ionization density parameter with angular β_eff(φ)."""
    sin_phi = jnp.sin(phi_drift)
    cos_phi = jnp.cos(phi_drift)
    angular_factor = jnp.sqrt(sin_phi**2 + cos_phi**2 / (params.R**2))
    effective_beta = params.beta_90 / jnp.maximum(angular_factor, 1e-10)
    return (effective_beta / params.density) * de_dx / jnp.maximum(field_kVcm, 1e-10)


XI_FN = {
    'modified_box': _xi_modified_box,
    'emb': _xi_emb,
}


def compute_quanta(de, dx, phi_drift, e_field_Vcm, params, xi_fn):
    """Compute charge (Q) and scintillation light (L) per segment.

    Parameters
    ----------
    de : jnp.ndarray, shape (N,)
        Energy depositions in MeV.
    dx : jnp.ndarray, shape (N,)
        Step lengths in cm.
    phi_drift : jnp.ndarray, shape (N,)
        Angle between track direction and drift electric field in radians.
    e_field_Vcm : scalar or jnp.ndarray, shape (N,)
        Local E-field magnitude in V/cm.
    params : ModifiedBoxParams or EMBParams
        Recombination parameters (includes w_value and excitation_ratio).
    xi_fn : callable
        Model-specific function: (de_dx, field_kVcm, phi_drift, params) -> xi.

    Returns
    -------
    Q : jnp.ndarray, shape (N,)
        Number of ionization electrons surviving recombination.
    L : jnp.ndarray, shape (N,)
        Number of scintillation photons produced.
    """
    field_kVcm = e_field_Vcm / 1000.0
    de_safe = jnp.maximum(de, 0.0)
    dx_positive = dx > 0.0
    de_dx = jnp.where(dx_positive, de_safe / jnp.maximum(dx, 1e-10), 0.0)

    xi = xi_fn(de_dx, field_kVcm, phi_drift, params)
    ln_arg = jnp.maximum(params.alpha + xi, 1.0)
    safe_xi = jnp.maximum(xi, 1e-10)
    R = jnp.where(xi > 1e-10, jnp.log(ln_arg) / safe_xi, 0.0)
    R = jnp.where(dx_positive, R, 0.0)

    W_ion_mev = params.w_value * 1e-6
    W_ph_mev = W_ion_mev / (1.0 + params.excitation_ratio)

    N_i = de_safe / W_ion_mev
    Q = N_i * R
    L = de_safe / W_ph_mev - Q

    # Invalid steps (dx <= 0) produce no quanta
    Q = jnp.where(dx_positive, Q, 0.0)
    L = jnp.where(dx_positive, L, 0.0)

    return Q, L
