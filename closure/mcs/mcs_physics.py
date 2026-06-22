"""Multiple-Coulomb-scattering (MCS) muon physics primitives.

Self-contained reconstruction of the small physics helpers that the MCS
closure (``closure/mcs/forward.py`` and the ``validate_*`` scripts) depends
on.  These used to live in an external ``MCS_muon.mcs_muon_generator`` module
that was never migrated into this repository; everything here is standard
muon kinematics + the PDG Highland formula, with JAX-friendly (smooth,
branchless) implementations so the forward model stays differentiable.

All quantities are in MeV / cm / radians unless suffixed otherwise.

Physics references
------------------
* Muon mass, LAr radiation length: PDG.
* Highland scattering width:
    theta0 = (13.6 MeV / beta c p) * z * sqrt(x / X0) * [1 + 0.038 ln(x / X0)]
  (PDG eq.; z = 1 for a muon).  The same ``highland_theta0`` is used both to
  *sample* the truth scattering and as the optimisation *prior*, so the
  closure is self-consistent regardless of the exact constant.
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MUON_MASS_MEV = 105.6583745          # PDG muon mass (MeV/c^2)

# Liquid-argon radiation length: X0 = 19.55 g/cm^2, rho = 1.396 g/cm^3
#   -> X0 = 19.55 / 1.396 = 14.0 cm.
X0_LAR_CM = 14.0


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def _ke_to_beta_p(kinetic_energy_mev):
    """Convert muon kinetic energy to (beta, p, beta*p).

    Parameters
    ----------
    kinetic_energy_mev : scalar or array
        Muon kinetic energy T in MeV.

    Returns
    -------
    beta : same shape — v/c.
    p : same shape — momentum in MeV/c.
    beta_p : same shape — beta * p in MeV (the combination entering Highland).

    Notes
    -----
    E = T + m,  p = sqrt(E^2 - m^2) = sqrt(T^2 + 2 T m),
    beta = p / E,  beta_p = p^2 / E.  A small floor keeps the muon from going
    exactly to rest (beta_p -> 0 would blow up Highland) so gradients stay
    finite at the track end.
    """
    T = jnp.maximum(kinetic_energy_mev, 1e-3)
    E = T + MUON_MASS_MEV
    p = jnp.sqrt(jnp.maximum(E * E - MUON_MASS_MEV * MUON_MASS_MEV, 1e-12))
    beta = p / E
    beta_p = p * beta
    return beta, p, beta_p


# ---------------------------------------------------------------------------
# Highland scattering width
# ---------------------------------------------------------------------------

def highland_theta0(beta_p, step_size_cm, highland_constant=13.6,
                    x0_cm=X0_LAR_CM):
    """Highland RMS scattering angle for one step.

    Parameters
    ----------
    beta_p : scalar or array — beta * p in MeV (from :func:`_ke_to_beta_p`).
    step_size_cm : float — step length x in cm.
    highland_constant : float — the 13.6 MeV Highland constant.
    x0_cm : float — radiation length in cm (default liquid argon).

    Returns
    -------
    theta0 : same shape as ``beta_p`` — RMS plane-projected scattering angle
        (radians) for a single step.
    """
    x_over_X0 = jnp.maximum(step_size_cm / x0_cm, 1e-12)
    log_corr = 1.0 + 0.038 * jnp.log(x_over_X0)
    # The log correction can in principle go negative for absurdly thin steps;
    # clamp so theta0 stays positive and the prior stays well-defined.
    log_corr = jnp.maximum(log_corr, 1e-3)
    return (highland_constant / jnp.maximum(beta_p, 1e-6)) * \
        jnp.sqrt(x_over_X0) * log_corr


# ---------------------------------------------------------------------------
# Orthonormal perpendicular basis (branchless, differentiable)
# ---------------------------------------------------------------------------

def _perpendicular_basis(d):
    """Two orthonormal vectors spanning the plane perpendicular to ``d``.

    Uses the branchless construction of Duff et al. (2017), "Building an
    Orthonormal Basis, Revisited" — smooth in ``d`` everywhere except the
    pole ``d = (0, 0, -1)``, which the MCS tracks never sit exactly on.

    Parameters
    ----------
    d : (3,) array — unit direction.

    Returns
    -------
    e1, e2 : (3,) arrays — orthonormal, each perpendicular to ``d``.
    """
    sign = jnp.where(d[2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + d[2])
    b = d[0] * d[1] * a
    e1 = jnp.array([1.0 + sign * d[0] * d[0] * a, sign * b, -sign * d[0]])
    e2 = jnp.array([b, sign + d[1] * d[1] * a, -d[1]])
    return e1, e2


# ---------------------------------------------------------------------------
# Reference (scan-based) truth generator
# ---------------------------------------------------------------------------

def generate_mcs_muon_segments(energy, start, theta, phi, step_size_mm,
                               n_segments, log_T, dedx, rng_key,
                               highland_constant=13.6, relax_steps=2.0):
    """Exact sequential-scan MCS muon track (the reference the cumsum model approximates).

    At each step the direction is deflected in the frame perpendicular to the
    *current* direction (recomputed every step), as opposed to the cumsum
    forward which deflects in the fixed initial frame.  For Highland-scale
    angles the two agree to high precision; this generator exists as the
    ground-truth cross-check.

    Parameters mirror :func:`closure.mcs.forward.generate_mcs_truth`.

    Returns
    -------
    positions : (N, 3) — segment positions in mm.
    de : (N,) — CSDA energy deposit per segment in MeV.
    dtheta1, dtheta2 : (N,) — sampled per-segment scattering angles (rad).
    """
    from tools.particle_generator import (
        _get_consistent_csda, _csda_energy_deposits,
    )

    step_size_cm = step_size_mm / 10.0

    # --- per-segment Highland width from CSDA midpoint energies ---
    R_cm, T_MeV = _get_consistent_csda(log_T, dedx)
    log_T_csda = jnp.log(T_MeV)
    R_initial = jnp.interp(jnp.log(energy), log_T_csda, R_cm)

    indices = jnp.arange(n_segments)
    R_at_mid = R_initial - (indices + 0.5) * step_size_cm
    R_floor = R_cm[0]
    relax = step_size_cm * relax_steps
    R_mid_soft = R_floor + jax.nn.softplus((R_at_mid - R_floor) / relax) * relax
    E_mid = jnp.interp(R_mid_soft, R_cm, T_MeV)

    _, _, bp_mid = _ke_to_beta_p(E_mid)
    theta0 = highland_theta0(bp_mid, step_size_cm, highland_constant)

    key1, key2 = jax.random.split(rng_key)
    dtheta1 = theta0 * jax.random.normal(key1, shape=(n_segments,))
    dtheta2 = theta0 * jax.random.normal(key2, shape=(n_segments,))

    # --- direction from (theta, phi) ---
    sin_th = jnp.sin(theta)
    d0 = jnp.array([sin_th * jnp.cos(phi), sin_th * jnp.sin(phi), jnp.cos(theta)])
    d0 = d0 / jnp.linalg.norm(d0)

    step_mm = step_size_mm

    def body(carry, inp):
        d, pos = carry
        a1, a2 = inp
        e1, e2 = _perpendicular_basis(d)
        d_new = d + a1 * e1 + a2 * e2
        d_new = d_new / jnp.linalg.norm(d_new)
        pos_out = pos                      # position at the *start* of this step
        pos_next = pos + step_mm * d       # advance along pre-deflection direction
        return (d_new, pos_next), pos_out

    (_, _), positions = jax.lax.scan(
        body, (d0, jnp.asarray(start, jnp.float32)),
        (dtheta1, dtheta2),
    )

    # --- CSDA energy deposits (decoupled from scattering) ---
    de = _csda_energy_deposits(energy, step_size_cm, n_segments,
                               R_cm, T_MeV, relax_steps)

    return positions, de, dtheta1, dtheta2
