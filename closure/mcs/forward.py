"""
MCS forward model using cumulative sums instead of jax.lax.scan.

Exploits the small-angle nature of MCS (~1-5 mrad/step) to replace the
sequential scan with parallel prefix sums (cumsums):

  d_k = normalize(d0 + Phi1_k * e1 + Phi2_k * e2)

where Phi1_k = cumsum(dtheta1)[k-1] (exclusive).  This is exact for zero
scattering and an excellent approximation for Highland-scale angles.

The model is end-to-end differentiable w.r.t. all parameters including
the per-segment scattering angles dtheta1, dtheta2.

Migrated from closure_analysis_MCS/mcs_cumsum_forward.py.
"""

import jax
import jax.numpy as jnp

from closure.mcs.mcs_physics import (
    _perpendicular_basis,
    highland_theta0,
    _ke_to_beta_p,
    MUON_MASS_MEV,
    X0_LAR_CM,
)
from tools.particle_generator import (
    load_dedx_table_jax,
    _get_consistent_csda,
    _csda_energy_deposits,
    mask_outside_volume,
)


# ---------------------------------------------------------------------------
# Cumsum utilities
# ---------------------------------------------------------------------------

def exclusive_cumsum(x, axis=0):
    """Exclusive cumulative sum: concat([zeros, cumsum(x)[:-1]]).

    Result[0] = 0, Result[k] = sum(x[0:k]) for k >= 1.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    axis : int
        Axis along which to compute.

    Returns
    -------
    jnp.ndarray
        Same shape as x, exclusive cumsum along axis.
    """
    cs = jnp.cumsum(x, axis=axis)
    # Prepend zeros and drop the last element
    zero_shape = list(x.shape)
    zero_shape[axis] = 1
    zeros = jnp.zeros(zero_shape, dtype=x.dtype)
    return jnp.concatenate([zeros, jnp.take(cs, jnp.arange(x.shape[axis] - 1), axis=axis)], axis=axis)


# ---------------------------------------------------------------------------
# Core 3-cumsum position model
# ---------------------------------------------------------------------------

def mcs_cumsum_positions(d0, e1, e2, start_mm, dtheta1, dtheta2, step_size_mm):
    """Compute segment positions from scattering angles via 3 cumsums.

    Model:
      1. Phi1 = exclusive_cumsum(dtheta1), Phi2 = exclusive_cumsum(dtheta2)
      2. dirs = normalize(d0 + Phi1*e1 + Phi2*e2)   (element-wise)
      3. positions = start + step_size * exclusive_cumsum(dirs)

    Parameters
    ----------
    d0 : (3,) array — initial unit direction
    e1, e2 : (3,) arrays — perpendicular basis vectors
    start_mm : (3,) array — starting position in mm
    dtheta1, dtheta2 : (N,) arrays — per-segment scattering angles (rad)
    step_size_mm : float — segment length in mm

    Returns
    -------
    positions : (N, 3) — segment positions in mm
    dirs : (N, 3) — unit direction at each segment
    """
    n = dtheta1.shape[0]

    # Cumulative scattering angles (exclusive: first segment has zero deflection)
    Phi1 = exclusive_cumsum(dtheta1)  # (N,)
    Phi2 = exclusive_cumsum(dtheta2)  # (N,)

    # Direction at each segment: d0 + Phi1*e1 + Phi2*e2, then normalize
    dirs_unnorm = d0[None, :] + Phi1[:, None] * e1[None, :] + Phi2[:, None] * e2[None, :]
    dirs = dirs_unnorm / jnp.linalg.norm(dirs_unnorm, axis=1, keepdims=True)

    # Positions via exclusive cumsum of direction vectors
    cum_dirs = exclusive_cumsum(dirs, axis=0)  # (N, 3)
    positions = start_mm[None, :] + step_size_mm * cum_dirs

    return positions, dirs


# ---------------------------------------------------------------------------
# Full forward model
# ---------------------------------------------------------------------------

def mcs_cumsum_forward(energy, start, sin_th, cos_th, sin_ph, cos_ph,
                       dtheta1, dtheta2, step_size_mm, n_segments,
                       log_T, dedx, relax_steps=2.0):
    """Full MCS forward model: positions + CSDA energy deposits.

    Parameters
    ----------
    energy : scalar — initial kinetic energy in MeV
    start : (3,) — starting position in mm
    sin_th, cos_th, sin_ph, cos_ph : scalars — trig direction params
    dtheta1, dtheta2 : (N,) — per-segment scattering angles (rad)
    step_size_mm : float — segment length in mm
    n_segments : int — number of segments (static)
    log_T, dedx : jnp.ndarray — PDG dE/dx table arrays
    relax_steps : float — softplus relaxation for CSDA

    Returns
    -------
    positions : (N, 3) — segment positions in mm
    de : (N,) — energy deposits per segment in MeV
    """
    # Direction from trig components
    dir_unnorm = jnp.array([sin_th * cos_ph, sin_th * sin_ph, cos_th])
    d0 = dir_unnorm / jnp.linalg.norm(dir_unnorm)

    # Perpendicular basis
    e1, e2 = _perpendicular_basis(d0)

    # Positions via cumsum model
    positions, dirs = mcs_cumsum_positions(
        d0, e1, e2, start, dtheta1, dtheta2, step_size_mm
    )

    # CSDA energy deposits (independent of scattering — path length preserved)
    R_cm, T_MeV = _get_consistent_csda(log_T, dedx)
    de = _csda_energy_deposits(
        energy, step_size_mm / 10.0, n_segments,
        R_cm, T_MeV, relax_steps,
    )

    return positions, de


# ---------------------------------------------------------------------------
# Truth generation: sample Highland angles, feed into cumsum forward
# ---------------------------------------------------------------------------

def generate_mcs_truth(energy, start, theta, phi, step_size_mm, n_segments,
                       log_T, dedx, rng_key, highland_constant=13.6,
                       relax_steps=2.0):
    """Generate truth MCS track by sampling Highland angles.

    1. Compute CSDA midpoint energies for each segment.
    2. Sample dtheta1, dtheta2 ~ N(0, theta_0(E_mid)) per segment.
    3. Feed into cumsum forward model.

    Parameters
    ----------
    energy : scalar — initial kinetic energy in MeV
    start : (3,) — starting position in mm
    theta : scalar — polar angle (radians)
    phi : scalar — azimuthal angle (radians)
    step_size_mm : float — segment length in mm
    n_segments : int — number of segments
    log_T, dedx : jnp.ndarray — PDG dE/dx table arrays
    rng_key : jax.random.PRNGKey
    highland_constant : float — Highland energy constant (MeV)
    relax_steps : float — softplus relaxation for CSDA

    Returns
    -------
    positions : (N, 3) — segment positions in mm
    de : (N,) — energy deposits per segment in MeV
    dtheta1_truth : (N,) — sampled scattering angles plane 1
    dtheta2_truth : (N,) — sampled scattering angles plane 2
    """
    step_size_cm = step_size_mm / 10.0

    # CSDA energy deposits
    R_cm, T_MeV = _get_consistent_csda(log_T, dedx)
    log_T_csda = jnp.log(T_MeV)
    R_initial = jnp.interp(jnp.log(energy), log_T_csda, R_cm)

    indices = jnp.arange(n_segments)

    # Midpoint ranges for Highland formula
    R_at_mid = R_initial - (indices + 0.5) * step_size_cm
    R_floor = R_cm[0]
    relax = step_size_cm * relax_steps
    R_mid_soft = R_floor + jax.nn.softplus((R_at_mid - R_floor) / relax) * relax
    E_mid = jnp.interp(R_mid_soft, R_cm, T_MeV)

    # Highland theta_0 at each segment
    _, _, bp_mid = _ke_to_beta_p(E_mid)
    theta0 = highland_theta0(bp_mid, step_size_cm, highland_constant)

    # Sample scattering angles
    key1, key2 = jax.random.split(rng_key)
    dtheta1 = theta0 * jax.random.normal(key1, shape=(n_segments,))
    dtheta2 = theta0 * jax.random.normal(key2, shape=(n_segments,))

    # Build positions via cumsum forward
    sin_th = jnp.sin(theta)
    cos_th = jnp.cos(theta)
    sin_ph = jnp.sin(phi)
    cos_ph = jnp.cos(phi)

    positions, de = mcs_cumsum_forward(
        energy, start, sin_th, cos_th, sin_ph, cos_ph,
        dtheta1, dtheta2, step_size_mm, n_segments,
        log_T, dedx, relax_steps,
    )

    return positions, de, dtheta1, dtheta2


# ---------------------------------------------------------------------------
# Highland prior (regularizer for optimization)
# ---------------------------------------------------------------------------

def highland_prior(dtheta1, dtheta2, energy, step_size_cm, n_segments,
                   log_T, dedx, relax_steps=2.0, highland_constant=13.6):
    """Highland scattering prior: -log p(dtheta | E).

    Computes sum_k [ dtheta1[k]^2 / (2*theta_0(E_k)^2) + dtheta2[k]^2 / (2*theta_0(E_k)^2) ]
    where E_k is the CSDA midpoint energy at segment k.

    Parameters
    ----------
    dtheta1, dtheta2 : (N,) — scattering angles
    energy : scalar — initial kinetic energy in MeV
    step_size_cm : float — segment length in cm
    n_segments : int — number of segments
    log_T, dedx : jnp.ndarray — PDG dE/dx table arrays
    relax_steps : float — softplus relaxation for CSDA
    highland_constant : float — Highland constant (MeV)

    Returns
    -------
    scalar — negative log prior (lower is more probable)
    """
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

    # Variance = theta0^2 per segment
    inv_var = 1.0 / jnp.maximum(theta0 ** 2, 1e-20)

    return 0.5 * jnp.sum((dtheta1 ** 2 + dtheta2 ** 2) * inv_var)


# ---------------------------------------------------------------------------
# Build forward function for wire simulation
# ---------------------------------------------------------------------------

def build_mcs_forward(simulator, n_segments, step_size_mm):
    """Build a forward function that maps (positions_mm, de) to wire signals.

    Uses simulator.forward_segments with default physics parameters.

    Parameters
    ----------
    simulator : DetectorSimulator
        Must have differentiable=True, n_segments=n_segments.
    n_segments : int
    step_size_mm : float

    Returns
    -------
    forward : callable
        forward(positions_mm, de) -> tuple of response arrays
    """
    sim_params = simulator.default_sim_params

    def forward(positions_mm, de):
        return simulator.forward_segments(sim_params, positions_mm, de, dx=step_size_mm)

    return forward
