"""
Differentiable muon track generator using JAX and PDG dE/dx tables.

Generates straight-line muon tracks with energy-dependent stopping power.
All operations use JAX primitives so the full pipeline
position/direction/energy -> segments -> wire signals is end-to-end
differentiable.

Design choices
--------------
- Energy evolution: CSDA range table for fully-parallel O(1) computation.
  The range function R(E) = integral(1/dE/dx) is pre-integrated from the
  PDG table, enabling E(s) = R^{-1}(R(E_0) - s) for each segment
  independently.  ~400x faster than the old jax.lax.scan approach.
- dE/dx lookup: log-energy interpolation of PDG table via jnp.interp
  (differentiable w.r.t. energy through the interpolation weights).
- Direction: spherical angles (theta, phi) -> unit vector, or trig
  components (sin_theta, cos_theta, sin_phi, cos_phi) for optimization.
- The old scan-based versions are available as ``generate_muon_segments_scan``
  and ``generate_muon_segments_trig_scan`` for testing/comparison.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os

# Liquid argon density (g/cm^3)
LAR_DENSITY = 1.396

# Path to PDG dE/dx data
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "data",
)
_DEDX_FILE = os.path.join(_DATA_DIR, "muon_dedx_lar.csv")

# Module-level caches
_DEDX_TABLE_JAX = None
_CSDA_TABLE_JAX = None
_CONSISTENT_CSDA = None


# ---------------------------------------------------------------------------
# PDG table loading
# ---------------------------------------------------------------------------

def load_dedx_table_jax():
    """Load PDG muon dE/dx table as JAX arrays (cached after first call).

    Returns
    -------
    log_T_table : jnp.ndarray
        Natural log of kinetic energies (MeV).
    dedx_table : jnp.ndarray
        Stopping power in MeV/cm (already multiplied by LAr density).
    """
    global _DEDX_TABLE_JAX
    if _DEDX_TABLE_JAX is None:
        data = np.loadtxt(_DEDX_FILE, delimiter=",", comments="#")
        T_MeV = data[:, 0]
        dedx_MeVcm = data[:, 2] * LAR_DENSITY
        _DEDX_TABLE_JAX = (
            jnp.array(np.log(T_MeV)),
            jnp.array(dedx_MeVcm),
        )
    return _DEDX_TABLE_JAX


def build_consistent_csda_table(log_T_table, dedx_table, n_points=2000):
    """Build CSDA range table by integrating 1/dE/dx from the same table
    the scan uses, ensuring perfect consistency between the two methods.

    Parameters
    ----------
    log_T_table : jnp.ndarray
        Log of kinetic energies from ``load_dedx_table_jax()``.
    dedx_table : jnp.ndarray
        dE/dx values in MeV/cm from ``load_dedx_table_jax()``.
    n_points : int
        Number of points in the dense integration grid.

    Returns
    -------
    R_cm_table : jnp.ndarray
        CSDA range in cm (n_points,).
    T_MeV_table : jnp.ndarray
        Kinetic energies in MeV (n_points,).
    """
    log_T_np = np.asarray(log_T_table)
    dedx_np = np.asarray(dedx_table)

    # Dense grid in log-E space (matches jnp.interp's linear interpolation)
    log_T_dense = np.linspace(log_T_np[0], log_T_np[-1], n_points)
    dedx_dense = np.interp(log_T_dense, log_T_np, dedx_np)
    T_dense = np.exp(log_T_dense)

    # Trapezoidal integration of 1/(dE/dx) over dT
    inv_dedx = 1.0 / dedx_dense
    dT = np.diff(T_dense)
    avg_inv = 0.5 * (inv_dedx[:-1] + inv_dedx[1:])
    R_dense = np.concatenate([[0.0], np.cumsum(dT * avg_inv)])

    return np.asarray(R_dense, dtype=np.float32), np.asarray(T_dense, dtype=np.float32)


def load_csda_range_table_jax(n_dense=0):
    """Load CSDA range table as JAX arrays (cached after first call).

    The range is read directly from the PDG table (column 4: g/cm^2)
    and converted to cm by dividing by LAr density.

    Parameters
    ----------
    n_dense : int
        If > 0, densify the table to this many points uniformly spaced
        in log-energy for improved interpolation accuracy.  0 uses the
        raw 82-point PDG table.

    Returns
    -------
    R_cm_table : jnp.ndarray
        CSDA range in cm (monotonically increasing with energy).
    T_MeV_table : jnp.ndarray
        Corresponding kinetic energies in MeV.
    """
    global _CSDA_TABLE_JAX
    if _CSDA_TABLE_JAX is None:
        data = np.loadtxt(_DEDX_FILE, delimiter=",", comments="#")
        T_MeV = data[:, 0]
        R_gcm2 = data[:, 3]
        R_cm = R_gcm2 / LAR_DENSITY

        if n_dense > 0:
            log_T_raw = np.log(T_MeV)
            log_T_dense = np.linspace(log_T_raw[0], log_T_raw[-1], n_dense)
            R_cm = np.interp(log_T_dense, log_T_raw, R_cm)
            T_MeV = np.exp(log_T_dense)

        _CSDA_TABLE_JAX = (jnp.array(R_cm), jnp.array(T_MeV))
    return _CSDA_TABLE_JAX


# ---------------------------------------------------------------------------
# Differentiable dE/dx
# ---------------------------------------------------------------------------

def diff_dedx(kinetic_energy_mev, log_T_table, dedx_table):
    """Differentiable dE/dx lookup via log-energy interpolation.

    Caller must ensure kinetic_energy_mev > 0 (use safe_energy clamp).

    Parameters
    ----------
    kinetic_energy_mev : scalar
        Muon kinetic energy in MeV.
    log_T_table : jnp.ndarray
        Log of table energies.
    dedx_table : jnp.ndarray
        dE/dx values in MeV/cm.

    Returns
    -------
    scalar
        dE/dx in MeV/cm.
    """
    return jnp.interp(jnp.log(kinetic_energy_mev), log_T_table, dedx_table)


# ---------------------------------------------------------------------------
# Differentiable muon segment generation
# ---------------------------------------------------------------------------

def _softplus(x, beta):
    """Softplus with temperature: log(1 + exp(beta*x)) / beta.

    Uses logaddexp for numerical stability (no overflow for large beta*x).
    """
    return jnp.logaddexp(beta * x, 0.0) / beta


def _make_scan_fn(step_vector, step_size_cm, log_T_table, dedx_table,
                  min_energy_mev, smooth_temperature):
    """Build the scan body for energy evolution.

    Factored out so both generate_muon_segments and
    generate_muon_segments_trig share the same implementation.
    """
    if smooth_temperature > 0:
        beta = 1.0 / smooth_temperature

        def scan_fn(carry, _):
            pos, energy = carry
            safe_energy = _softplus(energy - min_energy_mev, beta) + min_energy_mev
            dedx = diff_dedx(safe_energy, log_T_table, dedx_table)
            de_raw = dedx * step_size_cm
            remaining = _softplus(energy - min_energy_mev, beta)
            de = de_raw - _softplus(de_raw - remaining, beta)
            gate = jax.nn.sigmoid((energy - min_energy_mev) / smooth_temperature)
            de = de * gate
            return (pos + step_vector, energy - de), (pos, de)
    else:
        def scan_fn(carry, _):
            pos, energy = carry
            safe_energy = jnp.maximum(energy, min_energy_mev)
            dedx = diff_dedx(safe_energy, log_T_table, dedx_table)
            de_raw = dedx * step_size_cm
            remaining = jnp.maximum(energy - min_energy_mev, 0.0)
            de = jnp.minimum(de_raw, remaining)
            de = jnp.where(energy > min_energy_mev, de, 0.0)
            return (pos + step_vector, energy - de), (pos, de)

    return scan_fn


def generate_muon_segments_scan(
    kinetic_energy_mev,
    start_position_mm,
    theta,
    phi,
    step_size_mm,
    n_segments,
    log_T_table,
    dedx_table,
    min_energy_mev=10.0,
    smooth_temperature=0.0,
):
    """Generate muon segments using sequential ``jax.lax.scan`` (slow).

    Kept for testing/comparison.  Production code should use
    ``generate_muon_segments`` which calls the parallel CSDA path.
    """
    sin_theta = jnp.sin(theta)
    dir_vec = jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        jnp.cos(theta),
    ])
    step_vector = dir_vec * step_size_mm

    scan_fn = _make_scan_fn(step_vector, step_size_mm / 10.0,
                            log_T_table, dedx_table,
                            min_energy_mev, smooth_temperature)

    _, (positions, des) = jax.lax.scan(
        scan_fn, (start_position_mm, kinetic_energy_mev),
        None, length=n_segments,
    )
    return positions, des


def generate_muon_segments_trig_scan(
    kinetic_energy_mev,
    start_position_mm,
    sin_theta, cos_theta, sin_phi, cos_phi,
    step_size_mm,
    n_segments,
    log_T_table,
    dedx_table,
    min_energy_mev=10.0,
    smooth_temperature=0.0,
):
    """Generate muon segments using sequential scan with trig params (slow).

    Kept for testing/comparison.  Production code should use
    ``generate_muon_segments_trig`` which calls the parallel CSDA path.
    """
    dir_unnorm = jnp.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta,
    ])
    dir_vec = dir_unnorm / jnp.linalg.norm(dir_unnorm)
    step_vector = dir_vec * step_size_mm

    scan_fn = _make_scan_fn(step_vector, step_size_mm / 10.0,
                            log_T_table, dedx_table,
                            min_energy_mev, smooth_temperature)

    _, (positions, des) = jax.lax.scan(
        scan_fn, (start_position_mm, kinetic_energy_mev),
        None, length=n_segments,
    )
    return positions, des


# ---------------------------------------------------------------------------
# Volume masking
# ---------------------------------------------------------------------------

def mask_outside_volume(positions, de, half_extents_mm=(2160.0, 2160.0, 2160.0)):
    """Zero out dE for segments whose positions fall outside the detector volume.

    Parameters
    ----------
    positions : (N, 3) array — segment positions in mm.
    de : (N,) array — energy deposits per segment.
    half_extents_mm : tuple of 3 floats
        Half-extent of the detector volume per axis in mm.
        Default 2160.0 = 432 cm / 2 * 10 mm/cm.

    Returns
    -------
    de_masked : (N,) array — dE with out-of-volume segments zeroed.
    """
    hx, hy, hz = half_extents_mm
    in_volume = (
        (jnp.abs(positions[:, 0]) < hx) &
        (jnp.abs(positions[:, 1]) < hy) &
        (jnp.abs(positions[:, 2]) < hz)
    )
    return jnp.where(in_volume, de, 0.0)


# ---------------------------------------------------------------------------
# Forward function with correct dx
# ---------------------------------------------------------------------------

def build_muon_forward(simulator, n_segments, step_size_mm):
    """Build a forward function for muon tracks with correct step size.

    Delegates to ``simulator.build_forward(dx_mm=step_size_mm)`` and
    wraps it to accept ``(positions_mm, de)`` instead of SegmentData.

    Parameters
    ----------
    simulator : DetectorSimulator
        Must have been created with ``differentiable=True, n_segments=n_segments``.
    n_segments : int
        Segment count (must match ``simulator.n_segments``).
    step_size_mm : float
        Muon step size in mm.

    Returns
    -------
    forward : callable
        ``forward(positions_mm, de) -> tuple`` of 6 response arrays
        (east_U, east_V, east_Y, west_U, west_V, west_Y).
    """
    from tools.config import SegmentData
    inner = simulator.build_forward(dx_mm=step_size_mm)

    def forward(positions_mm, de):
        return inner(SegmentData(positions_mm=positions_mm, de=de))

    return forward


# ---------------------------------------------------------------------------
# CSDA range-based parallel segment generation
# ---------------------------------------------------------------------------

def _csda_energy_deposits(kinetic_energy_mev, step_size_cm, n_segments,
                          R_cm_table, T_MeV_table, relax_steps=2.0):
    """Compute energy deposits in parallel using CSDA range table.

    The CSDA range R(E) = integral of 1/(dE/dx) maps energy to total
    remaining path length.  The inverse R^{-1} gives energy at any
    distance along the track, turning the sequential recurrence into
    independent lookups:  E(s) = R^{-1}(R(E_0) - s).

    A soft clamp (softplus) replaces ``jnp.maximum(R, R_floor)`` at the
    stopping boundary.  This shifts the last few segments' dE by
    O(relax * ln2) — negligible for physics — but makes the gradient
    smooth at the Bragg peak.  JAX auto-diff handles the backward pass;
    no custom_jvp is needed because softplus + jnp.interp compose into
    a smooth enough gradient for the full simulation pipeline.
    """
    indices = jnp.arange(n_segments)
    log_T_csda = jnp.log(T_MeV_table)

    # Forward lookup: E_0 -> R(E_0)
    R_initial = jnp.interp(jnp.log(kinetic_energy_mev), log_T_csda, R_cm_table)

    # Remaining range at start and end of each segment
    R_at_start = R_initial - indices * step_size_cm
    R_at_end = R_initial - (indices + 1) * step_size_cm

    # Soft clamp: smoothed version of max(R, R_floor).
    # softplus(x/relax)*relax ≈ max(x, 0) but with smooth derivative (sigmoid).
    # relax controls the transition width at the stopping boundary.
    R_floor = R_cm_table[0]
    relax = step_size_cm * relax_steps
    R_start_soft = R_floor + jax.nn.softplus((R_at_start - R_floor) / relax) * relax
    R_end_soft = R_floor + jax.nn.softplus((R_at_end - R_floor) / relax) * relax

    E_start = jnp.interp(R_start_soft, R_cm_table, T_MeV_table)
    E_end = jnp.interp(R_end_soft, R_cm_table, T_MeV_table)

    return jnp.maximum(E_start - E_end, 0.0)


def generate_muon_segments_csda(
    kinetic_energy_mev,
    start_position_mm,
    theta,
    phi,
    step_size_mm,
    n_segments,
    log_T_table,
    dedx_table,
    R_cm_table,
    T_MeV_table,
    relax_steps=2.0,
):
    """Generate muon segments using CSDA range table (fully parallel).

    Same physics as ``generate_muon_segments`` but replaces the O(N)
    sequential ``jax.lax.scan`` with O(1) parallel table lookups.
    """
    sin_theta = jnp.sin(theta)
    dir_vec = jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        jnp.cos(theta),
    ])
    step_vector = dir_vec * step_size_mm

    indices = jnp.arange(n_segments)
    positions = start_position_mm[None, :] + indices[:, None] * step_vector[None, :]

    de = _csda_energy_deposits(
        kinetic_energy_mev, step_size_mm / 10.0, n_segments,
        R_cm_table, T_MeV_table, relax_steps,
    )
    return positions, de


def generate_muon_segments_trig_csda(
    kinetic_energy_mev,
    start_position_mm,
    sin_theta, cos_theta, sin_phi, cos_phi,
    step_size_mm,
    n_segments,
    log_T_table,
    dedx_table,
    R_cm_table,
    T_MeV_table,
    relax_steps=2.0,
):
    """Generate muon segments using CSDA range with trig parameterization."""
    dir_unnorm = jnp.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta,
    ])
    dir_vec = dir_unnorm / jnp.linalg.norm(dir_unnorm)
    step_vector = dir_vec * step_size_mm

    indices = jnp.arange(n_segments)
    positions = start_position_mm[None, :] + indices[:, None] * step_vector[None, :]

    de = _csda_energy_deposits(
        kinetic_energy_mev, step_size_mm / 10.0, n_segments,
        R_cm_table, T_MeV_table, relax_steps,
    )
    return positions, de


# ---------------------------------------------------------------------------
# Public API — CSDA-backed wrappers (drop-in replacements for scan versions)
# ---------------------------------------------------------------------------

def _get_consistent_csda(log_T_table, dedx_table):
    """Lazily build and cache the consistent CSDA table."""
    global _CONSISTENT_CSDA
    if _CONSISTENT_CSDA is None:
        _CONSISTENT_CSDA = build_consistent_csda_table(log_T_table, dedx_table)
    return _CONSISTENT_CSDA


def generate_muon_segments(
    kinetic_energy_mev,
    start_position_mm,
    theta,
    phi,
    step_size_mm,
    n_segments,
    log_T_table,
    dedx_table,
    relax_steps=2.0,
):
    """Generate a straight-line muon track as differentiable JAX arrays.

    Uses the CSDA range table for fully-parallel O(1) computation.

    Parameters
    ----------
    kinetic_energy_mev : scalar
        Initial muon kinetic energy in MeV.
    start_position_mm : (3,) array
        Starting (x, y, z) in mm.
    theta : scalar
        Polar angle from z-axis (radians).
    phi : scalar
        Azimuthal angle in xy-plane (radians).
    step_size_mm : float
        Segment length in mm (static, not traced by JAX).
    n_segments : int
        Maximum number of segments (static for JIT).
    log_T_table, dedx_table : jnp.ndarray
        PDG stopping-power table arrays from ``load_dedx_table_jax()``.
    relax_steps : float
        Softplus relaxation width in units of step size (default 2.0).

    Returns
    -------
    positions_mm : (n_segments, 3)
        Segment centre positions in mm.
    de : (n_segments,)
        Energy deposit per segment in MeV (0 for exhausted segments).
    """
    R_cm, T_MeV = _get_consistent_csda(log_T_table, dedx_table)
    return generate_muon_segments_csda(
        kinetic_energy_mev, start_position_mm, theta, phi,
        step_size_mm, n_segments, log_T_table, dedx_table,
        R_cm, T_MeV, relax_steps,
    )


def generate_muon_segments_trig(
    kinetic_energy_mev,
    start_position_mm,
    sin_theta, cos_theta, sin_phi, cos_phi,
    step_size_mm,
    n_segments,
    log_T_table,
    dedx_table,
    relax_steps=2.0,
):
    """Generate muon segments using trig parameterization (sin/cos).

    Uses the CSDA range table for fully-parallel O(1) computation.

    Parameters
    ----------
    kinetic_energy_mev : scalar
        Initial muon kinetic energy in MeV.
    start_position_mm : (3,) array
        Starting (x, y, z) in mm.
    sin_theta, cos_theta : scalar
        Polar angle trig components.
    sin_phi, cos_phi : scalar
        Azimuthal angle trig components.
    step_size_mm : float
        Segment length in mm (static, not traced by JAX).
    n_segments : int
        Maximum number of segments (static for JIT).
    log_T_table, dedx_table : jnp.ndarray
        PDG stopping-power table arrays from ``load_dedx_table_jax()``.
    relax_steps : float
        Softplus relaxation width in units of step size (default 2.0).

    Returns
    -------
    positions_mm : (n_segments, 3)
        Segment centre positions in mm.
    de : (n_segments,)
        Energy deposit per segment in MeV (0 for exhausted segments).
    """
    R_cm, T_MeV = _get_consistent_csda(log_T_table, dedx_table)
    return generate_muon_segments_trig_csda(
        kinetic_energy_mev, start_position_mm,
        sin_theta, cos_theta, sin_phi, cos_phi,
        step_size_mm, n_segments, log_T_table, dedx_table,
        R_cm, T_MeV, relax_steps,
    )
