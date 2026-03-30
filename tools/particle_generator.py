"""
Muon track generator using PDG dE/dx tables.

Two modes:
1. Numpy (production): ``generate_muon_track()`` — sequential stepping,
   variable-length output. For generating test data.
2. JAX (differentiable): ``generate_muon_segments()`` — parallel CSDA
   range-based computation, fixed-length output. For optimization with
   ``jax.grad`` through the full simulation pipeline.

The JAX path uses the CSDA range table R(E) = integral(1/dE/dx) to compute
energy deposits in O(1) per segment instead of O(N) sequential scan. A
softplus relaxation at the stopping boundary ensures smooth gradients.
"""

import numpy as np
import jax
import jax.numpy as jnp
import os
from typing import Tuple, Optional, Dict, Any

# Path to PDG data file
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DEDX_FILE = os.path.join(_DATA_DIR, "muon_dedx_lar.csv")

# Liquid argon density (g/cm³)
LAR_DENSITY = 1.396

# Cache for loaded table
_DEDX_TABLE = None


def _load_dedx_table():
    """Load PDG dE/dx table (cached)."""
    global _DEDX_TABLE
    if _DEDX_TABLE is None:
        data = np.loadtxt(_DEDX_FILE, delimiter=",", comments="#")
        # Columns: T_MeV, p_MeV, dedx_MeVcm2g, csda_range_gcm2, beta
        T_MeV = data[:, 0]
        dedx_MeVcm2g = data[:, 2]
        # Convert to MeV/cm
        dedx_MeVcm = dedx_MeVcm2g * LAR_DENSITY
        _DEDX_TABLE = (T_MeV, dedx_MeVcm)
    return _DEDX_TABLE


def get_dedx(kinetic_energy_mev: float) -> float:
    """
    Get dE/dx for muon at given kinetic energy via interpolation.

    Parameters
    ----------
    kinetic_energy_mev : float
        Muon kinetic energy in MeV.

    Returns
    -------
    float
        dE/dx in MeV/cm.
    """
    T_table, dedx_table = _load_dedx_table()
    # Log interpolation for better accuracy over wide energy range
    return np.interp(
        np.log(kinetic_energy_mev),
        np.log(T_table),
        dedx_table
    )


def generate_muon_track(
    start_position_mm: Tuple[float, float, float],
    direction: Tuple[float, float, float],
    kinetic_energy_mev: float,
    step_size_mm: float = 0.1,
    track_id: int = 1,
    detector_bounds_mm: Optional[Tuple[Tuple[float, float], ...]] = None,
    min_energy_mev: float = 10.0,
) -> Dict[str, Any]:
    """
    Generate a muon track with energy-dependent dE/dx from PDG tables.

    Parameters
    ----------
    start_position_mm : tuple
        Starting position (x, y, z) in mm.
    direction : tuple
        Direction vector (dx, dy, dz). Normalized internally.
    kinetic_energy_mev : float
        Initial kinetic energy in MeV.
    step_size_mm : float
        Step size in mm (default 0.1 mm).
    track_id : int
        Track identifier (default 1).
    detector_bounds_mm : tuple of tuples, optional
        Detector boundaries as ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    min_energy_mev : float
        Stop tracking below this energy (default 10 MeV, PDG table limit).

    Returns
    -------
    dict
        Step data compatible with JAXTPC pipeline:
        - 'position': (N, 3) positions in mm
        - 'de': (N,) energy deposits in MeV
        - 'dx': (N,) step lengths in mm
        - 'theta', 'phi': (N,) track angles in rad
        - 'track_id': (N,) track IDs
        - 'x', 'y', 'z': (N,) individual coordinates
    """
    # Normalize direction
    dir_arr = np.array(direction, dtype=np.float64)
    dir_norm = np.linalg.norm(dir_arr)
    if dir_norm < 1e-10:
        raise ValueError("Direction vector cannot be zero")
    dir_unit = dir_arr / dir_norm

    # Standard spherical: theta from z-axis, phi = atan2(y, x)
    theta = np.arccos(np.clip(dir_unit[2], -1.0, 1.0))
    phi = np.arctan2(dir_unit[1], dir_unit[0])

    step_size_cm = step_size_mm / 10.0
    step_vector = dir_unit * step_size_mm

    # Storage
    positions = []
    de_values = []

    pos = np.array(start_position_mm, dtype=np.float64)
    energy = kinetic_energy_mev

    # Generate steps
    while energy > min_energy_mev:
        # Check bounds
        if detector_bounds_mm is not None:
            outside = any(
                pos[i] < lo or pos[i] > hi
                for i, (lo, hi) in enumerate(detector_bounds_mm)
            )
            if outside:
                break

        # Get dE/dx at current energy
        dedx = get_dedx(energy)
        de = dedx * step_size_cm

        # Don't deposit more than remaining energy
        if de > energy - min_energy_mev:
            de = energy - min_energy_mev

        positions.append(pos.copy())
        de_values.append(de)

        energy -= de
        pos = pos + step_vector

    n_steps = len(positions)

    if n_steps == 0:
        return {
            "position": np.zeros((0, 3), dtype=np.float32),
            "x": np.zeros(0, dtype=np.float32),
            "y": np.zeros(0, dtype=np.float32),
            "z": np.zeros(0, dtype=np.float32),
            "de": np.zeros(0, dtype=np.float32),
            "dx": np.zeros(0, dtype=np.float32),
            "theta": np.zeros(0, dtype=np.float32),
            "phi": np.zeros(0, dtype=np.float32),
            "track_id": np.zeros(0, dtype=np.int32),
        }

    positions_arr = np.array(positions, dtype=np.float32)
    de_arr = np.array(de_values, dtype=np.float32)

    return {
        "position": positions_arr,
        "x": positions_arr[:, 0],
        "y": positions_arr[:, 1],
        "z": positions_arr[:, 2],
        "de": de_arr,
        "dx": np.full(n_steps, step_size_mm, dtype=np.float32),
        "theta": np.full(n_steps, theta, dtype=np.float32),
        "phi": np.full(n_steps, phi, dtype=np.float32),
        "track_id": np.full(n_steps, track_id, dtype=np.int32),
    }


def generate_multiple_tracks(
    track_params: list,
    detector_bounds_mm: Optional[Tuple[Tuple[float, float], ...]] = None,
) -> Dict[str, Any]:
    """
    Generate multiple muon tracks and combine them.

    Parameters
    ----------
    track_params : list of dict
        Each dict contains: start_position_mm, direction, kinetic_energy_mev,
        and optionally: track_id, step_size_mm.
    detector_bounds_mm : tuple of tuples, optional
        Detector boundaries.

    Returns
    -------
    dict
        Combined step data for all tracks.
    """
    tracks = []
    for i, params in enumerate(track_params):
        track = generate_muon_track(
            start_position_mm=params["start_position_mm"],
            direction=params["direction"],
            kinetic_energy_mev=params["kinetic_energy_mev"],
            step_size_mm=params.get("step_size_mm", 0.1),
            track_id=params.get("track_id", i + 1),
            detector_bounds_mm=detector_bounds_mm,
        )
        if len(track["de"]) > 0:
            tracks.append(track)

    if not tracks:
        return {
            "position": np.zeros((0, 3), dtype=np.float32),
            "x": np.zeros(0, dtype=np.float32),
            "y": np.zeros(0, dtype=np.float32),
            "z": np.zeros(0, dtype=np.float32),
            "de": np.zeros(0, dtype=np.float32),
            "dx": np.zeros(0, dtype=np.float32),
            "theta": np.zeros(0, dtype=np.float32),
            "phi": np.zeros(0, dtype=np.float32),
            "track_id": np.zeros(0, dtype=np.int32),
        }

    return {
        key: np.concatenate([t[key] for t in tracks], axis=0)
        for key in tracks[0].keys()
    }


# =========================================================================
# JAX differentiable track generation
# =========================================================================

def load_dedx_table_jax():
    """Load PDG muon dE/dx table as JAX arrays.

    Loads from disk on first call, returns cached arrays on subsequent calls.

    Returns
    -------
    log_T_table : jnp.ndarray
        Natural log of kinetic energies (MeV).
    dedx_table : jnp.ndarray
        Stopping power in MeV/cm (already multiplied by LAr density).
    """
    T_table, dedx_table = _load_dedx_table()
    return jnp.array(np.log(T_table)), jnp.array(dedx_table)


def build_csda_range_table(log_T_table, dedx_table, n_points=2000):
    """Build CSDA range table by integrating 1/dE/dx.

    The range function R(E) = integral(1/dE/dx) maps energy to total
    remaining path length. The inverse R^{-1} gives energy at any
    distance along the track, enabling parallel segment generation.

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
    R_cm_table : np.ndarray
        CSDA range in cm (n_points,).
    T_MeV_table : np.ndarray
        Kinetic energies in MeV (n_points,).
    """
    log_T_np = np.asarray(log_T_table)
    dedx_np = np.asarray(dedx_table)

    log_T_dense = np.linspace(log_T_np[0], log_T_np[-1], n_points)
    dedx_dense = np.interp(log_T_dense, log_T_np, dedx_np)
    T_dense = np.exp(log_T_dense)

    inv_dedx = 1.0 / dedx_dense
    dT = np.diff(T_dense)
    avg_inv = 0.5 * (inv_dedx[:-1] + inv_dedx[1:])
    R_dense = np.concatenate([[0.0], np.cumsum(dT * avg_inv)])

    return R_dense.astype(np.float32), T_dense.astype(np.float32)


def diff_dedx(kinetic_energy_mev, log_T_table, dedx_table):
    """Differentiable dE/dx lookup via log-energy interpolation.

    Parameters
    ----------
    kinetic_energy_mev : scalar
        Muon kinetic energy in MeV (must be > 0).
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


def _softplus(x, beta):
    """Softplus with temperature: log(1 + exp(beta*x)) / beta."""
    return jnp.logaddexp(beta * x, 0.0) / beta


def _csda_energy_deposits(kinetic_energy_mev, step_size_cm, n_segments,
                          R_cm_table, T_MeV_table, relax_steps=2.0):
    """Compute energy deposits in parallel using CSDA range table.

    E(s) = R^{-1}(R(E_0) - s) for each segment independently.
    Softplus relaxation at the stopping boundary ensures smooth gradients.

    Parameters
    ----------
    kinetic_energy_mev : scalar
        Initial muon kinetic energy in MeV.
    step_size_cm : float
        Segment length in cm.
    n_segments : int
        Number of segments.
    R_cm_table : jnp.ndarray
        CSDA range table in cm.
    T_MeV_table : jnp.ndarray
        Corresponding kinetic energies in MeV.
    relax_steps : float
        Softplus relaxation width in units of step size.

    Returns
    -------
    de : jnp.ndarray, shape (n_segments,)
        Energy deposit per segment in MeV.
    """
    indices = jnp.arange(n_segments)
    log_T_csda = jnp.log(T_MeV_table)

    R_initial = jnp.interp(jnp.log(kinetic_energy_mev), log_T_csda, R_cm_table)

    R_at_start = R_initial - indices * step_size_cm
    R_at_end = R_initial - (indices + 1) * step_size_cm

    R_floor = R_cm_table[0]
    relax = step_size_cm * relax_steps
    R_start_soft = R_floor + jax.nn.softplus((R_at_start - R_floor) / relax) * relax
    R_end_soft = R_floor + jax.nn.softplus((R_at_end - R_floor) / relax) * relax

    E_start = jnp.interp(R_start_soft, R_cm_table, T_MeV_table)
    E_end = jnp.interp(R_end_soft, R_cm_table, T_MeV_table)

    return jnp.maximum(E_start - E_end, 0.0)


# Cache for consistent CSDA table
_CONSISTENT_CSDA = None

def _get_consistent_csda(log_T_table, dedx_table):
    """Lazily build and cache the consistent CSDA table."""
    global _CONSISTENT_CSDA
    if _CONSISTENT_CSDA is None:
        _CONSISTENT_CSDA = build_csda_range_table(log_T_table, dedx_table)
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
    Direction specified as spherical angles (theta, phi).

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
        Segment length in mm.
    n_segments : int
        Number of segments (static for JIT).
    log_T_table, dedx_table : jnp.ndarray
        PDG stopping-power table arrays from ``load_dedx_table_jax()``.
    relax_steps : float
        Softplus relaxation width in units of step size (default 2.0).

    Returns
    -------
    positions_mm : jnp.ndarray, shape (n_segments, 3)
        Segment centre positions in mm.
    de : jnp.ndarray, shape (n_segments,)
        Energy deposit per segment in MeV.
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

    R_cm, T_MeV = _get_consistent_csda(log_T_table, dedx_table)
    de = _csda_energy_deposits(
        kinetic_energy_mev, step_size_mm / 10.0, n_segments,
        R_cm, T_MeV, relax_steps,
    )
    return positions, de


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
    Direction specified as trig components for stable optimization
    (avoids angle wrapping issues).

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
        Segment length in mm.
    n_segments : int
        Number of segments (static for JIT).
    log_T_table, dedx_table : jnp.ndarray
        PDG stopping-power table arrays from ``load_dedx_table_jax()``.
    relax_steps : float
        Softplus relaxation width in units of step size (default 2.0).

    Returns
    -------
    positions_mm : jnp.ndarray, shape (n_segments, 3)
        Segment centre positions in mm.
    de : jnp.ndarray, shape (n_segments,)
        Energy deposit per segment in MeV.
    """
    dir_unnorm = jnp.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta,
    ])
    dir_vec = dir_unnorm / jnp.linalg.norm(dir_unnorm)
    step_vector = dir_vec * step_size_mm

    indices = jnp.arange(n_segments)
    positions = start_position_mm[None, :] + indices[:, None] * step_vector[None, :]

    R_cm, T_MeV = _get_consistent_csda(log_T_table, dedx_table)
    de = _csda_energy_deposits(
        kinetic_energy_mev, step_size_mm / 10.0, n_segments,
        R_cm, T_MeV, relax_steps,
    )
    return positions, de


def mask_outside_volume(positions_mm, de, half_extents_mm):
    """Zero out dE for segments outside the detector volume.

    Parameters
    ----------
    positions_mm : jnp.ndarray, shape (N, 3)
        Segment positions in mm.
    de : jnp.ndarray, shape (N,)
        Energy deposits per segment.
    half_extents_mm : tuple of 3 floats
        Detector half-extent per axis (x, y, z) in mm.

    Returns
    -------
    de_masked : jnp.ndarray, shape (N,)
        dE with out-of-volume segments zeroed.
    """
    hx, hy, hz = half_extents_mm
    in_volume = (
        (jnp.abs(positions_mm[:, 0]) < hx) &
        (jnp.abs(positions_mm[:, 1]) < hy) &
        (jnp.abs(positions_mm[:, 2]) < hz)
    )
    return jnp.where(in_volume, de, 0.0)
