"""
Configuration classes for JAXTPC detector simulation.

This module defines all NamedTuple parameter bundles used throughout
the simulation code for clean parameter passing and type hints.
"""

from typing import NamedTuple
import numpy as np
import jax.numpy as jnp


class DepositData(NamedTuple):
    """Padded input data from particle simulation steps."""
    positions_mm: jnp.ndarray    # (N_pad, 3) - hit positions
    charges: jnp.ndarray         # (N_pad,) - recombined charge
    valid_mask: jnp.ndarray      # (N_pad,) - True for real hits
    theta: jnp.ndarray           # (N_pad,) - polar angle of step direction
    phi: jnp.ndarray             # (N_pad,) - azimuthal angle of step direction
    track_ids: jnp.ndarray       # (N_pad,) - particle track ID


class DriftParams(NamedTuple):
    """Drift physics parameters (detector-wide constants)."""
    detector_half_width_cm: float     # Maximum drift distance
    velocity_cm_us: float             # Drift velocity
    lifetime_us: float                # Electron lifetime (in μs)
    diffusion_long_cm2_us: float      # Longitudinal diffusion coefficient
    diffusion_trans_cm2_us: float     # Transverse diffusion coefficient


class TimeParams(NamedTuple):
    """Time axis discretization."""
    num_steps: int         # Number of time bins
    step_size_us: float    # Duration per bin (μs)


class PlaneGeometry(NamedTuple):
    """Geometry for one wire plane."""
    angle_rad: float                  # Wire angle in YZ plane
    wire_spacing_cm: float            # Wire pitch
    distance_from_anode_cm: float     # Plane position relative to anode
    index_offset: int                 # Offset for absolute wire indexing
    min_wire_idx: int                 # Minimum absolute wire index
    max_wire_idx: int                 # Maximum absolute wire index
    num_wires: int                    # Total wire count
    plane_type: str                   # 'U', 'V', or 'Y'


class DiffusionParams(NamedTuple):
    """Diffusion parameters for signal generation."""
    # Source values (from physics)
    max_sigma_trans_unitless: float   # Max transverse σ (wire spacing units)
    max_sigma_long_unitless: float    # Max longitudinal σ (time bin units)

    # Derived grid sizes (computed from sigmas)
    K_wire: int                       # Half-width in wire direction
    K_time: int                       # Half-width in time direction

    # Kernel interpolation
    num_s: int                        # Number of diffusion levels for kernel


class TrackHitsConfig(NamedTuple):
    """Configuration for track hit labeling."""
    threshold: float      # Minimum charge to keep
    max_tracks: int       # Max tracks for array pre-allocation
    max_keys: int         # Max unique (track, wire, time) combinations


def create_diffusion_params(
    max_sigma_trans_unitless: float,
    max_sigma_long_unitless: float,
    num_s: int = 16,
    n_sigma: float = 3.0
) -> DiffusionParams:
    """
    Create DiffusionParams with K values computed from sigmas.

    Parameters
    ----------
    max_sigma_trans_unitless : float
        Maximum transverse diffusion sigma in unitless grid coordinates (wires).
    max_sigma_long_unitless : float
        Maximum longitudinal diffusion sigma in unitless grid coordinates (time bins).
    num_s : int, optional
        Number of diffusion levels for kernel interpolation, by default 16.
    n_sigma : float, optional
        Number of sigma to cover (determines K values), by default 3.0.

    Returns
    -------
    DiffusionParams
        Configured diffusion parameters with derived K values.
    """
    K_wire = max(1, int(np.ceil(n_sigma * max_sigma_trans_unitless)))
    K_time = max(1, int(np.ceil(n_sigma * max_sigma_long_unitless)))

    return DiffusionParams(
        max_sigma_trans_unitless=max_sigma_trans_unitless,
        max_sigma_long_unitless=max_sigma_long_unitless,
        K_wire=K_wire,
        K_time=K_time,
        num_s=num_s,
    )


def create_drift_params(detector_config: dict) -> DriftParams:
    """
    Create DriftParams from detector configuration dictionary.

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().

    Returns
    -------
    DriftParams
        Configured drift parameters.
    """
    # Convert electron lifetime from ms to μs
    lifetime_us = detector_config['electron_lifetime_ms'] * 1000.0

    return DriftParams(
        detector_half_width_cm=detector_config['detector_half_width_x'],
        velocity_cm_us=detector_config['drift_velocity_cm_us'],
        lifetime_us=lifetime_us,
        diffusion_long_cm2_us=detector_config['longitudinal_diffusion_cm2_us'],
        diffusion_trans_cm2_us=detector_config['transverse_diffusion_cm2_us'],
    )


def create_time_params(detector_config: dict) -> TimeParams:
    """
    Create TimeParams from detector configuration dictionary.

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().

    Returns
    -------
    TimeParams
        Configured time parameters.
    """
    return TimeParams(
        num_steps=detector_config['num_time_steps'],
        step_size_us=detector_config['time_step_size_us'],
    )


def create_plane_geometry(
    detector_config: dict,
    side_idx: int,
    plane_idx: int,
    plane_type: str
) -> PlaneGeometry:
    """
    Create PlaneGeometry for a specific wire plane.

    Parameters
    ----------
    detector_config : dict
        Detector configuration from generate_detector().
    side_idx : int
        Side index (0 or 1).
    plane_idx : int
        Plane index (0, 1, or 2).
    plane_type : str
        Plane type ('U', 'V', or 'Y').

    Returns
    -------
    PlaneGeometry
        Configured plane geometry.
    """
    return PlaneGeometry(
        angle_rad=float(detector_config['angles_rad'][side_idx, plane_idx]),
        wire_spacing_cm=float(detector_config['wire_spacings_cm'][side_idx, plane_idx]),
        distance_from_anode_cm=float(detector_config['all_plane_distances_cm'][side_idx, plane_idx]),
        index_offset=int(detector_config['index_offsets'][side_idx, plane_idx]),
        min_wire_idx=int(detector_config['min_wire_indices_abs'][side_idx, plane_idx]),
        max_wire_idx=int(detector_config['max_wire_indices_abs'][side_idx, plane_idx]),
        num_wires=int(detector_config['num_wires_actual'][side_idx, plane_idx]),
        plane_type=plane_type,
    )


def create_track_hits_config(
    threshold: float = 1.0,
    max_tracks: int = 10000,
    max_keys: int = 1000000
) -> TrackHitsConfig:
    """
    Create TrackHitsConfig with specified parameters.

    Parameters
    ----------
    threshold : float, optional
        Minimum charge threshold for keeping hits, by default 1.0.
    max_tracks : int, optional
        Maximum number of tracks for array pre-allocation, by default 10000.
    max_keys : int, optional
        Maximum number of unique (track, wire, time) combinations, by default 1000000.

    Returns
    -------
    TrackHitsConfig
        Configured track hits parameters.
    """
    return TrackHitsConfig(
        threshold=threshold,
        max_tracks=max_tracks,
        max_keys=max_keys,
    )
