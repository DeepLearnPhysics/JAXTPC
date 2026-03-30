"""
Detector geometry configuration for LArTPC simulation.

This module handles loading and parsing detector configuration from YAML files,
and pre-calculates all geometry parameters needed for simulation including
wire positions, drift parameters, and diffusion coefficients.
"""

import yaml
import os
import numpy as np
from typing import Dict, Tuple, Optional, Any


def calculate_max_diffusion_sigmas(
    detector_half_width_cm,
    drift_velocity_cm_us,
    transverse_diffusion_cm2_us,
    longitudinal_diffusion_cm2_us,
    wire_spacing_cm,
    time_spacing_us
):
    """
    Calculate maximum diffusion sigmas for the detector in both physical and unitless coordinates.

    Parameters
    ----------
    detector_half_width_cm : float
        Half-width of detector (max drift distance) in cm
    drift_velocity_cm_us : float
        Drift velocity in cm/μs
    transverse_diffusion_cm2_us : float
        Transverse diffusion coefficient in cm²/μs
    longitudinal_diffusion_cm2_us : float
        Longitudinal diffusion coefficient in cm²/μs
    wire_spacing_cm : float
        Wire spacing in cm (for converting to unitless)
    time_spacing_us : float
        Time bin spacing in μs (for converting to unitless)

    Returns
    -------
    max_sigma_trans_cm : float
        Maximum transverse sigma in cm
    max_sigma_long_us : float
        Maximum longitudinal sigma in μs
    max_sigma_trans_unitless : float
        Maximum transverse sigma in unitless grid coordinates (wires)
    max_sigma_long_unitless : float
        Maximum longitudinal sigma in unitless grid coordinates (time bins)
    """
    # Maximum drift time
    max_drift_time_us = detector_half_width_cm / drift_velocity_cm_us

    # Transverse sigma (spatial - in cm)
    max_sigma_trans_cm = np.sqrt(2.0 * transverse_diffusion_cm2_us * max_drift_time_us)

    # Longitudinal sigma (temporal - in μs)
    D_long_temporal = longitudinal_diffusion_cm2_us / (drift_velocity_cm_us ** 2)
    max_sigma_long_us = np.sqrt(2.0 * D_long_temporal * max_drift_time_us)

    # Convert to unitless coordinates by dividing by grid spacing
    max_sigma_trans_unitless = max_sigma_trans_cm / wire_spacing_cm
    max_sigma_long_unitless = max_sigma_long_us / time_spacing_us

    return max_sigma_trans_cm, max_sigma_long_us, max_sigma_trans_unitless, max_sigma_long_unitless


def generate_detector(config_file_path: str) -> Dict[str, Any]:
    """
    Parse and validate a JAXTPC detector configuration YAML file.

    Returns the raw parsed config dict. Derived parameters (geometry, diffusion,
    wire layout) are computed by ``create_sim_config`` and ``create_sim_params``
    in ``config.py`` when building typed simulation structures.

    Parameters
    ----------
    config_file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    KeyError
        If a required section is missing from the configuration.
    yaml.YAMLError
        If the YAML file cannot be parsed.
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    with open(config_file_path, 'r') as file:
        detector_config = yaml.safe_load(file)

    required_keys = ['detector', 'wire_planes', 'readout', 'simulation', 'medium', 'electric_field']
    for key in required_keys:
        if key not in detector_config:
            raise KeyError(f"Missing required section '{key}' in configuration file: {config_file_path}")

    return detector_config


def _calculate_wire_lengths(dims_cm, angles_rad, wire_spacings_cm, index_offsets,
                            num_wires_actual):
    """
    Calculate wire lengths for all planes using vectorized numpy operations.

    Parameters
    ----------
    dims_cm : dict
        Detector dimensions in cm with keys 'x', 'y', 'z'.
    angles_rad : array-like, shape (2, 3)
        Wire angles in radians for each (side, plane).
    wire_spacings_cm : array-like, shape (2, 3)
        Wire spacing in cm for each (side, plane).
    index_offsets : array-like, shape (2, 3)
        Wire index offsets for each (side, plane).
    num_wires_actual : array-like, shape (2, 3)
        Number of wires for each (side, plane).

    Returns
    -------
    wire_lengths_m : dict
        Dictionary mapping (side_idx, plane_idx) -> np.ndarray of wire lengths in meters.
    """
    detector_y = dims_cm['y']
    detector_z = dims_cm['z']
    half_y = detector_y / 2.0
    half_z = detector_z / 2.0

    wire_lengths_m = {}

    for side_idx in range(2):
        for plane_idx in range(3):
            angle_rad = float(angles_rad[side_idx, plane_idx])
            num_wires = int(num_wires_actual[side_idx, plane_idx])
            wire_spacing = float(wire_spacings_cm[side_idx, plane_idx])
            offset = int(index_offsets[side_idx, plane_idx])

            sin_theta = np.sin(angle_rad)

            if abs(sin_theta) < 1e-9:  # Y-plane (angle ~ 0): all wires span full Y
                wire_lengths_m[(side_idx, plane_idx)] = np.full(num_wires, detector_y / 100.0)
            else:
                # Angled plane (U/V) - vectorized over all wires
                cos_theta = np.cos(angle_rad)

                wire_indices = np.arange(num_wires)
                relative_indices = wire_indices - offset
                r_values = relative_indices * wire_spacing

                # Parameterize wire as: y(t) = r*sin(θ) + t*cos(θ),
                #                       z(t) = r*cos(θ) - t*sin(θ)
                # Find t at each boundary intersection
                t_y1 = (-half_y - r_values * sin_theta) / cos_theta
                t_y2 = (+half_y - r_values * sin_theta) / cos_theta
                t_y_min = np.minimum(t_y1, t_y2)
                t_y_max = np.maximum(t_y1, t_y2)

                t_z1 = (r_values * cos_theta + half_z) / sin_theta
                t_z2 = (r_values * cos_theta - half_z) / sin_theta
                t_z_min = np.minimum(t_z1, t_z2)
                t_z_max = np.maximum(t_z1, t_z2)

                t_min = np.maximum(t_y_min, t_z_min)
                t_max = np.minimum(t_y_max, t_z_max)

                lengths_cm = np.maximum(0.0, t_max - t_min)
                wire_lengths_m[(side_idx, plane_idx)] = lengths_cm / 100.0

    return wire_lengths_m




def get_detector_dimensions(detector_config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract detector dimensions from config and convert them to floats.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.

    Returns
    -------
    dict
        Dictionary of detector dimensions in cm.
    """
    dims_cm = detector_config['detector']['dimensions']
    return {k: float(v) for k, v in dims_cm.items()}


def get_drift_params(detector_config: Dict[str, Any],
                     dims_cm: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    """
    Extract global drift parameters, converting velocity to cm/us.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.
    dims_cm : dict, optional
        Pre-calculated dimensions to avoid redundant calculation.
        If None, dimensions are calculated.

    Returns
    -------
    tuple
        (detector_half_width_x, drift_velocity) in cm and cm/us respectively.
    """
    if dims_cm is None:
        dims_cm = get_detector_dimensions(detector_config)

    detector_half_width_x = dims_cm['x'] / 2.0
    drift_velocity_mm_us = float(detector_config['simulation']['drift']['velocity'])
    drift_velocity_cm_us = drift_velocity_mm_us / 10.0  # Convert mm/us to cm/us

    if drift_velocity_cm_us <= 1e-9:
        raise ValueError("Drift velocity must be positive.")

    return detector_half_width_x, drift_velocity_cm_us


def get_plane_geometry(detector_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get distances for all planes from anode and identify furthest plane per side.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.

    Returns
    -------
    all_plane_distances_cm : np.ndarray
        Array of shape (2, 3) with distances from anode in cm.
    furthest_plane_indices : np.ndarray
        Array of shape (2,) with indices of furthest planes.
    """
    # Get plane distances
    distances_cm = np.zeros((2, 3), dtype=float)

    for side_idx in range(2):
        for plane_idx in range(3):
            plane_config = detector_config['wire_planes']['sides'][side_idx]['planes'][plane_idx]
            distances_cm[side_idx, plane_idx] = float(plane_config['distance_from_anode'])

    # Get furthest plane indices
    furthest_plane_indices = np.zeros(2, dtype=int)
    for side_idx in range(2):
        furthest_plane_indices[side_idx] = np.argmax(distances_cm[side_idx])

    return distances_cm, furthest_plane_indices


def get_single_plane_wire_params(detector_config: Dict[str, Any],
                                 side_idx: int,
                                 plane_idx: int,
                                 dims_cm: Optional[Dict[str, float]] = None) -> Tuple[float, float, int, int, int]:
    """
    Extract wire parameters for a single specified plane.

    Handles angles, spacing (cm), calculates offset and wire count/range.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.
    side_idx : int
        Index of the detector side (0 or 1).
    plane_idx : int
        Index of the plane (0, 1, or 2).
    dims_cm : dict, optional
        Pre-calculated dimensions to avoid redundant calculation.
        If None, dimensions are calculated.

    Returns
    -------
    angle_rad : float
        Wire angle in radians.
    wire_spacing_cm : float
        Spacing between wires in cm.
    index_offset : int
        Wire index offset.
    num_wires : int
        Number of wires in the plane.
    max_wire_idx_abs : int
        Maximum absolute wire index.
    """
    if dims_cm is None:
        dims_cm = get_detector_dimensions(detector_config)

    detector_y, detector_z = dims_cm['y'], dims_cm['z']
    plane_config = detector_config['wire_planes']['sides'][side_idx]['planes'][plane_idx]

    angle_deg = float(plane_config['angle'])
    angle_rad = np.radians(angle_deg)
    wire_spacing_cm = float(plane_config['wire_spacing'])

    if wire_spacing_cm <= 1e-9:
        raise ValueError("Wire spacing must be positive.")

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    half_y, half_z = detector_y / 2.0, detector_z / 2.0

    # Calculate corners of the detector in the YZ plane
    corners_centered = np.array([
        [-half_y, -half_z], [+half_y, -half_z], [-half_y, +half_z], [+half_y, +half_z]
    ], dtype=np.float64)

    # Project corners onto the wire direction
    r_values = corners_centered[:, 0] * sin_theta + corners_centered[:, 1] * cos_theta
    r_min = float(np.min(r_values))
    r_max = float(np.max(r_values))

    # Calculate offset
    index_offset = 0
    if r_min < -1e-9:
        index_offset = int(np.floor(np.abs(r_min / wire_spacing_cm) + 1e-9)) + 1

    # Calculate relative and absolute indices
    idx_min_rel = int(np.floor(r_min / wire_spacing_cm - 1e-9))
    idx_max_rel = int(np.ceil(r_max / wire_spacing_cm + 1e-9))
    abs_idx_min = idx_min_rel + index_offset
    abs_idx_max = idx_max_rel + index_offset

    # The offset formula guarantees abs_idx_min = 0 for any detector geometry
    # (algebraic identity: floor(-W-ε) + floor(W+ε) + 1 = 0).
    # This assertion catches float precision regressions (e.g., float32 swallows ε).
    assert abs_idx_min == 0, (
        f"Expected min_wire_idx_abs=0, got {abs_idx_min} for side={side_idx} "
        f"plane={plane_idx}. Check float precision in wire index computation.")

    # Calculate number of wires
    num_wires = abs_idx_max + 1  # since abs_idx_min is always 0
    max_wire_idx_abs = abs_idx_max

    return float(angle_rad), wire_spacing_cm, index_offset, num_wires, max_wire_idx_abs


def calculate_time_params(detector_config: Dict[str, Any],
                          dims_cm: Optional[Dict[str, float]] = None,
                          drift_velocity_cm_us: Optional[float] = None) -> Tuple[int, float, float]:
    """
    Calculate time-related parameters from the detector config.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.
    dims_cm : dict, optional
        Pre-calculated dimensions to avoid redundant calculation.
        If None, dimensions are calculated.
    drift_velocity_cm_us : float, optional
        Pre-calculated drift velocity to avoid redundant calculation.
        If None, velocity is calculated.

    Returns
    -------
    num_time_steps : int
        Number of time steps for simulation.
    time_step_size_us : float
        Size of time step in μs.
    max_drift_time_us : float
        Maximum drift time in μs.
    """
    if dims_cm is None:
        dims_cm = get_detector_dimensions(detector_config)

    if drift_velocity_cm_us is None:
        _, drift_velocity_cm_us = get_drift_params(detector_config, dims_cm)

    max_drift_dist_cm = dims_cm['x'] / 2.0

    if drift_velocity_cm_us <= 1e-9:
        max_drift_time_us = 0.0
    else:
        max_drift_time_us = max_drift_dist_cm / drift_velocity_cm_us

    sampling_rate_mhz = float(detector_config['readout']['sampling_rate'])

    if sampling_rate_mhz <= 1e-9:
        raise ValueError("Sampling rate must be positive.")

    time_step_size_us = 1.0 / sampling_rate_mhz  # us
    num_time_steps = int(np.ceil(max_drift_time_us / time_step_size_us)) + 1
    num_time_steps = max(1, num_time_steps)

    return num_time_steps, time_step_size_us, max_drift_time_us


def pre_calculate_all_wire_params(detector_config: Dict[str, Any],
                                  dims_cm: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, ...]:
    """
    Pre-calculate wire parameters for all planes and sides.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.
    dims_cm : dict, optional
        Pre-calculated dimensions to avoid redundant calculation.
        If None, dimensions are calculated.

    Returns
    -------
    angles_rad : np.ndarray
        Wire angles in radians, shape (2, 3).
    wire_spacings_cm : np.ndarray
        Spacing between wires in cm, shape (2, 3).
    index_offsets : np.ndarray
        Wire index offsets, shape (2, 3).
    num_wires_all : np.ndarray
        Number of wires for each plane, shape (2, 3).
    max_wire_indices_abs_all : np.ndarray
        Maximum absolute wire indices, shape (2, 3).
    """
    if dims_cm is None:
        dims_cm = get_detector_dimensions(detector_config)

    # Initialize arrays
    num_wires_all = np.zeros((2, 3), dtype=int)
    max_wire_indices_abs_all = np.zeros((2, 3), dtype=int)
    index_offsets_all = np.zeros((2, 3), dtype=int)
    wire_spacings_all = np.zeros((2, 3), dtype=float)
    angles_all = np.zeros((2, 3), dtype=float)

    for side_idx in range(2):
        for plane_idx in range(3):
            angle, spacing, offset, n_wires, max_idx_abs = \
                get_single_plane_wire_params(detector_config, side_idx, plane_idx, dims_cm)

            num_wires_all[side_idx, plane_idx] = n_wires
            max_wire_indices_abs_all[side_idx, plane_idx] = max_idx_abs
            index_offsets_all[side_idx, plane_idx] = offset
            wire_spacings_all[side_idx, plane_idx] = spacing
            angles_all[side_idx, plane_idx] = angle

    return (angles_all.astype(np.float32),
            wire_spacings_all.astype(np.float32),
            index_offsets_all.astype(np.int32),
            num_wires_all.astype(np.int32),
            max_wire_indices_abs_all.astype(np.int32))


def print_detector_summary(detector_config, cfg=None):
    """
    Print a summary of the detector configuration.

    Parameters
    ----------
    detector_config : dict
        Raw parsed YAML configuration.
    cfg : SimConfig, optional
        If provided, prints derived simulation parameters.
    """
    print("Detector Configuration Summary")
    print("==============================")

    # Basic detector properties
    print(f"Detector name: {detector_config['detector']['name']}")
    dims = detector_config['detector']['dimensions']
    print(f"Dimensions: {dims['x']} x {dims['y']} x {dims['z']} cm^3")

    # Wire planes information
    print("\nWire Plane Configuration:")
    print("------------------------")
    for side in detector_config['wire_planes']['sides']:
        side_id = side['side_id']
        print(f"Side {side_id}: {side['description']}")
        for plane in side['planes']:
            print(f"  Plane {plane['plane_id']} ({plane['type']}):")
            print(f"    Angle: {plane['angle']} degrees")
            print(f"    Wire spacing: {plane['wire_spacing']} cm")
            print(f"    Bias voltage: {plane['bias_voltage']} V")

    # Medium properties
    print(f"\nElectric field strength: {detector_config['electric_field']['field_strength']} V/cm")
    print(f"Medium: {detector_config['medium']['type']} at {detector_config['medium']['temperature']} K")

    # Derived parameters (from SimConfig if available)
    if cfg is not None:
        print("\nDerived Parameters:")
        print("------------------")
        print(f"Number of time steps: {cfg.num_time_steps}")
        print(f"Time step size: {cfg.time_step_us:.6f} us")
        print(f"Half-width: {cfg.side_geom[0].half_width_cm:.1f} cm")
        if cfg.diffusion:
            d = cfg.diffusion
            print(f"Drift velocity: {d.velocity_cm_us:.4f} cm/us")
            print(f"Diffusion (long): {d.long_cm2_us:.8f} cm^2/us")
            print(f"Diffusion (trans): {d.trans_cm2_us:.8f} cm^2/us")
            print(f"Max sigma trans: {d.max_sigma_trans_unitless:.3f} (unitless)")
            print(f"Max sigma long: {d.max_sigma_long_unitless:.3f} (unitless)")
            print(f"Kernel half-widths: K_wire={d.K_wire}, K_time={d.K_time}")
        for s in range(len(cfg.side_geom)):
            sg = cfg.side_geom[s]
            print(f"\nSide {s}: {len(sg.num_wires)} planes")
            for p in range(len(sg.num_wires)):
                print(f"  Plane {cfg.plane_names[s][p]}: "
                      f"{sg.num_wires[p]} wires, spacing={sg.wire_spacings_cm[p]:.3f} cm")


if __name__ == "__main__":
    from tools.config import create_sim_config
    config_path = "config/cubic_wireplane_config.yaml"
    detector = generate_detector(config_path)
    cfg = create_sim_config(detector)
    print_detector_summary(detector, cfg)