"""
Detector geometry configuration for LArTPC simulation.

This module handles loading and parsing detector configuration from YAML files,
and provides per-volume geometry computation functions called by create_sim_config.
"""

import yaml
import os
import numpy as np
from typing import Dict, Tuple, Optional, Any, List


def calculate_max_diffusion_sigmas(
    max_drift_cm,
    drift_velocity_cm_us,
    transverse_diffusion_cm2_us,
    longitudinal_diffusion_cm2_us,
    wire_spacing_cm,
    time_spacing_us
):
    """
    Calculate maximum diffusion sigmas in both physical and unitless coordinates.

    Parameters
    ----------
    max_drift_cm : float
        Half-width of detector volume (max drift distance) in cm.
    drift_velocity_cm_us : float
        Drift velocity in cm/us.
    transverse_diffusion_cm2_us : float
        Transverse diffusion coefficient in cm^2/us.
    longitudinal_diffusion_cm2_us : float
        Longitudinal diffusion coefficient in cm^2/us.
    wire_spacing_cm : float
        Wire spacing in cm (for converting to unitless).
    time_spacing_us : float
        Time bin spacing in us (for converting to unitless).

    Returns
    -------
    max_sigma_trans_cm : float
    max_sigma_long_us : float
    max_sigma_trans_unitless : float
    max_sigma_long_unitless : float
    """
    max_drift_time_us = max_drift_cm / drift_velocity_cm_us
    max_sigma_trans_cm = np.sqrt(2.0 * transverse_diffusion_cm2_us * max_drift_time_us)
    D_long_temporal = longitudinal_diffusion_cm2_us / (drift_velocity_cm_us ** 2)
    max_sigma_long_us = np.sqrt(2.0 * D_long_temporal * max_drift_time_us)
    max_sigma_trans_unitless = max_sigma_trans_cm / wire_spacing_cm
    max_sigma_long_unitless = max_sigma_long_us / time_spacing_us
    return max_sigma_trans_cm, max_sigma_long_us, max_sigma_trans_unitless, max_sigma_long_unitless


def generate_detector(config_file_path: str) -> Dict[str, Any]:
    """
    Parse and validate a JAXTPC detector configuration YAML file.

    Returns the raw parsed config dict. Derived parameters are computed by
    ``create_sim_config`` and ``create_sim_params`` in ``config.py``.

    Parameters
    ----------
    config_file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML configuration dictionary.
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    with open(config_file_path, 'r') as file:
        detector_config = yaml.safe_load(file)

    # Validate top-level keys
    required_keys = ['volumes', 'readout', 'simulation', 'medium', 'electric_field']
    for key in required_keys:
        if key not in detector_config:
            raise KeyError(f"Missing required section '{key}' in configuration file: {config_file_path}")

    # Validate per-volume structure
    for i, vol in enumerate(detector_config['volumes']):
        if 'geometry' not in vol:
            raise KeyError(f"Volume {i}: missing 'geometry' section")
        geo = vol['geometry']
        if 'ranges' not in geo:
            raise KeyError(f"Volume {i}: missing 'geometry.ranges'")
        if 'drift_direction' not in geo:
            raise KeyError(f"Volume {i}: missing 'geometry.drift_direction'")
        # Wire volumes have 'planes', pixel volumes have 'readout.type: pixel'
        readout_cfg = vol.get('readout', {})
        readout_type = readout_cfg.get('type', 'wire')
        if readout_type == 'pixel':
            if 'pixel_pitch' not in readout_cfg:
                raise KeyError(f"Volume {i}: pixel readout missing 'readout.pixel_pitch'")
            if 'pixel_shape' not in readout_cfg:
                raise KeyError(f"Volume {i}: pixel readout missing 'readout.pixel_shape'")
        else:
            if 'planes' not in vol:
                raise KeyError(f"Volume {i}: missing 'planes' section")

    return detector_config


def get_drift_velocity(detector_config: Dict[str, Any]) -> float:
    """
    Extract drift velocity from config, converting mm/us to cm/us.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.

    Returns
    -------
    float
        Drift velocity in cm/us.
    """
    drift_velocity_mm_us = float(detector_config['simulation']['drift']['velocity'])
    drift_velocity_cm_us = drift_velocity_mm_us / 10.0
    if drift_velocity_cm_us <= 1e-9:
        raise ValueError("Drift velocity must be positive.")
    return drift_velocity_cm_us


def get_plane_geometry_for_volume(planes_cfg: list) -> Tuple[np.ndarray, int]:
    """
    Get plane distances and furthest plane index for one volume.

    Parameters
    ----------
    planes_cfg : list
        List of plane config dicts from one volume.

    Returns
    -------
    distances_cm : np.ndarray
        Array of shape (n_planes,) with distances from anode in cm.
    furthest_idx : int
        Index of the furthest plane.
    """
    distances = np.array([float(p['distance_from_anode']) for p in planes_cfg])
    return distances, int(np.argmax(distances))


def get_single_plane_wire_params(plane_config: dict,
                                 dims_cm: Dict[str, float]) -> Tuple[float, float, int, int, int]:
    """
    Extract wire parameters for a single plane.

    Parameters
    ----------
    plane_config : dict
        Single plane configuration dict with 'angle', 'wire_spacing', etc.
    dims_cm : dict
        Volume dimensions in cm with keys 'y', 'z'.

    Returns
    -------
    angle_rad : float
    wire_spacing_cm : float
    index_offset : int
    num_wires : int
    max_wire_idx_abs : int
    """
    detector_y, detector_z = dims_cm['y'], dims_cm['z']

    angle_deg = float(plane_config['angle'])
    angle_rad = np.radians(angle_deg)
    wire_spacing_cm = float(plane_config['wire_spacing'])

    if wire_spacing_cm <= 1e-9:
        raise ValueError("Wire spacing must be positive.")

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    half_y, half_z = detector_y / 2.0, detector_z / 2.0

    corners_centered = np.array([
        [-half_y, -half_z], [+half_y, -half_z], [-half_y, +half_z], [+half_y, +half_z]
    ], dtype=np.float64)

    r_values = corners_centered[:, 0] * sin_theta + corners_centered[:, 1] * cos_theta
    r_min = float(np.min(r_values))
    r_max = float(np.max(r_values))

    index_offset = 0
    if r_min < -1e-9:
        index_offset = int(np.floor(np.abs(r_min / wire_spacing_cm) + 1e-9)) + 1

    idx_min_rel = int(np.floor(r_min / wire_spacing_cm - 1e-9))
    idx_max_rel = int(np.ceil(r_max / wire_spacing_cm + 1e-9))
    abs_idx_min = idx_min_rel + index_offset
    abs_idx_max = idx_max_rel + index_offset

    assert abs_idx_min == 0, (
        f"Expected min_wire_idx_abs=0, got {abs_idx_min}. "
        f"Check float precision in wire index computation.")

    num_wires = abs_idx_max + 1
    max_wire_idx_abs = abs_idx_max

    return float(angle_rad), wire_spacing_cm, index_offset, num_wires, max_wire_idx_abs


def _calculate_wire_lengths_for_volume(dims_cm, angles_rad, wire_spacings_cm,
                                        index_offsets, num_wires_actual):
    """
    Calculate wire lengths for all planes in one volume.

    Parameters
    ----------
    dims_cm : dict
        Volume dimensions in cm with keys 'y', 'z'.
    angles_rad : list of float
        Wire angle per plane.
    wire_spacings_cm : list of float
        Wire spacing per plane.
    index_offsets : list of int
        Wire index offset per plane.
    num_wires_actual : list of int
        Number of wires per plane.

    Returns
    -------
    list of np.ndarray
        Wire lengths in meters, one array per plane.
    """
    detector_y = dims_cm['y']
    detector_z = dims_cm['z']
    half_y = detector_y / 2.0
    half_z = detector_z / 2.0

    wire_lengths = []
    for p in range(len(angles_rad)):
        angle_rad = float(angles_rad[p])
        num_wires = int(num_wires_actual[p])
        wire_spacing = float(wire_spacings_cm[p])
        offset = int(index_offsets[p])

        sin_theta = np.sin(angle_rad)

        if abs(sin_theta) < 1e-9:  # Y-plane (angle ~ 0)
            wire_lengths.append(np.full(num_wires, detector_y / 100.0))
        else:
            cos_theta = np.cos(angle_rad)
            wire_indices = np.arange(num_wires)
            relative_indices = wire_indices - offset
            r_values = relative_indices * wire_spacing

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
            wire_lengths.append(lengths_cm / 100.0)

    return wire_lengths


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

    for vol_cfg in detector_config['volumes']:
        vol_id = vol_cfg['id']
        desc = vol_cfg.get('description', '')
        geo = vol_cfg['geometry']
        print(f"\nVolume {vol_id}: {desc}")
        print(f"  Ranges: {geo['ranges']}")
        print(f"  Drift direction: {geo['drift_direction']}")
        for plane in vol_cfg['planes']:
            print(f"  Plane {plane['plane_id']} ({plane['type']}):")
            print(f"    Angle: {plane['angle']} degrees")
            print(f"    Wire spacing: {plane['wire_spacing']} cm")
            print(f"    Bias voltage: {plane['bias_voltage']} V")

    print(f"\nElectric field strength: {detector_config['electric_field']['field_strength']} V/cm")
    print(f"Medium: {detector_config['medium']['type']} at {detector_config['medium']['temperature']} K")

    if cfg is not None:
        print("\nDerived Parameters:")
        print("------------------")
        print(f"Number of time steps: {cfg.num_time_steps}")
        print(f"Time step size: {cfg.time_step_us:.6f} us")
        print(f"Number of volumes: {cfg.n_volumes}")
        for v in range(cfg.n_volumes):
            vol = cfg.volumes[v]
            print(f"\nVolume {v}: max_drift={vol.max_drift_cm:.1f} cm, "
                  f"x_anode={vol.x_anode_cm:.1f} cm, "
                  f"drift_dir={vol.drift_direction}")
            if vol.diffusion:
                d = vol.diffusion
                print(f"  Diffusion: K_wire={d.K_wire}, K_time={d.K_time}, "
                      f"sigma_trans={d.max_sigma_trans_unitless:.3f}, "
                      f"sigma_long={d.max_sigma_long_unitless:.3f}")
            for p in range(vol.n_planes):
                print(f"  Plane {cfg.plane_names[v][p]}: "
                      f"{vol.num_wires[p]} wires, spacing={vol.wire_spacings_cm[p]:.3f} cm")


if __name__ == "__main__":
    from tools.config import create_sim_config
    config_path = "config/cubic_wireplane_config.yaml"
    detector = generate_detector(config_path)
    cfg = create_sim_config(detector)
    print_detector_summary(detector, cfg)
