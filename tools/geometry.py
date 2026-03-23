"""
Detector geometry configuration for LArTPC simulation.

This module handles loading and parsing detector configuration from YAML files,
and pre-calculates all geometry parameters needed for simulation including
wire positions, drift parameters, and diffusion coefficients.
"""

import yaml
import os
import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Any, List, Union


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


def generate_detector(config_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Read a JAXTPC detector configuration YAML file and return a detector dictionary.

    Parameters
    ----------
    config_file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict or None
        A dictionary containing all detector properties and derived parameters,
        or None if loading fails.
    """
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at {config_file_path}")
        return None

    try:
        with open(config_file_path, 'r') as file:
            detector_config = yaml.safe_load(file)

        # Basic validation to ensure the config has the expected structure
        required_keys = ['detector', 'wire_planes', 'readout', 'simulation', 'medium', 'electric_field']
        for key in required_keys:
            if key not in detector_config:
                print(f"Error: Missing required section '{key}' in configuration file")
                return None

        # Pre-calculate all parameters needed for simulation
        params = _precalculate_all_parameters(detector_config)
        detector_config.update(params)

        return detector_config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    except Exception as e:
        print(f"Error loading detector configuration: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_wire_lengths(dims_cm, angles_rad, wire_spacings_cm, index_offsets,
                            num_wires_actual, min_wire_indices_abs):
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
    min_wire_indices_abs : array-like, shape (2, 3)
        Minimum absolute wire index for each (side, plane).

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
            min_wire_idx = int(min_wire_indices_abs[side_idx, plane_idx])

            angle_deg = np.degrees(angle_rad)

            if abs(angle_deg) < 0.1:  # Y-plane (angle ~ 0)
                wire_lengths_m[(side_idx, plane_idx)] = np.full(num_wires, detector_y / 100.0)
            else:
                # Angled plane (U/V) - vectorized over all wires
                cos_theta = np.cos(angle_rad)
                sin_theta = np.sin(angle_rad)

                wire_indices = np.arange(min_wire_idx, min_wire_idx + num_wires)
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


def _calculate_noise_rms(wire_lengths_m, noise_config_path):
    """
    Calculate mean noise RMS in ADC for each plane using MicroBooNE noise model.

    Uses Equation 3.6: RMS = sqrt(x^2 + (y + z*L)^2)

    Parameters
    ----------
    wire_lengths_m : dict
        Dictionary mapping (side_idx, plane_idx) -> np.ndarray of wire lengths in meters.
    noise_config_path : str
        Path to noise_spectrum.npz containing noise_param_x/y/z.

    Returns
    -------
    noise_rms : np.ndarray, shape (2, 3)
        Mean noise RMS in ADC for each (side, plane).
    """
    noise_cfg = np.load(noise_config_path, allow_pickle=True)
    x = float(noise_cfg['noise_param_x'])
    y = float(noise_cfg['noise_param_y'])
    z = float(noise_cfg['noise_param_z'])

    noise_rms = np.zeros((2, 3))
    for side_idx in range(2):
        for plane_idx in range(3):
            lengths = wire_lengths_m[(side_idx, plane_idx)]
            rms_per_wire = np.sqrt(x**2 + (y + z * lengths)**2)
            noise_rms[side_idx, plane_idx] = np.mean(rms_per_wire)

    return noise_rms


def _precalculate_all_parameters(detector_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all derived parameters needed for simulation from the detector config.

    Parameters
    ----------
    detector_config : dict
        Original detector configuration dictionary.

    Returns
    -------
    dict
        Dictionary of all derived parameters.
    """
    params = {}

    # Get basic dimensions
    dims_cm = get_detector_dimensions(detector_config)
    params['dims_cm'] = dims_cm

    # Get drift parameters
    detector_half_width_x, drift_velocity_cm_us = get_drift_params(
        detector_config, dims_cm)
    params['detector_half_width_x'] = detector_half_width_x
    params['drift_velocity_cm_us'] = drift_velocity_cm_us

    # Get plane distances and furthest indices (combined function)
    all_plane_distances_cm, furthest_plane_indices = get_plane_geometry(detector_config)
    params['all_plane_distances_cm'] = all_plane_distances_cm
    params['furthest_plane_indices'] = furthest_plane_indices

    # Calculate time parameters
    num_time_steps, time_step_size_us, max_drift_time_us = calculate_time_params(
        detector_config, dims_cm, drift_velocity_cm_us)
    params['num_time_steps'] = num_time_steps
    params['time_step_size_us'] = time_step_size_us
    params['max_drift_time_us'] = max_drift_time_us

    # Calculate wire parameters for all planes
    (params['angles_rad'], params['wire_spacings_cm'], params['index_offsets'],
     params['num_wires_actual'], params['max_wire_indices_abs'],
     params['min_wire_indices_abs']) = pre_calculate_all_wire_params(
        detector_config, dims_cm)

    # Extract electron lifetime and diffusion parameters
    params['electron_lifetime_ms'] = float(detector_config['simulation']['drift']['electron_lifetime'])
    params['longitudinal_diffusion_cm2_s'] = float(detector_config['simulation']['drift']['longitudinal_diffusion'])
    params['transverse_diffusion_cm2_s'] = float(detector_config['simulation']['drift']['transverse_diffusion'])

    # Convert diffusion from cm²/s to cm²/μs for consistent units
    params['longitudinal_diffusion_cm2_us'] = params['longitudinal_diffusion_cm2_s'] / 1e6
    params['transverse_diffusion_cm2_us'] = params['transverse_diffusion_cm2_s'] / 1e6

    # Get wire spacing from config (using first plane as reference - all planes have same spacing)
    wire_spacing_cm = float(detector_config['wire_planes']['sides'][0]['planes'][0]['wire_spacing'])

    # Calculate maximum diffusion sigmas (both physical and unitless)
    (max_sigma_trans_cm, max_sigma_long_us,
     max_sigma_trans_unitless, max_sigma_long_unitless) = calculate_max_diffusion_sigmas(
        detector_half_width_x,
        drift_velocity_cm_us,
        params['transverse_diffusion_cm2_us'],
        params['longitudinal_diffusion_cm2_us'],
        wire_spacing_cm,
        time_step_size_us
    )
    params['max_sigma_trans_cm'] = max_sigma_trans_cm
    params['max_sigma_long_us'] = max_sigma_long_us
    params['max_sigma_trans_unitless'] = max_sigma_trans_unitless
    params['max_sigma_long_unitless'] = max_sigma_long_unitless

    # Load electrons per ADC conversion factor (default to MicroBooNE value if not specified)
    params['electrons_per_adc'] = float(detector_config['readout'].get('electrons_per_adc', 182))

    # Parse digitization config
    dig = detector_config['readout'].get('digitization', {})
    params['digitization'] = {
        'n_bits': int(dig.get('n_bits', 12)),
        'pedestal_collection': int(dig.get('pedestal_collection', 410)),
        'pedestal_induction': int(dig.get('pedestal_induction', 1843)),
        'gain_scale': float(dig.get('gain_scale', 1.0)),
    }

    # Wire lengths and noise RMS
    params['wire_lengths_m'] = _calculate_wire_lengths(
        dims_cm, params['angles_rad'], params['wire_spacings_cm'],
        params['index_offsets'], params['num_wires_actual'],
        params['min_wire_indices_abs'])

    noise_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'noise_spectrum.npz')
    if os.path.exists(noise_config_path):
        params['noise_rms'] = _calculate_noise_rms(params['wire_lengths_m'], noise_config_path)
    else:
        params['noise_rms'] = np.zeros((2, 3))

    return params


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


def get_plane_geometry(detector_config: Dict[str, Any]) -> Tuple[jnp.ndarray, np.ndarray]:
    """
    Get distances for all planes from anode and identify furthest plane per side.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary.

    Returns
    -------
    all_plane_distances_cm : jnp.ndarray
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

    all_plane_distances_cm = jnp.array(distances_cm)

    # Get furthest plane indices
    furthest_plane_indices = np.zeros(2, dtype=int)

    for side_idx in range(2):
        furthest_plane_indices[side_idx] = np.argmax(all_plane_distances_cm[side_idx])

    return all_plane_distances_cm, furthest_plane_indices


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
    # Assumes symmetrical geometry defined in the first side entry
    plane_config = detector_config['wire_planes']['sides'][0]['planes'][plane_idx]

    angle_deg = float(plane_config['angle'])
    angle_rad = jnp.radians(angle_deg)
    wire_spacing_cm = float(plane_config['wire_spacing'])

    if wire_spacing_cm <= 1e-9:
        raise ValueError("Wire spacing must be positive.")

    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)
    half_y, half_z = detector_y / 2.0, detector_z / 2.0

    # Calculate corners of the detector in the YZ plane
    corners_centered = jnp.array([
        [-half_y, -half_z], [+half_y, -half_z], [-half_y, +half_z], [+half_y, +half_z]
    ], dtype=jnp.float32)

    # Project corners onto the wire direction
    r_values = corners_centered[:, 0] * sin_theta + corners_centered[:, 1] * cos_theta
    r_min = jnp.min(r_values)
    r_max = jnp.max(r_values)

    # Calculate offset
    index_offset = 0
    if r_min < -1e-9:
        index_offset = int(jnp.floor(jnp.abs(r_min / wire_spacing_cm) + 1e-9)) + 1

    # Calculate relative and absolute indices
    idx_min_rel = jnp.floor(r_min / wire_spacing_cm - 1e-9).astype(jnp.int32)
    idx_max_rel = jnp.ceil(r_max / wire_spacing_cm + 1e-9).astype(jnp.int32)
    abs_idx_min = idx_min_rel + index_offset
    abs_idx_max = idx_max_rel + index_offset

    # Calculate number of wires
    num_wires = int(abs_idx_max - abs_idx_min + 1)
    max_wire_idx_abs = int(abs_idx_max)

    return angle_rad, wire_spacing_cm, index_offset, num_wires, max_wire_idx_abs


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
    num_time_steps = int(jnp.ceil(max_drift_time_us / time_step_size_us)) + 1
    num_time_steps = max(1, num_time_steps)

    return num_time_steps, time_step_size_us, max_drift_time_us


def pre_calculate_all_wire_params(detector_config: Dict[str, Any],
                                  dims_cm: Optional[Dict[str, float]] = None) -> Tuple[jnp.ndarray, ...]:
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
    angles_rad : jnp.ndarray
        Wire angles in radians, shape (2, 3).
    wire_spacings_cm : jnp.ndarray
        Spacing between wires in cm, shape (2, 3).
    index_offsets : jnp.ndarray
        Wire index offsets, shape (2, 3).
    num_wires_all : jnp.ndarray
        Number of wires for each plane, shape (2, 3).
    max_wire_indices_abs_all : jnp.ndarray
        Maximum absolute wire indices, shape (2, 3).
    min_wire_indices_abs_all : jnp.ndarray
        Minimum absolute wire indices, shape (2, 3).
    """
    if dims_cm is None:
        dims_cm = get_detector_dimensions(detector_config)

    # Initialize arrays
    num_wires_all = np.zeros((2, 3), dtype=int)
    max_wire_indices_abs_all = np.zeros((2, 3), dtype=int)
    min_wire_indices_abs_all = np.zeros((2, 3), dtype=int)
    index_offsets_all = np.zeros((2, 3), dtype=int)
    wire_spacings_all = np.zeros((2, 3), dtype=float)
    angles_all = np.zeros((2, 3), dtype=float)

    detector_y, detector_z = dims_cm['y'], dims_cm['z']
    half_y, half_z = detector_y / 2.0, detector_z / 2.0
    corners_centered = np.array([
        [-half_y, -half_z], [+half_y, -half_z], [-half_y, +half_z], [+half_y, +half_z]
    ], dtype=np.float32)

    for side_idx in range(2):
        for plane_idx in range(3):
            # Get wire parameters
            angle, spacing, offset, n_wires, max_idx_abs = get_single_plane_wire_params(
                detector_config, side_idx, plane_idx, dims_cm
            )

            # Calculate min_idx_abs
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            r_values = corners_centered[:, 0] * sin_theta + corners_centered[:, 1] * cos_theta
            r_min = np.min(r_values)
            idx_min_rel = np.floor(r_min / spacing - 1e-9).astype(np.int32)
            min_idx_abs = idx_min_rel + offset

            # Store values
            num_wires_all[side_idx, plane_idx] = n_wires
            max_wire_indices_abs_all[side_idx, plane_idx] = max_idx_abs
            min_wire_indices_abs_all[side_idx, plane_idx] = min_idx_abs
            index_offsets_all[side_idx, plane_idx] = offset
            wire_spacings_all[side_idx, plane_idx] = spacing
            angles_all[side_idx, plane_idx] = angle

    # Convert all arrays to JAX arrays for compatibility with JIT
    return (jnp.array(angles_all, dtype=jnp.float32),
            jnp.array(wire_spacings_all, dtype=jnp.float32),
            jnp.array(index_offsets_all, dtype=jnp.int32),
            jnp.array(num_wires_all, dtype=jnp.int32),
            jnp.array(max_wire_indices_abs_all, dtype=jnp.int32),
            jnp.array(min_wire_indices_abs_all, dtype=jnp.int32))


def print_detector_summary(detector_config: Dict[str, Any]) -> None:
    """
    Print a summary of the detector configuration.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary with pre-calculated parameters.
    """
    print("Detector Configuration Summary")
    print("==============================")

    # Basic detector properties
    print(f"Detector name: {detector_config['detector']['name']}")
    dimensions = detector_config['dims_cm']
    print(f"Dimensions: {dimensions['x']} × {dimensions['y']} × {dimensions['z']} cm³")

    # Wire planes information
    print("\nWire Plane Configuration:")
    print("------------------------")
    for side in detector_config['wire_planes']['sides']:
        side_id = side['side_id']
        print(f"Side {side_id}: {side['description']}")

        for plane in side['planes']:
            plane_id = plane['plane_id']
            plane_type = plane['type']
            print(f"  Plane {plane_id} ({plane_type}):")
            print(f"    Angle: {plane['angle']} degrees")
            print(f"    Wire spacing: {plane['wire_spacing']} cm")
            print(f"    Bias voltage: {plane['bias_voltage']} V")

    # Medium properties
    print(f"\nElectric field strength: {detector_config['electric_field']['field_strength']} V/cm")
    print(f"Medium: {detector_config['medium']['type']} at {detector_config['medium']['temperature']} K")

    # Derived simulation parameters
    print("\nDerived Parameters:")
    print("------------------")
    print(f"Number of time steps: {detector_config['num_time_steps']}")
    print(f"Time step size: {detector_config['time_step_size_us']:.6f} μs")
    print(f"Maximum drift time: {detector_config['max_drift_time_us']:.2f} μs")
    print(f"Drift velocity: {detector_config['drift_velocity_cm_us']:.2f} cm/μs")
    print(f"Electron lifetime: {detector_config['electron_lifetime_ms']:.2f} ms")
    print(f"Longitudinal diffusion: {detector_config['longitudinal_diffusion_cm2_s']:.6f} cm²/s")
    print(f"Transverse diffusion: {detector_config['transverse_diffusion_cm2_s']:.6f} cm²/s")
    print(f"\nDiffusion Sigmas (at max drift):")
    print(f"  Transverse:  {detector_config['max_sigma_trans_cm']:.3f} cm  ({detector_config['max_sigma_trans_unitless']:.3f} unitless)")
    print(f"  Longitudinal: {detector_config['max_sigma_long_us']:.3f} μs  ({detector_config['max_sigma_long_unitless']:.3f} unitless)")


if __name__ == "__main__":
    # Path to your detector configuration file
    config_path = "config/cubic_wireplane_config.yaml"

    # Generate the detector dictionary
    detector = generate_detector(config_path)

    if detector:
        print_detector_summary(detector)
    else:
        print("Failed to load detector configuration.")