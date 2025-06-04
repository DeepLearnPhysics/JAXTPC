# detector_geometry.py
import yaml
import os
import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Any, List, Union


def generate_detector(config_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Reads a JAXTPC detector configuration YAML file and returns a detector dictionary
    with all precalculated geometry parameters needed for simulation.

    Parameters:
        config_file_path (str): Path to the YAML configuration file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing all detector properties and
        derived parameters, or None if loading fails.
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


def _precalculate_all_parameters(detector_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all derived parameters needed for simulation from the detector config.

    Parameters:
        detector_config (Dict[str, Any]): Original detector configuration dictionary.

    Returns:
        Dict[str, Any]: Dictionary of all derived parameters.
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

    # Base detector resolution parameters
    params['sigma_wire_base_cm'] = 0.3
    params['sigma_time_base_us'] = 0.5

    return params


def get_detector_dimensions(detector_config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts detector dimensions from config and converts them to floats.

    Parameters:
        detector_config (Dict[str, Any]): Detector configuration dictionary.

    Returns:
        Dict[str, float]: Dictionary of detector dimensions in cm.
    """
    dims_cm = detector_config['detector']['dimensions']
    return {k: float(v) for k, v in dims_cm.items()}


def get_drift_params(detector_config: Dict[str, Any],
                     dims_cm: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    """
    Extracts global drift parameters, converting velocity to cm/us.

    Parameters:
        detector_config (Dict[str, Any]): Detector configuration dictionary.
        dims_cm (Optional[Dict[str, float]]): Pre-calculated dimensions to avoid redundant
                                             calculation. If None, dimensions are calculated.

    Returns:
        Tuple[float, float]: detector_half_width_x (cm), drift_velocity (cm/us)
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
    Gets distances for all planes from the anode and identifies the furthest plane for each side.

    Parameters:
        detector_config (Dict[str, Any]): Detector configuration dictionary.

    Returns:
        Tuple[jnp.ndarray, np.ndarray]:
            - all_plane_distances_cm: Array of shape (2, 3) with distances from anode
            - furthest_plane_indices: Array of shape (2,) with indices of furthest planes
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
    Extracts wire parameters for a single specified plane. Handles angles, spacing (cm),
    calculates offset and wire count/range.

    Parameters:
        detector_config (Dict[str, Any]): Detector configuration dictionary.
        side_idx (int): Index of the detector side (0 or 1).
        plane_idx (int): Index of the plane (0, 1, or 2).
        dims_cm (Optional[Dict[str, float]]): Pre-calculated dimensions to avoid redundant
                                             calculation. If None, dimensions are calculated.

    Returns:
        Tuple[float, float, int, int, int]: Tuple containing:
            - angle_rad (float): Wire angle in radians.
            - wire_spacing_cm (float): Spacing between wires in cm.
            - index_offset (int): Wire index offset.
            - num_wires (int): Number of wires in the plane.
            - max_wire_idx_abs (int): Maximum absolute wire index.
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
    Calculates time-related parameters from the detector config.

    Parameters:
        detector_config (Dict[str, Any]): Detector configuration dictionary.
        dims_cm (Optional[Dict[str, float]]): Pre-calculated dimensions to avoid redundant
                                             calculation. If None, dimensions are calculated.
        drift_velocity_cm_us (Optional[float]): Pre-calculated drift velocity to avoid
                                               redundant calculation. If None, velocity
                                               is calculated.

    Returns:
        Tuple[int, float, float]: Tuple containing:
            - num_time_steps (int): Number of time steps for simulation.
            - time_step_size_us (float): Size of time step in μs.
            - max_drift_time_us (float): Maximum drift time in μs.
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
    Pre-calculates wire parameters for all planes and sides.

    Parameters:
        detector_config (Dict[str, Any]): Detector configuration dictionary.
        dims_cm (Optional[Dict[str, float]]): Pre-calculated dimensions to avoid redundant
                                             calculation. If None, dimensions are calculated.

    Returns:
        Tuple[jnp.ndarray, ...]: Tuple containing six arrays of shape (2, 3):
            - angles_rad: Wire angles in radians.
            - wire_spacings_cm: Spacing between wires in cm.
            - index_offsets: Wire index offsets.
            - num_wires_all: Number of wires for each plane.
            - max_wire_indices_abs_all: Maximum absolute wire indices.
            - min_wire_indices_abs_all: Minimum absolute wire indices.
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
    Prints a summary of the detector configuration.

    Parameters:
        detector_config (Dict[str, Any]): Detector configuration dictionary with
                                         pre-calculated parameters.

    Returns:
        None
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


if __name__ == "__main__":
    # Path to your detector configuration file
    config_path = "config/cubic_wireplane_config.yaml"

    # Generate the detector dictionary
    detector = generate_detector(config_path)

    if detector:
        print_detector_summary(detector)
    else:
        print("Failed to load detector configuration.")