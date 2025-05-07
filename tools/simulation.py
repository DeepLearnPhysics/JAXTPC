import jax
import jax.numpy as jnp
import numpy as np
import time
import traceback
import os
from functools import partial

# Import our modules
from tools.drift import (_calculate_single_plane_drift_jit,
                  _calculate_single_plane_drift_correction,
                  calculate_drift_attenuation)
from tools.wires import (_calculate_single_plane_wire_distances_jit,
                  calculate_angular_scaling, calculate_angular_scaling_vmap,
                  calculate_segment_wire_angles, calculate_segment_wire_angles_vmap,
                  calculate_diffusion_response_normalized, 
                  prepare_segment_modified, fill_signals_array)
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import recombine_steps
from tools.responses import create_kernels_and_params, apply_response

def create_wire_signal_calculator(
    detector_config, response_path="tools/wire_responses/", n_dist=6, distance_falloff=1.0,
    K_wire=5, K_time=9, max_num_hits_pad=500000
):
    """
    Factory function: Performs setup and returns a JIT-compiled function
    to calculate wire signals for given PADDED position/charge arrays and mask.
    Now includes diffusion, electron lifetime effects, angle and wire distance interpolation,
    and wire response convolution.
    Returns signals as a dictionary of arrays with shape (num_wires, num_time_steps).
    """
    print("--- Creating Wire Signal Calculator (Factory Setup) ---")
    factory_start_time = time.time()

    # --- Extract the precalculated parameters ---
    print("   Reading detector parameters...")

    # Time parameters
    num_time_steps = detector_config['num_time_steps']
    time_step_size_us = detector_config['time_step_size_us']

    # Wire geometry parameters
    angles_rad_all = detector_config['angles_rad']
    wire_spacings_cm_all = detector_config['wire_spacings_cm']
    index_offsets_all_jax = detector_config['index_offsets']
    num_wires_actual = detector_config['num_wires_actual']
    max_wire_indices_abs_all_jax = detector_config['max_wire_indices_abs']
    min_wire_indices_abs_all_jax = detector_config['min_wire_indices_abs']

    # Drift parameters
    detector_half_width_x_cm = detector_config['detector_half_width_x']
    drift_velocity_cm_us = detector_config['drift_velocity_cm_us']

    # Plane distances
    all_plane_distances_cm = detector_config['all_plane_distances_cm']

    # Get closest plane indices
    furthest_plane_indices = detector_config['furthest_plane_indices']

    # Detector medium properties
    electron_lifetime_ms = detector_config['electron_lifetime_ms']
    longitudinal_diffusion_cm2_us = detector_config['longitudinal_diffusion_cm2_us']
    transverse_diffusion_cm2_us = detector_config['transverse_diffusion_cm2_us']

    # Resolution parameters
    sigma_wire_base_cm = detector_config.get('sigma_wire_base_cm', 0.3)
    sigma_time_base_us = detector_config.get('sigma_time_base_us', 0.5)

    # Define angle and wire distance interpolation points
    angle_points = jnp.linspace(0, 90, 10)  # 10 points from 0 to 90 degrees
    num_angles = len(angle_points)
    wire_distance_points = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # 6 distance points
    num_wire_distances = len(wire_distance_points)

    # --- Convert arrays needed as static args into hashable tuples ---
    index_offsets_np = np.array(index_offsets_all_jax)
    max_indices_np = np.array(max_wire_indices_abs_all_jax)
    min_indices_np = np.array(min_wire_indices_abs_all_jax)
    num_wires_np = np.array(num_wires_actual)

    index_offsets_static_tuple = tuple(tuple(int(x) for x in row) for row in index_offsets_np)
    max_indices_static_tuple = tuple(tuple(int(x) for x in row) for row in max_indices_np)
    min_indices_static_tuple = tuple(tuple(int(x) for x in row) for row in min_indices_np)
    num_wires_static_tuple = tuple(tuple(int(x) for x in row) for row in num_wires_np)

    # --- Create wire response kernels ---
    print("   Loading wire response kernels...")
    kernels_and_params = create_kernels_and_params(response_path, n_dist, distance_falloff)

    kernel_types = [
        kernels_and_params['U-plane']['kernels'],
        kernels_and_params['V-plane']['kernels'],
        kernels_and_params['Y-plane']['kernels']
    ]

    # Define the plane type mapping for each side/plane
    # 0=U, 1=V, 2=Y
    plane_type_indices = np.array([
        [0, 1, 2],  # Side 0: U, V, Y
        [0, 1, 2]   # Side 1: U, V, Y
    ])

    # Convert to tuple for JIT static argument
    plane_type_indices_tuple = tuple(tuple(int(x) for x in row) for row in plane_type_indices)
    
    precalc_end = time.time()
    print(f"   Parameters processed in {precalc_end - factory_start_time:.3f} s")
    print(f"   Config: K_wire={K_wire}, K_time={K_time}, Pad={max_num_hits_pad}")
    print(f"   Physics: Lifetime={electron_lifetime_ms} ms, LongDiff={detector_config['longitudinal_diffusion_cm2_s']} cm²/s, TransDiff={detector_config['transverse_diffusion_cm2_s']} cm²/s")
    print(f"   Interpolation: {num_angles} angles, {num_wire_distances} wire distances")
    print(f"   Response: {n_dist} distance steps, falloff={distance_falloff}")

    # Save parameters for later reference
    all_params = {
        "num_time_steps": num_time_steps,
        "time_step_size_us": time_step_size_us,
        "max_drift_time_us": detector_config['max_drift_time_us'],
        "angles_rad": angles_rad_all,
        "wire_spacings_cm": wire_spacings_cm_all,
        "index_offsets": index_offsets_all_jax,
        "max_abs_indices": max_wire_indices_abs_all_jax,
        "min_abs_indices": min_wire_indices_abs_all_jax,
        "num_wires_actual": num_wires_actual,
        "K_wire": K_wire,
        "K_time": K_time,
        "max_num_hits_pad": max_num_hits_pad,
        "all_plane_distances_cm": all_plane_distances_cm,
        "detector_half_width_x_cm": detector_half_width_x_cm,
        "drift_velocity_cm_us": drift_velocity_cm_us,
        "dims_cm": detector_config['dims_cm'],
        "electron_lifetime_ms": electron_lifetime_ms,
        "longitudinal_diffusion_cm2_s": detector_config['longitudinal_diffusion_cm2_s'],
        "transverse_diffusion_cm2_s": detector_config['transverse_diffusion_cm2_s'],
        "sigma_wire_base_cm": sigma_wire_base_cm,
        "sigma_time_base_us": sigma_time_base_us,
        "furthest_plane_indices": furthest_plane_indices,
        "angle_points": angle_points,
        "num_angles": num_angles,
        "wire_distance_points": wire_distance_points,
        "num_wire_distances": num_wire_distances,
        "plane_type_indices": plane_type_indices,
        "n_dist": n_dist,
        "distance_falloff": distance_falloff,
    }

    # --- Define the INNER JIT function ---
    @partial(jax.jit, static_argnames=('max_wire_indices_static_tuple', 'min_wire_indices_static_tuple',
                                      'index_offsets_static_tuple', 'num_wires_static_tuple', 'plane_type_indices_tuple'))
    def _calculate_signals_for_event_jit(
        positions_mm_padded_array,          # Dynamic input (max_hits_pad, 3)
        charge_padded_array,                # Dynamic input (max_hits_pad,)
        valid_hit_mask_padded_array,        # Dynamic input (max_hits_pad,)
        theta_padded_array,                 # Dynamic input (max_hits_pad,) - NEW
        phi_padded_array,                   # Dynamic input (max_hits_pad,) - NEW
        max_wire_indices_static_tuple,      # Static input (tuple of tuples)
        min_wire_indices_static_tuple,      # Static input (tuple of tuples)
        index_offsets_static_tuple,         # Static input (tuple of tuples)
        num_wires_static_tuple,             # Static input (tuple of tuples)
        plane_type_indices_tuple
        ):
        """ JIT - Calculates signals for pre-padded event data using captured & static args. """
        # 1. Prepare Padded Data (Units, Centering)
        positions_cm_padded = positions_mm_padded_array / 10.0
        positions_x_cm_padded = positions_cm_padded[:, 0]
        valid_hit_mask_padded = valid_hit_mask_padded_array
        charge_padded = charge_padded_array
        theta_padded = theta_padded_array
        phi_padded = phi_padded_array

        # 2. Perform Calculations and store results in a tuple of arrays
        # We'll return a tuple of (side0_plane0, side0_plane1, side0_plane2, side1_plane0, side1_plane1, side1_plane2)
        results = []

        for side_idx in range(2):
            is_on_side_mask_padded = (positions_x_cm_padded < 0) if side_idx == 0 else (positions_x_cm_padded >= 0)
            charge_side_masked_padded = jnp.where(is_on_side_mask_padded, charge_padded, 0.0)
            combined_mask_for_jit = valid_hit_mask_padded & is_on_side_mask_padded

            # Calculate drift for the closest plane on this side
            furthest_plane_idx = furthest_plane_indices[side_idx]
            furthest_plane_dist_cm = all_plane_distances_cm[side_idx, furthest_plane_idx]

            # Calculate drift time/distance to the furthest plane
            furthest_drift_distance_cm, furthest_drift_time_us, positions_yz_cm_padded = _calculate_single_plane_drift_jit(
                positions_cm_padded,
                detector_half_width_x_cm,
                drift_velocity_cm_us,
                furthest_plane_dist_cm
            )

            for plane_idx in range(3):
                # Access parameters
                plane_dist_cm = all_plane_distances_cm[side_idx, plane_idx]
                angle_rad = angles_rad_all[side_idx, plane_idx]
                spacing_cm = wire_spacings_cm_all[side_idx, plane_idx]
                offset = index_offsets_static_tuple[side_idx][plane_idx]
                max_idx_abs = max_wire_indices_static_tuple[side_idx][plane_idx]
                min_idx_abs = min_wire_indices_static_tuple[side_idx][plane_idx]
                num_wires = num_wires_static_tuple[side_idx][plane_idx]

                # --- Calculate Drift ---
                # Calculate distance difference between furthest plane and this plane
                plane_dist_difference_cm = furthest_plane_dist_cm - plane_dist_cm
                drift_distance_cm_padded, drift_time_us_padded = _calculate_single_plane_drift_correction(
                    furthest_drift_distance_cm,
                    furthest_drift_time_us,
                    drift_velocity_cm_us,
                    plane_dist_difference_cm
                )

                # --- Calculate electron lifetime attenuation ---
                electron_lifetime_us = electron_lifetime_ms * 1000.0  # Convert ms to us
                drift_time_us_safe = jnp.where(jnp.isnan(drift_time_us_padded), 0.0, drift_time_us_padded)
                attenuation_factors = jnp.exp(-drift_time_us_safe / electron_lifetime_us)

                # --- Calculate Closest Wire Distance ---
                # Calculate the closest wire for each hit
                closest_indices_abs, closest_distances = _calculate_single_plane_wire_distances_jit(
                    positions_yz_cm_padded, angle_rad, spacing_cm,
                    max_idx_abs, offset
                )

                theta_xz, theta_y = calculate_segment_wire_angles_vmap(
                    theta_padded, phi_padded, angle_rad
                )

                # Calculate the angular scaling factor
                angular_scaling_factor = calculate_angular_scaling_vmap(theta_xz, theta_y)

                # --- STEP 1: Process segments with prepare_segment ---
                # Create vmapped version to process all hits
                prepare_segment_vmap = jax.vmap(
                    prepare_segment_modified,
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None,
                            None, None, None, None, None, None, None, None),
                )

                # Process all hits and get indices and values
                indices_and_values = prepare_segment_vmap(
                    charge_side_masked_padded,
                    drift_time_us_padded,
                    drift_distance_cm_padded,
                    closest_indices_abs,
                    closest_distances,
                    attenuation_factors,
                    theta_xz,
                    theta_y,
                    angular_scaling_factor,
                    combined_mask_for_jit,
                    K_wire,
                    K_time,
                    spacing_cm,
                    time_step_size_us,
                    longitudinal_diffusion_cm2_us,
                    transverse_diffusion_cm2_us,
                    drift_velocity_cm_us,
                    num_angles,
                    num_wire_distances,
                    min_idx_abs,
                    num_wires,
                    num_time_steps
                )

                # --- STEP 2: Fill output array with signals ---
                wire_signals_plane = fill_signals_array(indices_and_values, num_wires, num_time_steps, num_angles, num_wire_distances)
                
                # --- STEP 3: Apply wire response convolution ---
                plane_type_idx = plane_type_indices_tuple[side_idx][plane_idx]
                kernels = kernel_types[plane_type_idx]
                
                # Apply convolution and collapse the result
                wire_signals_plane_collapsed = apply_response(
                    wire_signals_plane, kernels, num_angles, num_wire_distances
                )

                results.append(wire_signals_plane_collapsed)

        # Return tuple of results
        return tuple(results)
    # --- End of INNER JIT function definition ---

    # Use functools.partial to bind the static TUPLES to the JIT function
    final_calculator_jit = partial(_calculate_signals_for_event_jit,
                                  max_wire_indices_static_tuple=max_indices_static_tuple,
                                  min_wire_indices_static_tuple=min_indices_static_tuple,
                                  index_offsets_static_tuple=index_offsets_static_tuple,
                                  num_wires_static_tuple=num_wires_static_tuple,
                                  plane_type_indices_tuple=plane_type_indices_tuple)

    # Wrapper function to convert the tuple of results to a dictionary
    def calculate_event_signals(positions_mm_padded, charge_padded, valid_hit_mask_padded, theta_padded, phi_padded):
        # Call the JIT-compiled function
        result_tuple = final_calculator_jit(positions_mm_padded, charge_padded, valid_hit_mask_padded,
                                           theta_padded, phi_padded)

        # Convert tuple to dictionary
        result_dict = {}
        idx = 0
        for side_idx in range(2):
            for plane_idx in range(3):
                result_dict[(side_idx, plane_idx)] = result_tuple[idx]
                idx += 1

        return result_dict

    factory_end_time = time.time()
    print(f"--- Factory setup finished ({factory_end_time - factory_start_time:.2f} s) ---")

    # Return the wrapper function and the parameters dict
    return calculate_event_signals, all_params


def run_simulation(config_path, data_path, event_idx=0, 
                   K_wire=5, K_time=9, MAX_HITS_PADDING=500_00, 
                   n_dist=6, distance_falloff=1.0, response_path="tools/wire_responses/"):
    """
    Run the detector simulation for a specific event.

    Args:
        config_path: Path to detector configuration YAML file
        data_path: Path to particle step data HDF5 file
        event_idx: Index of event to process
        K_wire: Number of wire neighbors to consider in signal calculation
        K_time: Number of time bins to consider in signal calculation
        MAX_HITS_PADDING: Maximum number of hits to pad arrays for JIT compilation
        n_dist: Number of distance steps for wire response
        distance_falloff: Falloff parameter for wire response with distance
        response_path: Path to wire response data files

    Returns:
        wire_signals_dict: Dictionary of wire signals
        simulation_params: Dictionary of simulation parameters
    """
    print("="*60)
    print(" LArTPC Wire Signal Simulation")
    print("="*60)
    print(f"Config: {config_path}, Data: {data_path} (Event {event_idx})")
    print(f"Response path: {response_path}, n_dist={n_dist}, falloff={distance_falloff}")
    print(f"K={K_wire}, K_time={K_time}, MaxHitsPad={MAX_HITS_PADDING}")

    # --- Load Configuration ---
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        try:
            detector_config = generate_detector(config_path)
            if detector_config is None:
                raise ValueError(f"Error loading detector configuration from {config_path}")
            print(f"Successfully loaded detector config from {config_path}")
        except Exception as e:
            print(f"ERROR loading config file '{config_path}': {e}")
            traceback.print_exc()
            raise

    # --- Create the Specialized Calculator (ONCE) ---
    try:
        calculate_event_signals, simulation_params = create_wire_signal_calculator(
            detector_config, response_path=response_path, n_dist=n_dist, distance_falloff=distance_falloff,
            K_wire=K_wire, K_time=K_time, max_num_hits_pad=MAX_HITS_PADDING
        )
        # First call will trigger compilation
        print("\nTriggering JIT compilation (if not cached)...")
        dummy_pos_padded = jnp.zeros((MAX_HITS_PADDING, 3), dtype=jnp.float32)
        dummy_charge_padded = jnp.zeros((MAX_HITS_PADDING,), dtype=jnp.float32)
        dummy_mask_padded = jnp.zeros((MAX_HITS_PADDING,), dtype=bool)
        dummy_theta_padded = jnp.zeros((MAX_HITS_PADDING,), dtype=jnp.float32)
        dummy_phi_padded = jnp.zeros((MAX_HITS_PADDING,), dtype=jnp.float32)
        _ = calculate_event_signals(dummy_pos_padded, dummy_charge_padded, dummy_mask_padded,
                                   dummy_theta_padded, dummy_phi_padded)
        print("JIT compilation finished.")
    except Exception as e:
         print(f"\n--- An error occurred during calculator creation or compilation: ---")
         traceback.print_exc()
         raise

    # --- Process Event Data ---
    try:
        print(f"\n--- Processing Event {event_idx} ---")
        proc_start = time.time()
        step_data = load_particle_step_data(data_path, event_idx)
        event_positions_mm = jnp.asarray(step_data.get('position', jnp.empty((0,3))), dtype=jnp.float32)
        event_de = step_data.get('de', jnp.empty((0,)))

        # Extract theta and phi from step_data
        event_theta = jnp.asarray(step_data.get('theta', jnp.zeros((event_positions_mm.shape[0],))), dtype=jnp.float32)
        event_phi = jnp.asarray(step_data.get('phi', jnp.zeros((event_positions_mm.shape[0],))), dtype=jnp.float32)

        n_hits = event_positions_mm.shape[0]

        print(f"Loaded {n_hits} steps from event {event_idx}.")

        if n_hits == 0:
             print("WARNING: Event contains no particle steps. Skipping signal calculation.")
             # Create empty dictionary with correct format
             wire_signals_dict = {}
             for side_idx in range(2):
                 for plane_idx in range(3):
                     num_wires = simulation_params['num_wires_actual'][side_idx, plane_idx]
                     if num_wires > 0:
                         wire_signals_dict[(side_idx, plane_idx)] = jnp.zeros(
                             (num_wires, simulation_params['num_time_steps'])
                         )
        else:
            recomb_charge = recombine_steps(step_data, detector_config)
            recomb_charge = jnp.asarray(recomb_charge, dtype=jnp.float32)
            print(f"Calculated recombined charge (shape: {recomb_charge.shape})")

            # --- Perform Padding BEFORE JIT call ---
            print("Padding event data...")
            pad_start = time.time()
            pad_width = MAX_HITS_PADDING - n_hits

            if pad_width > 0:
                valid_hit_mask_padded = jnp.arange(MAX_HITS_PADDING) < n_hits
                charge_padded = jnp.pad(recomb_charge, (0, pad_width), constant_values=0.0)
                positions_mm_padded = jnp.pad(event_positions_mm, ((0, pad_width), (0, 0)), constant_values=0.0)

                # Pad theta and phi arrays
                theta_padded = jnp.pad(event_theta, (0, pad_width), constant_values=0.0)
                phi_padded = jnp.pad(event_phi, (0, pad_width), constant_values=0.0)

            else:
                valid_hit_mask_padded = jnp.ones(MAX_HITS_PADDING) < n_hits
                charge_padded = recomb_charge[:MAX_HITS_PADDING]
                positions_mm_padded = event_positions_mm[:MAX_HITS_PADDING]
                theta_padded = event_theta[:MAX_HITS_PADDING]
                phi_padded = event_phi[:MAX_HITS_PADDING]

            pad_end = time.time()
            print(f"Padding took {pad_end - pad_start:.4f} s")

            # --- Run the specialized JIT calculation ---
            print("Executing JIT calculator for the event...")
            exec_start = time.time()
            wire_signals_dict = calculate_event_signals(
                positions_mm_padded,
                charge_padded,
                valid_hit_mask_padded,
                theta_padded,
                phi_padded
            )
            # Ensure all calculations are complete
            for key, arr in wire_signals_dict.items():
                jax.device_get(arr)
            exec_end = time.time()
            print(f"Event calculation finished in {exec_end - exec_start:.3f} s")

        proc_end = time.time()
        print(f"Event processing took {proc_end - proc_start:.2f} s total.")

        return wire_signals_dict, simulation_params

    except Exception as e:
        print(f"\n--- An error occurred during event processing: ---")
        traceback.print_exc()
        raise