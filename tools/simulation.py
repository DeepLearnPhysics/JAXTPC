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
                  prepare_segment_no_diffusion, fill_signals_from_kernels)
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import recombine_steps
from tools.response_kernels import load_response_kernels, apply_diffusion_response

def create_wire_signal_calculator(
    detector_config, response_path="tools/responses/", num_s=16,
    max_num_hits_pad=500000
):
    """
    Factory function that creates a JIT-compiled function to calculate wire signals.
    
    Performs setup and returns a function to calculate wire signals for given 
    PADDED position/charge arrays and mask. Includes diffusion via response kernels,
    electron lifetime effects, and angular scaling.
    
    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary with pre-calculated parameters.
    response_path : str, optional
        Path to wire response kernel data, by default "tools/responses/".
    num_s : int, optional
        Number of diffusion levels in kernel interpolation, by default 16.
    max_num_hits_pad : int, optional
        Maximum number of hits to pad arrays for JIT compilation, by default 500000.
        
    Returns
    -------
    function
        Function to calculate wire signals.
    dict
        Dictionary of all parameters used in the calculation.
    """
    print("--- Creating Wire Signal Calculator (Factory Setup) ---")

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
    # Load kernels with proper wire and time spacing
    # The kernels use 0.1 cm bin spacing for sub-wire interpolation
    # This is independent of the actual physical wire spacing
    kernel_wire_spacing = 0.1  # Fixed kernel bin spacing
    time_spacing_float = float(time_step_size_us)
    
    kernel_info = load_response_kernels(
        response_path=response_path, 
        num_s=num_s,
        wire_spacing=kernel_wire_spacing,
        time_spacing=time_spacing_float
    )
    
    # Define the plane type mapping for each side/plane
    # 'U', 'V', 'Y'
    plane_types = [
        ['U', 'V', 'Y'],  # Side 0: U, V, Y
        ['U', 'V', 'Y']   # Side 1: U, V, Y
    ]
    
    print(f"   Config: Pad={max_num_hits_pad}, num_s={num_s}")
    print(f"   Physics: Lifetime={electron_lifetime_ms} ms, LongDiff={detector_config['longitudinal_diffusion_cm2_s']} cm²/s, TransDiff={detector_config['transverse_diffusion_cm2_s']} cm²/s")

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
        "max_num_hits_pad": max_num_hits_pad,
        "all_plane_distances_cm": all_plane_distances_cm,
        "detector_half_width_x_cm": detector_half_width_x_cm,
        "drift_velocity_cm_us": drift_velocity_cm_us,
        "dims_cm": detector_config['dims_cm'],
        "electron_lifetime_ms": electron_lifetime_ms,
        "longitudinal_diffusion_cm2_s": detector_config['longitudinal_diffusion_cm2_s'],
        "transverse_diffusion_cm2_s": detector_config['transverse_diffusion_cm2_s'],
        "longitudinal_diffusion_cm2_us": longitudinal_diffusion_cm2_us,
        "transverse_diffusion_cm2_us": transverse_diffusion_cm2_us,
        "furthest_plane_indices": furthest_plane_indices,
        "plane_types": plane_types,
        "num_s": num_s,
        "kernel_info": kernel_info,
    }

    # --- Define the INNER JIT function ---
    @partial(jax.jit, static_argnames=('max_wire_indices_static_tuple', 'min_wire_indices_static_tuple',
                                      'index_offsets_static_tuple', 'num_wires_static_tuple'))
    def _calculate_signals_for_event_jit(
        positions_mm_padded_array,          # Dynamic input (max_hits_pad, 3)
        charge_padded_array,                # Dynamic input (max_hits_pad,)
        valid_hit_mask_padded_array,        # Dynamic input (max_hits_pad,)
        theta_padded_array,                 # Dynamic input (max_hits_pad,)
        phi_padded_array,                   # Dynamic input (max_hits_pad,)
        max_wire_indices_static_tuple,      # Static input (tuple of tuples)
        min_wire_indices_static_tuple,      # Static input (tuple of tuples)
        index_offsets_static_tuple,         # Static input (tuple of tuples)
        num_wires_static_tuple             # Static input (tuple of tuples)
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

            # Calculate drift for the furthest plane on this side
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

                # Get plane type for kernel selection
                plane_type = plane_types[side_idx][plane_idx]

                # --- Calculate Drift ---
                # Calculate distance difference between furthest plane and this plane
                plane_dist_difference_cm = furthest_plane_dist_cm - plane_dist_cm
                drift_distance_cm_padded, drift_time_us_padded = _calculate_single_plane_drift_correction(
                    furthest_drift_distance_cm,
                    furthest_drift_time_us,
                    drift_velocity_cm_us,
                    plane_dist_difference_cm
                )

                # --- Calculate s parameter for diffusion ---
                # s = drift_distance / total_travel_distance
                # total_travel_distance is half the detector width
                total_travel_distance_cm = detector_half_width_x_cm
                s_values = drift_distance_cm_padded / total_travel_distance_cm
                # Clip s to [0, 1] range
                s_values = jnp.clip(s_values, 0.0, 1.0)

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

                # # Calculate the angular scaling factor
                # angular_scaling_factor = calculate_angular_scaling_vmap(theta_xz, theta_y)

                # --- STEP 1: Process segments without diffusion ---
                # Create vmapped version to process all hits
                prepare_segment_vmap = jax.vmap(
                    prepare_segment_no_diffusion,
                    in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None),
                )

                # Process all hits and get indices, offsets, and intensities
                segment_data = prepare_segment_vmap(
                    charge_side_masked_padded,
                    drift_time_us_padded,
                    closest_indices_abs,
                    closest_distances,
                    attenuation_factors,
                    combined_mask_for_jit,
                    spacing_cm,
                    time_step_size_us,
                    min_idx_abs,
                    num_wires
                )
                
                wire_indices_rel, wire_offsets, time_indices, time_offsets, intensities = segment_data

                # --- STEP 2: Apply diffusion response kernels ---
                # Get kernel data and parameters for this plane type
                plane_info = kernel_info[plane_type]
                DKernel = plane_info['DKernel']
                kernel_num_wires = plane_info['num_wires']
                kernel_height = plane_info['kernel_height']
                kernel_wire_stride = plane_info['wire_stride']
                kernel_wire_spacing = plane_info['wire_spacing']
                kernel_time_spacing = plane_info['time_spacing']
                
                # Get kernel contributions for this plane type
                contributions = apply_diffusion_response(
                    DKernel, s_values, wire_offsets, time_offsets,
                    kernel_wire_stride, kernel_wire_spacing, kernel_time_spacing, kernel_num_wires
                )

                # --- STEP 3: Fill output array with kernel contributions ---
                wire_signals_plane = fill_signals_from_kernels(
                    wire_indices_rel, time_indices, intensities, contributions,
                    num_wires, num_time_steps, kernel_num_wires, kernel_height
                )
                
                results.append(wire_signals_plane)

        # Return tuple of results
        return tuple(results)
    # --- End of INNER JIT function definition ---

    # Use functools.partial to bind the static TUPLES to the JIT function
    final_calculator_jit = partial(_calculate_signals_for_event_jit,
                                  max_wire_indices_static_tuple=max_indices_static_tuple,
                                  min_wire_indices_static_tuple=min_indices_static_tuple,
                                  index_offsets_static_tuple=index_offsets_static_tuple,
                                  num_wires_static_tuple=num_wires_static_tuple)

    # Wrapper function to convert the tuple of results to a dictionary
    def calculate_event_signals(positions_mm_padded, charge_padded, valid_hit_mask_padded, theta_padded, phi_padded):
        """
        Calculate wire signals for an event with padded inputs.
        
        Parameters
        ----------
        positions_mm_padded : jnp.ndarray
            Padded array of hit positions in mm, shape (max_hits_pad, 3).
        charge_padded : jnp.ndarray
            Padded array of hit charges, shape (max_hits_pad,).
        valid_hit_mask_padded : jnp.ndarray
            Padded boolean mask for valid hits, shape (max_hits_pad,).
        theta_padded : jnp.ndarray
            Padded array of theta angles in radians, shape (max_hits_pad,).
        phi_padded : jnp.ndarray
            Padded array of phi angles in radians, shape (max_hits_pad,).
            
        Returns
        -------
        dict
            Dictionary of wire signals, keyed by (side_idx, plane_idx).
        """
        
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

    # Return the wrapper function and the parameters dict
    return calculate_event_signals, all_params


def run_simulation(config_path, data_path, event_idx=0, 
                   MAX_HITS_PADDING=500000, num_s=16, response_path="tools/responses/"):
    """
    Run the detector simulation for a specific event.
    
    Parameters
    ----------
    config_path : str
        Path to detector configuration YAML file.
    data_path : str
        Path to particle step data HDF5 file.
    event_idx : int, optional
        Index of event to process, by default 0.
    MAX_HITS_PADDING : int, optional
        Maximum number of hits to pad arrays for JIT compilation, by default 500000.
    num_s : int, optional
        Number of diffusion levels in kernel interpolation, by default 16.
    response_path : str, optional
        Path to wire response kernel data, by default "tools/responses/".
        
    Returns
    -------
    dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    dict
        Dictionary of simulation parameters.
    """
    print("="*60)
    print(" LArTPC Wire Signal Simulation")
    print("="*60)
    print(f"Config: {config_path}, Data: {data_path} (Event {event_idx})")
    print(f"MaxHitsPad={MAX_HITS_PADDING}, num_s={num_s}")

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
            detector_config, response_path=response_path, num_s=num_s,
            max_num_hits_pad=MAX_HITS_PADDING
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
        step_data = load_particle_step_data(data_path, event_idx)
        event_positions_mm = jnp.asarray(step_data.get('position', jnp.empty((0,3))), dtype=jnp.float32)
        event_de = step_data.get('de', jnp.empty((0,)))

        positions_x = event_positions_mm[:, 0]
        print(f"number on east side: {jnp.sum(positions_x >= 0)}")
        print(f"number on west side: {jnp.sum(positions_x < 0)}")

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

            # --- Run the specialized JIT calculation ---
            print("Executing JIT calculator for the event...")
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

        return wire_signals_dict, simulation_params

    except Exception as e:
        print(f"\n--- An error occurred during event processing: ---")
        traceback.print_exc()
        raise