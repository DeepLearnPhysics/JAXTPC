import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

def visualize_wire_signals(wire_signals_dict, simulation_params, figsize=(20, 10), log_norm=False):
    """
    Visualize wire signals stored in a dictionary, using different color schemes per plane type.
    
    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    simulation_params : dict
        Dictionary containing simulation parameters.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization for all plots, by default False.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    # Extract pre-calculated parameters
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']
    
    # Plane name mapping
    plane_name_mapping = {
        (0, 0): 'U-plane',  # Side 0, Plane 0 -> U-plane
        (0, 1): 'V-plane',  # Side 0, Plane 1 -> V-plane
        (0, 2): 'Y-plane',  # Side 0, Plane 2 -> Y-plane
        (1, 0): 'U-plane',  # Side 1, Plane 0 -> U-plane
        (1, 1): 'V-plane',  # Side 1, Plane 1 -> V-plane
        (1, 2): 'Y-plane',  # Side 1, Plane 2 -> Y-plane
    }
    
    # Define colormap settings - using same colormap for all planes
    cmap_settings = {
        'U-plane': {'cmap': 'seismic'},
        'V-plane': {'cmap': 'seismic'},
        'Y-plane': {'cmap': 'seismic'}  # Now using same colormap as U and V
    }
    
    # Find min/max values for each plane type
    plane_min_max = {
        'U-plane': {'min': float('inf'), 'max': -float('inf')},
        'V-plane': {'min': float('inf'), 'max': -float('inf')},
        'Y-plane': {'min': float('inf'), 'max': -float('inf')}  # Treating Y-plane same as others
    }
    
    # Calculate min/max for each plane type
    for s in range(2):
        for p in range(3):
            if (s, p) in wire_signals_dict and num_wires_actual[s, p] > 0:
                plane_name = plane_name_mapping[(s, p)]
                signal_data = np.array(wire_signals_dict[(s, p)])
                if signal_data.size > 0:
                    # Track min/max for all plane types
                    plane_min_max[plane_name]['min'] = min(plane_min_max[plane_name]['min'], signal_data.min())
                    plane_min_max[plane_name]['max'] = max(plane_min_max[plane_name]['max'], signal_data.max())
    
    # Set fixed ranges if no data found or for specific plane types
    for plane_name in plane_min_max:
        # Uniform handling for all plane types - ensure symmetric range around zero
        if plane_min_max[plane_name]['min'] == float('inf'):
            # Default range if no data
            plane_min_max[plane_name]['min'] = -25
            plane_min_max[plane_name]['max'] = 25
        else:
            # Ensure range is symmetric around zero
            max_abs_val = max(abs(plane_min_max[plane_name]['min']), abs(plane_min_max[plane_name]['max']))
            plane_min_max[plane_name]['min'] = -max_abs_val
            plane_min_max[plane_name]['max'] = max_abs_val
    
    print("   Visualization Norms by Plane Type:")
    for plane_name in plane_min_max:
        print(f"   - {plane_name}: min={plane_min_max[plane_name]['min']:.2e}, max={plane_min_max[plane_name]['max']:.2e}")

    # Create figure and plot with white background
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor('black')
            ax.grid(False)
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
            plot_title = f"{side_names[side_idx]}\n{plane_types[plane_idx]}"
            plane_name = plane_name_mapping[(side_idx, plane_idx)]

            if (side_idx, plane_idx) not in wire_signals_dict or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", color='grey', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
                ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
                ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
                ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
                ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue

            signal_data_to_plot = np.array(wire_signals_dict[(side_idx, plane_idx)])
            extent_xmin = min_idx_abs
            extent_xmax = max_idx_abs + 1
            extent = [extent_xmin, extent_xmax, 0, max_time_axis]
            
            # Get colormap for this plane type
            cmap = cmap_settings[plane_name]['cmap']
            vmin = plane_min_max[plane_name]['min']
            vmax = plane_min_max[plane_name]['max']
            
            # Create normalization based on log_norm parameter
            if log_norm:
                # Use SymLogNorm for all planes with 5% threshold
                # This provides linear scaling near zero and logarithmic scaling for larger values
                max_abs_val = max(abs(vmin), abs(vmax))
                linthresh = max(1e-8, 0.015 * max_abs_val)  # Linear threshold - 1% of max value or at least 1e-8
                
                # Same treatment for all planes
                norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, clip=True)
                
                im = ax.imshow(
                    signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                    cmap=cmap, norm=norm
                )
            else:
                # Linear normalization
                im = ax.imshow(
                    signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                    cmap=cmap, vmin=vmin, vmax=vmax
                )
            
            ax.set_ylim(0, max_time_axis)
            ax.set_xlim(extent_xmin, extent_xmax)
            ax.set_box_aspect(1)
            ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
            ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
            ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
            ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.08)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=tick_size, colors='black')
            cbar.set_label('Signal Strength', fontsize=label_size, color='black')
    return fig


def visualize_single_plane(wire_signals_dict, simulation_params, side_idx=0, plane_idx=0, figsize=(10, 10), log_norm=False):
    """
    Visualize wire signals for a single side/plane combination using appropriate color scheme.
    
    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    simulation_params : dict
        Dictionary containing simulation parameters.
    side_idx : int, optional
        Index of the side to plot (0=West, 1=East), by default 0.
    plane_idx : int, optional
        Index of the plane to plot (0=U, 1=V, 2=Y), by default 0.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization, by default False.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    print(f"--- Starting Visualization for Side {side_idx}, Plane {plane_idx} ---")

    # Extract pre-calculated parameters
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    # Plane name mapping
    plane_name_mapping = {
        (0, 0): 'U-plane',  # Side 0, Plane 0 -> U-plane
        (0, 1): 'V-plane',  # Side 0, Plane 1 -> V-plane
        (0, 2): 'Y-plane',  # Side 0, Plane 2 -> Y-plane
        (1, 0): 'U-plane',  # Side 1, Plane 0 -> U-plane
        (1, 1): 'V-plane',  # Side 1, Plane 1 -> V-plane
        (1, 2): 'Y-plane',  # Side 1, Plane 2 -> Y-plane
    }
    
    # Define colormap settings - using same colormap for all planes
    cmap_settings = {
        'U-plane': {'cmap': 'seismic'},
        'V-plane': {'cmap': 'seismic'},
        'Y-plane': {'cmap': 'seismic'}  # Now using same colormap as U and V
    }
    
    # Get the corresponding plane name
    s, p = side_idx, plane_idx
    plane_name = plane_name_mapping[(s, p)]
    
    # Initialize min/max values
    min_val, max_val = float('inf'), -float('inf')
    
    # Check both sides but same plane type to find min/max
    for check_side in range(2):
        check_plane = p  # Same plane type
        check_key = (check_side, check_plane)
        
        if check_key in wire_signals_dict and num_wires_actual[check_side, check_plane] > 0:
            signal_data = np.array(wire_signals_dict[check_key])
            if signal_data.size > 0:
                # Track full range
                min_val = min(min_val, signal_data.min())
                max_val = max(max_val, signal_data.max())
    
    # Set fixed ranges if no data found or for specific plane types - uniform handling
    if min_val == float('inf'):
        # Default range if no data
        min_val, max_val = -25, 25
    else:
        # Ensure range is symmetric around zero for all plane types
        max_abs_val = max(abs(min_val), abs(max_val))
        min_val, max_val = -max_abs_val, max_abs_val
    
    print(f"   Visualization Norm for {plane_name}: min={min_val:.2e}, max={max_val:.2e}")

    # Create figure and plot with white background
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('black')
    ax.grid(False)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    min_idx_abs = int(min_abs_indices[s, p])
    max_idx_abs = int(max_abs_indices[s, p])
    actual_wire_count = int(num_wires_actual[s, p])
    plot_title = f"{side_names[s]}\n{plane_types[p]}"

    if (s, p) not in wire_signals_dict or actual_wire_count == 0:
        ax.text(0.5, 0.5, "(0 wires active)", color='grey', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
        ax.set_xlabel('Wire Index', fontsize=label_size, color='black')
        ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
        ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
        ax.set_xlim(min_idx_abs, max_idx_abs + 1)
        ax.set_ylim(0, max_time_axis)
        ax.set_box_aspect(1)
    else:
        signal_data_to_plot = np.array(wire_signals_dict[(s, p)])
        extent_xmin = min_idx_abs
        extent_xmax = max_idx_abs + 1
        extent = [extent_xmin, extent_xmax, 0, max_time_axis]
        
        # Get colormap for this plane type
        cmap = cmap_settings[plane_name]['cmap']
        vmin = min_val
        vmax = max_val
        
        # Create normalization based on log_norm parameter
        if log_norm:
            # Use SymLogNorm for all planes with 5% threshold
            max_abs_val = max(abs(vmin), abs(vmax))
            linthresh = max(1e-8, 0.01 * max_abs_val)  # Linear threshold - 1% of max value or at least 1e-8
            
            # Same treatment for all planes
            norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, clip=True)
            
            im = ax.imshow(
                signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                cmap=cmap, norm=norm
            )
        else:
            # Linear normalization
            im = ax.imshow(
                signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                cmap=cmap, vmin=vmin, vmax=vmax
            )

        ax.set_ylim(0, max_time_axis)
        ax.set_xlim(extent_xmin, extent_xmax)
        ax.set_box_aspect(1)
        ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
        ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
        ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
        ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=tick_size, colors='black')
        cbar.set_label('Signal Strength', fontsize=label_size, color='black')
        cbar.outline.set_edgecolor('white')
    return fig


def visualize_wire_planes_colored_by_index(detector_config, figsize=(15, 10)):
    """
    Visualize all 6 wire planes (3 on each side) of the LArTPC detector,
    coloring the wires based on their index.

    Parameters
    ----------
    detector_config : dict
        Detector configuration dictionary with pre-calculated parameters.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (15, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Extract detector dimensions
    detector_dims = detector_config['detector']['dimensions']
    detector_y = detector_dims['y']
    detector_z = detector_dims['z']

    # Extract wire plane information
    sides = detector_config['wire_planes']['sides']

    # Create figure and axes - 2 rows (for sides) x 3 columns (for planes)
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Set up titles for the planes
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    # Define a colormap for the wire indices
    cmap = plt.cm.viridis

    # Loop through each side (0: x < 0, 1: x > 0)
    for side_idx, side in enumerate(sides):
        side_desc = side['description']

        # Loop through each plane on this side
        for plane_idx, plane in enumerate(side['planes']):
            ax = axes[side_idx, plane_idx]

            # Extract plane parameters
            angle_deg = plane['angle']
            angle_rad = np.radians(angle_deg)
            wire_spacing = plane['wire_spacing']
            distance_from_anode = plane['distance_from_anode']

            # Display information
            title = f"{side_desc}\n{plane_types[plane_idx]}"
            ax.set_title(title)

            # Draw a representation of the detector boundaries
            # Z is horizontal (width) and Y is vertical (height)
            ax.add_patch(plt.Rectangle((0, 0), detector_z, detector_y, fill=False, color='black', linestyle='--'))

            # Calculate sine and cosine once
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)

            # Calculate the parameter values for all four corners of the detector
            corners = [
                (0, 0),  # Bottom-left (y=0, z=0)
                (detector_y, 0),  # Top-left (y=detector_y, z=0)
                (0, detector_z),  # Bottom-right (y=0, z=detector_z)
                (detector_y, detector_z)  # Top-right (y=detector_y, z=detector_z)
            ]

            # NEW PARAMETRIZATION: r = z * cos(θ) + y * sin(θ)
            r_values = [z * cos_theta + y * sin_theta for y, z in corners]
            r_min = min(r_values)
            r_max = max(r_values)

            # Calculate index offset for negative angles
            offset = 0
            if r_min < 0:
                offset = int(np.abs(np.floor(r_min / wire_spacing))) + 1

            # Calculate exact wire index range with offset applied
            idx_min = int(np.floor(r_min / wire_spacing)) + offset
            idx_max = int(np.ceil(r_max / wire_spacing)) + offset

            # Store the number of wires for normalization
            num_wires = idx_max - idx_min + 1

            # Draw each wire within this range
            for wire_idx in range(idx_min, idx_max + 1):
                # Wire parameter r (adjusted for offset)
                r = (wire_idx - offset) * wire_spacing

                # Calculate intersection points with the four boundaries
                # Using parametrization: r = z * cos(θ) + y * sin(θ)
                intersections = []

                # Check intersection with y=0 (bottom boundary)
                # r = z * cos(θ) + 0 * sin(θ) => z = r / cos(θ)
                if abs(cos_theta) > 1e-10:
                    z = r / cos_theta
                    if 0 <= z <= detector_z:
                        intersections.append((0, z))

                # Check intersection with y=detector_y (top boundary)
                # r = z * cos(θ) + detector_y * sin(θ) => z = (r - detector_y * sin(θ)) / cos(θ)
                if abs(cos_theta) > 1e-10:
                    z = (r - detector_y * sin_theta) / cos_theta
                    if 0 <= z <= detector_z:
                        intersections.append((detector_y, z))

                # Check intersection with z=0 (left boundary)
                # r = 0 * cos(θ) + y * sin(θ) => y = r / sin(θ)
                if abs(sin_theta) > 1e-10:
                    y = r / sin_theta
                    if 0 <= y <= detector_y:
                        intersections.append((y, 0))

                # Check intersection with z=detector_z (right boundary)
                # r = detector_z * cos(θ) + y * sin(θ) => y = (r - detector_z * cos(θ)) / sin(θ)
                if abs(sin_theta) > 1e-10:
                    y = (r - detector_z * cos_theta) / sin_theta
                    if 0 <= y <= detector_y:
                        intersections.append((y, detector_z))

                # Draw the wire if we have at least 2 intersections
                if len(intersections) >= 2:
                    # Sort intersections appropriately
                    if len(intersections) > 2:
                        # Remove duplicates first
                        unique_intersections = []
                        for pt in intersections:
                            is_duplicate = False
                            for existing in unique_intersections:
                                if abs(pt[0] - existing[0]) < 1e-6 and abs(pt[1] - existing[1]) < 1e-6:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                unique_intersections.append(pt)
                        intersections = unique_intersections

                    if len(intersections) >= 2:
                        # Sort by the coordinate that varies most
                        p1, p2 = intersections[0], intersections[1]
                        dy = abs(p2[0] - p1[0])
                        dz = abs(p2[1] - p1[1])

                        if dz > dy:
                            intersections.sort(key=lambda p: p[1])  # Sort by z
                        else:
                            intersections.sort(key=lambda p: p[0])  # Sort by y

                        p1, p2 = intersections[0], intersections[-1]

                        # Calculate normalized wire index for coloring
                        norm_idx = (wire_idx - idx_min) / max(1, num_wires - 1)
                        color = cmap(norm_idx)

                        # Plot with Z on x-axis and Y on y-axis
                        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color, linewidth=0.8, alpha=0.7)

            # Set axis properties - Z horizontal, Y vertical
            ax.set_xlabel('Z Position (cm)')
            ax.set_ylabel('Y Position (cm)')
            ax.set_xlim(0, detector_z)
            ax.set_ylim(0, detector_y)
            ax.grid(alpha=0.3)

            # Add plane info as text
            info_text = f"Angle: {angle_deg}°\nSpacing: {wire_spacing} cm\nDistance from anode: {distance_from_anode} cm"
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, va='top',
                    bbox=dict(facecolor='white', alpha=0.7))

            # Add a mini colorbar for this subplot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=idx_min, vmax=idx_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Wire Index')

    plt.tight_layout()
    return fig

def visualize_diffused_charge(wire_signals_dict, simulation_params, figsize=(20, 10), log_norm=False):
    """
    Visualize diffused charge signals (without detector response) stored in a dictionary.
    
    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    simulation_params : dict
        Dictionary containing simulation parameters.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization for all plots, by default False.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    # Extract pre-calculated parameters
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']
    
    # Find min/max values across all planes
    global_min = float('inf')
    global_max = -float('inf')
    
    for s in range(2):
        for p in range(3):
            if (s, p) in wire_signals_dict and num_wires_actual[s, p] > 0:
                signal_data = np.array(wire_signals_dict[(s, p)])
                if signal_data.size > 0:
                    # Find actual data range
                    data_min = signal_data.min()
                    data_max = signal_data.max()
                    if data_max > 0:  # Only update if we have positive values
                        global_min = min(global_min, data_min)
                        global_max = max(global_max, data_max)
    
    # Set default range if no data found
    if global_min == float('inf'):
        global_min, global_max = 0, 1

    # Calculate percentile-based range to avoid outliers
    percentiles = np.percentile(
        [np.array(wire_signals_dict[(s, p)]).flatten()
         for s in range(2) for p in range(3)
         if (s, p) in wire_signals_dict and num_wires_actual[s, p] > 0 and np.array(wire_signals_dict[(s, p)]).size > 0],
        [1, 99]
    )
    p1, p99 = percentiles
    global_min = max(global_min, p1)
    global_max = min(global_max, p99)
    
    print(f"   Diffused Charge Visualization Range: min={global_min:.2e}, max={global_max:.2e}")

    # Create figure and plot with white background
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor('white')
            ax.grid(True, alpha=0.3)
            
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
            plot_title = f"{side_names[side_idx]}\n{plane_types[plane_idx]}"

            if (side_idx, plane_idx) not in wire_signals_dict or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", color='grey', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=title_size, pad=10)
                ax.set_xlabel('Absolute Wire Index', fontsize=label_size)
                ax.set_ylabel('Time (μs)', fontsize=label_size)
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue

            signal_data_to_plot = np.array(wire_signals_dict[(side_idx, plane_idx)])
            extent_xmin = min_idx_abs
            extent_xmax = max_idx_abs + 1
            extent = [extent_xmin, extent_xmax, 0, max_time_axis]
            
            # Create normalization based on log_norm parameter
            if log_norm:
                # Use logarithmic normalization with vmin set to be 6 orders of magnitude below vmax
                vmax = global_max
                vmin = vmax / 1e6  # 6 orders of magnitude below maximum
                norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
                
                im = ax.imshow(
                    signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                    cmap='inferno', norm=norm
                )
            else:
                # Linear normalization
                im = ax.imshow(
                    signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                    cmap='inferno', vmin=global_min, vmax=global_max
                )
            
            ax.set_ylim(0, max_time_axis)
            ax.set_xlim(extent_xmin, extent_xmax)
            ax.set_box_aspect(1)
            ax.set_title(plot_title, fontsize=title_size, pad=10)
            ax.set_xlabel('Absolute Wire Index', fontsize=label_size)
            ax.set_ylabel('Time (μs)', fontsize=label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.08)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=tick_size)
            cbar.set_label('Diffused Charge', fontsize=label_size)
            
    return fig


def visualize_diffused_charge_single_plane(wire_signals_dict, simulation_params, side_idx=0, plane_idx=0, figsize=(10, 10), log_norm=False):
    """
    Visualize diffused charge signals for a single side/plane combination.
    
    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    simulation_params : dict
        Dictionary containing simulation parameters.
    side_idx : int, optional
        Index of the side to plot (0=West, 1=East), by default 0.
    plane_idx : int, optional
        Index of the plane to plot (0=U, 1=V, 2=Y), by default 0.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization, by default False.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    print(f"--- Visualizing Diffused Charge for Side {side_idx}, Plane {plane_idx} ---")

    # Extract pre-calculated parameters
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    # Get the corresponding plane data
    s, p = side_idx, plane_idx
    
    # Initialize min/max values
    min_val, max_val = float('inf'), -float('inf')
    
    # Check both sides but same plane type to find min/max
    for check_side in range(2):
        check_plane = p  # Same plane type
        check_key = (check_side, check_plane)
        
        if check_key in wire_signals_dict and num_wires_actual[check_side, check_plane] > 0:
            signal_data = np.array(wire_signals_dict[check_key])
            if signal_data.size > 0:
                data_min = signal_data.min()
                data_max = signal_data.max()
                if data_max > 0:  # Only update if we have positive values
                    min_val = min(min_val, data_min)
                    max_val = max(max_val, data_max)
    
    # Set default range if no data found
    if min_val == float('inf'):
        min_val, max_val = 0, 1
    
    print(f"   Visualization Range: min={min_val:.2e}, max={max_val:.2e}")

    # Create figure and plot with white background
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    min_idx_abs = int(min_abs_indices[s, p])
    max_idx_abs = int(max_abs_indices[s, p])
    actual_wire_count = int(num_wires_actual[s, p])
    plot_title = f"{side_names[s]}\n{plane_types[p]}"

    if (s, p) not in wire_signals_dict or actual_wire_count == 0:
        ax.text(0.5, 0.5, "(0 wires active)", color='grey', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(plot_title, fontsize=title_size, pad=10)
        ax.set_xlabel('Absolute Wire Index', fontsize=label_size)
        ax.set_ylabel('Time (μs)', fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.set_xlim(min_idx_abs, max_idx_abs + 1)
        ax.set_ylim(0, max_time_axis)
        ax.set_box_aspect(1)
    else:
        signal_data_to_plot = np.array(wire_signals_dict[(s, p)])
        extent_xmin = min_idx_abs
        extent_xmax = max_idx_abs + 1
        extent = [extent_xmin, extent_xmax, 0, max_time_axis]
        
        # Create normalization based on log_norm parameter
        if log_norm:
            # Use logarithmic normalization with vmin set to be 6 orders of magnitude below vmax
            vmax = max_val
            vmin = vmax / 1e6  # 6 orders of magnitude below maximum
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            
            im = ax.imshow(
                signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                cmap='inferno', norm=norm
            )
        else:
            # Linear normalization
            im = ax.imshow(
                signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                cmap='inferno', vmin=min_val, vmax=max_val
            )

        ax.set_ylim(0, max_time_axis)
        ax.set_xlim(extent_xmin, extent_xmax)
        ax.set_box_aspect(1)
        ax.set_title(plot_title, fontsize=title_size, pad=10)
        ax.set_xlabel('Absolute Wire Index', fontsize=label_size)
        ax.set_ylabel('Time (μs)', fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.set_label('Diffused Charge', fontsize=label_size)
        
    return fig


def visualize_wireplane_labels(truth_results, simulation_params, figsize=(20, 12)):
    """
    Visualize wire plane labels with track IDs as colors.
    
    Parameters
    ----------
    truth_results : dict
        Dictionary of truth matching results, keyed by (side_idx, plane_idx).
        Each entry should contain 'wires', 'times', and 'tracks' arrays.
    simulation_params : dict
        Dictionary containing simulation parameters.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 12).
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    import numpy as np
    
    # Extract pre-calculated parameters
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']
    
    # Collect all track IDs to understand the full range
    all_track_ids = []
    for plane_key, result in truth_results.items():
        if 'tracks' in result and len(result['tracks']) > 0:
            all_track_ids.extend(result['tracks'])
    
    if len(all_track_ids) == 0:
        print("No track IDs found in truth results")
        return None
    
    all_track_ids = np.array(all_track_ids)
    unique_tracks = np.unique(all_track_ids)
    n_unique_tracks = len(unique_tracks)
    
    print(f"Found {n_unique_tracks} unique tracks: {unique_tracks[:10]}..." if n_unique_tracks > 10 else f"Found {n_unique_tracks} unique tracks: {unique_tracks}")
    
    # Create a colormap that distinguishes neighboring track IDs
    # Use a strategy that maximizes visual separation between consecutive IDs
    if n_unique_tracks <= 10:
        # Use distinct qualitative colors for small numbers
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_unique_tracks]
    elif n_unique_tracks <= 20:
        # Use tab20 for medium numbers
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_unique_tracks]
    else:
        # For large numbers, use a strategy that spreads colors maximally
        # Use HSV colorspace and distribute hues to maximize separation
        hues = np.linspace(0, 1, n_unique_tracks, endpoint=False)
        # Shuffle hues to avoid neighboring tracks having similar colors
        np.random.seed(42)  # Reproducible shuffling
        shuffled_indices = np.random.permutation(n_unique_tracks)
        hues = hues[shuffled_indices]
        
        # Create colors with fixed saturation and value for good contrast
        colors = []
        for i, hue in enumerate(hues):
            # Alternate saturation and value to increase distinction
            sat = 0.8 if i % 2 == 0 else 0.6
            val = 0.9 if i % 3 == 0 else 0.7
            colors.append(mcolors.hsv_to_rgb([hue, sat, val]))
        colors = np.array(colors)
    
    # Create a custom colormap
    cmap = ListedColormap(colors)
    
    # Create mapping from track ID to color index
    track_to_color_idx = {track_id: idx for idx, track_id in enumerate(unique_tracks)}
    
    # Find top 10 most frequent tracks for legend
    track_counts = np.bincount(all_track_ids)
    track_frequencies = [(track_id, track_counts[track_id]) for track_id in unique_tracks if track_id < len(track_counts)]
    track_frequencies.sort(key=lambda x: x[1], reverse=True)
    top_tracks = track_frequencies[:10]
    
    print(f"Top 10 tracks by frequency: {[(tid, count) for tid, count in top_tracks]}")
    
    # Create figure and plot
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor('black')
            ax.grid(False)
            
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
            plot_title = f"{side_names[side_idx]}\n{plane_types[plane_idx]}"
            
            plane_key = (side_idx, plane_idx)
            
            if plane_key not in truth_results or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(No truth data)", color='grey', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
                ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
                ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
                ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue
            
            result = truth_results[plane_key]
            if 'wires' not in result or len(result['wires']) == 0:
                ax.text(0.5, 0.5, "(No truth-matched points)", color='grey', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
                ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
                ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
                ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue
            
            # Get truth matching data for this plane
            wires = result['wires']
            times = result['times']
            tracks = result['tracks']
            
            # Convert time indices to actual time values
            time_values = times * time_step_size_us
            
            # Create color array for this plane's tracks
            color_indices = [track_to_color_idx.get(track_id, 0) for track_id in tracks]
            
            # Create scatter plot with track ID colors
            scatter = ax.scatter(wires, time_values, c=color_indices, cmap=cmap, 
                               s=1, alpha=0.8, vmin=0, vmax=n_unique_tracks-1)
            
            ax.set_ylim(0, max_time_axis)
            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_box_aspect(1)
            ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
            ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
            ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
            ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
            
            # Add text showing number of truth-matched points
            n_points = len(wires)
            n_tracks_plane = len(np.unique(tracks))
            ax.text(0.02, 0.98, f"{n_points:,} points\n{n_tracks_plane} tracks", 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
                   fontsize=10)
    
    # Add legend for top tracks
    legend_elements = []
    for i, (track_id, count) in enumerate(top_tracks):
        color_idx = track_to_color_idx[track_id]
        color = colors[color_idx]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8,
                                        label=f'Track {track_id} ({count:,} points)'))
    
    # Add legend outside the plot area
    if legend_elements:
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  title="Top 10 Tracks", fontsize=10)
    
    plt.suptitle('Wire Plane Truth Matching - Track ID Labels', fontsize=16, y=0.95)
    
    return fig


def visualize_by_index(wire_signals_dict, simulation_params, indices_list, figsize=(10, 8)):
    """
    Visualize wire signals at specific wire indices across time.
    
    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
    simulation_params : dict
        Dictionary containing simulation parameters.
    indices_list : list
        List of wire indices to plot.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 8).
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    # Extract pre-calculated parameters
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']
    
    # Create time axis
    time_axis = np.arange(num_time_steps) * time_step_size_us
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    
    for side_idx in range(2):
        for plane_idx in range(3):
            ax = axes[side_idx, plane_idx]
            
            plane_key = (side_idx, plane_idx)
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
            
            plot_title = f"{side_names[side_idx]}\n{plane_types[plane_idx]}"
            ax.set_title(plot_title, fontsize=12)
            
            if plane_key not in wire_signals_dict or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel('Time (μs)')
                ax.set_ylabel('Signal Strength')
                continue
            
            signal_data = np.array(wire_signals_dict[plane_key])
            
            # Plot each requested wire index
            for wire_idx in indices_list:
                # Convert absolute index to relative index
                rel_idx = wire_idx - min_idx_abs
                
                if 0 <= rel_idx < signal_data.shape[0]:
                    wire_signal = signal_data[rel_idx, :]
                    ax.plot(time_axis, wire_signal, label=f'Wire {wire_idx}', alpha=0.8)
                    
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel('Signal Strength')
            ax.grid(True, alpha=0.3)
            
            if len(indices_list) <= 10:  # Only show legend if not too many lines
                ax.legend(fontsize=8)
    
    plt.tight_layout()
    return fig