import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

def visualize_wire_signals(wire_signals_dict, simulation_params, figsize=(20, 10), log_norm=False):
    """
    Visualize wire signals stored in a dictionary, using different color schemes per plane type.

    Args:
        wire_signals_dict: Dictionary of wire signals, keyed by (side_idx, plane_idx)
        simulation_params: Dictionary containing simulation parameters
        figsize: Figure size (width, height) in inches
        log_norm: If True, use logarithmic normalization for all plots

    Returns:
        matplotlib Figure object
    """
    print("--- Starting Visualization ---")
    vis_start = time.time()

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

    vis_end = time.time()
    print(f"--- Visualization Finished ({vis_end - vis_start:.3f} s) ---")
    return fig


def visualize_single_plane(wire_signals_dict, simulation_params, side_idx=0, plane_idx=0, figsize=(10, 10), log_norm=False):
    """
    Visualize wire signals for a single side/plane combination using appropriate color scheme.

    Args:
        wire_signals_dict: Dictionary of wire signals, keyed by (side_idx, plane_idx)
        simulation_params: Dictionary containing simulation parameters
        side_idx: Index of the side to plot (0=West, 1=East)
        plane_idx: Index of the plane to plot (0=U, 1=V, 2=Y)
        figsize: Figure size (width, height) in inches
        log_norm: If True, use logarithmic normalization

    Returns:
        matplotlib Figure object
    """
    print(f"--- Starting Visualization for Side {side_idx}, Plane {plane_idx} ---")
    vis_start = time.time()

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
        ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
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

    vis_end = time.time()
    print(f"--- Visualization Finished ({vis_end - vis_start:.3f} s) ---")
    return fig

# def visualize_wire_signals(wire_signals_dict, simulation_params, figsize=(20, 10)):
#     """
#     Visualize wire signals stored in a dictionary.

#     Args:
#         wire_signals_dict: Dictionary of wire signals, keyed by (side_idx, plane_idx)
#         simulation_params: Dictionary containing simulation parameters
#         figsize: Figure size (width, height) in inches

#     Returns:
#         matplotlib Figure object
#     """
#     print("--- Starting Visualization (White Fig BG, Black Axes BG, Black Text) ---")
#     vis_start = time.time()

#     # Extract pre-calculated parameters
#     num_time_steps = simulation_params['num_time_steps']
#     time_step_size_us = simulation_params['time_step_size_us']
#     num_wires_actual = simulation_params['num_wires_actual']
#     max_abs_indices = simulation_params['max_abs_indices']
#     min_abs_indices = simulation_params['min_abs_indices']

#     side_names = ['West Side (x < 0)', 'East Side (x > 0)']
#     plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

#     # Find global signal range for normalization
#     global_min, global_max = float('inf'), 1e-9
#     for s in range(2):
#         for p in range(3):
#             if (s, p) in wire_signals_dict and num_wires_actual[s, p] > 0:
#                 signal_data = np.array(wire_signals_dict[(s, p)])
#                 if signal_data.size > 0:
#                     non_zero = signal_data[signal_data > 1e-9]
#                     if non_zero.size > 0:
#                         global_min = min(global_min, non_zero.min())
#                         global_max = max(global_max, non_zero.max())
#     if global_min == float('inf'): global_min = 1e-6
#     if global_max <= global_min: global_max = global_min * 10
#     print(f"   Visualization Norm: min={global_min:.2e}, max={global_max:.2e}")

#     # Create figure and plot
#     fig = plt.figure(figsize=figsize, facecolor='white')
#     gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
#     max_time_axis = num_time_steps * time_step_size_us
#     title_size, label_size, tick_size = 14, 12, 10

#     for side_idx in range(2):
#         for plane_idx in range(3):
#             ax = fig.add_subplot(gs[side_idx, plane_idx])
#             ax.set_facecolor('black')
#             ax.grid(False)
#             min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
#             max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
#             actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
#             plot_title = f"{side_names[side_idx]}\n{plane_types[plane_idx]}"

#             if (side_idx, plane_idx) not in wire_signals_dict or actual_wire_count == 0:
#                 ax.text(0.5, 0.5, "(0 wires active)", color='grey', ha='center', va='center', transform=ax.transAxes)
#                 ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
#                 ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
#                 ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
#                 ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
#                 ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
#                 ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')
#                 ax.set_xlim(min_idx_abs, max_idx_abs + 1)
#                 ax.set_ylim(0, max_time_axis)
#                 ax.set_box_aspect(1)
#                 continue

#             signal_data_to_plot = np.array(wire_signals_dict[(side_idx, plane_idx)])
#             extent_xmin = min_idx_abs
#             extent_xmax = max_idx_abs + 1
#             extent = [extent_xmin, extent_xmax, 0, max_time_axis]

#             im = ax.imshow(
#                 signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
#                 cmap='inferno', norm=LogNorm(vmin=global_min, vmax=global_max, clip=True)
#             )
#             ax.set_ylim(0, max_time_axis)
#             ax.set_xlim(extent_xmin, extent_xmax)
#             ax.set_box_aspect(1)
#             ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
#             ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
#             ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
#             ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
#             ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
#             ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')

#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes('right', size='4%', pad=0.08)
#             cbar = fig.colorbar(im, cax=cax)
#             cbar.ax.tick_params(labelsize=tick_size, colors='black')
#             cbar.set_label('Signal Strength', fontsize=label_size, color='black')
#             cbar.outline.set_edgecolor('white')

#     vis_end = time.time()
#     print(f"--- Visualization Finished ({vis_end - vis_start:.3f} s) ---")
#     return fig


# def visualize_single_plane(wire_signals_dict, simulation_params, side_idx=0, plane_idx=0, figsize=(10, 10)):
#     """
#     Visualize wire signals for a single side/plane combination.

#     Args:
#         wire_signals_dict: Dictionary of wire signals, keyed by (side_idx, plane_idx)
#         simulation_params: Dictionary containing simulation parameters
#         side_idx: Index of the side to plot (0=West, 1=East)
#         plane_idx: Index of the plane to plot (0=U, 1=V, 2=Y)
#         figsize: Figure size (width, height) in inches

#     Returns:
#         matplotlib Figure object
#     """
#     print(f"--- Starting Visualization for Side {side_idx}, Plane {plane_idx} (White Fig BG, Black Axes BG, Black Text) ---")
#     vis_start = time.time()

#     # Extract pre-calculated parameters
#     num_time_steps = simulation_params['num_time_steps']
#     time_step_size_us = simulation_params['time_step_size_us']
#     num_wires_actual = simulation_params['num_wires_actual']
#     max_abs_indices = simulation_params['max_abs_indices']
#     min_abs_indices = simulation_params['min_abs_indices']

#     side_names = ['West Side (x < 0)', 'East Side (x > 0)']
#     plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

#     # Find signal range for normalization (using only the selected side/plane)
#     s, p = side_idx, plane_idx
#     global_min, global_max = float('inf'), 1e-9

#     if (s, p) in wire_signals_dict and num_wires_actual[s, p] > 0:
#         signal_data = np.array(wire_signals_dict[(s, p)])
#         if signal_data.size > 0:
#             non_zero = signal_data[signal_data > 1e-9]
#             if non_zero.size > 0:
#                 global_min = non_zero.min()
#                 global_max = non_zero.max()
#     if global_min == float('inf'): global_min = 1e-6
#     if global_max <= global_min: global_max = global_min * 10
#     print(f"   Visualization Norm: min={global_min:.2e}, max={global_max:.2e}")

#     # Create figure and plot
#     fig = plt.figure(figsize=figsize, facecolor='white')
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_facecolor('black')
#     ax.grid(False)
#     max_time_axis = num_time_steps * time_step_size_us
#     title_size, label_size, tick_size = 14, 12, 10

#     min_idx_abs = int(min_abs_indices[s, p])
#     max_idx_abs = int(max_abs_indices[s, p])
#     actual_wire_count = int(num_wires_actual[s, p])
#     plot_title = f"{side_names[s]}\n{plane_types[p]}"

#     if (s, p) not in wire_signals_dict or actual_wire_count == 0:
#         ax.text(0.5, 0.5, "(0 wires active)", color='grey', ha='center', va='center', transform=ax.transAxes)
#         ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
#         ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
#         ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
#         ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
#         ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
#         ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')
#         ax.set_xlim(min_idx_abs, max_idx_abs + 1)
#         ax.set_ylim(0, max_time_axis)
#         ax.set_box_aspect(1)
#     else:
#         signal_data_to_plot = np.array(wire_signals_dict[(s, p)])
#         extent_xmin = min_idx_abs
#         extent_xmax = max_idx_abs + 1
#         extent = [extent_xmin, extent_xmax, 0, max_time_axis]

#         im = ax.imshow(
#             signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
#             cmap='inferno', norm=LogNorm(vmin=global_min, vmax=global_max, clip=True)
#         )
#         ax.set_ylim(0, max_time_axis)
#         ax.set_xlim(extent_xmin, extent_xmax)
#         ax.set_box_aspect(1)
#         ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
#         ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
#         ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
#         ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
#         ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('white')
#         ax.spines['left'].set_color('white'); ax.spines['right'].set_color('white')

#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes('right', size='4%', pad=0.08)
#         cbar = fig.colorbar(im, cax=cax)
#         cbar.ax.tick_params(labelsize=tick_size, colors='black')
#         cbar.set_label('Signal Strength', fontsize=label_size, color='black')
#         cbar.outline.set_edgecolor('white')

#     vis_end = time.time()
#     print(f"--- Visualization Finished ({vis_end - vis_start:.3f} s) ---")
#     return fig