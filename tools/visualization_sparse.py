"""
Sparse visualization functions for wire signals.

These functions work directly with sparse data format (indices, values)
without converting to dense arrays, providing memory-efficient visualization.

Sparse format:
    - indices: (N, 2) int32 array with [wire_idx, time_idx] per row (relative indices)
    - values: (N,) float32 array with signal values
"""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_wire_signals(wire_signals_dict, simulation_params, figsize=(20, 10),
                           log_norm=False, sparse_data=False, point_size=0.1):
    """
    Visualize wire signals stored in a dictionary, using different color schemes per plane type.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
        If sparse_data=False: values are dense (num_wires, num_time_steps) arrays.
        If sparse_data=True: values are tuples (indices, values) where
            indices is (N, 2) with [wire_idx, time_idx] (relative) and values is (N,).
    simulation_params : dict
        Dictionary containing simulation parameters.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization for all plots, by default False.
    sparse_data : bool, optional
        If True, expect sparse format (indices, values). If False, expect dense arrays.
    point_size : float, optional
        Size of scatter points when using sparse_data=True, by default 0.1.

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
        (0, 0): 'U-plane', (0, 1): 'V-plane', (0, 2): 'Y-plane',
        (1, 0): 'U-plane', (1, 1): 'V-plane', (1, 2): 'Y-plane',
    }

    # Define colormap settings
    cmap_settings = {
        'U-plane': {'cmap': 'seismic'},
        'V-plane': {'cmap': 'seismic'},
        'Y-plane': {'cmap': 'seismic'}
    }

    # Find min/max values for each plane type
    plane_min_max = {
        'U-plane': {'min': float('inf'), 'max': -float('inf')},
        'V-plane': {'min': float('inf'), 'max': -float('inf')},
        'Y-plane': {'min': float('inf'), 'max': -float('inf')}
    }

    # Pre-convert all arrays once to avoid repeated device-to-host transfers
    converted_signals = {}
    for key, signal_data in wire_signals_dict.items():
        if sparse_data:
            indices, values = signal_data
            converted_signals[key] = (np.asarray(indices), np.asarray(values))
        else:
            converted_signals[key] = np.asarray(signal_data)

    # Calculate min/max for each plane type using pre-converted arrays
    for s in range(2):
        for p in range(3):
            if (s, p) in converted_signals and num_wires_actual[s, p] > 0:
                plane_name = plane_name_mapping[(s, p)]
                if sparse_data:
                    indices_np, values_np = converted_signals[(s, p)]
                    if len(values_np) > 0:
                        plane_min_max[plane_name]['min'] = min(plane_min_max[plane_name]['min'], values_np.min())
                        plane_min_max[plane_name]['max'] = max(plane_min_max[plane_name]['max'], values_np.max())
                else:
                    signal_data = converted_signals[(s, p)]
                    if signal_data.size > 0:
                        plane_min_max[plane_name]['min'] = min(plane_min_max[plane_name]['min'], signal_data.min())
                        plane_min_max[plane_name]['max'] = max(plane_min_max[plane_name]['max'], signal_data.max())

    # Set fixed ranges if no data found - ensure symmetric range around zero
    for plane_name in plane_min_max:
        if plane_min_max[plane_name]['min'] == float('inf'):
            plane_min_max[plane_name]['min'] = -25
            plane_min_max[plane_name]['max'] = 25
        else:
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

            if (side_idx, plane_idx) not in converted_signals or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", color='grey', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
                ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
                ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
                ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue

            # Get colormap settings
            cmap = cmap_settings[plane_name]['cmap']
            vmin = plane_min_max[plane_name]['min']
            vmax = plane_min_max[plane_name]['max']

            if sparse_data:
                # Sparse visualization using scatter plot (use pre-converted arrays)
                indices_np, values_np = converted_signals[(side_idx, plane_idx)]
                if len(values_np) == 0:
                    ax.text(0.5, 0.5, "(No data)", color='grey', ha='center', va='center', transform=ax.transAxes)
                else:

                    # Convert relative wire indices to absolute
                    wire_abs = indices_np[:, 0] + min_idx_abs
                    time_us = indices_np[:, 1] * time_step_size_us

                    # Create normalization
                    if log_norm:
                        max_abs_val = max(abs(vmin), abs(vmax))
                        linthresh = max(1e-8, 0.015 * max_abs_val)
                        norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, clip=True)
                    else:
                        norm = Normalize(vmin=vmin, vmax=vmax)

                    sc = ax.scatter(wire_abs, time_us, c=values_np, cmap=cmap, norm=norm,
                                    s=point_size, marker='s', linewidths=0)

                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='4%', pad=0.08)
                    cbar = fig.colorbar(sc, cax=cax)
                    cbar.ax.tick_params(labelsize=tick_size, colors='black')
                    cbar.set_label('Signal Strength', fontsize=label_size, color='black')
            else:
                # Dense visualization using imshow (use pre-converted arrays)
                signal_data_to_plot = converted_signals[(side_idx, plane_idx)]
                extent_xmin = min_idx_abs
                extent_xmax = max_idx_abs + 1
                extent = [extent_xmin, extent_xmax, 0, max_time_axis]

                if log_norm:
                    max_abs_val = max(abs(vmin), abs(vmax))
                    linthresh = max(1e-8, 0.015 * max_abs_val)
                    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, clip=True)
                    im = ax.imshow(signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                                   cmap=cmap, norm=norm)
                else:
                    im = ax.imshow(signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                                   cmap=cmap, vmin=vmin, vmax=vmax)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='4%', pad=0.08)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=tick_size, colors='black')
                cbar.set_label('Signal Strength', fontsize=label_size, color='black')

            ax.set_ylim(0, max_time_axis)
            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_box_aspect(1)
            ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
            ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
            ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
            ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')

    return fig


def visualize_single_plane(wire_signals_dict, simulation_params, side_idx=0, plane_idx=0,
                           figsize=(10, 10), log_norm=False, sparse_data=False, point_size=0.5):
    """
    Visualize wire signals for a single side/plane combination using appropriate color scheme.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
        If sparse_data=False: values are dense arrays.
        If sparse_data=True: values are tuples (indices, values).
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
    sparse_data : bool, optional
        If True, expect sparse format (indices, values).
    point_size : float, optional
        Size of scatter points when using sparse_data=True, by default 0.5.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    print(f"--- Starting Visualization for Side {side_idx}, Plane {plane_idx} ---")

    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    plane_name_mapping = {
        (0, 0): 'U-plane', (0, 1): 'V-plane', (0, 2): 'Y-plane',
        (1, 0): 'U-plane', (1, 1): 'V-plane', (1, 2): 'Y-plane',
    }

    cmap_settings = {
        'U-plane': {'cmap': 'seismic'},
        'V-plane': {'cmap': 'seismic'},
        'Y-plane': {'cmap': 'seismic'}
    }

    s, p = side_idx, plane_idx
    plane_name = plane_name_mapping[(s, p)]

    # Find min/max across both sides for the same plane type
    min_val, max_val = float('inf'), -float('inf')

    for check_side in range(2):
        check_key = (check_side, p)
        if check_key in wire_signals_dict and num_wires_actual[check_side, p] > 0:
            if sparse_data:
                indices, values = wire_signals_dict[check_key]
                if len(values) > 0:
                    values_np = np.array(values)
                    min_val = min(min_val, values_np.min())
                    max_val = max(max_val, values_np.max())
            else:
                signal_data = np.array(wire_signals_dict[check_key])
                if signal_data.size > 0:
                    min_val = min(min_val, signal_data.min())
                    max_val = max(max_val, signal_data.max())

    if min_val == float('inf'):
        min_val, max_val = -25, 25
    else:
        max_abs_val = max(abs(min_val), abs(max_val))
        min_val, max_val = -max_abs_val, max_abs_val

    print(f"   Visualization Norm for {plane_name}: min={min_val:.2e}, max={max_val:.2e}")

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
        return fig

    cmap = cmap_settings[plane_name]['cmap']
    vmin, vmax = min_val, max_val

    if sparse_data:
        indices, values = wire_signals_dict[(s, p)]
        if len(values) == 0:
            ax.text(0.5, 0.5, "(No data)", color='grey', ha='center', va='center', transform=ax.transAxes)
        else:
            indices_np = np.array(indices)
            values_np = np.array(values)

            wire_abs = indices_np[:, 0] + min_idx_abs
            time_us = indices_np[:, 1] * time_step_size_us

            if log_norm:
                max_abs_val = max(abs(vmin), abs(vmax))
                linthresh = max(1e-8, 0.01 * max_abs_val)
                norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

            sc = ax.scatter(wire_abs, time_us, c=values_np, cmap=cmap, norm=norm,
                            s=point_size, marker='s', linewidths=0)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.08)
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.tick_params(labelsize=tick_size, colors='black')
            cbar.set_label('Signal Strength', fontsize=label_size, color='black')
            cbar.outline.set_edgecolor('white')
    else:
        signal_data_to_plot = np.array(wire_signals_dict[(s, p)])
        extent_xmin = min_idx_abs
        extent_xmax = max_idx_abs + 1
        extent = [extent_xmin, extent_xmax, 0, max_time_axis]

        if log_norm:
            max_abs_val = max(abs(vmin), abs(vmax))
            linthresh = max(1e-8, 0.01 * max_abs_val)
            norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, clip=True)
            im = ax.imshow(signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                           cmap=cmap, norm=norm)
        else:
            im = ax.imshow(signal_data_to_plot.T, aspect='auto', origin='lower', extent=extent,
                           cmap=cmap, vmin=vmin, vmax=vmax)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=tick_size, colors='black')
        cbar.set_label('Signal Strength', fontsize=label_size, color='black')
        cbar.outline.set_edgecolor('white')

    ax.set_ylim(0, max_time_axis)
    ax.set_xlim(min_idx_abs, max_idx_abs + 1)
    ax.set_box_aspect(1)
    ax.set_title(plot_title, fontsize=title_size, pad=10, color='black')
    ax.set_xlabel('Absolute Wire Index', fontsize=label_size, color='black')
    ax.set_ylabel('Time (μs)', fontsize=label_size, color='black')
    ax.tick_params(axis='both', which='major', labelsize=tick_size, colors='black')

    return fig


def visualize_diffused_charge(wire_signals_dict, simulation_params, figsize=(20, 10),
                              log_norm=False, threshold=100, sparse_data=False, point_size=0.5):
    """
    Visualize diffused charge (hit signals) with proper scaling.

    Uses YlOrRd colormap with dark background and threshold masking for
    better visualization of charge deposits.

    Parameters
    ----------
    wire_signals_dict : dict
        Dictionary of wire signals, keyed by (side_idx, plane_idx).
        If sparse_data=False: values are dense arrays.
        If sparse_data=True: values are tuples (indices, values).
    simulation_params : dict
        Dictionary containing simulation parameters.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 10).
    log_norm : bool, optional
        If True, use logarithmic normalization for all plots, by default False.
    threshold : float, optional
        Values below this threshold are masked/hidden, by default 100.
    sparse_data : bool, optional
        If True, expect sparse format (indices, values).
    point_size : float, optional
        Size of scatter points when using sparse_data=True, by default 0.5.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    # Find global min/max
    global_min, global_max = float('inf'), -float('inf')
    all_values = []

    for s in range(2):
        for p in range(3):
            if (s, p) in wire_signals_dict and num_wires_actual[s, p] > 0:
                if sparse_data:
                    indices, values = wire_signals_dict[(s, p)]
                    if len(values) > 0:
                        values_np = np.array(values)
                        valid_data = values_np[values_np > threshold]
                        if len(valid_data) > 0:
                            global_min = min(global_min, valid_data.min())
                            global_max = max(global_max, valid_data.max())
                            all_values.append(valid_data)
                else:
                    signal_data = np.array(wire_signals_dict[(s, p)])
                    if signal_data.size > 0:
                        valid_data = signal_data[signal_data > threshold]
                        if valid_data.size > 0:
                            global_min = min(global_min, valid_data.min())
                            global_max = max(global_max, valid_data.max())
                            all_values.append(valid_data.flatten())

    if global_min == float('inf'):
        global_min, global_max = threshold, threshold * 10
    elif all_values:
        all_values_concat = np.concatenate(all_values)
        if len(all_values_concat) > 0:
            p1, p99 = np.percentile(all_values_concat, [1, 99])
            global_min = max(global_min, p1)
            global_max = min(global_max, p99)

    print(f"   Diffused Charge Range: min={global_min:.2e}, max={global_max:.2e}")
    background_color = '#1a1a1a'
    colormap_name = 'YlOrRd'

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor(background_color)
            ax.grid(True, alpha=0.3, color='#505050', linestyle='--', linewidth=0.5)

            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            actual_wire_count = int(num_wires_actual[side_idx, plane_idx])
            plot_title = f"{side_names[side_idx]}\n{plane_types[plane_idx]}"

            if (side_idx, plane_idx) not in wire_signals_dict or actual_wire_count == 0:
                ax.text(0.5, 0.5, "(0 wires active)", color='gray', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(plot_title, fontsize=14, pad=10)
                ax.set_xlim(min_idx_abs, max_idx_abs + 1)
                ax.set_ylim(0, max_time_axis)
                ax.set_box_aspect(1)
                continue

            cmap = plt.cm.get_cmap(colormap_name).copy()
            vmin_plot = max(threshold, global_min)
            vmax_plot = global_max

            if sparse_data:
                indices, values = wire_signals_dict[(side_idx, plane_idx)]
                if len(values) == 0:
                    ax.text(0.5, 0.5, "(No data)", color='gray', ha='center', va='center', transform=ax.transAxes)
                else:
                    indices_np = np.array(indices)
                    values_np = np.array(values)

                    # Apply threshold filter
                    mask = values_np > threshold
                    if np.sum(mask) == 0:
                        ax.text(0.5, 0.5, "(Below threshold)", color='gray', ha='center', va='center', transform=ax.transAxes)
                    else:
                        wire_abs = indices_np[mask, 0] + min_idx_abs
                        time_us = indices_np[mask, 1] * time_step_size_us
                        filtered_values = values_np[mask]

                        if log_norm:
                            norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
                        else:
                            norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)

                        sc = ax.scatter(wire_abs, time_us, c=filtered_values, cmap=cmap, norm=norm,
                                        s=point_size, marker='s', linewidths=0)

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='4%', pad=0.08)
                        cbar = fig.colorbar(sc, cax=cax)
                        cbar.set_label('Diffused Charge', fontsize=12)
            else:
                signal_data_to_plot = np.array(wire_signals_dict[(side_idx, plane_idx)])
                extent = [min_idx_abs, max_idx_abs + 1, 0, max_time_axis]
                masked_data = np.ma.masked_where(signal_data_to_plot.T <= threshold, signal_data_to_plot.T)

                cmap.set_bad(background_color)
                cmap.set_under(background_color)

                if log_norm:
                    norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
                    im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                                   cmap=cmap, norm=norm, interpolation='nearest')
                else:
                    im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                                   cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, interpolation='nearest')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='4%', pad=0.08)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Diffused Charge', fontsize=12)

            ax.set_ylim(0, max_time_axis)
            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_box_aspect(1)
            ax.set_title(plot_title, fontsize=14, pad=10)
            ax.set_xlabel('Absolute Wire Index', fontsize=12)
            ax.set_ylabel('Time (μs)', fontsize=12)

    return fig


def visualize_diffused_charge_single_plane(wire_signals_dict, simulation_params, side_idx=0, plane_idx=0,
                                           figsize=(10, 10), log_norm=False, threshold=100,
                                           sparse_data=False, point_size=1.0):
    """
    Visualize diffused charge for a single side/plane combination.

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
    threshold : float, optional
        Values below this threshold are masked/hidden, by default 100.
    sparse_data : bool, optional
        If True, expect sparse format (indices, values).
    point_size : float, optional
        Size of scatter points when using sparse_data=True, by default 1.0.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    print(f"--- Visualizing Diffused Charge for Side {side_idx}, Plane {plane_idx} ---")

    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    s, p = side_idx, plane_idx
    min_val, max_val = float('inf'), -float('inf')

    for check_side in range(2):
        check_key = (check_side, p)
        if check_key in wire_signals_dict and num_wires_actual[check_side, p] > 0:
            if sparse_data:
                indices, values = wire_signals_dict[check_key]
                if len(values) > 0:
                    values_np = np.array(values)
                    valid_data = values_np[values_np > threshold]
                    if len(valid_data) > 0:
                        min_val = min(min_val, valid_data.min())
                        max_val = max(max_val, valid_data.max())
            else:
                signal_data = np.array(wire_signals_dict[check_key])
                if signal_data.size > 0:
                    valid_data = signal_data[signal_data > threshold]
                    if valid_data.size > 0:
                        min_val = min(min_val, valid_data.min())
                        max_val = max(max_val, valid_data.max())

    if min_val == float('inf'):
        min_val, max_val = threshold, threshold * 10

    print(f"   Visualization Range: min={min_val:.2e}, max={max_val:.2e}")

    background_color = '#1a1a1a'
    colormap_name = 'YlOrRd'

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(background_color)
    ax.grid(True, alpha=0.3, color='#505050', linestyle='--', linewidth=0.5)
    max_time_axis = num_time_steps * time_step_size_us

    min_idx_abs = int(min_abs_indices[s, p])
    max_idx_abs = int(max_abs_indices[s, p])
    actual_wire_count = int(num_wires_actual[s, p])
    plot_title = f"{side_names[s]}\n{plane_types[p]}"

    if (s, p) not in wire_signals_dict or actual_wire_count == 0:
        ax.text(0.5, 0.5, "(0 wires active)", color='gray', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(plot_title, fontsize=14, pad=10)
        ax.set_xlabel('Absolute Wire Index', fontsize=12)
        ax.set_ylabel('Time (μs)', fontsize=12)
        ax.set_xlim(min_idx_abs, max_idx_abs + 1)
        ax.set_ylim(0, max_time_axis)
        ax.set_box_aspect(1)
        return fig

    cmap = plt.cm.get_cmap(colormap_name).copy()
    vmin_plot = max(threshold, min_val)
    vmax_plot = max_val

    if sparse_data:
        indices, values = wire_signals_dict[(s, p)]
        if len(values) == 0:
            ax.text(0.5, 0.5, "(No data)", color='gray', ha='center', va='center', transform=ax.transAxes)
        else:
            indices_np = np.array(indices)
            values_np = np.array(values)

            mask = values_np > threshold
            if np.sum(mask) == 0:
                ax.text(0.5, 0.5, "(Below threshold)", color='gray', ha='center', va='center', transform=ax.transAxes)
            else:
                wire_abs = indices_np[mask, 0] + min_idx_abs
                time_us = indices_np[mask, 1] * time_step_size_us
                filtered_values = values_np[mask]

                if log_norm:
                    norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
                else:
                    norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)

                sc = ax.scatter(wire_abs, time_us, c=filtered_values, cmap=cmap, norm=norm,
                                s=point_size, marker='s', linewidths=0)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='4%', pad=0.08)
                cbar = fig.colorbar(sc, cax=cax)
                cbar.set_label('Diffused Charge', fontsize=12)
    else:
        signal_data_to_plot = np.array(wire_signals_dict[(s, p)])
        extent = [min_idx_abs, max_idx_abs + 1, 0, max_time_axis]
        masked_data = np.ma.masked_where(signal_data_to_plot.T <= threshold, signal_data_to_plot.T)

        cmap.set_bad(background_color)
        cmap.set_under(background_color)

        if log_norm:
            norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot, clip=True)
            im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                           cmap=cmap, norm=norm, interpolation='nearest')
        else:
            im = ax.imshow(masked_data, aspect='auto', origin='lower', extent=extent,
                           cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, interpolation='nearest')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Diffused Charge', fontsize=12)

    ax.set_ylim(0, max_time_axis)
    ax.set_xlim(min_idx_abs, max_idx_abs + 1)
    ax.set_box_aspect(1)
    ax.set_title(plot_title, fontsize=14, pad=10)
    ax.set_xlabel('Absolute Wire Index', fontsize=12)
    ax.set_ylabel('Time (μs)', fontsize=12)

    return fig


def get_top_tracks_by_charge(track_hits_dict, top_n=20):
    """
    Find top tracks by total charge across all planes.

    Parameters
    ----------
    track_hits_dict : dict
        Dictionary of track hits results, keyed by (side_idx, plane_idx).
        Each entry should contain 'num_labeled' and 'labeled_hits' arrays.
    top_n : int, optional
        Number of top tracks to return, by default 20.

    Returns
    -------
    list
        List of tuples (track_id, total_charge) sorted by charge descending.
    """
    all_track_ids = []
    all_charges = []

    for plane_key, results in track_hits_dict.items():
        num_labeled = int(results['num_labeled'])
        if num_labeled > 0:
            labeled = results['labeled_hits'][:num_labeled]
            all_track_ids.append(jnp.asarray(labeled[:, 0], dtype=jnp.int32))
            all_charges.append(jnp.asarray(labeled[:, 3]))

    if not all_track_ids:
        return []

    all_track_ids = jnp.concatenate(all_track_ids)
    all_charges = jnp.concatenate(all_charges)

    sort_idx = jnp.argsort(all_track_ids)
    sorted_ids = all_track_ids[sort_idx]
    sorted_charges = all_charges[sort_idx]

    is_new_track = jnp.concatenate([jnp.array([True]), sorted_ids[1:] != sorted_ids[:-1]])
    unique_indices = jnp.where(is_new_track)[0]
    unique_tracks = sorted_ids[unique_indices]

    segment_ids = jnp.cumsum(is_new_track) - 1
    track_totals = jax.ops.segment_sum(sorted_charges, segment_ids, num_segments=len(unique_indices))

    top_indices = jnp.argsort(track_totals)[-top_n:][::-1]
    return [(int(unique_tracks[i]), float(track_totals[i])) for i in top_indices]


def visualize_track_labels(track_hits_dict, simulation_params, top_tracks_by_charge,
                           max_tracks=15, figsize=(20, 12)):
    """
    Visualize track labels with distinct colors for top tracks by charge.

    Uses hash-based coloring for non-top tracks to ensure visual distinction.

    Parameters
    ----------
    track_hits_dict : dict
        Dictionary of track hits results, keyed by (side_idx, plane_idx).
        Each entry should contain 'num_labeled' and 'labeled_hits' arrays.
    simulation_params : dict
        Dictionary containing simulation parameters.
    top_tracks_by_charge : list
        List of tuples (track_id, total_charge) from get_top_tracks_by_charge.
    max_tracks : int, optional
        Maximum number of tracks to show in legend, by default 15.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (20, 12).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30, width_ratios=[1, 1, 1, 0.12])
    max_time_axis = num_time_steps * time_step_size_us

    distinct_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF',
                       '#FFD700', '#FF8C00', '#8B008B', '#228B22', '#4B0082',
                       '#FF1493', '#00CED1', '#FF4500', '#9400D3', '#32CD32',
                       '#8B4513', '#20B2AA', '#FF69B4', '#4169E1', '#DC143C']
    distinct_colors_rgba = [mcolors.to_rgba(c) for c in distinct_colors]

    top_tracks = [tid for tid, _ in top_tracks_by_charge[:max_tracks]]
    top_track_to_color = {tid: distinct_colors_rgba[i] for i, tid in enumerate(top_tracks[:len(distinct_colors)])}
    cmap = plt.cm.hsv

    def get_track_colors_vectorized(track_ids):
        track_ids = np.asarray(track_ids, dtype=np.int64)
        colors = np.zeros((len(track_ids), 4))
        hash_values = (track_ids * 2654435761) % 2**32
        colors[:] = cmap(hash_values / (2**32 - 1))
        for tid, color in top_track_to_color.items():
            colors[track_ids == tid] = color
        return colors

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor('black')

            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])

            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_ylim(0, max_time_axis)
            ax.set_box_aspect(1)
            ax.set_title(f"{side_names[side_idx]}\n{plane_types[plane_idx]}", fontsize=14, pad=10)
            ax.set_xlabel('Absolute Wire Index', fontsize=12)
            ax.set_ylabel('Time (μs)', fontsize=12)

            plane_key = (side_idx, plane_idx)
            results = track_hits_dict[plane_key]
            num_labeled = int(results['num_labeled'])

            if num_labeled > 0:
                labeled = np.array(results['labeled_hits'][:num_labeled])
                tracks = labeled[:, 0].astype(np.int64)
                wires = labeled[:, 1]
                times = labeled[:, 2] * time_step_size_us

                colors = get_track_colors_vectorized(tracks)
                ax.scatter(wires, times, c=colors, s=0.5, alpha=0.8)
                ax.text(0.02, 0.98, f"{num_labeled:,} hits\n{len(np.unique(tracks))} tracks",
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
            else:
                ax.text(0.5, 0.5, "(No labeled hits)", color='grey', ha='center', va='center', transform=ax.transAxes)

    # Add colorbar
    if top_tracks:
        cbar_ax = fig.add_subplot(gs[:, 3])
        n_show = min(len(top_tracks), max_tracks)
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, n_show)
        for i, tid in enumerate(top_tracks[:n_show]):
            color = top_track_to_color.get(tid, cmap((tid * 2654435761 % 2**32) / (2**32 - 1)))
            y_pos = n_show - 1 - i
            rect = plt.Rectangle((0, y_pos), 0.4, 0.9, facecolor=color, edgecolor='black', linewidth=0.5)
            cbar_ax.add_patch(rect)
            cbar_ax.text(0.5, y_pos + 0.45, f'Track {tid}', ha='left', va='center', fontsize=8)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        cbar_ax.set_title(f'Top {n_show} Tracks\n(by total charge)', fontsize=11, pad=10)
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


def visualize_track_labels_single_plane(track_hits_dict, simulation_params, top_tracks_by_charge,
                                        side_idx=0, plane_idx=0, max_tracks=15, figsize=(12, 10)):
    """
    Visualize track labels for a single side/plane with distinct colors for top tracks.

    Parameters
    ----------
    track_hits_dict : dict
        Dictionary of track hits results, keyed by (side_idx, plane_idx).
    simulation_params : dict
        Dictionary containing simulation parameters.
    top_tracks_by_charge : list
        List of tuples (track_id, total_charge) from get_top_tracks_by_charge.
    side_idx : int, optional
        Index of the side to plot (0=West, 1=East), by default 0.
    plane_idx : int, optional
        Index of the plane to plot (0=U, 1=V, 2=Y), by default 0.
    max_tracks : int, optional
        Maximum number of tracks to show in legend, by default 15.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (12, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    print(f"--- Visualizing Track Labels for Side {side_idx}, Plane {plane_idx} ---")

    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, width_ratios=[1, 0.12])
    max_time_axis = num_time_steps * time_step_size_us

    distinct_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF',
                       '#FFD700', '#FF8C00', '#8B008B', '#228B22', '#4B0082',
                       '#FF1493', '#00CED1', '#FF4500', '#9400D3', '#32CD32',
                       '#8B4513', '#20B2AA', '#FF69B4', '#4169E1', '#DC143C']
    distinct_colors_rgba = [mcolors.to_rgba(c) for c in distinct_colors]

    top_tracks = [tid for tid, _ in top_tracks_by_charge[:max_tracks]]
    top_track_to_color = {tid: distinct_colors_rgba[i] for i, tid in enumerate(top_tracks[:len(distinct_colors)])}
    cmap = plt.cm.hsv

    def get_track_colors_vectorized(track_ids):
        track_ids = np.asarray(track_ids, dtype=np.int64)
        colors = np.zeros((len(track_ids), 4))
        hash_values = (track_ids * 2654435761) % 2**32
        colors[:] = cmap(hash_values / (2**32 - 1))
        for tid, color in top_track_to_color.items():
            colors[track_ids == tid] = color
        return colors

    s, p = side_idx, plane_idx
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor('black')

    min_idx_abs = int(min_abs_indices[s, p])
    max_idx_abs = int(max_abs_indices[s, p])

    ax.set_xlim(min_idx_abs, max_idx_abs + 1)
    ax.set_ylim(0, max_time_axis)
    ax.set_box_aspect(1)
    ax.set_title(f"{side_names[s]}\n{plane_types[p]}", fontsize=14, pad=10)
    ax.set_xlabel('Absolute Wire Index', fontsize=12)
    ax.set_ylabel('Time (μs)', fontsize=12)

    plane_key = (s, p)
    results = track_hits_dict[plane_key]
    num_labeled = int(results['num_labeled'])

    if num_labeled > 0:
        labeled = np.array(results['labeled_hits'][:num_labeled])
        tracks = labeled[:, 0].astype(np.int64)
        wires = labeled[:, 1]
        times = labeled[:, 2] * time_step_size_us

        colors = get_track_colors_vectorized(tracks)
        ax.scatter(wires, times, c=colors, s=0.5, alpha=0.8)
        ax.text(0.02, 0.98, f"{num_labeled:,} hits\n{len(np.unique(tracks))} tracks",
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        print(f"   {num_labeled:,} labeled hits, {len(np.unique(tracks))} unique tracks")
    else:
        ax.text(0.5, 0.5, "(No labeled hits)", color='grey', ha='center', va='center', transform=ax.transAxes)
        print("   No labeled hits found")

    # Add colorbar/legend
    if top_tracks:
        cbar_ax = fig.add_subplot(gs[0, 1])
        n_show = min(len(top_tracks), max_tracks)
        cbar_ax.set_xlim(0, 1)
        cbar_ax.set_ylim(0, n_show)
        for i, tid in enumerate(top_tracks[:n_show]):
            color = top_track_to_color.get(tid, cmap((tid * 2654435761 % 2**32) / (2**32 - 1)))
            y_pos = n_show - 1 - i
            rect = plt.Rectangle((0, y_pos), 0.4, 0.9, facecolor=color, edgecolor='black', linewidth=0.5)
            cbar_ax.add_patch(rect)
            cbar_ax.text(0.5, y_pos + 0.45, f'Track {tid}', ha='left', va='center', fontsize=8)
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])
        cbar_ax.set_title(f'Top {n_show} Tracks\n(by total charge)', fontsize=11, pad=10)
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


def visualize_active_buckets(response_signals, simulation_params, figsize=(20, 10)):
    """
    Visualize active buckets in the bucketed simulation output.

    Active buckets are shown as colored rectangles, empty buckets as dark background.
    Shows the number of active buckets and coverage percentage for each plane.

    Parameters
    ----------
    response_signals : dict
        Dictionary of response signals from bucketed simulation.
        Each entry is (buckets, num_active, compact_to_key, B1, B2).
    simulation_params : dict
        Dictionary containing simulation parameters.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['East Side (x < 0)', 'West Side (x >= 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    max_time_axis = num_time_steps * time_step_size_us
    title_size, label_size, tick_size = 14, 12, 10

    for side_idx in range(2):
        for plane_idx in range(3):
            ax = fig.add_subplot(gs[side_idx, plane_idx])
            ax.set_facecolor('#1a1a1a')

            plane_key = (side_idx, plane_idx)
            min_idx_abs = int(min_abs_indices[side_idx, plane_idx])
            max_idx_abs = int(max_abs_indices[side_idx, plane_idx])
            num_wires = int(num_wires_actual[side_idx, plane_idx])

            if plane_key not in response_signals:
                ax.text(0.5, 0.5, "(No data)", color='grey', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{side_names[side_idx]}\n{plane_types[plane_idx]}", fontsize=title_size, pad=10)
                continue

            buckets, num_active, compact_to_key, B1, B2 = response_signals[plane_key]
            num_active_int = int(num_active)
            B1_int = int(B1)
            B2_int = int(B2)

            num_buckets_w = (num_wires + B1_int - 1) // B1_int
            num_buckets_t = (num_time_steps + B2_int - 1) // B2_int
            total_buckets = num_buckets_w * num_buckets_t
            coverage_pct = (num_active_int / total_buckets) * 100 if total_buckets > 0 else 0

            active_keys = np.array(compact_to_key[:num_active_int])
            bucket_w_indices = active_keys // num_buckets_t
            bucket_t_indices = active_keys % num_buckets_t

            bucket_grid = np.zeros((num_buckets_w, num_buckets_t))
            for bw, bt in zip(bucket_w_indices, bucket_t_indices):
                if 0 <= bw < num_buckets_w and 0 <= bt < num_buckets_t:
                    bucket_grid[bw, bt] = 1

            extent = [
                min_idx_abs,
                min_idx_abs + num_buckets_w * B1_int,
                0,
                num_buckets_t * B2_int * time_step_size_us
            ]

            cmap = plt.cm.YlOrRd.copy()
            cmap.set_under('#1a1a1a')

            im = ax.imshow(
                bucket_grid.T,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap=cmap,
                vmin=0.5,
                vmax=1.0,
                interpolation='nearest'
            )

            ax.set_xlim(min_idx_abs, max_idx_abs + 1)
            ax.set_ylim(0, max_time_axis)
            ax.set_box_aspect(1)

            plot_title = f"{side_names[side_idx]}\n{plane_types[plane_idx]}"
            ax.set_title(plot_title, fontsize=title_size, pad=10)

            ax.set_xlabel('Absolute Wire Index', fontsize=label_size)
            ax.set_ylabel('Time (μs)', fontsize=label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size)

            info_text = f"Active: {num_active_int:,}\nTotal: {total_buckets:,}\nCoverage: {coverage_pct:.1f}%\nBucket: {B1_int}x{B2_int}"
            ax.text(
                0.02, 0.98, info_text,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3')
            )

    fig.suptitle('Active Buckets Visualization (Bucketed Mode)', fontsize=16, y=0.98)
    return fig


def visualize_by_index(wire_signals_dict, simulation_params, indices_list, figsize=(10, 8),
                       sparse_data=False):
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
    sparse_data : bool, optional
        If True, expect sparse format (indices, values).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    num_time_steps = simulation_params['num_time_steps']
    time_step_size_us = simulation_params['time_step_size_us']
    num_wires_actual = simulation_params['num_wires_actual']
    max_abs_indices = simulation_params['max_abs_indices']
    min_abs_indices = simulation_params['min_abs_indices']

    side_names = ['West Side (x < 0)', 'East Side (x > 0)']
    plane_types = ['First Induction (U)', 'Second Induction (V)', 'Collection (Y)']

    time_axis = np.arange(num_time_steps) * time_step_size_us

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

            if sparse_data:
                indices, values = wire_signals_dict[plane_key]
                if len(values) == 0:
                    ax.text(0.5, 0.5, "(No data)", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel('Time (μs)')
                    ax.set_ylabel('Signal Strength')
                    continue

                indices_np = np.array(indices)
                values_np = np.array(values)

                for wire_idx in indices_list:
                    rel_idx = wire_idx - min_idx_abs
                    mask = indices_np[:, 0] == rel_idx
                    if np.any(mask):
                        wire_times = indices_np[mask, 1] * time_step_size_us
                        wire_values = values_np[mask]
                        sort_order = np.argsort(wire_times)
                        ax.plot(wire_times[sort_order], wire_values[sort_order],
                               label=f'Wire {wire_idx}', alpha=0.8, marker='.', markersize=1, linestyle='-')
            else:
                signal_data = np.array(wire_signals_dict[plane_key])

                for wire_idx in indices_list:
                    rel_idx = wire_idx - min_idx_abs
                    if 0 <= rel_idx < signal_data.shape[0]:
                        wire_signal = signal_data[rel_idx, :]
                        ax.plot(time_axis, wire_signal, label=f'Wire {wire_idx}', alpha=0.8)

            ax.set_xlabel('Time (μs)')
            ax.set_ylabel('Signal Strength')
            ax.grid(True, alpha=0.3)

            if len(indices_list) <= 10:
                ax.legend(fontsize=8)

    plt.tight_layout()
    return fig
