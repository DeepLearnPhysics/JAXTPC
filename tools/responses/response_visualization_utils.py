"""
Visualization utilities for wire response data.

This module provides plotting functions for visualizing kernels,
smoothed data, and creating animations.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .response_extraction import PLANE_CONFIGS
from tools.kernels import calculate_wire_count


def visualize_kernel(kernel, kernel_x_coords, kernel_y_coords, plane='U', figsize=(8, 8)):
    """
    Visualize the extracted kernel.
    
    Parameters
    ----------
    kernel : np.ndarray
        Kernel array
    kernel_x_coords : np.ndarray
        Wire coordinates for kernel
    kernel_y_coords : np.ndarray
        Time coordinates for kernel
    plane : str
        Plane name
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    config = PLANE_CONFIGS[plane]

    fig, ax = plt.subplots(figsize=figsize)

    # Create continuous colormap
    cmap = plt.cm.RdBu_r

    # Plot kernel
    im = ax.imshow(kernel, aspect='auto',
                   extent=[kernel_x_coords[0], kernel_x_coords[-1],
                          kernel_y_coords[0], kernel_y_coords[-1]],
                   cmap=cmap, origin='lower')

    # Set axis properties
    ax.set_xlabel('Wire Number', fontsize=12)
    ax.set_ylabel('Time [μs]', fontsize=12)
    ax.set_title(f'{config["title"]} - Extracted Kernel ({kernel.shape[0]}x{kernel.shape[1]})',
                 fontsize=14, pad=15)

    # Add grid
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add crosshairs at (0,0)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Create colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, orientation='vertical')
    cbar.set_label('Log 10″', rotation=270, labelpad=15, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Set colorbar limits based on plane
    im.set_clim(config['value_range'])

    plt.tight_layout()
    return fig, ax


def visualize_diffusion_progression(DKernel, linear_s, x_coords, y_coords, plane='U', 
                                  figsize=(15, 3)):
    """
    Visualize how kernels change with diffusion parameter s.
    
    Parameters
    ----------
    DKernel : np.ndarray
        Array of diffused kernels, shape (num_s, height, width)
    linear_s : np.ndarray
        Array of s values
    x_coords : np.ndarray
        Wire coordinates
    y_coords : np.ndarray
        Time coordinates
    plane : str
        Plane name
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    num_s = len(linear_s)
    
    # Show a few representative kernels
    indices_to_show = [0, num_s//4, num_s//2, 3*num_s//4, num_s-1]
    
    fig, axes = plt.subplots(1, len(indices_to_show), figsize=figsize)
    
    config = PLANE_CONFIGS[plane]
    cmap = plt.cm.RdBu_r
    vmin, vmax = config['value_range']
    
    for i, idx in enumerate(indices_to_show):
        kernel = DKernel[idx]
        s_val = linear_s[idx]
        
        im = axes[i].imshow(kernel, aspect='auto',
                           extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                           cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        
        axes[i].set_title(f's = {s_val:.2f}')
        axes[i].set_xlabel('Wire Number')
        if i == 0:
            axes[i].set_ylabel('Time [μs]')
        
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'{plane} Plane Diffusion Progression', y=1.02)
    return fig


def create_parameter_sweep_gif(visualization_func, DKernels, plane='Y', 
                              parameter='w_offset', param_range=(0, 1), 
                              fixed_params=None, output_filename='parameter_sweep.gif',
                              n_frames=30, fps=10):
    """
    Create a GIF animation sweeping through any parameter.

    Parameters
    ----------
    visualization_func : callable
        Function to create visualization (e.g., visualize_interpolation_steps)
    DKernels : dict
        Dictionary of diffusion kernels
    plane : str
        Which plane to visualize
    parameter : str
        Which parameter to sweep ('w_offset', 's_observed', or 't_offset')
    param_range : tuple
        (min, max) values for the parameter
    fixed_params : dict, optional
        Dict of fixed parameters (defaults: s_observed=0.3, w_offset=0.25, t_offset=0.15)
    output_filename : str
        Name of output GIF file
    n_frames : int
        Number of frames for the animation
    fps : int
        Frames per second for the GIF
        
    Returns
    -------
    None
        Saves GIF to output_filename
    """
    # Set default fixed parameters
    if fixed_params is None:
        fixed_params = {
            's_observed': 0.3,
            'w_offset': 0.25,
            't_offset': 0.15
        }

    # Create temporary folder for frames
    temp_folder = 'temp_param_sweep_frames'
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    print(f"Creating parameter sweep GIF for {parameter}...")
    print(f"  Range: {param_range[0]:.3f} to {param_range[1]:.3f}")

    try:
        # Create parameter values that go from min to max and back to min
        half_frames = n_frames // 2
        param_values_forward = np.linspace(param_range[0], param_range[1], half_frames, endpoint=False)
        param_values_backward = np.linspace(param_range[1], param_range[0], n_frames - half_frames, endpoint=True)
        param_values = np.concatenate([param_values_forward, param_values_backward])

        # Generate frames
        frame_files = []
        for i, param_val in enumerate(param_values):
            print(f"  Generating frame {i+1}/{n_frames} ({parameter}={param_val:.3f})")

            # Set up parameters for this frame
            params = fixed_params.copy()
            params[parameter] = param_val

            # Create the visualization
            fig = visualization_func(
                DKernels,
                plane=plane,
                **params
            )

            if fig:
                # Save frame
                frame_filename = os.path.join(temp_folder, f'frame_{i:04d}.png')
                plt.figure(fig.number)
                plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
                plt.close(fig)
                frame_files.append(frame_filename)

        # Create GIF from frames
        print("\nCreating GIF...")
        images = []
        for filename in frame_files:
            images.append(Image.open(filename))

        if images:
            # Save GIF
            duration = 1000 // fps
            images[0].save(
                output_filename,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF saved as: {output_filename}")

    finally:
        # Clean up temporary folder
        print("\nCleaning up temporary files...")
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        print("Done!")


def visualize_interpolation_steps(DKernels, plane='Y', s_observed=0.3, w_offset=0.25, t_offset=0.15, 
                                 wire_spacing=0.1, time_spacing=0.5, verbose=False):
    """
    Visualize the interpolation process step by step in a 2x2 layout.
    
    Parameters
    ----------
    DKernels : dict
        Dictionary of diffusion kernels
    plane : str
        Which plane to visualize
    s_observed : float
        Diffusion parameter in [0, 1]
    w_offset : float
        Wire offset in [0, 1)
    t_offset : float
        Time offset in [0, 0.5)
    wire_spacing : float
        Wire spacing
    time_spacing : float
        Time spacing
    verbose : bool
        Print debug information
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if plane not in DKernels:
        print(f"Plane {plane} not available")
        return None

    DKernel, linear_s, kernel_shape, x_coords, y_coords = DKernels[plane]
    kernel_height, kernel_width = kernel_shape
    num_s = len(linear_s)

    # Calculate interpolation parameters
    s_continuous = s_observed * (num_s - 1)
    s_idx = int(np.floor(s_continuous))
    s_idx = min(s_idx, num_s - 2)
    s_alpha = s_continuous - s_idx

    center_w = kernel_width // 2
    bins_per_wire = int(1.0 / wire_spacing)
    w_bin_offset = w_offset * bins_per_wire
    w_base_bin = int(np.floor(w_bin_offset))
    w_alpha = w_bin_offset - w_base_bin

    # Calculate wire parameters
    num_wires = calculate_wire_count(kernel_width, wire_spacing)

    if num_wires % 2 == 0:
        half_wires = num_wires // 2
        wire_positions = np.arange(-half_wires, half_wires)
    else:
        half_wires = num_wires // 2
        wire_positions = np.arange(-half_wires, half_wires + 1)

    wire_base_positions = wire_positions * bins_per_wire + center_w
    actual_wire_positions = wire_base_positions + w_base_bin

    if verbose:
        print(f"\nInterpolation parameters:")
        print(f"  s_observed={s_observed:.3f} -> s_idx={s_idx}, s_alpha={s_alpha:.3f}")
        print(f"  w_offset={w_offset:.3f} -> w_base_bin={w_base_bin}, w_alpha={w_alpha:.3f}")
        print(f"  t_offset={t_offset:.3f}")
        print(f"  num_wires={num_wires}, Expected output shape: ({num_wires}, {kernel_height-1})")

    # Define specific wires and colors
    target_wires = [-1, 0, 1, 3, 5]
    wire_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # Red, Blue, Green, Orange, Purple

    # Find indices for target wires
    wire_indices_to_show = []
    for target in target_wires:
        idx = np.where(wire_positions == target)[0]
        if len(idx) > 0:
            wire_indices_to_show.append(idx[0])

    # Extract interpolation data for selected wires
    all_wire_results = []
    all_wire_steps = []

    for wire_idx in wire_indices_to_show:
        # Get the two adjacent wire bin indices for interpolation
        wire_bin_left = wire_base_positions[wire_idx] + w_base_bin
        wire_bin_right = wire_bin_left + 1

        # Clamp to valid range
        wire_bin_left = np.clip(wire_bin_left, 0, kernel_width - 1)
        wire_bin_right = np.clip(wire_bin_right, 0, kernel_width - 1)

        # Step 1: Extract values for s interpolation
        values_s_n_left = np.array(DKernel[s_idx, :, int(wire_bin_left)])
        values_s_n_plus_1_left = np.array(DKernel[s_idx + 1, :, int(wire_bin_left)])
        values_s_n_right = np.array(DKernel[s_idx, :, int(wire_bin_right)])
        values_s_n_plus_1_right = np.array(DKernel[s_idx + 1, :, int(wire_bin_right)])

        # Step 2: S interpolation for both left and right wire positions
        values_s_interp_left = (1 - s_alpha) * values_s_n_left + s_alpha * values_s_n_plus_1_left
        values_s_interp_right = (1 - s_alpha) * values_s_n_right + s_alpha * values_s_n_plus_1_right

        # Step 3: Wire interpolation
        values_w_interp = (1 - w_alpha) * values_s_interp_left + w_alpha * values_s_interp_right

        # Step 4: Time interpolation
        t_alpha = t_offset / time_spacing
        interpolated_values = (1 - t_alpha) * values_w_interp[:-1] + t_alpha * values_w_interp[1:]

        # Store results
        all_wire_results.append(interpolated_values)
        all_wire_steps.append({
            'left_s_n': values_s_n_left,
            'left_s_n_plus_1': values_s_n_plus_1_left,
            'right_s_n': values_s_n_right,
            'right_s_n_plus_1': values_s_n_plus_1_right,
            'left_s_interp': values_s_interp_left,
            'right_s_interp': values_s_interp_right,
            'w_interp': values_w_interp,
            'final': interpolated_values
        })

    # Create 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create time coordinates
    time_full = np.linspace(y_coords[0], y_coords[-1], kernel_height)
    time_reduced = np.linspace(y_coords[0], y_coords[-1], kernel_height - 1)

    # Get wire labels for the selected wires
    wire_labels = [wire_positions[i] for i in wire_indices_to_show]

    # Top left: s_n (dashed) and s_n+1 (solid) for left positions
    for i, (wire_idx, wire_label) in enumerate(zip(wire_indices_to_show, wire_labels)):
        step = all_wire_steps[i]
        color = wire_colors[i]

        axes[0, 0].plot(time_full, step['left_s_n'], color=color, alpha=0.7,
                       linestyle='--')
        axes[0, 0].plot(time_full, step['left_s_n_plus_1'], color=color, alpha=0.7,
                       linestyle='-')

    axes[0, 0].set_title(f'Left positions: s[{s_idx}] (dashed) & s[{s_idx+1}] (solid)')
    axes[0, 0].set_xlabel('Time [μs]')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: After S interpolation (solid=left, dashed=right)
    for i, (wire_idx, wire_label) in enumerate(zip(wire_indices_to_show, wire_labels)):
        step = all_wire_steps[i]
        color = wire_colors[i]

        axes[0, 1].plot(time_full, step['left_s_interp'], color=color, alpha=0.7,
                       linestyle='-')
        axes[0, 1].plot(time_full, step['right_s_interp'], color=color, alpha=0.7,
                       linestyle='--')

    axes[0, 1].set_title(f'After S interpolation\n(solid=left, dashed=right)')
    axes[0, 1].set_xlabel('Time [μs]')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Final results
    for i, (wire_idx, wire_label) in enumerate(zip(wire_indices_to_show, wire_labels)):
        step = all_wire_steps[i]
        color = wire_colors[i]

        axes[1, 0].plot(time_reduced, step['final'], color=color, alpha=0.8,
                       linewidth=2, label=f'Wire {wire_label}')

    axes[1, 0].set_title('Final Results (after all interpolations)')
    axes[1, 0].set_xlabel('Time [μs]')
    axes[1, 0].set_ylabel('Interpolated Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Full S-interpolated plane with colored lines
    config = PLANE_CONFIGS[plane]
    cmap = plt.cm.RdBu_r
    vmin, vmax = config['value_range']

    # Perform S interpolation on the full planes
    full_s_interp = (1 - s_alpha) * DKernel[s_idx] + s_alpha * DKernel[s_idx + 1]

    # Plot the interpolated plane
    im = axes[1, 1].imshow(full_s_interp.T, aspect='auto',
                          extent=[y_coords[0], y_coords[-1], x_coords[0], x_coords[-1]],
                          cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

    # Add colored lines at the actual wire positions
    for i, (wire_idx, wire_label) in enumerate(zip(wire_indices_to_show, wire_labels)):
        wire_pos = actual_wire_positions[wire_idx]
        if 0 <= wire_pos < kernel_width:
            wire_coord = x_coords[int(wire_pos)] if int(wire_pos) < len(x_coords) else x_coords[-1]
            axes[1, 1].axhline(y=wire_coord, color=wire_colors[i], alpha=0.8,
                             linewidth=2, label=f'Wire {wire_label}')

    axes[1, 1].set_title(f'S-interpolated plane (s={s_observed:.2f})\nwith wire positions (w_offset={w_offset:.2f})')
    axes[1, 1].set_xlabel('Time [μs]')
    axes[1, 1].set_ylabel('Wire Number')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig