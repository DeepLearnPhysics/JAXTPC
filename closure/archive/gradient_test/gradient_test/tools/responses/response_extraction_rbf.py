"""
Response Extraction from Images

This module handles extraction of wire response data from colorbar and plot images,
including conversion between the paper's "Log 10" scale and actual current values.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator
from skimage.restoration import denoise_bilateral
from PIL import Image


# Define color scale ranges for each plane
PLANE_CONFIGS = {
    'U': {
        'value_range': (-3.16, 3.16),
        'tick_values': [3, 2, 1, 0, -1, -2, -3],
        'title': 'U Plane'
    },
    'V': {
        'value_range': (-3.16, 3.16),
        'tick_values': [3, 2, 1, 0, -1, -2, -3],
        'title': 'V Plane'
    },
    'Y': {
        'value_range': (-3.16, 4.16),
        'tick_values': [4, 3, 2, 1, 0, -1, -2, -3],
        'title': 'Y Plane'
    }
}


def analyze_colorbar_for_plane(colorbar_path, plane='U', n_colors=20):
    """
    Analyze colorbar and extract colors with exact tick positions for specific plane.

    Parameters
    ----------
    colorbar_path : str
        Path to colorbar image file
    plane : str
        Which plane ('U', 'V', or 'Y')
    n_colors : int
        Number of color levels to extract

    Returns
    -------
    extracted_colors : np.ndarray
        RGB colors, shape (n_colors, 3)
    extracted_values : np.ndarray
        Corresponding "Log 10" values, shape (n_colors,)
    """
    # Load colorbar
    colorbar_img = np.array(Image.open(colorbar_path).convert('RGB'))
    height, width = colorbar_img.shape[:2]

    # Get value range for this plane
    value_range = PLANE_CONFIGS[plane]['value_range']

    # Extract colors evenly spaced along the colorbar
    positions = np.linspace(0, height-1, n_colors)
    center_x = width // 2

    extracted_colors = []
    for pos in positions:
        y_idx = int(pos)
        region = colorbar_img[
            max(0, y_idx-1):min(height, y_idx+2),
            max(0, center_x-2):min(width, center_x+3)
        ]
        mean_color = np.mean(region.reshape(-1, 3), axis=0)
        extracted_colors.append(mean_color)

    extracted_colors = np.array(extracted_colors)
    # Values go from top to bottom in "Log 10" scale
    extracted_values = np.linspace(value_range[1], value_range[0], n_colors)

    return extracted_colors, extracted_values


def paper_log10_to_actual(log10_values):
    """
    Convert from the paper's "Log 10" scale back to actual current values.

    The paper shows a symmetric log scale where:
    - For "Log 10" > 0: i = 10^(value) / 10^5
    - For "Log 10" = 0: i = 0
    - For "Log 10" < 0: i = -10^(abs(value)) / 10^5

    Parameters
    ----------
    log10_values : np.ndarray
        Values in paper's "Log 10" scale

    Returns
    -------
    actual_values : np.ndarray
        Actual current values (electrons per time bin)
    """
    actual_values = np.zeros_like(log10_values)

    # Positive Log10 values
    mask_pos = log10_values > 0
    actual_values[mask_pos] = 10**(log10_values[mask_pos]) / 10**5

    # Zero
    mask_zero = log10_values == 0
    actual_values[mask_zero] = 0

    # Negative Log10 values
    mask_neg = log10_values < 0
    actual_values[mask_neg] = -10**(-log10_values[mask_neg]) / 10**5

    return actual_values


def extract_with_bilateral_filter(plot_path, colorbar_colors, colorbar_values,
                                 x_range=(-10.5, 10.5), y_range=(-60, 40),
                                 time_bin_size=0.5, wire_spacing=0.1):
    """
    Extract data using bilateral filtering to smooth within contours while preserving edges.

    Parameters
    ----------
    plot_path : str
        Path to plot image file
    colorbar_colors : np.ndarray
        RGB colors from colorbar, shape (n_colors, 3)
    colorbar_values : np.ndarray
        Corresponding values, shape (n_colors,)
    x_range : tuple
        (x_min, x_max) wire number range
    y_range : tuple
        (y_min, y_max) time range in microseconds
    time_bin_size : float
        Time bin size in microseconds
    wire_spacing : float
        Wire number spacing

    Returns
    -------
    data_values : np.ndarray
        Extracted values on fixed grid
    confidence_map : np.ndarray
        Confidence values on fixed grid
    x_coords : np.ndarray
        Wire coordinates
    y_coords : np.ndarray
        Time coordinates
    """
    # Load plot image
    plot_img = np.array(Image.open(plot_path).convert('RGB'))
    plot_height, plot_width = plot_img.shape[:2]

    print(f"Image size: {plot_height} x {plot_width}")

    # Build color mapping
    color_tree = KDTree(colorbar_colors)

    # Map colors to values - discrete mapping, no interpolation
    print("Mapping colors to discrete values...")
    plot_img_flat = plot_img.reshape(-1, 3)
    distances, indices = color_tree.query(plot_img_flat)
    pixel_values = colorbar_values[indices].reshape(plot_height, plot_width)

    # Get unique contour levels and spacing
    unique_levels = np.unique(pixel_values)
    contour_spacing = np.median(np.diff(unique_levels))

    print(f"Found {len(unique_levels)} unique contour levels")
    print(f"Contour spacing: {contour_spacing:.3f}")
    print(f"Value range: {unique_levels.min():.2f} to {unique_levels.max():.2f}")

    # Apply bilateral filtering to smooth within contours while preserving edges
    print("Applying bilateral filter...")

    # Normalize values for bilateral filter (works better with values in [0,1] range)
    value_min, value_max = pixel_values.min(), pixel_values.max()
    pixel_values_normalized = (pixel_values - value_min) / (value_max - value_min)

    # Apply bilateral filter
    # sigma_color controls how much values can differ while still being smoothed together
    # Set it to about half the contour spacing (in normalized units)
    sigma_color = 0.01
    sigma_spatial = 5.0  # Spatial extent of smoothing

    filtered_normalized = denoise_bilateral(
        pixel_values_normalized,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=None  # Single channel data
    )

    # Convert back to original value range
    filtered_values = filtered_normalized * (value_max - value_min) + value_min

    print(f"Filtered value range: {filtered_values.min():.2f} to {filtered_values.max():.2f}")

    # Create coordinate arrays for the image
    x_coords_img = np.linspace(x_range[0], x_range[1], plot_width)
    y_coords_img = np.linspace(y_range[1], y_range[0], plot_height)  # Note: y flipped

    # Create output coordinate grids at target resolution
    n_time_bins = int((y_range[1] - y_range[0]) / time_bin_size) + 1
    y_coords_fixed = np.linspace(y_range[0], y_range[1], n_time_bins)

    n_wire_bins = int((x_range[1] - x_range[0]) / wire_spacing) + 1
    x_coords_fixed = np.linspace(x_range[0], x_range[1], n_wire_bins)

    # Interpolate filtered values to fixed grid
    print("Interpolating to target grid...")
    interp_func = RegularGridInterpolator(
        (y_coords_img[::-1], x_coords_img),  # Flip y for ascending order
        filtered_values[::-1, :],  # Flip data accordingly
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    # Create mesh grid for evaluation
    Y_fixed, X_fixed = np.meshgrid(y_coords_fixed, x_coords_fixed, indexing='ij')
    eval_points = np.column_stack((Y_fixed.ravel(), X_fixed.ravel()))

    # Evaluate at fixed grid points
    data_values_fixed = interp_func(eval_points).reshape(Y_fixed.shape)

    # Simple confidence map (uniform high confidence)
    confidence_map_fixed = np.ones_like(data_values_fixed) * 0.9

    print(f"Fixed grid resolution: {len(y_coords_fixed)} x {len(x_coords_fixed)}")
    print(f"Output value range: {data_values_fixed.min():.2f} to {data_values_fixed.max():.2f}")

    return data_values_fixed, confidence_map_fixed, x_coords_fixed, y_coords_fixed


def extract_kernel(data_values, x_coords, y_coords, kernel_size=(127, 201)):
    """
    Extract a kernel centered at (0,0) from the data.
    If kernel size is even, zero out one row/column to ensure (0,0) is at center.

    Parameters
    ----------
    data_values : np.ndarray
        2D array of values
    x_coords : np.ndarray
        Wire coordinates
    y_coords : np.ndarray
        Time coordinates
    kernel_size : tuple
        (height, width) of kernel to extract

    Returns
    -------
    kernel : np.ndarray
        Extracted kernel
    kernel_x_coords : np.ndarray
        Wire coordinates for kernel
    kernel_y_coords : np.ndarray
        Time coordinates for kernel
    """
    # Find indices closest to (0,0)
    x_idx_zero = np.argmin(np.abs(x_coords - 0))
    y_idx_zero = np.argmin(np.abs(y_coords - 0))

    kernel_height, kernel_width = kernel_size

    # Calculate extraction bounds
    # For even kernel size, we need to handle centering
    if kernel_height % 2 == 0:
        # Even height - need to zero out one row
        half_h = kernel_height // 2
        y_start = y_idx_zero - half_h + 1
        y_end = y_idx_zero + half_h + 1
    else:
        # Odd height - normal centering
        half_h = kernel_height // 2
        y_start = y_idx_zero - half_h
        y_end = y_idx_zero + half_h + 1

    if kernel_width % 2 == 0:
        # Even width - need to zero out one column
        half_w = kernel_width // 2
        x_start = x_idx_zero - half_w + 1
        x_end = x_idx_zero + half_w + 1
    else:
        # Odd width - normal centering
        half_w = kernel_width // 2
        x_start = x_idx_zero - half_w
        x_end = x_idx_zero + half_w + 1

    # Handle boundaries
    y_start = max(0, y_start)
    y_end = min(data_values.shape[0], y_end)
    x_start = max(0, x_start)
    x_end = min(data_values.shape[1], x_end)

    # Extract kernel
    kernel = np.zeros(kernel_size)
    extracted = data_values[y_start:y_end, x_start:x_end]

    # Calculate where to place the extracted data in the kernel
    kernel_y_start = max(0, -y_start + y_idx_zero - kernel_height//2 + (1 if kernel_height % 2 == 0 else 0))
    kernel_x_start = max(0, -x_start + x_idx_zero - kernel_width//2 + (1 if kernel_width % 2 == 0 else 0))

    kernel_y_end = kernel_y_start + extracted.shape[0]
    kernel_x_end = kernel_x_start + extracted.shape[1]

    kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end] = extracted

    # If even size, zero out the first row/column to ensure (0,0) is at center
    if kernel_height % 2 == 0:
        kernel[0, :] = 0
    if kernel_width % 2 == 0:
        kernel[:, 0] = 0

    # Extract corresponding coordinates
    kernel_x_coords = x_coords[x_start:x_end]
    kernel_y_coords = y_coords[y_start:y_end]

    # Pad coordinate arrays if needed
    if len(kernel_x_coords) < kernel_width:
        # Pad with appropriate values
        dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0.1
        if x_start == 0:
            # Pad on the left
            n_pad = kernel_width - len(kernel_x_coords)
            pad_values = kernel_x_coords[0] - dx * np.arange(n_pad, 0, -1)
            kernel_x_coords = np.concatenate([pad_values, kernel_x_coords])
        else:
            # Pad on the right
            n_pad = kernel_width - len(kernel_x_coords)
            pad_values = kernel_x_coords[-1] + dx * np.arange(1, n_pad + 1)
            kernel_x_coords = np.concatenate([kernel_x_coords, pad_values])

    if len(kernel_y_coords) < kernel_height:
        # Pad with appropriate values
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0.5
        if y_start == 0:
            # Pad on the bottom
            n_pad = kernel_height - len(kernel_y_coords)
            pad_values = kernel_y_coords[0] - dy * np.arange(n_pad, 0, -1)
            kernel_y_coords = np.concatenate([pad_values, kernel_y_coords])
        else:
            # Pad on the top
            n_pad = kernel_height - len(kernel_y_coords)
            pad_values = kernel_y_coords[-1] + dy * np.arange(1, n_pad + 1)
            kernel_y_coords = np.concatenate([kernel_y_coords, pad_values])

    return kernel, kernel_x_coords, kernel_y_coords


def save_kernel(kernel, kernel_x_coords, kernel_y_coords, plane,
               time_bin_size, wire_spacing, output_path):
    """
    Save kernel data to npz file in actual current values (not log scale).

    Parameters
    ----------
    kernel : np.ndarray
        Kernel array in paper's "Log 10" scale
    kernel_x_coords : np.ndarray
        Wire coordinates
    kernel_y_coords : np.ndarray
        Time coordinates
    plane : str
        Plane name ('U', 'V', or 'Y')
    time_bin_size : float
        Time bin size used
    wire_spacing : float
        Wire spacing used
    output_path : str
        Output file path
    """
    # Convert from log10 scale to actual current values before saving
    kernel_actual = paper_log10_to_actual(kernel)

    # Clamp very small values to exactly zero to avoid numerical noise
    threshold = 1e-5
    kernel_actual = np.where(np.abs(kernel_actual) < threshold, 0.0, kernel_actual)

    np.savez(output_path,
             kernel=kernel_actual,  # Save in actual values, not log scale
             kernel_x_coords=kernel_x_coords,
             kernel_y_coords=kernel_y_coords,
             plane=plane,
             time_bin_size=time_bin_size,
             wire_spacing=wire_spacing)

    print(f"Saved kernel to {output_path} (in actual values, not log scale)")


def process_single_plane(colorbar_path, plot_path, plane='U',
                        time_bin_size=0.5, wire_spacing=0.1,
                        kernel_size=(127, 201),
                        output_kernel_path=None):
    """
    Process a single plane using bilateral filtering approach.

    Parameters
    ----------
    colorbar_path : str
        Path to colorbar image
    plot_path : str
        Path to plot image
    plane : str
        Which plane ('U', 'V', or 'Y')
    time_bin_size : float
        Time bin size in microseconds
    wire_spacing : float
        Wire spacing
    kernel_size : tuple
        (height, width) of kernel to extract
    output_kernel_path : str, optional
        Path to save kernel, defaults to 'kernel_{plane}_kernel.npz'

    Returns
    -------
    dict
        Dictionary containing all results
    """
    print(f"\nProcessing {plane} plane with bilateral filtering...")

    # Extract colors from colorbar
    extracted_colors, extracted_values = analyze_colorbar_for_plane(colorbar_path, plane=plane)

    # Extract data using bilateral filtering
    data_values_log10, confidence_map, x_coords, y_coords = extract_with_bilateral_filter(
        plot_path, extracted_colors, extracted_values,
        time_bin_size=time_bin_size,
        wire_spacing=wire_spacing
    )

    # Extract kernel
    kernel, kernel_x_coords, kernel_y_coords = extract_kernel(
        data_values_log10, x_coords, y_coords, kernel_size
    )

    print(f"Extracted kernel shape: {kernel.shape}")
    print(f"Kernel center value at (0,0): {kernel[kernel.shape[0]//2, kernel.shape[1]//2]:.3f}")

    # Save kernel if path provided
    if output_kernel_path is None:
        output_kernel_path = f'{plane}_plane_kernel.npz'

    save_kernel(kernel, kernel_x_coords, kernel_y_coords, plane,
               time_bin_size, wire_spacing, output_kernel_path)

    return {
        'kernel': kernel,
        'kernel_x_coords': kernel_x_coords,
        'kernel_y_coords': kernel_y_coords,
        'data_values': data_values_log10,
        'confidence_map': confidence_map,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'plane': plane,
        'time_bin_size': time_bin_size,
        'wire_spacing': wire_spacing
    }


def main():
    """Extract kernels for all wire planes when run as a script."""
    import os

    # Configuration
    planes = ['U', 'V', 'Y']
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, 'images')
    output_dir = script_dir  # Save in tools/responses/

    # Extraction parameters
    kernel_size = (127, 201)
    time_bin_size = 0.5
    wire_spacing = 0.1

    print("="*60)
    print("Wire Response Kernel Extraction")
    print("(Bilateral Filtering Approach)")
    print("="*60)
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Kernel size: {kernel_size}")
    print(f"Time bin size: {time_bin_size} μs")
    print(f"Wire spacing: {wire_spacing}")
    
    # Process each plane
    for plane in planes:
        colorbar_path = os.path.join(image_dir, f'{plane}_response_colorbar.png')
        plot_path = os.path.join(image_dir, f'{plane}_response_image.png')
        output_path = os.path.join(output_dir, f'{plane}_plane_kernel.npz')
        
        # Check if input files exist
        if not os.path.exists(colorbar_path):
            print(f"\nERROR: Missing colorbar image: {colorbar_path}")
            continue
        if not os.path.exists(plot_path):
            print(f"\nERROR: Missing plot image: {plot_path}")
            continue
            
        try:
            # Process the plane
            result = process_single_plane(
                colorbar_path=colorbar_path,
                plot_path=plot_path,
                plane=plane,
                time_bin_size=time_bin_size,
                wire_spacing=wire_spacing,
                kernel_size=kernel_size,
                output_kernel_path=output_path
            )
            
            print(f"✓ Created {output_path}")
            
        except Exception as e:
            print(f"\n✗ ERROR processing {plane} plane: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run example scripts: python3 example_response_usage.py")
    print("2. Run timing benchmark: python3 timing/benchmark_response_interpolation.py")


if __name__ == "__main__":
    main()