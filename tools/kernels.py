"""
Response Kernel Module

Loads wire response kernels, builds diffusion tables via DCT-domain Gaussian
blurring, and provides JIT-compiled interpolation for runtime signal generation.

Contents:
1. Kernel Loading - Load NPZ kernel files
2. DCT Diffusion - Generate DKernel tables (vmap over s levels)
3. Runtime Interpolation - JIT-compiled batch interpolation
4. High-Level API - load_response_kernels(), apply_diffusion_response()
"""

import os
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np

from tools.config import ResponseKernel


# ============================================================================
# KERNEL LOADING
# ============================================================================

def load_kernel(filename):
    """
    Load kernel from npz file (stored in actual values, not log scale).

    Parameters
    ----------
    filename : str
        Path to kernel npz file

    Returns
    -------
    kernel : np.ndarray
        Kernel array in actual current values
    kernel_x_coords : np.ndarray
        Wire coordinates
    kernel_y_coords : np.ndarray
        Time coordinates
    plane : str
        Plane name
    dx : float
        Wire spacing
    dy : float
        Time spacing
    wire_zero_bin : int
        Index of wire=0 in kernel (where kernel_x_coords is closest to 0)
    time_zero_bin : int
        Index of t=0 in kernel (where kernel_y_coords is closest to 0)
    """
    data = np.load(filename, allow_pickle=True)

    # Kernel is now stored in actual values, not log scale
    kernel = data['kernel']

    kernel_x_coords = data['kernel_x_coords']
    kernel_y_coords = data['kernel_y_coords']
    plane = str(data['plane'])

    # Get spacing from coordinates
    dx = kernel_x_coords[1] - kernel_x_coords[0] if len(kernel_x_coords) > 1 else 0.1
    dy = kernel_y_coords[1] - kernel_y_coords[0] if len(kernel_y_coords) > 1 else 0.5

    # Read zero-bin indices from file, or compute from coordinates as fallback
    wire_zero_bin = int(data['wire_zero_bin']) if 'wire_zero_bin' in data else int(np.argmin(np.abs(kernel_x_coords)))
    time_zero_bin = int(data['time_zero_bin']) if 'time_zero_bin' in data else int(np.argmin(np.abs(kernel_y_coords)))

    return kernel, kernel_x_coords, kernel_y_coords, plane, dx, dy, wire_zero_bin, time_zero_bin


def calculate_wire_count(kernel_width, wire_spacing=0.1):
    """
    Calculate how many wire positions we can represent given kernel width.

    For wire_spacing = 0.1, we have 10 bins per unit wire spacing.
    So if kernel_width = 127, we have (127-1)/10 = 12.6, so floor(12.6) = 12 wires total.

    Parameters
    ----------
    kernel_width : int
        Width of kernel in bins
    wire_spacing : float
        Wire spacing

    Returns
    -------
    num_wires : int
        Number of representable wires
    """
    # Number of bins per wire unit
    bins_per_wire = int(1.0 / wire_spacing)  # 10

    # Total wire range (symmetric around center)
    wire_range = (kernel_width - 1) / bins_per_wire
    num_wires = int(wire_range)  # Use floor, not +1

    return num_wires


# ============================================================================
# DIFFUSION KERNEL TABLE GENERATION
# ============================================================================

# Number of sigma for Gaussian kernel truncation in reflect+conv.
# 5σ gives machine-precision accuracy; 4σ gives 0.02% max error.
_N_SIGMAS_BLUR = 5

def _gaussian_kernel_1d(size, sigma):
    """Normalized 1D Gaussian kernel of given size, centered."""
    x = jnp.linspace(-(size - 1) / 2., (size - 1) / 2., size, dtype=jnp.float32)
    g = jnp.exp(-0.5 * x**2 / jnp.maximum(sigma**2, 1e-10))
    return g / g.sum()


def generate_dkernel_table(sigma_trans_max, sigma_long_max,
                           base_kernel, kernel_dx, kernel_dy, s_levels,
                           ks_w=None, ks_t=None):
    """Generate diffusion kernel table via reflect padding + spatial convolution.

    Wire kernel is full two-sided (center at mid-array). Uses reflect padding
    at both edges and separable Gaussian convolution via lax.conv_general_dilated.
    Equivalent to exact linear convolution with the Gaussian.

    Parameters
    ----------
    sigma_trans_max : float
        Maximum transverse diffusion sigma (wire pitches, same units as kernel_dx).
    sigma_long_max : float
        Maximum longitudinal diffusion sigma (us, same units as kernel_dy).
    base_kernel : jnp.ndarray, shape (H, W)
        Raw response kernel (field response, no diffusion). Full two-sided.
    kernel_dx : float
        Wire axis bin spacing (cm per kernel bin).
    kernel_dy : float
        Time axis bin spacing (μs per kernel bin).
    s_levels : jnp.ndarray, shape (num_s,)
        Diffusion levels from 0 to 1. s=0 means no diffusion.
    ks_w : int, optional
        Gaussian kernel size for wire axis. If None, computed from sigma_trans_max.
        Pass explicitly when sigma_trans_max is a traced value (differentiable path).
    ks_t : int, optional
        Gaussian kernel size for time axis. If None, computed from sigma_long_max.

    Returns
    -------
    DKernel : jnp.ndarray, shape (num_s, H, W)
        Diffusion kernel table. Full two-sided.
    """
    H, W = base_kernel.shape

    # Compute Gaussian kernel sizes (bins) and padding
    if ks_w is None:
        max_sigma_w_bins = sigma_trans_max / kernel_dx
        max_sigma_t_bins = sigma_long_max / kernel_dy
        ks_w = int(2 * ((max_sigma_w_bins * 2 * _N_SIGMAS_BLUR) // 2) + 1)
        ks_t = int(2 * ((max_sigma_t_bins * 2 * _N_SIGMAS_BLUR) // 2) + 1)
    pw = ks_w // 2
    pt = ks_t // 2

    # Reflect-pad the kernel
    padded = jnp.pad(base_kernel, ((pt, 0), (pw, 0)), mode='reflect')

    def make_level(s):
        sigma_T_bins = sigma_trans_max * s / kernel_dx
        sigma_L_bins = sigma_long_max * s / kernel_dy

        res = padded[jnp.newaxis, jnp.newaxis, :, :]

        # Time axis convolution (axis 0 of kernel = axis 2 of NCHW)
        g_t = _gaussian_kernel_1d(ks_t, sigma_L_bins)
        k_t = g_t[:, None][jnp.newaxis, jnp.newaxis, :, :]
        res = jax.lax.conv_general_dilated(
            res, k_t, (1, 1), 'SAME', dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

        # Wire axis convolution (axis 1 of kernel = axis 3 of NCHW)
        g_w = _gaussian_kernel_1d(ks_w, sigma_T_bins)
        k_w = g_w[None, :][jnp.newaxis, jnp.newaxis, :, :]
        res = jax.lax.conv_general_dilated(
            res, k_w, (1, 1), 'SAME', dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

        return jnp.squeeze(res)[pt:, pw:]

    return vmap(make_level)(s_levels)


def create_diffusion_kernel_array(planes=['U', 'V', 'Y'], num_s=16, kernel_dir=None,
                                 time_spacing=0.5,
                                 max_sigma_trans_unitless=None, max_sigma_long_unitless=None):
    """
    Create the diffusion kernel array DKernel for each plane using reflect
    padding + Gaussian convolution. Vectorized via vmap (no Python loop over s).

    Parameters
    ----------
    planes : list
        List of planes to process.
    num_s : int
        Number of s values (diffusion levels).
    kernel_dir : str
        Directory containing kernel files.
    time_spacing : float
        Time spacing in us.
    max_sigma_trans_unitless : float, optional
        Maximum transverse diffusion sigma in unitless grid coordinates.
    max_sigma_long_unitless : float, optional
        Maximum longitudinal diffusion sigma in unitless grid coordinates.

    Returns
    -------
    DKernels : dict
        Dictionary mapping plane to
        (DKernel, s_levels, kernel_shape, x_coords, y_coords, dx, dy,
         wire_zero_bin, time_zero_bin, base_kernel).
    """
    if kernel_dir is None:
        kernel_dir = os.path.join(os.path.dirname(__file__), 'responses')

    s_levels = jnp.linspace(0, 1, num_s)

    DKernels = {}

    for plane in planes:
        try:
            filename = f'{kernel_dir}/{plane}_plane_kernel.npz'
            kernel, x_coords, y_coords, loaded_plane, dx, dy, wire_zero_bin, time_zero_bin = load_kernel(filename)
            kernel_shape = kernel.shape

            base_kernel = jnp.array(kernel)

            # max_sigma_trans_unitless is in physical units (wire pitches)
            # max_sigma_long_unitless is in grid bins, convert to us
            sigma_trans_max = max_sigma_trans_unitless
            sigma_long_max = max_sigma_long_unitless * time_spacing

            DKernel = generate_dkernel_table(
                sigma_trans_max, sigma_long_max,
                base_kernel, dx, dy, s_levels)

            DKernels[plane] = (DKernel, s_levels, kernel_shape, x_coords,
                               y_coords, dx, dy, wire_zero_bin, time_zero_bin,
                               base_kernel)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Kernel file not found: {kernel_dir}/{plane}_plane_kernel.npz")

    return DKernels


# ============================================================================
# RUNTIME INTERPOLATION (JIT-compiled)
# ============================================================================

@partial(jit, static_argnums=(4, 5))  # wire_spacing, num_wires are static
def interpolate_diffusion_kernel(DKernel, s_observed, w_offset, t_offset,
                               wire_spacing, num_wires):
    """
    Interpolate the diffusion kernel at given s, w, t offsets.

    This is the core runtime function for efficient kernel interpolation.
    Uses linear interpolation for all three dimensions (s, wire, time).

    Parameters
    ----------
    DKernel : jnp.ndarray
        Array of shape (num_s, kernel_height, kernel_width)
    s_observed : float
        Diffusion parameter in [0, 1]
    w_offset : float
        Wire offset in [0, 1.0) - wire offset in units of wire pitch
    t_offset : float
        Time offset in [0, 1.0) - fractional position within simulation time bin.
    wire_spacing : float
        Wire spacing in cm (static, read from kernel file).
    num_wires : int
        Number of output wire positions.

    Returns
    -------
    interpolated_values : jnp.ndarray
        Interpolated kernel values with shape (num_wires, kernel_height - 1)
    """
    num_s, kernel_height, kernel_width = DKernel.shape

    # 1. S interpolation - simple since we have linear points
    s_continuous = s_observed * (num_s - 1)  # Map to [0, num_s-1]
    s_idx = jnp.floor(s_continuous).astype(int)
    s_idx = jnp.clip(s_idx, 0, num_s - 2)  # Ensure we don't go out of bounds
    s_alpha = s_continuous - s_idx

    # 2. Wire interpolation setup
    center_w = kernel_width // 2
    bins_per_wire = int(round(1.0 / wire_spacing))

    # Convert w_offset to bin offset
    w_bin_offset = w_offset * bins_per_wire
    w_base_bin = jnp.floor(w_bin_offset).astype(int)
    w_alpha = w_bin_offset - w_base_bin

    # Generate wire bin indices for each output wire position
    if num_wires % 2 == 0:
        half_wires = num_wires // 2
        wire_positions = jnp.arange(-half_wires, half_wires)
    else:
        half_wires = num_wires // 2
        wire_positions = jnp.arange(-half_wires, half_wires + 1)

    wire_base_positions = center_w - wire_positions * bins_per_wire

    # 3. Time interpolation - linear interpolation between adjacent time bins
    t_alpha = t_offset  # Already in [0, 1)

    # Initialize output array (kernel_height - 1 due to interpolation)
    output_values = jnp.zeros((num_wires, kernel_height - 1))

    # Process each wire position
    for wire_idx in range(num_wires):
        wire_bin_left = wire_base_positions[wire_idx] + w_base_bin
        wire_bin_right = wire_bin_left + 1
        wire_bin_left = jnp.clip(wire_bin_left, 0, kernel_width - 1)
        wire_bin_right = jnp.clip(wire_bin_right, 0, kernel_width - 1)

        values_s_n_left = DKernel[s_idx, :, wire_bin_left]
        values_s_n_plus_1_left = DKernel[s_idx + 1, :, wire_bin_left]
        values_s_n_right = DKernel[s_idx, :, wire_bin_right]
        values_s_n_plus_1_right = DKernel[s_idx + 1, :, wire_bin_right]

        values_s_interp_left = (1 - s_alpha) * values_s_n_left + s_alpha * values_s_n_plus_1_left
        values_s_interp_right = (1 - s_alpha) * values_s_n_right + s_alpha * values_s_n_plus_1_right
        values_w_interp = (1 - w_alpha) * values_s_interp_left + w_alpha * values_s_interp_right

        # Time interpolation: linear blend between adjacent time bins
        # As t_alpha increases (later arrival), the kernel peak shifts RIGHT (later time)
        interpolated = t_alpha * values_w_interp[:-1] + (1 - t_alpha) * values_w_interp[1:]
        output_values = output_values.at[wire_idx, :].set(interpolated)

    return output_values


@partial(jit, static_argnums=(4, 5))  # wire_spacing, num_wires are static
def interpolate_diffusion_kernel_batch(DKernel, s_observed_batch, w_offset_batch, t_offset_batch,
                                     wire_spacing, num_wires):
    """
    Batch interpolation using vmap for multiple sets of parameters.

    Parameters
    ----------
    DKernel : jnp.ndarray
        Array of shape (num_s, kernel_height, kernel_width)
    s_observed_batch : jnp.ndarray
        Array of shape (N,) with s values
    w_offset_batch : jnp.ndarray
        Array of shape (N,) with w_offset values
    t_offset_batch : jnp.ndarray
        Array of shape (N,) with t_offset values
    wire_spacing : float
        Wire spacing in cm (static).
    num_wires : int
        Number of output wire positions.

    Returns
    -------
    batch_results : jnp.ndarray
        Batch results with shape (N, num_wires, kernel_height - 1)
    """
    vmapped_interpolate = vmap(
        lambda s, w, t: interpolate_diffusion_kernel(
            DKernel, s, w, t, wire_spacing, num_wires
        ),
        in_axes=(0, 0, 0),
        out_axes=0,
    )

    return vmapped_interpolate(s_observed_batch, w_offset_batch, t_offset_batch)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def load_response_kernels(response_path=None, num_s=16,
                         time_spacing=0.5,
                         max_sigma_trans_unitless=None, max_sigma_long_unitless=None):
    """
    Load response kernels and create diffusion kernel arrays.

    Wire spacing is read from the kernel files (not a parameter).

    Parameters
    ----------
    response_path : str
        Path to directory containing kernel NPZ files.
    num_s : int
        Number of diffusion levels to create.
    time_spacing : float
        Simulation time spacing in microseconds.
    max_sigma_trans_unitless : float, optional
        Maximum transverse diffusion sigma in unitless grid coordinates.
    max_sigma_long_unitless : float, optional
        Maximum longitudinal diffusion sigma in unitless grid coordinates.

    Returns
    -------
    dict[str, ResponseKernel]
        Dictionary mapping plane names ('U', 'V', 'Y') to ResponseKernel
        NamedTuples containing DKernel array and kernel metadata.
    """
    if response_path is None:
        response_path = os.path.join(os.path.dirname(__file__), 'responses')

    planes = ['U', 'V', 'Y']

    # Create diffusion kernel arrays
    DKernels = create_diffusion_kernel_array(
        planes=planes,
        num_s=num_s,
        kernel_dir=response_path,
        time_spacing=time_spacing,
        max_sigma_trans_unitless=max_sigma_trans_unitless,
        max_sigma_long_unitless=max_sigma_long_unitless
    )

    # Extract kernel info for each plane
    response_kernels = {}
    for plane in DKernels:
        (DKernel, s_levels, kernel_shape, x_coords, y_coords,
         dx, dy, wire_zero_bin, time_zero_bin,
         base_kernel) = DKernels[plane]

        wire_spacing = float(dx)
        num_wires = calculate_wire_count(kernel_shape[1], wire_spacing)
        bins_per_wire = int(round(1.0 / wire_spacing))

        # Output height is kernel_height - 1 due to linear time interpolation
        kernel_height_out = kernel_shape[0] - 1

        # time_zero_bin is already in kernel time bins (same as output since kernel is at sim resolution)
        # Adjust by -1: flipped time interpolation starts from K[1:] at t=0,
        # so the effective zero bin in the output is one index earlier
        time_zero_bin_out = time_zero_bin - 1

        # Calculate wire_zero_bin in output wire units
        # wire_zero_bin is in kernel bins, convert to output wire position
        wire_zero_bin_out = wire_zero_bin // bins_per_wire

        # Compute static conv filter sizes for differentiable path
        sigma_w_bins = (max_sigma_trans_unitless * wire_spacing) / dx
        sigma_t_bins = (max_sigma_long_unitless * time_spacing) / dy
        ks_w = int(2 * ((sigma_w_bins * 2 * _N_SIGMAS_BLUR) // 2) + 1)
        ks_t = int(2 * ((sigma_t_bins * 2 * _N_SIGMAS_BLUR) // 2) + 1)

        response_kernels[plane] = ResponseKernel(
            DKernel=DKernel,
            num_wires=num_wires,
            kernel_height=kernel_height_out,
            wire_spacing=wire_spacing,
            time_spacing=time_spacing,
            wire_zero_bin=wire_zero_bin_out,
            time_zero_bin=time_zero_bin_out,
            base_kernel=base_kernel,
            kernel_dx=dx,
            kernel_dy=dy,
            s_levels=s_levels,
            ks_w=ks_w,
            ks_t=ks_t,
        )

    return response_kernels


def apply_diffusion_response(DKernel, s_values, wire_offsets, time_offsets,
                           wire_spacing, num_wires):
    """
    Apply diffusion response using pre-computed kernels.

    Parameters
    ----------
    DKernel : jnp.ndarray
        Diffusion kernel array for the plane.
    s_values : jnp.ndarray
        Array of s values (diffusion parameters) for each segment.
    wire_offsets : jnp.ndarray
        Array of wire offsets in [0, 1) for each segment.
    time_offsets : jnp.ndarray
        Array of time offsets in [0, 1) for each segment.
    wire_spacing : float
        Wire spacing in cm (static).
    num_wires : int
        Number of output wire positions.

    Returns
    -------
    jnp.ndarray
        Response contributions with shape (N, num_wires, kernel_height - 1).
    """
    return interpolate_diffusion_kernel_batch(
        DKernel, s_values, wire_offsets, time_offsets,
        wire_spacing, num_wires
    )


# ============================================================================
# 3D PIXEL RESPONSE KERNELS
# ============================================================================

def generate_dkernel_table_3d(sigma_trans_max, sigma_long_max,
                              base_kernel, s_levels):
    """Generate 3D diffusion kernel table via reflect padding + spatial convolution.

    Pixel kernel is distance-indexed (bin 0 = pixel center). Uses reflect
    padding on the spatial axes (correct boundary: center sample once) and
    zero-padded time axis. Separable 3D Gaussian convolution via
    lax.conv_general_dilated.

    Parameters
    ----------
    sigma_trans_max : float
        Maximum transverse diffusion sigma (NPZ pixel bins).
    sigma_long_max : float
        Maximum longitudinal diffusion sigma (NPZ time bins).
    base_kernel : jnp.ndarray, shape (Hpy, Hpz, Ht)
        Raw pixel response kernel (no diffusion). Half-sided distance-indexed.
    s_levels : jnp.ndarray, shape (num_s,)
        Diffusion levels from 0 to 1.

    Returns
    -------
    DKernel : jnp.ndarray, shape (num_s, Hpy, Hpz, Ht)
    """
    Hpy, Hpz, Ht = base_kernel.shape

    # Compute Gaussian kernel sizes and padding
    ks_s = int(2 * ((sigma_trans_max * 2 * _N_SIGMAS_BLUR) // 2) + 1)
    ks_t = int(2 * ((sigma_long_max * 2 * _N_SIGMAS_BLUR) // 2) + 1)
    ps = ks_s // 2
    pt = ks_t // 2

    # Reflect-pad spatial axes (correct center boundary for distance-indexed),
    # zero-pad time axis (signal is zero beyond kernel extent)
    padded = jnp.pad(base_kernel, ((ps, 0), (ps, 0), (pt, 0)), mode='reflect')
    padded = padded.at[..., :pt].set(0.)  # zero the time reflection

    def make_level(s):
        sigma_T = sigma_trans_max * s
        sigma_L = sigma_long_max * s

        res = padded[jnp.newaxis, jnp.newaxis, ...]

        # Pixel Y axis (axis 0 of kernel = axis 2 of NCDHW)
        g_py = _gaussian_kernel_1d(ks_s, sigma_T)
        k_py = g_py[:, None, None][jnp.newaxis, jnp.newaxis, ...]
        res = jax.lax.conv_general_dilated(
            res, k_py, (1, 1, 1), 'SAME',
            dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'))

        # Pixel Z axis (axis 1 of kernel = axis 3 of NCDHW)
        g_pz = _gaussian_kernel_1d(ks_s, sigma_T)
        k_pz = g_pz[None, :, None][jnp.newaxis, jnp.newaxis, ...]
        res = jax.lax.conv_general_dilated(
            res, k_pz, (1, 1, 1), 'SAME',
            dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'))

        # Time axis (axis 2 of kernel = axis 4 of NCDHW)
        g_t = _gaussian_kernel_1d(ks_t, sigma_L)
        k_t = g_t[None, None, :][jnp.newaxis, jnp.newaxis, ...]
        res = jax.lax.conv_general_dilated(
            res, k_t, (1, 1, 1), 'SAME',
            dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'))

        return jnp.squeeze(res)[ps:, ps:, pt:]

    return vmap(make_level)(s_levels)


@partial(jit, static_argnums=(5, 6, 7, 8))
def interpolate_pixel_response_kernel(DKernel, s_observed,
                                       py_offset, pz_offset, t_offset,
                                       pixel_spacing, num_py, num_pz,
                                       rebin_factor):
    """Interpolate 3D pixel response kernel at given (s, py, pz, t) offsets.

    Interpolates the DKernel (at NPZ resolution) in s, pixel_y, pixel_z
    dimensions, applies time interpolation at NPZ resolution, then rebins
    the time axis to simulation resolution by summing groups of rebin_factor bins.

    Parameters
    ----------
    DKernel : jnp.ndarray, shape (num_s, Hpy, Hpz, Ht)
        Kernel table at NPZ resolution.
    s_observed : float
        Diffusion parameter in [0, 1].
    py_offset : float
        Pixel y offset in [-0.5, 0.5) in pixel pitch units.
    pz_offset : float
        Pixel z offset in [-0.5, 0.5) in pixel pitch units.
    t_offset : float
        Time offset in [0, 1) in simulation time step units.
    pixel_spacing : float
        NPZ_pixel_bin / pixel_pitch (unitless ratio, static).
    num_py, num_pz : int
        Number of output pixel positions in y and z (static).
    rebin_factor : int
        Simulation time step / NPZ time bin (static, e.g. 5).

    Returns
    -------
    jnp.ndarray, shape (num_py, num_pz, Ht_sim)
        Ht_sim = (Ht - 1) // rebin_factor
    """
    num_s, Hpy, Hpz, Ht = DKernel.shape
    bins_per_pixel = int(round(1.0 / pixel_spacing))

    # S interpolation
    s_continuous = s_observed * (num_s - 1)
    s_idx = jnp.clip(jnp.floor(s_continuous).astype(int), 0, num_s - 2)
    s_alpha = s_continuous - s_idx

    # Time offset: convert from simulation time step [0, 1) to NPZ bins
    t_alpha_npz = t_offset * rebin_factor
    t_sub_bin = jnp.floor(t_alpha_npz).astype(int)
    t_alpha = t_alpha_npz - t_sub_bin

    # Output pixel positions (offsets from deposit in pixel pitches)
    half_py = num_py // 2
    half_pz = num_pz // 2
    py_positions = jnp.arange(-half_py, half_py + (num_py % 2))
    pz_positions = jnp.arange(-half_pz, half_pz + (num_pz % 2))

    # Interpolate at NPZ resolution: (num_py, num_pz, Ht - 1)
    Ht_interp = Ht - 1
    output_fine = jnp.zeros((num_py, num_pz, Ht_interp))

    for ipy in range(num_py):
        for ipz in range(num_pz):
            # Distance from deposit to this output pixel, in kernel bins
            # py_offset/pz_offset are sub-pixel offsets [-0.5, 0.5)
            dist_py = jnp.abs(py_positions[ipy] - py_offset) * bins_per_pixel
            dist_pz = jnp.abs(pz_positions[ipz] - pz_offset) * bins_per_pixel

            py_left = jnp.clip(jnp.floor(dist_py).astype(int), 0, Hpy - 2)
            py_right = py_left + 1
            py_alpha = dist_py - py_left

            pz_left = jnp.clip(jnp.floor(dist_pz).astype(int), 0, Hpz - 2)
            pz_right = pz_left + 1
            pz_alpha = dist_pz - pz_left

            # Trilinear interpolation in (s, dist_py, dist_pz)
            v_s0_py0_pz0 = DKernel[s_idx, py_left, pz_left, :]
            v_s0_py0_pz1 = DKernel[s_idx, py_left, pz_right, :]
            v_s0_py1_pz0 = DKernel[s_idx, py_right, pz_left, :]
            v_s0_py1_pz1 = DKernel[s_idx, py_right, pz_right, :]
            v_s1_py0_pz0 = DKernel[s_idx + 1, py_left, pz_left, :]
            v_s1_py0_pz1 = DKernel[s_idx + 1, py_left, pz_right, :]
            v_s1_py1_pz0 = DKernel[s_idx + 1, py_right, pz_left, :]
            v_s1_py1_pz1 = DKernel[s_idx + 1, py_right, pz_right, :]

            v_py0_pz0 = (1 - s_alpha) * v_s0_py0_pz0 + s_alpha * v_s1_py0_pz0
            v_py0_pz1 = (1 - s_alpha) * v_s0_py0_pz1 + s_alpha * v_s1_py0_pz1
            v_py1_pz0 = (1 - s_alpha) * v_s0_py1_pz0 + s_alpha * v_s1_py1_pz0
            v_py1_pz1 = (1 - s_alpha) * v_s0_py1_pz1 + s_alpha * v_s1_py1_pz1

            v_pz0 = (1 - py_alpha) * v_py0_pz0 + py_alpha * v_py1_pz0
            v_pz1 = (1 - py_alpha) * v_py0_pz1 + py_alpha * v_py1_pz1

            v_interp = (1 - pz_alpha) * v_pz0 + pz_alpha * v_pz1

            # Sub-bin time interpolation at NPZ resolution
            interpolated = t_alpha * v_interp[:-1] + (1 - t_alpha) * v_interp[1:]
            output_fine = output_fine.at[ipy, ipz, :].set(interpolated)

    # Rebin time axis: average groups of rebin_factor NPZ bins → simulation bins
    # Kernel stores signal rate (current per unit time). Averaging preserves
    # the rate when changing bin width — matches pixlar's slice_and_merge.
    Ht_sim = Ht_interp // rebin_factor
    shifted = jnp.roll(output_fine, -t_sub_bin, axis=-1)
    truncated = shifted[:, :, :Ht_sim * rebin_factor]
    output = truncated.reshape(num_py, num_pz, Ht_sim, rebin_factor).mean(axis=-1)

    return output


@partial(jit, static_argnums=(5, 6, 7, 8))
def interpolate_pixel_response_kernel_batch(DKernel, s_batch, py_offset_batch,
                                             pz_offset_batch, t_offset_batch,
                                             pixel_spacing, num_py, num_pz,
                                             rebin_factor):
    """Batch pixel kernel interpolation via vmap."""
    return vmap(
        interpolate_pixel_response_kernel,
        in_axes=(None, 0, 0, 0, 0, None, None, None, None)
    )(DKernel, s_batch, py_offset_batch, pz_offset_batch, t_offset_batch,
      pixel_spacing, num_py, num_pz, rebin_factor)


def apply_pixel_diffusion_response(DKernel, s_values, py_offsets, pz_offsets,
                                    time_offsets, pixel_spacing, num_py, num_pz,
                                    rebin_factor):
    """Apply pixel diffusion response using pre-computed 3D kernels.

    Parameters
    ----------
    DKernel : jnp.ndarray, shape (num_s, Hpy, Hpz, Ht)
    s_values : jnp.ndarray, shape (N,)
    py_offsets : jnp.ndarray, shape (N,)
    pz_offsets : jnp.ndarray, shape (N,)
    time_offsets : jnp.ndarray, shape (N,)
    pixel_spacing : float (static)
    num_py, num_pz : int (static)
    rebin_factor : int (static)

    Returns
    -------
    jnp.ndarray, shape (N, num_py, num_pz, Ht_sim)
    """
    return interpolate_pixel_response_kernel_batch(
        DKernel, s_values, py_offsets, pz_offsets, time_offsets,
        pixel_spacing, num_py, num_pz, rebin_factor)


def load_pixel_response_kernel(npz_path, num_s=16, time_spacing=0.5,
                                pixel_pitch_cm=0.4,
                                max_sigma_trans_unitless=None,
                                max_sigma_long_unitless=None):
    """Load pixel response kernel from NPZ and build 3D DKernel table.

    The DKernel table is kept at native NPZ resolution (fine spatial and time
    bins). Spatial interpolation maps from NPZ bins to pixel-pitch positions.
    Time interpolation extracts at NPZ resolution, then rebins to simulation
    time steps by summing groups of rebin_factor bins.

    Parameters
    ----------
    npz_path : str
        Path to pixel response NPZ (pixlar format).
    num_s : int
        Number of diffusion levels.
    time_spacing : float
        Simulation time step in us.
    pixel_pitch_cm : float
        Pixel pitch of the detector in cm.
    max_sigma_trans_unitless : float, optional
        Max transverse sigma in pixel pitch units.
    max_sigma_long_unitless : float, optional
        Max longitudinal sigma in simulation time step units.

    Returns
    -------
    PixelResponseKernel
    """
    from tools.config import PixelResponseKernel

    data = np.load(npz_path)
    response = jnp.array(data['response'], dtype=jnp.float32)  # (Hpy, Hpz, Ht)
    npz_pixel_bin = float(data['pixel_bin_size'])
    npz_time_bin = float(data['time_bin_size'])

    Hpy, Hpz, Ht = response.shape

    # pixel_spacing: NPZ bin / pixel pitch (unitless ratio, like wire_spacing)
    # bins_per_pixel = 1 / pixel_spacing ≈ pixel_pitch / npz_bin
    pixel_spacing = npz_pixel_bin / pixel_pitch_cm

    # Rebin factor: how many NPZ time bins per simulation time step
    rebin_factor = int(round(time_spacing / npz_time_bin))

    s_levels = jnp.linspace(0.0, 1.0, num_s)

    # Convert max sigmas from detector units to NPZ kernel bin units
    sigma_trans = max_sigma_trans_unitless if max_sigma_trans_unitless is not None else 0.0
    sigma_long = max_sigma_long_unitless if max_sigma_long_unitless is not None else 0.0

    bins_per_pixel = pixel_pitch_cm / npz_pixel_bin
    bins_per_time = time_spacing / npz_time_bin
    sigma_trans_bins = sigma_trans * bins_per_pixel
    sigma_long_bins = sigma_long * bins_per_time

    DKernel = generate_dkernel_table_3d(
        sigma_trans_bins, sigma_long_bins,
        response, s_levels)

    # Output spatial dimensions (in pixel pitches)
    bins_per_pix = int(round(pixel_pitch_cm / npz_pixel_bin))
    num_py_out = int((Hpy - 1) / bins_per_pix)
    num_pz_out = int((Hpz - 1) / bins_per_pix)

    # Output time dimension (in simulation time steps)
    # After time interpolation: Ht - 1 NPZ bins, then rebin
    kernel_time_out = (Ht - 1) // rebin_factor

    # Spatial zero bins (kernel center in output pixel units)
    py_zero_bin = num_py_out // 2
    pz_zero_bin = num_pz_out // 2

    # Time zero bin: convert from NPZ bins to simulation time steps
    t_start_us = float(data['drift_time'])
    time_zero_bin_npz = int(round(t_start_us / npz_time_bin))
    # -1 for time interpolation (same as wire), then convert to sim bins
    time_zero_bin_out = (time_zero_bin_npz - 1) // rebin_factor

    return PixelResponseKernel(
        DKernel=DKernel,
        kernel_py=num_py_out,
        kernel_pz=num_pz_out,
        kernel_time=kernel_time_out,
        pixel_spacing=pixel_spacing,
        time_spacing=time_spacing,
        rebin_factor=rebin_factor,
        py_zero_bin=py_zero_bin,
        pz_zero_bin=pz_zero_bin,
        time_zero_bin=time_zero_bin_out,
        base_kernel=response,
        s_levels=s_levels,
    )
