import jax
import jax.numpy as jnp
import numpy as np
import time
from jax.scipy.signal import convolve
from functools import partial

def load_data(file_path):
    """
    Load convolution kernel data from NPZ files.
    
    Parameters
    ----------
    file_path : str
        Path to the directory containing NPZ files.
        
    Returns
    -------
    dict
        Dictionary with U-plane, V-plane, and Y-plane data.
    """
    u_data = np.load(file_path + "fig9_u_plane.npz", allow_pickle=True)['arr_0'].item()
    v_data = np.load(file_path + "fig9_v_plane.npz", allow_pickle=True)['arr_0'].item()
    y_data = np.load(file_path + "fig9_y_plane.npz", allow_pickle=True)['arr_0'].item()
    return {'U-plane': u_data, 'V-plane': v_data, 'Y-plane': y_data}


def create_distance_kernels(kernels, n_dist, distance_falloff=1.0):
    """
    Create n_dist copies of kernels with exponential decay based on distance.

    Parameters
    ----------
    kernels : list
        List of 1D JAX arrays of shape [n_angles].
    n_dist : int
        Number of distance copies to create.
    distance_falloff : float, optional
        Controls how quickly response falls off with distance, by default 1.0.

    Returns
    -------
    jnp.ndarray
        Array of shape (n_angles, n_dist, kernel_length).
    """
    n_angles = len(kernels)
    kernel_length = kernels[0].shape[0]

    # Create result array with shape (n_angles, n_dist, kernel_length)
    result = jnp.zeros((n_angles, n_dist, kernel_length))

    # Fill the result array with scaled copies of the kernels
    for angle_idx in range(n_angles):
        for dist_idx in range(n_dist):
            # Apply exponential decay based on distance
            # Using distance in increments of 0.5
            distance = dist_idx * 0.5
            scale = jnp.exp(-(distance*distance_falloff)**2/2)
            result = result.at[angle_idx, dist_idx].set(kernels[angle_idx] * scale)

    return result


def convolve_single_wire(wire_data, kernel):
    """
    Convolve a wire's time data with a kernel using FFT (more efficient for large kernels).

    Parameters
    ----------
    wire_data : jnp.ndarray
        1D array of time data for a wire.
    kernel : jnp.ndarray
        1D array, the convolution kernel.

    Returns
    -------
    jnp.ndarray
        1D array with convolution result.
    """
    n = len(wire_data)
    kernel_padded = jnp.pad(kernel, (0, n - len(kernel)), 'constant')

    # Perform FFT convolution
    result = jnp.fft.irfft(jnp.fft.rfft(wire_data) * jnp.fft.rfft(kernel_padded), n=n)

    # Adjust for phase shift if needed
    offset = len(kernel) // 2
    return jnp.roll(result, -offset)


# Vectorize the convolution function to apply to all wires at once
convolve_vmap = jax.vmap(convolve_single_wire, in_axes=(0, None))
convolve_vmap_jit = jax.jit(convolve_vmap)


def create_kernels_and_params(file_path="wire_responses/", n_dist=6, distance_falloff=1.0):
    """
    Create kernels and parameters for the convolution.

    Parameters
    ----------
    file_path : str, optional
        Path to the NPZ files, by default "wire_responses/".
    n_dist : int, optional
        Number of distance copies to create, by default 6.
    distance_falloff : float, optional
        Falloff parameter for the exponential decay, by default 1.0.

    Returns
    -------
    dict
        Dictionary containing kernels and parameters for each plane.
    """
    # Load the data
    data_dict = load_data(file_path)

    # Number of angles: 0°, 10°, 20°, 30°, 40°, 50°, 60°, 70°, 80°, 90°
    n_angles = 10

    # Process each plane separately
    planes = ['U-plane', 'V-plane', 'Y-plane']
    result = {}

    for plane in planes:
        # Find the kernel length for this plane (using the first angle's kernel length)
        kernel_length = len(data_dict[plane]['adc'][0])

        # Get kernels for all angles except 90° (which will be all zeros)
        kernels = []
        for i in range(n_angles - 1):  # All angles except 90°
            # Get the response for this angle
            response = data_dict[plane]['adc'][i]
            # Ensure it has the right length
            if len(response) != kernel_length:
                raise ValueError(f"Kernel length mismatch in {plane} at angle {i*10}°: "
                                 f"Expected {kernel_length}, got {len(response)}")
            # Convert to JAX array
            kernels.append(jnp.array(response))

        # Add zeros for 90° angle with the same shape as the other kernels
        kernels.append(jnp.zeros(kernel_length))

        # Create kernels with distance dimension
        kernels_dist = create_distance_kernels(kernels, n_dist, distance_falloff)

        # Store in result
        result[plane] = {
            'kernels': kernels_dist,
            'n_angles': n_angles,
            'n_dist': n_dist
        }

    return result


@partial(jax.jit, static_argnums=(2, 3))
def run_convolutions(C, kernels, num_angles, num_dist):
    """
    Run convolutions for each angle and distance sequentially.

    Parameters
    ----------
    C : jnp.ndarray
        Array of shape (n_wires, n_time, n_angles, n_dist).
    kernels : jnp.ndarray
        Array of shape (n_angles, n_dist, kernel_length).
    num_angles : int
        Number of angles.
    num_dist : int
        Number of distances.

    Returns
    -------
    jnp.ndarray
        Array of shape (n_wires, n_time) with convolution results.
    """
    results = jnp.empty_like(C)
    for angle_idx in range(num_angles):
        for dist_idx in range(num_dist):
            results = results.at[:, :, angle_idx, dist_idx].set(
                convolve_vmap_jit(C[:, :, angle_idx, dist_idx], kernels[angle_idx, dist_idx])
            )

    results_collapsed = jnp.sum(results, axis=(2, 3))
    return results_collapsed


def apply_response(wire_signals_plane, kernels, num_angles, num_dist):
    """
    Apply response kernels to signal plane.
    
    Parameters
    ----------
    wire_signals_plane : jnp.ndarray
        Array of shape (n_wires, n_time, n_angles, n_dist).
    kernels : jnp.ndarray
        Array of shape (n_angles, n_dist, kernel_length).
    num_angles : int
        Number of angles.
    num_dist : int
        Number of distances.
        
    Returns
    -------
    jnp.ndarray
        Array of shape (n_wires, n_time) with convolution results.
    """
    return run_convolutions(wire_signals_plane, kernels, num_angles, num_dist)